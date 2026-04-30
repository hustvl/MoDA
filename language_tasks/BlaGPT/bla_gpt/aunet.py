"""
AUNet implementation based on official codebase.
Reference: https://github.com/facebookresearch/lingua/blob/main/apps/aunet/hierarchical.py
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from coqpit import Coqpit


@dataclass
class HierarchicalConfig(Coqpit):
    dimensions: List[int] = field(default_factory=lambda: [384, 768, 1536])
    head_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    layers: List[int] = field(default_factory=lambda: [2, 2, 9])
    sliding_windows: Optional[List[Optional[int]]] = field(default_factory=lambda: [None, None, None])
    max_seqlens: Optional[List[int]] = field(default_factory=lambda: [2048, 1024, 512])
    residuals: Optional[List[bool]] = field(default_factory=lambda: [True, True, True])

    vocab_size: int = 256
    block_size: int = 2048
    n_kv_heads: int = 4
    ffn_dim_multiplier: float = 4.0
    bias: bool = False
    norm_eps: float = 1e-5

    seed: int = 42
    lambda_level: float = 0.0
    pooling_type: str = "simple_indexed_matmul"
    tie_embed_weights: bool = False

    dropout: float = 0.0

    # Override training hyperparameters
    device_batch_size: int = 16
    byte_level_training: bool = True

    def __post_init__(self):
        assert len(self.dimensions) == len(self.head_dims) == len(self.layers) == len(self.max_seqlens)
        assert self.block_size == self.max_seqlens[0]


def cross_entropy(logits, targets, ignore_index=-1):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        inv_freq = getattr(self, 'inv_freq')
        device = inv_freq.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        actual_seq_len = seq_len if seq_len is not None else x.shape[1]
        cos_cached = getattr(self, 'cos_cached')
        sin_cached = getattr(self, 'sin_cached')
        return (
            cos_cached[:actual_seq_len].to(dtype=x.dtype),
            sin_cached[:actual_seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    seq_len = q.shape[2]

    if position_ids is None:
        cos = cos[:seq_len, :]
        sin = sin[:seq_len, :]
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 max_seq_len: int, sliding_window: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.rope = RotaryEmbedding(head_dim, max_seq_len)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            if self.n_heads % self.n_kv_heads != 0:
                kv_rep = (self.n_heads + self.n_kv_heads - 1) // self.n_kv_heads
                k = k.repeat_interleave(kv_rep, dim=1)[:, :self.n_heads, :, :]
                v = v.repeat_interleave(kv_rep, dim=1)[:, :self.n_heads, :, :]
            else:
                kv_rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(kv_rep, dim=1)
                v = v.repeat_interleave(kv_rep, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        mask_buffer = getattr(self, 'mask')
        causal_mask = mask_buffer[:T, :T].clone()
        if self.sliding_window is not None and T > self.sliding_window:
            for i in range(T):
                start = max(0, i - self.sliding_window + 1)
                causal_mask[i, :start] = 0

        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.o_proj(y)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 max_seq_len: int, sliding_window: Optional[int] = None,
                 ffn_dim_multiplier: float = 4.0, bias: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, n_kv_heads, head_dim,
                                          max_seq_len, sliding_window, bias)
        self.feed_forward = SwiGLU(dim, int(dim * ffn_dim_multiplier), bias)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class CausalTransformer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 n_layers: int, max_seq_len: int, sliding_window: Optional[int] = None,
                 ffn_dim_multiplier: float = 4.0, bias: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seq_len
        self.sliding_window = sliding_window

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, max_seq_len,
                           sliding_window, ffn_dim_multiplier, bias, norm_eps)
            for _ in range(n_layers)
        ])

        class RopeEmbeddings:
            def __init__(self):
                self.freqs_cis = None
        self.rope_embeddings = RopeEmbeddings()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                attn_impl: str = "native") -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class MaxSumMask(nn.Module):
    def __init__(self, seq_len: int, numel: int):
        super().__init__()
        self.seq_len = seq_len
        self.numel = numel

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        B, T = mask.shape
        step = max(1, T // self.numel)
        idxs = torch.arange(0, min(T, self.numel * step), step, device=mask.device)[:self.numel]
        return idxs.unsqueeze(0).repeat(B, 1)


class SimpleTransition(nn.Module):
    def __init__(self, seqlen_in: int, seqlen_out: int, dim_in: int, dim_out: int,
                 head_dim: int, rope_theta: float, eps_norm: float,
                 non_parametric: bool = False, indexed_matmul: bool = False,
                 repeat: bool = False):
        super().__init__()
        self.seqlen_in = seqlen_in
        self.seqlen_out = seqlen_out
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.non_parametric = non_parametric
        self.repeat = repeat
        self.indexed_matmul = indexed_matmul

        self.max_sum_mask = MaxSumMask(seqlen_in, seqlen_out)

        if not self.non_parametric:
            self.down_norm = RMSNorm(dim_in, eps_norm)
            self.trans_down = nn.Linear(dim_in, dim_out, bias=False)
            self.up_norm = RMSNorm(dim_out, eps_norm)
            self.trans_up = nn.Linear(dim_out, dim_in, bias=False)
        else:
            if dim_out != dim_in:
                assert dim_out % dim_in == 0
                self.features_ratio = dim_out // dim_in
            else:
                self.features_ratio = 1

    def down(self, x: torch.Tensor, mask: torch.Tensor,
             freq_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        if mask is not None and mask.any():
            indices = mask.nonzero(as_tuple=True)[1]
            if len(indices) > 0:
                if len(indices) > self.seqlen_out:
                    step = len(indices) // self.seqlen_out
                    indices = indices[::step][:self.seqlen_out]
                x = x[:, indices, :]
        else:
            if T > self.seqlen_out:
                stride = T // self.seqlen_out
                x = x[:, ::stride, :][:, :self.seqlen_out, :]

        if hasattr(self, 'down_norm'):
            x = self.down_norm(x)
            x = self.trans_down(x)

        return x

    def up(self, x: torch.Tensor, res: torch.Tensor, mask: torch.Tensor,
           freq_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        target_seq_len = res.shape[1]

        if hasattr(self, 'up_norm'):
            x = self.up_norm(x)
            x = self.trans_up(x)

        if T < target_seq_len:
            repeat_factor = target_seq_len // T
            remainder = target_seq_len % T
            x = x.repeat_interleave(repeat_factor, dim=1)
            if remainder > 0:
                extra_tokens = x[:, :remainder, :]
                x = torch.cat([x, extra_tokens], dim=1)
        elif T > target_seq_len:
            x = x[:, :target_seq_len, :]

        return x

    def reset_parameters(self):
        if hasattr(self, 'trans_down'):
            nn.init.xavier_uniform_(self.trans_down.weight)
        if hasattr(self, 'trans_up'):
            nn.init.xavier_uniform_(self.trans_up.weight)


class HierarchicalTransformer(nn.Module):
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        self.dim = config.dimensions[0]

        self.tok_embeddings = nn.Embedding(config.vocab_size, self.dim)
        self.vocab_norm = RMSNorm(self.dim, config.norm_eps)
        self.vocab = nn.Linear(self.dim, config.vocab_size, bias=False)

        self.lambda_level = config.lambda_level
        if self.lambda_level > 0:
            self.level_mask_norm = RMSNorm(self.dim * 2, config.norm_eps)
            self.level_mask = nn.Linear(self.dim * 2, len(config.dimensions), bias=False)

        self.adding_residuals = config.residuals

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(len(config.dimensions) - 1):
            dim_in = config.dimensions[i]
            dim_out = config.dimensions[i + 1]
            head_dim = config.head_dims[i]
            n_heads = max(1, dim_in // head_dim)
            n_kv_heads = min(config.n_kv_heads, n_heads)
            for possible_kv in range(n_kv_heads, 0, -1):
                if n_heads % possible_kv == 0:
                    n_kv_heads = possible_kv
                    break
            layers = config.layers[i]
            max_seq_in = config.max_seqlens[i]
            max_seq_out = config.max_seqlens[i + 1]
            sliding_window = config.sliding_windows[i] if config.sliding_windows and config.sliding_windows[i] is not None else None

            encoder = CausalTransformer(
                dim_in, n_heads, n_kv_heads, head_dim, layers, max_seq_in,
                sliding_window, config.ffn_dim_multiplier, config.bias, config.norm_eps
            )
            self.encoders.append(encoder)

            transition = SimpleTransition(
                max_seq_in, max_seq_out, dim_in, dim_out, head_dim, 10000.0, config.norm_eps,
                non_parametric="non_param" in config.pooling_type,
                indexed_matmul="indexed_matmul" in config.pooling_type,
                repeat="repeat" in config.pooling_type
            )
            self.transitions.append(transition)

            decoder = CausalTransformer(
                dim_in, n_heads, n_kv_heads, head_dim, layers, max_seq_in,
                sliding_window, config.ffn_dim_multiplier, config.bias, config.norm_eps
            )
            self.decoders.append(decoder)

        trunk_dim = config.dimensions[-1]
        trunk_head_dim = config.head_dims[-1]
        trunk_n_heads = max(1, trunk_dim // trunk_head_dim)
        trunk_n_kv_heads = min(config.n_kv_heads, trunk_n_heads)
        for possible_kv in range(trunk_n_kv_heads, 0, -1):
            if trunk_n_heads % possible_kv == 0:
                trunk_n_kv_heads = possible_kv
                break
        trunk_layers = config.layers[-1]
        trunk_max_seq = config.max_seqlens[-1]
        trunk_sliding_window = config.sliding_windows[-1] if config.sliding_windows and config.sliding_windows[-1] is not None else None

        self.trunk = CausalTransformer(
            trunk_dim, trunk_n_heads, trunk_n_kv_heads, trunk_head_dim, trunk_layers,
            trunk_max_seq, trunk_sliding_window, config.ffn_dim_multiplier,
            config.bias, config.norm_eps
        )

        if config.tie_embed_weights:
            self.tok_embeddings.weight = self.vocab.weight

        self.apply(self._init_weights)

        print(f"Hierarchical model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_pool_mask(self, level_mask: torch.Tensor, max_seqlen: Optional[List[int]] = None,
                     return_idcs: bool = True, force_first: bool = False):
        B, T = level_mask.shape
        nb_levels = len(self.encoders)
        pool_mask = []
        pool_mask_idcs = []
        nb_toks = []

        for i in range(nb_levels):
            target_seq_len = max_seqlen[i+1] if max_seqlen and i+1 < len(max_seqlen) else T // (2 ** (i+1))

            if target_seq_len >= T:
                mask = torch.ones(B, T, dtype=torch.bool, device=level_mask.device)
            else:
                mask = torch.zeros(B, T, dtype=torch.bool, device=level_mask.device)
                step = T // target_seq_len
                indices = torch.arange(0, T, step, device=level_mask.device)[:target_seq_len]
                mask[:, indices] = True

            nb_toks.append(mask.sum().item())
            pool_mask.append(mask)

            if return_idcs:
                pool_mask_idcs.append(mask.nonzero(as_tuple=True)[1])

        if return_idcs:
            return pool_mask, pool_mask_idcs, nb_toks
        else:
            return pool_mask, None, nb_toks

    def _level_mask_logits(self, features: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if self.lambda_level <= 0:
            return None

        target_emb = torch.cat([features, self.tok_embeddings(target)], dim=-1)
        logits = self.level_mask(self.level_mask_norm(target_emb))
        return logits

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                level_mask: Optional[torch.Tensor] = None,
                target_level_mask: Optional[torch.Tensor] = None):
        device = idx.device
        B, T = idx.shape

        if level_mask is None:
            level_mask = torch.zeros(B, T, dtype=torch.long, device=device)

        max_seqlens = [int(e.max_seqlen) for e in self.encoders] + [int(self.trunk.max_seqlen)]
        masks, _, nb_toks = self.get_pool_mask(
            level_mask,
            max_seqlens,
            return_idcs=False,
            force_first=True
        )

        x = self.tok_embeddings(idx)

        residuals = []
        for i, (encoder, transition, mask) in enumerate(zip(self.encoders, self.transitions, masks)):
            x = encoder(x, mask=None)
            residuals.append(x)
            x = transition.down(x, mask, None)

        x = self.trunk(x, mask=None)

        adding_residuals = self.adding_residuals if self.adding_residuals is not None else [True] * len(self.decoders)
        decoder_transition_residual_mask = list(zip(
            self.decoders, self.transitions, residuals, adding_residuals, masks
        ))

        for decoder, transition, residual, add_res, mask in reversed(decoder_transition_residual_mask):
            x = transition.up(x, residual, mask, None)
            if add_res:
                x = residual + x
            x = decoder(x, mask=None)

        logits = self.vocab(self.vocab_norm(x))

        if targets is not None:
            loss = cross_entropy(logits, targets)

            mask_loss = None
            if self.lambda_level > 0 and target_level_mask is not None:
                mask_logits = self._level_mask_logits(x, targets)
                if mask_logits is not None:
                    mask_loss = cross_entropy(mask_logits, target_level_mask)
                    loss = loss + self.lambda_level * mask_loss

            return logits, loss
        else:
            return logits[:, [-1], :], None

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
