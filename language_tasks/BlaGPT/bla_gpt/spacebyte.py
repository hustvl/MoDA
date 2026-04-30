"""
SpaceByte implementation for BlaGPT framework.
Based on the original SpaceByte paper and implementation.
FIXED: Proper UTF-8 patching with token ignoring for excess boundaries.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from numpy import byte
import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit


@dataclass
class SpaceByteConfig(Coqpit):
    """Configuration for SpaceByte model"""
    vocab_size: int = 256
    hidden_size: int = 768  # d_model in original
    num_hidden_layers: int = 12  # n_layers for global model
    pad_token_id: int = 0
    intermediate_size: int = 3072  # d_ff
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048  # context_size
    block_size: int = 2048

    # SpaceByte specific parameters
    patch_method: str = 'utf8'  # 'utf8' or 'periodic'
    global_context_size: int = 256  # TG in original
    d_local: int = 384
    n_initial_layers: int = 6
    n_local_layers: int = 12
    local_attention_window: int = 384
    print_patches: float = 0.0  # Debug printing

    # BlaGPT integration
    byte_level_training: bool = True
    compile_model: bool = True
    optimizer_name: str = "AdamW"
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.98),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "fused": True,
        }
    )
    grad_clip: float = 1.0
    learning_rate: float = 0.04  # 0.005 * sqrt(64) for batch_size=64

    def __post_init__(self):
        if self.d_local is None:
            self.d_local = self.hidden_size // 2
        if self.n_local_layers is None:
            self.n_local_layers = self.num_hidden_layers
        if self.n_initial_layers is None:
            self.n_initial_layers = self.n_local_layers // 2
        if self.local_attention_window is None:
            self.local_attention_window = self.d_local
        if self.global_context_size is None:
            self.global_context_size = self.block_size // 6
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4


class SpaceByteRMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.type_as(hidden_states)


class SpaceByteRotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotate half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SpaceByteAttention(nn.Module):
    """Multi-head attention with optional windowed attention"""
    def __init__(self, config: SpaceByteConfig, hidden_size: int, attention_window: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.attention_window = attention_window

        head_dim = 64
        self.num_heads = max(1, hidden_size // head_dim)
        self.head_dim = hidden_size // self.num_heads

        self.num_key_value_heads = max(1, self.num_heads // 3)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_layernorm = SpaceByteRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = SpaceByteRMSNorm(self.head_dim, eps=config.norm_eps)

        if hasattr(config, 'rope_theta'):
            self.rotary_emb = SpaceByteRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_layernorm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_layernorm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + attention_mask
        else:

            causal_mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=hidden_states.device), diagonal=1)
            attn_weights = attn_weights + causal_mask


        if self.attention_window is not None and self.attention_window < q_len:
            window_mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=hidden_states.device),
                                   diagonal=self.attention_window)
            window_mask = window_mask + torch.tril(torch.full((q_len, q_len), float('-inf'), device=hidden_states.device),
                                                 diagonal=-self.attention_window)
            attn_weights = attn_weights + window_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class SpaceByteMLP(nn.Module):
    """Feed-forward network with SiLU activation"""
    def __init__(self, config: SpaceByteConfig, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SpaceByteTransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config: SpaceByteConfig, hidden_size: int, attention_window: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = SpaceByteAttention(config, hidden_size, attention_window)
        self.mlp = SpaceByteMLP(config, hidden_size)
        self.input_layernorm = SpaceByteRMSNorm(hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = SpaceByteRMSNorm(hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, attention_mask=None):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states


        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SpaceByte(nn.Module):
    """SpaceByte model implementation with proper UTF-8 patching"""
    def __init__(self, config: SpaceByteConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size


        self.token_embedding = nn.Embedding(config.vocab_size, config.d_local)


        self.local_position_encoding = nn.Parameter(torch.randn(config.max_position_embeddings, config.d_local))
        self.global_position_encoding = nn.Parameter(torch.randn(config.global_context_size, config.hidden_size))


        self.initial_blocks = nn.ModuleList([
            SpaceByteTransformerBlock(config, config.d_local, config.local_attention_window)
            for _ in range(config.n_initial_layers)
        ])


        self.global_blocks = nn.ModuleList([
            SpaceByteTransformerBlock(config, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])


        self.final_blocks = nn.ModuleList([
            SpaceByteTransformerBlock(config, config.d_local, config.local_attention_window)
            for _ in range(config.n_initial_layers, config.n_local_layers)
        ])


        self.norm = SpaceByteRMSNorm(config.d_local, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_local, config.vocab_size, bias=False)


        self.apply(self._init_weights)


        self.BOS = 255  # Special byte value for beginning of sequence

    def _init_weights(self, module):
        """Initialize weights following standard practice"""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _get_utf8_patch_positions(self, tokens: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get positions for UTF-8 based patching and return positions to ignore.

        Returns:
            global_ts: Tensor of selected patch positions [batch_size, max_global_T]
            global_T: Number of patches per sequence [batch_size]
            ignored_positions: Positions beyond global capacity to ignore [batch_size, seq_len]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # UTF-8 patch boundaries: non-alphanumeric and continuation bytes
        use_global = (
            (tokens < ord('0')) |
            ((ord('9') < tokens) & (tokens < ord('A'))) |
            ((ord('Z') < tokens) & (tokens < ord('a'))) |
            ((ord('z') < tokens) & (tokens < 0b1000_0000)) |
            (0b1100_0000 <= tokens)
        )

        # Don't use consecutive boundaries
        use_global[:, 1:] &= ~use_global[:, :-1]

        # Always use BOS token as boundary
        use_global |= (tokens == self.BOS)


        max_global_T = min(self.config.global_context_size, seq_len)
        global_T = torch.full((batch_size,), 0, dtype=torch.long, device=device)
        global_ts = torch.full((batch_size, max_global_T), seq_len - 1, dtype=torch.long, device=device)


        ignored_positions = torch.zeros_like(tokens, dtype=torch.bool)

        for b in range(batch_size):
            global_positions = use_global[b].nonzero(as_tuple=True)[0]

            # CRITICAL FIX: Handle excess boundaries like original implementation
            if len(global_positions) > self.config.global_context_size:

                if targets is not None:
                    excess_positions = global_positions[self.config.global_context_size:]
                    ignored_positions[b, excess_positions] = True


                global_positions = global_positions[:self.config.global_context_size]

            global_T[b] = len(global_positions)
            if len(global_positions) > 0:
                global_ts[b, :len(global_positions)] = global_positions

        return global_ts, global_T, ignored_positions

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_all_logits: bool = False):
        """Forward pass of SpaceByte model"""
        device = idx.device
        batch_size, seq_len = idx.size()


        x = self.token_embedding(idx)  # [B, T, d_local]


        x = x + self.local_position_encoding[:seq_len]


        for block in self.initial_blocks:
            x = block(x)


        if self.config.patch_method == 'utf8':

            global_ts, global_T, ignored_positions = self._get_utf8_patch_positions(idx, targets)
            max_global_T = min(self.config.global_context_size, seq_len)


            if targets is not None:
                targets = targets.clone()
                targets[ignored_positions] = -1


            y = x.gather(1, global_ts[:, :max_global_T, None].expand(batch_size, max_global_T, self.config.d_local))


            y = torch.cat([
                torch.zeros(batch_size, max_global_T, self.config.hidden_size - self.config.d_local,
                           device=device, dtype=x.dtype),
                y
            ], dim=-1)

        else:  # periodic patching
            patch_size = seq_len // self.config.global_context_size
            y = x[:, ::patch_size]
            max_global_T = y.shape[1]


            y = torch.cat([
                torch.zeros(batch_size, max_global_T, self.config.hidden_size - self.config.d_local,
                           device=device, dtype=x.dtype),
                y
            ], dim=-1)


        y = y + self.global_position_encoding[:max_global_T]


        for block in self.global_blocks:
            y = block(y)

        # Scatter global information back to local (avoiding in-place operations)
        if self.config.patch_method == 'utf8':

            x_new = x.clone()
            for b in range(batch_size):
                valid_positions = global_ts[b, :global_T[b]]
                if len(valid_positions) > 0:
                    x_new[b, valid_positions] = x[b, valid_positions] + y[b, :global_T[b], -self.config.d_local:]
            x = x_new
        else:  # periodic
            patch_size = seq_len // self.config.global_context_size
            x_new = x.clone()
            x_new[:, ::patch_size] = x[:, ::patch_size] + y[:, :, -self.config.d_local:]
            x = x_new


        for block in self.final_blocks:
            x = block(x)


        x = self.norm(x)

        if targets is not None:

            logits = self.lm_head(x)


            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()


            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-1
            )

            return logits, loss

        else:

            if return_all_logits:
                logits = self.lm_head(x)
            else:
                logits = self.lm_head(x[:, [-1], :])

            return logits, None

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie input and output embeddings"""
        self.lm_head.weight = self.token_embedding.weight
