import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit


@dataclass
class LFM2Config(Coqpit):
    vocab_size: int = 50304
    hidden_size: int = 768
    num_hidden_layers: int = 12
    pad_token_id: int = 0
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 2048
    conv_kernel: int = 4
    conv_bias: bool = True
    use_conv_bias: bool = True
    full_attn_idxs: tuple = (2, 5, 8, 11)
    attn_implementation: str = "eager"
    block_size: int = 2048

    # training overrides
    compile_model: bool = True

    # optimizer - overriding Hyperparameters
    optimizer_name: str = (
        "Muon"  # check get_optimizer() in bla_gpt/optimizers/__init__.py
    )
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    )

    def __post_init__(self):
        if self.full_attn_idxs is None:
            self.full_attn_idxs = tuple(range(self.num_hidden_layers))

    def layers_block_type(self, layer_idx: int) -> str:
        if layer_idx in self.full_attn_idxs:
            return "attention"
        else:
            return "conv"


class LFM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, hidden_states):
        return hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )

    def forward(self, hidden_states):
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LFM2RotaryEmbedding(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            self.max_position_embeddings, self.inv_freq.device, self.inv_freq.dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LFM2Attention(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_layernorm = LFM2RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = LFM2RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_layernorm(
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_layernorm(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


class LFM2MLP(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LFM2ShortConv(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.conv_kernel = config.conv_kernel

        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            bias=config.conv_bias,
            padding=self.conv_kernel - 1,
        )

        self.in_proj = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=config.conv_bias
        )
        self.out_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.conv_bias
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Gated convolution with projections
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        # Apply gating: B * x
        Bx = B * x

        # Apply convolution
        conv_out = self.conv1d(Bx)
        conv_out = conv_out[..., :seq_len]

        # Apply second gate: C * conv_out
        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()

        # Output projection
        y = self.out_proj(y)
        return y


class LFM2AttentionDecoderLayer(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LFM2Attention(config, layer_idx)
        self.mlp = LFM2MLP(config)
        self.input_layernorm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = LFM2RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LFM2ShortConvDecoderLayer(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.short_conv = LFM2ShortConv(config)
        self.mlp = LFM2MLP(config)
        self.input_layernorm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_conv_layernorm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, cos=None, sin=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.short_conv(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_conv_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LFM2(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.rotary_emb = LFM2RotaryEmbedding(config)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in config.full_attn_idxs:
                self.layers.append(LFM2AttentionDecoderLayer(config, layer_idx))
            else:
                self.layers.append(LFM2ShortConvDecoderLayer(config, layer_idx))

        self.norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, idx, targets=None, return_all_logits=False):
        device = idx.device
        b, t = idx.size()

        hidden_states = self.embed_tokens(idx)
        cos, sin = self.rotary_emb(hidden_states, seq_len=t)
        attention_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=device), diagonal=1
        )

        for layer in self.layers:
            if isinstance(layer, LFM2AttentionDecoderLayer):
                hidden_states = layer(hidden_states, cos, sin, attention_mask)
            else:
                hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.norm(hidden_states)

        if targets is not None:
            logits = self.lm_head(hidden_states)

            # Shift targets
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-1,
            )

            return logits, loss
        else:
            if return_all_logits:
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states[:, [-1], :])
            return logits, None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
