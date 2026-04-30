import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit


@dataclass
class HNetConfig(Coqpit):
    vocab_size: int = 256
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    pad_token_id: int = 0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    # HNet specific parameters
    num_stages: int = 3
    stage_layers: List[int] = field(default_factory=lambda: [4, 4, 4])
    stage_types: List[str] = field(default_factory=lambda: ['mamba', 'attention', 'mamba'])
    use_routing: bool = True

    # Training configuration
    block_size: int = 2048
    compile_model: bool = True
    byte_level_training: bool = True

    # Optimizer configuration
    optimizer_name: str = "Muon"
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    )

    def __post_init__(self):
        if len(self.stage_layers) != self.num_stages:
            self.stage_layers = [self.num_hidden_layers // self.num_stages] * self.num_stages
        if len(self.stage_types) != self.num_stages:
            self.stage_types = ['mamba'] * self.num_stages


class HNetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HNetRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HNetAttention(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = HNetRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'), device=hidden_states.device), diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return self.o_proj(attn_output)


class HNetMLP(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HNetMambaBlock(nn.Module):
    """Simplified Mamba-like block for HNet."""
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expand_factor = 2
        self.d_conv = 4
        self.d_inner = self.expand_factor * self.hidden_size

        self.in_proj = nn.Linear(self.hidden_size, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1
        )
        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # Simplified state space operation
        x = F.silu(x)

        # Gating
        x = x * F.silu(z)

        return self.out_proj(x)


class HNetRoutingModule(nn.Module):
    """Simple routing module for dynamic chunking."""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.boundary_predictor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Simple boundary prediction based on hidden states
        boundary_logits = self.boundary_predictor(x)
        boundary_prob = torch.sigmoid(boundary_logits).squeeze(-1)
        return boundary_prob


class HNetLayer(nn.Module):
    def __init__(self, config: HNetConfig, layer_type: str = 'mamba'):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size

        self.input_layernorm = HNetRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = HNetRMSNorm(config.hidden_size, eps=config.norm_eps)

        if layer_type == 'attention':
            self.mixer = HNetAttention(config)
        elif layer_type == 'mamba':
            self.mixer = HNetMambaBlock(config)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.mlp = HNetMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == 'attention':
            hidden_states = self.mixer(hidden_states, attention_mask)
        else:
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HNetStage(nn.Module):
    def __init__(self, config: HNetConfig, stage_idx: int):
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.num_layers = config.stage_layers[stage_idx]
        self.layer_type = config.stage_types[stage_idx]

        self.layers = nn.ModuleList([
            HNetLayer(config, self.layer_type)
            for _ in range(self.num_layers)
        ])

        if config.use_routing and stage_idx > 0:
            self.routing_module = HNetRoutingModule(config.hidden_size)
        else:
            self.routing_module = None

    def forward(self, hidden_states, attention_mask=None):
        # Apply routing if available
        if self.routing_module is not None:
            boundary_prob = self.routing_module(hidden_states)
            # For now, just use routing info but don't modify the computation
            # In a full implementation, this would affect chunking

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class HNet(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.stages = nn.ModuleList([
            HNetStage(config, i) for i in range(config.num_stages)
        ])

        self.norm = HNetRMSNorm(config.hidden_size, eps=config.norm_eps)
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

        # Get embeddings
        hidden_states = self.embed_tokens(idx)

        # Create attention mask
        attention_mask = torch.triu(
            torch.full((t, t), float('-inf'), device=device), diagonal=1
        )

        # Process through stages
        for stage in self.stages:
            hidden_states = stage(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Compute loss if targets are provided
        if targets is not None:
            logits = self.lm_head(hidden_states)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-1)

            return logits, loss
        else:
            if return_all_logits:
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states[:, [-1], :])
            return logits, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

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
