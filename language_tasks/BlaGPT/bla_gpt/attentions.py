import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F

from modules.pattention import Pattention

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("FlashAttention not available")
    flash_attn_func = None

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    from norms import RMSNorm


#
# Attention Modules
#


def softpick(x, dim=-1, eps=1e-8):
    # from https://github.com/zaydzuhri/softpick-attention
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (
        torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps
    )  # epsilon is only useful if all inputs are EXACTLY 0. we might not even need it


def soft_cap(x, cap):
    return x.div_(cap).tanh_().mul_(cap)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    with torch.autocast(device_type=x.device.type, enabled=False):
        d = x.shape[3] // 2
        x1 = x[..., :d].float()
        x2 = x[..., d:].float()
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


def apply_simplified_rotary_emb(x, cos, sin):
    """Apply simplified RoPE: concat(x[:d/2]·cos, x[:d/2]·sin)

    Takes first half of input dimensions, applies element-wise multiplication
    with cos and sin, then concatenates to produce full head_dim output.

    Note: Only first half of x receives positional encoding. This is by design.
    """
    assert x.ndim == 4  # multihead attention
    # Use same autocast context as standard RoPE for fair comparison
    with torch.autocast(device_type=x.device.type, enabled=False):
        d = x.shape[3] // 2
        x_half = x[..., :d].float()  # Take first half
        x_cos = x_half * cos
        x_sin = x_half * sin
    return torch.cat([x_cos, x_sin], dim=-1).type_as(x)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, seq_len=1024):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos", freqs.cos().float(), persistent=False)
        self.register_buffer("sin", freqs.sin().float(), persistent=False)

    def forward(self, x):
        return self.cos[None, : x.size(-3), None, :], self.sin[
            None, : x.size(-3), None, :
        ]


class SimplifiedRotary(torch.nn.Module):
    """Simplified RoPE using concatenation instead of 2D rotations.

    Instead of rotating dimension pairs, this applies element-wise cos/sin
    multiplication and concatenates: q' = concat(q·cos(θ), q·sin(θ))

    Args:
        dim: Half of head_dim (since concat doubles the dimension)
        base: Frequency base (default 10000)
        seq_len: Maximum sequence length for precomputation
    """
    def __init__(self, dim, base=10000, seq_len=1024):
        super().__init__()
        # No longer need even dimension check since we use all dims

        # FIXED: Creates dim frequencies (one per dimension, no stepping)
        # This matches the apply_simplified_rotary_emb expectation
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)

        # Register buffers (non-persistent, same as standard RoPE)
        self.register_buffer("cos", freqs.cos().float(), persistent=False)
        self.register_buffer("sin", freqs.sin().float(), persistent=False)

    def forward(self, x):
        # Slice to sequence length (same indexing as standard Rotary)
        return self.cos[None, : x.size(-3), None, :], self.sin[None, : x.size(-3), None, :]


#
# Base Attention Module
#


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head  # New parameter for number of key-value heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.soft_cap = 50.0 if config.use_soft_logit_capping else 0.0
        self.use_softpick = getattr(config, "use_softpick", False)
        self.causal = True

        # RMSNorm before q and k projections
        if config.rmsnorm_before_qk:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        if config.pos_encoding == "rotary":
            # Get rope_variant with backward compatibility
            rope_variant = getattr(config, 'rope_variant', 'standard')

            if rope_variant == 'standard':
                self.rotary = Rotary(self.head_dim, base=config.rope_theta)
                self.apply_rope_fn = apply_rotary_emb
            elif rope_variant == 'simplified':
                # SimplifiedRotary takes head_dim/2 (will be doubled by concat)
                self.rotary = SimplifiedRotary(self.head_dim // 2, base=config.rope_theta)
                self.apply_rope_fn = apply_simplified_rotary_emb
            else:
                raise ValueError(f"Unknown rope_variant: {rope_variant}")

        elif config.pos_encoding == "relative":
            self.rel_pos_emb = nn.Parameter(
                torch.zeros(2 * config.block_size - 1, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_emb, std=0.02)
        elif config.pos_encoding == "none" or config.pos_encoding is None:
            pass
        else:
            raise ValueError(f"Unknown positional encoding: {config.pos_encoding}")

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Causal mask
        self.mask = None

        self.set_layers(config)

    def set_layers(self, config):
        # Projections for query, key, and value
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, q=None, mask=None):
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Update mask if provided
        if mask is not None:
            self.mask = mask
        # else:
        #     self.mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and rotary embeddings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare attention inputs
        q, k, v = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0 and not self.use_softpick:
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, T_q, T)

        # Project output
        return self._project_output(y, B, T_q, C)

    def _project_query(self, x, B, T):
        return self.q_proj(x).view(B, T, self.n_head, self.head_dim)

    def _project_kv(self, x, B, T):
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        return kv.unbind(dim=2)

    def _apply_norm(self, q, k):
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def _apply_rotary(self, q, k, T_q, T):
        cos, sin = self.rotary(q)
        q = self.apply_rope_fn(q, cos, sin)
        # Recompute cos/sin for k only if sequence lengths differ
        cos, sin = self.rotary(k) if T_q != T else (cos, sin)
        k = self.apply_rope_fn(k, cos, sin)
        return q, k

    def _apply_relative_pos(self, q, k, T_q, T):
        # Get relative position embeddings
        pos_emb = self._get_rel_pos_emb(T_q, T)

        # Apply relative position embeddings
        q = q + pos_emb[:T_q].unsqueeze(0).unsqueeze(0)
        k = k + pos_emb[:T].unsqueeze(0).unsqueeze(0)

        return q, k

    def _get_rel_pos_emb(self, T_q, T):
        # Get relative position embeddings centered around each position
        seq_length = max(T_q, T)
        positions = torch.arange(seq_length, device=self.rel_pos_emb.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = (
            relative_positions + seq_length - 1
        )  # shift to all positive
        return self.rel_pos_emb[relative_positions]

    def _prepare_qkv(self, q, k, v):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k,v for multi-query attention
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        return q, k, v

    def _flash_attention(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.causal if self.mask is None else False,
        )

    def _manual_attention(self, q, k, v, T_q, T):
        if self.causal and self.mask is None:
            self.mask = torch.tril(
                torch.ones(T_q, T, dtype=torch.bool, device=q.device)
            ).view(1, 1, T_q, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.soft_cap > 0:
            att = soft_cap(att, self.soft_cap)

        att = att.masked_fill(self.mask[:, :, :T_q, :T] == 0, float("-inf"))
        if self.use_softpick:
            att = softpick(att, dim=-1)
        else:
            att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def _project_output(self, y, B, T_q, C):
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.resid_dropout(self.c_proj(y))


#
# Specialized Attention Modules
#


class MultiHeadLatentAttention(Attention):
    def __init__(self, config):
        assert config.n_latentd > 0, "Must provide number of latent dimensions"
        self.n_latentd = config.n_latentd
        super().__init__(config)

    def set_layers(self, config):
        # Projections for query, key, and value
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_latent = nn.Linear(
            config.n_embd,
            self.n_latentd,
            bias=config.bias or config.use_qkv_bias,
        )
        self.kv_proj = nn.Linear(
            self.n_latentd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def _project_kv(self, x, B, T):
        kv = self.kv_latent(x)
        kv = self.kv_proj(kv).view(B, T, 2, self.n_kv_head, self.head_dim)
        return kv.unbind(dim=2)


class PattentionSelfAttention(Attention):
    """
    TokenFormer: https://arxiv.org/abs/2410.23168
    """

    def __init__(self, config):
        super().__init__(config)

    def set_layers(self, config):
        self.q_proj = Pattention(config)
        self.k_proj = Pattention(config)
        self.v_proj = Pattention(config)
        self.c_proj = Pattention(config)

    def _project_kv(self, x, B, T):
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        return k, v


class KVShiftingAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

        # Initialize KV shifting parameters
        # Following paper's initialization: randomly initialize from U(0,1)
        # and make them sum to 1
        self.alpha1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.alpha2 = nn.Parameter(torch.ones(self.n_kv_head) - self.alpha1)
        self.beta1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.beta2 = nn.Parameter(torch.ones(self.n_kv_head) - self.beta1)

    def _shift_kv(self, x):
        """Perform shifting operation on key/value tensors.
        Shifts the sequence by padding a zero at the beginning and dropping last element.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_kv_head, head_dim)

        Returns:
            Shifted tensor of same shape
        """
        # Get shifted version by padding front and removing last element
        # Keep same dimensions by dropping last element after padding
        x_shifted = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        return x_shifted

    def _project_kv(self, x, B, T):
        """Override parent's _project_kv to add KV shifting.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            B: Batch size
            T: Sequence length

        Returns:
            Tuple of processed key and value tensors
        """
        # Get initial K,V projections using parent method
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Get shifted versions
        k_shifted = self._shift_kv(k)
        v_shifted = self._shift_kv(v)

        # Combine original and shifted versions with learned parameters
        k = (
            self.alpha1.view(1, 1, -1, 1) * k
            + self.alpha2.view(1, 1, -1, 1) * k_shifted
        )
        v = self.beta1.view(1, 1, -1, 1) * v + self.beta2.view(1, 1, -1, 1) * v_shifted

        return k, v


class ForgettingAttention(Attention):
    """
    Forgetting Transformer Attention: https://openreview.net/pdf?id=q2Lnyegkr8
    """

    def __init__(self, config):
        super().__init__(config)
        # Add parameters for the forget gate (one for each attention head)
        self.wf = nn.Linear(config.n_embd, self.n_head, bias=True)

    def forward(self, x, q=None, mask=None):
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Update mask if provided
        if mask is not None:
            self.mask = mask

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and rotary embeddings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Compute forget gates for each position and head: ft = σ(wf^T * xt + bf)
        forget_gates = torch.sigmoid(self.wf(x))  # [B, T, n_head]

        # Prepare attention inputs
        q, k, v = self._prepare_qkv(q, k, v)

        # Compute attention with forget gates
        y = self._forgetting_attention(q, k, v, forget_gates, T)

        # Project output
        return self._project_output(y, B, T_q, C)

    def _forgetting_attention(self, q, k, v, forget_gates, T):
        # Compute attention logits
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply soft cap if configured
        if self.soft_cap > 0:
            att = soft_cap(att, self.soft_cap)

        # Arrange forget gates to match attention dimensions
        forget_gates = forget_gates.transpose(1, 2)  # [B, n_head, T]

        # Create the logit bias matrix D based on forget gates
        # Compute log of forget gates
        log_forget = torch.log(forget_gates + 1e-10)  # [B, n_head, T]

        # Compute cumulative sums c[t] = Σ_i=1^t log(f_i)
        # This is used to efficiently compute dij = Σ_l=j+1^i log(f_l)
        cum_log_forget = torch.zeros_like(log_forget)
        cum_log_forget[..., 1:] = torch.cumsum(log_forget[..., :-1], dim=-1)

        # Compute D[i,j] = c[i] - c[j]
        # This gives us the sum of log forget gates from j+1 to i
        c_i = cum_log_forget.unsqueeze(-1)  # [B, n_head, T, 1]
        c_j = cum_log_forget.unsqueeze(-2)  # [B, n_head, 1, T]
        D = c_i - c_j  # [B, n_head, T, T]

        # Add logit bias to attention scores (applying the forget gate effect)
        att = att + D

        # Apply causal mask (this enforces attention only to previous tokens)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Compute weighted sum of values
        return att @ v


class MultiTokenAttention(Attention):
    """
    Multi-Token Attention (MTA) allows LLMs to condition their attention weights
    on multiple query and key vectors simultaneously through convolution operations.

    Key components:
    1. Key-query convolution: Combines information from multiple query-key pairs
    2. Head mixing convolution: Shares information between attention heads
    3. Group normalization with depth scaling: Improves gradient flow
    """

    def __init__(self, config):
        super().__init__(config)

        # MTA specific parameters
        self.use_key_query_conv = getattr(config, "use_key_query_conv", True)
        self.use_head_conv = getattr(config, "use_head_conv", True)
        self.use_group_norm = getattr(config, "use_group_norm", True)

        # Convolution kernel dimensions
        self.cq = getattr(config, "mta_query_kernel_size", 6)  # Query dimension
        self.ck = getattr(config, "mta_key_kernel_size", 11)  # Key dimension
        self.ch = getattr(config, "mta_head_kernel_size", 2)  # Head dimension

        # Whether to apply convolution pre or post softmax
        self.pre_softmax_key_query = getattr(config, "pre_softmax_key_query", True)
        self.pre_softmax_head = getattr(config, "pre_softmax_head", False)

        # Ensure head count is divisible by head kernel size
        assert self.n_head % self.ch == 0, (
            f"Head count {self.n_head} must be divisible by head kernel size {self.ch}"
        )

        # Initialize convolution kernels
        if self.use_key_query_conv:
            # One convolution kernel per head
            self.key_query_conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(self.cq, self.ck),
                padding=((self.cq - 1) // 2, (self.ck - 1) // 2),
                bias=False,
                groups=1,
            )

            # Initialize identity kernel (only central value is 1.0)
            with torch.no_grad():
                kernel = torch.zeros(1, 1, self.cq, self.ck)
                kernel[0, 0, self.cq // 2, self.ck // 2] = 1.0
                self.key_query_conv.weight.copy_(kernel)

        if self.use_head_conv:
            # Convolution across groups of heads
            self.head_groups = self.n_head // self.ch

            # Create separate convolution kernel for each group
            # Note: This is a simplified implementation using Parameter instead of Conv1d
            # to avoid dimension issues with standard Conv1d application
            self.head_kernel = nn.Parameter(
                torch.zeros(self.head_groups, self.ch, self.ch)
            )

            # Initialize identity mapping (diagonal elements = 1)
            with torch.no_grad():
                for g in range(self.head_groups):
                    self.head_kernel[g] = torch.eye(self.ch)

        # Group normalization for heads
        if self.use_group_norm:
            self.group_norm = nn.GroupNorm(self.n_head, self.n_head)

    def _manual_attention(self, q, k, v, T):
        # Get actual sequence lengths
        B, H, T_q, _ = q.shape
        _, _, T_k, _ = k.shape

        # Calculate attention logits (B, n_head, T_q, T_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Create fresh causal mask every time, expanded for batch and head dimensions
        causal_mask = torch.tril(torch.ones(T_q, T_k, device=q.device)).view(
            1, 1, T_q, T_k
        )

        # Apply key-query convolution if enabled
        if self.use_key_query_conv and self.pre_softmax_key_query:
            # Causal masking
            att_masked_for_conv = att * causal_mask

            # Process each head separately with the same convolution
            B, H, T_q, T_k = att_masked_for_conv.shape
            att_reshaped = att_masked_for_conv.view(B * H, 1, T_q, T_k)

            # Apply convolution
            att_conv = self.key_query_conv(att_reshaped)

            # Reshape back
            att = att_conv.view(B, H, T_q, T_k)

            # Re-apply causal masking to ensure convolution didn't leak information
            att = att * causal_mask

            # Now apply softmax masking with -inf
            att = att.masked_fill((1 - causal_mask).bool(), float("-inf"))
        else:
            # Standard masking for softmax
            att = att.masked_fill((1 - causal_mask).bool(), float("-inf"))

        # Apply head convolution if enabled and set to pre-softmax
        if self.use_head_conv and self.pre_softmax_head:
            # Reshape to group heads (B, head_groups, ch, T_q, T_k)
            B, H, T_q, T_k = att.shape
            att = att.view(B, self.head_groups, self.ch, T_q, T_k)

            # Apply head mixing for each batch and position
            mixed_att = torch.zeros_like(att)
            for g in range(self.head_groups):
                # For each group, mix the heads using the kernel
                mixed_att[:, g] = torch.einsum(
                    "bhtk,hj->bjtk", att[:, g], self.head_kernel[g]
                )

            att = mixed_att
            # Reshape back to original shape
            att = att.view(B, H, T_q, T_k)

            # Re-apply causal masking to ensure no information leakage
            att = att.masked_fill((1 - causal_mask).bool(), float("-inf"))

        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)

        # Apply dropout
        att = self.attn_dropout(att)

        # Apply key-query convolution if enabled and post-softmax
        if self.use_key_query_conv and not self.pre_softmax_key_query:
            # Apply strict causal masking before convolution
            att_masked_for_conv = att * causal_mask

            # Process each head separately
            B, H, T_q, T_k = att_masked_for_conv.shape
            att_reshaped = att_masked_for_conv.view(B * H, 1, T_q, T_k)

            # Apply convolution
            att_conv = self.key_query_conv(att_reshaped)

            # Reshape back
            att = att_conv.view(B, H, T_q, T_k)

            # Re-apply causal masking
            att = att * causal_mask

            # Renormalize attention weights to sum to 1 after masking
            att_sum = att.sum(dim=-1, keepdim=True)
            att = att / (att_sum + 1e-8)  # Add small epsilon to avoid division by zero

        # Apply head convolution if enabled and set to post-softmax
        if self.use_head_conv and not self.pre_softmax_head:
            # Reshape to group heads (B, head_groups, ch, T_q, T_k)
            B, H, T_q, T_k = att.shape
            att = att.view(B, self.head_groups, self.ch, T_q, T_k)

            # Apply head mixing for each batch and position
            mixed_att = torch.zeros_like(att)
            for g in range(self.head_groups):
                # For each group, mix the heads using the kernel
                mixed_att[:, g] = torch.einsum(
                    "bhtk,hj->bjtk", att[:, g], self.head_kernel[g]
                )

            att = mixed_att
            # Reshape back to original shape
            att = att.view(B, H, T_q, T_k)

            # Re-apply causal masking
            att = att * causal_mask

            # Renormalize attention weights after masking
            att_sum = att.sum(dim=-1, keepdim=True)
            att = att / (att_sum + 1e-8)  # Add small epsilon to avoid division by zero

        # Apply attention to values
        y = att @ v

        return y

    def _flash_attention(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # Flash attention handles causal masking internally with is_causal
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,  # Always use causal masking
        )

    def _project_output(self, y, B, T_q, C):
        # Transpose and reshape attention output
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)

        # Apply group normalization if enabled
        if self.use_group_norm:
            # Reshape for group normalization
            y = y.view(B * T_q, self.n_head, self.head_dim)
            # Apply group norm (treats each head as a group)
            y = self.group_norm(y)
            # Reshape back to original dimensions
            y = y.view(B, T_q, C)

        # Apply output projection and dropout
        return self.resid_dropout(self.c_proj(y))

    def forward(self, x, q=None, mask=None):
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Always create a fresh causal mask for the correct sequence length
        causal_mask = torch.tril(torch.ones(T_q, T, device=x.device)).view(1, 1, T_q, T)

        # Use the fresh mask for this forward pass
        return super().forward(x, q, causal_mask)


class KDAAttention(Attention):
    """
    Kimi Delta Attention (KDA) - Linear attention with fine-grained gating.

    Reference: https://github.com/MoonshotAI/Kimi-Linear

    Key features:
    - Channel-wise decay gates (Diag(alpha)) instead of scalar gating
    - Delta rule with weight decay for associative memory
    - Hardware-efficient chunkwise parallel computation
    - O(T) complexity with fixed-size recurrent state
    """

    def __init__(self, config):
        # KDA-specific parameters
        self.chunk_size = getattr(config, 'kda_chunk_size', 64)
        self.use_short_conv = getattr(config, 'kda_use_short_conv', True)
        self.conv_kernel_size = 4

        # Compute decay rank before super().__init__() which calls set_layers()
        head_dim = config.n_embd // config.n_head
        self.decay_rank = getattr(config, 'kda_decay_rank', None) or head_dim

        super().__init__(config)

    def set_layers(self, config):
        """Override to add KDA-specific projections."""
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )

        # Short convolutions for Q, K, V (following paper)
        if self.use_short_conv:
            self.q_conv = nn.Conv1d(
                config.n_embd, config.n_embd,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size - 1,
                groups=config.n_embd
            )
            self.k_conv = nn.Conv1d(
                config.n_embd // (config.n_head // config.n_kv_head),
                config.n_embd // (config.n_head // config.n_kv_head),
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size - 1,
                groups=config.n_embd // (config.n_head // config.n_kv_head)
            )
            self.v_conv = nn.Conv1d(
                config.n_embd // (config.n_head // config.n_kv_head),
                config.n_embd // (config.n_head // config.n_kv_head),
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size - 1,
                groups=config.n_embd // (config.n_head // config.n_kv_head)
            )

        # Channel-wise decay parameters (low-rank for efficiency)
        self.alpha_down = nn.Linear(config.n_embd, self.decay_rank, bias=False)
        self.alpha_up = nn.Linear(self.decay_rank, config.n_embd, bias=False)

        # Learning rate beta (per head, scalar)
        self.beta_proj = nn.Linear(config.n_embd, config.n_head, bias=False)

        # L2 normalization layers for Q and K
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def _apply_short_conv(self, x, conv_layer):
        """Apply short convolution with causal padding."""
        # x: (B, T, C)
        B, T, C = x.shape
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = conv_layer(x)
        # Remove extra padding to maintain causality
        x = x[:, :, :T]
        return x.transpose(1, 2)

    def _project_query(self, x, B, T):
        """Override to add short conv and activation."""
        q = self.q_proj(x)
        if self.use_short_conv:
            q = self._apply_short_conv(q, self.q_conv)
        q = F.silu(q)  # Swish activation
        return q.view(B, T, self.n_head, self.head_dim)

    def _project_kv(self, x, B, T):
        """Override to add short conv and activation."""
        kv = self.kv_proj(x)
        # Split into k and v first
        kv = kv.view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Flatten back to apply conv
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)

        if self.use_short_conv:
            # Apply conv separately to k and v
            k = self._apply_short_conv(k, self.k_conv)
            v = self._apply_short_conv(v, self.v_conv)

        # Apply activation
        k = F.silu(k)
        v = F.silu(v)

        # Reshape back
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        return k, v

    def _apply_norm(self, q, k):
        """Apply L2 normalization to Q and K."""
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def _compute_decay_and_beta(self, x):
        """Compute channel-wise decay alpha and learning rate beta."""
        B, T, C = x.shape

        # Channel-wise decay via low-rank projection
        # alpha = sigmoid(up(down(x))) in [0, 1]^d
        alpha = self.alpha_down(x)
        alpha = self.alpha_up(alpha)
        alpha = torch.sigmoid(alpha)
        alpha = alpha.view(B, T, self.n_head, self.head_dim)

        # Learning rate beta (per head, scalar)
        beta = torch.sigmoid(self.beta_proj(x))  # (B, T, n_head)

        return alpha, beta

    def _manual_attention(self, q, k, v, T_q, T):
        """
        KDA computation using official FLA kernel.

        Uses the optimized chunk_kda kernel from flash-linear-attention library.
        This ensures correctness and performance.
        """
        from fla.ops.kda import chunk_kda

        B, H, T_seq, D = q.shape
        V = v.shape[-1]

        # Get alpha and beta from cache (computed in forward())
        # These were computed from the input x
        if hasattr(self, '_alpha_cache') and hasattr(self, '_beta_cache'):
            alpha = self._alpha_cache  # [B, T, H, D]
            beta = self._beta_cache    # [B, T, H]
        else:
            # Fallback: use random values for testing
            alpha = torch.sigmoid(torch.randn(B, T_seq, H, D, device=q.device))
            beta = torch.sigmoid(torch.randn(B, T_seq, H, device=q.device))

        # Compute g as per-timestep log decay (not cumsum!)
        # The cumsum is handled internally by chunk_kda
        g = torch.log(alpha + 1e-10)  # [B, T, H, D]

        # Transpose from [B, H, T, D] to [B, T, H, D] for FLA kernel
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Call official FLA kernel
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=D ** -0.5,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,  # Let FLA handle normalization
        )

        # Transpose back to [B, H, T, V]
        o = o.transpose(1, 2).contiguous()

        return o

    def forward(self, x, q=None, mask=None):
        """Forward pass with KDA attention."""
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Compute decay and beta from input
        alpha, beta = self._compute_decay_and_beta(x)

        # Store for use in _manual_attention
        # (This is a workaround - ideally we'd pass these through)
        self._alpha_cache = alpha
        self._beta_cache = beta

        # Project inputs
        q_proj = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Skip L2 normalization - FLA kernel will handle it
        # q_proj, k = self._apply_norm(q_proj, k)

        # No rotary embeddings for KDA (uses learnable decay instead)
        # Prepare for attention computation
        q_proj, k, v = self._prepare_qkv(q_proj, k, v)

        # Compute KDA attention (always use manual, no flash attention for linear attention)
        y = self._manual_attention(q_proj, k, v, T_q, T)

        # Project output
        return self._project_output(y, B, T_q, C)


class GatedAttention(Attention):
    """
    Gated Attention with head-specific sigmoid gates applied after SDPA.

    From paper: Gated Attention for Large Language Models (arXiv:2505.06708)
    Qwen Team, Alibaba Group (May 2025)

    Applies elementwise multiplicative gating: Y' = Y ⊙ sigmoid(X·W_gate)
    where X is the pre-normalized input hidden states.

    Key features:
    - Head-specific gates (each head has own gate parameters)
    - Query-dependent gating (gates computed from input hidden states)
    - Eliminates attention sink phenomenon (46.7% → 4.8%)
    - Improves training stability and tolerates larger learning rates
    - Reduces massive activations

    Performance gains (from paper, 15B MoE model):
    - PPL: -0.265 improvement
    - MMLU: +2.03 points
    - GSM8k: +2.35 points
    """

    def __init__(self, config):
        super().__init__(config)

        # Gate projection: maps d_model → d_model (full dimensionality)
        # This will be reshaped to (n_head, head_dim) for per-head gating
        self.gate_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

    def forward(self, x, q=None, mask=None):
        """
        Forward pass with gated attention.

        Stores pre-norm input for gate computation, then calls parent forward.
        """
        # Store pre-norm input for gate computation
        # This is the hidden state after layer normalization (ln_1 in Block)
        self.x_input = x  # Shape: (B, T, n_embd)

        # Call parent forward which will eventually call our overridden methods
        return super().forward(x, q, mask)

    def _flash_attention(self, q, k, v):
        """
        Flash attention with gating applied to output.

        Args:
            q: Query tensor (B, n_head, T, head_dim)
            k: Key tensor (B, n_head, T, head_dim)
            v: Value tensor (B, n_head, T, head_dim)

        Returns:
            Gated attention output (B, n_head, T, head_dim)
        """
        # Get standard SDPA output from parent
        y = super()._flash_attention(q, k, v)
        # y shape: (B, n_head, T, head_dim)

        # Apply per-head gating
        y = self._apply_gate(y)
        return y

    def _manual_attention(self, q, k, v, T_q, T):
        """
        Manual attention with gating applied to output.

        Args:
            q: Query tensor (B, n_head, T, head_dim)
            k: Key tensor (B, n_head, T, head_dim)
            v: Value tensor (B, n_head, T, head_dim)
            T_q: Query sequence length
            T: Key/Value sequence length

        Returns:
            Gated attention output (B, n_head, T, head_dim)
        """
        # Get standard SDPA output from parent
        y = super()._manual_attention(q, k, v, T_q, T)
        # y shape: (B, n_head, T, head_dim)

        # Apply per-head gating
        y = self._apply_gate(y)
        return y

    def _apply_gate(self, y):
        """
        Apply head-specific sigmoid gates to SDPA output.

        Computes gates from pre-norm input hidden states, then applies
        multiplicative gating element-wise to attention output.

        Args:
            y: SDPA output, shape (B, n_head, T, head_dim)

        Returns:
            gated_y: Gated output, same shape (B, n_head, T, head_dim)
        """
        B, n_head, T, head_dim = y.shape

        # Compute gates from pre-norm input: sigmoid(x @ W_gate)
        # x_input: (B, T, n_embd)
        # gate_proj: (n_embd, n_embd)
        gates = torch.sigmoid(self.gate_proj(self.x_input))
        # gates: (B, T, n_embd)

        # Reshape gates to match per-head structure
        # (B, T, n_embd) → (B, T, n_head, head_dim)
        gates = gates.view(B, T, n_head, head_dim)

        # Transpose to match y's shape
        # (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        gates = gates.transpose(1, 2)

        # Apply multiplicative gating element-wise
        # This creates query-dependent sparsity in the attention output
        gated_y = y * gates

        return gated_y


@torch.compile
def diff_attn_v2_func(attn1: torch.Tensor, attn2: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    """Differential attention v2: attn1 - sigmoid(lambda) * attn2"""
    return attn1 - torch.sigmoid(lambda_val).unsqueeze(-1) * attn2


class MultiheadDiffAttnv2(Attention):
    """
    Differential Attention v2 (Microsoft, Jan 2025)

    Key improvements over v1:
    - Doubled query heads (2h), same KV heads (native GQA support)
    - Token-specific sigmoid lambda (vs global exponential)
    - No per-head RMSNorm (stable gradients)
    - Single FlashAttention call (efficient)

    From: https://huggingface.co/blog/microsoft/diff-attn-v2
    """

    def __init__(self, config):
        # Store doubled head count before calling super().__init__
        self._doubled_n_head = config.n_head * 2
        super().__init__(config)

        # Lambda projection: (n_embd) -> (n_head) per token
        self.lambda_proj = nn.Linear(config.n_embd, config.n_head, bias=False)

    def set_layers(self, config):
        # Query projection with DOUBLED heads
        self.q_proj = nn.Linear(
            config.n_embd,
            self._doubled_n_head * self.head_dim,
            bias=config.bias or config.use_qkv_bias
        )
        # KV projection stays the same
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        # Output projection: back to n_embd (not doubled)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def _project_query(self, x, B, T):
        # Project to doubled heads
        return self.q_proj(x).view(B, T, self._doubled_n_head, self.head_dim)

    def _prepare_qkv(self, q, k, v):
        # q has doubled heads, k/v have normal heads
        q = q.transpose(1, 2)  # (B, 2*n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)

        # Expand k,v to match doubled query heads
        # Each KV head serves 2x the query heads now
        n_rep = self._doubled_n_head // self.n_kv_head
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        return q, k, v

    def forward(self, x, q=None, mask=None):
        # Store input for lambda computation
        self._x_input = x
        return super().forward(x, q, mask)

    def _apply_diff_attn(self, y):
        """Apply differential attention: attn1 - sigmoid(lambda) * attn2"""
        # y shape: (B, 2*n_head, T, head_dim)
        # Split into odd/even heads (same GQA group)
        attn1 = y[:, 0::2]  # (B, n_head, T, head_dim)
        attn2 = y[:, 1::2]  # (B, n_head, T, head_dim)

        # Compute token-specific lambda
        lambda_val = self.lambda_proj(self._x_input)  # (B, T, n_head)
        lambda_val = lambda_val.transpose(1, 2)  # (B, n_head, T)

        # Apply diff function
        return diff_attn_v2_func(attn1, attn2, lambda_val)

    def _flash_attention(self, q, k, v):
        y = super()._flash_attention(q, k, v)
        return self._apply_diff_attn(y)

    def _manual_attention(self, q, k, v, T_q, T):
        y = super()._manual_attention(q, k, v, T_q, T)
        return self._apply_diff_attn(y)

    def _project_output(self, y, B, T_q, C):
        # y now has n_head (not doubled), normal output projection
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.resid_dropout(self.c_proj(y))


#
# WIP Implementations
#


class ForgettingTransformerPro(ForgettingAttention):  # OOM !
    def __init__(self, config):
        super().__init__(config)

        # Output gate and normalization
        self.output_gate = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.output_norm = RMSNorm(config.n_embd)

        # KV-shift components - single projection for efficiency
        self.kv_shift_proj = nn.Linear(config.n_embd, 2 * self.n_kv_head, bias=False)

        # Remove the stateful buffers - we'll handle this differently
        self.register_buffer("zeros", torch.zeros(1), persistent=False)

    def _project_kv(self, x, B, T):
        """Override to implement KV-shift (data-dependent token shift) in a vectorized way"""
        # Original projection for K and V
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k_raw, v_raw = kv.unbind(dim=2)

        # If sequence length is 1, just return the raw projections
        if T == 1:
            return k_raw, v_raw

        # Compute shift gates for both K and V at once (more efficient)
        shift_gates = torch.sigmoid(self.kv_shift_proj(x))
        shift_gates = shift_gates.view(B, T, 2, self.n_kv_head, 1)
        key_gates, value_gates = shift_gates[:, :, 0], shift_gates[:, :, 1]

        # Vectorized computation
        k_shifted = torch.zeros_like(k_raw)
        v_shifted = torch.zeros_like(v_raw)

        # First position is unchanged
        k_shifted[:, 0] = k_raw[:, 0]
        v_shifted[:, 0] = v_raw[:, 0]

        # Use a simple recurrence relation for the rest
        # k_t = α_t * k_{t-1} + (1-α_t) * k̃_t
        for t in range(1, T):
            k_shifted[:, t] = (
                key_gates[:, t] * k_shifted[:, t - 1]
                + (1 - key_gates[:, t]) * k_raw[:, t]
            )
            v_shifted[:, t] = (
                value_gates[:, t] * v_shifted[:, t - 1]
                + (1 - value_gates[:, t]) * v_raw[:, t]
            )

        # Apply normalization to keys if needed
        if hasattr(self, "k_norm"):
            k_shifted = self.k_norm(k_shifted)

        return k_shifted, v_shifted

    def _project_output(self, y, B, T_q, C):
        """Override to add output gate and normalization"""
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)

        # Apply output normalization and gating
        y = self.output_norm(y)
        y = y * torch.sigmoid(self.output_gate(y))

        # Final projection and dropout
        return self.resid_dropout(self.c_proj(y))


class DilatedAttention(Attention):  # TOOO SLOW !!
    def __init__(
        self,
        config,
    ):
        """
        Implements the dilated attention mechanism from the LongNet paper.

        Args:
            config: Configuration object with attention parameters
            segment_sizes: List of window sizes for each dilated attention
            dilation_rates: List of dilation rates corresponding to each segment size
        """
        super().__init__(config)
        assert len(config.segment_sizes) == len(config.dilation_rates), (
            "Must provide same number of segment sizes and dilation rates"
        )

        self.segment_sizes = config.segment_sizes
        self.dilation_rates = config.dilation_rates

    def _get_dilated_indices(self, seq_len, segment_size, dilation_rate, num_heads):
        """
        Generate dilated indices for all heads at once.
        Ensures output size is consistent with the input sequence length.
        """
        # Calculate how many tokens per segment after dilation
        tokens_per_segment = math.ceil(segment_size / dilation_rate)

        # Calculate number of segments
        num_segments = math.ceil(seq_len / segment_size)

        # Initialize indices for all heads
        all_indices = []
        for head in range(num_heads):
            head_indices = []
            offset = head % dilation_rate

            # Generate indices for each segment
            for seg in range(num_segments):
                start_idx = seg * segment_size
                # Generate base indices for this segment
                seg_indices = torch.arange(
                    start_idx + offset,
                    min(start_idx + segment_size, seq_len),
                    dilation_rate,
                )
                head_indices.append(seg_indices)

            # Concatenate all segment indices
            head_indices = torch.cat(head_indices)

            # Pad or truncate to match sequence length
            if len(head_indices) < seq_len:
                padding = torch.full(
                    (seq_len - len(head_indices),),
                    seq_len - 1,
                    dtype=head_indices.dtype,
                )
                head_indices = torch.cat([head_indices, padding])
            else:
                head_indices = head_indices[:seq_len]

            all_indices.append(head_indices)

        # Stack indices for all heads
        return torch.stack(all_indices)  # (num_heads, seq_len)

    def _dilate_qkv(self, qkv, segment_size, dilation_rate):
        """
        Apply dilation to query, key, or value tensor while maintaining sequence length.
        """
        batch_size, num_heads, seq_len, head_dim = qkv.shape
        indices = self._get_dilated_indices(
            seq_len, segment_size, dilation_rate, num_heads
        )
        indices = indices.to(qkv.device)

        # Expand indices for batch and head dimensions
        expanded_indices = (
            indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        )

        # Gather along sequence length dimension
        return torch.gather(qkv, 2, expanded_indices)

    def _prepare_qkv(self, q, k, v):
        """
        Prepare query, key, and value tensors for dilated attention computation.
        """
        # First apply the standard preparation from parent class
        q, k, v = super()._prepare_qkv(q, k, v)

        dilated_outputs = []

        # Process each segment size and dilation rate
        for segment_size, dilation_rate in zip(self.segment_sizes, self.dilation_rates):
            # Apply dilation to q, k, v while maintaining sequence length
            q_dilated = self._dilate_qkv(q, segment_size, dilation_rate)
            k_dilated = self._dilate_qkv(k, segment_size, dilation_rate)
            v_dilated = self._dilate_qkv(v, segment_size, dilation_rate)

            dilated_outputs.append((q_dilated, k_dilated, v_dilated))

        return dilated_outputs

    def _flash_attention(self, dilated_qkv):
        """
        Compute attention using flash attention for each dilated version.
        """
        outputs = []
        attention_weights = []

        for q, k, v in dilated_qkv:
            # Compute attention with flash attention
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
            outputs.append(out)

            # Compute attention weights for dynamic weighting
            with torch.no_grad():
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (
                    1.0 / math.sqrt(k.size(-1))
                )
                attn_weights = attn_weights.softmax(dim=-1).mean(dim=(0, 1))
                attention_weights.append(attn_weights.max().item())

        # Compute dynamic weights based on attention scores
        weights = torch.softmax(
            torch.tensor(attention_weights, device=outputs[0].device), dim=0
        )

        # Combine outputs using dynamic weights
        combined_output = sum(w * out for w, out in zip(weights, outputs))
        return combined_output

    def _manual_attention(self, dilated_qkv, T):
        """
        Compute attention manually for each dilated version.
        """
        outputs = []
        attention_weights = []

        for q, k, v in dilated_qkv:
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.soft_cap > 0:
                attn = soft_cap(attn, self.soft_cap)

            # Create causal mask for dilated attention
            causal_mask = torch.ones_like(attn, dtype=torch.bool).triu_()
            attn = attn.masked_fill(~causal_mask, float("-inf"))

            # Compute attention weights
            attn_probs = F.softmax(attn, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)

            # Compute output
            out = attn_probs @ v
            outputs.append(out)

            # Store attention weights for dynamic weighting
            with torch.no_grad():
                attention_weights.append(attn_probs.mean(dim=(0, 1)).max().item())

        # Compute dynamic weights based on attention scores
        weights = torch.softmax(
            torch.tensor(attention_weights, device=outputs[0].device), dim=0
        )

        # Combine outputs using dynamic weights
        combined_output = sum(w * out for w, out in zip(weights, outputs))
        return combined_output

    def forward(self, x, q=None, mask=None):
        """
        Forward pass for dilated attention.
        """
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and positional encodings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare dilated attention inputs
        dilated_qkv = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0:
            y = self._flash_attention(dilated_qkv)
        else:
            y = self._manual_attention(dilated_qkv, T)

        # Project output
        return self._project_output(y, B, T_q, C)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):  # OOM
    def __init__(
        self,
        config,
        depth,
    ):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        # num_heads set to half of Transformer's #heads
        self.num_kv_heads = (
            config.n_kv_head if config.n_kv_head is not None else self.num_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)
        self.rotary = Rotary(self.head_dim)

        assert flash_attn_func, (
            "FlashAttention is not available. Please install it to use this module."
        )

    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        cos, sin = self.rotary(q)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn


if __name__ == "__main__":
    #
    # Testing - Forgetting Transformer Attention
    #

    # Simple configuration class for testing
    class Config:
        def __init__(self):
            # Basic model configuration
            self.n_embd = 128
            self.n_head = 4
            self.n_kv_head = 4
            self.block_size = 16
            self.dropout = 0.1
            self.bias = False
            self.use_qkv_bias = False
            self.use_soft_logit_capping = False
            self.rmsnorm_before_qk = True  # Enable for Pro version
            self.pos_encoding = None

    # Test both implementations
    config = Config()

    # Test original Forgetting Transformer
    print("\n=== Testing Forgetting Transformer ===")
    model = ForgettingAttention(config)
    batch_size, seq_length = 2, 8
    x = torch.randn(batch_size, seq_length, config.n_embd)

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")

    # Test Pro version
    print("\n=== Testing Forgetting Transformer Pro ===")
    pro_model = ForgettingTransformerPro(config)

    with torch.no_grad():
        pro_output = pro_model(x)

    print(f"Pro output shape: {pro_output.shape}")

    # Compare some statistics
    print("\n=== Comparison ===")
    print(f"Standard mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
    print(
        f"Pro mean: {pro_output.mean().item():.4f}, std: {pro_output.std().item():.4f}"
    )

    #
    # Testing - Multi-Token Attention
    #

    # Create a test configuration
    config = SimpleNamespace(
        n_embd=384,  # Embedding dimension
        n_head=12,  # Number of attention heads
        n_kv_head=12,  # Number of key-value heads (same as n_head for standard attention)
        block_size=1024,  # Maximum sequence length
        dropout=0.1,  # Dropout rate
        bias=False,  # Whether to include bias in linear projections
        use_qkv_bias=False,  # Whether to include bias in QKV projections
        pos_encoding="rotary",  # Position encoding type
        rope_theta=10000,  # RoPE base
        rmsnorm_before_qk=False,  # Whether to apply RMSNorm before QK projections
        use_soft_logit_capping=False,  # Soft logit capping setting
        # MTA specific parameters
        use_key_query_conv=True,  # Whether to use key-query convolution
        use_head_conv=True,  # Whether to use head convolution
        use_group_norm=True,  # Whether to use group normalization
        mta_query_kernel_size=6,  # Query dimension for convolution kernel
        mta_key_kernel_size=11,  # Key dimension for convolution kernel
        mta_head_kernel_size=2,  # Head dimension for convolution kernel
        pre_softmax_key_query=True,  # Whether to apply key-query convolution before softmax
        pre_softmax_head=False,  # Whether to apply head convolution before softmax
    )

    # Create a dummy Rotary class for testing (since it's used in parent class)
    class MockRotary(nn.Module):
        def __init__(self, dim, base=10000, seq_len=1024):
            super().__init__()
            self.base = base
            self.dim = dim
            self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            self.register_buffer("cos", freqs.cos().float())
            self.register_buffer("sin", freqs.sin().float())

        def forward(self, x):
            return self.cos[None, : x.size(-3), None, :], self.sin[
                None, : x.size(-3), None, :
            ]

    # Create a mock apply_rotary_emb function
    def mock_apply_rotary_emb(x, cos, sin):
        # Just return x without modification for testing
        return x

    # Override the necessary methods for testing
    Attention.rotary = None
    MultiTokenAttention._apply_rotary = lambda self, q, k, T_q, T: (q, k)

    # Print test configuration
    print("Testing Multi-Token Attention with configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    # Create model
    try:
        model = MultiTokenAttention(config)
        print("\nModel created successfully.")

        # Generate random input
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        # Forward pass
        print(f"\nRunning forward pass with input shape: {x.shape}")
        output = model(x)

        # Verify output
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, config.n_embd), (
            "Output shape doesn't match input shape"
        )

        # Test with different key-query and head convolution settings
        test_configs = [
            {
                "pre_softmax_key_query": True,
                "pre_softmax_head": False,
                "label": "Pre-softmax key-query, Post-softmax head (paper's best)",
            },
            {
                "pre_softmax_key_query": False,
                "pre_softmax_head": False,
                "label": "Post-softmax key-query, Post-softmax head",
            },
            {
                "pre_softmax_key_query": True,
                "pre_softmax_head": True,
                "label": "Pre-softmax key-query, Pre-softmax head",
            },
            {
                "pre_softmax_key_query": False,
                "pre_softmax_head": True,
                "label": "Post-softmax key-query, Pre-softmax head",
            },
        ]

        print("\nTesting different convolution configurations:")
        for test_config in test_configs:
            # Update the model configuration
            model.pre_softmax_key_query = test_config["pre_softmax_key_query"]
            model.pre_softmax_head = test_config["pre_softmax_head"]

            # Forward pass
            output = model(x)
            print(f"  ✓ {test_config['label']} - Output shape: {output.shape}")

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
