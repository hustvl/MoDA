from typing import Optional
import torch
from torch import nn
from bla_gpt.attentions import apply_rotary_emb
from bla_gpt.norms import LayerNorm
from einops import rearrange


import torch
from typing import Tuple


def fp8_index_pytorch(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of fp8_index algorithm.

    Computes index scores for attention routing without FP8 quantization.

    Args:
        q (torch.Tensor): Query tensor of shape (batch, m, n_heads, head_dim)
        q_s (torch.Tensor): Query scale/weights of shape (batch, m, n_heads)
        k (torch.Tensor): Key tensor of shape (batch, n, head_dim)
        k_s (torch.Tensor): Key scales of shape (batch, n)

    Returns:
        torch.Tensor: Index scores of shape (batch, m, n)

    Algorithm:
        1. Compute logits = q @ k.T for each head
        2. Apply ReLU(logits) * q_s (scale by query weights)
        3. Sum over heads dimension
        4. Multiply by k_s scales
    """
    # Validate input shapes
    b, m, h, d = q.shape
    assert q_s.shape == (b, m, h), f"q_s shape {q_s.shape} doesn't match expected {(b, m, h)}"
    assert k.shape[0] == b and k.shape[2] == d, f"k shape {k.shape} incompatible with q shape {q.shape}"
    assert k_s.shape[0] == b and k_s.shape[1] == k.shape[1], f"k_s shape {k_s.shape} incompatible with k shape {k.shape}"

    n = k.shape[1]

    # Step 1: Compute q @ k.T for all heads
    # q: (b, m, h, d), k: (b, n, d) -> logits: (b, m, h, n)
    logits = torch.einsum('bmhd,bnd->bmhn', q, k)

    # Step 2: Apply ReLU and scale by query weights
    # logits: (b, m, h, n), q_s: (b, m, h) -> (b, m, h, n)
    logits = torch.relu(logits) * q_s.unsqueeze(-1)

    # Step 3: Sum over heads dimension
    # (b, m, h, n) -> (b, m, n)
    logits_sum = logits.sum(dim=2)

    # Step 4: Multiply by key scales
    # logits_sum: (b, m, n), k_s: (b, n) -> (b, m, n)
    index_score = logits_sum * k_s.unsqueeze(1)

    return index_score


def act_quant_pytorch(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of block-wise quantization to FP8.

    This is a reference implementation that mimics the quantization behavior
    without actually converting to FP8 (keeps data in original dtype).

    Args:
        x (torch.Tensor): Input tensor with last dimension divisible by block_size
        block_size (int): Size of blocks for quantization (default: 128)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Quantized tensor (clipped to FP8 range but not converted to FP8)
            - Scale factors of shape (*x.shape[:-1], x.shape[-1] // block_size)

    Note: This function simulates quantization for testing purposes but doesn't
    actually use FP8 dtype. For real FP8 quantization, use the TileLang version.
    """
    assert x.size(-1) % block_size == 0, (
        f"Last dimension {x.size(-1)} must be divisible by block_size {block_size}"
    )

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Reshape to separate blocks
    # (..., N) -> (..., N // block_size, block_size)
    shape = x.shape
    x_blocked = x.view(*shape[:-1], shape[-1] // block_size, block_size)

    # Compute absolute max per block
    # (..., num_blocks, block_size) -> (..., num_blocks)
    amax = x_blocked.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=1e-4)

    # Compute scale factors
    scale = amax * fp8_max_inv

    # Quantize: x / scale, clipped to FP8 range
    # scale: (..., num_blocks) -> (..., num_blocks, 1) for broadcasting
    x_quantized = x_blocked / scale.unsqueeze(-1)
    x_quantized = torch.clamp(x_quantized, fp8_min, fp8_max)

    # Reshape back to original shape
    x_quantized = x_quantized.view(shape)

    return x_quantized, scale



def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate activation using Hadamard transform."""
    # For compatibility, check if bf16
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    try:
        from fast_hadamard_transform import hadamard_transform
        hidden_size = x.size(-1)
        return hadamard_transform(x, scale=hidden_size ** -0.5)
    except ImportError:
        # Fallback: identity (or could use FFT-based Hadamard)
        print("Warning: fast_hadamard_transform not available, using identity")
        return x


class Indexer(torch.nn.Module):

    def __init__(self, args, block_size):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.block_size = block_size

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        # Full precision cache (bf16 instead of fp8)
        self.register_buffer(
            "k_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.bfloat16),
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Query projection
        q = self.wq_b(qr)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)

        # Apply RoPE
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Key projection
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Rotate activations
        q = rotate_activation(q)
        k = rotate_activation(k)

        # Update cache (no quantization)
        self.k_cache[:bsz, start_pos:end_pos] = k.to(torch.bfloat16)

        # Compute weights
        weights = self.weights_proj(x) * self.n_heads ** -0.5
        # Note: No quantization scales needed for PyTorch version
        weights = weights.unsqueeze(-1) * self.softmax_scale

        # Create k_s (all ones since no quantization)
        k_s = torch.ones(bsz, end_pos, dtype=torch.float32, device=x.device)

        # Compute index scores using pure PyTorch
        index_score = fp8_index_pytorch(
            q,
            weights,
            self.k_cache[:bsz, :end_pos],
            k_s
        )

        # Apply mask if provided
        if mask is not None:
            index_score = index_score + mask

        # Get top-k indices
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]

        return topk_indices
