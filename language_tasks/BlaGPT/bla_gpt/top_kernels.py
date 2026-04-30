"""
Optimized Triton kernels for Token Order Prediction (TOP).
Based on official TOP repository: https://github.com/zaydzuhri/token-order-prediction

These kernels provide significant performance improvements for TOP target generation
and ListNet loss computation on GPU.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_V': 256}, num_warps=1),
            triton.Config({'BLOCK_SIZE_V': 512}, num_warps=2),
            triton.Config({'BLOCK_SIZE_V': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE_V': 2048}, num_warps=8),
        ],
        key=['vocab_size']
    )
    @triton.jit
    def seq_to_top_kernel(
        seq_ptr, out_ptr,
        batch_size, total_len, seq_len, vocab_size, window_size,
        seq_stride_b, seq_stride_t,
        out_stride_b, out_stride_t, out_stride_v,
        BLOCK_SIZE_V: tl.constexpr
    ):
        """Triton kernel for sequence-to-TOP conversion."""
        # Get program IDs
        pid_b = tl.program_id(0)  # batch dimension
        pid_v = tl.program_id(1)  # vocabulary dimension
        
        # Calculate vocabulary block range
        v_start = pid_v * BLOCK_SIZE_V
        v_end = min(v_start + BLOCK_SIZE_V, vocab_size)
        v_range = tl.arange(0, BLOCK_SIZE_V)
        v_mask = v_range < (v_end - v_start)
        v_indices = v_start + v_range
        
        # Initialize next occurrence positions to total_len (infinity)
        next_occurrence = tl.full((BLOCK_SIZE_V,), total_len, dtype=tl.int32)
        
        # Process sequence in reverse
        for t in range(total_len - 1, -1, -1):
            # Load token at position t for current batch
            token_ptr = seq_ptr + pid_b * seq_stride_b + t * seq_stride_t
            token = tl.load(token_ptr)
            
            # Update next occurrence if token matches vocabulary indices
            token_match = (v_indices == token) & v_mask & (token >= 0) & (token < vocab_size)
            next_occurrence = tl.where(token_match, t, next_occurrence)
            
            # Compute scores for output positions
            if t < seq_len:
                # Calculate distances
                distances = next_occurrence - t
                
                # Create window mask and compute scores
                window_mask = (distances > 0) & (distances <= window_size) & v_mask
                scores = tl.where(window_mask, window_size - distances, float('-inf'))
                
                # Store scores to output
                out_ptrs = (out_ptr + pid_b * out_stride_b + t * out_stride_t + 
                           v_indices * out_stride_v)
                tl.store(out_ptrs, scores.to(tl.float32), mask=v_mask)


def seq_to_top_triton(seq, vocab_size, window_size):
    """
    Triton-optimized sequence-to-TOP conversion.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len + window_size)
        vocab_size: Size of vocabulary
        window_size: Window size for proximity scoring
        
    Returns:
        TOP target tensor of shape (batch_size, seq_len, vocab_size) with proximity scores
    """
    if not HAS_TRITON:
        # Fallback to naive implementation
        from losses import construct_top_targets
        return construct_top_targets(seq, vocab_size, window_size)
    
    device = seq.device
    
    # Only use Triton on CUDA tensors
    if device.type != 'cuda':
        from losses import construct_top_targets
        return construct_top_targets(seq, vocab_size, window_size)
    
    batch_size, total_len = seq.shape
    seq_len = total_len - window_size
    
    if seq_len <= 0:
        return torch.full((batch_size, 1, vocab_size), float('-inf'), device=device)
    
    # Ensure input is contiguous and on CUDA
    seq = seq.contiguous()
    
    # Initialize output tensor
    out = torch.empty((batch_size, seq_len, vocab_size), device=device, dtype=torch.float)
    
    # Calculate grid dimensions
    grid = (batch_size, triton.cdiv(vocab_size, 256))  # Assuming BLOCK_SIZE_V=256 default
    
    # Launch kernel
    seq_to_top_kernel[grid](
        seq, out,
        batch_size, total_len, seq_len, vocab_size, window_size,
        seq.stride(0), seq.stride(1),
        out.stride(0), out.stride(1), out.stride(2)
    )
    
    return out


# Conditional import based on Triton availability
if HAS_TRITON:
    # Use optimized version when Triton is available
    construct_top_targets_optimized = seq_to_top_triton
else:
    # Use naive version as fallback
    from .losses import construct_top_targets as construct_top_targets_optimized


def get_top_target_fn(force_optimized=True):
    """Get the best available TOP target construction function."""
    if HAS_TRITON and torch.cuda.is_available():
        return seq_to_top_triton  # This function handles device checking internally
    elif force_optimized:
        raise RuntimeError(
            "Optimized TOP implementation requested but requirements not met. "
            f"Triton available: {HAS_TRITON}, CUDA available: {torch.cuda.is_available()}. "
            "Set top_force_optimized=False to use fallback implementation."
        )
    else:
        print("Warning: Triton not available or not on CUDA, using tensor-optimized implementation")
        # Import the optimized tensor-based version from losses
        from losses import construct_top_targets
        return construct_top_targets