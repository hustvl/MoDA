"""
Loss functions for BlaGPT models.

This module contains various loss functions used in training:
- Standard cross-entropy loss utilities
- Z-loss regularization 
- Token Order Prediction (TOP) loss
- Multi-Token Prediction (MTP) loss utilities
"""

import torch
import torch.nn.functional as F
from attentions import soft_cap


def compute_z_loss(logits, dim=-1, eps=1e-20):
    """
    Compute z-loss regularization to prevent model from being too confident.

    Args:
        logits: Raw logits from model of shape [batch, seq_len, vocab_size]
        dim: Dimension along which to compute z-loss (usually vocab dimension)
        eps: Small constant for numerical stability

    Returns:
        z_loss: Scalar z-loss term to add to training loss
    """
    # Get log of the partition function (logsumexp)
    log_z = torch.logsumexp(logits, dim=dim, keepdim=True)

    # Compute mean of log_z squared
    z_loss = torch.square(torch.max(log_z, torch.zeros_like(log_z)))
    z_loss = torch.mean(z_loss)

    return z_loss


def construct_top_targets(seq, vocab_size, window_size):
    """
    Convert a token sequence to TOP target sequence using optimized tensor operations.
    Based on official implementation from TOP paper repository.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len + window_size)
        vocab_size: Size of vocabulary  
        window_size: Window size for proximity scoring
        
    Returns:
        TOP target tensor of shape (batch_size, seq_len, vocab_size) with proximity scores
    """
    batch_size, total_len = seq.shape
    seq_len = total_len - window_size
    device = seq.device
    dtype = seq.dtype
    
    if seq_len <= 0:
        return torch.full((batch_size, 1, vocab_size), float('-inf'), device=device)
    
    # Initialize output tensor with -inf
    out = torch.full((batch_size, seq_len, vocab_size), float('-inf'), device=device, dtype=torch.float)
    
    # Track next occurrence positions for all tokens
    next_occurrence = torch.full((batch_size, vocab_size), total_len, device=device, dtype=torch.long)
    
    # Process sequence in reverse to find next occurrences efficiently
    for t in range(total_len - 1, -1, -1):
        # Get tokens at position t for all sequences in batch
        tokens_at_t = seq[:, t]  # Shape: (batch_size,)
        
        # Create valid mask for tokens within vocabulary
        valid_mask = (tokens_at_t >= 0) & (tokens_at_t < vocab_size)
        
        # Update next occurrence positions using one-hot encoding
        if valid_mask.any():
            # Convert tokens to one-hot for efficient updating
            token_one_hot = F.one_hot(tokens_at_t, num_classes=vocab_size).float()  # (B, V)
            
            # Update next occurrence positions where tokens are valid
            update_mask = valid_mask.unsqueeze(1)  # (B, 1)
            next_occurrence = torch.where(
                update_mask & (token_one_hot > 0), 
                t, 
                next_occurrence
            )
        
        # Compute distances and scores for output positions
        if t < seq_len:
            # Calculate distances to next occurrence
            distances = next_occurrence - t  # (B, V)
            
            # Create window mask: 0 < distance <= window_size
            window_mask = (distances > 0) & (distances <= window_size)
            
            # Compute proximity scores: window_size - distance (closer = higher score)
            scores = torch.where(window_mask, window_size - distances, torch.tensor(float('-inf'), device=device))
            
            # Assign scores to output tensor
            out[:, t, :] = scores.float()
    
    return out


def listnet_loss(y_pred, y_true):
    """
    ListNet loss from "Learning to Rank: From Pairwise Approach to Listwise Approach".
    Official implementation from TOP paper repository.
    
    Args:
        y_pred: Model predictions of shape [*, slate_length] 
        y_true: Ground truth labels of shape [*, slate_length]
        
    Returns:
        Loss value as a torch.Tensor
    """
    return torch.mean(-torch.sum(
        F.softmax(y_true, dim=-1).nan_to_num(nan=0) * 
        F.log_softmax(y_pred, dim=-1), 
        dim=-1
    ))


def compute_top_loss(model, x, idx, targets):
    """
    Compute Token Order Prediction (TOP) loss for the given model and inputs.
    
    Args:
        model: The GPT model instance
        x: Hidden states from final transformer layer of shape (batch_size, seq_len, n_embd)
        idx: Input token sequence of shape (batch_size, seq_len)
        targets: Target token sequence of shape (batch_size, seq_len)
        
    Returns:
        TOP loss scalar or None if not applicable
    """
    if not (model.config.use_top and hasattr(model, "top_head")):
        return None
        
    # Create extended sequence for TOP target construction
    window_size = min(model.config.top_window_size, targets.size(1))
    
    if window_size <= 0:
        return None
        
    # Create extended sequence: current input + future targets (for window)
    extended_seq = torch.cat([idx, targets[:, :window_size]], dim=1)
    
    # Generate TOP targets - use optimized implementation  
    from top_kernels import get_top_target_fn
    force_optimized = getattr(model.config, 'top_force_optimized', True)
    target_fn = get_top_target_fn(force_optimized=force_optimized)
    top_targets = target_fn(extended_seq, model.config.vocab_size, window_size)
    
    # Compute TOP predictions (only for positions we have targets for)
    seq_len = min(x.size(1), top_targets.size(1))
    top_logits = model.top_head(x[:, :seq_len]).float()
    
    # Apply soft capping if enabled
    if model.soft_cap > 0.0:
        top_logits = soft_cap(top_logits, model.soft_cap)
    
    # Compute TOP loss
    top_loss = listnet_loss(top_logits, top_targets[:, :seq_len])
    
    return top_loss