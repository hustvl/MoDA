# Token Order Prediction (TOP): Auxiliary Training Objective

*An auxiliary training loss that trains language models to rank upcoming tokens by their proximity in the sequence.*

---

## Overview

Token Order Prediction (TOP) is an auxiliary training objective introduced by Zuhri et al. (2024) as an alternative to Multi-Token Prediction (MTP). Rather than predicting exact future token values, TOP trains models to predict the relative ordering of upcoming tokens within a specified window based on their proximity to the current position.

The method addresses limitations observed in MTP, where exact future token prediction shows inconsistent improvements and requires substantial additional parameters (multiple transformer layers per predicted token).

## Method

### Target Construction

For each sequence position t, proximity scores are computed for all vocabulary tokens using Algorithm 1:

1. Iterate backwards through the sequence to track next occurrence positions
2. For each vocabulary token v, compute distance d to its next occurrence after position t
3. Assign proximity score: score = W - d (where W is window size)
4. Tokens with no occurrence within window W receive score -∞

### Architecture

The method adds a single linear projection layer (TOP head) parallel to the standard language modeling head:

```
hidden_state = transformer_layers(input)
ntp_logits = ntp_head(hidden_state)     # Next-token prediction
top_logits = top_head(hidden_state)     # Token order prediction
```

### Loss Function

Training optimizes a combined objective:
```
L_total = L_NTP + L_TOP
```

Where L_NTP is standard cross-entropy loss and L_TOP uses ListNet ranking loss to train the model to match target proximity distributions.

## Empirical Results

### Comparison with Baselines

Zuhri et al. (2024) evaluated TOP against NTP and MTP baselines across three model scales (340M, 1.8B, 7B parameters) on eight standard NLP benchmarks.

### Parameter Efficiency

TOP requires only a single additional unembedding matrix (hidden_dim × vocab_size parameters), compared to MTP which adds N transformer layers for N future tokens. For 4-token MTP, this represents approximately 97% parameter reduction while maintaining comparable or superior performance.

### Performance Across Scales

The authors report that TOP shows consistent improvements across model scales, unlike MTP which demonstrates benefits primarily for models exceeding 1-3B parameters. TOP improvements appear to scale monotonically with model size.

### Task Performance

Results on standard benchmarks show TOP outperforming both NTP and MTP baselines across most evaluated tasks, with particularly notable improvements on knowledge-intensive tasks like TriviaQA and reasoning tasks like ARC Challenge.

## Performance Characteristics

### Computational Optimization

The implementation includes Triton kernels for GPU acceleration of target construction:

| Configuration | Reference Implementation | Triton Kernel | Speedup Factor |
|---------------|-------------------------|---------------|----------------|
| 4×512×1K vocab | 159.67ms | 0.21ms | 752x |
| 8×1024×5K vocab | 320.64ms | 0.38ms | 843x |
| 16×2048×32K vocab | 654.22ms | 1.82ms | 360x |

### Implementation Efficiency

- Target construction uses vectorized tensor operations
- On-the-fly target generation during training
- Autotuned block sizes for different vocabulary dimensions

## Algorithm Details

### Target Construction Algorithm

The target construction follows Algorithm 1 from the paper:

```python
def construct_top_targets(seq, vocab_size, window_size):
    # Backward iteration to track next token occurrences
    for t in range(total_len - 1, -1, -1):
        tokens_at_t = seq[:, t]

        # Update next occurrence positions using one-hot encoding
        token_one_hot = F.one_hot(tokens_at_t, vocab_size)
        next_occurrence = torch.where(valid_mask, t, next_occurrence)

        # Compute proximity scores for output positions
        if t < seq_len:
            distances = next_occurrence - t
            scores = torch.where(in_window_mask, window_size - distances, -inf)
            out[:, t, :] = scores
```

### ListNet Loss Function

```python
def listnet_loss(y_pred, y_true):
    return torch.mean(-torch.sum(
        F.softmax(y_true, dim=-1).nan_to_num(nan=0) *
        F.log_softmax(y_pred, dim=-1),
        dim=-1
    ))
```

This loss function treats ranking as a probability matching problem, converting proximity scores to distributions and computing cross-entropy between target and predicted rankings.

## Usage Considerations

### Suitable Applications

- Language model pretraining across different scales
- Models with vocabulary sizes where ranking provides meaningful signal
- Training setups where auxiliary objectives are practical

### Observed Performance Patterns

Based on the original paper's experiments:
- ARC Challenge: 3.67% improvement at 1.8B parameters
- TriviaQA: 6.63% improvement at 7B parameters
- SciQ: 3.00% improvement at 7B parameters
- Consistent gains across multiple benchmark tasks

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_top` | `False` | Enable TOP auxiliary loss |
| `top_window_size` | `1024` | Proximity scoring window (≤ block_size) |
| `top_loss_weight` | `1.0` | Weight for TOP loss in total loss |
| `top_force_optimized` | `True` | Require Triton optimization |

### Constraints

- Window size must not exceed sequence block size
- Triton kernels provide optimal performance on CUDA devices
- Fallback implementations available for CPU execution

## Analysis

### Ranking vs Exact Prediction

The method is based on the hypothesis that learning relative token ordering is more tractable than exact future token prediction. The authors observe that MTP's performance degrades with increased look-ahead distance, suggesting that exact prediction becomes prohibitively difficult.

### Comparison with Multi-Token Prediction

| Aspect | MTP (4 tokens) | TOP |
|--------|----------------|-----|
| Additional Parameters | ~4.2B | ~24M |
| Computational Overhead | 4× transformer layers | 1× linear projection |
| Training Objective | Exact token prediction | Proximity ranking |
| Scaling Behavior | Inconsistent across tasks | Consistent improvements |

### Training Dynamics

The authors note that TOP models exhibit higher NTP training loss while achieving superior validation performance, suggesting the auxiliary objective may provide implicit regularization effects.


---

**Paper**: [Predicting the Order of Upcoming Tokens Improves Language Modeling](https://arxiv.org/abs/2508.19228)
**PDF**: https://arxiv.org/pdf/2508.19228v1
**Official Code**: https://github.com/zaydzuhri/token-order-prediction
**Implementation**: `/bla_gpt/losses.py`, `/bla_gpt/top_kernels.py`
**Status**: Production-ready with Triton optimization