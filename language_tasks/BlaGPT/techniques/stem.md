# STEM: Scaling Transformers with Embedding Modules

*Replace dense up-projections with token-indexed embedding tables for improved efficiency and performance.*

---

## Overview

STEM (Scaling Transformers with Embedding Modules) replaces the dense up-projection in transformer FFN layers with a lookup table indexed by input tokens. This modification reduces computation while improving performance on knowledge-intensive and long-context tasks.

The method addresses the computational cost of FFN layers, which typically account for two-thirds of model parameters and a significant portion of FLOPs. By replacing the up-projection matrix with an embedding table, STEM reduces active parameters per forward pass while maintaining or improving model quality.

## Method

### Architecture Modification

Standard SwiGLU FFN:
```
y = W_down(SiLU(W_gate(x)) ⊙ W_up(x))
```

STEM FFN:
```
y = W_down(SiLU(W_gate(x)) ⊙ U[token_id])
```

Where:
- `W_gate` and `W_down` remain as dense projections
- `W_up` is replaced with embedding table `U` of shape `[vocab_size, d_ff]`
- `U[token_id]` performs direct lookup instead of matrix multiplication
- Context-dependent modulation through `W_gate` is preserved

### Layer Selection

STEM selectively replaces FFN layers based on a specified ratio:

| Ratio | Replacement Pattern | Typical Usage |
|-------|-------------------|---------------|
| 1/3 | Every third layer (balanced) | Default, ROI: 1.08x |
| 1/2 | Every other layer | Best accuracy, ROI: 1.20x |
| Full | All layers except layer 0 | Best efficiency, ROI: 1.33x |

Layer 0 is never replaced, as it handles critical initial processing of raw token embeddings.

### Parameter Tradeoff

Per STEM layer (example: d=2048, d_ff=5632, vocab_size=50304):
- Dense W_up: 2048 × 5632 = 11.5M parameters
- STEM table U: 50304 × 5632 = 283M parameters
- Net increase: +271.5M parameters

Despite higher total parameters, only 11.5M fewer parameters are active per forward pass, reducing FLOPs by approximately 20-25% per STEM layer.

## Implementation in BlaGPT

### Configuration

```python
from bla_gpt import GPT, GPTConfig

config = GPTConfig()
config.use_stem = True
config.stem_ratio = "1/3"  # or "1/2" or "full"
config.activation = "swiglu"
```

Or specify exact layers:
```python
config.stem_layers = [2, 5, 8, 11]  # Custom selection
```

### Training

```bash
# Balanced (1/3 replacement)
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
    --run_name stem_1b --model_name stem

# Best accuracy (1/2 replacement)
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
    --run_name stem_half_1b --model_name stem_half

# Best efficiency (full replacement)
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
    --run_name stem_full_1b --model_name stem_full
```

---

**Paper**: [STEM: Scaling Transformers with Embedding Modules](https://arxiv.org/abs/2601.10639)
**Implementation**: `/bla_gpt/mlps.py`, `/bla_gpt/bla_gpt.py`
**Status**: Production-ready
