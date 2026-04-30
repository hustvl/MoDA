# Cautious Weight Decay (CWD)

*A weight decay variant that applies regularization selectively based on momentum-parameter sign alignment.*

---

## Overview

Cautious Weight Decay is a modification to standard decoupled weight decay that applies regularization only when the optimizer's momentum and the parameter have the same sign. This selective application preserves the original optimization objective while standard weight decay implicitly optimizes a regularized objective.

## Method

### Standard Weight Decay
```
x_{t+1} = x_t - η_t(u_t + λx_t)
```

### Cautious Weight Decay
```
x_{t+1} = x_t - η_t(u_t + λ·I(u_t ⊙ x_t ≥ 0) ⊙ x_t)
```

Where:
- `u_t` is the optimizer update (momentum for SGD/Muon, bias-corrected gradient for Adam)
- `λ` is the weight decay coefficient
- `I(·)` is an element-wise indicator function
- `⊙` denotes element-wise multiplication

The key difference is the mask `I(u_t ⊙ x_t ≥ 0)`, which equals 1 when signs align and 0 otherwise.

## Implementation

CWD is implemented for the Muon optimizer in BlaGPT, covering both Muon parameters (2D weights) and AdamW backup parameters (1D biases/norms).

### Algorithm

For each parameter update:
1. Compute momentum buffer (standard optimizer step)
2. Calculate sign-alignment mask: `mask = (momentum * param >= 0).float()`
3. Apply selective weight decay: `param *= (1 - lr * wd * mask)`
4. Apply optimizer update

The implementation uses biased momentum (raw momentum buffer) rather than bias-corrected values, as specified in the paper.

### Usage

Enable via optimizer configuration:

```python
# In model config
optimizer_args = {
    "wd": 0.1,
    "momentum": 0.95,
    "use_cautious_weight_decay": True,
}
```

Or in training config:
```python
from bla_gpt.train import Hyperparameters

args = Hyperparameters(
    optimizer_name="Muon",
    optimizer_args={
        "wd": 0.1,
        "use_cautious_weight_decay": True,
    }
)
```

## Performance Characteristics

### Computational Overhead
- One element-wise comparison per parameter
- One element-wise multiplication per parameter
- Temporary mask tensor (freed each iteration)
- Expected overhead: <0.5%

### Expected Improvements
Based on paper results:
- Validation loss: 0.3-1% improvement over standard weight decay
- Mask activation rate: ~40-50% of coordinates after warmup
- Training stability: Equal or better than baseline

---

**Paper**: [Cautious Weight Decay](https://arxiv.org/abs/2510.12402)
**Implementation**: `/bla_gpt/optimizers/muon.py`
