# NorMuon: Normalized Muon Optimizer

*Newton-Schulz orthogonalization + adaptive normalization with norm preservation.*

---

## What is NorMuon?

NorMuon extends the [Muon optimizer](https://github.com/KellerJordan/Muon) by adding **adaptive normalization through second-moment tracking** while **preserving update norms**. It maintains Muon's geometric optimization properties through Newton-Schulz orthogonalization while introducing variance-based step size adaptation.

The key innovation: apply second-moment normalization to orthogonalized updates element-wise, then restore the original update magnitude. This creates adaptive per-parameter scaling without changing overall update magnitudes.

**Compared to related optimizers:**
- **Muon**: Geometric conditioning, uniform scaling
- **AdaMuon**: Geometric conditioning + adaptive scaling (changes magnitudes via RMS rescaling)
- **NorMuon**: Geometric conditioning + adaptive scaling + norm preservation

## How Does It Work?

NorMuon's update algorithm:

### 1. Momentum Accumulation
```
Mt = β₁ · Mt-1 + (1 - β₁) · Gt
```

### 2. Optional Nesterov Acceleration
```
Update = Gt + β₁ · Mt  (or just Mt without Nesterov)
```

### 3. Newton-Schulz Orthogonalization
```
Ot = Newton-Schulz(Update, T=5)
```
Transforms updates into orthogonal matrices for well-conditioned optimization.

### 4. Norm-Preserving Adaptive Normalization
```
vnorm_orig = ||Ot||                         # Store original norm
v_mean = mean(Ot²)                          # Element-wise variance
Vt = β₂ · Vt-1 + (1 - β₂) · v_mean          # Second momentum
step_size = 1 / (√Vt + ε)                   # Adaptive per-element scaling
Ot' = Ot · step_size                        # Apply scaling
vnorm_new = ||Ot'||                         # New norm after scaling
Ot_final = Ot' · (vnorm_orig / vnorm_new)   # Restore original norm
```

**Key insight**: Adaptive scaling happens per-element based on variance, but the overall update magnitude stays constant.

### 5. Dimensional Adjustment
```
Final_update = Ot_final · √max(1, m/n)
```

## Why Norm Preservation Matters

The norm restoration step differentiates NorMuon from AdaMuon:

```python
update.mul_(original_norm / new_norm)  # After adaptive scaling, restore norm
```

**Benefits:**
- Consistent learning dynamics throughout training
- Predictable update magnitudes for LR scheduling
- Better late-stage stability when variances are established
- Adaptive per-element scaling without magnitude drift

**Without norm preservation**: Adaptive scaling can unpredictably shrink/expand updates, requiring careful LR tuning.

**With norm preservation**: Get adaptive benefits while maintaining predictable training behavior.

## Usage

Train with NorMuon via command line:

```bash
# Multi-GPU training
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
    --run_name normuon_exp --model_name blagpt --optimizer_name normuon

# Single-GPU
python bla_gpt/train.py --run_name normuon_test --model_name blagpt --optimizer_name normuon
```

Or via optimizer registry:

```python
from bla_gpt.optimizers import get_optimizer

optimizer = get_optimizer(
    optimizer_name="normuon",
    optimizer_params={
        "momentum": 0.95,      # β₁ for first momentum
        "beta2": 0.95,         # β₂ for variance tracking
        "weight_decay": 0.01,
        "nesterov": True
    },
    lr=0.02,
    model=model
)
```

**Parameter separation** (handled automatically):
- **NorMuon**: 2D+ weight matrices (excluding embeddings/lm_head)
- **AdamW**: 1D parameters (biases, norms) + embeddings/lm_head

## When to Use NorMuon

**✅ Good for:**
- Large-scale transformer training with heterogeneous parameters
- Long training runs requiring late-stage stability
- Distributed training (built-in multi-GPU support)
- When you want adaptive scaling without magnitude unpredictability

**❌ Consider alternatives for:**
- Small batch sizes (< 16) - Muon family works best with larger batches
- Fine-tuning where Adam's aggressive adaptivity may be preferable
- Memory-constrained settings (requires additional variance buffers)

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.02 | Learning rate for NorMuon parameters |
| `momentum` | 0.95 | First momentum coefficient (β₁) |
| `beta2` | 0.95 | Second momentum for variance tracking (β₂) |
| `weight_decay` | 0.01 | L2 regularization |
| `nesterov` | True | Enable Nesterov acceleration |
| `ns_steps` | 5 | Newton-Schulz iterations |

**Memory**: ~1× parameter size additional memory (comparable to Adam/AdamW).

## Implementation Notes

- **Distributed support**: `NorMuon` class handles multi-GPU synchronization
- **Single-device**: `SingleDeviceNorMuon` for non-distributed training
- **Hybrid optimizer**: `HybridNorMuon` combines NorMuon (weights) + AdamW (biases/norms)
- BlaGPT uses `HybridNorMuon` by default via the optimizer registry

---

**Original Code**: https://github.com/zichongli5/NorMuon
**Implementation**: `/bla_gpt/optimizers/normuon.py`
**Related**: [Muon](https://github.com/KellerJordan/Muon), [AdaMuon](./adamuon.md)
**Status**: Production-ready with distributed training support
