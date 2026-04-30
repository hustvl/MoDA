# ResFormer: Value Residual Learning

*An architecture that adds residual connections from the first layer's values to all subsequent layers.*

---

## 🤔 What is ResFormer?

ResFormer is a transformer variant introduced by Zhou et al. (2025) that adds value residual connections from the first attention layer to all subsequent layers. The modification preserves token-level information from initial embeddings throughout the network, reducing over-smoothing and attention concentration effects.

The paper reports training efficiency improvements: models achieve equivalent validation loss with 16% fewer parameters and 20% less training data compared to standard transformers.

## 🔬 How Does It Work?

ResFormer uses a 3-step process:

### 1. **First Layer Computes and Caches V₁**
```
V₁ = H₀ · W^V₁
```
The first attention layer computes value vectors normally and stores them for later use. V₁ is detached from the computation graph to avoid memory overhead.

### 2. **Subsequent Layers Mix V₁ with Current Values**
```
V'ₙ = λ₁ · V₁ + λ₂ · Vₙ
```
For layers 2 and beyond, the current layer's value vectors (Vₙ) are mixed with the cached first-layer values (V₁) before attention.

### 3. **Shared Attention Matrix**
```
output = Attention(Q, K, V'ₙ)
```
Both value components share the current layer's attention matrix. The mixing happens before attention computation, not after.

## 💡 Why Use ResFormer?

### **Issues with Standard Transformers**

Research shows that deeper transformer layers experience:
- **Over-smoothing**: Token representations become increasingly similar
- **Information dilution**: Initial token-level features are lost
- **Attention concentration**: Attention focuses on few tokens (attention sink)

### **ResFormer's Approach**

✅ **Preserves Token Information** (V₁ retains initial embeddings)
✅ **Reduces Over-Smoothing** (maintains token diversity in deeper layers)
✅ **Improves Information Flow** (direct path from layer 1 to all layers)
✅ **Minimal Overhead** (<5% slower, similar memory)

### **Measured Results**

From experiments on 468M parameter models trained on 20B tokens (SlimPajama):

| Model | Val Loss | Params for Equal Loss | Data for Equal Loss |
|-------|----------|----------------------|---------------------|
| Baseline | 2.739 | 468M | 20B tokens |
| ResFormer-Plus | 2.681 | 392M (-16%) | 16B tokens (-20%) |

Downstream task improvements (zero-shot):
- Average: +1.7 points across benchmarks
- HellaSwag: +1.5 points
- OpenBookQA: +1.2 points
- WinoGrande: +1.3 points

## 🚀 Implementation in BlaGPT

ResFormer is available in BlaGPT with five variants differing in how lambda coefficients are determined.

### **Basic Usage**
```python
from bla_gpt.resformer import ResFormerPlusConfig, ResFormer

# Best variant (learnable-plus)
config = ResFormerPlusConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
)

model = ResFormer(config)
```

### **Via Training Command**
```bash
# Best performance
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
  --run_name resformer --model_name resformer_plus

# Best tradeoff (performance vs simplicity)
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
  --run_name resformer --model_name resformer_sparse

# Simplest (no hyperparameters)
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
  --run_name resformer --model_name resformer_constant
```

### **Available Variants**

| Variant | Lambda Values | Parameters | Performance | Use Case |
|---------|--------------|------------|-------------|----------|
| `resformer` | λ₁=0.5, λ₂=0.5 (fixed) | +0 | +1.0% | Testing, first time |
| `resformer_constant` | λ₁=2.0, λ₂=1.0 (fixed) | +0 | +1.4% | Production, simple |
| `resformer_learnable` | Both trainable | +2N | +1.2% | Experimentation |
| `resformer_plus` | Softmax distribution | +N+1 | +2.1% | Best performance |
| `resformer_sparse` | Last 1/3 layers only | +0 | +1.9% | Best tradeoff |

Where N is the number of layers. For a 12-layer model, the additional parameters are negligible (<0.01%).

## 📊 When to Use ResFormer

### **✅ Suitable for:**

- **Models with 12+ layers** (benefits increase with depth)
- **When parameter efficiency matters** (16% reduction for same loss)
- **Data-constrained settings** (20% less data needed)
- **Production deployments** (minimal overhead)

### **❓ Consider alternatives for:**

- **Very shallow models** (<8 layers) (limited benefit)
- **When exact baseline replication needed** (changes architecture)
- **Extremely latency-critical inference** (adds 1-5% overhead)

## 🔍 Key Insights

### **Why V₁ Instead of H₀?**

The paper tested residual connections from different sources:

| Connection Source | Validation Loss | Notes |
|------------------|----------------|-------|
| Standard (none) | 2.739 | Baseline |
| Hidden state H₀ | 2.781 | Worse than baseline |
| Value vectors V₁ | 2.712 | Best performance |

V₁ works better because:
- Contains token-level information after one transformation
- Less likely to disrupt attention distributions than H₀
- Higher similarity to early hidden states than deeper ones

### **Why Not Connect to V₂ or Later?**

Testing showed that connections from V₂ onwards provide minimal benefit. This occurs because the original hidden residual already propagates H₁ information (which V₂ is computed from). The gap between V₁ and later layers is what needs bridging.

### **Sparse Pattern Discovery**

Ablation studies found that:
- Applying value residual to all layers: 2.709 loss
- Applying only to layers 6-8: 2.687 loss (better)

The sparse variant works better because:
- Early layers need to build abstract representations
- Deep layers benefit most from raw token information
- Stronger residual (λ₁=5.0) in later layers is effective

### **Comparison with Related Methods**

| Method | Modification Point | Performance | Overhead |
|--------|-------------------|-------------|----------|
| **ResFormer** | Values (before attention) | +2.1% | <5% |
| NeuTRENO | Values (after attention) | +1.2% | <5% |
| DenseFormer | Hidden states | +1.6% | 10-15% |

ResFormer modifies values before attention computation, which is more effective than post-attention modifications or hidden state connections.

## 📈 Variant Selection Guide

### **Choose `resformer_plus` when:**
- You want maximum performance
- Parameter efficiency is critical
- Training time allows optimization

### **Choose `resformer_sparse` when:**
- You want good performance with less complexity
- You prefer fixed hyperparameters
- You're unsure which variant to use

### **Choose `resformer_constant` when:**
- You want simplicity
- You need a stable baseline
- Production deployment with no tuning

## 🎯 Bottom Line

ResFormer adds a simple residual connection that addresses information flow limitations in deep transformers:

> *"By preserving first-layer value vectors throughout the network, ResFormer achieves equivalent performance with fewer parameters and less training data."*

The modification is compatible with existing transformer features (GQA, Flash Attention, RoPE) and adds minimal overhead. The sparse variant provides a good balance between performance improvement and implementation simplicity.

For most use cases, start with `resformer_sparse` (90% of the benefit with fixed hyperparameters) or `resformer_plus` (maximum performance with learnable distribution).

---

**Paper**: [Value Residual Learning](https://arxiv.org/abs/2410.17897)
**Authors**: Zhou et al. (2025)
**Implementation**: `/bla_gpt/resformer.py`
**Status**: Production-ready, all variants tested
