# AdaMuon: Adaptive Muon Optimizer

*An enhanced version of Muon that combines geometric optimization with adaptive per-parameter scaling.*

---

## 🤔 What is AdaMuon?

AdaMuon is a state-of-the-art optimizer that builds upon the recently introduced [Muon optimizer](https://github.com/KellerJordan/Muon), adding **adaptive second-moment estimation** to make it more robust and efficient for large-scale transformer training.

While Muon revolutionized optimization by using Newton-Schulz orthogonalization to create well-conditioned updates, it lacked the adaptive scaling that makes optimizers like Adam so effective. AdaMuon bridges this gap by applying variance-aware scaling **after** orthogonalization, giving you the best of both worlds.

## 🔬 How Does It Work?

AdaMuon follows a carefully designed 5-step process:

### 1. **Standard Momentum Update**
```
Mt = β₁ · Mt-1 + (1 - β₁) · Gt
```
Just like SGD with momentum, but this is just the beginning.

### 2. **Newton-Schulz Orthogonalization**
```
Ot = Newton-Schulz(Mt, T=5)
```
This is Muon's key innovation - transforms the momentum into an orthogonal matrix that preserves geometric properties while ensuring well-conditioned updates.

### 3. **Second-Moment Estimation on Orthogonalized Updates**
```
vt = β₂ · vt-1 + (1 - β₂) · ot²
```
**Key insight**: Apply adaptive scaling to the *clean* orthogonalized signal, not the noisy raw gradients.

### 4. **Element-wise Adaptive Scaling**
```
ôt = ot / (√vt + ε)
```
Scale each parameter based on its historical variance, just like Adam, but on the orthogonal updates.

### 5. **RMS-Aligned Rescaling**
```
Final_update = ôt · (0.2√min(n,m)) / (||ôt|| + ε)
```
Ensures compatibility with existing Adam learning rate schedules - no hyperparameter retuning needed!

## 💡 Why Use AdaMuon?

### **The Problem with Vanilla Optimizers**
- **SGD**: Fast but sensitive to learning rates and poorly conditioned
- **Adam**: Adaptive but can have poor conditioning in high dimensions
- **Muon**: Well-conditioned but uniform scaling doesn't handle parameter heterogeneity

### **AdaMuon's Solution**
✅ **Well-Conditioned Updates** (from Muon's orthogonalization)  
✅ **Adaptive Per-Parameter Scaling** (from Adam-style second moments)  
✅ **Learning Rate Compatibility** (from RMS-aligned rescaling)  
✅ **Better Noise Handling** (adaptive scaling on clean signals)

### **Quantified Benefits**

Based on experiments across GPT-2 model scales (125M to 1.5B parameters):

| Improvement | AdaMuon vs Muon | AdaMuon vs AdamW |
|-------------|------------------|------------------|
| **Convergence Speed** | 3-5% faster | 16-42% fewer tokens |
| **Training Efficiency** | 2-5% less wall-clock time | Up to 26% time reduction |
| **Stability** | More robust to hyperparameters | Better late-stage training |

## 🚀 Implementation in BlaGPT

AdaMuon is now the **default optimizer** in BlaGPT for maximum performance out of the box.

### **Basic Usage**
```python
# Already configured as default!
config = GPTConfig()  # Uses AdaMuon automatically
```

### **Direct Usage**
```python
from bla_gpt.optimizers.adamuon import AdaMuon

optimizer = AdaMuon(
    lr=1e-3,                    # Learning rate
    betas=(0.9, 0.95),          # Momentum coefficients  
    wd=0.1,                     # Weight decay
    muon_params=muon_params,    # 2D parameters (matrices)
    adamw_params=adamw_params   # 1D parameters (biases, norms)
)
```

### **Via Optimizer Registry**
```python
from bla_gpt.optimizers import get_optimizer

optimizer = get_optimizer(
    optimizer_name="adamuon", 
    optimizer_params={"betas": (0.9, 0.95), "wd": 0.1},
    lr=1e-3,
    model=model
)
```

## 📊 When to Use AdaMuon

### **✅ Great for:**
- **Large-scale transformer training** (where it was tested)
- **Heterogeneous parameter structures** (attention + MLP layers)
- **Long training runs** (prevents late-stage stagnation)
- **When you want Adam-level adaptivity with better conditioning**

### **❓ Consider alternatives for:**
- **Very small batch sizes** (Muon family optimizers prefer larger batches)
- **Fine-tuning pretrained models** (less tested, though likely works well)
- **Extremely memory-constrained scenarios** (slightly more memory overhead)

## 🔍 Key Insights

### **Why Apply Second-Moment Estimation After Orthogonalization?**

The paper's key insight is that applying adaptive scaling to **orthogonalized updates** rather than raw gradients is more effective:

- **Raw gradients** are noisy and ill-conditioned
- **Orthogonalized updates** are clean, normalized signals  
- **Variance estimation** on clean signals is more meaningful and stable

This is a fundamental difference from other optimizers and explains why AdaMuon outperforms naive combinations of Muon + Adam-style scaling.

### **RMS-Aligned Rescaling Magic**

The RMS rescaling step is crucial for practical deployment:
- Makes AdaMuon a **drop-in replacement** for Adam in existing codebases
- Prevents **vanishing updates** that can occur with aggressive second-moment dampening
- Maintains **consistent update magnitudes** throughout training

## 📈 Performance Tips

1. **Use default hyperparameters** - they're well-tuned from the paper's experiments
2. **Stick with larger batch sizes** (≥32) for best performance  
3. **No need to retune learning rates** from your Adam configurations
4. **Monitor early training** - benefits should be visible within first few hundred steps

## 🎯 Bottom Line

AdaMuon represents the evolution of optimization for modern deep learning:

> **"If Adam taught us the value of adaptive scaling, and Muon showed us the power of geometric conditioning, AdaMuon proves we can have both."**

It's particularly powerful for transformer architectures where different parameter matrices (Q, K, V projections vs MLP weights) have vastly different gradient characteristics. The combination of Muon's orthogonalization with Adam-style adaptivity makes it especially suited for the heterogeneous parameter landscape of modern language models.

---

**Paper**: [AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005)  
**Implementation**: `/bla_gpt/optimizers/adamuon.py`  
**Status**: Production-ready, default optimizer in BlaGPT