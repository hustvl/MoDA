# Gated Attention: Non-linearity, Sparsity, and Attention-Sink-Free

*An attention mechanism that applies sigmoid gates after SDPA.*

---

## 🤔 What is Gated Attention?

Gated Attention is an attention mechanism from Qwen Team at Alibaba Group (May 2025) that adds learnable sigmoid gates after the Scaled Dot-Product Attention (SDPA) computation. The modification adds one linear projection per attention layer and shows improvements across model scales, reduces the "attention sink" phenomenon, and improves training stability.

The technique comes from a study testing 30 variants across 15B Mixture-of-Experts (MoE) models and 1.7B dense models, trained on 3.5 trillion tokens. Applying head-specific sigmoid gates after SDPA (the "G1 position") performed best among the tested variants.

## 🔬 How Does It Work?

Gated Attention uses a 4-step process:

### 1. **Standard SDPA Computation**
```
Y = softmax(Q·K^T / √d) · V
```
First, compute attention exactly as usual—nothing changes here.

### 2. **Gate Computation from Pre-Normalized Input**
```
gates = sigmoid(X · W_gate)
```
Compute gating scores from the input hidden states (after layer normalization). The gate projection W_gate has shape (d_model, d_model), enabling per-head, per-dimension gating.

### 3. **Reshape for Head-Specific Gating**
```
gates: (B, T, d_model) → (B, n_head, T, head_dim)
```
Reshape gates to match the multi-head attention structure, ensuring each attention head gets its own gate values.

### 4. **Multiplicative Gating**
```
Y' = Y ⊙ gates
```
Apply element-wise multiplication between attention output and sigmoid gates. The gates modulate the attention output before it's projected back to the residual stream.

### The "G1 Position" Advantage

The paper tested five different gating positions:
- **G4**: After query projection
- **G3**: After key projection
- **G2**: After value projection
- **G1**: After SDPA (before concatenation) ← **Best performance**
- **G5**: After final output projection

The G1 position performs best because it:
1. Introduces non-linearity between the value projection (W_V) and output projection (W_O), breaking up their otherwise low-rank composition
2. Applies query-dependent gating scores that filter attention outputs based on the current token's context

## 💡 Why Use Gated Attention?

### **Problems with Standard Attention**

Standard transformers have several attention-related issues:
- **Attention Sink**: Initial tokens disproportionately dominate attention scores (often 40-50%)
- **Low-Rank Bottleneck**: Consecutive linear layers (W_V → W_O) form a low-rank mapping
- **Training Instability**: Loss spikes during training, especially with large learning rates
- **Massive Activations**: Certain tokens accumulate extremely large activation values

### **Gated Attention's Solutions**

✅ **Non-Linearity**: Breaks the low-rank W_V·W_O composition with sigmoid gates
✅ **Query-Dependent Sparsity**: Creates input-dependent sparse gating (mean ~0.116)
✅ **Attention-Sink-Free**: Reduces first-token attention from 46.7% to 4.8%
✅ **Training Stability**: Reduces loss spikes, allows larger learning rates
✅ **Lower Peak Activations**: Reduces peak activations from 1053 to 94

### **Quantified Benefits**

Experimental results from 15B MoE models trained on 400B tokens:

| Metric | Improvement | Baseline → Gated |
|--------|-------------|------------------|
| **Validation PPL** | -0.265 | 6.026 → 5.761 |
| **MMLU** | +2.03 points | 58.79 → 60.82 |
| **GSM8k** | +2.35 points | 52.92 → 55.27 |
| **Hellaswag** | +1.57 points | 73.07 → 74.64 |
| **Attention Sink** | -41.9 percentage points | 46.7% → 4.8% |
| **Wall-time Overhead** | <2% | Minimal impact |

Results hold consistently across 1.7B dense models and remain robust across different scales.

## 🚀 Implementation in BlaGPT

Gated Attention is available in BlaGPT and can be enabled with a configuration change.

### **Basic Usage**
```python
from bla_gpt import GPT, GPTConfig

# Enable gated attention
config = GPTConfig(
    attention="gated",  # That's it!
    n_layer=12,
    n_head=12,
    n_embd=768,
)

model = GPT(config)
```

### **Via Training Command**
```bash
torchrun --standalone --nproc_per_node=8 bla_gpt/train.py \
  --run_name gated_experiment \
  --model_name blagpt \
  --config_override '{"attention": "gated"}'
```

### **Via JSON Config**
```json
{
  "attention": "gated",
  "n_layer": 12,
  "n_head": 12,
  "n_embd": 768
}
```

### **Architecture Details**

For each attention layer, Gated Attention adds:
- **Parameters**: d_model² (e.g., 768² = 590K params for d_model=768)
- **Memory**: One additional tensor of shape (B, n_head, T, head_dim) during forward pass
- **Computation**: One linear projection + sigmoid activation per attention call

The implementation:
- Works with **Grouped Query Attention (GQA)**
- Supports **flash attention** and **manual attention** paths
- Compatible with **RoPE**, **soft logit capping**, and other BlaGPT features
- Same inference speed (gates computed during forward pass)

## 📊 When to Use Gated Attention

### **✅ Suitable for:**

- **Large-scale pretraining** (tested on models up to 15B parameters)
- **Long-context applications** (reduces attention sink)
- **When training stability matters** (reduces loss spikes)
- **When attention sink is an issue** (46.7% → 4.8% first-token attention)
- **Models with 1B+ parameters** (benefits increase with model size)

### **❓ Consider alternatives for:**

- **Very small models (<100M params)** (overhead may outweigh benefits)
- **Inference-only applications** (training benefits don't apply)
- **Extremely memory-constrained scenarios** (adds ~2-3% memory overhead)
- **When you need to match baseline exactly** (changes attention behavior)

## 🔍 Key Insights

### **Why Does G1 Position Work Best?**

The paper compared five gating positions and found G1 (after SDPA) performed best for two reasons:

**1. Non-linearity in the Right Place**

In standard attention, the value and output projections form a low-rank bottleneck:
```
output = (attention_weights · V·W_V) · W_O
       = attention_weights · V · (W_V·W_O)  ← low-rank!
```

By inserting a sigmoid gate at G1, we break this composition:
```
output = (attention_weights · V·W_V · sigmoid(gate)) · W_O
```

This is more expressive than gating at other positions because it modulates the full attention-weighted values before the output projection.

**2. Query-Dependent Sparsity**

G1 gates are computed from the **current token's hidden state** (after pre-norm), making them query-dependent. This means:
- Different queries get different gating patterns
- Irrelevant attention weights are suppressed based on the query's context
- The model learns when to trust attention vs. when to dampen it

In contrast, G2 (value gating) depends on keys/values, not queries, making it less effective at filtering irrelevant context.

### **The Sparsity Mechanism**

Gated attention creates sparse attention outputs:
- **Mean gate value**: ~0.116 (after training with sigmoid)
- **Distribution**: Heavily skewed toward 0
- **Effect**: Most attention outputs are dampened

This sparsity is learned and input-dependent. The model learns which attention patterns to suppress based on the query token.

### **How It Eliminates Attention Sink**

The "attention sink" phenomenon (where early tokens receive disproportionate attention) has been explained as an artifact of softmax's non-negative normalization—the model needs somewhere to "dump" attention when no tokens are relevant.

Gated attention solves this by:
1. Allowing attention outputs to be **zeroed out** via sparse gates
2. Removing the need to allocate attention to irrelevant tokens
3. Creating query-dependent decisions about what attention to preserve

Result: First-token attention drops from 46.7% to 4.8%, and long-context extrapolation improves (10+ point gain on RULER benchmark).

### **Training Stability Improvements**

The paper shows that gated attention:
- **Reduces loss spikes**: Fewer training instabilities
- **Allows higher learning rates**: Can train with 8e-3 vs 4e-3 for baseline
- **Lowers peak activations**: Peak activations drop from 1053 to 94
- **Scales better**: Performance improves more as model size increases

This occurs because sparse gates prevent accumulation of large activation values that cause numerical issues in BF16 training.

## 📈 Performance Tips

1. **Use the default configuration** - The implemented version (elementwise, G1, sigmoid, multiplicative) matches the paper's best-performing variant. Don't modify unless experimenting.

2. **Monitor gate sparsity during training** - Gates should become sparse (mean ~0.1-0.2) as training progresses. If they stay around 0.5, something may be wrong.

3. **Expect ~2% wall-time overhead** - The additional computation is minimal and primarily manifests as slightly longer forward passes.

4. **Benefits appear during training** - Don't just compare final metrics; gated attention's stability benefits are visible throughout training curves.

5. **Long-context extrapolation** - If you plan to extend context length post-training (via YaRN, etc.), gated attention shows improvements (10+ points on RULER).

6. **Learning rate schedules** - The paper used the same learning rate schedules as baseline models. No special tuning needed.

## 🎯 Bottom Line

Gated Attention is a simple modification with measurable effects:

> *"A sigmoid gate after SDPA reduces attention sinks, improves training stability, and shows performance gains across benchmarks with <2% overhead."*

The technique differs from earlier work by adding learnable modulation rather than changing attention patterns. This addresses attention sink and training stability issues without architectural complexity.

To use gated attention in BlaGPT, set `attention="gated"` in your config. The technique was tested by Qwen Team on models up to 15B parameters trained on 3.5T tokens.

---

**Paper**: [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
**Authors**: Qiu et al., Qwen Team, Alibaba Group (May 2025)
**Implementation**: `/bla_gpt/attentions.py` (GatedAttention class, lines 950-1073)
**Status**: Production-ready, fully tested, drop-in replacement for standard attention
