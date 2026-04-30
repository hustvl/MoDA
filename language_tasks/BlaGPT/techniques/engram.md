# Engram: N-gram Hash Memory for Transformers

*A deterministic memory module that augments transformers with static n-gram knowledge through hash-based embedding lookups.*

---

## Overview

Engram addresses the lack of native knowledge lookup mechanisms in transformer architectures. While Mixture-of-Experts (MoE) enables conditional computation, transformers lack a complementary approach for accessing static, pre-computed memories. Engram introduces deterministic n-gram hashing to retrieve token-pattern embeddings with O(1) complexity, offloading large tables to CPU memory with minimal inference overhead.

The method treats static knowledge retrieval independently from neural computation, enabling models to balance between dynamic processing and static memory access.

## Method

### Core Mechanism

Engram computes hash indices for n-gram token patterns and retrieves corresponding embeddings from per-layer tables:

1. **Hash n-gram patterns**: For each position in the sequence, combine n consecutive tokens using XOR hashing
2. **Lookup embeddings**: Index into layer-specific embedding tables
3. **Gate with hidden states**: Modulate retrieved embeddings using context-dependent gating
4. **Add as residual**: Combine gated embeddings with the transformer's forward pass

### Hash Function

For n consecutive tokens at position i:
```
hash = (t[i] × m[0]) ⊕ (t[i-1] × m[1]) ⊕ ... ⊕ (t[i-n+1] × m[n-1])
index = hash mod table_size
```

Where:
- `t[i]` is the token ID at position i
- `m[k]` are fixed random multipliers (odd integers)
- `⊕` denotes XOR operation
- Padding tokens are used for positions near sequence boundaries

### Gating Mechanism

Retrieved n-gram embeddings are modulated by the current hidden state:

```
key = W_key(concat(ngram_embeds))
value = W_value(concat(ngram_embeds))
gate = sigmoid((norm(key) · norm(hidden)) / √d)
output = gate × value
```

This preserves context-awareness while accessing static memories.

## Variants in BlaGPT

BlaGPT includes 3 Engram variants with these common changes:

Removed the following things to keep it simple and focus on the core idea:

1. **CompressedTokenizer**: No text normalization before hashing
2. **Multi-head hashing**: Single hash function instead of finding optimal primes
3. **ShortConv refinement**: No additional convolution over n-gram embeddings
4. **Hyper-connection complexity**: Simplified residual connection patterns

and retrained:

- XOR-based n-gram hashing with random multipliers
- Per-layer embedding tables
- Context-dependent gating
- Deterministic addressing (no learned routing)


### SimpleEngram

Standard implementation with multiple n-gram sizes (default: bigrams and trigrams):
- Separate embedding tables per n-gram size
- Projects concatenated embeddings to key/value
- RMSNorm for stable gating

### NgramLambdaEngram

from: https://x.com/classiclarryd/status/2013520088297558274?s=20

Layer-wise mixing variant inspired by bigram models:
- Single n-gram embedding table shared across layers
- Per-layer learnable mixing coefficients (λ_x, λ_x0, λ_ngram)
- Mixes current hidden state, initial embedding, and n-gram embedding
- Applied between transformer blocks

### MinimalEngram

*Best results in BlaGPT experiments*

I simplified version the original idea to:
- Single n-gram size (default: trigrams)
- Concatenates hidden state with n-gram embedding for gating
- Minimal projections

## Implementation

### Basic Usage

```python
from bla_gpt.engram import SimpleEngram, NgramLambdaEngram, MinimalEngram

# Standard variant with bigrams and trigrams
engram = SimpleEngram(
    hidden_size=512,
    vocab_size=32000,
    ngram_sizes=(2, 3),
    embed_dim=256,
    table_size=500_000,
)

# Apply in transformer block
output = engram(hidden_states, input_ids)  # [B, L, D]
hidden_states = hidden_states + output
```

### Shared Embeddings

Reduce memory by sharing tables across layers:

```python
from bla_gpt.engram import create_shared_simple_engram_embeddings

# Create shared tables
shared_embs = create_shared_simple_engram_embeddings(
    table_size=500_000,
    embed_dim=256,
    ngram_sizes=(2, 3),
)

# Use in multiple layers
engram_layer_1 = SimpleEngram(hidden_size=512, vocab_size=32000,
                               shared_embeddings=shared_embs)
engram_layer_2 = SimpleEngram(hidden_size=512, vocab_size=32000,
                               shared_embeddings=shared_embs)
```

### Layer-wise Mixing

```python
# Create model-level wrapper
ngram_engram = NgramLambdaEngram(
    vocab_size=32000,
    model_dim=512,
    num_layers=6,
    ngram=2,  # bigrams
    ngram_vocab_mult=5,
)

# Get n-gram embeddings once
x0_ngram = ngram_engram.get_ngram_embedding(input_ids)

# Mix at each layer
for layer_idx in range(num_layers):
    x = ngram_engram.mix_at_layer(x, x0, x0_ngram, layer_idx)
    x = transformer_block(x)
```
