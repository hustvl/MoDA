"""
Simplified Engram Implementation

Core idea: Hash N-grams → lookup embeddings → gate with hidden states → residual add

Strips away:
- CompressedTokenizer (normalization)
- Multi-head hashing with prime finding
- ShortConv refinement
- Hyper-connection (hc_mult) complexity
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================================
# Helper functions for creating shared embeddings
# ============================================================================

def create_shared_engram_embedding(
    table_size: int,
    hidden_size: int,
    zero_init: bool = True,
) -> nn.Embedding:
    """
    Create a shared embedding table for MinimalEngram instances.

    Args:
        table_size: Size of the hash table
        hidden_size: Embedding dimension (should match model hidden size)
        zero_init: Whether to zero-initialize weights (default: True)

    Returns:
        nn.Embedding that can be passed to MinimalEngram's shared_embedding parameter

    Example:
        shared_emb = create_shared_engram_embedding(500_000, hidden_size=512)
        engram1 = MinimalEngram(hidden_size=512, shared_embedding=shared_emb)
        engram2 = MinimalEngram(hidden_size=512, shared_embedding=shared_emb)
        # engram1 and engram2 now share the same embedding table
    """
    emb = nn.Embedding(table_size, hidden_size)
    if zero_init:
        nn.init.zeros_(emb.weight)
    return emb


def create_shared_simple_engram_embeddings(
    table_size: int,
    embed_dim: int,
    ngram_sizes: tuple[int, ...] = (2, 3),
    normal_init_std: float = 0.02,
) -> nn.ModuleList:
    """
    Create shared embedding tables for SimpleEngram instances.

    Args:
        table_size: Size of each hash table
        embed_dim: Embedding dimension per N-gram
        ngram_sizes: Which N-grams to use (default: (2, 3) for bigrams and trigrams)
        normal_init_std: Standard deviation for normal initialization (default: 0.02)

    Returns:
        nn.ModuleList that can be passed to SimpleEngram's shared_embeddings parameter

    Example:
        shared_embs = create_shared_simple_engram_embeddings(500_000, embed_dim=256)
        engram1 = SimpleEngram(hidden_size=512, vocab_size=32000, shared_embeddings=shared_embs)
        engram2 = SimpleEngram(hidden_size=512, vocab_size=32000, shared_embeddings=shared_embs)
        # engram1 and engram2 now share the same embedding tables
    """
    embeddings = nn.ModuleList([
        nn.Embedding(table_size, embed_dim)
        for _ in ngram_sizes
    ])
    # Initialize with small values
    for emb in embeddings:
        nn.init.normal_(emb.weight, std=normal_init_std)
    return embeddings


class SimpleEngram(nn.Module):
    """
    Simplified Engram: N-gram hash lookup with gated fusion.

    Args:
        hidden_size: Model hidden dimension
        vocab_size: Tokenizer vocabulary size
        ngram_sizes: Which N-grams to use (default: [2, 3] for bigrams and trigrams)
        embed_dim: Embedding dimension per N-gram (default: 256)
        table_size: Size of each hash table (default: 500000)
        pad_id: Padding token ID for shifted positions
        shared_embeddings: Optional shared embedding tables (nn.ModuleList). If provided,
            uses these instead of creating new embedding tables. Useful for sharing
            embeddings across multiple layers to reduce memory footprint.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        ngram_sizes: tuple[int, ...] = (2, 3),
        embed_dim: int = 256,
        table_size: int = 500_000,
        pad_id: int = 0,
        shared_embeddings: Optional[nn.ModuleList] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.ngram_sizes = ngram_sizes
        self.embed_dim = embed_dim
        self.pad_id = pad_id

        # Use shared embeddings if provided, else create own
        if shared_embeddings is not None:
            self.ngram_embeddings = shared_embeddings
            self.table_size = shared_embeddings[0].num_embeddings
        else:
            self.table_size = table_size
            # One embedding table per N-gram size
            self.ngram_embeddings = nn.ModuleList([
                nn.Embedding(table_size, embed_dim)
                for _ in ngram_sizes
            ])
            # Zero-initialize the embeddings
            for emb in self.ngram_embeddings:
                nn.init.zeros_(emb.weight)

        # Random multipliers for hashing (fixed after init)
        # Using large odd numbers for better hash distribution
        self.register_buffer(
            "hash_multipliers",
            torch.randint(1, 2**31, (max(ngram_sizes),), dtype=torch.long) | 1
        )

        # Total N-gram embedding size
        total_ngram_dim = len(ngram_sizes) * embed_dim

        # Projections for gating
        self.key_proj = nn.Linear(total_ngram_dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(total_ngram_dim, hidden_size, bias=False)

        # Layer norms for stable gating
        self.key_norm = RMSNorm(hidden_size)
        self.query_norm = RMSNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        # Small init for embeddings (they'll be learned)
        for emb in self.ngram_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        # Small init for value projection (starts near-zero residual)
        nn.init.normal_(self.value_proj.weight, std=0.02 / math.sqrt(2))

    def _compute_ngram_hash(
        self,
        input_ids: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Compute hash indices for N-grams.

        Args:
            input_ids: [B, L] token IDs
            n: N-gram size (e.g., 2 for bigrams)

        Returns:
            [B, L] hash indices into the embedding table
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Compute hash by combining n consecutive tokens
        # hash = (t[i] * m[0]) XOR (t[i-1] * m[1]) XOR ... XOR (t[i-n+1] * m[n-1])

        hash_val = torch.zeros(B, L, dtype=torch.long, device=device)

        for k in range(n):
            if k == 0:
                shifted = input_ids
            else:
                # Shift right by k positions, pad with pad_id
                pad = torch.full((B, k), self.pad_id, dtype=torch.long, device=device)
                shifted = torch.cat([pad, input_ids[:, :-k]], dim=1)

            hash_val = hash_val ^ (shifted * self.hash_multipliers[k])

        # Modulo to get table index
        hash_val = hash_val % self.table_size

        return hash_val

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] current hidden states
            input_ids: [B, L] original token IDs

        Returns:
            [B, L, D] output to add as residual
        """
        # 1. Compute N-gram hashes and lookup embeddings
        ngram_embeds = []
        for i, n in enumerate(self.ngram_sizes):
            hash_ids = self._compute_ngram_hash(input_ids, n)  # [B, L]
            emb = self.ngram_embeddings[i](hash_ids)  # [B, L, embed_dim]
            ngram_embeds.append(emb)

        # Concatenate all N-gram embeddings
        ngram_embeds = torch.cat(ngram_embeds, dim=-1)  # [B, L, num_ngrams * embed_dim]

        # 2. Project to key and value
        key = self.key_proj(ngram_embeds)    # [B, L, D]
        value = self.value_proj(ngram_embeds)  # [B, L, D]

        # 3. Compute gating score via normalized dot product
        key_norm = self.key_norm(key)
        query_norm = self.query_norm(hidden_states)

        # Dot product gating (element-wise, then sum, then sigmoid)
        gate_logits = (key_norm * query_norm).sum(dim=-1, keepdim=True)  # [B, L, 1]
        gate_logits = gate_logits / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate_logits)  # [B, L, 1]

        # 4. Gated output
        output = gate * value  # [B, L, D]

        return output


class EngramLayer(nn.Module):
    """
    A Transformer layer with Engram.

    Engram is applied before attention as a pre-residual addition.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 8,
        engram_config: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Engram module
        engram_config = engram_config or {}
        self.engram = SimpleEngram(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            **engram_config
        )

        # Standard transformer components (simplified)
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

        # Mock attention (replace with real implementation)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Engram: add N-gram memory
        hidden_states = hidden_states + self.engram(hidden_states, input_ids)

        # Attention
        normed = self.attn_norm(hidden_states)
        attn_out, _ = self.attn(normed, normed, normed)
        hidden_states = hidden_states + attn_out

        # FFN
        normed = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.ffn(normed)

        return hidden_states


# ============================================================================
# N-gram Lambda Mixer (from NanoGPT variant, extended to N-grams)
# ============================================================================

class NgramLambdaEngram(nn.Module):
    """
    N-gram Hash Embedding with per-layer lambda mixing.

    Extended from the NanoGPT bigram variant to support configurable N-grams.
    Uses learnable per-layer weights to mix:
    - x: current hidden state
    - x0: initial token embedding
    - x0_ngram: n-gram hash embedding

    At each layer: x = x_lambda * x + x0_lambda * x0 + ngram_lambda * x0_ngram

    This is a MODEL-LEVEL component that wraps the transformer blocks.

    Args:
        vocab_size: Vocabulary size
        model_dim: Model hidden dimension
        num_layers: Number of transformer layers
        ngram: N-gram size (default: 2 for bigrams, 3 for trigrams, etc.)
        ngram_vocab_mult: Multiplier for hash table size (table_size = mult * vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        ngram: int = 2,
        ngram_vocab_mult: int = 5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.ngram = ngram
        self.ngram_vocab_size = ngram_vocab_mult * vocab_size

        # Zero-initialized ngram embedding (learns to add ngram info)
        self.ngram_embed = nn.Embedding(self.ngram_vocab_size, model_dim)
        nn.init.zeros_(self.ngram_embed.weight)

        # Per-layer mixing weights
        # x_lambdas: weight for current hidden state (init to 1.0)
        # x0_lambdas: weight for initial embedding (init to 0.0)
        # ngram_lambdas: weight for ngram embedding (init to 0.1)
        self.x_lambdas = nn.Parameter(torch.ones(num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.ngram_lambdas = nn.Parameter(0.1 * torch.ones(num_layers))

        # Random multipliers for hashing (one per ngram position)
        # Using large odd numbers for better hash distribution
        self.register_buffer(
            "hash_multipliers",
            torch.randint(1, 2**31, (ngram,), dtype=torch.long) | 1
        )

    def get_ngram_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute n-gram hash indices using XOR hashing.

        Args:
            x: [B, L] token IDs

        Returns:
            [B, L] ngram hash indices
        """
        B, L = x.shape
        device = x.device
        mod = self.ngram_vocab_size - 1

        # Compute hash by XOR-combining n consecutive tokens
        # hash = (t[i] * m[0]) XOR (t[i-1] * m[1]) XOR ... XOR (t[i-n+1] * m[n-1])
        hash_val = torch.zeros(B, L, dtype=torch.long, device=device)

        for k in range(self.ngram):
            if k == 0:
                shifted = x
            else:
                # Shift right by k positions, pad with zeros
                shifted = F.pad(x[:, :-k], (k, 0), value=0)

            hash_val = hash_val ^ (shifted * self.hash_multipliers[k])

        return hash_val % mod

    def get_ngram_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute n-gram embeddings from input token IDs.

        Args:
            input_ids: [B, L] token IDs

        Returns:
            [B, L, D] ngram embeddings
        """
        ngram_hash = self.get_ngram_hash(input_ids)
        return self.ngram_embed(ngram_hash)

    def mix_at_layer(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        x0_ngram: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply per-layer lambda mixing.

        Args:
            x: [B, L, D] current hidden state
            x0: [B, L, D] initial token embedding
            x0_ngram: [B, L, D] ngram embedding
            layer_idx: which layer (0-indexed)

        Returns:
            [B, L, D] mixed representation
        """
        return (
            self.x_lambdas[layer_idx] * x +
            self.x0_lambdas[layer_idx] * x0 +
            self.ngram_lambdas[layer_idx] * x0_ngram
        )


# ============================================================================
# Even simpler: Minimal Engram (< 50 lines of core logic)
# ============================================================================

class MinimalEngram(nn.Module):
    """
    Bare-bones Engram in ~40 lines.

    Just: hash N-grams → lookup → project → gate → output

    Args:
        hidden_size: Model hidden dimension
        table_size: Size of hash table (default: 500000)
        ngram: N-gram size (default: 3)
        pad_id: Padding token ID for shifted positions
        shared_embedding: Optional shared embedding table. If provided, uses this
            instead of creating a new embedding table. Useful for sharing embeddings
            across multiple layers to reduce memory footprint.
    """

    def __init__(
        self,
        hidden_size: int,
        table_size: int = 500_000,
        ngram: int = 3,
        pad_id: int = 0,
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.ngram = ngram
        self.pad_id = pad_id

        # Use shared embedding if provided, else create own
        if shared_embedding is not None:
            self.embedding = shared_embedding
            self.table_size = shared_embedding.num_embeddings  # Sync table_size
        else:
            self.table_size = table_size
            self.embedding = nn.Embedding(table_size, hidden_size)
            nn.init.zeros_(self.embedding.weight)

        self.gate_proj = nn.Linear(hidden_size * 2, 1)

        # Fixed random multipliers for hashing
        self.register_buffer(
            "multipliers",
            torch.randint(1, 2**62, (ngram,), dtype=torch.long) | 1
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        B, L = input_ids.shape
        device = input_ids.device

        # Hash N-gram at each position
        h = torch.zeros(B, L, dtype=torch.long, device=device)
        for k in range(self.ngram):
            shifted = F.pad(input_ids[:, k:], (0, k), value=self.pad_id) if k > 0 else input_ids
            shifted = F.pad(input_ids, (k, 0), value=self.pad_id)[:, :L]
            h = h ^ (shifted * self.multipliers[k])
        h = h % self.table_size

        # Lookup and gate
        mem = self.embedding(h)  # [B, L, D]
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, mem], dim=-1)))

        return gate * mem


# ============================================================================
# Factory
# ============================================================================

def create_engram(
    variant: str,
    hidden_size: int,
    vocab_size: int,
    num_layers: int,
    ngram: int = 3,
    vocab_mult: int = 5,
) -> nn.Module:
    """
    Factory function to create an Engram module.

    Args:
        variant: "ngram_lambda", "simple", or "minimal"
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers (used by ngram_lambda)
        ngram: N-gram size (default: 3)
        vocab_mult: Hash table size multiplier (default: 5)

    Returns:
        Engram module instance
    """
    if variant == "ngram_lambda":
        return NgramLambdaEngram(
            vocab_size=vocab_size,
            model_dim=hidden_size,
            num_layers=num_layers,
            ngram=ngram,
            ngram_vocab_mult=vocab_mult,
        )
    elif variant == "simple":
        return SimpleEngram(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            ngram_sizes=tuple(range(2, ngram + 1)),
            table_size=vocab_mult * vocab_size,
        )
    elif variant == "minimal":
        return MinimalEngram(
            hidden_size=hidden_size,
            table_size=vocab_mult * vocab_size,
            ngram=ngram,
        )
    else:
        raise ValueError(f"Unknown engram variant: {variant}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    B, L, D = 2, 32, 512
    vocab_size = 32000

    # Test SimpleEngram
    print("Testing SimpleEngram...")
    model = SimpleEngram(hidden_size=D, vocab_size=vocab_size)
    hidden = torch.randn(B, L, D)
    input_ids = torch.randint(0, vocab_size, (B, L))

    out = model(hidden, input_ids)
    print(f"  Input:  hidden {hidden.shape}, input_ids {input_ids.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test MinimalEngram
    print("\nTesting MinimalEngram...")
    model2 = MinimalEngram(hidden_size=D)
    out2 = model2(hidden, input_ids)
    print(f"  Output: {out2.shape}")
    print(f"  Params: {sum(p.numel() for p in model2.parameters()):,}")

    # Test full layer
    print("\nTesting EngramLayer...")
    layer = EngramLayer(hidden_size=D, vocab_size=vocab_size)
    out3 = layer(hidden, input_ids)
    print(f"  Output: {out3.shape}")
    print(f"  Params: {sum(p.numel() for p in layer.parameters()):,}")

    # Test NgramLambdaEngram with different n-gram sizes
    num_layers = 6
    for ngram in [2, 3, 4]:
        print(f"\nTesting NgramLambdaEngram (ngram={ngram})...")
        ngram_engram = NgramLambdaEngram(
            vocab_size=vocab_size,
            model_dim=D,
            num_layers=num_layers,
            ngram=ngram,
        )

        # Simulate model forward pass
        x0 = torch.randn(B, L, D)  # Initial embedding
        x0_ngram = ngram_engram.get_ngram_embedding(input_ids)
        print(f"  N-gram embedding shape: {x0_ngram.shape}")

        x = x0.clone()
        for i in range(num_layers):
            x = ngram_engram.mix_at_layer(x, x0, x0_ngram, i)
            # In real use, x = block(x) would go here

        print(f"  Layer lambdas: x={ngram_engram.x_lambdas[0].item():.2f}, "
              f"x0={ngram_engram.x0_lambdas[0].item():.2f}, "
              f"ngram={ngram_engram.ngram_lambdas[0].item():.2f}")
        print(f"  Final output: {x.shape}")
        print(f"  Params: {sum(p.numel() for p in ngram_engram.parameters()):,}")

    # Test shared embedding for MinimalEngram
    print("\nTesting MinimalEngram with shared embedding...")
    shared_emb = create_shared_engram_embedding(500_000, D)
    engram1 = MinimalEngram(hidden_size=D, shared_embedding=shared_emb)
    engram2 = MinimalEngram(hidden_size=D, shared_embedding=shared_emb)

    # Verify they share the same embedding
    assert engram1.embedding is engram2.embedding, "Embeddings should be shared!"
    assert engram1.table_size == 500_000, "table_size should be synced from shared embedding"
    print("  ✓ Embeddings are shared")

    # Test forward pass with shared embedding
    out_shared1 = engram1(hidden, input_ids)
    out_shared2 = engram2(hidden, input_ids)
    print(f"  ✓ Forward pass works: {out_shared1.shape}")
    print(f"  Params per engram (excluding shared): {sum(p.numel() for p in engram1.gate_proj.parameters()):,}")
    print(f"  Shared embedding params: {shared_emb.weight.numel():,}")

    # Test shared embeddings for SimpleEngram
    print("\nTesting SimpleEngram with shared embeddings...")
    shared_embs = create_shared_simple_engram_embeddings(500_000, embed_dim=256)
    simple1 = SimpleEngram(hidden_size=D, vocab_size=vocab_size, shared_embeddings=shared_embs)
    simple2 = SimpleEngram(hidden_size=D, vocab_size=vocab_size, shared_embeddings=shared_embs)

    # Verify they share the same embeddings
    assert simple1.ngram_embeddings is simple2.ngram_embeddings, "Embeddings should be shared!"
    assert simple1.table_size == 500_000, "table_size should be synced from shared embeddings"
    print("  ✓ Embeddings are shared")

    # Test forward pass with shared embeddings
    out_simple1 = simple1(hidden, input_ids)
    out_simple2 = simple2(hidden, input_ids)
    print(f"  ✓ Forward pass works: {out_simple1.shape}")

    print("\n✅ All tests passed!")
