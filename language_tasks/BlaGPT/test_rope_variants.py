#!/usr/bin/env python3
"""Unit test for RoPE variants (standard and simplified)."""

import sys
sys.path.insert(0, '/home/devuser/BlaGPT/bla_gpt')

import torch
from bla_gpt import GPT, GPTConfig

def test_rope_variant(variant_name, config):
    """Test a specific RoPE variant."""
    print(f"\nTesting {variant_name.upper()} RoPE...")
    print(f"  Config: n_embd={config.n_embd}, n_head={config.n_head}, "
          f"head_dim={config.n_embd//config.n_head}, block_size={config.block_size}")

    config.rope_variant = variant_name

    try:
        # Create model
        model = GPT(config)
        param_count = sum(p.numel() for p in model.parameters())/1e6
        print(f"  Model created: {param_count:.2f}M parameters")

        # Create test data
        batch_size = 2
        seq_len = config.block_size
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            logits, loss = model(x, y)

        # Validate shapes
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, \
            f"Expected logits shape {expected_shape}, got {logits.shape}"

        # Validate loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {logits.shape}")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ Test PASSED")
        return True

    except Exception as e:
        print(f"  ✗ Test FAILED: {type(e).__name__}")
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("RoPE VARIANTS UNIT TEST")
    print("=" * 70)

    # Test configuration - small model for fast testing
    base_config = GPTConfig(
        n_layer=2,           # Just 2 layers
        n_head=4,            # 4 attention heads
        n_embd=128,          # 128 embedding dim (head_dim=32)
        block_size=64,       # 64 token sequences
        vocab_size=50304,    # BlaGPT vocab size
        dropout=0.0,         # No dropout for testing
        pos_encoding="rotary"
    )

    results = {}

    # Test 1: Standard RoPE
    results['standard'] = test_rope_variant('standard', base_config)

    # Test 2: Simplified RoPE
    results['simplified'] = test_rope_variant('simplified', base_config)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for variant, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {variant.upper()} RoPE: {status}")

    all_passed = all(results.values())
    print("\n" + ("="*70))
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
