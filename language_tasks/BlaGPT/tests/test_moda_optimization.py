"""Correctness + efficiency tests for the BlaGPT MoDA depth-scaling
optimisations.

Three layers of coverage:

1. **Phase 1 bit-equivalence** -- the new ``(B, T, depth, h_kv, d)`` cache
   layout, fed to the kernel via a zero-copy ``reshape``, must produce the
   same kernel-input bytes as the OLD ``(B*T, h_kv, depth, d)`` layout
   followed by the per-block ``permute().contiguous().view()`` rebuild.
   Pure tensor-shape test, no kernel call required.

2. **v14 / v17 model-level equivalence** -- build two
   :class:`MoDADepthScalingGPT` models with identical weights, run
   forward + backward in v14 (``torch.cat`` cache) and v17 (K1
   pre-allocated buffer + ``current_depth`` mask) modes. Forward outputs
   and parameter gradients must match within fp16/bf16 noise.

3. **Wall-clock + memory benchmark** -- time forward-only and
   forward+backward at multiple depths (12, 24, 48, 108 layers) to surface
   how the K1 backend scales as the cache rebuild cost dominates v14.

Run::

    pytest tests/test_moda_optimization.py -q       # correctness
    python  tests/test_moda_optimization.py         # correctness + benchmark
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

# ``conftest.py`` already inserts ``bla_gpt`` into ``sys.path`` for pytest;
# add it explicitly so the file also runs as a plain script (``python ...``).
HERE = os.path.dirname(os.path.abspath(__file__))
BLA_GPT = os.path.normpath(os.path.join(HERE, "..", "bla_gpt"))
if BLA_GPT not in sys.path:
    sys.path.insert(0, BLA_GPT)

from bla_gpt_moda_depth_scaling import (  # noqa: E402
    DepthScalingMoDA512T64LPostNorm1p3BConfig,
    MoDADepthScalingGPT,
    _MODA_BACKENDS,
)


# -----------------------------------------------------------------------------
# Phase 1: cache layout bit-equivalence (pure tensor test, no kernel).
# -----------------------------------------------------------------------------
def _build_old_cache_kernel_input(per_block_kvs, B, T, H, D):
    """Mimic the PRE-Phase-1 pipeline: build cache as ``(B*T, H, depth, D)``
    by per-block ``reshape + .contiguous() + .cat(dim=2)``, then convert
    to the kernel-friendly ``(B, T*L, H, D)`` via
    ``view().permute(0,1,3,2,4).contiguous().view(...)``.
    """
    cache = None
    for block_k in per_block_kvs:
        origin = block_k.reshape(B * T, H, 1, D).contiguous()
        cache = origin if cache is None else torch.cat([cache, origin], dim=2)
    L = cache.size(2)
    return (
        cache.view(B, T, H, L, D)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(B, T * L, H, D)
    )


def _build_new_cache_kernel_input(per_block_kvs, B, T, H, D):
    """POST-Phase-1 pipeline: cache as ``(B, T, depth, H, D)`` via
    ``unsqueeze(2) + cat(dim=2)``; kernel feed via zero-copy ``reshape``.
    """
    cache = None
    for block_k in per_block_kvs:
        origin = block_k.unsqueeze(2)  # (B, T, 1, H, D), no copy
        cache = origin if cache is None else torch.cat([cache, origin], dim=2)
    L = cache.size(2)
    return cache.reshape(B, T * L, H, D)


def test_phase1_cache_layout_bitequal():
    """The Phase 1 layout change must produce bit-identical kernel input
    bytes to the old pipeline.
    """
    torch.manual_seed(0)
    B, T, H, D, L = 2, 16, 2, 32, 5
    per_block_kvs = [torch.randn(B, T, H, D) for _ in range(L)]

    old = _build_old_cache_kernel_input(per_block_kvs, B, T, H, D)
    new = _build_new_cache_kernel_input(per_block_kvs, B, T, H, D)

    assert old.shape == new.shape == (B, T * L, H, D)
    assert torch.equal(old, new), (
        "Phase 1 cache layout transformation is not bit-equivalent."
    )


# -----------------------------------------------------------------------------
# Test config / model factory.
# -----------------------------------------------------------------------------
@dataclass
class _SmallTestConfig(DepthScalingMoDA512T64LPostNorm1p3BConfig):
    """Tiny config for fast end-to-end correctness + quick benchmark."""

    block_size: int = 128
    sequence_length: int = 128
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 1
    n_embd: int = 64
    bias: bool = False
    rmsnorm_before_qk: bool = False
    tie_embed_weights: bool = True
    pos_encoding: str = "rotary"
    use_top: bool = False
    use_engram: bool = False
    use_canon_layers: bool = False
    use_per_layer_token_emb: bool = False
    use_parallel_blocks: bool = False
    use_pre_post_norm: bool = False
    enable_weight_sharing: bool = False
    n_predict: int = 1
    dropout: float = 0.0
    moda_backend: str = "v14"


def _build_model(
    backend: str,
    config_overrides: Optional[dict] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    cfg_kwargs = dict(moda_backend=backend)
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = _SmallTestConfig(**cfg_kwargs)
    torch.manual_seed(0)
    model = MoDADepthScalingGPT(config).to(device=device, dtype=dtype)
    return config, model


# -----------------------------------------------------------------------------
# v14 / v17 model-level equivalence (forward + backward).
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_v14_v17_forward_match():
    """v14 and v17 must produce the same logits (within bf16 noise) for the
    same weights and input.
    """
    cfg14, m14 = _build_model("v14")
    cfg17, m17 = _build_model("v17")
    m17.load_state_dict(m14.state_dict())
    m14.eval(); m17.eval()

    torch.manual_seed(1)
    idx = torch.randint(0, cfg14.vocab_size, (4, cfg14.sequence_length), device="cuda")

    with torch.no_grad():
        logits14, _ = m14(idx)
        logits17, _ = m17(idx)

    diff = (logits14.float() - logits17.float()).abs().max().item()
    print(f"\n[v14<->v17 fwd] max abs logit diff: {diff:.3e}")
    # bf16 model with 4 layers + small embed: rounding error is small.
    assert diff < 5e-2, f"Forward outputs diverge: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_v14_v17_backward_match():
    """v14 and v17 must produce matching parameter gradients (within bf16
    noise) for the same weights, input, and target.
    """
    cfg14, m14 = _build_model("v14")
    cfg17, m17 = _build_model("v17")
    m17.load_state_dict(m14.state_dict())
    m14.train(); m17.train()

    torch.manual_seed(2)
    idx = torch.randint(0, cfg14.vocab_size, (4, cfg14.sequence_length), device="cuda")
    targets = torch.randint(0, cfg14.vocab_size, (4, cfg14.sequence_length), device="cuda")

    def run(m):
        m.zero_grad(set_to_none=True)
        _, loss = m(idx, targets=targets)
        if isinstance(loss, dict):
            loss = loss["total"]
        loss.backward()
        grads = {
            n: p.grad.detach().float().clone()
            for n, p in m.named_parameters()
            if p.grad is not None
        }
        return loss.detach().float().item(), grads

    loss14, g14 = run(m14)
    loss17, g17 = run(m17)
    print(
        f"\n[v14<->v17 bwd] loss v14={loss14:.6f} v17={loss17:.6f} "
        f"diff={abs(loss14 - loss17):.3e}"
    )

    worst_name, worst_diff = "", 0.0
    for name in g14:
        d = (g14[name] - g17[name]).abs().max().item()
        if d > worst_diff:
            worst_name, worst_diff = name, d
    print(
        f"[v14<->v17 bwd] worst param-grad diff: {worst_name:60s} {worst_diff:.3e}"
    )
    # bf16 4-layer noise tolerance.
    assert worst_diff < 0.1, f"Param grad mismatch in {worst_name}: {worst_diff}"


# -----------------------------------------------------------------------------
# Benchmarks.
# -----------------------------------------------------------------------------
def _bench(model, idx, targets, train: bool, n_warmup: int = 5, n_iter: int = 20):
    """Return (avg ms, peak MB)."""
    if train:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(n_warmup):
            model.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            if isinstance(loss, dict):
                loss = loss["total"]
            loss.backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model.zero_grad(set_to_none=True)
            _, loss = model(idx, targets=targets)
            if isinstance(loss, dict):
                loss = loss["total"]
            loss.backward()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iter * 1e3
    else:
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(n_warmup):
                model(idx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                model(idx)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / n_iter * 1e3
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return elapsed, peak_mb


def bench_v14_vs_v17(label, n_layer, n_embd, n_head, n_kv_head, T, B):
    print(
        f"\n=== {label}: n_layer={n_layer}, B={B}, T={T}, "
        f"n_embd={n_embd}, n_head={n_head}, n_kv_head={n_kv_head} ==="
    )
    overrides = dict(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        block_size=max(T, 64),
        sequence_length=T,
    )
    cfg14, m14 = _build_model("v14", overrides)
    cfg17, m17 = _build_model("v17", overrides)
    m17.load_state_dict(m14.state_dict())

    idx = torch.randint(0, cfg14.vocab_size, (B, T), device="cuda")
    targets = torch.randint(0, cfg14.vocab_size, (B, T), device="cuda")

    for train in (False, True):
        label2 = "fwd+bwd" if train else "fwd-only"
        t14, m14_mb = _bench(m14, idx, targets, train=train)
        t17, m17_mb = _bench(m17, idx, targets, train=train)
        print(
            f"  {label2:9s}: v14 {t14:7.2f} ms ({m14_mb:7.1f} MB)  "
            f"v17 {t17:7.2f} ms ({m17_mb:7.1f} MB)  "
            f"v17/v14: {t17/t14:5.3f}  Δmem: {(m17_mb - m14_mb):+.1f} MB"
        )

    # Free model state between configurations to keep peak memory clean.
    del m14, m17
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Phase 1 cache-layout bit-equivalence test:")
    test_phase1_cache_layout_bitequal()
    print("  PASSED.\n")

    if not torch.cuda.is_available():
        print("CUDA not available; skipping model-level tests + benchmarks.")
        sys.exit(0)

    print("Model-level v14<->v17 equivalence tests:")
    test_v14_v17_forward_match()
    test_v14_v17_backward_match()
    print("  PASSED.\n")

    print("Benchmarks (forward-only / fwd+bwd, peak memory):")
    # Two sweeps:
    #
    # (a) Short context (T=128) sweep across depth: surfaces v14's O(L^2)
    #     cache-rebuild cost vs v17's O(L) in-place writes. v17 holds peak
    #     memory roughly flat in L while v14 grows quadratically.
    #
    # (b) Production-shape (T=512) sweep including the recipe depth
    #     (``DepthScalingMoDA512T64LPostNorm1p3BConfig``, 64 layers) plus a
    #     deeper 108L stress shape that exercises memory pressure (this is
    #     where v14 risks OOM at non-trivial B).
    bench_v14_vs_v17("BlaGPT 12L shallow",  n_layer=12,  n_embd=256,
                     n_head=4, n_kv_head=1, T=128, B=8)
    bench_v14_vs_v17("BlaGPT 24L mid",      n_layer=24,  n_embd=256,
                     n_head=4, n_kv_head=2, T=128, B=4)
    bench_v14_vs_v17("BlaGPT 48L deep",     n_layer=48,  n_embd=256,
                     n_head=4, n_kv_head=2, T=128, B=2)
    bench_v14_vs_v17("BlaGPT 64L T=512 (recipe shape)",
                     n_layer=64,  n_embd=256,
                     n_head=4, n_kv_head=2, T=512, B=1)
    bench_v14_vs_v17("BlaGPT 108L T=128",   n_layer=108, n_embd=256,
                     n_head=4, n_kv_head=2, T=128, B=1)
    bench_v14_vs_v17("BlaGPT 108L T=512 (deep stress)",
                     n_layer=108, n_embd=256,
                     n_head=4, n_kv_head=2, T=512, B=1)
