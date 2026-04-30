# -*- coding: utf-8 -*-
"""Tests + benchmark for ``fla.ops.moda.parallel_moda_v17`` (K1 variant).

The K1 plan (see ``libs/moda_triton/fla/ops/moda/moda_v17.py`` docstring)
modifies the MoDA Triton kernels so that the depth cache is addressed by
``L_max`` (the pre-allocated buffer's depth) but only the first ``L_cur``
slots per token are valid (the rest are masked out). This swaps the
per-block cache-management cost on the Python/PyTorch side from
``O(L*BNHK)`` per layer (``torch.stack`` of the growing list) down to
``O(BNHK)`` per layer (one strided write into the pre-allocated buffer),
trading a small amount of wasted in-kernel compute for a large reduction
in cache-build memory bandwidth.

This file contains:

* ``test_v17_default_matches_v14``         -- ``current_depth=None`` is
  bit-identical to v14 forward (drop-in compatibility).
* ``test_v17_k1_matches_v14_tight``        -- the K1 mode (max-depth buffer
  + ``current_depth=L_cur``) matches v14 with a tightly packed ``L_cur``
  cache, within fp16 numerical noise.
* ``test_v17_invalid_slot_grads_are_zero`` -- gradients written by the K1
  backward to slot positions ``slot >= L_cur`` are exactly zero.

* ``main_bench_multi_layer``  -- end-to-end ``n_layer``-block forward-only
  simulation. Compares three host-side cache-management strategies
  (all pure PyTorch, only the underlying parallel_moda kernel differs):

    A. v14 + per-block ``torch.stack`` rebuild from a per-layer ``kv_list``
       (= the strategy used in ``vision_tasks/deit/models.py``).
    B. v14 + per-block rolling ``torch.cat`` (= the strategy used in
       ``language_tasks/BlaGPT/bla_gpt/bla_gpt_moda_depth_scaling.py``).
    C. v17 + pre-allocated ``[B, T_kv*n_layer, H, K]`` buffer and a
       ``current_depth=i`` parameter per block (the K1 plan).

  All three produce numerically equivalent attention outputs (modulo fp16
  noise in C); the difference is purely the cache-build cost.

* ``main_bench_kernel`` -- single-call kernel-only benchmark across L,
  comparing v14, v17(default), v17(K1 with L_max > L_cur) for both
  forward and forward+backward. This isolates the per-kernel-call cost
  difference introduced by the K1 mask from the host-side savings.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch

from fla.ops.moda import parallel_moda, parallel_moda_v17


# =============================================================================
# Helpers
# =============================================================================


def _make_attention_inputs(
    B: int,
    T_kv: int,
    H_q: int,
    H_kv: int,
    D: int,
    L_max: int,
    L_cur: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> dict:
    """Random inputs for a single MoDA attention call.

    Builds two depth caches:
      * kd_tight, vd_tight: shape ``[B, T_kv*L_cur, H, D]`` -- what v14
        expects.
      * kd_full,  vd_full:  shape ``[B, T_kv*L_max, H, D]`` -- what v17 K1
        expects, with the first L_cur slots per token equal to kd_tight /
        vd_tight and the rest random garbage.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    T_q = T_kv * (H_q // H_kv)

    q = torch.randn(B, T_q, H_kv, D, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, T_kv, H_kv, D, generator=g, dtype=dtype, device=device)
    v = torch.randn(B, T_kv, H_kv, D, generator=g, dtype=dtype, device=device)

    kd_tight = torch.randn(
        B, T_kv * L_cur, H_kv, D, generator=g, dtype=dtype, device=device
    )
    vd_tight = torch.randn(
        B, T_kv * L_cur, H_kv, D, generator=g, dtype=dtype, device=device
    )

    # Use randn (finite) for the unwritten suffix; the kernel computes
    # a dot over the full L_max strip before the slot mask is applied,
    # so the suffix must be finite to avoid NaN propagation through
    # tensor-core paths.
    kd_full = torch.randn(
        B, T_kv * L_max, H_kv, D, generator=g, dtype=dtype, device=device
    )
    vd_full = torch.randn(
        B, T_kv * L_max, H_kv, D, generator=g, dtype=dtype, device=device
    )
    with torch.no_grad():
        kd_full.view(B, T_kv, L_max, H_kv, D)[:, :, :L_cur].copy_(
            kd_tight.view(B, T_kv, L_cur, H_kv, D)
        )
        vd_full.view(B, T_kv, L_max, H_kv, D)[:, :, :L_cur].copy_(
            vd_tight.view(B, T_kv, L_cur, H_kv, D)
        )

    for t in (q, k, v, kd_tight, vd_tight, kd_full, vd_full):
        t.requires_grad_(True)

    return dict(
        q=q, k=k, v=v,
        kd_tight=kd_tight, vd_tight=vd_tight,
        kd_full=kd_full, vd_full=vd_full,
        T_kv=T_kv, H_kv=H_kv, D=D, L_max=L_max, L_cur=L_cur,
        scale=1.0 / math.sqrt(D),
        moda_group_num=H_q // H_kv,
    )


def _make_layer_inputs(
    n_layer: int, B: int, T_kv: int, H_kv: int, D: int, moda_group_num: int,
    dtype: torch.dtype, device: str,
):
    """K and V tensors for n_layer transformer blocks (no autograd)."""
    g = torch.Generator(device=device).manual_seed(2026)
    T_q = T_kv * moda_group_num
    q_per_layer = [
        torch.randn(B, T_q, H_kv, D, generator=g, dtype=dtype, device=device)
        for _ in range(n_layer)
    ]
    k_per_layer = [
        torch.randn(B, T_kv, H_kv, D, generator=g, dtype=dtype, device=device)
        for _ in range(n_layer)
    ]
    v_per_layer = [
        torch.randn(B, T_kv, H_kv, D, generator=g, dtype=dtype, device=device)
        for _ in range(n_layer)
    ]
    return q_per_layer, k_per_layer, v_per_layer


def _cuda_bench(fn, warmup: int = 3, iters: int = 10) -> float:
    """Run fn() warmup+iters times and return mean wall time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# =============================================================================
# Pytest correctness tests
# =============================================================================


CORRECTNESS_CASES = [
    # (B, T_kv, H_q, H_kv, D, L_max, L_cur, dtype_name)
    (2,  64, 4, 1, 64,  8, 5, "fp16"),
    (2, 128, 4, 2, 64,  6, 3, "fp16"),
    (1,  96, 8, 4, 64, 12, 1, "fp16"),
    (1,  96, 8, 4, 64, 12, 7, "fp16"),
    (1,  96, 8, 4, 64, 12, 12, "fp16"),  # boundary: L_cur == L_max
    (2,  47, 4, 2, 64,  5, 2, "bf16"),
]


@pytest.mark.parametrize(
    ("B", "T_kv", "H_q", "H_kv", "D", "L_max", "L_cur", "dtype_name"),
    [pytest.param(*c, id=f"B{c[0]}-T{c[1]}-Hq{c[2]}-Hkv{c[3]}-D{c[4]}-Lmax{c[5]}-Lcur{c[6]}-{c[7]}")
     for c in CORRECTNESS_CASES],
)
def test_v17_default_matches_v14(B, T_kv, H_q, H_kv, D, L_max, L_cur, dtype_name):
    """``current_depth=None`` reproduces v14 bit-by-bit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    args = _make_attention_inputs(
        B, T_kv, H_q, H_kv, D, L_max=L_cur, L_cur=L_cur, dtype=dtype, device="cuda",
    )
    out_v14 = parallel_moda(
        args["q"], args["k"], args["v"],
        cached_k=args["kd_tight"], cached_v=args["vd_tight"],
        scale=args["scale"], moda_group_num=args["moda_group_num"],
        is_causal=True, warn_shape=False,
    )
    out_v17 = parallel_moda_v17(
        args["q"], args["k"], args["v"],
        cached_k=args["kd_tight"], cached_v=args["vd_tight"],
        scale=args["scale"], moda_group_num=args["moda_group_num"],
        is_causal=True, warn_shape=False,
    )
    diff = (out_v14 - out_v17).abs().max().item()
    assert diff == 0.0, f"v17 default must equal v14 bit-by-bit; got {diff}"


@pytest.mark.parametrize(
    ("B", "T_kv", "H_q", "H_kv", "D", "L_max", "L_cur", "dtype_name"),
    [pytest.param(*c, id=f"B{c[0]}-T{c[1]}-Hq{c[2]}-Hkv{c[3]}-D{c[4]}-Lmax{c[5]}-Lcur{c[6]}-{c[7]}")
     for c in CORRECTNESS_CASES],
)
def test_v17_k1_matches_v14_tight(B, T_kv, H_q, H_kv, D, L_max, L_cur, dtype_name):
    """K1 mode matches v14 with a tightly packed L_cur cache."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    args = _make_attention_inputs(
        B, T_kv, H_q, H_kv, D, L_max=L_max, L_cur=L_cur, dtype=dtype, device="cuda",
    )
    out_v14 = parallel_moda(
        args["q"], args["k"], args["v"],
        cached_k=args["kd_tight"], cached_v=args["vd_tight"],
        scale=args["scale"], moda_group_num=args["moda_group_num"],
        is_causal=True, warn_shape=False,
    )
    out_v17 = parallel_moda_v17(
        args["q"], args["k"], args["v"],
        cached_k=args["kd_full"], cached_v=args["vd_full"],
        scale=args["scale"], moda_group_num=args["moda_group_num"],
        is_causal=True, current_depth=L_cur, warn_shape=False,
    )
    atol = 5e-3 if dtype == torch.float16 else 8e-3
    diff = (out_v14.float() - out_v17.float()).abs().max().item()
    assert diff < atol, f"v17 K1 must match v14 tight within {atol}; got {diff}"


def test_v17_invalid_slot_grads_are_zero():
    """Backward through K1 mode writes exactly zero to slots >= L_cur."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = _make_attention_inputs(
        B=2, T_kv=64, H_q=4, H_kv=2, D=64, L_max=8, L_cur=5,
        dtype=torch.float16, device="cuda",
    )
    out = parallel_moda_v17(
        args["q"], args["k"], args["v"],
        cached_k=args["kd_full"], cached_v=args["vd_full"],
        scale=args["scale"], moda_group_num=args["moda_group_num"],
        is_causal=True, current_depth=args["L_cur"], warn_shape=False,
    )
    grad_o = torch.randn_like(out)
    g_kd_full, g_vd_full = torch.autograd.grad(
        out, [args["kd_full"], args["vd_full"]], grad_o
    )
    invalid_k = (
        g_kd_full.view(2, 64, 8, 2, 64)[:, :, args["L_cur"]:].abs().max().item()
    )
    invalid_v = (
        g_vd_full.view(2, 64, 8, 2, 64)[:, :, args["L_cur"]:].abs().max().item()
    )
    assert invalid_k == 0.0, f"d(cached_k) over invalid slots not zero: {invalid_k}"
    assert invalid_v == 0.0, f"d(cached_v) over invalid slots not zero: {invalid_v}"


# =============================================================================
# End-to-end multi-layer benchmark (forward only, no autograd)
# =============================================================================
#
# Why forward-only:
#
#   Strategies A and B build the cache via ``torch.stack`` / ``torch.cat``,
#   which keep the autograd graph intact: backward through the cached_k
#   path naturally flows back to the per-layer ``k`` tensors. Strategy C's
#   K1 buffer requires either an in-place (graph-breaking) ``.copy_()`` or
#   a manual ``index_copy``-into-leaf gradient routing on the user side --
#   the latter is implementation-specific and not part of the kernel API.
#   We therefore compare cache-management cost in the forward pass only;
#   the kernel-only forward+backward comparison is in ``main_bench_kernel``.


def _strategy_A_stack(
    n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
):
    """v14 + stack-from-list cache (DeiT-style)."""
    kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    out_list = []
    for i in range(n_layer):
        if kv_list:
            keys = torch.stack([t[0] for t in kv_list], dim=2)
            vals = torch.stack([t[1] for t in kv_list], dim=2)
            B_, N_, L_, H_, D_ = keys.shape
            cached_k = keys.reshape(B_, N_ * L_, H_, D_)
            cached_v = vals.reshape(B_, N_ * L_, H_, D_)
        else:
            cached_k = cached_v = None
        out = parallel_moda(
            q_per_layer[i], k_per_layer[i], v_per_layer[i],
            cached_k=cached_k, cached_v=cached_v,
            scale=scale, moda_group_num=moda_group_num,
            is_causal=False, warn_shape=False,
        )
        out_list.append(out)
        kv_list.append((k_per_layer[i], v_per_layer[i]))
    return out_list


def _strategy_B_cat(
    n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
):
    """v14 + rolling torch.cat cache (BlaGPT-style)."""
    cached_k = cached_v = None
    out_list = []
    for i in range(n_layer):
        if cached_k is None:
            depth_k = depth_v = None
        else:
            B_, N_, L_, H_, D_ = cached_k.shape
            depth_k = cached_k.view(B_, N_ * L_, H_, D_)
            depth_v = cached_v.view(B_, N_ * L_, H_, cached_v.size(-1))
        out = parallel_moda(
            q_per_layer[i], k_per_layer[i], v_per_layer[i],
            cached_k=depth_k, cached_v=depth_v,
            scale=scale, moda_group_num=moda_group_num,
            is_causal=False, warn_shape=False,
        )
        out_list.append(out)
        new_k = k_per_layer[i].unsqueeze(2)
        new_v = v_per_layer[i].unsqueeze(2)
        cached_k = new_k if cached_k is None else torch.cat([cached_k, new_k], dim=2)
        cached_v = new_v if cached_v is None else torch.cat([cached_v, new_v], dim=2)
    return out_list


def _strategy_C_v17_k1(
    n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
):
    """v17 K1: pre-allocated max-depth buffer + ``current_depth=i``.

    Note: the buffer must be initialised with finite values (zeros);
    ``torch.empty`` is unsafe because the kernel still loads + dots the
    unwritten suffix before masking, and NaN/Inf garbage can leak through
    tensor-core paths in some Triton/HMMA configurations.
    """
    B, T_kv, H_kv, D = k_per_layer[0].shape
    Vdim = v_per_layer[0].shape[-1]
    buf_k = torch.zeros(
        B, T_kv * n_layer, H_kv, D,
        dtype=k_per_layer[0].dtype, device=k_per_layer[0].device,
    )
    buf_v = torch.zeros(
        B, T_kv * n_layer, H_kv, Vdim,
        dtype=v_per_layer[0].dtype, device=v_per_layer[0].device,
    )
    out_list = []
    for i in range(n_layer):
        if i == 0:
            depth_k = depth_v = None
        else:
            depth_k = buf_k
            depth_v = buf_v
        out = parallel_moda_v17(
            q_per_layer[i], k_per_layer[i], v_per_layer[i],
            cached_k=depth_k, cached_v=depth_v,
            scale=scale, moda_group_num=moda_group_num,
            is_causal=False, current_depth=i, warn_shape=False,
        )
        out_list.append(out)
        # O(BNHK) strided write into the pre-allocated buffer.
        buf_k.view(B, T_kv, n_layer, H_kv, D)[:, :, i].copy_(k_per_layer[i])
        buf_v.view(B, T_kv, n_layer, H_kv, Vdim)[:, :, i].copy_(v_per_layer[i])
    return out_list


def _measure_forward_strategy(name, strategy_fn, *args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ms = _cuda_bench(lambda: strategy_fn(*args))
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return ms, peak_mb


def _verify_strategies_match(
    n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale, atol,
):
    """Sanity check that A, B, C produce numerically equivalent outputs."""
    with torch.no_grad():
        out_a = _strategy_A_stack(
            n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
        )
        out_b = _strategy_B_cat(
            n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
        )
        out_c = _strategy_C_v17_k1(
            n_layer, q_per_layer, k_per_layer, v_per_layer, moda_group_num, scale,
        )
    diffs = []
    for i in range(n_layer):
        d_ab = (out_a[i].float() - out_b[i].float()).abs().max().item()
        d_ac = (out_a[i].float() - out_c[i].float()).abs().max().item()
        diffs.append((i, d_ab, d_ac))
    max_ab = max(d for _, d, _ in diffs)
    max_ac = max(d for _, _, d in diffs)
    print(f"  sanity: max layer diff A vs B = {max_ab:.2e}, A vs C = {max_ac:.2e}")
    assert max_ab < atol, f"A and B should agree (max={max_ab})"
    assert max_ac < atol, f"A and C should agree (max={max_ac})"


def main_bench_multi_layer(
    B: int = 16,
    T_kv: int = 197,
    H_kv: int = 1,
    moda_group_num: int = 4,
    D: int = 64,
    n_layer: int = 12,
    dtype_name: str = "fp16",
):
    """Multi-layer end-to-end forward-only benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    H_q = H_kv * moda_group_num

    print("=" * 80)
    print("MoDA depth-cache strategies, multi-layer forward-only benchmark")
    print(
        f"  B={B}, T_kv={T_kv}, H_q={H_q}, H_kv={H_kv}, D={D}, "
        f"n_layer={n_layer}, dtype={dtype_name}"
    )
    print(f"  GPU={torch.cuda.get_device_name(0)}")
    print("=" * 80)

    q_per_layer, k_per_layer, v_per_layer = _make_layer_inputs(
        n_layer, B, T_kv, H_kv, D, moda_group_num,
        dtype=dtype, device="cuda",
    )

    _verify_strategies_match(
        n_layer, q_per_layer, k_per_layer, v_per_layer,
        moda_group_num, 1.0 / math.sqrt(D),
        atol=5e-3 if dtype == torch.float16 else 8e-3,
    )

    rows = []
    for name, fn in [
        ("A: v14 + stack-from-list  (DeiT today)",      _strategy_A_stack),
        ("B: v14 + rolling torch.cat (BlaGPT today)",   _strategy_B_cat),
        ("C: v17 + pre-alloc buf + current_depth (K1)", _strategy_C_v17_k1),
    ]:
        # Run with no_grad so requires_grad=False inputs participate in pure
        # forward (no graph build cost interfering with measurement).
        with torch.no_grad():
            ms, peak_mb = _measure_forward_strategy(
                name, fn,
                n_layer, q_per_layer, k_per_layer, v_per_layer,
                moda_group_num, 1.0 / math.sqrt(D),
            )
        rows.append((name, ms, peak_mb))

    ms_a = rows[0][1]
    print(f"\n  {'strategy':50s} {'ms':>10s} {'speedup vs A':>14s} {'peak MB':>10s}")
    for name, ms, peak_mb in rows:
        print(f"  {name:48s} {ms:>9.3f}  {ms_a/ms:>13.2f}x  {peak_mb:>9.1f}")
    print()


# =============================================================================
# Single-call kernel-only benchmark
# =============================================================================


def main_bench_kernel(
    B: int = 16,
    T_kv: int = 197,
    H_kv: int = 1,
    moda_group_num: int = 4,
    D: int = 64,
    L_max: int = 12,
    L_cur_values: List[int] = None,
    dtype_name: str = "fp16",
):
    """For each L_cur, time the kernel for v14(L=L_cur) vs v17(default,
    L=L_cur) vs v17(K1, L_max=L_max, current_depth=L_cur)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return
    if L_cur_values is None:
        L_cur_values = [1, 3, 6, 9, 11]
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    H_q = H_kv * moda_group_num

    print("=" * 80)
    print("Single-call kernel-only benchmark")
    print(
        f"  B={B}, T_kv={T_kv}, H_q={H_q}, H_kv={H_kv}, D={D}, "
        f"L_max={L_max}, dtype={dtype_name}"
    )
    print(f"  GPU={torch.cuda.get_device_name(0)}")
    print("=" * 80)

    print(
        f"\n  {'L_cur':>5s} | {'v14 fwd':>10s} {'v17 fwd':>10s} {'v17K1 fwd':>10s} | "
        f"{'v14 fwd+bwd':>14s} {'v17 fwd+bwd':>14s} {'v17K1 fwd+bwd':>14s}"
    )

    for L_cur in L_cur_values:
        args = _make_attention_inputs(
            B, T_kv, H_q, H_kv, D, L_max=L_max, L_cur=L_cur,
            dtype=dtype, device="cuda",
        )

        def _fwd_v14():
            return parallel_moda(
                args["q"], args["k"], args["v"],
                cached_k=args["kd_tight"], cached_v=args["vd_tight"],
                scale=args["scale"], moda_group_num=args["moda_group_num"],
                is_causal=False, warn_shape=False,
            )

        def _fwd_v17_default():
            return parallel_moda_v17(
                args["q"], args["k"], args["v"],
                cached_k=args["kd_tight"], cached_v=args["vd_tight"],
                scale=args["scale"], moda_group_num=args["moda_group_num"],
                is_causal=False, warn_shape=False,
            )

        def _fwd_v17_k1():
            return parallel_moda_v17(
                args["q"], args["k"], args["v"],
                cached_k=args["kd_full"], cached_v=args["vd_full"],
                scale=args["scale"], moda_group_num=args["moda_group_num"],
                is_causal=False, current_depth=L_cur, warn_shape=False,
            )

        with torch.no_grad():
            ms_f_v14   = _cuda_bench(_fwd_v14)
            ms_f_v17   = _cuda_bench(_fwd_v17_default)
            ms_f_v17k1 = _cuda_bench(_fwd_v17_k1)

        # forward+backward: include autograd setup. Use a fresh grad_o each
        # call but reuse the same input tensors (their .grad fields are
        # zeroed each call to avoid accumulation cost).
        grad_o = torch.randn(B, T_kv * args["moda_group_num"], H_kv, D,
                             dtype=dtype, device="cuda")

        def _step_v14():
            for t in (args["q"], args["k"], args["v"], args["kd_tight"], args["vd_tight"]):
                t.grad = None
            o = _fwd_v14()
            o.backward(grad_o)

        def _step_v17_default():
            for t in (args["q"], args["k"], args["v"], args["kd_tight"], args["vd_tight"]):
                t.grad = None
            o = _fwd_v17_default()
            o.backward(grad_o)

        def _step_v17_k1():
            for t in (args["q"], args["k"], args["v"], args["kd_full"], args["vd_full"]):
                t.grad = None
            o = _fwd_v17_k1()
            o.backward(grad_o)

        ms_b_v14   = _cuda_bench(_step_v14)
        ms_b_v17   = _cuda_bench(_step_v17_default)
        ms_b_v17k1 = _cuda_bench(_step_v17_k1)

        print(
            f"  {L_cur:>5d} | {ms_f_v14:>10.3f} {ms_f_v17:>10.3f} {ms_f_v17k1:>10.3f} | "
            f"{ms_b_v14:>14.3f} {ms_b_v17:>14.3f} {ms_b_v17k1:>14.3f}"
        )
    print()


def main_correctness():
    """Standalone correctness check, mirrors the pytest assertions."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping correctness.")
        return
    print("Running v17 correctness checks (stand-alone)...")
    for cfg in CORRECTNESS_CASES:
        B, T_kv, H_q, H_kv, D, L_max, L_cur, dtype_name = cfg
        dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
        args = _make_attention_inputs(
            B, T_kv, H_q, H_kv, D, L_max=L_max, L_cur=L_cur,
            dtype=dtype, device="cuda",
        )
        out14 = parallel_moda(
            args["q"], args["k"], args["v"],
            cached_k=args["kd_tight"], cached_v=args["vd_tight"],
            scale=args["scale"], moda_group_num=args["moda_group_num"],
            is_causal=True, warn_shape=False,
        )
        out17_default = parallel_moda_v17(
            args["q"], args["k"], args["v"],
            cached_k=args["kd_tight"], cached_v=args["vd_tight"],
            scale=args["scale"], moda_group_num=args["moda_group_num"],
            is_causal=True, warn_shape=False,
        )
        out17_k1 = parallel_moda_v17(
            args["q"], args["k"], args["v"],
            cached_k=args["kd_full"], cached_v=args["vd_full"],
            scale=args["scale"], moda_group_num=args["moda_group_num"],
            is_causal=True, current_depth=L_cur, warn_shape=False,
        )
        d_default = (out14 - out17_default).abs().max().item()
        d_k1 = (out14.float() - out17_k1.float()).abs().max().item()
        tag = f"B{B}-T{T_kv}-Hq{H_q}-Hkv{H_kv}-Lmax{L_max}-Lcur{L_cur}-{dtype_name}"
        print(
            f"  [{tag}] v17(default) diff = {d_default:.2e}, "
            f"v17(K1) diff = {d_k1:.2e}"
        )


if __name__ == "__main__":
    main_correctness()
    print()
    # Sweep depth to expose K1's O(L) cache-build savings, which only
    # outweigh its in-kernel mask overhead at large n_layer.
    #
    #   * DeiT-Tiny  (n_layer=12)
    #   * ViT-L-ish  (n_layer=24)
    #   * Mid-deep   (n_layer=48)
    #   * BlaGPT-ish (n_layer=96)
    main_bench_multi_layer(B=16, T_kv=197, H_kv=1, moda_group_num=4,
                           D=64, n_layer=12)
    main_bench_multi_layer(B=8,  T_kv=197, H_kv=2, moda_group_num=2,
                           D=64, n_layer=24)
    main_bench_multi_layer(B=4,  T_kv=512, H_kv=2, moda_group_num=4,
                           D=64, n_layer=48)
    main_bench_multi_layer(B=2,  T_kv=512, H_kv=2, moda_group_num=4,
                           D=64, n_layer=96)

    # Per-call kernel cost for various L_cur values (DeiT-Tiny shape).
    main_bench_kernel(B=16, T_kv=197, H_kv=1, moda_group_num=4, D=64,
                      L_max=12, L_cur_values=[1, 3, 6, 9, 11])
