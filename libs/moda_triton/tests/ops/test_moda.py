# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import pytest
import torch

from fla.utils import (
    device as default_device,
    check_shared_mem,
)


from fla.ops.moda import parallel_moda


def naive_mixture_of_depth_causal_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kd: Optional[torch.Tensor] = None,
    vd: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    moda_group_num: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.ndim == k.ndim == v.ndim == 4
    B, T_q, H, Kdim = q.shape
    Bk, T_kv, Hk, Kk = k.shape
    Bv, Tv, Hv, Vdim = v.shape
    assert (Bk, Hk, Kk) == (B, H, Kdim)
    assert (Bv, Tv, Hv) == (B, T_kv, H)
    assert T_q == T_kv * moda_group_num, "T_q must equal T_kv * moda_group_num"

    use_depth = (kd is not None) and (vd is not None)
    if use_depth:
        assert kd.shape[0] == B and kd.shape[2] == H and kd.shape[3] == Kdim
        assert vd.shape[0] == B and vd.shape[2] == H and vd.shape[3] == Vdim
        assert kd.shape[1] % T_kv == 0 and vd.shape[1] == kd.shape[1]
        L = kd.shape[1] // T_kv
        kd_reshaped = kd.view(B, T_kv, L, H, Kdim)
        vd_reshaped = vd.view(B, T_kv, L, H, Vdim)
    else:
        L = 0

    if scale is None:
        scale = 1.0 / math.sqrt(Kdim)
    ln2 = math.log(2.0)

    out = torch.empty(B, T_q, H, Vdim, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, T_q, H, dtype=q.dtype, device=q.device)

    for b in range(B):
        for h in range(H):
            q_bh = q[b, :, h].to(torch.float32)
            k_bh = k[b, :, h].to(torch.float32)
            v_bh = v[b, :, h].to(torch.float32)
            if use_depth:
                kd_bh = kd_reshaped[b, :, :, h].to(torch.float32)
                vd_bh = vd_reshaped[b, :, :, h].to(torch.float32)
            for t_q_idx in range(T_q):
                base_t = t_q_idx // moda_group_num
                scores_space = (q_bh[t_q_idx] @ k_bh[: base_t + 1].T) * scale
                if use_depth:
                    scores_depth = (kd_bh[base_t] @ q_bh[t_q_idx]) * scale
                    scores = torch.cat([scores_space, scores_depth], dim=0)
                else:
                    scores = scores_space
                weights = torch.softmax(scores, dim=0)
                if use_depth:
                    w_space = weights[: base_t + 1]
                    w_depth = weights[base_t + 1 :]
                    ctx_space = (w_space.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)
                    ctx_depth = (w_depth.unsqueeze(1) * vd_bh[base_t]).sum(0)
                    out[b, t_q_idx, h] = (ctx_space + ctx_depth).to(out.dtype)
                else:
                    out[b, t_q_idx, h] = (
                        (weights.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)
                    ).to(out.dtype)
                lse[b, t_q_idx, h] = torch.logsumexp(scores, dim=0) / ln2

    return out, lse


def report_and_check(tag, ref, val, atol=5e-3, rtol=5e-3):
    diff = (val - ref).float()
    max_abs = diff.abs().max().item()
    l2_rel = diff.pow(2).sum().sqrt() / (ref.pow(2).sum().sqrt() + 1e-12)
    ok = max_abs <= atol or l2_rel <= rtol * 10
    print(
        f"[{tag}] max_abs={max_abs:.4e} l2_rel={l2_rel:.4e} -> {'OK' if ok else 'FAIL'}"
    )
    assert ok, f"{tag} mismatch (max_abs={max_abs}, l2_rel={l2_rel})"


TEST_CASES = [
    (1, 63, 1, 64, 64, 0, 1, "float16", 1.0),
    (2, 128, 2, 64, 64, 0, 2, "float16", 1.0),
    (2, 111, 2, 64, 64, 4, 1, "float16", 1.0),
    (2, 256, 4, 64, 64, 4, 2, "float16", 1.0),
    (2, 192, 4, 64, 64, 0, 4, "float16", 0.1),
    (1, 160, 2, 64, 64, 8, 2, "bfloat16", 0.1),
    (1, 128, 2, 64, 64, 8, 4, "bfloat16", 0.1),
    (1, 1, 1, 64, 64, 0, 1, "float16", None),
    (1, 2, 2, 64, 64, 0, 2, "float16", None),
    (2, 3, 2, 64, 64, 0, 2, "float16", 1.0),
    (1, 63, 1, 64, 64, 0, 1, "float16", 1.0),
    (1, 64, 2, 64, 64, 0, 2, "float16", None),
    (1, 65, 2, 64, 64, 0, 4, "float16", None),
    (2, 47, 3, 64, 64, 0, 4, "float16", 0.5),
    (2, 95, 4, 64, 64, 0, 4, "float16", None),
    (3, 80, 3, 64, 64, 0, 2, "float16", None),
    (2, 111, 5, 64, 64, 0, 2, "float16", 1.0),
    (1, 32, 2, 64, 64, 0, 8, "float16", None),
    (1, 40, 2, 64, 64, 0, 8, "float16", 0.25),
    (1, 64, 2, 64, 64, 1, 1, "float16", None),
    (1, 63, 2, 64, 64, 1, 2, "float16", 1.0),
    (2, 65, 2, 64, 64, 2, 1, "float16", None),
    (2, 47, 3, 64, 64, 2, 2, "float16", None),
    (1, 64, 2, 64, 64, 3, 2, "float16", None),
    (1, 63, 2, 64, 64, 3, 4, "float16", 0.1),
    (2, 80, 2, 64, 64, 4, 2, "float16", None),
    (2, 95, 4, 64, 64, 4, 1, "float16", None),
    (1, 64, 2, 64, 64, 5, 2, "float16", None),
    (1, 63, 2, 64, 64, 5, 2, "float16", 0.2),
    (2, 47, 2, 64, 64, 7, 2, "float16", None),
    (1, 32, 2, 64, 64, 8, 2, "float16", None),
    (1, 40, 2, 64, 64, 8, 4, "float16", None),
    (1, 64, 2, 64, 64, 0, 2, "bfloat16", None),
    (1, 63, 2, 64, 64, 4, 2, "bfloat16", None),
    (2, 80, 3, 64, 64, 4, 4, "bfloat16", 0.1),
    (2, 95, 4, 64, 64, 8, 2, "bfloat16", None),
    (1, 64, 2, 64, 64, 0, 1, "float16", 1.0),
    (1, 64, 2, 64, 64, 0, 1, "float16", 0.03125),
    (1, 127, 2, 64, 64, 0, 1, "float16", None),
    (1, 128, 2, 64, 64, 0, 2, "float16", None),
    (1, 129, 2, 64, 64, 0, 4, "float16", None),
    (2, 191, 4, 64, 64, 0, 2, "float16", None),
    (2, 192, 4, 64, 64, 0, 4, "float16", None),
    (2, 193, 4, 64, 64, 0, 8, "float16", None),
    (1, 128, 2, 64, 64, 4, 2, "float16", None),
    (1, 128, 2, 64, 64, 7, 2, "float16", None),
    (1, 129, 2, 64, 64, 5, 2, "float16", None),
    (2, 192, 4, 64, 64, 4, 2, "float16", 0.1),
    (2, 192, 4, 64, 64, 8, 2, "float16", None),
    (4, 64, 8, 64, 64, 4, 2, "float16", None),
    (4, 63, 8, 64, 64, 5, 2, "float16", None),
    (4, 95, 8, 64, 64, 7, 2, "float16", None),
    (3, 111, 3, 64, 64, 3, 2, "float16", None),
    (3, 111, 3, 64, 64, 5, 2, "float16", None),
    (3, 111, 3, 64, 64, 7, 2, "float16", None),
    (1, 48, 2, 64, 64, 4, 8, "float16", None),
    (1, 48, 2, 64, 64, 5, 8, "float16", None),
    (2, 128, 4, 64, 64, 4, 2, "bfloat16", None),
    (2, 128, 4, 64, 64, 7, 2, "bfloat16", None),
    (2, 128, 4, 64, 64, 8, 2, "bfloat16", 0.05),
    (2, 192, 4, 64, 64, 8, 2, "bfloat16", None),
    (3, 97, 3, 64, 64, 5, 2, "float16", None),
    (3, 97, 3, 64, 64, 7, 2, "float16", None),
    (3, 129, 5, 64, 64, 5, 2, "float16", None),
    (2, 256, 8, 64, 64, 4, 2, "float16", None),
    (2, 256, 8, 64, 64, 8, 2, "float16", None),
    (1, 96, 6, 64, 64, 9, 2, "float16", None),
    (2, 96, 6, 64, 64, 12, 4, "float16", None),
    (2, 96, 7, 64, 64, 13, 2, "float16", 0.5),
    (3, 96, 7, 64, 64, 15, 2, "float16", None),
    (3, 96, 7, 64, 64, 16, 2, "float16", None),
    (4, 160, 6, 64, 64, 9, 2, "float16", None),
    (4, 160, 6, 64, 64, 12, 2, "float16", None),
    (4, 160, 6, 64, 64, 13, 4, "float16", None),
    (5, 160, 6, 64, 64, 15, 2, "float16", None),
    (5, 160, 6, 64, 64, 16, 2, "float16", 0.1),
    (6, 128, 5, 64, 64, 9, 4, "float16", None),
    (6, 128, 5, 64, 64, 12, 4, "float16", None),
    (6, 128, 5, 64, 64, 13, 2, "bfloat16", None),
    (6, 128, 5, 64, 64, 15, 4, "bfloat16", None),
    (6, 128, 5, 64, 64, 16, 4, "bfloat16", 0.2),
    (2, 48, 4, 64, 64, 9, 8, "float16", None),
    (2, 48, 4, 64, 64, 12, 8, "float16", None),
    (2, 48, 4, 64, 64, 13, 8, "float16", 0.25),
    (5, 72, 5, 64, 64, 0, 8, "float16", None),
    (6, 72, 5, 64, 64, 0, 8, "bfloat16", None),
    (1, 64, 4, 64, 64, 0, 1, "float16", 0.015625),
    (1, 64, 4, 64, 64, 0, 1, "float16", 2.0),
    (2, 80, 4, 64, 64, 8, 2, "bfloat16", 2.0),
    (2, 256, 8, 64, 64, 0, 2, "float16", None),
    (2, 256, 8, 64, 64, 8, 2, "float16", None),
    (2, 256, 8, 64, 64, 16, 2, "float16", None),
    (1, 224, 8, 64, 64, 12, 2, "float16", None),
    (1, 224, 8, 64, 64, 15, 2, "float16", None),
    (1, 224, 8, 64, 64, 16, 2, "bfloat16", None),
    (1, 128, 6, 64, 64, 9, 8, "float16", None),
    (1, 128, 6, 64, 64, 12, 8, "float16", None),
    (1, 128, 6, 64, 64, 16, 8, "float16", None),
    (5, 96, 7, 64, 64, 16, 4, "float16", None),
    (5, 96, 7, 64, 64, 15, 4, "bfloat16", None),
    (2, 191, 6, 64, 64, 16, 2, "float16", None),
    (2, 193, 6, 64, 64, 16, 2, "float16", None),
]


@pytest.mark.parametrize(
    ("B", "T_kv", "H", "K", "V", "L", "group", "dtype_name", "scale"),
    [
        pytest.param(
            *cfg,
            id=f"B{cfg[0]}-T{cfg[1]}-H{cfg[2]}-K{cfg[3]}-V{cfg[4]}-L{cfg[5]}-G{cfg[6]}-{cfg[7]}",
        )
        for cfg in TEST_CASES
    ],
)
def test_parallel_moda(
    B: int,
    T_kv: int,
    H: int,
    K: int,
    V: int,
    L: int,
    group: int,
    dtype_name: str,
    scale: float,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(default_device if torch.cuda.is_available() else "cpu")

    if not check_shared_mem("hopper") and K > 128:
        pytest.skip("Skip: shared-memory/architecture constraint (K>128 non-Hopper)")
    if group > 8:
        pytest.skip("Skip: group>8 is not covered")

    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

    torch.manual_seed(2025)
    T_q = T_kv * group
    scale_used = scale if scale is not None else 1.0 / math.sqrt(K)

    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device, requires_grad=True)

    if L > 0:
        kd = torch.randn(
            B, T_kv * L, H, K, dtype=dtype, device=device, requires_grad=True
        )
        vd = torch.randn(
            B, T_kv * L, H, V, dtype=dtype, device=device, requires_grad=True
        )
    else:
        kd = None
        vd = None

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    if kd is not None:
        kd_ref = kd.detach().clone().requires_grad_(True)
        vd_ref = vd.detach().clone().requires_grad_(True)
    else:
        kd_ref = vd_ref = None

    ref_out, ref_lse = naive_mixture_of_depth_causal_ref(
        q_ref,
        k_ref,
        v_ref,
        kd=kd_ref,
        vd=vd_ref,
        scale=scale_used,
        moda_group_num=group,
    )

    do = torch.randn_like(ref_out)
    (ref_out * do).sum().backward()

    q_impl = q.detach().clone().requires_grad_(True)
    k_impl = k.detach().clone().requires_grad_(True)
    v_impl = v.detach().clone().requires_grad_(True)
    if kd is not None:
        kd_impl = kd.detach().clone().requires_grad_(True)
        vd_impl = vd.detach().clone().requires_grad_(True)
    else:
        kd_impl = vd_impl = None

    impl_out = parallel_moda(
        q_impl,
        k_impl,
        v_impl,
        g=None,
        cached_k=kd_impl,
        cached_v=vd_impl,
        scale=scale_used,
        cu_seqlens=None,
        moda_group_num=group,
        head_first=False,
        need_lse=False,
    )
    (impl_out * do).sum().backward()

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-3
    rtol = 1e-2

    report_and_check("Forward/O", ref_out, impl_out, atol=atol, rtol=rtol)
    report_and_check("dQ", q_ref.grad, q_impl.grad, atol=atol, rtol=rtol)
    report_and_check("dK", k_ref.grad, k_impl.grad, atol=atol, rtol=rtol)
    report_and_check("dV", v_ref.grad, v_impl.grad, atol=atol, rtol=rtol)

    if L > 0:
        report_and_check("dK_depth", kd_ref.grad, kd_impl.grad, atol=atol, rtol=rtol)
        report_and_check("dV_depth", vd_ref.grad, vd_impl.grad, atol=atol, rtol=rtol)


def test_quick_moda_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    B, T_kv, H, K, V, L, group = 1, 64, 2, 64, 64, 4, 2
    dtype = torch.float16
    device = torch.device(default_device)
    scale = 1.0 / math.sqrt(K)

    T_q = T_kv * group
    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device, requires_grad=True)
    kd = torch.randn(B, T_kv * L, H, K, dtype=dtype, device=device, requires_grad=True)
    vd = torch.randn(B, T_kv * L, H, V, dtype=dtype, device=device, requires_grad=True)

    out = parallel_moda(
        q, k, v, cached_k=kd, cached_v=vd, scale=scale, moda_group_num=group
    )
    do = torch.randn_like(out)
    (out * do).sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert kd.grad is not None and vd.grad is not None
    print("Smoke test passed: grads computed.")


if __name__ == "__main__":

    pytest.main([__file__, "-k", "parallel_moda", "-s"])
