# -*- coding: utf-8 -*-

"""MoDA Triton kernels and reference helpers."""

import math
import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp2, log2
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    check_shared_mem,
    contiguous,
)


def is_target_gpu(
    device_index: int | None = None, target_gpu_name: str = "H800"
) -> bool:

    try:
        if device_index is None:
            device_index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_index)
        return target_gpu_name in name.upper()
    except Exception:
        pass


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_DEPTH": lambda args: args["cached_k"] is not None
        and args["cached_v"] is not None,
    }
)
@triton.jit
def parallel_moda_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    lse,
    cached_k,
    cached_v,
    L,
    moda_group_num,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_DEPTH: tl.constexpr,
):

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_hq = i_bh // HQ, i_bh % HQ

    i_h = i_hq // G

    if IS_VARLEN:

        i_n, i_t_seq = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)

        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)

        T_q = eos - bos

        i_t_effective = i_t_seq
    else:
        i_n = i_b

        T_q = T
        bos, eos = i_n * T_q, i_n * T_q + T_q

        i_t_effective = i_t

    T_kv = T_q // moda_group_num

    bos_q = bos

    bos_kv = i_n * T_kv

    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(
        q + (bos_q * HQ + i_hq) * K,
        (T_q, K),
        (HQ * K, 1),
        (i_t_effective * BT, 0),
        (BT, BK),
        (1, 0),
    )

    p_o = tl.make_block_ptr(
        o + (bos_q * HQ + i_hq) * V,
        (T_q, V),
        (HQ * V, 1),
        (i_t_effective * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    p_lse = tl.make_block_ptr(
        lse + bos_q * HQ + i_hq, (T_q,), (HQ,), (i_t_effective * BT,), (BT,), (0,)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)

    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_G:

        p_g = tl.make_block_ptr(
            g_cumsum + (bos_q * HQ + i_hq),
            (T_q,),
            (HQ,),
            (i_t_effective * BT,),
            (BT,),
            (0,),
        )
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

    else:
        b_gq = None

    q_block_start = i_t_effective * BT
    q_block_end = tl.minimum((i_t_effective + 1) * BT, T_q)

    o_q = q_block_start + tl.arange(0, BT)
    o_q_base = o_q // moda_group_num

    base_t_block_start = q_block_start // moda_group_num

    for i_s in range(0, base_t_block_start, BS):

        p_k = tl.make_block_ptr(
            k + (bos_kv * H + i_h) * K,
            (K, T_kv),
            (1, H * K),
            (0, i_s),
            (BK, BS),
            (0, 1),
        )

        p_v = tl.make_block_ptr(
            v + (bos_kv * H + i_h) * V,
            (T_kv, V),
            (H * V, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        o_k = i_s + tl.arange(0, BS)
        m_k = (o_k < base_t_block_start) & (o_k < T_kv)
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos_kv + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )

            b_s += b_gq[:, None] - b_gk[None, :]

        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m

        b_r = exp2(b_mp - b_m)

        b_p = exp2(b_s - b_m[:, None])

        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    base_t_block_end = (q_block_end - 1) // moda_group_num + 1
    base_t_block_end = tl.minimum(base_t_block_end, T_kv)

    for i_s in range(base_t_block_start, base_t_block_end, BS):

        p_k = tl.make_block_ptr(
            k + (bos_kv * H + i_h) * K,
            (K, T_kv),
            (1, H * K),
            (0, i_s),
            (BK, BS),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (bos_kv * H + i_h) * V,
            (T_kv, V),
            (H * V, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T_kv

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos_kv + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )
            b_s += b_gq[:, None] - b_gk[None, :]

        b_s = tl.where(
            (o_q_base[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float("-inf")
        )

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m

    if USE_DEPTH:

        bos_cached = i_n * (T_kv * L)

        row_valid = o_q < T
        valid_rows = tl.minimum(BT, T_q - q_block_start)

        base_rows_start = q_block_start // moda_group_num
        base_rows_end = (q_block_start + valid_rows - 1) // moda_group_num + 1
        base_rows_end = tl.minimum(base_rows_end, T_kv)

        depth_block_start = base_rows_start * L
        depth_block_end = base_rows_end * L

        query_row_ids = o_q

        base_k_cached = cached_k + (bos_cached * H + i_h) * K
        base_v_cached = cached_v + (bos_cached * H + i_h) * V

        for depth_col_start in range(depth_block_start, depth_block_end, BS):
            p_k_depth = tl.make_block_ptr(
                base_k_cached,
                (K, T_kv * L),
                (1, H * K),
                (0, depth_col_start),
                (BK, BS),
                (0, 1),
            )

            p_v_depth = tl.make_block_ptr(
                base_v_cached,
                (T_kv * L, V),
                (H * V, 1),
                (depth_col_start, i_v * BV),
                (BS, BV),
                (1, 0),
            )

            b_k_depth = tl.load(p_k_depth, boundary_check=(0, 1))
            b_v_depth = tl.load(p_v_depth, boundary_check=(0, 1))

            b_s_depth = tl.dot(b_q, b_k_depth) * scale * RCP_LN2

            depth_col_ids = depth_col_start + tl.arange(0, BS)
            depth_row_ids = depth_col_ids // L

            row_mask = row_valid[:, None]

            col_mask = depth_col_ids[None, :] < depth_block_end

            row_match_mask = (
                depth_row_ids[None, :] == (query_row_ids // moda_group_num)[:, None]
            )

            mask_depth = row_mask & col_mask & row_match_mask

            b_s_depth = tl.where(mask_depth, b_s_depth, float("-inf"))

            b_m_new = tl.maximum(b_m, tl.max(b_s_depth, 1))

            b_r = exp2(b_m - b_m_new)

            b_p_depth = exp2(b_s_depth - b_m_new[:, None])

            b_acc = b_acc * b_r + tl.sum(b_p_depth, 1)

            b_o = b_o * b_r[:, None] + tl.dot(b_p_depth.to(b_q.dtype), b_v_depth)

            b_m = b_m_new

    b_o = b_o / b_acc[:, None]

    b_m += log2(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


def parallel_moda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cached_k: Optional[torch.Tensor] = None,
    cached_v: Optional[torch.Tensor] = None,
    moda_group_num: int = 1,
    customized_BT: int = None,
    customized_BS: int = None,
    customized_BK: int = None,
    customized_BV: int = None,
    customized_num_warps: int = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    T_kv = T
    TQ = q.shape[1]
    HQ = q.shape[2]
    G = HQ // H
    BT = 128

    L = 0
    if cached_k is not None and cached_v is not None:
        L = cached_k.shape[1] // T_kv
        assert L > 0, "Depth L must be positive when using cached KV"

    if moda_group_num > 1:
        TKV = k.shape[1]
        assert (
            TQ / TKV == moda_group_num
        ), "For now, TQ / TKV must be equal to moda_group_num"

    if check_shared_mem("hopper", q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T_kv)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
        num_warps = 8
    elif check_shared_mem("ampere", q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T_kv)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        num_warps = 4
    else:
        BS = min(32, max(16, triton.next_power_of_2(T_kv)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
        num_warps = 2

    if customized_BT is not None:
        BT = customized_BT
    if customized_BS is not None:
        BS = customized_BS
    if customized_BK is not None:
        BK = customized_BK
    if customized_BV is not None:
        BV = customized_BV
    if customized_num_warps is not None:
        num_warps = customized_num_warps

    device_id = q.device.index

    if is_target_gpu(device_id, "H800"):
        BT = 64
        BS = 64
        BK = 64
        BV = 64
        num_warps = 4

    elif is_target_gpu(device_id, "H20"):
        BT = 64
        BS = 64
        BK = 64
        BV = 64
        num_warps = 4

    elif is_target_gpu(device_id, "A100"):
        BT = 64
        BS = 64
        BK = 64
        BV = 64
        num_warps = 4

    else:
        print("Target GPU is not optimized")

    assert (
        BT % moda_group_num == 0
    ), "To keep depth alignment consistent with backward, BT % moda_group_num must be 0"

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(TQ, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, TQ, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, TQ, HQ, dtype=torch.float, device=q.device)

    grid = (NV, NT, B * HQ)

    parallel_moda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        lse=lse,
        cached_k=cached_k,
        cached_v=cached_v,
        L=L,
        moda_group_num=moda_group_num,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=TQ,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )
    return o, lse


@triton.jit
def parallel_moda_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):

    i_n = tl.program_id(0)

    o_d = tl.arange(0, B)

    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)

    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_DEPTH": lambda args: args["cached_k"] is not None
        and args["cached_v"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_moda_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    g_cumsum,
    cached_k,
    cached_v,
    L,
    moda_group_num,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_DEPTH: tl.constexpr,
):

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:

        i_n, i_t_seq = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_q = eos - bos
        q_block_index = i_t_seq
    else:
        i_n = i_b
        T_q = T
        bos = i_n * T_q
        eos = bos + T_q
        q_block_index = i_t

    T_kv = T_q // moda_group_num

    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K,
        (T_q, K),
        (HQ * K, 1),
        (q_block_index * BT, 0),
        (BT, BK),
        (1, 0),
    )
    p_dq = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K,
        (T_q, K),
        (HQ * K, 1),
        (q_block_index * BT, 0),
        (BT, BK),
        (1, 0),
    )
    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V,
        (T_q, V),
        (HQ * V, 1),
        (q_block_index * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_lse = tl.make_block_ptr(
        lse + bos * HQ + i_hq, (T_q,), (HQ,), (q_block_index * BT,), (BT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta + bos * HQ + i_hq, (T_q,), (HQ,), (q_block_index * BT,), (BT,), (0,)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros(
            [
                BT,
            ],
            dtype=tl.float32,
        )
        p_gq = tl.make_block_ptr(
            g_cumsum + bos * HQ + i_hq,
            (T_q,),
            (HQ,),
            (q_block_index * BT,),
            (BT,),
            (0,),
        )
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None

    q_block_start = q_block_index * BT
    q_block_end = tl.minimum((q_block_index + 1) * BT, T_q)
    o_q = q_block_start + tl.arange(0, BT)
    o_q_base = o_q // moda_group_num

    base_t_block_start = q_block_start // moda_group_num

    for i_s in range(0, base_t_block_start, BS):
        p_k = tl.make_block_ptr(
            k + (i_n * T_kv * H + i_h) * K,
            (K, T_kv),
            (1, H * K),
            (0, i_s),
            (BK, BS),
            (0, 1),
        )

        p_v = tl.make_block_ptr(
            v + (i_n * T_kv * H + i_h) * V,
            (V, T_kv),
            (1, H * V),
            (i_v * BV, i_s),
            (BV, BS),
            (0, 1),
        )

        o_k = i_s + tl.arange(0, BS)
        m_k = (o_k < base_t_block_start) & (o_k < T_kv)

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            b_gk = tl.load(
                g_cumsum + (i_n * T_kv + o_k) * HQ + i_hq, mask=m_k, other=0
            ).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])

        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    base_t_block_end = (q_block_end - 1) // moda_group_num + 1
    base_t_block_end = tl.minimum(base_t_block_end, T_kv)

    for i_s in range(base_t_block_start, base_t_block_end, BS):
        p_k = tl.make_block_ptr(
            k + (i_n * T_kv * H + i_h) * K,
            (K, T_kv),
            (1, H * K),
            (0, i_s),
            (BK, BS),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (i_n * T_kv * H + i_h) * V,
            (V, T_kv),
            (1, H * V),
            (i_v * BV, i_s),
            (BV, BS),
            (0, 1),
        )

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T_kv

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:

            b_gk = tl.load(
                g_cumsum + (i_n * T_kv + o_k) * HQ + i_hq, mask=m_k, other=0
            ).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        causal_mask = (o_q_base[:, None] >= o_k[None, :]) & m_k[None, :]
        b_p = tl.where(causal_mask, exp2(b_s - b_lse[:, None]), 0)

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    if USE_DEPTH:

        bos_cached = i_n * (T_kv * L)

        row_valid = o_q < T_q
        valid_rows = tl.minimum(BT, T_q - q_block_start)

        base_rows_start = q_block_start // moda_group_num
        base_rows_end = (q_block_start + valid_rows - 1) // moda_group_num + 1
        base_rows_end = tl.minimum(base_rows_end, T_kv)

        depth_block_start = base_rows_start * L
        depth_block_end = base_rows_end * L

        query_base_rows = o_q_base

        base_k_cached = cached_k + (bos_cached * H + i_h) * K
        base_v_cached = cached_v + (bos_cached * H + i_h) * V

        for depth_col_start in range(depth_block_start, depth_block_end, BS):

            p_k_depth = tl.make_block_ptr(
                base_k_cached,
                (K, T_kv * L),
                (1, H * K),
                (0, depth_col_start),
                (BK, BS),
                (0, 1),
            )

            p_v_depth = tl.make_block_ptr(
                base_v_cached,
                (V, T_kv * L),
                (1, H * V),
                (i_v * BV, depth_col_start),
                (BV, BS),
                (0, 1),
            )

            b_k_depth = tl.load(p_k_depth, boundary_check=(0, 1))
            b_v_depth = tl.load(p_v_depth, boundary_check=(0, 1))

            b_s_depth = tl.dot(b_q, b_k_depth) * scale * RCP_LN2

            depth_col_ids = depth_col_start + tl.arange(0, BS)
            depth_row_ids = depth_col_ids // L

            col_valid = depth_col_ids < depth_block_end

            match_row = depth_row_ids[None, :] == query_base_rows[:, None]
            mask_depth = (row_valid[:, None]) & col_valid[None, :] & match_row

            b_p_depth = tl.where(mask_depth, exp2(b_s_depth - b_lse[:, None]), 0)

            b_dp_depth = tl.dot(b_do, b_v_depth)
            b_ds_depth = b_p_depth * (b_dp_depth.to(tl.float32) - b_delta[:, None])

            b_dq += tl.dot(b_ds_depth.to(b_k_depth.dtype), tl.trans(b_k_depth))

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(
            dg_cumsum + bos * HQ + i_hq,
            (T_q,),
            (HQ,),
            (q_block_index * BT,),
            (BT,),
            (0,),
        )
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T_kv", "T_q"])
def parallel_attn_bwd_kernel_dkv_group_parallel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T_kv,
    T_q,
    moda_group_num,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    MoDA_G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):

    i_v, i_t, i_bhg = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b = i_bhg // (HQ * MoDA_G)
    i_hq = (i_bhg % (HQ * MoDA_G)) // MoDA_G
    i_g = i_bhg % MoDA_G

    i_h = i_hq // G

    if IS_VARLEN:
        return

    bos_q = i_b * T_q
    bos_kv = i_b * T_kv
    RCP_LN2: tl.constexpr = 1.4426950216

    k_start = i_t * BT

    p_k = tl.make_block_ptr(
        k + (bos_kv * H + i_h) * K,
        (T_kv, K),
        (H * K, 1),
        (k_start, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos_kv * H + i_h) * V,
        (T_kv, V),
        (H * V, 1),
        (k_start, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    o_k = k_start + tl.arange(0, BT)
    full_k = (k_start + BT) <= T_kv
    if full_k:
        b_k = tl.load(p_k)
        b_v = tl.load(p_v)
        m_k = tl.full([BT], True, tl.int1)
    else:
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_k = o_k < T_kv

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    for base_start in range(k_start, T_kv, BS):
        o_base = base_start + tl.arange(0, BS)
        full_base = (base_start + BS) <= T_kv

        o_q = o_base * moda_group_num + i_g

        m_base = o_base < T_kv
        m_q = (o_q < T_q) & m_base

        q_ptrs = (
            q
            + (bos_q * HQ + i_hq) * K
            + o_q[:, None] * (HQ * K)
            + tl.arange(0, BK)[None, :]
        )
        do_ptrs = (
            do
            + (bos_q * HQ + i_hq) * V
            + o_q[:, None] * (HQ * V)
            + (i_v * BV + tl.arange(0, BV)[None, :])
        )
        lse_ptrs = lse + bos_q * HQ + i_hq + o_q * HQ
        delta_ptrs = delta + bos_q * HQ + i_hq + o_q * HQ

        if full_base:
            b_q = tl.load(q_ptrs)
            b_do = tl.load(do_ptrs)
            b_lse = tl.load(lse_ptrs)
            b_delta = tl.load(delta_ptrs)

            b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
            causal = (o_k[:, None] <= o_base[None, :]) & m_k[:, None]
            b_p = tl.where(causal, exp2(b_s - b_lse[None, :]), 0)
        else:
            b_q = tl.load(q_ptrs, mask=m_q[:, None], other=0.0)
            b_do = tl.load(do_ptrs, mask=m_q[:, None], other=0.0)
            b_lse = tl.load(lse_ptrs, mask=m_q, other=float("inf"))
            b_delta = tl.load(delta_ptrs, mask=m_q, other=0.0)

            b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
            causal = (o_k[:, None] <= o_base[None, :]) & m_k[:, None] & m_q[None, :]
            b_p = tl.where(causal, exp2(b_s - b_lse[None, :]), 0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)

        b_dp = tl.dot(b_v, tl.trans(b_do))

        b_ds = b_p * (b_dp - b_delta[None, :])

        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    b_dk = b_dk * scale

    p_dk = tl.make_block_ptr(
        dk + ((bos_kv * HQ + i_hq) * MoDA_G + i_g) * K,
        (T_kv, K),
        (HQ * MoDA_G * K, 1),
        (k_start, 0),
        (BT, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + ((bos_kv * HQ + i_hq) * MoDA_G + i_g) * V,
        (T_kv, V),
        (HQ * MoDA_G * V, 1),
        (k_start, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    moda_group_num,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        seq_id, t_block = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos_seq = tl.load(cu_seqlens + seq_id).to(tl.int32)
        eos_seq = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
        T_kv = eos_seq - bos_seq

        T_q = T_kv * moda_group_num

        key_tile_index = t_block
        bos_kv = bos_seq
        bos_q = bos_seq * moda_group_num
    else:
        T_kv = T
        T_q = T_kv * moda_group_num
        key_tile_index = i_t
        bos_kv = i_b * T_kv
        bos_q = i_b * T_q

    RCP_LN2: tl.constexpr = 1.4426950216

    p_k = tl.make_block_ptr(
        k + (bos_kv * H + i_h) * K,
        (T_kv, K),
        (H * K, 1),
        (key_tile_index * BT, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos_kv * H + i_h) * V,
        (T_kv, V),
        (H * V, 1),
        (key_tile_index * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (bos_kv * HQ + i_hq) * K,
        (T_kv, K),
        (HQ * K, 1),
        (key_tile_index * BT, 0),
        (BT, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos_kv * HQ + i_hq) * V,
        (T_kv, V),
        (HQ * V, 1),
        (key_tile_index * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = key_tile_index * BT + tl.arange(0, BT)
    m_k = o_k < T_kv

    q_partial_start = (key_tile_index * BT) * moda_group_num
    q_partial_end = tl.minimum(((key_tile_index + 1) * BT) * moda_group_num, T_q)

    for qs in range(q_partial_start, q_partial_end, BS):
        o_q = qs + tl.arange(0, BS)
        m_q = o_q < q_partial_end
        m_q = m_q & (o_q < T_q)

        p_q = tl.make_block_ptr(
            q + (bos_q * HQ + i_hq) * K,
            (T_q, K),
            (HQ * K, 1),
            (qs, 0),
            (BS, BK),
            (1, 0),
        )
        p_do = tl.make_block_ptr(
            do + (bos_q * HQ + i_hq) * V,
            (T_q, V),
            (HQ * V, 1),
            (qs, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        p_lse = tl.make_block_ptr(
            lse + bos_q * HQ + i_hq, (T_q,), (HQ,), (qs,), (BS,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta + bos_q * HQ + i_hq, (T_q,), (HQ,), (qs,), (BS,), (0,)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        o_q_base = o_q // moda_group_num

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2

        causal = (o_k[:, None] <= o_q_base[None, :]) & m_k[:, None] & m_q[None, :]
        b_p = tl.where(causal, exp2(b_s - b_lse[None, :]), 0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)

        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])

        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    for qs in range(q_partial_end, tl.cdiv(T_q, BS) * BS, BS):
        o_q = qs + tl.arange(0, BS)
        m_q = o_q < T_q

        p_q = tl.make_block_ptr(
            q + (bos_q * HQ + i_hq) * K,
            (T_q, K),
            (HQ * K, 1),
            (qs, 0),
            (BS, BK),
            (1, 0),
        )
        p_do = tl.make_block_ptr(
            do + (bos_q * HQ + i_hq) * V,
            (T_q, V),
            (HQ * V, 1),
            (qs, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        p_lse = tl.make_block_ptr(
            lse + bos_q * HQ + i_hq, (T_q,), (HQ,), (qs,), (BS,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta + bos_q * HQ + i_hq, (T_q,), (HQ,), (qs,), (BS,), (0,)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2

        b_p = tl.where(m_k[:, None] & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)

        b_dp = tl.dot(b_v, tl.trans(b_do))

        b_ds = b_p * (b_dp - b_delta[None, :])

        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    b_dk = b_dk * scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_DEPTH": lambda args: args["cached_k"] is not None
        and args["cached_v"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def parallel_attn_bwd_kernel_dkv_depth(
    q,
    cached_k,
    cached_v,
    lse,
    delta,
    do,
    d_cached_k,
    d_cached_v,
    L,
    moda_group_num,
    T_q,
    T_kv,
    scale,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_DEPTH: tl.constexpr,
):

    if not USE_DEPTH:
        return

    if IS_VARLEN:
        return

    if tl.cdiv(V, BV) > 1:
        return

    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    q_block_start = i_t * BT
    q_block_end = tl.minimum((i_t + 1) * BT, T_q)

    o_q = q_block_start + tl.arange(0, BT)
    row_valid = o_q < T_q
    o_q_base = o_q // moda_group_num

    base_rows_start = q_block_start // moda_group_num
    base_rows_end = (q_block_end - 1) // moda_group_num + 1
    base_rows_end = tl.minimum(base_rows_end, T_kv)

    depth_block_start = base_rows_start * L
    depth_block_end = base_rows_end * L

    RCP_LN2: tl.constexpr = 1.4426950216

    bos_q = i_b * T_q
    bos_cached = i_b * (T_kv * L)

    p_q = tl.make_block_ptr(
        q + (bos_q * HQ + i_hq) * K,
        (T_q, K),
        (HQ * K, 1),
        (q_block_start, 0),
        (BT, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))

    p_do = tl.make_block_ptr(
        do + (bos_q * HQ + i_hq) * V,
        (T_q, V),
        (HQ * V, 1),
        (q_block_start, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    b_do = tl.load(p_do, boundary_check=(0, 1))

    p_lse = tl.make_block_ptr(
        lse + bos_q * HQ + i_hq, (T_q,), (HQ,), (q_block_start,), (BT,), (0,)
    )
    b_lse = tl.load(p_lse, boundary_check=(0,))

    p_delta = tl.make_block_ptr(
        delta + bos_q * HQ + i_hq, (T_q,), (HQ,), (q_block_start,), (BT,), (0,)
    )
    b_delta = tl.load(p_delta, boundary_check=(0,))

    base_k_cached = cached_k + (bos_cached * H + i_h) * K
    base_v_cached = cached_v + (bos_cached * H + i_h) * V

    for depth_col_start in range(depth_block_start, depth_block_end, BS):

        p_k_depth = tl.make_block_ptr(
            base_k_cached,
            (K, T_kv * L),
            (1, H * K),
            (0, depth_col_start),
            (BK, BS),
            (0, 1),
        )
        p_v_depth = tl.make_block_ptr(
            base_v_cached,
            (T_kv * L, V),
            (H * V, 1),
            (depth_col_start, i_v * BV),
            (BS, BV),
            (1, 0),
        )

        b_k_depth = tl.load(p_k_depth, boundary_check=(0, 1))
        b_v_depth = tl.load(p_v_depth, boundary_check=(0, 1))

        b_s_depth = tl.dot(b_q, b_k_depth) * scale * RCP_LN2

        depth_col_ids = depth_col_start + tl.arange(0, BS)
        depth_row_ids = depth_col_ids // L

        col_valid = depth_col_ids < depth_block_end
        row_match = depth_row_ids[None, :] == o_q_base[:, None]
        mask_depth = (row_valid[:, None]) & col_valid[None, :] & row_match

        b_s_depth = tl.where(mask_depth, b_s_depth, float("-inf"))

        b_p_depth = exp2(b_s_depth - b_lse[:, None])

        b_dp_depth = tl.dot(b_do, tl.trans(b_v_depth))

        b_ds_depth = b_p_depth * (b_dp_depth - b_delta[:, None])

        b_dk_depth = tl.dot(tl.trans(b_ds_depth).to(b_q.dtype), b_q) * scale

        b_dv_depth = tl.dot(tl.trans(b_p_depth).to(b_do.dtype), b_do)

        is_full_tile = (depth_col_start + BS) <= depth_block_end

        if is_full_tile:

            p_dk_out = tl.make_block_ptr(
                d_cached_k + ((bos_cached + depth_col_start) * HQ + i_hq) * K,
                (T_kv * L, K),
                (HQ * K, 1),
                (0, 0),
                (BS, BK),
                (1, 0),
            )
            p_dv_out = tl.make_block_ptr(
                d_cached_v + ((bos_cached + depth_col_start) * HQ + i_hq) * V,
                (T_kv * L, V),
                (HQ * V, 1),
                (0, 0),
                (BS, BV),
                (1, 0),
            )
            tl.store(
                p_dk_out,
                b_dk_depth.to(p_dk_out.dtype.element_ty),
                boundary_check=(0, 1),
            )
            tl.store(
                p_dv_out,
                b_dv_depth.to(p_dv_out.dtype.element_ty),
                boundary_check=(0, 1),
            )

        else:

            depth_ids = depth_col_ids
            valid_rows = depth_ids < depth_block_end

            k_idx = tl.arange(0, K)[None, :]
            dk_row_offsets = ((bos_cached + depth_ids) * HQ + i_hq) * K
            dk_ptrs = d_cached_k + dk_row_offsets[:, None] + k_idx

            tl.store(dk_ptrs, b_dk_depth.to(b_k_depth.dtype), mask=valid_rows[:, None])

            v_idx = tl.arange(0, V)[None, :]
            dv_row_offsets = ((bos_cached + depth_ids) * HQ + i_hq) * V
            dv_ptrs = d_cached_v + dv_row_offsets[:, None] + v_idx
            tl.store(dv_ptrs, b_dv_depth.to(b_v_depth.dtype), mask=valid_rows[:, None])


def parallel_moda_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_moda_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_moda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    scale: float = None,
    chunk_size: int = 128,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cached_k: Optional[torch.Tensor] = None,
    cached_v: Optional[torch.Tensor] = None,
    moda_group_num: int = 1,
    customized_BT: int = None,
    customized_BS: int = None,
    customized_BK: int = None,
    customized_BV: int = None,
    customized_num_warps: int = None,
    group_bs: int = None,
    group_warps: int = None,
    depth_bs: int = None,
    depth_warps: int = None,
):
    assert g_cumsum is None, "g_cumsum is not supported in this version"

    B, T_kv, H, K, V = *k.shape, v.shape[-1]
    T_q = q.shape[1]
    HQ = q.shape[2]
    assert T_q % moda_group_num == 0, "T_q must be divisible by moda_group_num"
    assert (
        T_kv == T_q // moda_group_num
    ), "q/k time dimensions mismatch (T_q = T_kv * moda_group_num)"

    G = HQ // H

    if check_shared_mem("hopper"):
        BT = 128
        BS = 64
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        num_warps = 8
    elif check_shared_mem("ampere"):
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        BT = 128 if K <= 64 else 64
        num_warps = 4
    else:
        BT = 64
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = min(max(triton.next_power_of_2(V), 16), 64)
        num_warps = 2

    if customized_BT is not None:
        BT = customized_BT
    if customized_BS is not None:
        BS = customized_BS
    if customized_BK is not None:
        BK = customized_BK
    if customized_BV is not None:
        BV = customized_BV
    if customized_num_warps is not None:
        num_warps = customized_num_warps

    device_id = q.device.index

    if is_target_gpu(device_id, "H800"):
        BT = 128
        BS = 32
        BK = 64
        BV = 64
        num_warps = 8
        default_group_bs, default_group_warps, default_depth_bs, default_depth_warps = (
            32,
            4,
            64,
            4,
        )

    elif is_target_gpu(device_id, "H20"):
        BT = 32
        BS = 32
        BK = 64
        BV = 64
        num_warps = 4
        default_group_bs, default_group_warps, default_depth_bs, default_depth_warps = (
            64,
            2,
            128,
            4,
        )

    elif is_target_gpu(device_id, "A100"):
        BT = 32
        BS = 32
        BK = 64
        BV = 64
        num_warps = 4
        default_group_bs, default_group_warps, default_depth_bs, default_depth_warps = (
            128,
            4,
            128,
            8,
        )

    else:
        print("Target GPU is not optimized")
        default_group_bs, default_group_warps, default_depth_bs, default_depth_warps = (
            64,
            2,
            128,
            4,
        )

    if group_bs is None:
        group_bs = default_group_bs
    if group_warps is None:
        group_warps = default_group_warps
    if depth_bs is None:
        depth_bs = default_depth_bs
    if depth_warps is None:
        depth_warps = default_depth_warps

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T_q, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)

    delta = parallel_moda_bwd_preprocess(o, do)

    dq = torch.empty(
        B, T_q, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device
    )

    dk = torch.empty(
        B, T_kv, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device
    )
    dv = torch.empty(
        B, T_kv, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device
    )

    dg_cumsum = None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T_q, HQ, dtype=torch.float, device=q.device)

    L = 0
    use_depth = (cached_k is not None) and (cached_v is not None)
    if use_depth:
        L = cached_k.shape[1] // T_kv
        assert (
            L > 0 and cached_v.shape[1] == T_kv * L
        ), "cached_k / cached_v shape mismatch"

    d_cached_k = None
    d_cached_v = None
    if use_depth:
        d_cached_k = torch.empty(
            B,
            T_kv * L,
            HQ,
            K,
            dtype=k.dtype if H == HQ else torch.float,
            device=q.device,
        )
        d_cached_v = torch.empty(
            B,
            T_kv * L,
            HQ,
            V,
            dtype=v.dtype if H == HQ else torch.float,
            device=q.device,
        )

    grid = (NV, NT, B * HQ)

    dg_cumsum_k = None

    parallel_moda_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        g_cumsum=g_cumsum,
        cached_k=cached_k,
        cached_v=cached_v,
        L=L,
        moda_group_num=moda_group_num,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_q,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )

    USE_GROUP_PARALLEL = (moda_group_num > 1) and (cu_seqlens is None)

    import os

    _debug = os.environ.get("MoDA_DEBUG", "0") == "1"

    FAIR_GROUP_PARALLEL = os.environ.get("FAIR_GROUP_PARALLEL", "0") == "1"

    if USE_GROUP_PARALLEL:
        if FAIR_GROUP_PARALLEL:

            BS_GROUP = BS
            NUM_WARPS_GROUP = num_warps
            BT_GROUP = BT
        else:

            BS_GROUP = group_bs
            NUM_WARPS_GROUP = group_warps
            BT_GROUP = BT

        if _debug:
            print(
                f"[DEBUG] dKV using: GROUP-PARALLEL kernel, moda_group_num={moda_group_num}, BS={BS_GROUP}, num_warps={NUM_WARPS_GROUP}, grid=({NV}, {triton.cdiv(T_kv, BT)}, {B * HQ * moda_group_num})"
            )

        dk_grouped = q.new_zeros(B, T_kv, HQ, moda_group_num, K, dtype=torch.float32)
        dv_grouped = q.new_zeros(B, T_kv, HQ, moda_group_num, V, dtype=torch.float32)

        NT_kv_group = triton.cdiv(T_kv, BT_GROUP)
        grid_group = (NV, NT_kv_group, B * HQ * moda_group_num)

        parallel_attn_bwd_kernel_dkv_group_parallel[grid_group](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dk=dk_grouped,
            dv=dv_grouped,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T_kv=T_kv,
            T_q=T_q,
            moda_group_num=moda_group_num,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            MoDA_G=moda_group_num,
            K=K,
            V=V,
            BT=BT_GROUP,
            BS=BS_GROUP,
            BK=BK,
            BV=BV,
            num_warps=NUM_WARPS_GROUP,
        )

        dk = dk_grouped.sum(dim=3).to(dk.dtype)
        dv = dv_grouped.sum(dim=3).to(dv.dtype)
    else:
        if _debug:
            print(
                f"[DEBUG] dKV using: KEY-PARALLEL kernel (original), moda_group_num={moda_group_num}"
            )

        parallel_attn_bwd_kernel_dkv[grid](
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            lse=lse,
            delta=delta,
            do=do,
            dk=dk,
            dv=dv,
            dg_cumsum=dg_cumsum_k,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T_kv,
            moda_group_num=moda_group_num,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
        )

    if use_depth:

        assert NV == 1, "Depth dkv backward currently supports only NV == 1 (BV >= V)"

        if FAIR_GROUP_PARALLEL:
            BS_DEPTH = BS
            NUM_WARPS_DEPTH = num_warps
        else:

            BS_DEPTH = depth_bs
            NUM_WARPS_DEPTH = depth_warps

        parallel_attn_bwd_kernel_dkv_depth[grid](
            q=q,
            cached_k=cached_k,
            cached_v=cached_v,
            lse=lse,
            delta=delta,
            do=do,
            d_cached_k=d_cached_k,
            d_cached_v=d_cached_v,
            L=L,
            moda_group_num=moda_group_num,
            T_q=T_q,
            T_kv=T_kv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BT=BT,
            BS=BS_DEPTH,
            BK=BK,
            BV=BV,
            num_warps=NUM_WARPS_DEPTH,
        )
    dk = reduce(dk, "b t (h g) k -> b t h k", g=G, reduction="sum")
    dv = reduce(dv, "b t (h g) v -> b t h v", g=G, reduction="sum")
    if g_cumsum is not None and dg_cumsum_k is not None:
        dg_cumsum.add_(dg_cumsum_k)

    if use_depth:
        d_cached_k = reduce(
            d_cached_k, "b tl (h g) k -> b tl h k", g=G, reduction="sum"
        )
        d_cached_v = reduce(
            d_cached_v, "b tl (h g) v -> b tl h v", g=G, reduction="sum"
        )

    return dq, dk, dv, dg_cumsum, d_cached_k, d_cached_v


def naive_causal_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    moda_group_num: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    ), "q/k/v must be 4D [B,T(or T_q),H,D]"
    Bq, T_q, Hq, Kq = q.shape
    Bk, T_kv, Hk, Kk = k.shape
    Bv, Tv, Hv, Vdim = v.shape
    assert (
        (Bq, Hq, Kq) == (Bk, Hk, Kk) == (Bv, Hv, Kk)
    ), "Batch / head / K dimensions do not match"
    assert T_kv == Tv, "k/v sequence lengths do not match"
    if moda_group_num == 1:
        assert T_q == T_kv, "When moda_group_num=1, q length must equal k/v length"
    else:
        assert (
            T_q == T_kv * moda_group_num
        ), f"T_q must equal T_kv * moda_group_num, got {T_q} vs {T_kv * moda_group_num}"

    if scale is None:
        scale = 1.0 / math.sqrt(Kq)
    ln2 = math.log(2.0)

    out = torch.empty(Bq, T_q, Hq, Vdim, device=q.device, dtype=torch.float32)
    lse = torch.empty(Bq, T_q, Hq, device=q.device, dtype=torch.float32)

    for b in range(Bq):
        for h in range(Hq):
            q_bh = q[b, :, h].to(torch.float32)
            k_bh = k[b, :, h].to(torch.float32)
            v_bh = v[b, :, h].to(torch.float32)
            for t_q_idx in range(T_q):
                base_t = t_q_idx // moda_group_num

                scores = (q_bh[t_q_idx] @ k_bh[: base_t + 1].T) * scale

                w = torch.softmax(scores, dim=0)
                ctx = (w.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)
                out[b, t_q_idx, h] = ctx
                lse[b, t_q_idx, h] = torch.logsumexp(scores, dim=0) / ln2

    return out, lse


def naive_mixture_of_depth_causal_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kd: Optional[torch.Tensor] = None,
    vd: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    moda_group_num: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    ), "q/k/v must be [B,T(or T_q),H,D]"
    Bq, T_q, Hq, Kq = q.shape
    Bk, T_kv, Hk, Kk = k.shape
    Bv, Tv, Hv, Vdim = v.shape
    assert (
        (Bq, Hq, Kq) == (Bk, Hk, Kk) == (Bv, Hv, Kk)
    ), "Batch / head / K dimensions do not match"
    assert T_kv == Tv, "k/v sequence lengths do not match"
    if moda_group_num == 1:
        assert T_q == T_kv, "When moda_group_num=1, q length must equal k/v length"
    else:
        assert (
            T_q == T_kv * moda_group_num
        ), f"T_q must equal T_kv * moda_group_num, got {T_q} vs {T_kv * moda_group_num}"

    use_depth = (kd is not None) and (vd is not None)
    if use_depth:
        assert kd.ndim == 4 and vd.ndim == 4, "kd/vd must be [B,T_kv*L,H,D]"
        Bkd, TLkd, Hkd, Kkd = kd.shape
        Bvd, TLvd, Hvd, Vd = vd.shape
        assert (Bkd, Hkd, Kkd) == (Bq, Hq, Kq), "kd and q dimensions do not match"
        assert (Bvd, TLvd, Hvd, Vd) == (
            Bq,
            TLkd,
            Hq,
            Vdim,
        ), "vd does not match kd / v dimensions"
        assert (
            TLkd % T_kv == 0
        ), "kd sequence length must be divisible by T_kv (T_kv * L)"
        L = TLkd // T_kv

        kd_reshaped = kd.view(Bq, T_kv, L, Hq, Kq)
        vd_reshaped = vd.view(Bq, T_kv, L, Hq, Vdim)
    else:
        L = 0

    if scale is None:
        scale = 1.0 / math.sqrt(Kq)
    ln2 = math.log(2.0)

    out = torch.empty(Bq, T_q, Hq, Vdim, device=q.device, dtype=torch.float32)
    lse = torch.empty(Bq, T_q, Hq, device=q.device, dtype=torch.float32)

    for b in range(Bq):
        for h in range(Hq):
            q_bh = q[b, :, h].to(torch.float32)
            k_bh = k[b, :, h].to(torch.float32)
            v_bh = v[b, :, h].to(torch.float32)
            if use_depth:
                kd_bh = kd_reshaped[b, :, :, h].to(torch.float32)
                vd_bh = vd_reshaped[b, :, :, h].to(torch.float32)
            for t_q_idx in range(T_q):
                q_vec = q_bh[t_q_idx]
                base_t = t_q_idx // moda_group_num

                logits_space = (q_vec @ k_bh[: base_t + 1].T) * scale

                if use_depth:

                    logits_depth = (kd_bh[base_t] @ q_vec) * scale
                    logits = torch.cat([logits_space, logits_depth], dim=0)
                else:
                    logits = logits_space

                w = torch.softmax(logits, dim=0)

                if use_depth:
                    w_space = w[: base_t + 1]
                    w_depth = w[base_t + 1 :]
                    ctx_space = (w_space.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)
                    ctx_depth = (w_depth.unsqueeze(1) * vd_bh[base_t]).sum(0)
                    out[b, t_q_idx, h] = ctx_space + ctx_depth
                else:
                    out[b, t_q_idx, h] = (w.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)

                lse[b, t_q_idx, h] = torch.logsumexp(logits, dim=0) / ln2

    return out, lse


def naive_mixture_of_depth_causal_ref_vis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kd: Optional[torch.Tensor] = None,
    vd: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    moda_group_num: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    ), "q/k/v must be [B,T(or T_q),H,D]"
    Bq, T_q, Hq, Kq = q.shape
    Bk, T_kv, Hk, Kk = k.shape
    Bv, Tv, Hv, Vdim = v.shape
    assert (
        (Bq, Hq, Kq) == (Bk, Hk, Kk) == (Bv, Hv, Kk)
    ), "Batch / head / K dimensions do not match"
    assert T_kv == Tv, "k/v sequence lengths do not match"
    if moda_group_num == 1:
        assert T_q == T_kv, "When moda_group_num=1, q length must equal k/v length"
    else:
        assert (
            T_q == T_kv * moda_group_num
        ), f"T_q must equal T_kv * moda_group_num, got {T_q} vs {T_kv * moda_group_num}"

    use_depth = (kd is not None) and (vd is not None)
    if use_depth:
        assert kd.ndim == 4 and vd.ndim == 4, "kd/vd must be [B,T_kv*L,H,D]"
        Bkd, TLkd, Hkd, Kkd = kd.shape
        Bvd, TLvd, Hvd, Vd = vd.shape
        assert (Bkd, Hkd, Kkd) == (Bq, Hq, Kq), "kd and q dimensions do not match"
        assert (Bvd, TLvd, Hvd, Vd) == (
            Bq,
            TLkd,
            Hq,
            Vdim,
        ), "vd does not match kd / v dimensions"
        assert (
            TLkd % T_kv == 0
        ), "kd sequence length must be divisible by T_kv (T_kv * L)"
        L = TLkd // T_kv

        kd_reshaped = kd.view(Bq, T_kv, L, Hq, Kq)
        vd_reshaped = vd.view(Bq, T_kv, L, Hq, Vdim)
    else:
        L = 0

    print("layer idx: ", L)

    if scale is None:
        scale = 1.0 / math.sqrt(Kq)
    ln2 = math.log(2.0)

    out = torch.empty(Bq, T_q, Hq, Vdim, device=q.device, dtype=torch.float32)
    lse = torch.empty(Bq, T_q, Hq, device=q.device, dtype=torch.float32)
    attn = torch.zeros(Bq, Hq, T_q, T_kv + 70, device=q.device, dtype=torch.float32)

    for b in range(Bq):
        for h in range(Hq):
            q_bh = q[b, :, h].to(torch.float32)
            k_bh = k[b, :, h].to(torch.float32)
            v_bh = v[b, :, h].to(torch.float32)
            if use_depth:
                kd_bh = kd_reshaped[b, :, :, h].to(torch.float32)
                vd_bh = vd_reshaped[b, :, :, h].to(torch.float32)
            for t_q_idx in range(T_q):
                q_vec = q_bh[t_q_idx]
                base_t = t_q_idx // moda_group_num

                logits_space = (q_vec @ k_bh[: base_t + 1].T) * scale

                if use_depth:

                    logits_depth = (kd_bh[base_t] @ q_vec) * scale
                    logits = torch.cat([logits_space, logits_depth], dim=0)
                else:
                    logits = logits_space

                w = torch.softmax(logits, dim=0)

                w_depth = w[-L:]
                w_spatial = w[:-L]
                attn[b, h, t_q_idx, : len(w_spatial)] = w_spatial
                attn[b, h, t_q_idx, T_kv : T_kv + len(w_depth)] = w_depth

                if use_depth:
                    w_space = w[: base_t + 1]
                    w_depth = w[base_t + 1 :]
                    ctx_space = (w_space.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)
                    ctx_depth = (w_depth.unsqueeze(1) * vd_bh[base_t]).sum(0)
                    out[b, t_q_idx, h] = ctx_space + ctx_depth
                else:
                    out[b, t_q_idx, h] = (w.unsqueeze(1) * v_bh[: base_t + 1]).sum(0)

                lse[b, t_q_idx, h] = torch.logsumexp(logits, dim=0) / ln2

    save_layer_id = L // 2 - 1

    torch.save(attn, f"attn_layer_{save_layer_id}.pt")

    return out, lse


def accuracy_report(out_impl, out_ref, name):
    diff = (out_impl - out_ref).float()
    abs_diff = diff.abs()
    abs_ref = out_ref.abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    mask = abs_ref > 1e-2
    if mask.any():
        max_rel_filtered = (abs_diff[mask] / abs_ref[mask]).max().item()
    else:
        max_rel_filtered = 0.0
    l2_rel = diff.pow(2).sum().sqrt() / (out_ref.pow(2).sum().sqrt() + 1e-12)
    print(
        f"[{name}] max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
        f"max_rel(|ref|>1e-2)={max_rel_filtered:.4e} l2_rel={l2_rel.item():.4e}"
    )


def test_once(
    B=2,
    T=21,
    H=4,
    K=64,
    V=64,
    dtype=torch.float32,
    seed=0,
    device="cuda",
    L: int = 64,
    moda_group_num: int = 1,
):
    torch.manual_seed(seed)
    print(
        f"\n==== Test B={B} T={T} H={H} K={K} V={V} L={L} dtype={dtype} moda_group_num={moda_group_num} ===="
    )

    TQ = T * moda_group_num

    HQ = H
    scale = 1.0 / math.sqrt(K)

    q = torch.randn(B, TQ, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)

    use_depth = L > 0
    if use_depth:
        kd = torch.randn(B, T * L, H, K, dtype=dtype, device=device)
        vd = torch.randn(B, T * L, H, V, dtype=dtype, device=device)
    else:
        kd = None
        vd = None

    ref_out, ref_lse = naive_mixture_of_depth_causal_ref(
        q, k, v, kd=kd, vd=vd, scale=scale, moda_group_num=moda_group_num
    )

    out_impl, lse_impl = parallel_moda_fwd(
        q=q,
        k=k,
        v=v,
        g_cumsum=None,
        scale=scale,
        cu_seqlens=None,
        cached_k=kd,
        cached_v=vd,
        moda_group_num=moda_group_num,
    )

    out_impl_f32 = out_impl.float()
    lse_impl_f32 = lse_impl.float()

    accuracy_report(out_impl_f32, ref_out, "MoDA 6p0 Kernel O")
    accuracy_report(lse_impl_f32, ref_lse, "MoDA 6p0 Kernel LSE")

    return out_impl_f32, ref_out, lse_impl_f32, ref_lse


def main_basic_accuracy():
    device = "cuda"

    test_once(
        B=1,
        T=4096 * 8,
        H=4,
        K=64,
        V=64,
        L=64,
        dtype=torch.float16,
        seed=2,
        device=device,
        moda_group_num=2,
    )


def main_speed_benchmark():
    device = "cuda"
    dtype = torch.float16

    B, T, H, K, V, L = 1, 32768, 4, 64, 64, 72

    moda_group_num = 8

    TQ = T * moda_group_num
    TKV = T
    HKV = H

    print(f"\n==== Speed Benchmark B={B} T={TQ} H={HKV} K={K} V={V} L={L} ====")

    q = torch.randn(B, TQ, HKV, K, dtype=dtype, device=device)
    k = torch.randn(B, TKV, HKV, K, dtype=dtype, device=device)
    v = torch.randn(B, TKV, HKV, V, dtype=dtype, device=device)
    kd = torch.randn(B, TKV * L, HKV, K, dtype=dtype, device=device)
    vd = torch.randn(B, TKV * L, HKV, V, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(K)

    def benchmark(func, name, warmup=5, repeat=20):

        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        import time

        start = time.time()
        for _ in range(repeat):
            func()
        torch.cuda.synchronize()
        avg_time = (time.time() - start) / repeat * 1000
        print(f"{name}: {avg_time:.3f} ms")
        return avg_time

    def modav14p0_spatial_only_test():
        return parallel_moda_fwd(
            q,
            k,
            v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=None,
            cached_v=None,
            moda_group_num=moda_group_num,
        )

    def modav14p0_triton_test():
        return parallel_moda_fwd(
            q,
            k,
            v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=kd,
            cached_v=vd,
            moda_group_num=moda_group_num,
        )

    modav14p0_time = benchmark(
        modav14p0_triton_test, "MoDA v14.0 Triton Implementation"
    )
    modav14p0_spatial_only_time = benchmark(
        modav14p0_spatial_only_test, "Flash Attention (spatial only)"
    )

    print(f"MoDA v14.0 Triton: {modav14p0_time:.3f} ms")
    print(f"Flash Attention V2 (spatial only): {modav14p0_spatial_only_time:.3f} ms")


def causal_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    B, T, H, K = q.shape
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q_, k_.transpose(-1, -2)) * scale
    causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), 1)
    scores.masked_fill_(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_)
    return out.permute(0, 2, 1, 3)


def test_dq_backward(
    B=2, T=32, H=4, K=64, V=64, dtype=torch.float32, seed=0, device="cuda"
):
    torch.manual_seed(seed)
    print(
        f"\n==== DQ TEST (NO DEPTH, moda_group_num=1) B={B} T={T} H={H} K={K} V={V} dtype={dtype} ===="
    )
    scale = 1.0 / math.sqrt(K)

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)

    with torch.no_grad():
        o_kernel, lse_kernel = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=None,
            cached_v=None,
            moda_group_num=1,
        )

    do = torch.randn_like(o_kernel)

    with torch.no_grad():
        dq_impl, _, _, _, _, _ = parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o_kernel,
            g_cumsum=None,
            lse=lse_kernel,
            do=do,
            scale=scale,
            cu_seqlens=None,
            cached_k=None,
            cached_v=None,
            moda_group_num=1,
        )

    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach()
    v_ref = v.clone().detach()

    out_ref, _ = naive_mixture_of_depth_causal_ref(
        q_ref, k_ref, v_ref, kd=None, vd=None, scale=scale, moda_group_num=1
    )
    loss = (out_ref * do).sum()
    loss.backward()
    dq_ref = q_ref.grad

    accuracy_report(dq_impl.float(), dq_ref.float(), "DQ Kernel vs Ref")
    return dq_impl, dq_ref


def test_dkv_backward(
    B=2,
    T=4096,
    H=4,
    K=64,
    V=64,
    dtype=torch.float32,
    device="cuda",
    seed=0,
    moda_group_num=1,
):
    torch.manual_seed(seed)
    print(
        f"\n==== DKV TEST (NO DEPTH, moda_group_num={moda_group_num}) B={B} T={T} H={H} K={K} V={V} dtype={dtype} ===="
    )
    scale = 1.0 / math.sqrt(K)

    if moda_group_num > 1:
        Tq = T * moda_group_num
    else:
        Tq = T

    q = torch.randn(B, Tq, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)

    with torch.no_grad():
        o_kernel, lse_kernel = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=None,
            cached_v=None,
            moda_group_num=moda_group_num,
        )

    do = torch.randn_like(o_kernel)

    with torch.no_grad():
        dq_impl, dk_impl, dv_impl, _, _, _ = parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o_kernel,
            g_cumsum=None,
            lse=lse_kernel,
            do=do,
            scale=scale,
            cu_seqlens=None,
            cached_k=None,
            cached_v=None,
            moda_group_num=moda_group_num,
        )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref, _ = naive_mixture_of_depth_causal_ref(
        q_ref,
        k_ref,
        v_ref,
        kd=None,
        vd=None,
        scale=scale,
        moda_group_num=moda_group_num,
    )
    loss = (out_ref * do).sum()
    loss.backward()

    dk_ref = k_ref.grad
    dv_ref = v_ref.grad

    accuracy_report(dk_impl.float(), dk_ref.float(), "dK Kernel vs Ref")
    accuracy_report(dv_impl.float(), dv_ref.float(), "dV Kernel vs Ref")
    return dk_impl, dk_ref, dv_impl, dv_ref


def test_dkv_backward_depth(
    B=1,
    T_kv=4096,
    H=4,
    K=64,
    V=64,
    L=0,
    moda_group_num=1,
    dtype=torch.float16,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)
    print(
        f"\n==== DKV DEPTH TEST B={B} T_kv={T_kv} H={H} K={K} V={V} L={L} moda_group_num={moda_group_num} dtype={dtype} ===="
    )
    T_q = T_kv * moda_group_num
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device)
    if L > 0:
        kd = torch.randn(B, T_kv * L, H, K, dtype=dtype, device=device)
        vd = torch.randn(B, T_kv * L, H, V, dtype=dtype, device=device)
    else:
        kd = None
        vd = None
    with torch.no_grad():
        o_kernel, lse_kernel = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=kd,
            cached_v=vd,
            moda_group_num=moda_group_num,
        )
    do = torch.randn_like(o_kernel)
    with torch.no_grad():
        dq_impl, dk_impl, dv_impl, _, dkd_impl, dvd_impl = parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o_kernel,
            g_cumsum=None,
            lse=lse_kernel,
            do=do,
            scale=scale,
            cu_seqlens=None,
            cached_k=kd,
            cached_v=vd,
            moda_group_num=moda_group_num,
        )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    kd_ref = kd.detach().clone().requires_grad_(True) if kd is not None else None
    vd_ref = vd.detach().clone().requires_grad_(True) if vd is not None else None
    out_ref, _ = naive_mixture_of_depth_causal_ref(
        q_ref,
        k_ref,
        v_ref,
        kd=kd_ref,
        vd=vd_ref,
        scale=scale,
        moda_group_num=moda_group_num,
    )
    loss = (out_ref * do).sum()
    loss.backward()
    dkd_ref = kd_ref.grad if kd is not None else None
    dvd_ref = vd_ref.grad if vd is not None else None

    if kd is not None:
        accuracy_report(dkd_impl.float(), dkd_ref.float(), "dK_depth Kernel vs Ref")
    if vd is not None:
        accuracy_report(dvd_impl.float(), dvd_ref.float(), "dV_depth Kernel vs Ref")


def test_dq_backward_depth(
    B=1,
    T_kv=4096,
    H=4,
    K=64,
    V=64,
    L=0,
    moda_group_num=1,
    dtype=torch.float16,
    device="cuda",
    seed=0,
):
    print(
        f"\n==== DQ TEST (WITH DEPTH, moda_group_num={moda_group_num}) B={B} T_kv={T_kv} H={H} K={K} V={V} L={L} dtype={dtype} ===="
    )
    torch.manual_seed(seed)
    T_q = T_kv * moda_group_num
    scale = 1.0 / (K**0.5)

    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device)

    if L > 0:
        kd = torch.randn(B, T_kv * L, H, K, dtype=dtype, device=device)
        vd = torch.randn(B, T_kv * L, H, V, dtype=dtype, device=device)
    else:
        kd = None
        vd = None

    with torch.no_grad():
        o, lse = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cached_k=kd,
            cached_v=vd,
            moda_group_num=moda_group_num,
        )

    do = torch.randn_like(o)

    with torch.no_grad():
        dq_impl, _, _, _, _, _ = parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=None,
            lse=lse,
            do=do,
            scale=scale,
            cached_k=kd,
            cached_v=vd,
            moda_group_num=moda_group_num,
        )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach()
    v_ref = v.detach()
    kd_ref = kd.detach() if kd is not None else None
    vd_ref = vd.detach() if vd is not None else None
    out_ref, _ = naive_mixture_of_depth_causal_ref(
        q_ref,
        k_ref,
        v_ref,
        kd=kd_ref,
        vd=vd_ref,
        scale=scale,
        moda_group_num=moda_group_num,
    )
    loss = (out_ref * do).sum()
    loss.backward()
    dq_ref = q_ref.grad
    accuracy_report(dq_impl.float(), dq_ref.float(), "DQ Depth Kernel vs Ref")


def main_dq_accuracy():
    test_dq_backward(B=2, T=4096, H=4, K=64, V=64, dtype=torch.float16)


def main_dkv_accuracy():
    test_dkv_backward(B=2, T=4096, H=4, K=64, V=64, dtype=torch.float16)


RCP_LN2_CONST: float = 1.4426950216


class ParallelMixtureOfDepthAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        scale: float,
        cu_seqlens: Optional[torch.Tensor],
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
        moda_group_num: int = 1,
        group_bs: Optional[int] = None,
        group_warps: Optional[int] = None,
        depth_bs: Optional[int] = None,
        depth_warps: Optional[int] = None,
    ):

        assert g is None, "g is not supported in this version"
        assert cu_seqlens is None, "cu_seqlens is not supported in this version"

        if g is not None:
            g_cumsum = chunk_global_cumsum(
                g, cu_seqlens=cu_seqlens, scale=RCP_LN2_CONST
            )
        else:
            g_cumsum = None

        o, lse = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=moda_group_num,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            g_cumsum,
            lse,
            *(
                [cached_k, cached_v]
                if (cached_k is not None and cached_v is not None)
                else []
            ),
        )
        ctx.has_depth = cached_k is not None and cached_v is not None
        ctx.num_saved_no_depth = 6
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.moda_group_num = moda_group_num
        ctx.has_g = g is not None
        ctx.dtype = q.dtype

        ctx.group_bs = group_bs
        ctx.group_warps = group_warps
        ctx.depth_bs = depth_bs
        ctx.depth_warps = depth_warps

        return o.to(q.dtype), lse

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dlse_unused):

        if ctx.has_depth:
            q, k, v, o, g_cumsum, lse, cached_k, cached_v = ctx.saved_tensors
        else:
            q, k, v, o, g_cumsum, lse = ctx.saved_tensors
            cached_k = cached_v = None

        dq, dk, dv, dg_cumsum, d_cached_k, d_cached_v = parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=ctx.moda_group_num,
            group_bs=ctx.group_bs,
            group_warps=ctx.group_warps,
            depth_bs=ctx.depth_bs,
            depth_warps=ctx.depth_warps,
        )

        if ctx.has_g and dg_cumsum is not None:

            dg = chunk_global_cumsum(dg_cumsum, cu_seqlens=ctx.cu_seqlens, reverse=True)
        else:
            dg = None

        d_scale = None
        d_cu = None
        d_moda = None
        d_group_bs = None
        d_group_warps = None
        d_depth_bs = None
        d_depth_warps = None

        if not ctx.has_depth:
            d_cached_k = None
            d_cached_v = None

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg,
            d_scale,
            d_cu,
            d_cached_k,
            d_cached_v,
            d_moda,
            d_group_bs,
            d_group_warps,
            d_depth_bs,
            d_depth_warps,
        )


def parallel_moda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    cached_k: Optional[torch.Tensor] = None,
    cached_v: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    moda_group_num: int = 1,
    head_first: bool = False,
    need_lse: bool = False,
    return_depth_grads: bool = True,
    warn_shape: bool = True,
    customized_BT: int = None,
    customized_BS: int = None,
    customized_BT_backward: int = None,
    customized_BS_backward: int = None,
    group_bs: int = None,
    group_warps: int = None,
    depth_bs: int = None,
    depth_warps: int = None,
):

    if head_first:

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if g is not None:
            g = g.transpose(1, 2)
        if cached_k is not None and cached_v is not None:
            cached_k = cached_k.transpose(1, 2)
            cached_v = cached_v.transpose(1, 2)
    else:
        if warn_shape and q.shape[1] < q.shape[2]:
            warnings.warn(
                f"[parallel_moda] detected seq_len({q.shape[1]}) < num_heads({q.shape[2]}). "
                "input may be in head_first format while head_first=True is not set."
            )

    B, T_q, HQ, Kdim = q.shape
    T_kv = k.shape[1]
    H = k.shape[2]
    assert HQ % H == 0, "HQ must be an integer multiple of H (GQA)"
    if scale is None:
        scale = Kdim**-0.5

    if moda_group_num > 1:
        assert (
            T_q == T_kv * moda_group_num
        ), "When moda_group_num > 1, T_q must equal T_kv * moda_group_num"
    else:
        assert T_q == T_kv, "When moda_group_num=1, T_q must equal T_kv"

    use_depth = cached_k is not None and cached_v is not None
    if use_depth:
        assert cached_k.shape[0] == B and cached_v.shape[0] == B
        assert (
            cached_k.shape[2] == H and cached_v.shape[2] == H
        ), "cached_k/v head count mismatch"
        assert cached_k.shape[1] % T_kv == 0, "cached_k time dimension must be T_kv * L"
        L = cached_k.shape[1] // T_kv
        assert cached_v.shape[1] == T_kv * L, "cached_v shape mismatch"
        if (
            cached_k.requires_grad or cached_v.requires_grad
        ) and not return_depth_grads:
            warnings.warn(
                "cached_k/v requires gradients, but return_depth_grads=False. Gradients are still computed and returned."
            )

    if g is not None:
        assert g.shape[:3] == (B, T_q, HQ), "g must be [B,T_q,HQ]"

    o, lse = ParallelMixtureOfDepthAttentionFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        cu_seqlens,
        cached_k,
        cached_v,
        moda_group_num,
        group_bs,
        group_warps,
        depth_bs,
        depth_warps,
    )

    if need_lse:
        return o, lse
    else:
        return o


def metric(name, x_impl, x_ref, atol=1e-4, rtol=1e-4):
    diff = (x_impl - x_ref).float()
    max_abs = diff.abs().max().item()
    l2_rel = diff.pow(2).sum().sqrt() / (x_ref.pow(2).sum().sqrt() + 1e-12)
    ok = (max_abs <= atol) or (l2_rel <= rtol * 10)
    print(
        f"[{name}] max_abs={max_abs:.3e} l2_rel={l2_rel:.3e} -> {'OK' if ok else 'MISMATCH'}"
    )
    return ok, max_abs, l2_rel


def run_case(
    case_name,
    B=2,
    T_kv=16,
    H=2,
    K=32,
    V=32,
    L=0,
    moda_group_num=1,
    device="cuda",
    dtype=torch.float32,
    seed=0,
):
    torch.manual_seed(seed)
    T_q = T_kv * moda_group_num
    print(f"\n=== {case_name} ===")
    print(
        f"B={B}, T_kv={T_kv}, T_q={T_q}, H={H}, K={K}, V={V}, L={L}, group={moda_group_num}, dtype={dtype}"
    )

    q = torch.randn(B, T_q, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T_kv, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T_kv, H, V, device=device, dtype=dtype, requires_grad=True)

    if L > 0:
        kd = torch.randn(
            B, T_kv * L, H, K, device=device, dtype=dtype, requires_grad=True
        )
        vd = torch.randn(
            B, T_kv * L, H, V, device=device, dtype=dtype, requires_grad=True
        )
    else:
        kd = None
        vd = None

    scale = 1.0 / math.sqrt(K)

    q_kernel = q.clone().detach().requires_grad_(True)
    k_kernel = k.clone().detach().requires_grad_(True)
    v_kernel = v.clone().detach().requires_grad_(True)
    if kd is not None:
        kd_kernel = kd.clone().detach().requires_grad_(True)
        vd_kernel = vd.clone().detach().requires_grad_(True)
    else:
        kd_kernel = vd_kernel = None

    out_kernel = parallel_moda(
        q_kernel,
        k_kernel,
        v_kernel,
        g=None,
        cached_k=kd_kernel,
        cached_v=vd_kernel,
        scale=scale,
        cu_seqlens=None,
        moda_group_num=moda_group_num,
        head_first=False,
        need_lse=False,
    )

    grad_out = torch.randn_like(out_kernel)
    loss_kernel = (out_kernel * grad_out).sum()
    loss_kernel.backward()

    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    if kd is not None:
        kd_ref = kd.clone().detach().requires_grad_(True)
        vd_ref = vd.clone().detach().requires_grad_(True)
    else:
        kd_ref = vd_ref = None

    out_ref, _lse_ref = naive_mixture_of_depth_causal_ref(
        q_ref,
        k_ref,
        v_ref,
        kd=kd_ref,
        vd=vd_ref,
        scale=scale,
        moda_group_num=moda_group_num,
    )
    loss_ref = (out_ref * grad_out).sum()
    loss_ref.backward()

    metric("Forward/O", out_kernel, out_ref)

    metric("dQ", q_kernel.grad, q_ref.grad)
    metric("dK", k_kernel.grad, k_ref.grad)
    metric("dV", v_kernel.grad, v_ref.grad)
    if L > 0:
        metric("dK_depth", kd_kernel.grad, kd_ref.grad)
        metric("dV_depth", vd_kernel.grad, vd_ref.grad)


def run_one_case(
    *,
    name: str,
    B: int,
    T_kv: int,
    H: int,
    K: int,
    V: int,
    L: int,
    group: int,
    dtype: torch.dtype,
    device: str = "cuda",
    warmup: int = 10,
    repeat: int = 50,
    fixed_seed: int = 1234,
    customized_BT: int = None,
    customized_BS: int = None,
):
    torch.manual_seed(fixed_seed)
    T_q = T_kv * group
    scale = 1.0 / math.sqrt(K)

    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device)
    if L > 0:
        cached_k = torch.randn(B, T_kv * L, H, K, dtype=dtype, device=device)
        cached_v = torch.randn(B, T_kv * L, H, V, dtype=dtype, device=device)
    else:
        cached_k = cached_v = None

    for _ in range(warmup):
        parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(repeat):
        start_event.record()
        parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )
        end_event.record()
        torch.cuda.synchronize()
        total_ms += start_event.elapsed_time(end_event)

    avg_ms = total_ms / repeat
    print(
        f"{name:<28}  B={B} T_kv={T_kv} T_q={T_q} H={H} K={K} V={V} L={L} G={group}  dtype={str(dtype).replace('torch.',''):<9}  {avg_ms:7.3f} ms"
    )
    return avg_ms, T_q


def benchmark_suite(
    *,
    B=2,
    T_kv=4096,
    H=4,
    K=64,
    V=64,
    L_depth=64,
    device="cuda",
    dtypes=(torch.float16, torch.bfloat16),
    warmup=50,
    repeat=100,
    need_baseline_g2=True,
    customized_BT=None,
    customized_BS=None,
):
    print("==== Depth + MoDA Forward Benchmark (minimal) ====")
    print(
        f"Config: B={B}, T_kv={T_kv}, H={H}, K={K}, V={V}, L_depth={L_depth}, warmup={warmup}, repeat={repeat}, customized_BT={customized_BT}, customized_BS={customized_BS}"
    )

    print("-" * 110)

    for dt in dtypes:
        print(f"\n--- DType = {str(dt).replace('torch.','')} ---")

        t_case1, _ = run_one_case(
            name="moda",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=L_depth,
            group=1,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )

        t_case2, _ = run_one_case(
            name="fa2",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=0,
            group=1,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )

        t_case3, _ = run_one_case(
            name="moda + gqa=2",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=L_depth,
            group=2,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )

        if need_baseline_g2:
            t_base_g2, _ = run_one_case(
                name="fa2 + gqa=2",
                B=B,
                T_kv=T_kv,
                H=H,
                K=K,
                V=V,
                L=0,
                group=2,
                dtype=dt,
                device=device,
                warmup=warmup,
                repeat=repeat,
                customized_BT=customized_BT,
                customized_BS=customized_BS,
            )


def run_one_case_bwd(
    *,
    name: str,
    B: int,
    T_kv: int,
    H: int,
    K: int,
    V: int,
    L: int,
    group: int,
    dtype: torch.dtype,
    device: str = "cuda",
    warmup: int = 8,
    repeat: int = 30,
    fixed_seed: int = 1234,
    customized_BT: int = None,
    customized_BS: int = None,
    customized_BT_backward: int = None,
    customized_BS_backward: int = None,
):
    torch.manual_seed(fixed_seed)
    T_q = T_kv * group
    scale = 1.0 / math.sqrt(K)

    q = torch.randn(B, T_q, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device)
    if L > 0:
        cached_k = torch.randn(B, T_kv * L, H, K, dtype=dtype, device=device)
        cached_v = torch.randn(B, T_kv * L, H, V, dtype=dtype, device=device)
    else:
        cached_k = cached_v = None

    for _ in range(warmup):
        o, lse = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )
        do = torch.ones_like(o)
        parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=None,
            lse=lse,
            do=do,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT_backward,
            customized_BS=customized_BS_backward,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(repeat):
        start.record()
        o, lse = parallel_moda_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
        )
        do = torch.ones_like(o)
        parallel_moda_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=None,
            lse=lse,
            do=do,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=group,
            customized_BT=customized_BT_backward,
            customized_BS=customized_BS_backward,
        )
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    avg_ms = total_ms / repeat
    print(
        f"[BWD] {name:<18} B={B} T_kv={T_kv} T_q={T_q} H={H} K={K} V={V} "
        f"L={L:<4} GQA={group:<2} dtype={str(dtype).replace('torch.',''):<9} {avg_ms:7.3f} ms"
    )
    return avg_ms, T_q


def benchmark_suite_backward(
    *,
    B=2,
    T_kv=4096,
    H=4,
    K=64,
    V=64,
    L_depth=64,
    device="cuda",
    dtypes=(torch.float16,),
    warmup=8,
    repeat=30,
    need_baseline_g2=True,
    customized_BT=None,
    customized_BS=None,
    customized_BT_backward=None,
    customized_BS_backward=None,
):
    print("==== MoDA / FA2 Backward Benchmark (Forward+Backward) ====")
    print(
        f"Config: B={B}, T_kv={T_kv}, H={H}, K={K}, V={V}, L_depth={L_depth}, "
        f"warmup={warmup}, repeat={repeat}, customized_BT={customized_BT}, customized_BS={customized_BS}, customized_BT_backward={customized_BT_backward}, customized_BS_backward={customized_BS_backward}"
    )
    print("-" * 115)
    for dt in dtypes:
        print(f"\n--- DType = {str(dt).replace('torch.','')} ---")

        run_one_case_bwd(
            name="moda",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=L_depth,
            group=1,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
            customized_BT_backward=customized_BT_backward,
            customized_BS_backward=customized_BS_backward,
        )

        run_one_case_bwd(
            name="fa2",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=0,
            group=1,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
            customized_BT_backward=customized_BT_backward,
            customized_BS_backward=customized_BS_backward,
        )

        run_one_case_bwd(
            name="moda + gqa=2",
            B=B,
            T_kv=T_kv,
            H=H,
            K=K,
            V=V,
            L=L_depth,
            group=2,
            dtype=dt,
            device=device,
            warmup=warmup,
            repeat=repeat,
            customized_BT=customized_BT,
            customized_BS=customized_BS,
            customized_BT_backward=customized_BT_backward,
            customized_BS_backward=customized_BS_backward,
        )

        if need_baseline_g2:
            run_one_case_bwd(
                name="fa2 + gqa=2",
                B=B,
                T_kv=T_kv,
                H=H,
                K=K,
                V=V,
                L=0,
                group=2,
                dtype=dt,
                device=device,
                warmup=warmup,
                repeat=repeat,
                customized_BT=customized_BT,
                customized_BS=customized_BS,
                customized_BT_backward=customized_BT_backward,
                customized_BS_backward=customized_BS_backward,
            )


def _bench_fwd_only(fwd_fn, warmup=10, iters=50):
    with torch.no_grad():
        for _ in range(warmup):
            fwd_fn()
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            fwd_fn()
        end_evt.record()
        torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters


def _bench_fwd_bwd_timing(fwd_fn, grad_tensors, warmup=10, iters=50):
    def _zero():
        for t in grad_tensors:
            if t.grad is not None:
                t.grad = None

    out = fwd_fn()
    do = torch.randn_like(out)
    out.backward(do)
    _zero()

    for _ in range(warmup):
        out = fwd_fn()
        out.backward(do)
        _zero()

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        out = fwd_fn()
        out.backward(do)
        _zero()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters


def _run_single_method(
    method_name, T_kv, G, B, H, K, V, L, dtype, device, mode, warmup, repeat
):
    from fla.ops.attn import parallel as fa2_mod

    T_q = T_kv * G
    HQ = H * G
    scale = K**-0.5
    need_grad = mode == "fwd_bwd"

    torch.manual_seed(42)
    k = torch.randn(B, T_kv, H, K, dtype=dtype, device=device, requires_grad=need_grad)
    v = torch.randn(B, T_kv, H, V, dtype=dtype, device=device, requires_grad=need_grad)

    if method_name == "FA2_ref":
        q = torch.randn(
            B, T_kv, HQ, K, dtype=dtype, device=device, requires_grad=need_grad
        )

        def fwd_fn():
            return fa2_mod.parallel_attn(q, k, v, g=None, scale=scale, cu_seqlens=None)

        grad_tensors = [q, k, v]

    elif method_name == "MoDAv14+D":
        q = torch.randn(
            B, T_q, H, K, dtype=dtype, device=device, requires_grad=need_grad
        )
        kd = torch.randn(
            B, T_kv * L, H, K, dtype=dtype, device=device, requires_grad=need_grad
        )
        vd = torch.randn(
            B, T_kv * L, H, V, dtype=dtype, device=device, requires_grad=need_grad
        )

        def fwd_fn():
            return parallel_moda(
                q=q,
                k=k,
                v=v,
                g=None,
                scale=scale,
                cu_seqlens=None,
                cached_k=kd,
                cached_v=vd,
                moda_group_num=G,
            )

        grad_tensors = [q, k, v, kd, vd]

    elif method_name == "MoDAv14":
        q = torch.randn(
            B, T_q, H, K, dtype=dtype, device=device, requires_grad=need_grad
        )

        def fwd_fn():
            return parallel_moda(
                q=q,
                k=k,
                v=v,
                g=None,
                scale=scale,
                cu_seqlens=None,
                cached_k=None,
                cached_v=None,
                moda_group_num=G,
            )

        grad_tensors = [q, k, v]

    else:
        raise ValueError(f"Unknown method: {method_name}")

    if mode == "fwd":
        return _bench_fwd_only(fwd_fn, warmup, repeat)
    else:
        return _bench_fwd_bwd_timing(fwd_fn, grad_tensors, warmup, repeat)


def comprehensive_benchmark(
    T_kv_list, G_list, B, H, K, V, L, dtype, device, warmup, repeat, mode
):
    import csv

    method_names = ["FA2_ref", "MoDAv14+D", "MoDAv14"]
    results = []

    print(f"\n{'='*140}")
    print(
        f"Comprehensive Benchmark [{mode}]  B={B} H={H} K={K} V={V} L={L} dtype={dtype}"
    )
    print(f"{'='*140}")

    hdr = f"{'T_kv':>8} {'G':>4} {'T_q':>10}"
    for mn in method_names:
        hdr += f" | {mn:>16}"
    print(hdr)
    print("-" * len(hdr))

    for G in G_list:
        for T_kv in T_kv_list:
            T_q = T_kv * G
            row = {"T_kv": T_kv, "G": G, "T_q": T_q, "mode": mode}
            line = f"{T_kv:>8} {G:>4} {T_q:>10}"

            for mn in method_names:
                torch.cuda.empty_cache()
                try:
                    ms = _run_single_method(
                        mn,
                        T_kv,
                        G,
                        B,
                        H,
                        K,
                        V,
                        L,
                        dtype,
                        device,
                        mode,
                        warmup,
                        repeat,
                    )
                    row[mn] = f"{ms:.3f}"
                    line += f" | {ms:>13.3f} ms"
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        row[mn] = "OOM"
                        line += f" | {'OOM':>16}"
                        torch.cuda.empty_cache()
                    else:
                        raise

            print(line)
            results.append(row)
        print()

    csv_file = f"benchmark_results_{mode}.csv"
    fieldnames = ["T_kv", "G", "T_q", "mode"] + method_names
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_file}")

    return results


if __name__ == "__main__":

    torch.set_printoptions(precision=4, sci_mode=False)

    device = "cuda"
    assert torch.cuda.is_available(), "GPU is required to run this test"

    run_case(
        "Case1: no depth, group=1",
        B=1,
        T_kv=128,
        H=2,
        K=64,
        V=64,
        L=0,
        moda_group_num=1,
        device=device,
        dtype=torch.float16,
        seed=0,
    )

    run_case(
        "Case2: depth L=64, group=1",
        B=1,
        T_kv=128,
        H=2,
        K=64,
        V=64,
        L=4,
        moda_group_num=1,
        device=device,
        dtype=torch.float16,
        seed=1,
    )

    run_case(
        "Case3: depth L=64, group=2",
        B=1,
        T_kv=128,
        H=2,
        K=64,
        V=64,
        L=72,
        moda_group_num=2,
        device=device,
        dtype=torch.float16,
        seed=2,
    )

    comprehensive_benchmark(
        T_kv_list=[1024, 2048, 4096, 8192, 16384],
        G_list=[2, 4, 8, 16, 32],
        B=1,
        H=8,
        K=64,
        V=64,
        L=64,
        dtype=torch.float16,
        device=device,
        warmup=10,
        repeat=50,
        mode="fwd",
    )

    comprehensive_benchmark(
        T_kv_list=[4096],
        G_list=[8],
        B=1,
        H=8,
        K=64,
        V=64,
        L=64,
        dtype=torch.float16,
        device=device,
        warmup=10,
        repeat=50,
        mode="fwd_bwd",
    )
