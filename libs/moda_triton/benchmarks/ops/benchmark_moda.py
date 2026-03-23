# -*- coding: utf-8 -*-
import math
import torch
import triton

from fla.utils import device as _device


from fla.ops.attn.parallel import parallel_attn_fwd, parallel_attn_bwd


from fla.ops.moda import parallel_moda_bwd, parallel_moda_fwd

B = 4
H = 8
K = 64
V = 64
DTYPE = torch.bfloat16
REQUIRES_GRAD = True


x_vals = [512 * 2**i for i in range(0, 7)]


DEPTH_CONFIGS = [0, 64]
GROUP_CONFIGS = [1, 2, 4]


ENABLE_GROUP8 = True
if ENABLE_GROUP8:
    GROUP_CONFIGS.append(8)


LINE_SPECS = []


for g in GROUP_CONFIGS:
    LINE_SPECS.append(
        ("fa2_fwd_g{}".format(g), dict(kind="fa2", pass_type="fwd", L=0, group=g))
    )
    LINE_SPECS.append(
        ("fa2_bwd_g{}".format(g), dict(kind="fa2", pass_type="bwd", L=0, group=g))
    )


for L in DEPTH_CONFIGS:
    for g in GROUP_CONFIGS:

        LINE_SPECS.append(
            (
                "moda_fwd_L{}_g{}".format(L, g),
                dict(kind="moda", pass_type="fwd", L=L, group=g),
            )
        )
        LINE_SPECS.append(
            (
                "moda_bwd_L{}_g{}".format(L, g),
                dict(kind="moda", pass_type="bwd", L=L, group=g),
            )
        )


STYLES = []
_base_colors = [
    "green",
    "blue",
    "red",
    "cyan",
    "magenta",
    "orange",
    "black",
    "purple",
    "brown",
    "olive",
]
_line_types = ["-", "--", "-.", ":", "dotted"]
for i in range(200):
    STYLES.append(
        (_base_colors[i % len(_base_colors)], _line_types[i % len(_line_types)])
    )


def _alloc_fa2_inputs(T_kv: int, group: int, dtype, requires_grad):
    Hq = H * group
    q = torch.randn(
        B, T_kv, Hq, K, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    k = torch.randn(
        B, T_kv, H, K, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    v = torch.randn(
        B, T_kv, H, V, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    return q, k, v


def _alloc_moda_inputs(T_kv: int, L: int, group: int, dtype, requires_grad):
    T_q = T_kv * group
    q = torch.randn(
        B, T_q, H, K, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    k = torch.randn(
        B, T_kv, H, K, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    v = torch.randn(
        B, T_kv, H, V, device=_device, dtype=dtype, requires_grad=requires_grad
    )
    if L > 0:
        cached_k = torch.randn(
            B, T_kv * L, H, K, device=_device, dtype=dtype, requires_grad=requires_grad
        )
        cached_v = torch.randn(
            B, T_kv * L, H, V, device=_device, dtype=dtype, requires_grad=requires_grad
        )
    else:
        cached_k = None
        cached_v = None
    return q, k, v, cached_k, cached_v, T_q


def _fa2_forward(q, k, v):

    scale = 1.0 / math.sqrt(K)
    o, lse = parallel_attn_fwd(q, k, v, g_cumsum=None, scale=scale, cu_seqlens=None)
    return o


def _fa2_backward(q, k, v):
    scale = 1.0 / math.sqrt(K)
    o, lse = parallel_attn_fwd(q, k, v, g_cumsum=None, scale=scale, cu_seqlens=None)
    do = torch.ones_like(o)
    parallel_attn_bwd(
        q, k, v, o, g_cumsum=None, lse=lse, do=do, scale=scale, cu_seqlens=None
    )


def _moda_forward(q, k, v, cached_k, cached_v, group):
    scale = 1.0 / math.sqrt(K)
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
    )
    return o


def _moda_backward(q, k, v, cached_k, cached_v, group):
    scale = 1.0 / math.sqrt(K)
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
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T_kv"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=[name for name, _ in LINE_SPECS],
        line_names=[name for name, _ in LINE_SPECS],
        styles=STYLES[: len(LINE_SPECS)],
        ylabel="Execution Time (ms)",
        plot_name="fa2_vs_moda_performance",
        args={},
    )
)
def benchmark(T_kv: int, provider: str):
    torch.cuda.empty_cache()
    spec = None
    for n, s in LINE_SPECS:
        if n == provider:
            spec = s
            break
    if spec is None:
        return (0, 0, 0)

    kind = spec["kind"]
    pass_type = spec["pass_type"]
    L = spec["L"]
    group = spec["group"]

    max_tokens = 4096 * 32
    if kind == "moda":
        T_q = T_kv * group
        if T_q > max_tokens:
            return (0, 0, 0)
        if L > 0 and (T_kv * L > 1_000_000):
            return (0, 0, 0)
    else:

        if T_kv > max_tokens:
            return (0, 0, 0)

    if kind == "fa2":
        q, k, v = _alloc_fa2_inputs(T_kv, group, DTYPE, REQUIRES_GRAD)
        if pass_type == "fwd":

            def run():
                _fa2_forward(q, k, v)

        else:

            def run():
                _fa2_backward(q, k, v)

    else:
        q, k, v, cached_k, cached_v, T_q = _alloc_moda_inputs(
            T_kv, L, group, DTYPE, REQUIRES_GRAD
        )
        if pass_type == "fwd":

            def run():
                _moda_forward(q, k, v, cached_k, cached_v, group)

        else:

            def run():
                _moda_backward(q, k, v, cached_k, cached_v, group)

    quantiles = [0.5, 0.2, 0.8]
    try:
        ms = triton.testing.do_bench(run, quantiles=quantiles)
    except RuntimeError as e:

        print(f"[WARN] provider={provider} T_kv={T_kv} failed: {e}")
        return (0, 0, 0)
    return ms


if __name__ == "__main__":
    print("==== FA2 vs MoDA Benchmark ====")
    print(
        f"Batch={B}, Heads={H}, K={K}, V={V}, dtype={DTYPE}, groups={GROUP_CONFIGS}, depths={DEPTH_CONFIGS}"
    )
    print("Line specs:")
    for name, cfg in LINE_SPECS:
        print(f"  {name}: {cfg}")
    benchmark.run(print_data=True)
