"""
Based on the GPT code in "101424_ModernArch" folder (please diff it to see the changes)

Changes:
*) CausalSelfAttention => RWKV-7
*) FFN 4x => 3.5x (to keep params count)
*) rms_norm => LayerNorm
*) use Adam for misc weights (lora, time, etc.)

Note:
Currently runs at 50% GPT speed (when using --wind_cuda) due to:

*) Strangely, takes more VRAM, so I have to reduce device_bsz to 32. I have some theory:
1. Maybe current method of allocating tensor within custom op is suboptimal.
2. Maybe torch amp has trouble deciding fp32 / bf16 especially when using custom op.

*) Torch compile is not efficiently fusing the loras. I noticed JIT can be faster. This becomes worse when using custom op.

I think we can get it to 85% GPT speed @ ctxlen 1024 (can be faster than GPT @ ctxlen 4096) after more work.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from coqpit import Coqpit
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# The main GPT-2 model
HEAD_SIZE = 64
WIND_CUDA = True


@dataclass
class RWKV7Config(Coqpit):
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 768 // HEAD_SIZE
    n_embd: int = 768


# -----------------------------------------------------------------------------
# RWKV-7 kernel

sequence_length = 1024

from torch.utils.cpp_extension import load

if not WIND_CUDA:
    DTYPE = torch.bfloat16
    XTYPE = torch.float
    T = sequence_length
    CHUNK_LEN = 16

    load(
        name="wkv7g",
        sources=["rwkv7/rwkv_cuda/wkv7g_op.cpp", f"rwkv7/rwkv_cuda/wkv7g_v1.cu"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
        ],
    )

    class WKV_7g(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                A = T // CHUNK_LEN
                assert HEAD_SIZE == C // H
                assert T % CHUNK_LEN == 0
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                y = torch.empty(
                    (B, T, C),
                    device=k.device,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                saa = torch.empty(
                    (B, T, H, N),
                    device=k.device,
                    dtype=torch.float,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                sss = torch.empty(
                    (B, H, A, N, N),
                    device=k.device,
                    dtype=torch.float,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                torch.ops.wkv7g.forward(B, T, C, H, r, w, k, v, a, b, y, saa, sss)
                ctx.save_for_backward(r, w, k, v, a, b, saa, sss)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                N = HEAD_SIZE
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                A = T // CHUNK_LEN
                assert gy.dtype == DTYPE
                assert gy.is_contiguous()
                r, w, k, v, a, b, saa, sss = ctx.saved_tensors
                gr = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .zero_()#.uniform_(-100, 100)
                gw = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .zero_()#.uniform_(-100, 100)
                gk = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .zero_()#.uniform_(-100, 100)
                gv = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .zero_()#.uniform_(-100, 100)
                ga = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                gb = torch.empty(
                    (B, T, C),
                    device=gy.device,
                    requires_grad=False,
                    dtype=DTYPE,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                zzz = torch.empty(
                    (B, H, A - 1, N, N),
                    device=gy.device,
                    dtype=XTYPE,
                    memory_format=torch.contiguous_format,
                )  # .uniform_(-100, 100)
                torch.ops.wkv7g.backward(
                    B,
                    T,
                    C,
                    H,
                    r,
                    w,
                    k,
                    v,
                    a,
                    b,
                    saa,
                    sss,
                    zzz,
                    gy,
                    gr,
                    gw,
                    gk,
                    gv,
                    ga,
                    gb,
                )
                del saa
                del sss
                del zzz
                return (gr, gw, gk, gv, ga, gb)

    @torch.compiler.disable
    def RUN_CUDA_RWKV7g(r, w, k, v, a, b):
        return WKV_7g.apply(r, w, k, v, a, b)

else:
    assert HEAD_SIZE == 64  # only for headsz 64

    class WindRWKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, s0):
            B, T, H, C = w.shape
            ctx.s0_is_None = s0 is None
            s0 = (
                torch.zeros(B, H, C, C, dtype=w.dtype, device=w.device)
                if s0 is None
                else s0
            )
            assert T % 16 == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b, s0])
            assert all(i.is_contiguous() for i in [w, q, k, v, a, b, s0])
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B, H, T // 16, C, C, dtype=w.dtype, device=w.device)
            torch.ops.wind.forward(w, q, k, v, a, b, s0, y, s, sT)
            ctx.save_for_backward(w, q, k, v, a, b, s)
            return y, sT

        @staticmethod
        def backward(ctx, dy, dsT):
            assert all(i.dtype == torch.bfloat16 for i in [dy, dsT])
            assert all(i.is_contiguous() for i in [dy, dsT])
            w, q, k, v, a, b, s = ctx.saved_tensors
            dw, dq, dk, dv, da, db, ds0 = [
                torch.empty_like(x) for x in [w, q, k, v, a, b, dsT]
            ]
            torch.ops.wind.backward(
                w, q, k, v, a, b, dy, s, dsT, dw, dq, dk, dv, da, db, ds0
            )
            return dw, dq, dk, dv, da, db, (None if ctx.s0_is_None else ds0)

    wind_rwkv7 = WindRWKV7.apply

    @torch.compile
    def wind(w, q, k, v, a, b, s0=None, return_state=False, path=""):
        if not "wind" in dir(torch.ops):
            B, T, H, C = w.shape
            load(
                name="wind",
                sources=[path + "wind_rwkv7.cu", path + "wind_rwkv7.cpp"],
                is_python_module=False,
                verbose=True,
                extra_cuda_cflags=[
                    f"-D_C_={C}",
                    "-res-usage",
                    "--use_fast_math",
                    "-O3",
                    "-Xptxas -O3",
                    "--extra-device-vectorization",
                ],
            )
        return wind_rwkv7(w, q, k, v, a, b, s0)[: return_state + 1]

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [
            i.view(B, T, HC // HEAD_SIZE, HEAD_SIZE) for i in [q, w, k, v, a, b]
        ]
        return wind(w, q, k, v, a, b, path="rwkv7/rwkv_cuda_wind/")[0].view(B, T, HC)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the RWKV-7 model (optimized for 123M params performance)


class RWKV7(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(
                1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0**0.9)
            )
            self.time_maa_rg = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            )
            self.time_maa_wa = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            )
            self.time_maa_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.time_decay = nn.Parameter(
                decay_speed.reshape(1, 1, args.dim_att) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.head_size)
            )
            self.time_aaaaa = nn.Parameter(torch.zeros(1, 1, args.dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 4))
            self.time_maa_w2 = nn.Parameter(
                ortho_init(torch.zeros(4, D_MIX_LORA, args.n_embd), 0.1)
            )

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_DECAY_LORA, args.dim_att), 0.1)
            )

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_AAA_LORA, args.dim_att), 0.1)
            )

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_KKK_LORA, args.dim_att), 0.1)
            )

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(
                ortho_init(torch.zeros(args.n_embd, D_GATE_LORA), 0.1)
            )
            self.gate_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_GATE_LORA, args.dim_att), 0.1)
            )

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_MA_LORA, args.dim_att), 0.1)
            )
            self.time_misc_a = nn.Parameter(torch.zeros(1, 1, args.n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_MK_LORA, args.dim_att), 0.1)
            )
            self.time_misc_k = nn.Parameter(torch.zeros(1, 1, args.n_embd))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

            self.receptance.weight.data.uniform_(
                -0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5)
            )
            self.key.weight.data.uniform_(
                -0.05 / (self.n_embd**0.5), 0.05 / (self.n_embd**0.5)
            )
            self.value.weight.data.uniform_(
                -0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5)
            )
            self.output.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        mrg, mwa, mk, mv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + mrg)
        xwa = x + xx * (self.time_maa_wa + mwa)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)

        r = self.receptance(xrg)
        w = (
            -F.softplus(
                -(
                    self.time_decay
                    + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2
                )
            )
            - 0.5
        )
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        a = torch.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k * a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w * mk, max=0).exp()

        x = RUN_CUDA_RWKV7g(
            r.bfloat16(),
            w.bfloat16(),
            k.bfloat16(),
            v.bfloat16(),
            -kk.bfloat16(),
            (kk * a).bfloat16(),
        )
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.time_faaaa).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 7 * config.n_embd // 2, bias=False)
        self.c_proj = nn.Linear(7 * config.n_embd // 2, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = RWKV7(config, layer_id)
        self.mlp = MLP(config)

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RWKV7Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config, layer_id) for layer_id in range(config.n_layer)]
                ),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.ln_out = nn.LayerNorm(config.n_embd)

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = x * 2.0  # rescale embedding
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_out(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
