# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_


from fla.ops.moda import parallel_moda, parallel_moda_v17

_MODA_BACKENDS = ("v14", "v17")

__all__ = [
    'deit_tiny_patch16_224', 'deit_tiny_gqa_patch16_224', 'deit_tiny_moda_patch16_224',
    'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


class GQAAttention(nn.Module):
    """Grouped-query attention with shared K/V heads."""

    def __init__(self, dim, num_heads=8, num_kv_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.num_kv_heads != self.num_heads:
            # Share each K/V head across a group of query heads.
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GQABlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GQAAttention(
            dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GQAVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, num_kv_heads=1, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            GQABlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class _WriteDepthSlotKV(torch.autograd.Function):
    """Fused K+V variant: write both ``k_data`` into slot ``slot_idx`` of
    ``buf_k`` and ``v_data`` into the same slot of ``buf_v`` in a single
    autograd Function call. Returns ``(buf_k, buf_v)`` so the caller can
    chain them into the next block.

    This is the autograd-correct enabler for the K1 strategy used by the
    ``moda_backend="v17"`` path: per-block cache update stays at
    ``O(BNHK)`` (one strided slot write per buffer) while subsequent
    blocks' depth attention reads of slot ``slot_idx`` still send
    gradients back to the block that wrote that slot.

    Two implementation details matter for correctness:

    * The forward uses ``buf.data.copy_(...)`` to bypass PyTorch's
      version counter on each buffer. Without this bypass, a later
      slot-write would invalidate the saved ``cached_k``/``cached_v``
      references inside earlier kernel calls' autograd contexts (the
      kernel snapshot was taken *before* the slot was written), tripping
      the standard "tensor was modified by an inplace operation" check.
    * The backward routes only slot ``slot_idx`` of each ``grad_buf``
      to the corresponding source tensor; the rest of each ``grad_buf``
      (carrying gradients for other slots) is passed through unchanged
      so earlier ``_WriteDepthSlotKV`` calls in the chain can extract
      their own slots.

    Why fused K+V (vs one Function per buffer): every ``apply`` call
    pays the Python-level autograd-dispatch overhead (build node, hook,
    etc.), which dominates the per-call cost for small slot writes.
    Halving the call count from ``2*(L-1)`` to ``L-1`` halves that
    overhead.
    """

    @staticmethod
    def forward(ctx, buf_k, buf_v, k_data, v_data, slot_idx, n_layer):
        B, NLK, H, D = buf_k.shape
        Vdim = buf_v.shape[-1]
        N = NLK // n_layer
        assert buf_v.shape == (B, N * n_layer, H, Vdim), "buf_v shape mismatch"
        ctx.slot_idx = slot_idx
        ctx.n_layer = n_layer
        ctx.B, ctx.N, ctx.H, ctx.D, ctx.Vdim = B, N, H, D, Vdim
        # `.data.copy_()` writes the storage without bumping the buffer's
        # _version, so prior kernel calls' saved-for-backward references
        # stay valid. ``slot_data.detach()`` strips the autograd metadata
        # on the source tensor before the in-place copy (the gradient
        # path back to it is restored explicitly by ``backward`` below).
        buf_k.data.view(B, N, n_layer, H, D)[:, :, slot_idx].copy_(k_data.detach())
        buf_v.data.view(B, N, n_layer, H, Vdim)[:, :, slot_idx].copy_(v_data.detach())
        return buf_k, buf_v

    @staticmethod
    def backward(ctx, grad_buf_k, grad_buf_v):
        slot_grad_k = (
            grad_buf_k.view(ctx.B, ctx.N, ctx.n_layer, ctx.H, ctx.D)[:, :, ctx.slot_idx]
            .contiguous()
        )
        slot_grad_v = (
            grad_buf_v.view(ctx.B, ctx.N, ctx.n_layer, ctx.H, ctx.Vdim)[:, :, ctx.slot_idx]
            .contiguous()
        )
        # Pass each grad_buf back unchanged so earlier WriteSlot calls
        # can extract their own slots; residuals eventually arrive at
        # the (no-grad) initial buffers and are harmlessly discarded.
        return grad_buf_k, grad_buf_v, slot_grad_k, slot_grad_v, None, None


class MoDAAttention(nn.Module):
    """MoDA attention: replaces standard attention with parallel_moda kernel (is_causal=False)
    and supports attending to depth KV history from previous layers.

    GQA groups are unfolded into the sequence dimension (not head dimension) as
    required by the MoDA kernel: Q is [B, N*G, num_kv_heads, head_dim] while
    K/V remain [B, N, num_kv_heads, head_dim], with moda_group_num=G.

    The ``moda_backend`` argument selects the underlying kernel:

    * ``"v14"`` -- the original ``parallel_moda`` kernel; the caller is
      expected to pass a tightly packed ``[B, N*L_cur, H, D]`` cache built
      via ``torch.stack`` / ``torch.cat`` (autograd flows through the
      stack op naturally).
    * ``"v17"`` -- the K1 variant ``parallel_moda_v17``; the caller is
      expected to pass a pre-allocated ``[B, N*n_layer, H, D]`` buffer
      and the per-layer ``current_depth`` index. Gradient flow back to
      earlier layers is preserved by routing buffer writes through
      ``_WriteDepthSlotKV``.
    """

    def __init__(self, dim, num_heads=8, num_kv_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 moda_backend="v14"):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if moda_backend not in _MODA_BACKENDS:
            raise ValueError(
                f"moda_backend must be one of {_MODA_BACKENDS}, got {moda_backend!r}"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.moda_group_num = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.moda_backend = moda_backend

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, depth_k=None, depth_v=None, current_depth=None):
        B, N, C = x.shape
        G = self.moda_group_num

        # Layout in N*G axis: [tok0_g0, tok0_g1, ..., tok1_g0, ...] -- matches
        # parallel_moda's `o_q_base = o_q // moda_group_num` decoding.
        q = self.q(x).reshape(B, N * G, self.num_kv_heads, self.head_dim)

        # Materialize contiguous K/V exactly once. The non-contiguous slices
        # `kv[:, :, 0/1]` would otherwise be copied twice per layer: once
        # internally by parallel_moda's `@contiguous` (input_guard) wrapper,
        # and again by the `.contiguous()` we used to put on the return path.
        kv = self.kv(x).reshape(B, N, 2, self.num_kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.contiguous()
        v = v.contiguous()

        # Autocast can leave ``q`` / ``k`` / ``v`` as bf16 (Linear is
        # autocasted) while ``depth_k`` / ``depth_v`` -- which were allocated
        # with ``dtype=x.dtype`` in ``_forward_blocks_v17`` *before* the
        # patch_embed Conv2d's autocast applied -- stay fp32. The kernel
        # requires identical dtypes for ``q`` and ``cached_k`` (its
        # ``tl.dot(b_q, b_k_depth)`` raises ``Both operands must be same
        # dtype`` otherwise). ``Tensor.to(self.dtype)`` is a no-op when the
        # dtype already matches, so this is free in the matched case and
        # safe under any future precision changes.
        if depth_k is not None:
            depth_k = depth_k.to(q.dtype)
            depth_v = depth_v.to(q.dtype)

        # ``is_causal=False``: ViT spatial attention is fully bidirectional
        # (every patch attends to every other patch). The kernel applies
        # ``is_causal`` only to the *spatial* K/V branch (the
        # ``o_q_base >= o_k`` mask in ``parallel_moda_fwd_kernel``); the
        # depth K/V branch (``cached_k`` / ``cached_v``) is *unconditionally*
        # masked by ``row_match_mask`` (each query token only attends to its
        # own history slots) and ``slot_valid_mask`` (``slot_in_token <
        # L_cur``), so layer-dimension "causality" is enforced by the
        # forward-pass scheduling -- it does not piggyback on this flag.
        # If we mistakenly passed ``is_causal=True`` here, the kernel would
        # zero out the upper triangle of spatial attention and patch j
        # could no longer see patch j+1..N-1, which is wrong for ViT.
        if self.moda_backend == "v17":
            # ``depth_bs`` / ``depth_warps`` override the autotuner's default
            # config for ``parallel_attn_bwd_kernel_dkv_depth`` (the depth
            # pass of the backward). The library's per-GPU default for A100
            # is ``(128, 8)``; in fp32 this needs ~192KB shared memory and
            # exceeds A100 80GB's 163KB limit on small head-dim cases (see
            # the same fix in ``bla_gpt_moda_depth_scaling.py``). In bf16
            # the same config only uses ~96KB and runs ~2-4% faster than
            # the safe ``(64, 4)`` H800 default, so we apply the override
            # only when the kernel actually risks OOR (fp32 path). Real
            # training is bf16 autocast and lets the autotuner pick.
            kernel_extra = (
                {"depth_bs": 64, "depth_warps": 4}
                if q.dtype == torch.float32
                else {}
            )
            o = parallel_moda_v17(
                q=q, k=k, v=v, g=None,
                scale=self.scale, cu_seqlens=None,
                cached_k=depth_k, cached_v=depth_v,
                moda_group_num=G, is_causal=False,
                head_first=False, warn_shape=False,
                current_depth=current_depth,
                **kernel_extra,
            )
        else:
            o = parallel_moda(
                q=q, k=k, v=v, g=None,
                scale=self.scale, cu_seqlens=None,
                cached_k=depth_k, cached_v=depth_v,
                moda_group_num=G, is_causal=False,
                head_first=False, warn_shape=False,
            )

        # o: [B, N*G, num_kv_heads, D] -> [B, N, G*num_kv_heads*D] = [B, N, C]
        x = o.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k, v


class MoDAMlp(nn.Module):
    """DeiT-style MLP that *also* contributes a fresh (K, V) pair to the
    depth cache (BlaGPT-style "dual-slot" MoDA layout).

    The plain timm ``Mlp`` is augmented with an extra ``kv_proj`` whose
    ``2 * num_kv_heads * head_dim`` output is reshaped into per-token
    ``(k, v)`` of shape ``[B, N, num_kv_heads, head_dim]`` and returned
    alongside the MLP output. The block / vision transformer is responsible
    for routing those tensors into the depth buffer (v17) or the per-layer
    list (v14).

    Why the last block's ``kv_proj`` is stripped at construction time
    (``is_last_layer=True``):
      * In dual-slot mode the last block's MLP slot is ``2*L - 1`` (the
        very last slot of the K1 v17 buffer), which no later attention
        ever reads. Computing it is pure waste.
      * It also leaves ``kv_proj.weight`` *outside* the autograd graph,
        which trips DDP ``find_unused_parameters=False``. Stripping the
        Linear keeps the parameter set clean.
    """

    def __init__(self, in_features, hidden_features, num_kv_heads, head_dim,
                 act_layer=nn.GELU, drop=0., qkv_bias=True,
                 use_depth_kv_projection=True, is_last_layer=False):
        super().__init__()
        out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # ``use_depth_kv_projection=False`` and ``is_last_layer=True`` both
        # disable the kv_proj branch; the latter wins so we never allocate
        # parameters that would land at an unread slot.
        if use_depth_kv_projection and not is_last_layer:
            self.kv_proj = nn.Linear(
                in_features, 2 * num_kv_heads * head_dim, bias=qkv_bias
            )
        else:
            self.kv_proj = None

    def forward(self, x):
        if self.kv_proj is not None:
            B, N, _ = x.shape
            kv = self.kv_proj(x).reshape(B, N, 2, self.num_kv_heads, self.head_dim)
            # ``unbind`` returns non-contiguous views; we materialise once
            # here so the kernel's ``@contiguous`` input_guard doesn't pay
            # for the copy on every layer (mirrors MoDAAttention.forward).
            k, v = kv.unbind(dim=2)
            k = k.contiguous()
            v = v.contiguous()
        else:
            k, v = None, None
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out, k, v


class MoDABlock(nn.Module):
    """Transformer block with MoDA depth attention.

    With ``mlp_depth_kv_projection=True`` (default), the MLP also produces
    a per-token (k, v) pair via :class:`MoDAMlp`, and the block returns the
    full ``(x, k_attn, v_attn, k_mlp, v_mlp)`` tuple so the parent
    transformer can write *both* slots into the depth buffer (one for
    attention, one for the MLP). This mirrors BlaGPT's
    ``BlockPostNormMoDA`` / ``SwiGLUMLPDepthScalingMoDA`` design.

    With ``mlp_depth_kv_projection=False``, the block falls back to the
    timm ``Mlp`` (no extra projection) and returns ``(x, k_attn, v_attn,
    None, None)`` -- this is the legacy "single-slot per layer" layout.
    """

    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moda_backend="v14", mlp_depth_kv_projection=True, is_last_layer=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MoDAAttention(
            dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            moda_backend=moda_backend,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_depth_kv_projection = mlp_depth_kv_projection
        if mlp_depth_kv_projection:
            self.mlp = MoDAMlp(
                in_features=dim, hidden_features=mlp_hidden_dim,
                num_kv_heads=num_kv_heads, head_dim=dim // num_heads,
                act_layer=act_layer, drop=drop, qkv_bias=qkv_bias,
                use_depth_kv_projection=True,
                is_last_layer=is_last_layer,
            )
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, depth_k=None, depth_v=None, current_depth=None):
        attn_out, k_attn, v_attn = self.attn(
            self.norm1(x), depth_k=depth_k, depth_v=depth_v,
            current_depth=current_depth,
        )
        x = x + self.drop_path(attn_out)
        if self.mlp_depth_kv_projection:
            mlp_out, k_mlp, v_mlp = self.mlp(self.norm2(x))
            x = x + self.drop_path(mlp_out)
            return x, k_attn, v_attn, k_mlp, v_mlp
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        return x, k_attn, v_attn, None, None


class MoDAVisionTransformer(nn.Module):
    """MoDA-augmented ViT.

    The ``moda_backend`` argument selects the depth-cache management
    strategy:

    * ``"v14"`` -- per-block ``torch.stack`` rebuild from a per-layer
      ``kv_list``. ``O(L^2 * BNHK)`` cache construction bandwidth across
      the forward pass; full autograd semantics straight out of PyTorch.
    * ``"v17"`` (recommended) -- the K1 plan: a single
      ``[B, N*max_depth, H, D]`` buffer is pre-allocated once per forward
      pass; each block writes its new ``[B, N, H, D]`` slice into a slot
      via ``_WriteDepthSlotKV`` (``O(BNHK)`` per block,
      ``O(slots * BNHK)`` total). The K1 kernel reads only the first
      ``current_depth`` slots per token. The autograd graph is preserved
      (gradients flow from later layers' depth attention back through
      the slot to the writing layer's ``self.kv`` / MLP ``kv_proj``
      parameters), so v14 and v17 are bit-identical in both forward and
      backward.

    The ``mlp_depth_kv_projection`` argument (default ``True``) controls
    the per-layer cache layout:

    * ``True`` (BlaGPT-style "dual slot per layer"): each block writes two
      slots, one from attention and one from the MLP's ``kv_proj`` branch
      (see :class:`MoDAMlp`). Total cache depth is ``2 * depth`` and the
      MLP gains a learnable ``[in_features, 2 * num_kv_heads * head_dim]``
      Linear per non-final block.
    * ``False`` (legacy "single slot per layer"): only attention writes a
      slot; the block uses the timm ``Mlp`` and the cache depth is
      ``depth``. Useful for ablations / re-creating the original DeiT
      MoDA experiments.

    Within a fixed ``mlp_depth_kv_projection`` setting, ``v14`` and
    ``v17`` produce numerically identical forward and backward outputs.
    Different ``mlp_depth_kv_projection`` settings give different models
    (extra parameters and an extra read row per layer), so checkpoints
    are *not* interchangeable across this flag.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, num_kv_heads=1, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 moda_backend="v14", mlp_depth_kv_projection=True, **kwargs):
        super().__init__()
        if moda_backend not in _MODA_BACKENDS:
            raise ValueError(
                f"moda_backend must be one of {_MODA_BACKENDS}, got {moda_backend!r}"
            )
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.moda_backend = moda_backend
        # ``mlp_depth_kv_projection=True`` activates the BlaGPT-style "dual
        # slot per layer" cache layout: each block writes 2 slots (one for
        # attention, one for the MLP) instead of 1, giving the depth
        # attention twice as many history rows to attend to per layer at
        # the cost of an extra ``(2 * num_kv_heads * head_dim)`` Linear per
        # block (last block excluded, see :class:`MoDAMlp`). Buffer / cache
        # depth is ``depth * slots_per_layer``.
        self.mlp_depth_kv_projection = mlp_depth_kv_projection
        self.slots_per_layer = 2 if mlp_depth_kv_projection else 1
        self.max_depth = depth * self.slots_per_layer
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MoDABlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                moda_backend=moda_backend,
                mlp_depth_kv_projection=mlp_depth_kv_projection,
                # Last block's slot lands at ``max_depth - 1`` and is never
                # read; :class:`MoDAMlp` skips its kv_proj entirely so the
                # parameter stays out of the autograd graph (DDP-friendly).
                is_last_layer=(i == depth - 1),
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _build_depth_cache(self, kv_list):
        """Stack per-layer KV into the depth cache format expected by the MoDA kernel.

        Each entry in kv_list is (k, v) with shape [B, N, H, K].
        Returns cached_k, cached_v each of shape [B, N*L, H, K] where L = len(kv_list),
        laid out as [tok0_depth0, tok0_depth1, ..., tok1_depth0, ...] -- matches
        parallel_moda's `depth_row_ids = depth_col_ids // L` decoding.

        ``torch.stack(..., dim=2)`` already yields a contiguous [B, N, L, H, K]
        tensor, so the subsequent ``reshape(B, N*L, H, K)`` is a free view and
        no extra ``.contiguous()`` is needed.
        """
        if not kv_list:
            return None, None
        keys = torch.stack([kv[0] for kv in kv_list], dim=2)   # [B, N, L, H, K]
        vals = torch.stack([kv[1] for kv in kv_list], dim=2)   # [B, N, L, H, V]
        B, N, L, H, K = keys.shape
        cached_k = keys.reshape(B, N * L, H, K)
        cached_v = vals.reshape(B, N * L, H, vals.shape[-1])
        return cached_k, cached_v

    def _forward_blocks_v14(self, x):
        """v14 backend: per-block ``torch.stack`` rebuild from a per-layer
        list. ``O(slots^2 * BNHK)`` cache build bandwidth across the forward.

        With ``mlp_depth_kv_projection=True`` each block appends *two*
        entries to ``kv_list`` (attention's k/v then MLP's k/v); with
        ``False`` it appends one. The last block's contributions are
        intentionally dropped: no future attention reads them, so packing
        them into the cache would only inflate the next layer's depth
        attention -- but there is no next layer. (For the dual-slot case
        the MLP's last-block ``kv_proj`` is also stripped at construction
        time, so we never compute it in the first place.)
        """
        kv_list = []
        n_layer = self.depth
        for i, blk in enumerate(self.blocks):
            depth_k, depth_v = self._build_depth_cache(kv_list)
            x, k_attn, v_attn, k_mlp, v_mlp = blk(x, depth_k=depth_k, depth_v=depth_v)
            if i + 1 < n_layer:
                kv_list.append((k_attn, v_attn))
                if self.mlp_depth_kv_projection:
                    # k_mlp / v_mlp are guaranteed non-None for non-final
                    # blocks (MoDAMlp only strips kv_proj on the last layer).
                    kv_list.append((k_mlp, v_mlp))
        return x

    def _forward_blocks_v17(self, x):
        """v17 / K1 backend: pre-allocate one ``[B, N*max_depth, H, D]``
        buffer for the whole forward pass and write per-block slot(s) in
        place. ``O(slots * BNHK)`` total cache-build bandwidth.

        Slot layout (dual-slot, ``mlp_depth_kv_projection=True``)::

            block 0: attn -> slot 0, mlp  -> slot 1
            block 1: attn -> slot 2, mlp  -> slot 3   (reads slots 0..1)
            ...
            block i: attn -> slot 2i,    mlp  -> slot 2i+1  (reads slots 0..2i-1)

        Slot layout (single-slot, ``mlp_depth_kv_projection=False``)::

            block i: attn -> slot i  (reads slots 0..i-1)

        The ``_WriteDepthSlotKV`` autograd Function (fused K+V) handles
        the in-place slot writes so that gradients still flow from later
        blocks' depth attention back to the source ``self.kv`` (or MLP
        ``kv_proj``) of the writing block. Fusing K and V into one
        Function call halves the Python-level autograd-dispatch overhead
        vs a per-buffer Function.
        """
        B, N = x.shape[:2]
        n_layer = self.depth
        max_depth = self.max_depth
        slots_per_layer = self.slots_per_layer
        H = self.num_kv_heads
        D = self.head_dim
        # The buffers themselves do not need a grad slot; gradients are
        # routed to the source K/V tensors by ``_WriteDepthSlotKV.backward``.
        #
        # We use ``zeros`` (not ``empty``) because the K1 kernel still issues
        # a ``tl.dot(b_q, b_k_depth)`` over slot positions ``>= current_depth``
        # before masking them out, and dot-products that include NaN/Inf in
        # an unwritten slot can leak NaN into the masked-out lane in some
        # Triton/HMMA paths. The one-time zero-fill is cheap (one full-buffer
        # write per forward, vs the per-block slot writes K1 already pays).
        #
        # Buffer dtype must match the autocast dtype (e.g. bf16 under
        # ``torch.autocast``) rather than ``x.dtype``: under autocast, ``x`` is
        # the LayerNorm input/output (fp32) but the per-layer ``q``/``k``/``v``
        # produced by the autocasted ``nn.Linear`` in ``MoDAAttention`` are
        # bf16, and the kernel requires identical dtypes for ``q`` and
        # ``cached_k``. Allocating the buffer in fp32 here would force
        # ``MoDAAttention.forward`` to materialise a *fresh* bf16 copy
        # ``cached_k.to(q.dtype)`` *every block*, which then gets
        # ``save_for_backward``'d by the K1 kernel; for L=12 that's ~427 MB
        # of redundant activations vs the v14 path. ``get_autocast_gpu_dtype``
        # gives us the active autocast dtype when enabled, falling back to
        # ``x.dtype`` otherwise.
        if torch.is_autocast_enabled():
            cache_dtype = torch.get_autocast_gpu_dtype()
        else:
            cache_dtype = x.dtype
        buf_k = torch.zeros(B, N * max_depth, H, D, dtype=cache_dtype, device=x.device)
        buf_v = torch.zeros(B, N * max_depth, H, D, dtype=cache_dtype, device=x.device)

        bk_state, bv_state = buf_k, buf_v
        for i, blk in enumerate(self.blocks):
            attn_slot = i * slots_per_layer
            current_depth = attn_slot
            depth_k = bk_state if current_depth > 0 else None
            depth_v = bv_state if current_depth > 0 else None
            x, k_attn, v_attn, k_mlp, v_mlp = blk(
                x, depth_k=depth_k, depth_v=depth_v,
                current_depth=current_depth,
            )
            # Last block's slot(s) are never read; skip the fused write
            # entirely so we don't pay the tiny BNHK bandwidth + extra
            # autograd nodes.
            if i + 1 < n_layer:
                bk_state, bv_state = _WriteDepthSlotKV.apply(
                    bk_state, bv_state, k_attn, v_attn, attn_slot, max_depth,
                )
                if self.mlp_depth_kv_projection:
                    # k_mlp / v_mlp are guaranteed non-None for non-final
                    # blocks; MoDAMlp strips kv_proj on the last layer.
                    mlp_slot = attn_slot + 1
                    bk_state, bv_state = _WriteDepthSlotKV.apply(
                        bk_state, bv_state, k_mlp, v_mlp, mlp_slot, max_depth,
                    )
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.moda_backend == "v17":
            x = self._forward_blocks_v17(x)
        else:
            x = self._forward_blocks_v14(x)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_gqa_patch16_224(pretrained=False, **kwargs):
    model = GQAVisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=4, num_kv_heads=1, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError("No pretrained weights are available for deit_tiny_gqa_patch16_224.")
    return model


@register_model
def deit_tiny_moda_patch16_224(pretrained=False, moda_backend="v17",
                               mlp_depth_kv_projection=True, **kwargs):
    """DeiT-Tiny with MoDA depth-attention.

    ``moda_backend`` selects the depth-cache strategy:
      * ``"v14"`` -- ``parallel_moda`` + per-block stack-from-list cache.
        Drop-in autograd semantics, ``O(slots^2)`` cache-build cost. Kept
        available for parity testing / debugging.
      * ``"v17"`` (default) -- ``parallel_moda_v17`` (K1) + pre-allocated
        buffer + per-block ``current_depth``. Bit-identical forward +
        backward to v14 but ``O(slots)`` cache-build cost and ~12-17%
        lower peak memory; this is the production-recommended path.

    ``mlp_depth_kv_projection`` (default ``True``) controls whether the
    MLP also writes a slot to the depth cache (BlaGPT-style "dual slot
    per layer" layout). See :class:`MoDAVisionTransformer` for the
    semantics. Setting this to ``False`` recovers the original
    "single slot per layer" DeiT MoDA design.
    """
    model = MoDAVisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=4, num_kv_heads=1, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        moda_backend=moda_backend,
        mlp_depth_kv_projection=mlp_depth_kv_projection,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError("No pretrained weights are available for deit_tiny_moda_patch16_224.")
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
