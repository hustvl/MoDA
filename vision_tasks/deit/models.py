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

        if self.moda_backend == "v17":
            o = parallel_moda_v17(
                q=q, k=k, v=v, g=None,
                scale=self.scale, cu_seqlens=None,
                cached_k=depth_k, cached_v=depth_v,
                moda_group_num=G, is_causal=False,
                head_first=False, warn_shape=False,
                current_depth=current_depth,
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


class MoDABlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moda_backend="v14"):
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, depth_k=None, depth_v=None, current_depth=None):
        attn_out, k, v = self.attn(
            self.norm1(x), depth_k=depth_k, depth_v=depth_v,
            current_depth=current_depth,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, k, v


class MoDAVisionTransformer(nn.Module):
    """MoDA-augmented ViT.

    The ``moda_backend`` argument selects the depth-cache management
    strategy:

    * ``"v14"`` (default, drop-in compatible) -- per-block ``torch.stack``
      rebuild from a per-layer ``kv_list``. ``O(L^2 * BNHK)`` cache
      construction bandwidth across the forward pass; full autograd
      semantics straight out of PyTorch.
    * ``"v17"`` -- the K1 plan: a single ``[B, N*depth, H, D]`` buffer is
      pre-allocated once per forward pass; each block writes its new
      ``[B, N, H, D]`` slice into a slot via ``_WriteDepthSlotKV``
      (``O(BNHK)`` per block, ``O(L * BNHK)`` total). The K1 kernel
      reads only the first ``current_depth`` slots per token. The
      autograd graph is preserved (gradients flow from later layers'
      depth attention back through the slot to the writing layer's
      ``self.kv`` parameters), so v14 and v17 are bit-identical in both
      forward and backward.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, num_kv_heads=1, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 moda_backend="v14", **kwargs):
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
        list. ``O(L^2 * BNHK)`` cache build bandwidth across the forward."""
        kv_list = []
        for blk in self.blocks:
            depth_k, depth_v = self._build_depth_cache(kv_list)
            x, k, v = blk(x, depth_k=depth_k, depth_v=depth_v)
            kv_list.append((k, v))
        return x

    def _forward_blocks_v17(self, x):
        """v17 / K1 backend: pre-allocate one ``[B, N*depth, H, D]`` buffer
        for the whole forward pass and write a single new slot per block.
        ``O(L * BNHK)`` total cache-build bandwidth.

        The ``_WriteDepthSlotKV`` autograd Function (fused K+V) handles
        the in-place slot writes so that gradients still flow from later
        blocks' depth attention back to the source ``self.kv`` of the
        writing block. Fusing K and V into one Function call halves the
        Python-level autograd-dispatch overhead vs a per-buffer Function.
        """
        B, N = x.shape[:2]
        n_layer = self.depth
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
        # write per forward, vs the per-layer slot writes K1 already pays).
        buf_k = torch.zeros(B, N * n_layer, H, D, dtype=x.dtype, device=x.device)
        buf_v = torch.zeros(B, N * n_layer, H, D, dtype=x.dtype, device=x.device)

        bk_state, bv_state = buf_k, buf_v
        for i, blk in enumerate(self.blocks):
            depth_k = bk_state if i > 0 else None
            depth_v = bv_state if i > 0 else None
            x, k, v = blk(x, depth_k=depth_k, depth_v=depth_v, current_depth=i)
            # Last block's slot is never read; skip the fused write entirely
            # so we don't pay the tiny BNHK bandwidth + an extra autograd node.
            if i + 1 < n_layer:
                bk_state, bv_state = _WriteDepthSlotKV.apply(
                    bk_state, bv_state, k, v, i, n_layer,
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
def deit_tiny_moda_patch16_224(pretrained=False, moda_backend="v14", **kwargs):
    """DeiT-Tiny with MoDA depth-attention.

    ``moda_backend`` selects the depth-cache strategy:
      * ``"v14"`` (default) -- ``parallel_moda`` + per-block stack-from-list
        cache. Drop-in autograd semantics, ``O(L^2)`` cache-build cost.
      * ``"v17"`` -- ``parallel_moda_v17`` (K1) + pre-allocated buffer + per-
        block ``current_depth``. Bit-identical forward + backward to v14
        but ``O(L)`` cache-build cost and ~12-17% lower peak memory.
    """
    model = MoDAVisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=4, num_kv_heads=1, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        moda_backend=moda_backend, **kwargs)
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
