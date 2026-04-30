"""
MoDA depth-scaling recipe for BlaGPT.

64 post-norm transformer blocks where attention and MLP share a depth KV
cache; the attention layer in block ``N`` attends to its current spatial KVs
plus a per-position cache of (K, V) pairs accumulated from blocks ``0..N-1``
(the "depth" axis). The Triton ``parallel_moda`` kernel computes the depth +
spatial attention in a single fused pass.

Components defined here:

* :class:`MixtureOfDepthsAttention` -- attention with depth + spatial KV cache,
  using ``parallel_moda`` for layers ``>= 1`` and SDPA for the first layer
  (where the cache is empty).
* :class:`SwiGLUMLPDepthScalingMoDA` -- SwiGLU MLP that also projects fresh
  K/V from its input and appends them to the depth cache so that the next
  layer's attention can attend to them.
* :class:`BlockPostNormMoDA` -- post-norm transformer block that threads the
  ``(k_list, v_list)`` cache through both attention and MLP branches:
  ``x = ln_1(x + attn(x))`` then ``x = ln_2(x + mlp(x))``.
* :class:`DepthScalingMoDA512T64LPostNorm1p3BConfig` -- the 1.3B-token,
  64-layer training recipe.
* :class:`MoDADepthScalingGPT` -- :class:`bla_gpt.GPT` subclass with a
  depth-cache-aware :meth:`forward` that unwraps ``(x, k_list, v_list)``
  tuples between blocks.

Implementation notes
--------------------
* The Triton kernel is :func:`fla.ops.moda.parallel_moda`; it has the same API
  as the seed's ``fla.ops.dsa.parallel_dsa``, with ``moda_group_num`` playing
  the role of ``dsa_group_num``.
* For the no-cache path (block 0) the seed prefers FlexAttention; we use plain
  SDPA with ``repeat_interleave`` for GQA, which is numerically equivalent at
  ``dropout=0`` and ``scale=1/sqrt(head_dim)`` and avoids the FlexAttention
  dependency.

Seed-name correspondence
------------------------
Ported from the seed BlaGPT repo
(``BlaGPT-seed/bla_gpt/bla_gpt_merged_depth_scaling.py`` and friends). The
seed names are kept here only as cross-references for old configs / logs::

    MoDA name (this file)                            <-  Seed name
    ----------------------------------------------       -------------------
    blagpt_depth_scaling_moda_512t_64l_post_norm_1p3b
                                                     <-  blagpt_depth_scaling_mergedv2dsav4_512t_12l_post_norm_1p3b
    DepthScalingMoDA512T64LPostNorm1p3BConfig        <-  DepthScalingMergedV2DSAV4512T12LPostNorm1p3BConfig
    MoDADepthScalingGPT                              <-  (seed builds via the merged Block factory)
    MixtureOfDepthsAttention                         <-  MergedAttentionV2DSAV4
    SwiGLUMLPDepthScalingMoDA                        <-  MergedV2SwiGLU_MLP
    BlockPostNormMoDA                                <-  BlockMergedPostNorm
    fla.ops.moda.parallel_moda                       <-  fla.ops.dsa.parallel_dsa

The seed config's ``attention="mergedv2dsav4"`` / ``activation="mergedv2swiglu"`` /
``use_block_merged_post_norm=True`` / ``dsa_parallel_impl="parallel_dsa"`` /
``use_triton_depth_spatial=True`` flags are not carried over -- this codebase
hard-wires the equivalent behaviour in :class:`MoDADepthScalingGPT` instead.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from attentions import Attention
from bla_gpt import GPT, GPTConfig, Block
from bla_gpt_depth_scaling import SwiGLUMLPDepthScaling


_MODA_BACKENDS = ("v14", "v17")


def _load_parallel_moda(backend: str = "v14"):
    """Resolve the MoDA Triton entry-point.

    ``backend="v14"`` returns :func:`fla.ops.moda.parallel_moda` (the original
    O(L^2) cache-rebuild path; the caller passes a tightly packed
    ``(B, T*L_cur, h_kv, d)`` cache built via ``torch.cat``).

    ``backend="v17"`` returns :func:`fla.ops.moda.parallel_moda_v17` (the K1
    variant with an extra ``current_depth`` argument; the caller passes a
    pre-allocated ``(B, T*L_max, h_kv, d)`` buffer and the kernel masks out
    slots ``>= current_depth``). Bit-identical to v14 when ``current_depth``
    equals the buffer's L_max.

    Raises a clear :class:`ImportError` describing the missing package install
    if the kernel cannot be located. The kernel lives in ``libs/moda_triton``;
    the debug shell scripts ``pip install`` it before launching ``train.py``.
    """
    if backend not in _MODA_BACKENDS:
        raise ValueError(
            f"Unknown MoDA backend {backend!r}; must be one of {_MODA_BACKENDS}"
        )
    try:
        if backend == "v17":
            from fla.ops.moda import parallel_moda_v17 as _impl
        else:
            from fla.ops.moda import parallel_moda as _impl
    except Exception as exc:
        raise ImportError(
            "MixtureOfDepthsAttention requires "
            f"`fla.ops.moda.parallel_moda{'_v17' if backend == 'v17' else ''}` "
            "from `libs/moda_triton`. Run "
            "`pip install -e /opt/tiger/MoDA/libs/moda_triton` "
            "(or `pip install .` from that directory) before launching "
            "training."
        ) from exc
    return _impl


class _WriteDepthSlotKV(torch.autograd.Function):
    """Fused K+V variant of the autograd-correct in-place slot writer used by
    the v17 / K1 backend.

    Writes ``k_data`` / ``v_data`` into slot ``slot_idx`` of the pre-allocated
    depth buffers ``buf_k`` / ``buf_v`` in a single autograd Function call,
    and on backward extracts slot ``slot_idx`` of each upstream gradient and
    routes it to the corresponding source tensor.

    Layout convention here: buffers are ``(B, T, max_depth, H, D)``, source
    K/V are ``(B, T, H, D)``. The slot is the depth axis (``dim=2``).

    Two implementation details matter for correctness:

    * ``forward`` uses ``buf.data.copy_(...)`` to bypass PyTorch's version
      counter on the buffers. Without this bypass, a later WriteSlot call
      would invalidate the saved ``cached_k`` / ``cached_v`` references inside
      earlier kernel calls' autograd contexts (the kernel snapshot was taken
      *before* the slot was written), tripping the standard "tensor was
      modified by an inplace operation" check.
    * ``backward`` routes only slot ``slot_idx`` of each ``grad_buf`` to the
      corresponding source tensor; the rest of each ``grad_buf`` (carrying
      gradients for other slots) is passed through unchanged so earlier
      ``_WriteDepthSlotKV`` calls in the chain can extract their own slots.

    Why fused K+V (vs one Function per buffer): every ``apply`` call pays the
    Python-level autograd-dispatch overhead (build node, hook, etc.), which
    dominates the per-call cost for small slot writes. Halving the call count
    from ``2*(L-1)`` to ``L-1`` halves that overhead.
    """

    @staticmethod
    def forward(ctx, buf_k, buf_v, k_data, v_data, slot_idx, max_depth):
        B, T, L, H, D = buf_k.shape
        Vdim = buf_v.shape[-1]
        assert L == max_depth, "buf_k depth dim must equal max_depth"
        assert buf_v.shape == (B, T, max_depth, H, Vdim), "buf_v shape mismatch"
        ctx.slot_idx = slot_idx
        ctx.max_depth = max_depth
        ctx.B, ctx.T, ctx.H, ctx.D, ctx.Vdim = B, T, H, D, Vdim
        # ``.data.copy_()`` writes the storage without bumping the buffer's
        # _version, so prior kernel calls' saved-for-backward references stay
        # valid. ``.detach()`` strips autograd metadata on the source tensor
        # before the in-place copy (the gradient path back to it is restored
        # explicitly by ``backward`` below).
        buf_k.data[:, :, slot_idx].copy_(k_data.detach())
        buf_v.data[:, :, slot_idx].copy_(v_data.detach())
        return buf_k, buf_v

    @staticmethod
    def backward(ctx, grad_buf_k, grad_buf_v):
        slot_grad_k = grad_buf_k[:, :, ctx.slot_idx].contiguous()
        slot_grad_v = grad_buf_v[:, :, ctx.slot_idx].contiguous()
        # Pass each grad_buf back unchanged so earlier WriteSlot calls can
        # extract their own slots; residuals eventually arrive at the
        # (no-grad) initial buffers and are harmlessly discarded.
        return grad_buf_k, grad_buf_v, slot_grad_k, slot_grad_v, None, None


class MixtureOfDepthsAttention(Attention):
    """Depth + spatial attention head for the MoDA depth-scaling recipe.

    Reuses :class:`attentions.Attention` for everything that does not depend
    on the depth cache:

    * ``q_proj`` / ``kv_proj`` / ``c_proj`` (with the right GQA shapes)
    * optional ``q_norm`` / ``k_norm`` (RMSNorm before Q/K)
    * RoPE (``rotary`` + ``apply_rope_fn``)
    * ``resid_dropout``
    * ``_project_query`` / ``_project_kv`` / ``_apply_norm`` /
      ``_apply_rotary`` / ``_prepare_qkv`` / ``_flash_attention`` /
      ``_project_output``

    Forward signature::

        forward(x, k_list=None, v_list=None) -> (out, k_list, v_list)

    The depth cache ``k_list`` / ``v_list`` has shape
    ``(B, T, depth, h_kv, head_dim)`` and stores the *pre-RoPE, pre-norm* K/V
    contributions of every preceding attention/MLP block at each position.
    This layout is chosen so that the kernel-feed view
    ``cache.reshape(B, T*depth, h_kv, head_dim)`` is a zero-copy view (T outer,
    depth inner -- exactly the order ``parallel_moda`` consumes), avoiding the
    per-block ``permute().contiguous()`` rebuild of the whole cache that the
    naive ``(B*T, h_kv, depth, head_dim)`` layout would force.

    On block 0 the cache is ``None`` and we delegate to ``Attention``'s
    ``_prepare_qkv`` + ``_flash_attention`` (causal SDPA with GQA expansion).
    From block 1 onwards we feed the current post-RoPE Q/K/V plus the cached
    depth K/V into :func:`parallel_moda` (the MoDA-rebranded ``parallel_dsa``).

    Backend switching
    -----------------
    Two interchangeable kernel backends, selected by ``config.moda_backend``:

    * ``"v14"`` (default, drop-in compatible) -- per-block ``torch.cat``
      append on the depth axis; the cache grows from 1 to ``2*n_layer`` slots
      and each block re-allocates the whole cache (``O(L^2)`` total bandwidth
      across the forward pass).
    * ``"v17"`` -- the K1 plan: a single ``(B, T, max_depth, h_kv, d)``
      buffer is pre-allocated once per forward pass; each block writes its
      new slot via :class:`_WriteDepthSlotKV` (``O(BTHd)`` per block,
      ``O(L * BTHd)`` total). The K1 kernel reads only the first
      ``current_depth`` slots per token, with the rest masked out. The
      autograd graph is preserved (gradients flow from later layers' depth
      attention back through the slot to the writing layer's source K/V),
      so v14 and v17 are bit-equivalent in both forward and backward up to
      fp16/bf16 noise.
    """

    def __init__(self, config):
        super().__init__(config)

        # Triton kernel scale (matches SDPA's default 1/sqrt(d)). Mirrors the
        # seed's ``attention_regularization`` config knob.
        self.attention_regularization = getattr(
            config,
            "attention_regularization",
            None,
        )
        if self.attention_regularization is None:
            self.attention_regularization = 1.0 / math.sqrt(self.head_dim)

        self.moda_backend = getattr(config, "moda_backend", "v17")
        if self.moda_backend not in _MODA_BACKENDS:
            raise ValueError(
                f"Unknown moda_backend={self.moda_backend!r}; "
                f"must be one of {_MODA_BACKENDS}"
            )

        # Eagerly resolve the kernel so configuration mistakes surface during
        # ``__init__`` rather than mid-training.
        self._parallel_moda = _load_parallel_moda(self.moda_backend)

    def forward(
        self,
        x,
        k_list=None,
        v_list=None,
        current_depth=None,
        slot=None,
        max_depth=None,
    ):
        """Dispatch to v14 (cat-based cache) or v17 (K1 buffer) backend.

        v14 mode (``self.moda_backend == "v14"``): only ``k_list`` / ``v_list``
        are consumed; they are tightly packed ``(B, T, depth, h_kv, d)``
        caches. The depth axis grows by 1 per block via ``torch.cat``.

        v17 mode (``self.moda_backend == "v17"``): ``k_list`` / ``v_list``
        are pre-allocated ``(B, T, max_depth, h_kv, d)`` buffers; ``slot``
        is the depth index this block writes to (= number of slots already
        written by earlier blocks); ``current_depth`` is the number of valid
        slots for the kernel to consume (== ``slot`` since this block writes
        *after* its kernel call); ``max_depth`` is the buffer's depth axis.
        """
        if self.moda_backend == "v17":
            return self._forward_v17(
                x, k_list, v_list, current_depth=current_depth,
                slot=slot, max_depth=max_depth,
            )
        return self._forward_v14(x, k_list, v_list)

    def _forward_v14(self, x, k_list=None, v_list=None):
        B, T, C = x.size()

        # Steps 1-2: project Q / K / V using inherited helpers.
        q = self._project_query(x, B, T)            # (B, T, h_q, d)
        k, v = self._project_kv(x, B, T)            # each (B, T, h_kv, d)

        # Cache the *pre-norm, pre-RoPE* K/V (matches seed behaviour).
        # Layout note: ``(B, T, 1, h_kv, d)`` -- ``unsqueeze(2)`` instead of
        # ``reshape(B*T, h_kv, 1, d).contiguous()`` because (a) ``unsqueeze``
        # is a free metadata op (no copy) and (b) the kernel-friendly cache
        # layout is ``(B, T, depth, h_kv, d)`` so the new slot inserts along
        # ``dim=2`` and the kernel feed becomes a zero-copy ``reshape``.
        origin_k = k.unsqueeze(2)                   # (B, T, 1, h_kv, d)
        origin_v = v.unsqueeze(2)

        # Steps 3-4: optional RMSNorm + RoPE on Q / K.
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)
        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T, T)

        if k_list is None or v_list is None:
            # Block 0: standard causal SDPA via inherited helpers
            # (transpose + GQA repeat_interleave + scaled_dot_product_attention).
            q_h, k_h, v_h = self._prepare_qkv(q, k, v)
            y = self._flash_attention(q_h, k_h, v_h)        # (B, h_q, T, d)
        else:
            # Block N>=1: depth-spatial attention via parallel_moda.
            y = self._depth_sequence_attention(q, k, v, k_list, v_list, B, T)

        out = self._project_output(y, B, T, C)              # (B, T, C)

        if k_list is None or v_list is None:
            k_list = origin_k
            v_list = origin_v
        else:
            # ``torch.cat`` along dim=2 grows the depth axis. The output
            # is contiguous, so the next block's ``cached_k.reshape(B, T*L,
            # h_kv, d)`` is a zero-copy view.
            k_list = torch.cat([k_list, origin_k], dim=2)
            v_list = torch.cat([v_list, origin_v], dim=2)

        return out, k_list, v_list

    def _depth_sequence_attention(
        self, q, k, v, k_list, v_list, B, T, current_depth=None,
    ):
        """Call ``parallel_moda`` (v14) or ``parallel_moda_v17`` (K1) on the
        (current spatial) + (depth cache) KVs.

        Inputs:
            q, k, v   (post-RoPE) of shapes (B, T, h_q, d) and (B, T, h_kv, d).
            k_list, v_list shape (B, T, depth, h_kv, d) -- pre-RoPE per-position
            cache, in the kernel-friendly layout (T outer, depth inner).
            For v17/K1 the ``depth`` axis is ``max_depth`` (the buffer's full
            depth); the kernel reads only the first ``current_depth`` slots
            per token. For v14 the ``depth`` axis is the tightly packed
            ``cache_depth`` and ``current_depth`` must be ``None``.

        Returns ``(B, h_q, T, d)`` (head-major) so the caller can route the
        result through the inherited ``_project_output``.
        """
        cache_depth = k_list.size(2)
        d = self.head_dim
        h_kv = self.n_kv_head
        h_q = self.n_head
        group = h_q // h_kv

        # q is (B, T, h_q, d) post-rotary. parallel_moda expects q in
        # ``(B, T_q, H, d)`` with ``T_q = T_kv * group`` and ``H == h_kv``.
        # Reorder ``h_q`` heads so the ``group`` axis becomes the
        # *fast-moving* one inside ``T_q`` (matches seed
        # ``_heads_split_into_time``):
        #   q :: (B, T, h_kv, group, d) -> (B, T, group, h_kv, d)
        #     -> (B, T*group, h_kv, d)
        # ``reshape`` after a non-contig ``permute`` does ``contig + view``
        # in one Python call (one materialisation, not two).
        q_moda = (
            q.reshape(B, T, h_kv, group, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, T * group, h_kv, d)
        )

        # The kernel's ``@input_guard`` already calls ``contiguous()`` on
        # non-contig inputs, so an extra ``.contiguous()`` here would either
        # be a no-op (when already contig) or duplicate the kernel's copy.
        k_moda = k
        v_moda = v

        # k_list / v_list :: (B, T, L, h_kv, d) -- already in kernel-friendly
        # layout, so the (B, T*L, h_kv, d) view is zero-copy. This is the
        # main per-block win vs the original ``(B*T, h_kv, L, d)`` layout
        # which forced an ``O(B*T*L*h_kv*d)`` ``permute().contiguous()`` here.
        cached_k = k_list.reshape(B, T * cache_depth, h_kv, d)
        cached_v = v_list.reshape(B, T * cache_depth, h_kv, d)

        if hasattr(self, "k_norm"):
            cached_k = self.k_norm(cached_k)

        # No-op when dtypes already match (the common case during training);
        # PyTorch returns ``self`` for ``Tensor.to(self.dtype)``.
        cached_k = cached_k.to(q_moda.dtype)
        cached_v = cached_v.to(q_moda.dtype)
        k_moda = k_moda.to(q_moda.dtype)
        v_moda = v_moda.to(q_moda.dtype)

        # v14 ignores ``current_depth``; v17 uses it to mask buffer slots
        # ``>= current_depth``. Passing ``None`` makes v17 fall back to
        # ``L_cur = L_max``, i.e. bit-identical to v14 on the same input.
        kernel_kwargs = (
            {"current_depth": current_depth}
            if current_depth is not None and self.moda_backend == "v17"
            else {}
        )
        y = self._parallel_moda(
            q_moda,
            k_moda,
            v_moda,
            cached_k=cached_k,
            cached_v=cached_v,
            scale=self.attention_regularization,
            moda_group_num=group,
            is_causal=True,
            **kernel_kwargs,
        )  # (B, T*group, h_kv, d) -- contiguous

        # (B, T*group, h_kv, d) -> (B, T, group, h_kv, d) -> permute to
        # (B, T, h_kv, group, d) -> reshape (one materialisation) to
        # (B, T, h_q, d) -> transpose to (B, h_q, T, d).
        # We deliberately omit a trailing ``.contiguous()`` here:
        # ``_project_output`` does ``y.transpose(1, 2).contiguous().view(...)``
        # and ``transpose(1,2)`` of our ``transpose(1,2)`` output reverts the
        # strides exactly, returning a contiguous tensor for free, so the
        # subsequent ``.contiguous()`` becomes a no-op.
        y = (
            y.view(B, T, group, h_kv, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, T, h_q, d)
            .transpose(1, 2)
        )
        return y

    def _forward_v17(
        self, x, buf_k, buf_v, current_depth, slot, max_depth,
    ):
        """K1 path: kernel reads ``buf_{k,v}[:, :, :current_depth]`` (via
        in-kernel masking), then we write the new slot via
        :class:`_WriteDepthSlotKV`.

        Block 0 (``current_depth == 0``) falls back to SDPA so the output is
        bit-identical to v14's block 0 path.
        """
        B, T, C = x.size()

        q = self._project_query(x, B, T)
        k, v = self._project_kv(x, B, T)

        # Save the *pre-norm, pre-RoPE* k/v references for the buffer write
        # (matches v14's ``origin_k`` / ``origin_v``); ``_apply_norm`` and
        # ``_apply_rotary`` return new tensors, so these references stay
        # valid even after ``q, k = self._apply_*(q, k)`` rebinds ``k``.
        k_for_cache = k
        v_for_cache = v

        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)
        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T, T)

        if current_depth == 0:
            # Block 0: empty cache, use SDPA (matches v14 numerics).
            q_h, k_h, v_h = self._prepare_qkv(q, k, v)
            y = self._flash_attention(q_h, k_h, v_h)
        else:
            y = self._depth_sequence_attention(
                q, k, v, buf_k, buf_v, B, T,
                current_depth=current_depth,
            )

        out = self._project_output(y, B, T, C)

        # Write the new slot. The very last slot (``slot == max_depth - 1``)
        # is never read by any future block, so skip its write to save the
        # BTHd bandwidth + an autograd-graph node.
        if slot + 1 < max_depth:
            buf_k, buf_v = _WriteDepthSlotKV.apply(
                buf_k, buf_v, k_for_cache, v_for_cache, slot, max_depth,
            )

        return out, buf_k, buf_v


class SwiGLUMLPDepthScalingMoDA(SwiGLUMLPDepthScaling):
    """SwiGLU MLP that contributes a fresh (K, V) pair to the depth cache.

    Reuses :class:`bla_gpt_depth_scaling.SwiGLUMLPDepthScaling` for the
    ``mlp_multiplier``-aware SwiGLU branch (and through it, ``mlps.SwiGLU_MLP``
    for the actual ``c_proj(silu(c_gate(x)) * c_fc(x))`` forward) and only
    adds an extra ``kv_proj`` whose ``(n_kv_head, head_dim)`` output is
    appended to the depth cache so the next block's attention can see it.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_head = config.n_kv_head
        bias = config.bias or getattr(config, "use_qkv_bias", False)
        self.kv_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim * 2, bias=bias
        )
        self.moda_backend = getattr(config, "moda_backend", "v14")
        if self.moda_backend not in _MODA_BACKENDS:
            raise ValueError(
                f"Unknown moda_backend={self.moda_backend!r}; "
                f"must be one of {_MODA_BACKENDS}"
            )

    def forward(
        self,
        x,
        k_list=None,
        v_list=None,
        slot=None,
        max_depth=None,
    ):
        """Dispatch to v14 (cat-based cache) or v17 (K1 buffer) backend.

        v14: ``(k_list, v_list)`` is the tightly packed
        ``(B, T, depth, h_kv, d)`` cache; the MLP appends one slot via
        ``torch.cat`` along ``dim=2``.

        v17: ``(k_list, v_list)`` is the pre-allocated
        ``(B, T, max_depth, h_kv, d)`` buffer; the MLP writes its new K/V
        into slot ``slot`` via :class:`_WriteDepthSlotKV` (in-place,
        autograd-correct).
        """
        if self.moda_backend == "v17":
            return self._forward_v17(x, k_list, v_list, slot, max_depth)
        return self._forward_v14(x, k_list, v_list)

    def _forward_v14(self, x, k_list=None, v_list=None):
        if k_list is None or v_list is None:
            raise RuntimeError(
                "SwiGLUMLPDepthScalingMoDA requires a non-empty depth cache. "
                "Block 0's attention should have produced (k_list, v_list) "
                "before this MLP runs."
            )

        B, T, _ = x.shape
        k, v = (
            self.kv_proj(x)
            .view(B, T, 2, self.n_kv_head, self.head_dim)
            .unbind(dim=2)
        )
        # Match :class:`MixtureOfDepthsAttention` cache layout so the kernel
        # feed stays a zero-copy ``reshape``: store as ``(B, T, 1, h_kv, d)``.
        origin_k = k.unsqueeze(2)
        origin_v = v.unsqueeze(2)
        k_list = torch.cat([k_list, origin_k], dim=2)
        v_list = torch.cat([v_list, origin_v], dim=2)

        # SwiGLU forward is inherited (mlp_multiplier hidden dim).
        x = super().forward(x)
        return x, k_list, v_list

    def _forward_v17(self, x, buf_k, buf_v, slot, max_depth):
        if buf_k is None or buf_v is None:
            raise RuntimeError(
                "SwiGLUMLPDepthScalingMoDA (v17) requires pre-allocated "
                "(buf_k, buf_v) buffers; the block must allocate them in "
                "block 0 and thread them through."
            )
        B, T, _ = x.shape
        k, v = (
            self.kv_proj(x)
            .view(B, T, 2, self.n_kv_head, self.head_dim)
            .unbind(dim=2)
        )
        # The very last slot is never read; skip the write to save bandwidth
        # + an autograd-graph node.
        if slot + 1 < max_depth:
            buf_k, buf_v = _WriteDepthSlotKV.apply(
                buf_k, buf_v, k, v, slot, max_depth,
            )
        x = super().forward(x)
        return x, buf_k, buf_v


class BlockPostNormMoDA(Block):
    """Post-norm transformer block with depth-cache-aware branches.

    Forward::

        x, k_list, v_list = attn(x, k_list, v_list)
        x = ln_1(x + attn_out)

        x, k_list, v_list = mlp(x, k_list, v_list)
        x = ln_2(x + mlp_out)

    Reuses :class:`bla_gpt.Block`'s ``__init__`` (which wires up ``ln_1`` /
    ``ln_2`` via ``get_norm`` and builds throwaway ``attn`` / ``mlp`` from
    ``get_attention`` / ``get_mlp``) and only overrides:

    * ``self.attn``: replaced by :class:`MixtureOfDepthsAttention` so the
      attention branch can read/write the depth ``(k_list, v_list)`` cache.
    * ``self.mlp``: replaced by :class:`SwiGLUMLPDepthScalingMoDA` so the MLP also
      contributes a fresh K/V pair to the cache.
    * :meth:`forward`: applies post-norm residuals while threading the
      ``(x, k_list, v_list)`` tuple through both branches.
    """

    def __init__(self, config, depth):
        super().__init__(config, depth)
        # Replace the placeholder pre-norm modules built by Block.__init__
        # (using the config's "regular" attention and "swiglu" activation
        # placeholders) with the merged-v4 + merged-v2 variants that own
        # the depth KV cache plumbing.
        self.attn = MixtureOfDepthsAttention(config)
        self.mlp = SwiGLUMLPDepthScalingMoDA(config)

        self.moda_backend = getattr(config, "moda_backend", "v14")
        # v17 / K1 needs the layer index + total depth to compute slot
        # indices into the pre-allocated (B, T, max_depth, h_kv, d) buffer.
        # Each block contributes 2 cache entries (attn + mlp), so the total
        # buffer depth is ``2 * n_layer``.
        self.layer_idx = depth
        self.n_layer = config.n_layer
        self.max_depth = 2 * config.n_layer
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x, token_ids=None, **_kwargs):
        if self.moda_backend == "v17":
            return self._forward_v17(x)
        return self._forward_v14(x)

    def _forward_v14(self, x):
        # The previous block returns ``(x, k_list, v_list)``. The first block
        # is given the bare hidden state as ``x`` and threads its outputs
        # forward.
        if isinstance(x, tuple):
            x, k_list, v_list = x
        else:
            k_list = None
            v_list = None

        attn_out, k_list, v_list = self.attn(x, k_list=k_list, v_list=v_list)
        x = self.ln_1(x + attn_out)

        mlp_out, k_list, v_list = self.mlp(x, k_list=k_list, v_list=v_list)
        x = self.ln_2(x + mlp_out)

        return x, k_list, v_list

    def _forward_v17(self, x):
        """K1 path: thread a single pre-allocated (buf_k, buf_v) pair through
        the whole block stack.

        Block 0 receives the bare hidden state and allocates the buffers
        once for the entire forward pass; subsequent blocks unpack the
        ``(x, buf_k, buf_v)`` tuple from the previous block. Slot indices
        are deterministic from ``self.layer_idx``: the attention writes
        slot ``2*layer_idx`` (after reading the first ``2*layer_idx``
        slots), and the MLP writes slot ``2*layer_idx + 1``. The last MLP's
        slot is intentionally skipped inside ``_forward_v17`` (no future
        read).
        """
        if isinstance(x, tuple):
            x, buf_k, buf_v = x
        else:
            B, T, _ = x.shape
            # ``zeros`` (not ``empty``) because the K1 kernel still issues
            # a ``tl.dot(b_q, b_k_depth)`` over slot positions
            # ``>= current_depth`` before masking them out, and
            # dot-products that include NaN/Inf in an unwritten slot can
            # leak NaN into the masked-out lane in some Triton/HMMA paths.
            # The one-time zero-fill is cheap (one full-buffer write per
            # forward, vs the per-block slot writes K1 already pays).
            buf_k = torch.zeros(
                B, T, self.max_depth, self.n_kv_head, self.head_dim,
                dtype=x.dtype, device=x.device,
            )
            buf_v = torch.zeros_like(buf_k)

        attn_slot = 2 * self.layer_idx
        attn_out, buf_k, buf_v = self.attn(
            x,
            k_list=buf_k, v_list=buf_v,
            current_depth=attn_slot, slot=attn_slot, max_depth=self.max_depth,
        )
        x = self.ln_1(x + attn_out)

        mlp_slot = 2 * self.layer_idx + 1
        mlp_out, buf_k, buf_v = self.mlp(
            x,
            k_list=buf_k, v_list=buf_v,
            slot=mlp_slot, max_depth=self.max_depth,
        )
        x = self.ln_2(x + mlp_out)

        return x, buf_k, buf_v


@dataclass
class DepthScalingMoDA512T64LPostNorm1p3BConfig(GPTConfig):
    """1.3B-token, 64-layer post-norm MoDA depth-spatial attention recipe.

    The ``attention`` / ``activation`` strings are placeholders that
    ``bla_gpt.get_attention`` / ``get_mlp`` recognise; the base
    ``GPT.__init__`` builds throwaway pre-norm blocks with them and
    :class:`MoDADepthScalingGPT.__init__` then swaps in
    :class:`BlockPostNormMoDA`.
    """

    # Placeholders so the base ``GPT.__init__`` can construct the standard
    # pre-norm blocks before we throw them away. The actual model uses
    # MixtureOfDepthsAttention + SwiGLUMLPDepthScalingMoDA regardless of these
    # values.
    attention: str = "regular"
    activation: str = "swiglu"

    n_head: int = 8
    n_embd: int = 512
    n_kv_head: int = 2
    n_layer: int = 64

    block_size: int = 512

    mlp_multiplier: float = 2.0
    zero_init_proj_layers: bool = False

    # MoDA Triton kernel scale; default 1/sqrt(head_dim) is filled in by
    # :class:`MixtureOfDepthsAttention.__init__` when ``None``.
    attention_regularization: Optional[float] = None

    # MoDA depth-cache backend. ``"v14"`` uses the original ``torch.cat``
    # cache (O(L^2) total cache-rebuild bandwidth) and the ``parallel_moda``
    # kernel. ``"v17"`` uses the K1 plan: a single pre-allocated
    # ``(B, T, 2*n_layer, h_kv, d)`` buffer with in-place slot writes via
    # :class:`_WriteDepthSlotKV`, and the ``parallel_moda_v17`` kernel with
    # a ``current_depth`` mask. v14/v17 are bit-equivalent up to fp16/bf16
    # noise; v17 is significantly faster + lower-memory at deep
    # configurations like this 64-layer recipe (cache rebuild drops from
    # O(L^2) to O(L)).
    moda_backend: str = "v17"

    # Hyperparameters overrides applied via train.py's
    # ``for k,v in model_config.to_dict(): setattr(args, k, v)`` loop.
    sequence_length: int = 512
    device_batch_size: int = 8
    num_iterations: int = 5100  # 1.3B tokens with batch_size=512, seq_len=512


class _TupleAwareNorm(nn.Module):
    """Wraps a norm so that ``ln_f((x, k_list, v_list))`` returns ``ln(x)``.

    The base :class:`bla_gpt.GPT.forward` does
    ``x = self.transformer.ln_f(x)`` *without* first unwrapping a tuple, so
    when the depth-cache blocks return ``(x, k_list, v_list)`` we need the
    final norm to drop the cache trail before normalising.
    """

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return self.norm(x)


class MoDADepthScalingGPT(GPT):
    """GPT variant whose layers thread a depth KV cache between blocks.

    The base :class:`bla_gpt.GPT` loops ``x = block(x, token_ids=idx)``;
    :class:`BlockPostNormMoDA` returns a 3-tuple, so the next block sees a
    tuple as its input ``x`` and unwraps it. After the loop we wrap
    ``ln_f`` with :class:`_TupleAwareNorm` so the final norm + LM head still
    receive a plain tensor.
    """

    def __init__(self, config):
        super().__init__(config)

        new_blocks = nn.ModuleList(
            [BlockPostNormMoDA(config, d) for d in range(config.n_layer)]
        )
        self.transformer.h = new_blocks
        self.transformer.ln_f = _TupleAwareNorm(self.transformer.ln_f)

        # Re-run init for the freshly constructed blocks, then apply the
        # GPT-2 residual-projection scaling (mirrors GPT.__init__).
        new_blocks.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                if self.zero_init_proj_layers:
                    torch.nn.init.zeros_(p)
                else:
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                    )

        moda_backend = getattr(config, "moda_backend", "v17")
        kernel_name = "parallel_moda_v17" if moda_backend == "v17" else "parallel_moda"
        print(
            f"MoDADepthScalingGPT: BlockPostNormMoDA + {kernel_name} "
            f"(moda_backend={moda_backend}), "
            f"n_layer={config.n_layer}, n_head={config.n_head}, "
            f"n_kv_head={config.n_kv_head}, mlp_multiplier={config.mlp_multiplier}"
        )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
