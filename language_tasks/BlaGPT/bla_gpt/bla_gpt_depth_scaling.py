"""
Depth-scaling baselines for BlaGPT.

This file is a thin add-on on top of `bla_gpt.py` that introduces:

* `SwiGLUMLPDepthScaling` - subclass of :class:`mlps.SwiGLU_MLP` that swaps
  the hard-coded ``4 * n_embd`` hidden dimension for
  ``int(config.mlp_multiplier * n_embd)``. The forward (gate * silu * proj)
  is inherited unchanged.
* `BlockPostNorm` - subclass of :class:`bla_gpt.Block` that swaps in the
  depth-scaling SwiGLU MLP and overrides the forward to apply post-norm
  residuals (``norm(x + branch(x))``).
* `DepthScalingBaseline512T64LPostNorm1p3BConfig` - the 1.3B-token, 64-layer
  training recipe (depth-scaled from the seed BlaGPT 12-layer baseline at
  `bla_gpt_merged_depth_scaling.DepthScalingBaseline512T12LPostNorm1p3BConfig`).
* `DepthScalingGPT` - GPT subclass that swaps the standard pre-norm blocks for
  `BlockPostNorm`.

The seed recipe (12 layers) is mirrored except for ``n_layer``, which we scale
to 64 here:
    attention            = "regularflex"   (mapped to MoDA's "regular" SDPA Attention)
    use_block_merged_post_norm = True       (mapped to BlockPostNorm here)
    n_head=8, n_embd=512, n_kv_head=2, n_layer=64  (seed: 12)
    block_size=512, sequence_length=512
    mlp_multiplier=2.0
    num_iterations=5100  (~1.3B tokens with batch_size=512, seq=512)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from bla_gpt import GPT, GPTConfig, Block
from mlps import SwiGLU_MLP


class SwiGLUMLPDepthScaling(SwiGLU_MLP):
    """SwiGLU MLP with hidden dim = ``int(config.mlp_multiplier * n_embd)``.

    Reuses :class:`mlps.SwiGLU_MLP` for the SwiGLU forward
    (``c_proj(silu(c_gate(x)) * c_fc(x))`` + dropout) and only overrides the
    three Linear layers so the hidden width follows ``config.mlp_multiplier``
    instead of the hard-coded ``4 * n_embd``.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden = int(config.mlp_multiplier * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)


class BlockPostNorm(Block):
    """Transformer block with post-norm residuals.

    Reuses :class:`bla_gpt.Block`'s ``__init__`` (which wires up ``ln_1``,
    ``attn``, ``ln_2`` via ``get_norm`` / ``get_attention`` / ``get_mlp``) and
    only overrides:

    * ``self.mlp``: replaced by :class:`SwiGLUMLPDepthScaling` so the hidden
      dim follows ``config.mlp_multiplier`` instead of the hard-coded
      ``4 * n_embd`` that ``get_mlp`` would build.
    * :meth:`forward`: applies post-norm residuals
      (``x = ln(x + branch(x))``) instead of ``Block``'s pre-norm pattern
      (``x = x + branch(ln(x))``). The seed reference
      (``BlockMergedPostNorm`` for non-merged attention) reduces to exactly
      this pair of post-norm updates.
    """

    def __init__(self, config, depth):
        super().__init__(config, depth)
        # Replace the default get_mlp(config, depth) (which would be
        # SwiGLU_MLP with the hard-coded 4*n_embd hidden) with the
        # mlp_multiplier-aware variant required by the depth-scaling recipe.
        self.mlp = SwiGLUMLPDepthScaling(config)

    def forward(self, x, **kwargs):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


@dataclass
class DepthScalingBaseline512T64LPostNorm1p3BConfig(GPTConfig):
    """1.3B-token, 64-layer post-norm baseline for depth-scaling sweeps."""

    attention: str = "regular"
    activation: str = "swiglu"

    n_head: int = 8
    n_embd: int = 512
    n_kv_head: int = 2
    n_layer: int = 64

    block_size: int = 512

    mlp_multiplier: float = 2.0

    zero_init_proj_layers: bool = False

    # Hyperparameters overrides applied via train.py's
    # ``for k,v in model_config.to_dict(): setattr(args, k, v)`` loop.
    sequence_length: int = 512
    device_batch_size: int = 8
    num_iterations: int = 5100  # ~1.3B tokens with batch_size=512, seq_len=512


class DepthScalingGPT(GPT):
    """GPT variant that uses :class:`BlockPostNorm` for every transformer layer.

    The base ``GPT.__init__`` builds standard pre-norm blocks first; we then
    swap them for post-norm blocks (which also use the depth-scaling MLP).
    Re-running the GPT-2 style residual init keeps parameter scaling correct
    after the swap.
    """

    def __init__(self, config):
        super().__init__(config)

        new_blocks = nn.ModuleList(
            [BlockPostNorm(config, d) for d in range(config.n_layer)]
        )
        self.transformer.h = new_blocks

        new_blocks.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                if self.zero_init_proj_layers:
                    torch.nn.init.zeros_(p)
                else:
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                    )

        print(
            "DepthScalingGPT: post-norm blocks, "
            f"mlp_multiplier={config.mlp_multiplier}, n_layer={config.n_layer}"
        )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
