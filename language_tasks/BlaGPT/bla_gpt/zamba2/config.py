from dataclasses import dataclass, field
from typing import Callable, List

import torch.nn.functional as F
from coqpit import Coqpit
from zamba2.utils import init_method_normal, scaled_init_method_normal


@dataclass
class MambaConfig(Coqpit):
    # model architecture
    base_model_type: str = "zamba2"
    max_sequence_length: int = 1024
    num_layers: int = 18
    hidden_size: int = 768
    state_size: int = 64
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    expansion_factor: int = 2
    conv_dimension: int = 4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    use_module_layernorm: bool = True

    rms_norm: bool = True
    fused_add_norm: bool = False
    residual_in_fp32: bool = False
    hidden_dropout: float = 0.0
    ffn_hidden_size: int = 768 * 4
    gated_linear_unit: bool = True
    kv_channels: int = 80
    kv_mem_channels: int = None
    num_attention_heads: int = 12
    num_query_groups: int = 12
    num_mem_query_groups: int = None
    attention_dropout: float = 0.0
    num_mem_heads: int = 0
    use_mem_mlp: bool = False
    window_size: int = None
    gateconv_expansion_factor: int = 2
    layer_mapping: List[str] = field(
        default_factory=lambda: [
            "m",
            "m",
            "m",
            "m",
            "m",
            "g",
            "m",
            "m",
            "m",
            "m",
            "g",
            "m",
            "m",
            "m",
            "g",
            "m",
            "m",
            "m",
        ]
    )
    vocab_size: int = 50304
    device: str = "cuda"
    use_mem_rope: bool = False
    use_shared_block_lora: bool = True
    lora_rank: int = 128
    num_mem_blocks: int = 2
    use_low_rank_mamba_proj: bool = False
    num_shared_mamba_proj: int = 1
    mamba_lora_rank: int = 1
    use_shared_attention_lora: bool = False
    rope_theta: int = 10000

    fp32_residual_connection: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = False
    activation_func: Callable = F.gelu
    num_moe_experts: int = None

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    gated_linear_unit: bool = False
    bias_gelu_fusion: bool = False
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None

    def __post_init__(self):
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None and self.num_attention_heads is not None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.kv_mem_channels is None and self.num_mem_heads > 0:
            self.kv_mem_channels = self.hidden_size // self.num_mem_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_mem_query_groups is None:
            self.num_mem_query_groups = self.num_mem_heads

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_gelu_fusion:
            if not self.add_bias_linear:
                raise ValueError(
                    "When bias_gelu_fusion is True, add_bias_linear must also be True."
                )

            if self.activation_func != F.gelu:
                raise ValueError(
                    f"When bias_gelu_fusion is True, activation_func must be F.gelu."
                )

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )
