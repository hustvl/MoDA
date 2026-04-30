import torch
import torch.nn as nn
import torch.nn.functional as F
from zamba2.config import MambaConfig
from zamba2.utils import bias_gelu_impl


class MLP(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        is_expert: bool = False,
        layer_idx=None,
        num_mem_blocks=None,
    ):
        super().__init__()

        self.num_mem_blocks = num_mem_blocks

        self.config: MambaConfig = config
        self.layer = layer_idx
        ffn_hidden_size_1 = self.config.ffn_hidden_size
        ffn_hidden_size_2 = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size_1 *= 2

        if self.layer == -1:
            ffn_hidden_size_1 = 8 * self.config.hidden_size

        self.linear_fc1 = nn.Linear(
            self.config.hidden_size, ffn_hidden_size_1, bias=self.config.add_bias_linear
        )

        if self.config.gated_linear_unit or self.layer == -1:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = nn.Linear(
            ffn_hidden_size_2, self.config.hidden_size, bias=self.config.add_bias_linear
        )

        if self.config.use_shared_block_lora:
            self.linear_fc1_lora_A_list = nn.ParameterList([])
            self.linear_fc1_lora_B_list = nn.ParameterList([])

            for i in range(self.num_mem_blocks):
                linear_fc1_lora_A = nn.Linear(
                    self.config.hidden_size, self.config.lora_rank, bias=False
                )
                linear_fc1_lora_B = nn.Linear(
                    self.config.lora_rank, ffn_hidden_size_1, bias=False
                )
                self.linear_fc1_lora_A_list.append(linear_fc1_lora_A)
                self.linear_fc1_lora_B_list.append(linear_fc1_lora_B)

    def forward(self, hidden_states, inference_params=None, forward_layer_idx=None):
        if self.config.use_shared_block_lora:
            linear_fc1_lora_A = self.linear_fc1_lora_A_list[forward_layer_idx]
            linear_fc1_lora_B = self.linear_fc1_lora_B_list[forward_layer_idx]
            lora_output = linear_fc1_lora_A(hidden_states)
            lora_output = linear_fc1_lora_B(lora_output)
            intermediate_parallel = self.linear_fc1(hidden_states)
            intermediate_parallel = intermediate_parallel + lora_output
        else:
            intermediate_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_gelu_fusion:
            assert self.config.add_bias_linear is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel)
        else:
            intermediate_parallel = intermediate_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.linear_fc2(intermediate_parallel)
        return output

    def sharded_state_dict(
        self, prefix="", sharded_key_prefix=None, sharded_offsets=()
    ):
        sharded_key_prefix = (
            prefix if sharded_key_prefix is None else sharded_key_prefix
        )
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(
                prefix=f"{prefix}{name}.",
                sharded_key_prefix=f"{sharded_key_prefix}{name}.",
                sharded_offsets=sharded_offsets,
            )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict
