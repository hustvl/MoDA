import torch
import transformer_engine as te
from torch import nn
from zamba2.enums import AttnMaskType
from zamba2.rotary import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    def __init__(
        self, config, layer_number, attn_mask_type=AttnMaskType.padding, **kwargs
    ):
        super().__init__()
        assert config.hidden_size % config.num_mem_heads == 0
        self.config = config
        self.linear_qkv = nn.Linear(
            2 * config.hidden_size, 6 * config.hidden_size, bias=config.add_bias_linear
        )
        self.linear_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=config.add_bias_linear
        )
        self.n_head = config.num_mem_heads
        self.n_embd = config.hidden_size * 2
        self.query_projection_size = (
            self.config.kv_channels * self.config.num_attention_heads
        )
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups
        world_size = 1
        self.world_size = world_size

        self.hidden_size_per_attention_head = (
            self.query_projection_size // self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = self.config.num_attention_heads
        self.num_query_groups_per_partition = self.config.num_query_groups
        self.dpa = te.pytorch.DotProductAttention(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=0.0,
            layer_number=layer_number,
            attn_mask_type="causal",
        )
        self.dpa_generation = te.pytorch.DotProductAttention(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=0.0,
            layer_number=layer_number,
            attn_mask_type="no_mask",
        )

        if self.config.use_shared_attention_lora:
            self.linear_q_lora_A_list = nn.ParameterList([])
            self.linear_q_lora_B_list = nn.ParameterList([])
            self.linear_k_lora_A_list = nn.ParameterList([])
            self.linear_k_lora_B_list = nn.ParameterList([])
            self.linear_v_lora_A_list = nn.ParameterList([])
            self.linear_v_lora_B_list = nn.ParameterList([])

            for i in range(self.num_mem_blocks):
                linear_q_lora_A = nn.Linear(
                    2 * self.config.hidden_size, self.config.lora_rank, bias=False
                )
                linear_q_lora_B = nn.Linear(
                    self.config.lora_rank, 2 * self.query_projection_size, bias=False
                )
                self.linear_q_lora_A_list.append(linear_q_lora_A)
                self.linear_q_lora_B_list.append(linear_q_lora_B)
                linear_k_lora_A = nn.Linear(
                    self.config.hidden_size, self.config.lora_rank, bias=False
                )
                linear_k_lora_B = nn.Linear(
                    self.config.lora_rank, 2 * self.kv_projection_size, bias=False
                )
                self.linear_k_lora_A_list.append(linear_k_lora_A)
                self.linear_k_lora_B_list.append(linear_k_lora_B)
                linear_v_lora_A = nn.Linear(
                    2 * self.config.hidden_size, self.config.lora_rank, bias=False
                )
                linear_v_lora_B = nn.Linear(
                    self.config.lora_rank, 2 * self.kv_projection_size, bias=False
                )
                self.linear_v_lora_A_list.append(linear_v_lora_A)
                self.linear_v_lora_B_list.append(linear_v_lora_B)

    def _allocate_memory(self, inference_max_sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head * 2,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(
        self, inference_params, key, value, rotary_pos_emb, layer_number
    ):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        if inference_params is None:
            return key, value, rotary_pos_emb

        is_first_step = False
        if layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = (
                inference_params.key_value_memory_dict[layer_number]
            )

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        inference_key_memory[
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = key

        inference_value_memory[
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]

        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if not is_first_step:
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        forward_layer_idx=None,
    ):
        qkv_out = self.linear_qkv(hidden_states)
        new_tensor_shape = qkv_out.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
                * 2
            ),
        )
        qkv_out = qkv_out.view(*new_tensor_shape)

        (query, key, value) = torch.split(
            qkv_out,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                    * 2
                ),
                self.hidden_size_per_attention_head * 2,
                self.hidden_size_per_attention_head * 2,
            ],
            dim=3,
        )

        if self.config.use_shared_attention_lora:
            new_lora_tensor_shape = new_tensor_shape[:-1] + (-1,)
            linear_q_lora_A = self.linear_q_lora_A_list[forward_layer_idx]
            linear_q_lora_B = self.linear_q_lora_B_list[forward_layer_idx]
            q_lora = linear_q_lora_A(hidden_states)
            q_lora = linear_q_lora_B(q_lora)
            query = query + q_lora.view(new_lora_tensor_shape)
            linear_k_lora_A = self.linear_k_lora_A_list[forward_layer_idx]
            linear_k_lora_B = self.linear_k_lora_B_list[forward_layer_idx]
            k_lora = linear_k_lora_A(hidden_states)
            k_lora = linear_k_lora_B(k_lora)
            key = key + k_lora.view(new_lora_tensor_shape)
            linear_v_lora_A = self.linear_v_lora_A_list[forward_layer_idx]
            linear_v_lora_B = self.linear_v_lora_B_list[forward_layer_idx]
            v_lora = linear_v_lora_A(hidden_states)
            v_lora = linear_v_lora_B(v_lora)
            value = value + v_lora.view(new_lora_tensor_shape)

        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head * 2
        )

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        key, value, rotary_pos_emb = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb, forward_layer_idx
        )

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if inference_params is None or inference_params.sequence_len_offset == 0:
            y = self.dpa(query, key, value)
        else:
            y = self.dpa_generation(query, key, value)

        y = self.linear_proj(y)

        return y
