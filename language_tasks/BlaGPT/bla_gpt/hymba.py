import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import (index_first_axis, pad_input,  # noqa
                                     unpad_input)
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa)
from transformers.modeling_outputs import (MoeCausalLMOutputWithPast,
                                           MoeModelOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (add_start_docstrings_to_model_forward,
                                is_flash_attn_greater_or_equal_2_10, logging,
                                replace_return_docstrings)
from transformers.utils.import_utils import is_torch_fx_available

_flash_supports_window_size = "window_size" in list(
    inspect.signature(flash_attn_func).parameters
)

from einops import pack, rearrange, repeat, unpack

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import (mamba_inner_fn,
                                                    selective_scan_fn)
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

is_fast_path_available = all(
    (
        selective_state_update,
        selective_scan_fn,
        causal_conv1d_fn,
        causal_conv1d_update,
        mamba_inner_fn,
    )
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HymbaConfig"


class HymbaConfig(PretrainedConfig):
    model_type = "hymba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        tie_word_embeddings=True,
        hidden_size=768,
        intermediate_size=768 * 3,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        calc_logits_for_entire_prompt=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=1024,
        max_position_embeddings=8192,
        orig_max_position_embeddings=2048,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_experts=1,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_inner_layernorms=True,
        kv_reuse_every_i_layer=-1,
        kv_reuse_group=[
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ],
        kv_weight_reuse=False,
        global_attn_idx=[0, 15, 31],
        num_mamba=1,
        attn_implementation_new="flex",
        rope_type="ntk",
        attn_hidden_size=-1,
        kq_head_dim=-1,
        v_head_dim=128,
        kq_norm="none",
        rope=True,
        rope_theta=10000.0,
        num_memory_tokens=128,
        memory_tokens_interspersed_every=0,
        attn_implementation="flex",
        mlp_hidden_act="silu",
        seq_length=1024,
        torch_dtype="bfloat16",
        layer_type=["h"] * 32,
        conv_dim={str(i): 3200 for i in range(32)},
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.orig_max_position_embeddings = orig_max_position_embeddings
        self.attention_dropout = attention_dropout

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(self.hidden_size / 16)
            if mamba_dt_rank == "auto"
            else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms

        self.attn_hidden_size = attn_hidden_size
        self.kq_head_dim = kq_head_dim
        self.v_head_dim = v_head_dim
        self.kq_norm = kq_norm
        self.rope = rope
        self.rope_theta = rope_theta
        self.num_memory_tokens = num_memory_tokens
        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        self.kv_reuse_every_i_layer = kv_reuse_every_i_layer
        self.kv_reuse_group = kv_reuse_group
        self.kv_weight_reuse = kv_weight_reuse

        self.global_attn_idx = global_attn_idx
        self.num_mamba = num_mamba
        self.attn_implementation_new = attn_implementation_new
        self.rope_type = rope_type
        self.mlp_hidden_act = mlp_hidden_act
        self.attn_implementation = attn_implementation
        self.seq_length = seq_length
        self.torch_dtype = torch_dtype
        self.layer_type = layer_type
        self.conv_dim = conv_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def pad_at_dim(t, pad: Tuple[int, int], dim=-1, value=0.0):
    if pad == (0, 0):
        return t

    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


# Adapted from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [
                layer_gate.to(compute_device)
                for layer_gate in gate_logits
                if layer_gate.shape[1] > 1
            ],
            dim=0,
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PerheadHymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, num_heads, eps=1e-6):
        """
        For per-head kq normalization
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_heads, 1, hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # assert 1==0, f"hiddens_states shape: {hidden_states.shape}" # [bsz, num_heads, seq_len, head_dim]
        assert (
            hidden_states.shape[1] == self.weight.shape[1]
        ), f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        assert (
            hidden_states.shape[3] == self.weight.shape[3]
        ), f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)

        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HymbaOnlyNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        self.config = config

        self.rope_type = config.rope_type

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if config.rope_type is None or config.rope_type == "default":
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                    / self.dim
                )
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            base = base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                    / self.dim
                )
            )

            self.max_seq_len_cached = orig_max_position_embeddings

        elif config.rope_type == "dynamic_ntk":
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                    / self.dim
                )
            )
            self.original_inv_freq = inv_freq
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """

        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            base = self.base * (
                (self.factor * seq_len / self.config.orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                    / self.dim
                )
            )

            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.config.orig_max_position_embeddings
            and self.max_seq_len_cached > self.config.orig_max_position_embeddings
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.rope_type == "dynamic_ntk":
            self._dynamic_frequency_update(position_ids, device=x.device)

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(
        self, config, batch_size, dtype=torch.float16, device=None, layer_type=None
    ):
        self.dtype = dtype
        # self.layers_block_type = config.layers_block_type
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []

        self.layer_type = layer_type

        for i in range(config.num_hidden_layers):
            if layer_type is None:
                has_mamba_state = True
            else:
                has_mamba_state = self.layer_type[i] == "h" or self.layer_type[i] == "m"

            if has_mamba_state:
                if hasattr(config, "conv_dim"):
                    conv_dim = config.conv_dim[i]
                else:
                    conv_dim = intermediate_size
                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        conv_dim,
                        conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size,
                        intermediate_size,
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]

        self.key_cache = [
            torch.tensor([[]] * batch_size, device=device)
            for _ in range(config.num_hidden_layers)
        ]
        self.value_cache = [
            torch.tensor([[]] * batch_size, device=device)
            for _ in range(config.num_hidden_layers)
        ]

        self.mamba_past_length = [0 for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(
                0, beam_idx.to(device)
            )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor

        if self.layer_type[layer_idx] == "m":
            return self.mamba_past_length[layer_idx]

        if self.key_cache[layer_idx].shape[-1] == 0:
            return 0

        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError(
            "HybridMambaAttentionDynamicCache does not have a legacy cache equivalent."
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        raise NotImplementedError(
            "HybridMambaAttentionDynamicCache does not have a legacy cache equivalent."
        )


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Hymba
class HymbaAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        config: HymbaConfig,
        layer_idx: Optional[int] = None,
        reuse_kv=False,
        output_hidden_size=None,
        attn_only_wo_proj=False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # self.hidden_size = config.hidden_size
        self.hidden_size = (
            config.attn_hidden_size
            if config.attn_hidden_size > 0
            else config.hidden_size
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.attn_only_wo_proj = attn_only_wo_proj

        self.kq_head_dim = (
            config.kq_head_dim if config.kq_head_dim > 0 else self.head_dim
        )
        self.v_head_dim = config.v_head_dim if config.v_head_dim > 0 else self.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if not self.attn_only_wo_proj:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.kq_head_dim, bias=False
            )

        self.reuse_kv = reuse_kv

        if not self.attn_only_wo_proj and not self.reuse_kv:
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.kq_head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False
            )

        if output_hidden_size is None:
            output_hidden_size = self.hidden_size

        if not self.attn_only_wo_proj:
            self.o_proj = nn.Linear(
                self.num_heads * self.v_head_dim, output_hidden_size, bias=False
            )

        if self.config.kq_norm == "rms":
            self.k_norm = HymbaRMSNorm(self.kq_head_dim)
            self.q_norm = HymbaRMSNorm(self.kq_head_dim)
        elif self.config.kq_norm == "perhead-rms":
            self.k_norm = PerheadHymbaRMSNorm(
                self.kq_head_dim, self.num_key_value_heads
            )
            self.q_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_heads)
        elif self.config.kq_norm == "none":
            self.k_norm = None
            self.q_norm = None
        else:
            raise NotImplementedError(f"Unknown kq_norm: {self.config.kq_norm}")

        if self.config.rope:
            self._init_rope()

    def set_rope(
        self, rope_type, orig_max_position_embeddings, max_position_embeddings
    ):
        self.config.rope_type = rope_type
        self.config.orig_max_position_embeddings = orig_max_position_embeddings
        self.config.max_position_embeddings = max_position_embeddings

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            dim=self.kq_head_dim,
            base=self.rope_theta,
            device=torch.device("cuda"),
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_last_layer=None,
        # kv_proj_last_layer = None,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError(
            "HymbaAttention is an abstract class. Use one of the subclasses."
        )


# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->Hymba
class HymbaFlashAttention2(HymbaAttention):
    """
    Hymba flash attention module. This module inherits from `HymbaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_last_layer=None,
        # kv_proj_last_layer = None,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        if self.attn_only_wo_proj:
            assert query_states is not None
            bsz, q_len, _ = query_states.size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if not self.attn_only_wo_proj:
            query_states = self.q_proj(hidden_states)

        query_states = (
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.config.rope:
            if self.attn_only_wo_proj:
                cos, sin = self.rotary_emb(query_states, position_ids)
            else:
                cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        if self.reuse_kv:
            assert kv_last_layer is not None
            key_states, value_states = (
                kv_last_layer  # (batch, num_heads, slen, head_dim)
            )

        else:
            if not self.attn_only_wo_proj:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.kq_head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.v_head_dim
            ).transpose(1, 2)

            if self.k_norm is not None:
                key_states = self.k_norm(key_states)

            if self.config.rope:
                _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not self.reuse_kv:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len
            > (
                self.config.sliding_window + self.config.num_memory_tokens
                if self.config.num_memory_tokens > 0
                else self.config.sliding_window
            )
            and use_swa
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        swa_processed_flag = False
        if past_key_value is not None and use_cache and not self.reuse_kv:
            kv_layer_idx = self.layer_idx

            cache_has_contents = past_key_value.get_seq_length(kv_layer_idx) > 0

            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len
                > (
                    self.config.sliding_window + self.config.num_memory_tokens
                    if self.config.num_memory_tokens > 0
                    else self.config.sliding_window
                )
                and cache_has_contents
                and use_swa
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[kv_layer_idx][0]
                past_value = past_key_value[kv_layer_idx][1]

                if self.config.num_memory_tokens > 0:
                    # num_fetched_memory_tokens = min(kv_seq_len - self.config.sliding_window, self.config.num_memory_tokens)
                    num_fetched_memory_tokens = self.config.num_memory_tokens

                    past_key = torch.cat(
                        [
                            past_key[:, :, :num_fetched_memory_tokens, :],
                            past_key[:, :, slicing_tokens:, :],
                        ],
                        dim=-2,
                    ).contiguous()
                    past_value = torch.cat(
                        [
                            past_value[:, :, :num_fetched_memory_tokens, :],
                            past_value[:, :, slicing_tokens:, :],
                        ],
                        dim=-2,
                    ).contiguous()

                else:
                    past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                    past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                past_key_value.key_cache[kv_layer_idx] = past_key
                past_key_value.value_cache[kv_layer_idx] = past_value

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

                swa_processed_flag = True

            key_states, value_states = past_key_value.update(
                key_states, value_states, kv_layer_idx
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(
            1, 2
        )  # (batch, slen, num_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        value_states = value_states.transpose(
            1, 2
        )  # (batch, slen, num_heads, head_dim)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows and not swa_processed_flag,
        )

        v_dim = value_states.shape[-2] * value_states.shape[-1]
        attn_output = attn_output.reshape(bsz, q_len, v_dim).contiguous()

        if self.attn_only_wo_proj:
            return attn_output, (key_states_no_repeat, value_states_no_repeat)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return (
            attn_output,
            attn_weights,
            past_key_value,
            (key_states_no_repeat, value_states_no_repeat),
        )

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if value_states.shape[-1] == query_states.shape[-1] * 2:
                value_states1 = value_states[..., : query_states.shape[-1]]

                batch_size = query_states.shape[0]

                (
                    query_states1,
                    key_states1,
                    value_states1,
                    indices_q,
                    cu_seq_lens,
                    max_seq_lens,
                ) = self._upad_input(
                    query_states,
                    key_states,
                    value_states1,
                    attention_mask,
                    query_length,
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad1 = flash_attn_varlen_func(
                        query_states1,
                        key_states1,
                        value_states1,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad1 = flash_attn_varlen_func(
                        query_states1,
                        key_states1,
                        value_states1,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

                attn_output1 = pad_input(
                    attn_output_unpad1, indices_q, batch_size, query_length
                )

                value_states2 = value_states[..., query_states.shape[-1] :]

                (
                    query_states2,
                    key_states2,
                    value_states2,
                    indices_q,
                    cu_seq_lens,
                    max_seq_lens,
                ) = self._upad_input(
                    query_states,
                    key_states,
                    value_states2,
                    attention_mask,
                    query_length,
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad2 = flash_attn_varlen_func(
                        query_states2,
                        key_states2,
                        value_states2,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad2 = flash_attn_varlen_func(
                        query_states2,
                        key_states2,
                        value_states2,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

                attn_output2 = pad_input(
                    attn_output_unpad2, indices_q, batch_size, query_length
                )

                attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

            else:
                batch_size = query_states.shape[0]
                (
                    query_states,
                    key_states,
                    value_states,
                    indices_q,
                    cu_seq_lens,
                    max_seq_lens,
                ) = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

                attn_output = pad_input(
                    attn_output_unpad, indices_q, batch_size, query_length
                )
        else:
            if value_states.shape[-1] == query_states.shape[-1] * 2:
                if not use_sliding_windows:
                    attn_output1 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[..., : query_states.shape[-1]],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )

                    attn_output2 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[..., query_states.shape[-1] :],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )

                    attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

                else:
                    attn_output1 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[..., : query_states.shape[-1]],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

                    attn_output2 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[..., query_states.shape[-1] :],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

                    attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

            else:
                if not use_sliding_windows:
                    attn_output = flash_attn_func(
                        query_states,
                        key_states,
                        value_states,
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output = flash_attn_func(
                        query_states,
                        key_states,
                        value_states,
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(
                            self.config.sliding_window,
                            self.config.sliding_window,
                        ),
                    )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Adapted from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Hymba
class HymbaSdpaAttention(HymbaAttention):
    """
    Hymba attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `HymbaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from HymbaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_last_layer=None,
        # kv_proj_last_layer = None,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        if self.attn_only_wo_proj:
            assert query_states is not None
            bsz, q_len, _ = query_states.size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if not self.attn_only_wo_proj:
            query_states = self.q_proj(hidden_states)

        query_states = (
            query_states.view(bsz, q_len, self.num_heads, self.kq_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.config.rope:
            if self.attn_only_wo_proj:
                cos, sin = self.rotary_emb(query_states, position_ids)
            else:
                cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        if self.reuse_kv:
            assert kv_last_layer is not None
            key_states, value_states = (
                kv_last_layer  # (batch, num_heads, slen, head_dim)
            )

        else:
            if not self.attn_only_wo_proj:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.kq_head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.v_head_dim
            ).transpose(1, 2)

            if self.k_norm is not None:
                key_states = self.k_norm(key_states)

            if self.config.rope:
                _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not self.reuse_kv and use_cache:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.v_head_dim * self.num_heads)

        if self.attn_only_wo_proj:
            return attn_output, (key_states_no_repeat, value_states_no_repeat)

        attn_output = self.o_proj(attn_output)

        return (
            attn_output,
            None,
            past_key_value,
            (key_states_no_repeat, value_states_no_repeat),
        )


# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->Hymba
class HymbaFlexAttention(HymbaFlashAttention2):
    """
    Hymba flash attention module. This module inherits from `HymbaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.config.num_memory_tokens > 0
        # assert self.config.sliding_window is not None

        from functools import partial

        from torch.nn.attention.flex_attention import (and_masks,
                                                       create_block_mask,
                                                       flex_attention,
                                                       or_masks)

        self.create_block_mask = create_block_mask

        def sliding_window(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= self.config.sliding_window

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        if (
            self.config.sliding_window is not None
            and self.config.global_attn_idx is not None
            and self.layer_idx not in self.config.global_attn_idx
        ):
            attn_mask = and_masks(causal_mask, sliding_window)
        else:
            attn_mask = causal_mask

        if self.config.memory_tokens_interspersed_every > 0:
            # !If see errors, note that deprecated n_ctx, using seq_length or max_position_embeddings instead
            num_memory_band = (
                self.config.seq_length // self.config.memory_tokens_interspersed_every
            )
            qk_length = (
                self.config.seq_length + num_memory_band * self.config.num_memory_tokens
            )
            num_tokens_per_band = qk_length // num_memory_band

            for i in range(num_memory_band):
                left_mask = (
                    lambda b, h, q_idx, kv_idx, i=i: kv_idx > i * num_tokens_per_band
                )
                right_mask = (
                    lambda b, h, q_idx, kv_idx, i=i: kv_idx
                    < i * num_tokens_per_band + self.config.num_memory_tokens
                )

                band_mask = and_masks(left_mask, right_mask)

                if i == 0:
                    prefix_mask_interspersed = band_mask
                else:
                    prefix_mask_interspersed = or_masks(
                        prefix_mask_interspersed, band_mask
                    )

            register_mask = and_masks(causal_mask, prefix_mask_interspersed)
        else:

            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx < self.config.num_memory_tokens

            register_mask = and_masks(causal_mask, prefix_mask)
            qk_length = self.config.seq_length + self.config.num_memory_tokens

        self.attn_mask = or_masks(attn_mask, register_mask)

        self.block_mask = create_block_mask(
            self.attn_mask,
            B=None,
            H=None,
            Q_LEN=qk_length,
            KV_LEN=qk_length,
            _compile=True,
        )

        self.flex_attention = torch.compile(flex_attention)

    def recompile_flexattn(self):
        from torch.nn.attention.flex_attention import flex_attention

        self.flex_attention = torch.compile(flex_attention)

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_last_layer=None,
        # kv_proj_last_layer = None,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            attention_mask = kwargs.pop("padding_mask")

        if self.attn_only_wo_proj:
            assert query_states is not None
            bsz, q_len, _ = query_states.size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if not self.attn_only_wo_proj:
            query_states = self.q_proj(hidden_states)

        query_states = (
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.config.rope:
            if self.attn_only_wo_proj:
                cos, sin = self.rotary_emb(query_states, position_ids)
            else:
                cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        if self.reuse_kv:
            assert kv_last_layer is not None
            key_states, value_states = (
                kv_last_layer  # (batch, num_heads, slen, head_dim)
            )

        else:
            if not self.attn_only_wo_proj:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.kq_head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.v_head_dim
            ).transpose(1, 2)

            if self.k_norm is not None:
                key_states = self.k_norm(key_states)

            if self.config.rope:
                # cos, sin = self.rotary_emb(hidden_states, position_ids)
                _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not self.reuse_kv:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len
            > (
                self.config.sliding_window + self.config.num_memory_tokens
                if self.config.num_memory_tokens > 0
                else self.config.sliding_window
            )
            and use_swa
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        swa_processed_flag = False
        if past_key_value is not None and use_cache and not self.reuse_kv:
            kv_layer_idx = self.layer_idx

            cache_has_contents = past_key_value.get_seq_length(kv_layer_idx) > 0

            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len
                > (
                    self.config.sliding_window + self.config.num_memory_tokens
                    if self.config.num_memory_tokens > 0
                    else self.config.sliding_window
                )
                and cache_has_contents
                and use_swa
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[kv_layer_idx][0]
                past_value = past_key_value[kv_layer_idx][1]

                if self.config.num_memory_tokens > 0:
                    # num_fetched_memory_tokens = min(kv_seq_len - self.config.sliding_window, self.config.num_memory_tokens)
                    num_fetched_memory_tokens = self.config.num_memory_tokens

                    past_key = torch.cat(
                        [
                            past_key[:, :, :num_fetched_memory_tokens, :],
                            past_key[:, :, slicing_tokens:, :],
                        ],
                        dim=-2,
                    ).contiguous()
                    past_value = torch.cat(
                        [
                            past_value[:, :, :num_fetched_memory_tokens, :],
                            past_value[:, :, slicing_tokens:, :],
                        ],
                        dim=-2,
                    ).contiguous()

                else:
                    past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                    past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                ### only keep sliding_window tokens in kv cache: Removed as this will impact the kv_seq_len calculation, resulting in errors for all swa cases
                past_key_value.key_cache[kv_layer_idx] = past_key
                past_key_value.value_cache[kv_layer_idx] = past_value

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

                swa_processed_flag = True

            key_states, value_states = past_key_value.update(
                key_states, value_states, kv_layer_idx
            )

            # print(key_states.shape, value_states.shape)
        else:
            cache_has_contents = False

        # repeat k/v heads if n_kv_heads < n_heads
        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                # target_dtype = self.q_proj.weight.dtype
                target_dtype = query_states.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if (
            past_key_value is not None
            and use_cache
            and (not use_swa or query_states.shape[-2] <= self.config.sliding_window)
        ):
            query_states = query_states.transpose(
                1, 2
            )  # (batch, slen, num_heads, head_dim)
            key_states = key_states.transpose(
                1, 2
            )  # (batch, slen, num_heads, head_dim)
            value_states = value_states.transpose(
                1, 2
            )  # (batch, slen, num_heads, head_dim)

            attn_output = self._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                use_sliding_windows=use_sliding_windows and not swa_processed_flag,
            )

            v_dim = value_states.shape[-2] * value_states.shape[-1]
            attn_output = attn_output.reshape(bsz, q_len, v_dim).contiguous()

        else:
            if (
                key_states.shape[-2] <= self.block_mask.shape[-2] - 128
                or key_states.shape[-2] > self.block_mask.shape[-2]
            ):
                block_mask = self.create_block_mask(
                    self.attn_mask,
                    B=None,
                    H=None,
                    Q_LEN=key_states.shape[-2],
                    KV_LEN=key_states.shape[-2],
                )  # , _compile=True)
            else:
                block_mask = self.block_mask

            if value_states.shape[-1] == query_states.shape[-1] * 2:
                attn_output1 = self.flex_attention(
                    query_states,
                    key_states,
                    value_states[..., : query_states.shape[-1]],
                    block_mask=block_mask,
                )
                attn_output2 = self.flex_attention(
                    query_states,
                    key_states,
                    value_states[..., query_states.shape[-1] :],
                    block_mask=block_mask,
                )

                attn_output = torch.cat([attn_output1, attn_output2], dim=-1)
            else:
                attn_output = self.flex_attention(
                    query_states, key_states, value_states, block_mask=block_mask
                )

            attn_output = attn_output.transpose(
                1, 2
            ).contiguous()  ## [batch_size, seq_length, num_head, v_head_dim]

            if hasattr(self, "head_mask") and self.head_mask is not None:
                head_mask = self.head_mask.to(attn_output)
                head_mask = head_mask.view(1, 1, -1, 1)
                attn_output = attn_output * head_mask

            attn_output = attn_output.reshape(
                bsz, q_len, self.v_head_dim * self.num_heads
            )

        if self.attn_only_wo_proj:
            return attn_output, (key_states_no_repeat, value_states_no_repeat)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return (
            attn_output,
            attn_weights,
            past_key_value,
            (key_states_no_repeat, value_states_no_repeat),
        )

    def set_head_mask(self, mask):
        self.head_mask = mask


JAMBA_ATTENTION_CLASSES = {
    "eager": HymbaAttention,
    "flash_attention_2": HymbaFlashAttention2,
    "sdpa": HymbaSdpaAttention,  ## the default attention
    "flex": HymbaFlexAttention,
}


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class HymbaBlock(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: HymbaConfig, layer_idx, reuse_kv=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv

        self.intermediate_size = int(config.mamba_expand * config.hidden_size)

        self.reuse_kv = reuse_kv

        self.attn_hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        config.v_head_dim = self.intermediate_size // self.num_attention_heads

        self.k_hidden_size = int(
            self.num_key_value_heads / self.num_attention_heads * self.attn_hidden_size
        )
        self.v_hidden_size = int(
            self.num_key_value_heads
            / self.num_attention_heads
            * self.attn_hidden_size
            * config.mamba_expand
        )

        self.self_attn = JAMBA_ATTENTION_CLASSES[config.attn_implementation](
            config, layer_idx, attn_only_wo_proj=True, reuse_kv=reuse_kv
        )

        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        self.use_fast_kernels = True  # config.use_mamba_kernels

        if self.reuse_kv:
            self.latent_dim = (
                self.intermediate_size + self.attn_hidden_size
            )  ## mamba plus q
        else:
            self.latent_dim = (
                self.intermediate_size
                + self.attn_hidden_size
                + self.k_hidden_size
                + self.v_hidden_size
            )  ## mamba plus qkv

        self.pre_avg_layernorm1 = HymbaRMSNorm(
            self.intermediate_size, eps=config.rms_norm_eps
        )
        self.pre_avg_layernorm2 = HymbaRMSNorm(
            self.intermediate_size, eps=config.rms_norm_eps
        )

        self.in_proj = nn.Linear(
            self.hidden_size,
            self.latent_dim + self.intermediate_size,
            bias=self.use_bias,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=self.use_bias
        )

        num_ssm_param = 1

        if not hasattr(config, "conv_dim"):
            config.conv_dim = {i: 0 for i in range(config.num_hidden_layers)}

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        config.conv_dim[self.layer_idx] = self.intermediate_size

        self.x_proj = nn.ModuleList(
            [
                nn.Linear(
                    self.intermediate_size,
                    self.time_step_rank + self.ssm_state_size * 2,
                    bias=False,
                )
                for _ in range(num_ssm_param)
            ]
        )
        self.dt_proj = nn.ModuleList(
            [
                nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
                for _ in range(num_ssm_param)
            ]
        )

        # A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.ParameterList(
            [nn.Parameter(torch.log(A)) for _ in range(num_ssm_param)]
        )

        self.D = nn.ParameterList(
            [
                nn.Parameter(torch.ones(self.intermediate_size))
                for _ in range(num_ssm_param)
            ]
        )

        if self.apply_inner_layernorms:
            self.dt_layernorm = HymbaRMSNorm(
                self.time_step_rank, eps=config.rms_norm_eps
            )
            self.B_layernorm = HymbaRMSNorm(
                self.ssm_state_size, eps=config.rms_norm_eps
            )
            self.C_layernorm = HymbaRMSNorm(
                self.ssm_state_size, eps=config.rms_norm_eps
            )

        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def set_attn_mamba_mask(self, attn_branch_mask, mamba_branch_mask):
        self.attn_branch_mask = attn_branch_mask
        self.mamba_branch_mask = mamba_branch_mask

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            input_dtype = dt.dtype
            dt = self.dt_layernorm(dt).to(input_dtype)
        if self.B_layernorm is not None:
            input_dtype = B.dtype
            B = self.B_layernorm(B).to(input_dtype)
        if self.C_layernorm is not None:
            input_dtype = C.dtype
            C = self.C_layernorm(C).to(input_dtype)
        return dt, B, C

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache = None,
        attention_mask=None,
        position_ids=None,
        kv_last_layer=None,
        use_cache=False,
        use_swa=False,
    ):
        projected_states = self.in_proj(hidden_states).transpose(
            1, 2
        )  ## (bs, latent_dim, seq_len)

        if (
            self.training and cache_params is None and not self.apply_inner_layernorms
        ):  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            batch_size, seq_len, _ = hidden_states.shape
            use_precomputed_states = (
                cache_params is not None
                and cache_params.has_previous_state
                and seq_len == 1
                and cache_params.conv_states[self.layer_idx].shape[0]
                == cache_params.ssm_states[self.layer_idx].shape[0]
                == batch_size
                and use_cache
            )

            hidden_states, gate = projected_states.tensor_split(
                (self.latent_dim,), dim=1
            )

            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )

            if self.reuse_kv:
                query_states, hidden_states = hidden_states.tensor_split(
                    (self.attn_hidden_size,), dim=1
                )
                query_states = query_states.transpose(1, 2)
            else:
                query_states, key_states, value_states, hidden_states = (
                    hidden_states.tensor_split(
                        (
                            self.attn_hidden_size,
                            self.attn_hidden_size + self.k_hidden_size,
                            self.attn_hidden_size
                            + self.k_hidden_size
                            + self.v_hidden_size,
                        ),
                        dim=1,
                    )
                )

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

            if use_precomputed_states:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)

                cache_params.mamba_past_length[self.layer_idx] += seq_len
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states,
                        (self.conv_kernel_size - hidden_states.shape[-1], 0),
                    )

                    cache_params.conv_states[self.layer_idx].copy_(conv_states)

                    cache_params.mamba_past_length[self.layer_idx] += seq_len

                hidden_states = causal_conv1d_fn(
                    hidden_states,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                )

            if self.reuse_kv:
                assert kv_last_layer is not None
                attn_outputs, attn_key_value = self.self_attn(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    query_states=query_states,
                    kv_last_layer=kv_last_layer,
                    use_swa=use_swa,
                    use_cache=use_cache,
                    past_key_value=cache_params,
                )
            else:
                attn_outputs, attn_key_value = self.self_attn(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    use_swa=use_swa,
                    use_cache=use_cache,
                    past_key_value=cache_params,
                )

            ## Mamba head
            index = 0
            ssm_parameters = self.x_proj[index](hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters,
                [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
                dim=-1,
            )
            time_step, B, C = self._apply_layernorms(time_step, B, C)

            if hasattr(self.dt_proj[index], "base_layer"):
                time_proj_bias = self.dt_proj[index].base_layer.bias
                self.dt_proj[index].base_layer.bias = None
            else:
                time_proj_bias = self.dt_proj[index].bias
                self.dt_proj[index].bias = None
            discrete_time_step = self.dt_proj[index](time_step).transpose(
                1, 2
            )  # [batch, intermediate_size, seq_len]

            if hasattr(self.dt_proj[index], "base_layer"):
                self.dt_proj[index].base_layer.bias = time_proj_bias
            else:
                self.dt_proj[index].bias = time_proj_bias

            A = -torch.exp(self.A_log[index].float())

            time_proj_bias = (
                time_proj_bias.float() if time_proj_bias is not None else None
            )
            if use_precomputed_states:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D[index],
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                outputs = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D[index].float(),
                    z=gate,
                    delta_bias=time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )

                if len(outputs) == 3:
                    scan_outputs, ssm_state, _ = outputs
                else:
                    scan_outputs, ssm_state = outputs

                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            scan_outputs = scan_outputs.transpose(1, 2)

            hidden_states = (
                self.pre_avg_layernorm1(attn_outputs)
                + self.pre_avg_layernorm2(scan_outputs)
            ) / 2
            contextualized_states = self.out_proj(hidden_states)

        return contextualized_states, attn_key_value

    def mixer_forward(
        self,
        hidden_states,
        cache_params: HybridMambaAttentionDynamicCache = None,
        attention_mask=None,
        position_ids=None,
        kv_last_layer=None,
        use_cache=False,
        use_swa=False,
    ):
        if self.use_fast_kernels:
            if (
                not is_fast_path_available
                or "cuda" not in self.x_proj[0].weight.device.type
            ):
                # if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device"
                )
            return self.cuda_kernels_forward(
                hidden_states,
                cache_params,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_last_layer=kv_last_layer,
                use_cache=use_cache,
                use_swa=use_swa,
            )
        else:
            raise ValueError("Support Mamba kernel only")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        res, attn_key_value = self.mixer_forward(
            hidden_states,
            cache_params=past_key_value,
            attention_mask=kwargs["attention_mask"],
            kv_last_layer=kwargs["kv_last_layer"],
            position_ids=kwargs["position_ids"],
            use_cache=kwargs["use_cache"],
            use_swa=kwargs["use_swa"],
        )

        return res, attn_key_value, past_key_value


class HymbaMLP(nn.Module):
    def __init__(self, config: HymbaConfig):
        super().__init__()
        # self.config = config
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        if self.act_fn_name == "silu":
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            return self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock with Mistral->Hymba
class HymbaSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: HymbaConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size

        #   these values are decided on runtime depending on the layer index
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            # expert routing
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList(
            [HymbaMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        if len(hidden_states.shape) == 3:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            bs_times_seq_len = batch_size * sequence_length
        elif len(hidden_states.shape) == 2:
            assert self.num_experts == 1
            bs_times_seq_len, hidden_dim = hidden_states.shape
        else:
            batch_size, sequence_length, _, hidden_dim = hidden_states.shape
            bs_times_seq_len = batch_size * sequence_length

        if self.num_experts == 1:
            # in this case we have a single MLP block and don't need to do any routing
            final_hidden_states = self.experts[0](hidden_states)
            router_logits = torch.ones(
                (bs_times_seq_len, 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                requires_grad=hidden_states.requires_grad,
            )
            return final_hidden_states, router_logits

        # in this case we have multiple experts and need to do routing
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class HymbaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: HymbaConfig,
        num_experts: int,
        layer_idx: int,
        reuse_kv: bool = False,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.reuse_kv = reuse_kv

        self.mamba = HymbaBlock(config=config, layer_idx=layer_idx, reuse_kv=reuse_kv)

        self.input_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1

            self.moe = HymbaSparseMoeBlock(
                config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok
            )

            self.pre_moe_layernorm = HymbaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_raw: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        kv_last_layer=None,
        use_swa=False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_key_value, present_key_value = self.mamba(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_last_layer=kv_last_layer,
            use_cache=use_cache,
            use_swa=use_swa,
        )

        bs, seqlen, _ = hidden_states.shape
        past_seqlen = self._get_past_seqlen(past_key_value, seqlen)
        num_attention_heads = self.mamba.config.num_attention_heads
        self_attn_weights = torch.empty(
            bs, num_attention_heads, seqlen, past_seqlen, device="meta"
        )

        # residual connection after mamba
        hidden_states = residual + hidden_states

        if self.intermediate_size > 0:
            residual = hidden_states
            hidden_states = self.pre_moe_layernorm(hidden_states)
            hidden_states, router_logits = self.moe(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        outputs += (attn_key_value,)

        return outputs

    def _get_past_seqlen(self, past_key_value, seqlen):
        if past_key_value is None:
            return seqlen
        past_seqlen = past_key_value.get_seq_length()

        if past_seqlen == 0:
            return seqlen

        return past_seqlen


class HymbaPreTrainedModel(PreTrainedModel):
    config_class = HymbaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HymbaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. have the seqlen as the third dim
        also for mamba layers
        """
        attn_layer_index = [k.shape == v.shape for k, v in past_key_value].index(True)
        seqlen = past_key_value[attn_layer_index][0].shape[2]
        standard_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                # expand doesn't use more memory, so it's fine to do it here
                standard_past_key_value += (
                    (k.expand(-1, -1, seqlen, -1), v.expand(-1, -1, seqlen, -1)),
                )
            else:
                standard_past_key_value += ((k, v),)
        return standard_past_key_value

    @staticmethod
    def _convert_to_hymba_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Hymba, i.e. dummy seqlen dimesion with size 1 for mamba layers
        """
        hymba_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                hymba_past_key_value += ((k[:, :, :1, :], v[:, :, :1, :]),)
            else:
                hymba_past_key_value += ((k, v),)
        return hymba_past_key_value


HYMBA_INPUTS_DOCSTRING = r"""
    Args: To be added later. Please refer to the forward function.
"""


# Adapted from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->JAMBA, Mistral->Hymba
class HymbaModel(HymbaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HymbaDecoderLayer`]

    Args:
        config: HymbaConfig
    """

    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        config.attn_implementation = config.attn_implementation_new
        config._attn_implementation = config.attn_implementation_new

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.inter_layer_kv_reuse = (
            config.kv_reuse_every_i_layer > 0 or config.kv_reuse_group is not None
        )
        self.kv_reuse_group = config.kv_reuse_group
        self.kv_reuse_every_i_layer = config.kv_reuse_every_i_layer

        decoder_layers = []

        if self.kv_reuse_group is not None:
            self.kv_reuse_group = [
                {"producer": group[0], "consumer": group[1:]}
                for group in self.kv_reuse_group
            ]

        layer_type = []
        for i in range(config.num_hidden_layers):
            if self.inter_layer_kv_reuse:
                if self.kv_reuse_group is not None:
                    reuse_kv = False
                    for group_id, item in enumerate(self.kv_reuse_group):
                        if i in item["consumer"]:
                            reuse_kv = True

                else:
                    if i % config.kv_reuse_every_i_layer == 0:
                        reuse_kv = False
                    else:
                        reuse_kv = True
            else:
                reuse_kv = False

            layer_type.append("h")
            decoder_layer = HymbaDecoderLayer(
                config, num_experts=1, layer_idx=i, reuse_kv=reuse_kv
            )

            decoder_layers.append(decoder_layer)

        config.layer_type = layer_type

        if config.sliding_window is not None:
            self.sliding_window = config.sliding_window
            self.global_attn_idx = config.global_attn_idx
        else:
            self.sliding_window = None
            self.global_attn_idx = None

        self._attn_layer_index = []
        self._hymba_layer_index = [
            isinstance(layer, HymbaDecoderLayer) for layer in decoder_layers
        ].index(True)

        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config.attn_implementation
        self.final_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(self.config.num_memory_tokens, self.config.hidden_size)
            )
        self.gradient_checkpointing = False

        self.post_init()

    # Ignore copy
    @add_start_docstrings_to_model_forward(HYMBA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[List[torch.FloatTensor], HybridMambaAttentionDynamicCache]
        ] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = False
            # past_key_values_length = past_key_values.get_usable_length(seq_length, self._attn_layer_index)
            if past_key_values is not None:
                past_key_values_length = past_key_values.get_usable_length(
                    seq_length, 0
                )
            else:
                use_cache = False

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if (
                self.config.num_memory_tokens > 0
                and past_key_values is not None
                and past_key_values.get_seq_length() == 0
            ):
                position_ids = position_ids.view(
                    -1, seq_length + self.config.num_memory_tokens
                ).long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.num_memory_tokens > 0 and (
            past_key_values is None or past_key_values.get_seq_length() == 0
        ):
            ori_b, ori_n = inputs_embeds.shape[0], inputs_embeds.shape[1]

            if self.config.memory_tokens_interspersed_every > 0:
                mem_every = self.config.memory_tokens_interspersed_every
                next_seq_len = math.ceil(ori_n / mem_every) * mem_every

                # print(f"before padding: {inputs_embeds.shape}")
                inputs_embeds = pad_at_dim(
                    inputs_embeds, (0, next_seq_len - ori_n), dim=-2, value=0.0
                )
                # print(f"after padding: {inputs_embeds.shape}")
                inputs_embeds = rearrange(
                    inputs_embeds, "b (n m) d -> (b n) m d", m=mem_every
                )  # m is the segment length

            mem = repeat(
                self.memory_tokens, "n d -> b n d", b=inputs_embeds.shape[0]
            )  # prepend the memory to every segment of m by repeating the memory tokens
            inputs_embeds, mem_packed_shape = pack((mem, inputs_embeds), "b * d")

            if self.config.memory_tokens_interspersed_every > 0:
                inputs_embeds = rearrange(
                    inputs_embeds, "(b n) m d -> b (n m) d", b=ori_b
                )

            if (
                position_ids is not None
                and position_ids.shape[1] != inputs_embeds.shape[1]
            ):
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)

        attention_mask_raw = attention_mask

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Hymba. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if (
            self._attn_implementation == "flash_attention_2"
            or self._attn_implementation == "flex"
        ):
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
            attention_mask_swa = attention_mask

        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask_input = attention_mask

            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

            if self.sliding_window is not None:
                attention_mask_swa = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask_input,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.sliding_window,
                )

        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

            if self.sliding_window is not None:
                attention_mask_swa = _prepare_4d_causal_attention_mask(
                    attention_mask_input,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.sliding_window,
                )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        kv_last_layer = None

        shared_kv_cache_dict = {}

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.inter_layer_kv_reuse and self.kv_reuse_group is not None:
                no_reuse_flag = True
                for group_id, item in enumerate(self.kv_reuse_group):
                    if i in item["consumer"]:
                        kv_last_layer = shared_kv_cache_dict[group_id]
                        no_reuse_flag = False
                        # print(f'[Layer-{i}]: Reuse KV cache from Layer-{self.kv_reuse_group[group_id]["producer"]}')
                        break

                if no_reuse_flag:
                    kv_last_layer = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask
                    if (self.sliding_window is None or i in self.global_attn_idx)
                    else attention_mask_swa,
                    attention_mask_raw,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    kv_last_layer,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask
                    if (self.sliding_window is None or i in self.global_attn_idx)
                    else attention_mask_swa,
                    attention_mask_raw=attention_mask_raw,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    kv_last_layer=kv_last_layer if self.inter_layer_kv_reuse else None,
                    use_swa=self.sliding_window is not None
                    and i not in self.global_attn_idx,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[3],)

            if self.inter_layer_kv_reuse:
                kv_last_layer = layer_outputs[-1]

                if self.kv_reuse_group is not None:
                    for group_id, item in enumerate(self.kv_reuse_group):
                        if i == item["producer"]:
                            shared_kv_cache_dict[group_id] = kv_last_layer
                            break

        del shared_kv_cache_dict

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.config.num_memory_tokens > 0 and (
            past_key_values is None or past_key_values.get_seq_length() == 0
        ):
            if self.config.memory_tokens_interspersed_every > 0:
                hidden_states = rearrange(
                    hidden_states,
                    "b (n m) d -> (b n) m d",
                    m=(
                        self.config.num_memory_tokens
                        + self.config.memory_tokens_interspersed_every
                    ),
                )

            mem, hidden_states = unpack(hidden_states, mem_packed_shape, "b * d")

            if self.config.memory_tokens_interspersed_every > 0:
                hidden_states = rearrange(
                    hidden_states, "(b n) m d -> b (n m) d", b=ori_b
                )

            hidden_states = hidden_states[:, :ori_n, :]

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM with MIXTRAL->JAMBA, Mixtral->Hymba
class HymbaForCausalLM(HymbaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        self.config = config
        self.model = HymbaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(HYMBA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        calc_logits_for_entire_prompt: Optional[bool] = True,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            calc_logits_for_entire_prompt (`bool`, *optional*):
                Whether or not to calculate the logits for the entire prompt, or just the last token. Only last token
                logits are needed for generation, and calculating them only for that token can save memory,
                which becomes pretty significant for long sequences.

        Returns:
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -1:, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        # print("hidden_states.shape:", hidden_states.shape, "input_ids.shape:", input_ids.shape, "logits.shape:", logits.shape)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs,
    ):
        if self.config.num_memory_tokens > 0:
            attention_mask = torch.cat(
                [
                    torch.ones(
                        input_ids.shape[0],
                        self.config.num_memory_tokens,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )

        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if isinstance(past_key_values, Tuple):
                if past_key_values[self.model._hymba_layer_index][0].shape[2] > 1:
                    past_key_values = self._convert_to_hymba_cache(past_key_values)

            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()

                past_length = cache_length

            else:
                cache_length = past_length = past_key_values[
                    self.model._attn_layer_index
                ][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif (
                self.config.num_memory_tokens <= 0 and past_length < input_ids.shape[1]
            ):
                input_ids = input_ids[:, past_length:]

            elif (
                self.config.num_memory_tokens > 0
                and past_length < input_ids.shape[1] + self.config.num_memory_tokens
            ):
                new_query_id = past_length - self.config.num_memory_tokens
                input_ids = input_ids[:, new_query_id:]

            if self.config.sliding_window is not None and (
                self.config.global_attn_idx is None
                or len(self.config.global_attn_idx) == 0
            ):
                input_ids = input_ids[:, -1:]

            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config,
                input_ids.shape[0],
                self.dtype,
                device=self.device,
                layer_type=self.config.layer_type,
            )

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values.get_seq_length() > 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "calc_logits_for_entire_prompt": self.config.calc_logits_for_entire_prompt,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class HymbaForBlaGPT(HymbaForCausalLM):
    def __init__(self, config: HymbaConfig):
        super().__init__(config)

    def forward(self, idx, targets):
        output = super().forward(input_ids=idx, labels=targets)
        return output["logits"], output["loss"]


if __name__ == "__main__":
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        config = HymbaConfig(kq_norm="none")
        model = HymbaForBlaGPT(config)
        model.cuda()
        idx = torch.randint(0, 50256, (2, 1024)).cuda()
        targets = torch.randint(0, 50256, (2, 1024)).cuda()
        logits, loss = model.forward(idx, targets)
        print(loss)
