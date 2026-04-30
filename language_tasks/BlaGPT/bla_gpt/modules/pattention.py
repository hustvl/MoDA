import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pattention(nn.Module):
    """Pattention Layer."""

    def __init__(self, config):
        super().__init__()

        self.param_token_num = config.param_token_num
        self.param_key_dim = config.n_embd
        self.param_value_dim = config.n_embd
        self.norm_activation_type = "l2_norm_gelu"

        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim))
        )
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim))
        )

        gain = 3.0 / math.sqrt(self.param_key_dim)
        nn.init.normal_(self.key_param_tokens, std=gain)
        nn.init.normal_(self.value_param_tokens, std=gain)

    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == "softmax":
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = (
                nonlinear_outputs
                / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True)
                * inputs.shape[dim]
            )
            outputs = norm_outputs
        elif normalize_type == "gelu_l2_norm":
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = (
                nonlinear_outputs
                / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True)
                * math.sqrt(nonlinear_outputs.shape[dim])
            )
            outputs = norm_outputs
        elif normalize_type == "l2_norm_gelu":
            norm_outputs = (
                inputs
                / torch.norm(inputs, p=2, dim=dim, keepdim=True)
                * math.sqrt(inputs.shape[dim])
            )
            nonlinear_outputs = F.gelu(norm_outputs)
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        return outputs

    def forward(
        self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None
    ):
        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = (
                self.key_param_tokens[router_index],
                self.value_param_tokens[router_index],
            )

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale

        if self.norm_activation_type == "softmax":
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        else:
            attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if self.norm_activation_type == "softmax":
                    attn_bias.masked_fill_(attn_mask, float("-inf"))
                else:
                    attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if self.norm_activation_type == "softmax":
            attn_weight += attn_bias
        else:
            attn_weight *= attn_bias

        attn_weight = self.nonlinear_norm_func(
            attn_weight, self.norm_activation_type, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output
