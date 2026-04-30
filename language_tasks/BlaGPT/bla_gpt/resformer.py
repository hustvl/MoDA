"""
ResFormer: Transformer with Value Residual Learning

Paper: Value Residual Learning (arXiv:2410.17897v5)
Authors: Zhou et al., 2025

Key innovation: Add residual connections from first layer's value vectors (V_1)
to all subsequent layers before attention, improving information flow and
reducing over-smoothing.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from coqpit import Coqpit

from bla_gpt import GPT, GPTConfig, Block, get_norm, get_mlp
from attentions import Attention


@dataclass
class ResFormerConfig(GPTConfig):
    """
    ResFormer configuration extending GPTConfig with value residual parameters.
    """

    # Value residual type
    value_residual_type: str = "identity"  # Options: "identity", "constant", "learnable", "learnable-plus", "sparse"

    # Lambda coefficients (initial values for learnable, fixed for constant/identity)
    value_residual_lambda_1: float = 0.5  # Weight for V_first (from layer 1)
    value_residual_lambda_2: float = 0.5  # Weight for V_current

    # Sparse ResFormer: only apply value residual to specific layers
    value_residual_start_layer: int = 2  # Start applying from this layer (1-indexed, default=2 means layers 2+)
    value_residual_end_layer: int = -1  # End layer (-1 means all layers)

    # Learnable-Plus: shared scale parameter initialization
    value_residual_scale_init: float = None  # Will default to n_layer if None


class ResFormerAttention(Attention):
    """
    Attention module with value residual learning.

    Extends standard Attention to support mixing current layer's value vectors
    with the first layer's value vectors before applying attention.

    Formula: V'_n = λ_1 * V_1 + λ_2 * V_n
    where V_1 is from the first layer, V_n is current layer.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx  # 0-indexed

        # Storage for first layer's V (only used by layer 0)
        self.last_v = None

        # Initialize lambda parameters based on residual type
        self._init_lambda_parameters(config)

    def _init_lambda_parameters(self, config):
        """Initialize lambda coefficients based on residual type."""
        residual_type = config.value_residual_type

        if residual_type == "identity":
            # Fixed 0.5/0.5 split
            self.register_buffer("lambda_1", torch.tensor(0.5))
            self.register_buffer("lambda_2", torch.tensor(0.5))

        elif residual_type == "constant":
            # Fixed custom values
            self.register_buffer("lambda_1", torch.tensor(config.value_residual_lambda_1))
            self.register_buffer("lambda_2", torch.tensor(config.value_residual_lambda_2))

        elif residual_type == "learnable":
            # Simple learnable parameters
            self.lambda_1 = nn.Parameter(torch.tensor(config.value_residual_lambda_1))
            self.lambda_2 = nn.Parameter(torch.tensor(config.value_residual_lambda_2))

        elif residual_type == "learnable-plus":
            # Learnable with softmax initialization (deeper layers get more V_1)
            # Lambda_1 uses softmax distribution across layers
            self.lambda_1_logit = nn.Parameter(torch.randn(1) * 0.1)  # Small random init
            self.lambda_2 = nn.Parameter(torch.tensor(config.value_residual_lambda_2))
            # Note: lambda_scale will be shared at model level, passed during forward

        elif residual_type == "sparse":
            # Sparse variant - some layers don't use residual
            if self._should_apply_residual(config):
                # Use constant or learnable for active layers
                lambda_1 = config.value_residual_lambda_1 if config.value_residual_lambda_1 != 0.5 else 5.0
                self.register_buffer("lambda_1", torch.tensor(lambda_1))
                self.register_buffer("lambda_2", torch.tensor(config.value_residual_lambda_2))
            else:
                # Inactive layers: no residual (lambda_1 = 0, lambda_2 = 1)
                self.register_buffer("lambda_1", torch.tensor(0.0))
                self.register_buffer("lambda_2", torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown value_residual_type: {residual_type}")

    def _should_apply_residual(self, config):
        """Check if this layer should apply value residual (for sparse variant)."""
        start = config.value_residual_start_layer - 1  # Convert to 0-indexed
        end = config.value_residual_end_layer - 1 if config.value_residual_end_layer > 0 else config.n_layer
        return start <= self.layer_idx < end

    def forward(self, x, q=None, mask=None, V_first=None, lambda_scale=None):
        """
        Forward pass with optional value residual.

        Args:
            x: Input tensor [B, T, C]
            q: Optional query input
            mask: Optional attention mask
            V_first: Value vectors from first layer [B, n_kv_head, T, head_dim] (or None for layer 0)
            lambda_scale: Shared scale parameter for learnable-plus variant
        """
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Update mask if provided
        if mask is not None:
            self.mask = mask

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply value residual if this is not the first layer and V_first is provided
        if self.layer_idx > 0 and V_first is not None:
            v = self._apply_value_residual(v, V_first, lambda_scale)

        # Store V if this is the first layer (layer_idx == 0)
        if self.layer_idx == 0:
            self.last_v = v.detach()  # Detach to avoid keeping full computation graph

        # Apply normalization and rotary embeddings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare attention inputs
        q, k, v = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0 and not self.use_softpick:
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, T_q, T)

        # Project output
        return self._project_output(y, B, T_q, C)

    def _apply_value_residual(self, v_current, v_first, lambda_scale=None):
        """
        Mix current layer's value with first layer's value.

        Args:
            v_current: Current layer values [B, T, n_kv_head, head_dim]
            v_first: First layer values [B, T, n_kv_head, head_dim]
            lambda_scale: Precomputed lambda coefficient for this layer (learnable-plus only)

        Returns:
            Mixed values [B, T, n_kv_head, head_dim]
        """
        if self.config.value_residual_type == "learnable-plus" and lambda_scale is not None:
            # Use the precomputed softmax-normalized coefficient
            # Computed at model level: softmax(logits) * lambda_scale
            lambda_1 = lambda_scale
        else:
            lambda_1 = self.lambda_1

        lambda_2 = self.lambda_2

        # Mix: V' = λ_1 * V_first + λ_2 * V_current
        # Both should have shape [B, T, n_kv_head, head_dim]
        v_mixed = lambda_1 * v_first + lambda_2 * v_current

        return v_mixed


class ResFormerBlock(Block):
    """
    Transformer block for ResFormer that passes V_first through attention.
    """

    def __init__(self, config, depth):
        # Override attention with ResFormerAttention before calling super
        super().__init__(config, depth)
        # Replace the attention module
        self.attn = ResFormerAttention(config, layer_idx=depth)

    def forward(self, x, idx=None, V_first=None, lambda_scale=None):
        """
        Forward pass with V_first propagation.

        Args:
            x: Input tensor [B, T, C]
            idx: Token indices (for compatibility)
            V_first: First layer's value vectors
            lambda_scale: Shared scale for learnable-plus
        """
        # Attention branch with V_first
        x_norm = self.ln_1(x)
        attn_out = self.attn(x_norm, V_first=V_first, lambda_scale=lambda_scale)

        if self.ln_3 is not None:
            attn_out = self.ln_3(attn_out)

        if self.res_w1 is not None:
            x = self.res_w1 * x + attn_out
        else:
            x = x + attn_out

        # MLP branch (unchanged)
        x = self._process_branch(x, self.ln_2, self.mlp, self.ln_4, self.res_w2)

        return x


class ResFormer(GPT):
    """
    ResFormer: GPT with Value Residual Learning.

    Extends the standard GPT model by:
    1. Using ResFormerAttention in all layers
    2. Caching first layer's value vectors (V_1)
    3. Passing V_1 to all subsequent layers for value mixing

    This improves information flow from initial token embeddings to deeper layers,
    reducing over-smoothing and attention concentration.
    """

    def __init__(self, config):
        # Call parent init to set up all standard GPT components
        super().__init__(config)

        # Replace standard blocks with ResFormer blocks
        blocks = [ResFormerBlock(config, d) for d in range(config.n_layer)]
        self.transformer.h = nn.ModuleList(blocks)

        # Initialize lambda_scale for learnable-plus variant
        if config.value_residual_type == "learnable-plus":
            scale_init = config.value_residual_scale_init
            if scale_init is None:
                scale_init = float(config.n_layer)
            self.lambda_scale = nn.Parameter(torch.tensor(scale_init))
            print(f"  ResFormer: Initialized lambda_scale to {scale_init}")
        else:
            self.lambda_scale = None

        # Print ResFormer-specific info
        print(f"  ResFormer: Using {config.value_residual_type} value residuals")
        if config.value_residual_type == "sparse":
            print(f"    Applied to layers {config.value_residual_start_layer} to {config.value_residual_end_layer if config.value_residual_end_layer > 0 else config.n_layer}")

    def _forward_blocks_with_value_residual(self, x, idx):
        """
        Forward through transformer blocks with value residual learning.

        This is the core ResFormer logic:
        1. First layer computes and caches V_1
        2. Subsequent layers receive V_1 and mix it with their own V_n
        3. For learnable-plus, compute lambda coefficients via softmax
        """
        # First layer: standard forward, no V_first
        first_block = self.transformer.h[0]
        x = first_block(x, idx, V_first=None, lambda_scale=None)

        # Capture V_first from first layer's attention
        V_first = first_block.attn.last_v

        # Compute lambda_scale for learnable-plus (softmax over all layer logits)
        if self.config.value_residual_type == "learnable-plus" and self.lambda_scale is not None:
            # Gather all lambda_1_logits from layers 1+ (layer 0 doesn't have one)
            logits = torch.stack([
                block.attn.lambda_1_logit.squeeze()
                for block in self.transformer.h[1:]
            ])
            # Apply softmax and scale
            lambda_coeffs = torch.softmax(logits, dim=0) * self.lambda_scale
        else:
            lambda_coeffs = None

        # Subsequent layers: pass V_first for value mixing
        for i, block in enumerate(self.transformer.h[1:], start=1):
            # Get this layer's lambda coefficient if using learnable-plus
            layer_lambda_scale = lambda_coeffs[i-1] if lambda_coeffs is not None else None
            x = block(x, idx, V_first=V_first, lambda_scale=layer_lambda_scale)

        return x

    def forward(self, idx, targets=None):
        """
        Forward pass with value residual learning.

        Only difference from GPT: Uses _forward_blocks_with_value_residual()
        instead of standard block loop. Everything else is inherited.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Standard GPT embedding logic
        tok_emb = self.transformer.wte(idx)
        if self.transformer.wpe is not None:
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # *** RESFORMER-SPECIFIC: Value residual block loop ***
        x = self._forward_blocks_with_value_residual(x, idx)

        # Continue with standard GPT logic
        x = self.transformer.ln_f(x)

        # Standard GPT loss computation (inherited logic, duplicated here for clarity)
        if targets is not None:
            if hasattr(self, "lm_head"):
                if self.config.use_per_token_output_bias:
                    x = x + self.output_bias_emb(idx)
                logits = self.lm_head(x)
                logits = logits.float()
                if self.soft_cap > 0.0:
                    from attentions import soft_cap
                    logits = soft_cap(logits, self.soft_cap)

                ntp_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )

                total_loss = ntp_loss
                loss_dict = {"ntp_loss": ntp_loss.detach().item()}

                # Add TOP loss if enabled
                from losses import compute_top_loss, compute_z_loss
                top_loss = compute_top_loss(self, x, idx, targets)
                if top_loss is not None:
                    total_loss += self.config.top_loss_weight * top_loss
                    loss_dict["top_loss"] = top_loss.detach().item()

                if self.config.z_loss_weight > 0.0:
                    z_loss = compute_z_loss(logits)
                    total_loss += self.config.z_loss_weight * z_loss
                    loss_dict["z_loss"] = z_loss.detach().item()

                if len(loss_dict) > 1:
                    loss_dict["total"] = total_loss
                    loss = loss_dict
                else:
                    loss = total_loss
            else:
                raise NotImplementedError("Multi-token prediction not yet supported in ResFormer")
        else:
            # Inference
            if self.config.use_per_token_output_bias:
                x = x + self.output_bias_emb(idx)
            logits = self.lm_head(x)
            logits = logits.float()
            if self.soft_cap > 0.0:
                from attentions import soft_cap
                logits = soft_cap(logits, self.soft_cap)
            loss = None

        return logits, loss


# =============================================================================
# Configuration Variants
# =============================================================================

from best_model_config import BestConfig


class ResFormerIdentityConfig(ResFormerConfig, BestConfig):
    """
    ResFormer with Identity value residuals (λ=0.5 fixed).
    Simplest variant - no hyperparameter tuning needed.
    """

    value_residual_type: str = "identity"  # Fixed λ_1 = λ_2 = 0.5
    value_residual_start_layer: int = 2  # Apply to all layers ≥ 2
    use_per_layer_token_emb: bool = True  # Not yet supported in ResFormerBlock


class ResFormerConstantConfig(ResFormerConfig, BestConfig):
    """
    ResFormer with Constant value residuals (λ_1=2.0, λ_2=1.0).
    Best performing fixed-coefficient variant from paper.
    """

    value_residual_type: str = "constant"
    value_residual_lambda_1: float = 2.0  # Empirically optimal
    value_residual_lambda_2: float = 1.0
    value_residual_start_layer: int = 2
    use_per_layer_token_emb: bool = True


class ResFormerLearnableConfig(ResFormerConfig, BestConfig):
    """
    ResFormer with Learnable value residuals.
    Lambda parameters learned during training.
    """

    value_residual_type: str = "learnable"
    value_residual_lambda_1: float = 0.5  # Initial value
    value_residual_lambda_2: float = 0.5  # Initial value
    value_residual_start_layer: int = 2
    use_per_layer_token_emb: bool = True


class ResFormerPlusConfig(ResFormerConfig, BestConfig):
    """
    ResFormer with Learnable-Plus value residuals.
    Best performing variant - uses softmax distribution across layers.
    Deeper layers automatically learn to use more V_1.

    Expected results (from paper):
    - ~2% better validation loss than vanilla Transformer
    - 16-20% more parameter efficient
    - 20% more data efficient
    """

    value_residual_type: str = "learnable-plus"
    value_residual_lambda_1: float = 0.5  # Initial value for logits
    value_residual_lambda_2: float = 0.5  # Initial value
    value_residual_start_layer: int = 2
    value_residual_scale_init: float = None  # Will default to n_layer
    use_per_layer_token_emb: bool = True


class ResFormerSparseConfig(ResFormerConfig, BestConfig):
    """
    ResFormer with Sparse value residuals.
    Only applies value residual to last 1/3 of layers.
    Slightly better performance with fewer parameters.
    """

    value_residual_type: str = "sparse"
    value_residual_lambda_1: float = 5.0  # Stronger residual for active layers
    value_residual_lambda_2: float = 1.0
    value_residual_start_layer: int = 9  # Last 1/3 of 12 layers (9, 10, 11, 12)
    value_residual_end_layer: int = -1  # To end
    use_per_layer_token_emb: bool = True
