import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions import Attention
from coqpit import Coqpit
from mlps import SwiGLU_MLP
from norms import RMSNorm


@dataclass
class HourglassConfig(Coqpit):
    """Configuration class for Hourglass Transformer model."""

    # Model architecture
    n_embd: int = 768
    n_layer: List[int] = field(
        default_factory=lambda: [3, 9, 3]
    )  # [pre_vanilla, middle, post_vanilla]
    shorten_factors: List[int] = field(
        default_factory=lambda: [4]
    )  # Shortening factors for each level
    tie_embed_weights: bool = True
    use_input_pos_embedding: bool = False

    # Attention settings
    n_head: int = 12
    n_kv_head: int = 12
    use_soft_logit_capping: bool = False
    bias: bool = False
    use_qkv_bias: bool = True
    rmsnorm_before_qk: bool = True
    pos_encoding: bool = "rotary"
    rope_theta: float = 10000.0

    # Vocabulary and sequence settings
    vocab_size: int = 50304
    block_size: int = 1024
    pad_token_id: int = -1

    # Regularization
    dropout: float = 0.0

    # Architecture choices
    shortening_method: str = "attention"  # Options: "linear", "avg", "attention"
    upsampling_method: str = "attention"  # Options: "attention", "linear", "repeat"

    # Training settings (optional but useful to keep with model config)
    learning_rate: float = 1e-3
    weight_decay: float = 0.00
    warmup_steps: int = 10000

    def __post_init__(self):
        if self.n_kv_head > self.n_head:
            self.n_kv_head = self.n_head


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)  # Pre-norm architecture as mentioned
        # self.attn = RelativeMultiHeadAttention(n_embd, n_head, dropout)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ff = SwiGLU_MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed))
        normed = self.norm2(x)
        x = x + self.dropout(self.ff(normed))
        return x


class Shortening(nn.Module):
    def __init__(self, config, shorten_factor: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.shorten_factor = shorten_factor
        self.method = config.shortening_method

        if config.shortening_method == "linear":
            self.proj = nn.Linear(shorten_factor * config.n_embd, config.n_embd)

        if config.shortening_method == "attention":
            self.proj = nn.Linear(shorten_factor * config.n_embd, config.n_embd)
            self.attn = Attention(config)

        # Store shift size to prevent information leaks
        self.shift_size = shorten_factor - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        assert seq_len % self.shorten_factor == 0

        # Apply right shift to prevent information leaks
        # Pad with zeros at the beginning and remove from the end
        x = F.pad(x, (0, 0, self.shift_size, -self.shift_size), value=0.0)

        if self.method == "avg":
            x = x.view(
                batch_size, seq_len // self.shorten_factor, self.shorten_factor, n_embd
            )
            return x.mean(dim=2)
        elif self.method == "linear":
            x = x.view(
                batch_size,
                seq_len // self.shorten_factor,
                self.shorten_factor * n_embd,
            )
            return self.proj(x)
        elif self.method == "attention":
            x_s = x.view(
                batch_size,
                seq_len // self.shorten_factor,
                self.shorten_factor * n_embd,
            )
            x_s = self.proj(x_s)
            x = x_s + self.attn(x, q=x_s)
            return x
        else:
            raise ValueError(f"Unknown shortening method: {self.method}")


# Update Upsampling class to handle masks
class Upsampling(nn.Module):
    def __init__(self, config, shorten_factor: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.shorten_factor = shorten_factor
        self.method = config.upsampling_method

        if self.method == "linear":
            self.proj = nn.Linear(config.n_embd, self.shorten_factor * config.n_embd)
        elif self.method == "attention":
            self.proj = nn.Linear(config.n_embd, self.shorten_factor * config.n_embd)
            self.attn = Attention(config)

    def forward(
        self, x: torch.Tensor, orig_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        if self.method == "repeat":
            return x.repeat_interleave(self.shorten_factor, dim=1)
        elif self.method == "linear":
            x = self.proj(x)
            return x.view(batch_size, seq_len * self.shorten_factor, n_embd)
        elif self.method == "attention":
            assert orig_x is not None

            x_up = self.proj(x)
            x_up = x_up.view(batch_size, seq_len * self.shorten_factor, n_embd)
            x_up = x_up + orig_x

            upsampled_len = seq_len * self.shorten_factor
            key_len = seq_len
            query_positions = torch.arange(upsampled_len, device=x.device)
            key_positions = torch.arange(key_len, device=x.device)
            source_positions = query_positions // self.shorten_factor
            causal_mask = key_positions.unsqueeze(0) <= source_positions.unsqueeze(1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            x_up = x_up + self.attn(x=x, q=x_up, mask=causal_mask)
            return x_up
        else:
            raise ValueError(f"Unknown upsampling method: {self.method}")


class HourglassTransformer(nn.Module):
    def __init__(self, config: HourglassConfig):
        super().__init__()

        self.config = config

        # Input embedding
        self.embedding = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=config.pad_token_id
        )
        self.pos_embedding = (
            nn.Parameter(torch.randn(1, config.block_size, config.n_embd))
            if config.use_input_pos_embedding
            else None
        )
        self.dropout = nn.Dropout(config.dropout)

        # Pre vanilla layers
        self.pre_vanilla = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layer[0])]
        )

        # Build recursive structure for each shortening level
        def build_recursive_layers(level: int) -> nn.ModuleList:
            if level == len(config.shorten_factors):
                return nn.ModuleList(
                    [TransformerLayer(config) for _ in range(config.n_layer[level + 1])]
                )

            layers = nn.ModuleList(
                [
                    Shortening(
                        config,
                        config.shorten_factors[level],
                    ),
                    *[
                        TransformerLayer(config)
                        for _ in range(config.n_layer[level + 1])
                    ],
                    build_recursive_layers(level + 1),
                    Upsampling(config, shorten_factor=config.shorten_factors[level]),
                ]
            )
            return layers

        self.hourglass_layers = build_recursive_layers(0)

        # Post vanilla layers
        self.post_vanilla = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layer[-1])]
        )

        # Output layer
        self.output = nn.Linear(config.n_embd, config.vocab_size)

        # Tie embedding weights to the output layer
        if config.tie_embed_weights:
            self.output.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward_recursive(
        self, layers: nn.ModuleList, x: torch.Tensor, mask: torch.Tensor, level: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(layers) > 0 and all(
            isinstance(layer, TransformerLayer) for layer in layers
        ):
            for layer in layers:
                x = layer(x, mask)
            return x, mask

        orig_x = x
        orig_mask = mask

        # Shortening
        if isinstance(layers[0], Shortening):
            x = layers[0](x)

        # Process transformer layers
        for layer in layers[1:-2]:
            if isinstance(layer, TransformerLayer):
                x = layer(x, mask)

        # Process recursive layers
        if len(layers) > 2:
            x, mask = self.forward_recursive(layers[-2], x, mask, level + 1)

        # Upsampling
        x = layers[-1](x, orig_x)
        mask = orig_mask

        return x, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        assert seq_len % self.config.shorten_factors[0] == 0, (
            "Sequence length must be divisible by the first shortening factor."
        )

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device), diagonal=1
        ).bool()

        # Input embedding with proper scaling
        x = self.embedding(input_ids) * math.sqrt(self.config.n_embd)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :seq_len]
        x = self.dropout(x)

        # Pre vanilla layers
        for layer in self.pre_vanilla:
            x = layer(x, causal_mask)

        # Store original sequence for residual connections
        orig_x = x.clone()

        # Process through hourglass structure
        x, _ = self.forward_recursive(self.hourglass_layers, x, causal_mask)

        # Add residual connection from pre-vanilla to post-hourglass
        x = x + orig_x

        # Post vanilla layers
        for layer in self.post_vanilla:
            x = layer(x, causal_mask)

        # Output projection
        logits = self.output(x)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Compute cross entropy loss
            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
        return logits, loss


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a small config for testing
    config = HourglassConfig(
        n_embd=256,
        n_head=4,
        n_layer=[2, 4, 2],
        shorten_factors=[2],
        vocab_size=1000,
        block_size=128,
        dropout=0.1,
    )

    print("\nModel configuration:")
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")

    # Create model
    print("\nInitializing model...")
    model = HourglassTransformer(config)
    model = model.to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create random input
    batch_size = 4
    seq_length = 64

    print(
        f"\nTesting with random input (batch_size={batch_size}, seq_length={seq_length})"
    )

    # Generate random input
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), device=device
    )

    labels = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), device=device
    )

    # Create attention mask (simulating padded sequences)
    attention_mask = torch.ones_like(input_ids, device=device)
    # Randomly mask some positions
    attention_mask[:, seq_length // 2 :] = torch.randint(
        0, 2, (batch_size, seq_length // 2), device=device
    )

    print("\nInput shapes:")
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"attention_mask: {attention_mask.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # Test without labels first
            print("Testing without labels...")
            logits, loss = model(input_ids=input_ids)
            print("Logits shape:", logits.shape)

            # Test with labels
            print("\nTesting with labels...")
            outputs = model(input_ids=input_ids, labels=labels)
            print("Logits shape:", logits.shape)
            print("Loss:", loss)

            # Test attention patterns
            print("\nTesting different sequence lengths...")
            for length in [32, 48, 64, 96]:
                test_input = torch.randint(
                    0, config.vocab_size, (2, length), device=device
                )

                logits, loss = model(input_ids=test_input)
                print(f"Sequence length {length}: logits shape {logits.shape}")

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise e

    # Test memory efficiency
    print("\nTesting memory usage...")
    torch.cuda.empty_cache()  # Clear cache before memory test

    try:
        torch.cuda.reset_peak_memory_stats()
        outputs = model(input_ids=input_ids, labels=labels)

        if torch.cuda.is_available():
            max_memory = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )  # Convert to MB
            print(f"Peak memory usage: {max_memory:.2f} MB")
    except Exception as e:
        print(f"Error during memory testing: {str(e)}")

    print("\nBasic gradient test...")
    try:
        model.train()
        logits, loss = model(input_ids=input_ids, labels=labels)
        print("Loss:", loss.item())
        loss.backward()

        # Check if gradients are flowing
        has_grad = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        print(f"Gradients present for all parameters: {has_grad}")

        # Print a few gradient statistics
        grad_norms = [
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        ]
        if grad_norms:
            print(f"Mean gradient norm: {np.mean(grad_norms):.6f}")
            print(f"Max gradient norm: {np.max(grad_norms):.6f}")
            print(f"Min gradient norm: {np.min(grad_norms):.6f}")

    except Exception as e:
        print(f"Error during gradient testing: {str(e)}")
        raise e

    print("\nAll tests completed!")
    print("\nAll tests completed!")
