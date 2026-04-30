import torch
import torch.nn as nn
import torch.nn.functional as F

class CanonLayer(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 4):
        """
        Canon layer using a 1D causal convolution with residual connection.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Use causal convolution with explicit initialization
        self.causal_conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,  # Depthwise convolution
            padding=0,  # No automatic padding
            bias=True
        )

        # Initialize weights more conservatively
        nn.init.zeros_(self.causal_conv1d.weight)
        nn.init.zeros_(self.causal_conv1d.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Applies the Canon layer transformation with causal masking.
        """
        # Conv1d expects input shape (batch_size, channels, sequence_length)
        h_permuted = h.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)

        # Add padding of (kernel_size - 1) only to the left side
        padding = self.kernel_size - 1
        h_padded = F.pad(h_permuted, (padding, 0))

        # Apply causal convolution
        conv_out = self.causal_conv1d(h_padded)

        # Permute back to the original shape
        conv_out_permuted = conv_out.permute(0, 2, 1)

        # Add the residual connection
        output = h + conv_out_permuted

        return output


if __name__ == "__main__":
    # Normal test
    batch_size = 4
    sequence_length = 1024
    hidden_dim = 512

    input_tensor = torch.randn(batch_size, sequence_length, hidden_dim)
    canon_layer = CanonLayer(hidden_dim=hidden_dim, kernel_size=4)
    output_tensor = canon_layer(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # Causality test
    print("\n--- Causality Test ---")
    # Create a sequence where each position has a unique signal
    seq_len = 20
    test_dim = 5
    test_input = torch.zeros(1, seq_len, test_dim)

    # Set unique identifiers for each position
    for pos in range(seq_len):
        test_input[0, pos, :] = pos + 1

    # Create layer with small kernel
    test_layer = CanonLayer(hidden_dim=test_dim, kernel_size=3)

    # Process normally
    result = test_layer(test_input)

    # Now corrupt future tokens and see if output changes
    corrupt_input = test_input.clone()
    corrupt_pos = 10  # Choose a position to test
    corrupt_input[0, corrupt_pos:, :] = 99  # Corrupt all future tokens

    corrupt_result = test_layer(corrupt_input)

    # Check if positions before the corrupt point are identical
    is_causal = torch.allclose(result[0, :corrupt_pos, :],
                              corrupt_result[0, :corrupt_pos, :])

    print(f"Layer is causal: {is_causal}")
    if not is_causal:
        print("CAUSALITY VIOLATION DETECTED!")
        diff = (result[0, :corrupt_pos, :] - corrupt_result[0, :corrupt_pos, :]).abs()
        print(f"Max difference: {diff.max().item()}")