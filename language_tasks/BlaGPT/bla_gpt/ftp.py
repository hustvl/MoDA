import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions import Rotary, apply_rotary_emb
from coqpit import Coqpit
from norms import RMSNorm
from utils import detach_loss


@dataclass
class FTPConfig(Coqpit):
    vocab_size: int = 50304  # GPT-2 vocabulary size
    dim: int = 768  # Embedding dimension
    hidden_dim: int = 768 * 3  # MLP hidden dimension (3x dim for SwiGLU)
    num_heads: int = 12  # Number of attention heads
    encoder_layers: int = 6  # Number of encoder layers
    decoder_layers: int = 6  # Number of decoder layers
    pseudo_seq_len: int = 12  # Length of pseudo-sequence
    max_seq_len: int = 1024  # Maximum sequence length
    dropout: float = 0.0  # Dropout probability
    future_tokens: int = 8  # Number of future tokens to predict
    gamma: float = 0.8  # Loss weighting factor for future tokens
    use_encoder_masking_loss: bool = False  # Use encoder output for masking loss
    use_chunked_encoder_masking: bool = True  # Use chunked attention for encoder

    def __post_init__(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        if self.hidden_dim is None:
            self.hidden_dim = 3 * self.dim  # Default SwiGLU expansion factor


class BaseAttention(nn.Module):
    def __init__(self, config: FTPConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = Rotary(self.head_dim)


class MultiHeadAttention(BaseAttention):
    def __init__(self, config: FTPConfig):
        super().__init__(config)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class SwiGLUMLP(nn.Module):
    def __init__(self, config: FTPConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        gate = self.w2(x)
        return self.dropout(self.w3(swish * gate))


class EncoderLayer(nn.Module):
    def __init__(self, config: FTPConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.dim)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttention(BaseAttention):
    def __init__(self, config: FTPConfig):
        super().__init__(config)
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, context):
        batch_size = x.size(0)
        seq_len = x.size(1)
        context_len = context.size(1)  # This is pseudo_seq_len

        # Project queries from decoder sequence
        q = self.q(x)  # [batch, seq_len, dim]

        # Project keys and values from context (pseudo-sequence)
        kv = self.kv(context)  # [batch, pseudo_seq_len, dim*2]
        k, v = kv.chunk(2, dim=-1)  # 2 x [batch, pseudo_seq_len, dim]

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, pseudo_seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, pseudo_seq_len, head_dim]

        # Apply rotary embeddings only to query
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)

        # Compute attention
        attn = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # [batch, heads, seq_len, pseudo_seq_len]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, heads, seq_len, head_dim]

        # Restore original dimensions
        out = out.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        out = out.view(batch_size, seq_len, self.dim)  # [batch, seq_len, dim]

        out = self.proj(out)
        out = self.dropout(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: FTPConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.dim)
        self.self_attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.dim)
        self.cross_attn = CrossAttention(config)
        self.ln3 = RMSNorm(config.dim)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, context, self_mask=None):
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, self_mask)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.cross_attn(x, context)
        x = residual + x

        residual = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = residual + x

        return x


class FTPModel(nn.Module):
    def __init__(self, config: FTPConfig):
        super().__init__()
        self.config = config

        # Shared embeddings
        self.mask_token = config.vocab_size  # Add MASK token at end of vocab
        self.effective_vocab_size = (
            config.vocab_size + 1
            if self.config.use_encoder_masking_loss
            else config.vocab_size
        )
        self.token_embedding = nn.Embedding(self.effective_vocab_size, config.dim)
        self.decoder_pos_embedding = nn.Embedding(config.future_tokens, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.start_token = nn.Parameter(torch.zeros(1, 1, config.dim))

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.encoder_to_decoder_proj = nn.Linear(
            config.dim, config.dim * self.config.future_tokens, bias=False
        )
        # self.encoder_to_decoder_proj = nn.GRU(
        #     input_size=config.dim,
        #     hidden_size=config.dim,
        #     num_layers=1,
        #     batch_first=True,
        # )

        # Pseudo-sequence projection
        self.pseudo_seq_proj = nn.Linear(
            config.dim, config.pseudo_seq_len * config.dim, bias=False
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )

        # Output projection (shared with embedding)
        self.final_norm = RMSNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight

        # Add encoder output projection for masking
        if config.use_encoder_masking_loss:
            self.encoder_output_norm = RMSNorm(config.dim)
            self.encoder_output_proj = nn.Linear(
                config.dim, self.effective_vocab_size, bias=False
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def get_causal_mask(self, seq_len, is_encoder=False, device=None):
        device = self.start_token.device

        # For encoder, use chunked causal masking
        if is_encoder:
            # Calculate number of chunks
            num_chunks = math.ceil(seq_len / self.config.future_tokens)

            # Create base mask of sequential chunks - note the transpose here
            chunk_mask = torch.ones(num_chunks, num_chunks, device=device)
            chunk_mask = torch.tril(
                chunk_mask
            ).bool()  # Lower triangular instead of upper

            # Expand to full sequence length
            chunk_mask = chunk_mask.repeat_interleave(self.config.future_tokens, dim=0)
            chunk_mask = chunk_mask.repeat_interleave(self.config.future_tokens, dim=1)

            # Within each chunk, allow full attention
            for i in range(num_chunks):
                start_idx = i * self.config.future_tokens
                end_idx = min(start_idx + self.config.future_tokens, seq_len)
                chunk_mask[start_idx:end_idx, start_idx:end_idx] = 1

            # Handle padding for incomplete last chunk
            if seq_len % self.config.future_tokens != 0:
                chunk_mask = chunk_mask[:seq_len, :seq_len]

            return chunk_mask

        # For decoder, use regular causal masking
        else:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            return ~mask

    def create_masked_input(self, input_ids, mask_prob=0.15):
        device = input_ids.device

        # Create masked input following BERT paper:
        # - 80% replace with [MASK]
        # - 10% replace with random token
        # - 10% keep original

        masked_input = input_ids.clone()
        labels = input_ids.clone()

        # Create masking mask
        probability_matrix = torch.full(input_ids.shape, mask_prob, device=device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of time, replace with [MASK] token
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool()
            & masked_indices
        )
        masked_input[indices_replaced] = self.mask_token

        # 10% of time, replace with random token
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.config.vocab_size,
            input_ids.shape,
            dtype=input_ids.dtype,
            device=device,
        )
        masked_input[indices_random] = random_words[indices_random]

        # Rest 10% of time, keep original tokens

        return masked_input, labels

    def forward(self, input_ids, targets):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        future_tokens = self.config.future_tokens

        # Create masked version of input for MLM task
        if self.training and self.config.use_encoder_masking_loss:
            masked_input, mlm_labels = self.create_masked_input(input_ids)

        # Get embeddings and positional encoding
        x = (
            self.token_embedding(masked_input)
            if self.training and self.config.use_encoder_masking_loss
            else self.token_embedding(input_ids)
        )
        x = self.dropout(x)

        # Encoder
        encoder_mask = self.get_causal_mask(
            seq_len, is_encoder=self.config.use_chunked_encoder_masking, device=device
        )
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)

        # Compute MLM loss
        mlm_loss = 0.0
        if self.training and self.config.use_encoder_masking_loss:
            mlm_logits = self.encoder_output_proj(self.encoder_output_norm(x))
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size + 1),  # +1 for MASK token
                mlm_labels.view(-1),
                ignore_index=-100,  # Ignore non-masked tokens
                reduction="mean",
            )

        # Project to pseudo-sequence
        # First future_tokens positions use repeated start token as context
        context_start = x[:, :1, :].repeat(1, future_tokens, 1)
        # context_start = x[:, :1, :].expand(-1, future_tokens, -1)

        # Remaining sequence is chunked into future_tokens groups
        # Only use previous tokens as context (shift by future_tokens)
        context_rest = x[:, :-future_tokens]  # Remove last future_tokens positions

        # Concatenate start context with rest of sequence
        context = torch.cat([context_start, context_rest], dim=1)

        # Reshape into chunks of future_tokens
        context = context.view(
            batch_size * (seq_len // future_tokens),
            future_tokens,
            self.config.dim,
        )

        # pseudo_seq = self.pseudo_seq_proj(x)
        # pseudo_seq = pseudo_seq.view(
        #     batch_size, seq_len, self.config.pseudo_seq_len, self.config.dim
        # )
        if targets is not None:
            # Ensure sequence length is divisible by future_tokens
            effective_seq_len = (seq_len // future_tokens) * future_tokens

            # Reshape inputs for parallel processing
            y = self.token_embedding(input_ids[:, :effective_seq_len])
            y = y.view(
                batch_size, -1, future_tokens, self.config.dim
            )  # Group by future_tokens
            y = self.dropout(y)

            # Add positional embeddings
            y = y + self.decoder_pos_embedding.weight.view(1, 1, future_tokens, -1)

            # Reshape for processing
            y = y.view(
                -1, future_tokens, self.config.dim
            )  # Combine batch and seq_len dimensions

            # Get corresponding context
            # context = pseudo_seq[:, :effective_seq_len:future_tokens].reshape(
            #     -1, self.config.pseudo_seq_len, self.config.dim
            # )

            decoder_mask = self.get_causal_mask(
                future_tokens, is_encoder=False, device=device
            )

            y = y + self.encoder_to_decoder_proj(context.mean(1, keepdim=True)).view(
                y.shape[0], future_tokens, -1
            )
            # Process through decoder
            for layer in self.decoder_layers:
                y = layer(
                    y,
                    context,
                    decoder_mask,
                )

            logits = self.output_proj(self.final_norm(y))

            # Compute weighted loss
            position_weights = (
                self.config.gamma ** torch.arange(future_tokens, device=device)
            ).view(1, -1)

            # Expand position weights to match batch dimension
            position_weights = position_weights.expand(
                batch_size * (seq_len // future_tokens), -1
            ).reshape(-1)

            # Reshape targets to match logits
            targets = targets[:, :effective_seq_len].view(-1)
            logits = logits.view(-1, self.config.vocab_size)

            # Compute loss
            loss = F.cross_entropy(logits, targets, reduction="none").mean()
            # weighted_loss = (loss * position_weights).mean()

            return logits, {
                "total": loss + mlm_loss,
                "ce": detach_loss(loss),
                "mlm": detach_loss(mlm_loss),
            }

        else:
            # Inference mode - only process last token
            y = self.token_embedding(input_ids[:, -1:])
            y = self.dropout(y)

            # Use last context directly
            curr_context = pseudo_seq[:, -1]  # [batch, pseudo_seq_len, dim]

            for layer in self.decoder_layers:
                y = layer(y, curr_context, None)

            return self.output_proj(y)



if __name__ == "__main__":
    import gc

    import numpy as np
    from torch.cuda import (max_memory_allocated, memory_allocated,
                            reset_peak_memory_stats)

    def get_size(bytes):
        """Convert bytes to human readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes < 1024:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024

    def print_memory_stats(prefix=""):
        print(f"\n{prefix} Memory Stats:")
        print(f"Current Memory Usage: {get_size(memory_allocated())}")
        print(f"Peak Memory Usage: {get_size(max_memory_allocated())}")

    # Clear any existing cached memory
    torch.cuda.empty_cache()
    gc.collect()
    reset_peak_memory_stats()

    config = FTPConfig(
        vocab_size=50257,
        dim=768,
        num_heads=12,
        encoder_layers=12,
        decoder_layers=3,
    )

    print("\nInitializing model...")
    model = FTPModel(config)
    model.cuda()
    print_memory_stats("After model initialization")

    batch_size = 8
    seq_len = 1024
    decoder_len = config.future_tokens

    print("\nGenerating input data...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len + 1))
    print_memory_stats("After data generation")

    print("\nRunning forward pass...")
    # Time the forward pass
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    outputs = model(x.cuda(), targets=y.cuda())
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)
    print(f"Forward pass time: {elapsed_time:.2f} ms")
    print_memory_stats("After forward pass")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Model dimensionality: {config.dim}")
    print(f"Number of encoder layers: {config.encoder_layers}")
    print(f"Number of decoder layers: {config.decoder_layers}")
