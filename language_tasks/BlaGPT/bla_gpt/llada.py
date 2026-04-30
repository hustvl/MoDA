import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
from coqpit import Coqpit


@dataclass
class LLaDAConfig(Coqpit):
    """Configuration class for LLaDA (Large Language Diffusion with mAsking) model"""

    # override train config parameters
    compile_model: bool = True
    num_iterations: int = 5100 * 100

    learning_rate: float = 1e-4
    optimizer_name: str = (
        "AdamW"  # check get_optimizer() in bla_gpt/optimizers/__init__.py
    )
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
    )

    # Model architecture parameters
    vocab_size: int = 50304  # Vocabulary size
    dim: int = 768  # Model dimension
    n_layers: int = 12  # Number of transformer layers
    n_heads: int = 12  # Number of attention heads
    ffn_dim: int = 768 * 4  # Feed-forward network dimension
    max_seq_len: int = 1024  # Maximum sequence length

    # LLaDA-specific parameters
    mask_token_id: Optional[int] = None  # Mask token ID (defaults to vocab_size if None)
    eos_token_id: Optional[int] = None  # EOS token ID (defaults to vocab_size + 1 if None)

    # Diffusion parameters
    num_diffusion_steps: int = 64  # Number of diffusion steps for generation
    min_time_step: float = 1e-6  # Minimum time step to avoid t=0

    # Generation parameters
    generation_temperature: float = 1.0  # Default temperature for generation

    # Training parameters
    dropout: float = 0.0  # Dropout rate

    # Positional encoding
    rope_theta: float = 10000.0  # RoPE theta parameter
    rope_full_precision: bool = True  # Use full precision for RoPE computations

    def __post_init__(self):
        # Set mask_token_id to vocab_size if not specified
        if self.mask_token_id is None:
            self.mask_token_id = self.vocab_size

        # Validate parameters
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")

        if self.num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")

        if not (0.0 < self.min_time_step < 1.0):
            raise ValueError("min_time_step must be between 0 and 1")


class RMSNorm(nn.Module):
    """RMS Normalization as used in LLaMA"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        return self.weight * x


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 4096, rope_full_precision: bool = True):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_full_precision = rope_full_precision

        # Precompute frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

        # Type annotations for registered buffers
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len):
        if self.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            cos = self.cos_cached[:seq_len, :]
            sin = self.sin_cached[:seq_len, :]

            # Ensure cos and sin have the same dtype as q_ and k_
            cos = cos.to(q_.dtype)
            sin = sin.to(q_.dtype)

            q_embed = (q_ * cos) + (self.rotate_half(q_) * sin)
            k_embed = (k_ * cos) + (self.rotate_half(k_) * sin)

        return q_embed.to(q.dtype), k_embed.to(k.dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head attention without causal masking (bidirectional)"""
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 4096, rope_full_precision: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        assert self.head_dim * n_heads == dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_full_precision)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )

        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block without causal masking"""
    def __init__(self, dim: int, n_heads: int, ffn_dim: int, max_seq_len: int = 4096, rope_full_precision: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len, rope_full_precision)
        self.feed_forward = SwiGLU(dim, ffn_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        x = x + self.attention(self.attention_norm(x), attention_mask)

        # Feed-forward with residual connection
        x = x + self.feed_forward(self.ffn_norm(x))

        return x


class MaskPredictor(nn.Module):
    """The core mask predictor Transformer for LLaDA"""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        ffn_dim: int = 12288,
        max_seq_len: int = 4096,
        mask_token_id: int = None,
        rope_full_precision: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id or vocab_size  # Use vocab_size as default mask token

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size + 1, dim)  # +1 for mask token

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_dim, max_seq_len, rope_full_precision)
            for _ in range(n_layers)
        ])

        # Final norm and output projection
        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the mask predictor

        Args:
            input_ids: Token IDs, potentially with mask tokens (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, 1, 1, seq_len) or None

        Returns:
            Logits for predicting original tokens (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.token_embedding(input_ids)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits


class LLaDA(nn.Module):
    """Large Language Diffusion with mAsking (LLaDA)"""
    def __init__(
        self,
        config: LLaDAConfig
    ):
        super().__init__()

        # Use config values if provided, otherwise use individual parameters
        self.config = config
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id if config.mask_token_id is not None else config.vocab_size
        self.eos_token_id = config.eos_token_id if config.eos_token_id is not None else (config.vocab_size + 1)
        self.max_seq_len = config.max_seq_len

        # Initialize tiktoken tokenizer for text generation
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # The core mask predictor
        self.mask_predictor = MaskPredictor(
            vocab_size=config.vocab_size,
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
            mask_token_id=self.mask_token_id,
            rope_full_precision=config.rope_full_precision
        )

    @classmethod
    def from_config(cls, config: LLaDAConfig):
        """Create LLaDA model from config"""
        return cls(config=config)

    def apply_forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward diffusion process to mask tokens

        Args:
            x0: Original sequence (batch_size, seq_len)
            t: Time step (batch_size,) - values in [0, 1]

        Returns:
            xt: Masked sequence (batch_size, seq_len)
        """
        batch_size, seq_len = x0.shape

        # Sample masking pattern
        # Each token is masked with probability t
        mask_probs = t.unsqueeze(1).expand(batch_size, seq_len)  # (batch_size, seq_len)
        should_mask = torch.rand_like(mask_probs) < mask_probs

        # Create masked sequence
        xt = x0.clone()
        xt[should_mask] = self.mask_token_id

        return xt, should_mask

    def apply_forward_process_sft(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward process for SFT - only mask response tokens, not prompt

        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            response_ids: Response token IDs (batch_size, response_len)
            t: Time step (batch_size,)

        Returns:
            masked_sequence: Full sequence with masked responses (batch_size, total_len)
            mask_positions: Boolean mask for masked positions (batch_size, total_len)
        """
        batch_size, prompt_len = prompt_ids.shape
        _, response_len = response_ids.shape
        total_len = prompt_len + response_len

        # Concatenate prompt and response
        full_sequence = torch.cat([prompt_ids, response_ids], dim=1)

        # Only mask response tokens
        response_mask_probs = t.unsqueeze(1).expand(batch_size, response_len)
        should_mask_response = torch.rand_like(response_mask_probs) < response_mask_probs

        # Create mask for full sequence (False for prompt, computed for response)
        mask_positions = torch.zeros(batch_size, total_len, dtype=torch.bool, device=full_sequence.device)
        mask_positions[:, prompt_len:] = should_mask_response

        # Apply masking
        masked_sequence = full_sequence.clone()
        masked_sequence[mask_positions] = self.mask_token_id

        return masked_sequence, mask_positions

    def compute_loss(self, x0: torch.Tensor, xt: torch.Tensor, mask_positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss for LLaDA

        Args:
            x0: Original sequence (batch_size, seq_len)
            xt: Masked sequence (batch_size, seq_len)
            mask_positions: Boolean tensor indicating masked positions (batch_size, seq_len)
            t: Time step (batch_size,)

        Returns:
            loss: Training loss
        """
        # Get predictions from mask predictor
        logits = self.mask_predictor(xt)  # (batch_size, seq_len, vocab_size)

        # Compute cross-entropy loss only on masked tokens
        batch_size, seq_len = x0.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = x0.view(-1)  # (batch_size * seq_len)
        mask_flat = mask_positions.view(-1)  # (batch_size * seq_len)

        # Compute cross-entropy loss
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (batch_size * seq_len)

        # Apply mask - only compute loss on masked tokens
        masked_loss = loss_per_token * mask_flat.float()

        # Average over masked tokens, weighted by 1/t as in the paper
        # Loss = -E[1/t * sum(masked_tokens) * log p(x0|xt)]
        t_expanded = t.unsqueeze(1).expand(batch_size, seq_len).contiguous().view(-1)  # (batch_size * seq_len)

        # Avoid division by zero
        t_safe = torch.clamp(t_expanded, min=1e-8)

        # Weight by 1/t and sum over masked positions
        weighted_loss = masked_loss / t_safe

        # Average over batch and sequence - but only count masked tokens
        num_masked_tokens = mask_flat.sum()
        if num_masked_tokens > 0:
            loss = weighted_loss.sum() / num_masked_tokens
        else:
            loss = torch.tensor(0.0, device=x0.device, requires_grad=True)

        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        response_ids: Optional[torch.Tensor] = None,
        is_sft: bool = False,
        use_variable_length: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass for training

        Args:
            input_ids: Input token IDs (batch_size, seq_len) - for pre-training
            targets: Target token IDs (batch_size, seq_len) - same as input_ids for LLaDA
            prompt_ids: Prompt IDs for SFT (batch_size, prompt_len)
            response_ids: Response IDs for SFT (batch_size, response_len)
            is_sft: Whether this is supervised fine-tuning
            use_variable_length: Whether to use variable length training (1% of time)

        Returns:
            Dictionary containing loss and other metrics
        """
        if is_sft:
            # SFT mode: only mask response tokens
            assert prompt_ids is not None and response_ids is not None
            batch_size = prompt_ids.shape[0]

            # Sample time steps
            t = torch.rand(batch_size, device=prompt_ids.device) * (1.0 - 1e-6) + 1e-6

            # Apply SFT forward process
            masked_sequence, mask_positions = self.apply_forward_process_sft(prompt_ids, response_ids, t)

            # Compute loss on response tokens only
            full_targets = torch.cat([prompt_ids, response_ids], dim=1)
            loss = self.compute_loss(full_targets, masked_sequence, mask_positions, t)

            return {
                'masked_sequence': masked_sequence,
                'mask_positions': mask_positions,
                'time_steps': t,
                'prompt_ids': prompt_ids,
                'response_ids': response_ids
            }, loss
        else:
            # Pre-training mode
            if targets is None:
                targets = input_ids

            # ❗FIXIT: workaround for the current training script
            targets = input_ids

            x0 = targets  # Original clean sequence
            batch_size, seq_len = x0.shape

            # Variable length training (1% of the time as in paper)
            if use_variable_length and torch.rand(1).item() < 0.01:
                # Randomly truncate sequence length
                new_seq_len = torch.randint(1, seq_len + 1, (1,)).item()
                x0 = x0[:, :new_seq_len]

            # Sample time steps uniformly from [0, 1]
            t = torch.rand(batch_size, device=x0.device) * (1.0 - 1e-6) + 1e-6

            # Apply forward diffusion process
            xt, mask_positions = self.apply_forward_process(x0, t)

            # Compute loss
            loss = self.compute_loss(x0, xt, mask_positions, t)

            return {
                'masked_sequence': xt,
                'mask_positions': mask_positions,
                'time_steps': t
            }, loss

    def validate(self, input_ids, targets):
        with torch.no_grad():
            # ❗FIXIT
            loss = self.evaluate_cross_entropy(input_ids)
        return None, loss

    def evaluate_likelihood(
        self,
        input_ids: torch.Tensor,
        num_monte_carlo: int = 128,
        return_perplexity: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate likelihood using the stable method from Algorithm 3 in the paper.
        This gives likelihood estimates comparable to autoregressive models.

        Args:
            input_ids: Input sequences (batch_size, seq_len)
            num_monte_carlo: Number of Monte Carlo samples for stable estimation
            return_perplexity: Whether to return perplexity in addition to log likelihood

        Returns:
            Dictionary with log_likelihood and optionally perplexity
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        total_log_likelihood = torch.zeros(batch_size, device=device)

        self.eval()
        with torch.no_grad():
            for _ in range(num_monte_carlo):
                # Uniformly sample number of tokens to mask from {1, 2, ..., L}
                num_mask = torch.randint(1, seq_len + 1, (batch_size,), device=device)

                # Create masked sequences by uniformly sampling l tokens for masking
                masked_sequences = input_ids.clone()

                for i in range(batch_size):
                    num_to_mask = num_mask[i].item()
                    # Uniformly sample positions to mask without replacement
                    positions_to_mask = torch.randperm(seq_len, device=device)[:num_to_mask]
                    masked_sequences[i, positions_to_mask] = self.mask_token_id

                # Get predictions
                logits = self.mask_predictor(masked_sequences)  # (batch_size, seq_len, vocab_size)

                # Compute log probabilities for the original tokens
                log_probs = F.log_softmax(logits, dim=-1)

                # Extract log probabilities for the original tokens
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
                token_log_probs = log_probs[batch_indices, seq_indices, input_ids]

                # Apply mask and weight by L/l
                for i in range(batch_size):
                    num_masked = num_mask[i].item()
                    # Find which positions were masked
                    mask_positions = (masked_sequences[i] == self.mask_token_id)

                    # Sum log probabilities of masked tokens, weighted by L/l
                    masked_log_probs = token_log_probs[i, mask_positions].sum()
                    total_log_likelihood[i] += (seq_len / num_masked) * masked_log_probs

        # Average over Monte Carlo samples
        log_likelihood = total_log_likelihood / num_monte_carlo

        results = {'log_likelihood': log_likelihood}

        if return_perplexity:
            # Perplexity = exp(-log_likelihood / num_tokens)
            perplexity = torch.exp(-log_likelihood / seq_len)
            results['perplexity'] = perplexity

        return results

    def evaluate_cross_entropy(
        self,
        input_ids: torch.Tensor,
        num_monte_carlo: int = 128
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss comparable to autoregressive models.
        This is the negative log-likelihood normalized by sequence length.

        Args:
            input_ids: Input sequences (batch_size, seq_len)
            num_monte_carlo: Number of Monte Carlo samples

        Returns:
            Cross-entropy loss (scalar)
        """
        likelihood_results = self.evaluate_likelihood(
            input_ids,
            num_monte_carlo=num_monte_carlo,
            return_perplexity=False
        )

        # Cross-entropy is negative log-likelihood normalized by sequence length
        seq_len = input_ids.shape[1]
        cross_entropy = -likelihood_results['log_likelihood'] / seq_len

        return cross_entropy.mean()  # Average over batch

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Convert token IDs to text using tiktoken tokenizer

        Args:
            tokens: Token IDs tensor (1D)

        Returns:
            Decoded text string
        """
        # Filter out mask tokens and convert to list
        valid_tokens = [t for t in tokens.cpu().numpy() if t != self.mask_token_id and t < self.vocab_size]
        if len(valid_tokens) == 0:
            return ""

        try:
            return self.tokenizer.decode(valid_tokens)
        except Exception as e:
            # Fallback to token visualization if decoding fails
            return " ".join([f"<{t}>" for t in valid_tokens])

    def text_to_tokens(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert text to token IDs using tiktoken tokenizer

        Args:
            text: Input text string
            device: Device to place tensor on (uses model device if None)

        Returns:
            Token IDs tensor (1D)
        """
        if device is None:
            device = next(self.parameters()).device

        try:
            tokens = self.tokenizer.encode(text)
            return torch.tensor(tokens, dtype=torch.long, device=device)
        except Exception:
            # Fallback for encoding errors
            print(f"Warning: Failed to encode text '{text}'")
            return torch.tensor([], dtype=torch.long, device=device)

    def generate_from_text(
        self,
        prompt_text: str = "",
        max_length: int = 512,
        num_steps: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate text from a text prompt using the reverse diffusion process

        Args:
            prompt_text: Input text prompt (empty string for unconditional generation)
            max_length: Maximum generation length
            num_steps: Number of diffusion steps (uses config default if None)
            temperature: Sampling temperature (uses config default if None)

        Returns:
            Generated text string
        """
        # Convert text prompt to tokens
        if prompt_text:
            prompt_tokens = self.text_to_tokens(prompt_text)
            if len(prompt_tokens) > 0:
                prompt_ids = prompt_tokens.unsqueeze(0)  # Add batch dimension
            else:
                prompt_ids = None
        else:
            prompt_ids = None

        # Generate text using the existing generate method
        return self.generate(
            prompt_ids=prompt_ids,
            max_length=max_length,
            num_steps=num_steps or 256,
            temperature=temperature or 1.0,
            return_text=True,
            **kwargs
        )

    def generate(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_steps: int = 256,
        temperature: float = 1.0,
        remasking_strategy: str = "random",  # "random", "low_confidence", "semi_autoregressive"
        block_length: int = 32,  # For semi-autoregressive
        cfg_weight: float = 0.0,  # Classifier-free guidance weight
        top_p: float = 1.0,  # Nucleus sampling
        eos_token_id: Optional[int] = None,
        return_text: bool = False
    ) -> Union[torch.Tensor, str, list]:
        """
        Generate text using the reverse diffusion process

        Args:
            prompt_ids: Optional prompt token IDs (batch_size, prompt_len)
            max_length: Maximum generation length
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            remasking_strategy: Strategy for remasking ("random", "low_confidence", "semi_autoregressive")
            block_length: Block length for semi-autoregressive sampling
            cfg_weight: Classifier-free guidance weight (0 = no guidance)
            top_p: Nucleus sampling parameter
            eos_token_id: EOS token ID for early stopping
            return_text: If True, return decoded text instead of token IDs

        Returns:
            Generated token IDs (batch_size, total_length) or decoded text if return_text=True
        """
        device = next(self.parameters()).device

        if prompt_ids is not None:
            batch_size, prompt_len = prompt_ids.shape
            total_len = prompt_len + max_length
        else:
            batch_size = 1
            prompt_len = 0
            total_len = max_length

        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        # Start with fully masked sequence
        sequence = torch.full((batch_size, total_len), self.mask_token_id, device=device)

        # Fill in prompt if provided
        if prompt_ids is not None:
            sequence[:, :prompt_len] = prompt_ids

        # Semi-autoregressive: generate block by block
        if remasking_strategy == "semi_autoregressive":
            sequence = self._generate_semi_autoregressive(
                sequence, prompt_len, num_steps, temperature, block_length, cfg_weight, top_p, eos_token_id
            )
        else:
            # Standard diffusion sampling with different remasking strategies
            for step in range(num_steps):
                t = 1.0 - step / num_steps
                s = 1.0 - (step + 1) / num_steps

                if t <= 0:
                    break

                with torch.no_grad():
                    # Get predictions with optional classifier-free guidance
                    if cfg_weight > 0 and prompt_ids is not None:
                        logits = self._apply_classifier_free_guidance(sequence, prompt_ids, cfg_weight)
                    else:
                        logits = self.mask_predictor(sequence)

                    # Sample predictions
                    predictions = self._sample_tokens(logits, temperature, top_p)

                    # Keep unmasked tokens, update predictions for masked tokens
                    mask = (sequence == self.mask_token_id)
                    sequence = torch.where(mask, predictions, sequence)

                    # Check for early stopping (EOS tokens)
                    if eos_token_id is not None:
                        eos_positions = (sequence == eos_token_id)
                        if eos_positions.any():
                            # Stop generation after EOS in each sequence
                            for b in range(batch_size):
                                eos_idx = torch.where(eos_positions[b])[0]
                                if len(eos_idx) > 0:
                                    first_eos = eos_idx[0].item()
                                    if first_eos < total_len - 1:
                                        sequence[b, first_eos + 1:] = eos_token_id

                    # Remask for next step
                    if s > 0 and step < num_steps - 1:
                        sequence = self._apply_remasking(
                            sequence, logits, t, s, prompt_len, remasking_strategy
                        )

        if return_text:
            # Convert to text using tiktoken
            if sequence.shape[0] == 1:
                return self.tokens_to_text(sequence[0])
            else:
                return [self.tokens_to_text(seq) for seq in sequence]

        return sequence

    def _generate_semi_autoregressive(
        self,
        sequence: torch.Tensor,
        prompt_len: int,
        num_steps: int,
        temperature: float,
        block_length: int,
        cfg_weight: float,
        top_p: float,
        eos_token_id: int
    ) -> torch.Tensor:
        """Generate using semi-autoregressive strategy (block by block from left to right)"""
        batch_size, total_len = sequence.shape
        gen_len = total_len - prompt_len

        # Generate each block from left to right
        for block_start in range(prompt_len, total_len, block_length):
            block_end = min(block_start + block_length, total_len)

            # Mask the current block
            sequence[:, block_start:block_end] = self.mask_token_id

            # Run diffusion process for this block
            for step in range(num_steps):
                t = 1.0 - step / num_steps
                s = 1.0 - (step + 1) / num_steps

                if t <= 0:
                    break

                with torch.no_grad():
                    # Get predictions
                    if cfg_weight > 0 and prompt_len > 0:
                        logits = self._apply_classifier_free_guidance(
                            sequence, sequence[:, :prompt_len], cfg_weight
                        )
                    else:
                        logits = self.mask_predictor(sequence)

                    # Sample predictions for current block only
                    block_logits = logits[:, block_start:block_end, :]
                    block_predictions = self._sample_tokens(block_logits, temperature, top_p)

                    # Update current block
                    block_mask = (sequence[:, block_start:block_end] == self.mask_token_id)
                    sequence[:, block_start:block_end] = torch.where(
                        block_mask, block_predictions, sequence[:, block_start:block_end]
                    )

                    # Remask within block using low-confidence strategy
                    if s > 0 and step < num_steps - 1:
                        remask_ratio = s / t
                        sequence = self._apply_low_confidence_remasking_block(
                            sequence, logits, block_start, block_end, remask_ratio
                        )

        return sequence

    def _apply_classifier_free_guidance(
        self,
        sequence: torch.Tensor,
        prompt_ids: torch.Tensor,
        cfg_weight: float
    ) -> torch.Tensor:
        """
        Apply unsupervised classifier-free guidance as described in the paper

        Args:
            sequence: Current sequence with masks
            prompt_ids: Original prompt
            cfg_weight: Guidance weight (w in the paper)

        Returns:
            Guided logits
        """
        # Get conditional predictions p(r0|p0, rt)
        conditional_logits = self.mask_predictor(sequence)

        # Create unconditional sequence (mask the prompt)
        unconditional_sequence = sequence.clone()
        prompt_len = prompt_ids.shape[1]
        unconditional_sequence[:, :prompt_len] = self.mask_token_id

        # Get unconditional predictions p(r0|mask, rt)
        unconditional_logits = self.mask_predictor(unconditional_sequence)

        # Apply classifier-free guidance: p~(r0|p0, rt) ∝ p(r0|p0, rt)^(1+w) / p(r0|mask, rt)^w
        # In log space: log p~ = (1+w) * log p_cond - w * log p_uncond
        guided_logits = (1 + cfg_weight) * conditional_logits - cfg_weight * unconditional_logits

        return guided_logits

    def _sample_tokens(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample tokens with temperature and nucleus sampling"""
        if temperature <= 0:
            return logits.argmax(dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Apply nucleus (top-p) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
            sorted_indices_to_remove[:, :, 0] = 0

            # Scatter back to original indexing
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, logits.size(-1)), 1).view(logits.shape[:-1])

    def _apply_remasking(
        self,
        sequence: torch.Tensor,
        logits: torch.Tensor,
        t: float,
        s: float,
        prompt_len: int,
        strategy: str
    ) -> torch.Tensor:
        """Apply remasking strategy for next diffusion step"""
        if strategy == "random":
            return self._apply_random_remasking(sequence, t, s, prompt_len)
        elif strategy == "low_confidence":
            return self._apply_low_confidence_remasking(sequence, logits, t, s, prompt_len)
        else:
            return sequence  # No remasking for semi-autoregressive (handled separately)

    def _apply_random_remasking(
        self,
        sequence: torch.Tensor,
        t: float,
        s: float,
        prompt_len: int
    ) -> torch.Tensor:
        """Random remasking as in original diffusion"""
        remask_ratio = s / t

        # Only remask in generation region
        gen_region = sequence[:, prompt_len:] if prompt_len > 0 else sequence
        should_remask = torch.rand_like(gen_region.float()) < remask_ratio

        if prompt_len > 0:
            sequence[:, prompt_len:] = torch.where(
                should_remask,
                torch.full_like(gen_region, self.mask_token_id),
                gen_region
            )
        else:
            sequence = torch.where(
                should_remask,
                torch.full_like(sequence, self.mask_token_id),
                sequence
            )

        return sequence

    def _apply_low_confidence_remasking(
        self,
        sequence: torch.Tensor,
        logits: torch.Tensor,
        t: float,
        s: float,
        prompt_len: int
    ) -> torch.Tensor:
        """Low-confidence remasking - remask tokens with lowest prediction confidence"""
        batch_size, seq_len = sequence.shape
        remask_ratio = s / t

        # Calculate confidence scores (max probability for each token)
        probs = F.softmax(logits, dim=-1)
        confidences = torch.gather(probs, -1, sequence.unsqueeze(-1)).squeeze(-1)

        # Only consider generation region
        if prompt_len > 0:
            # Set prompt region to high confidence so it won't be remasked
            confidences[:, :prompt_len] = 1.0

        # Calculate how many tokens to remask
        gen_len = seq_len - prompt_len
        num_to_remask = int(gen_len * remask_ratio)

        if num_to_remask > 0:
            # Find tokens with lowest confidence
            _, lowest_confidence_indices = torch.topk(
                confidences, num_to_remask, dim=-1, largest=False
            )

            # Create remasking pattern
            remask_pattern = torch.zeros_like(sequence, dtype=torch.bool)
            for b in range(batch_size):
                remask_pattern[b, lowest_confidence_indices[b]] = True

            # Apply remasking (but not to prompt region)
            if prompt_len > 0:
                remask_pattern[:, :prompt_len] = False

            sequence[remask_pattern] = self.mask_token_id

        return sequence

    def _apply_low_confidence_remasking_block(
        self,
        sequence: torch.Tensor,
        logits: torch.Tensor,
        block_start: int,
        block_end: int,
        remask_ratio: float
    ) -> torch.Tensor:
        """Apply low-confidence remasking within a specific block"""
        batch_size = sequence.shape[0]
        block_len = block_end - block_start

        # Calculate confidence for block only
        block_logits = logits[:, block_start:block_end, :]
        block_sequence = sequence[:, block_start:block_end]

        probs = F.softmax(block_logits, dim=-1)
        confidences = torch.gather(probs, -1, block_sequence.unsqueeze(-1)).squeeze(-1)

        # Number of tokens to remask in this block
        num_to_remask = int(block_len * remask_ratio)

        if num_to_remask > 0:
            # Find lowest confidence tokens in block
            _, lowest_indices = torch.topk(
                confidences, min(num_to_remask, block_len), dim=-1, largest=False
            )

            # Apply remasking within block
            for b in range(batch_size):
                sequence[b, block_start + lowest_indices[b]] = self.mask_token_id

        return sequence

# Example usage and configuration matching LLaDA 8B from the paper
def create_llada_8b(vocab_size: int) -> LLaDA:
    """Create LLaDA 8B model with architecture from the paper"""
    config = LLaDAConfig(
        vocab_size=vocab_size,
        dim=4096,
        n_layers=32,
        n_heads=32,
        ffn_dim=12288,  # Reduced from LLaMA3's 14336 to compensate for vanilla attention
        max_seq_len=4096,
    )
    return LLaDA.from_config(config)

def create_llada_1b(vocab_size: int) -> LLaDA:
    """Create LLaDA 1B model for experiments"""
    config = LLaDAConfig(
        vocab_size=vocab_size,
        dim=2048,
        n_layers=22,
        n_heads=32,
        ffn_dim=5634,
        max_seq_len=4096,
    )
    return LLaDA.from_config(config)

def create_llada_8b_config(vocab_size: Optional[int] = None) -> LLaDAConfig:
    """Create LLaDA 8B config with architecture from the paper"""
    if vocab_size is None:
        # Use tiktoken GPT-2 vocab size by default
        vocab_size = tiktoken.get_encoding("gpt2").n_vocab

    return LLaDAConfig(
        vocab_size=vocab_size,
        dim=4096,
        n_layers=32,
        n_heads=32,
        ffn_dim=12288,  # Reduced from LLaMA3's 14336 to compensate for vanilla attention
        max_seq_len=4096,
    )

def create_llada_1b_config(vocab_size: Optional[int] = None) -> LLaDAConfig:
    """Create LLaDA 1B config for experiments"""
    if vocab_size is None:
        # Use tiktoken GPT-2 vocab size by default
        vocab_size = tiktoken.get_encoding("gpt2").n_vocab

    return LLaDAConfig(
        vocab_size=vocab_size,
        dim=2048,
        n_layers=22,
        n_heads=32,
        ffn_dim=5634,
        max_seq_len=4096,
    )

def create_llada_8b_tiktoken() -> LLaDAConfig:
    """Create LLaDA 8B config with tiktoken GPT-2 tokenizer settings"""
    return create_llada_8b_config()

def create_llada_1b_tiktoken() -> LLaDAConfig:
    """Create LLaDA 1B config with tiktoken GPT-2 tokenizer settings"""
    return create_llada_1b_config()

# Example training step
def training_example():
    """Example of how to use the model for training"""

    # Example 1: Using tiktoken configuration
    config = create_llada_1b_tiktoken()
    model = LLaDA.from_config(config)

    # Get vocab size from tiktoken
    vocab_size = tiktoken.get_encoding("gpt2").n_vocab

    # Example batch
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    outputs = model(input_ids)
    loss = outputs['loss']

    print(f"Loss: {loss.item():.4f}")
    print(f"Using tiktoken GPT-2 vocab size: {vocab_size}")

    # Example text generation
    print("\nGenerating sample text...")
    with torch.no_grad():
        generated_text = model.generate(
            prompt_ids=None,
            max_length=32,
            num_steps=16,
            temperature=0.8,
            return_text=True
        )
        print(f"Generated: {generated_text}")

    return loss

def load_checkpoint(checkpoint_path: str, config: LLaDAConfig, device: str = 'cpu') -> LLaDA:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")

    # Create model
    model = LLaDA.from_config(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for key in checkpoint['model']:
        new_state_dict[key.replace('_orig_mod.', '')] = checkpoint['model'][key]
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    return model


def generate_sample_texts(model: LLaDA, tokenizer=None, device: str = 'cpu'):
    """Generate sample texts to demonstrate model performance"""
    print("\n" + "="*80)
    print("GENERATING SAMPLE TEXTS")
    print("="*80)

    # Use model's built-in tokenizer or provided one
    def tokens_to_text(tokens):
        if tokenizer is not None:
            try:
                valid_tokens = [t for t in tokens.cpu().numpy() if t != model.mask_token_id and t < model.vocab_size]
                return tokenizer.decode(valid_tokens) if valid_tokens else ""
            except:
                return " ".join([f"<{t}>" for t in tokens.cpu().numpy() if t != model.mask_token_id])
        else:
            return model.tokens_to_text(tokens)

    # First, demonstrate text-to-text generation
    print("\n" + "="*60)
    print("TEXT-TO-TEXT GENERATION DEMOS")
    print("="*60)

    text_prompts = [
        "",  # Unconditional
        "The quick brown fox",
        "In a world where",
        "Once upon a time",
        "The future of AI",
    ]

    for i, prompt_text in enumerate(text_prompts):
        print(f"\n--- Text Demo {i+1} ---")
        if prompt_text == "":
            print("Prompt: <EMPTY> (unconditional generation)")
        else:
            print(f"Prompt: '{prompt_text}'")

        try:
            generated_text = model.generate_from_text(
                prompt_text=prompt_text,
                max_length=64,
                num_steps=32,
                temperature=0.8
            )
            print(f"Generated: '{generated_text}'")
        except Exception as e:
            print(f"Error during text generation: {e}")

        print("-" * 40)

    # Then demonstrate token-based generation for comparison
    print("\n" + "="*60)
    print("TOKEN-BASED GENERATION DEMOS")
    print("="*60)

    # Test cases with different prompt styles
    test_prompts = [
        # Empty prompt (unconditional generation)
        None,

        # Single token prompts (if using simple vocab)
        torch.tensor([[1]], device=device),
        torch.tensor([[5]], device=device),

        # Multi-token prompts
        torch.tensor([[1, 2, 3]], device=device),
        torch.tensor([[10, 20, 30, 40]], device=device),
    ]

    generation_configs = [
        {"max_length": 32, "num_steps": 16, "temperature": 0.0},  # Deterministic
        {"max_length": 32, "num_steps": 32, "temperature": 0.8},  # Creative
        {"max_length": 64, "num_steps": 64, "temperature": 1.0},  # Long and diverse
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test Case {i+1} ---")

        if prompt is None:
            print("Prompt: <NONE> (unconditional generation)")
        else:
            print(f"Prompt tokens: {prompt.cpu().numpy().tolist()}")
            print(f"Prompt text: {tokens_to_text(prompt[0])}")

        for j, gen_config in enumerate(generation_configs):
            print(f"\nGeneration Config {j+1}: {gen_config}")

            try:
                with torch.no_grad():
                    generated = model.generate(
                        prompt_ids=prompt,
                        max_length=gen_config.get('max_length', 512),
                        num_steps=gen_config.get('num_steps', 256),
                        temperature=gen_config.get('temperature', 1.0),
                        return_text=False
                    )

                    # Also generate with return_text=True for direct text output
                    generated_text = model.generate(
                        prompt_ids=prompt,
                        max_length=gen_config.get('max_length', 512),
                        num_steps=gen_config.get('num_steps', 256),
                        temperature=gen_config.get('temperature', 1.0),
                        return_text=True
                    )

                if isinstance(generated, torch.Tensor):
                    print(f"Generated tokens: {generated[0].cpu().numpy().tolist()}")
                    print(f"Generated text (tokens_to_text): {tokens_to_text(generated[0])}")
                print(f"Generated text (direct): {generated_text}")

                # Calculate some basic stats
                if isinstance(generated, torch.Tensor):
                    unique_tokens = len(set(generated[0].cpu().numpy()))
                    total_tokens = len(generated[0])
                    print(f"Stats: {unique_tokens}/{total_tokens} unique tokens, "
                          f"diversity: {unique_tokens/total_tokens:.2f}")

            except Exception as e:
                print(f"Error during generation: {e}")

        print("-" * 40)


def benchmark_generation_speed(model: LLaDA, device: str = 'cpu', num_runs: int = 5):
    """Benchmark generation speed"""
    print("\n" + "="*80)
    print("BENCHMARKING GENERATION SPEED")
    print("="*80)

    import time

    configs = [
        {"max_length": 32, "num_steps": 16},
        {"max_length": 64, "num_steps": 32},
        {"max_length": 128, "num_steps": 64},
    ]

    for config in configs:
        print(f"\nConfig: {config}")
        times = []

        for run in range(num_runs):
            torch.cuda.empty_cache() if device.startswith('cuda') else None

            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(
                    prompt_ids=None,
                    max_length=config.get('max_length', 512),
                    num_steps=config.get('num_steps', 256),
                    temperature=1.0
                )
            end_time = time.time()

            run_time = end_time - start_time
            times.append(run_time)
            print(f"  Run {run+1}: {run_time:.3f}s")

        avg_time = sum(times) / len(times)
        tokens_per_sec = config["max_length"] / avg_time
        print(f"  Average: {avg_time:.3f}s ({tokens_per_sec:.1f} tokens/sec)")


if __name__ == "__main__":
    print("LLaDA Text Generation Demo")
    print("="*50)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Checkpoint path (modify this to your checkpoint location)
    checkpoint_path = "/home/ubuntu/BlaGPT/bla_gpt/logs/llada_long_7/best_model_9125.pt"  # Update this path

    # Try to load from checkpoint, otherwise create new model
    try:
        # Create config with tiktoken (use smaller model for demo)
        config = LLaDAConfig()

        # Try to load checkpoint
        import os
        if os.path.exists(checkpoint_path):
            print(f"\nFound checkpoint at {checkpoint_path}")
            model = load_checkpoint(checkpoint_path, config, device)
            print("✓ Model loaded from checkpoint")
        else:
            print(f"\nNo checkpoint found at {checkpoint_path}")
            print("Creating new model for demonstration...")
            model = LLaDA.from_config(config)
            model.to(device)
            model.eval()
            print("✓ New model created")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating new model for demonstration...")
        config = create_llada_1b_tiktoken()
        model = LLaDA.from_config(config)
        model.to(device)
        model.eval()
        print("✓ New model created")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Info:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"  Mask token ID: {model.mask_token_id}")

    # Quick text generation demo
    print("\nQuick Text Generation Demo:")
    print("-" * 40)

    demo_prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The secret to happiness is"
    ]

    for prompt in demo_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            generated = model.generate_from_text(
                prompt_text=prompt,
                max_length=32,
                num_steps=64,
                temperature=0.8,
                remasking_strategy="semi_autoregressive"
            )
            print(f"Generated: '{generated}'")
        except Exception as e:
            print(f"Error: {e}")

    # Generate sample texts
    print("\nStarting comprehensive text generation...")
    # Initialize tiktoken for text generation demo
    tokenizer = tiktoken.get_encoding("gpt2")
    generate_sample_texts(model, tokenizer=tokenizer, device=device)

    # Benchmark speed
    print("\nStarting speed benchmark...")
    benchmark_generation_speed(model, device=device, num_runs=3)

    # Interactive generation (optional)
    print("\n" + "="*80)
    print("INTERACTIVE GENERATION")
    print("="*80)
    print("Enter custom text prompts or 'quit' to exit")
    print("Example: 'The quick brown fox' will use that text as prompt")
    print("Press Enter for unconditional generation")

    while True:
        try:
            user_input = input("\nPrompt (text): ").strip()

            if user_input.lower() == 'quit':
                break

            if user_input == "":
                # Unconditional generation
                prompt_text = ""
                print("Generating unconditionally...")
            else:
                prompt_text = user_input
                print(f"Using prompt: '{prompt_text}'")

            # Generate using text-to-text method
            with torch.no_grad():
                generated_text = model.generate_from_text(
                    prompt_text=prompt_text,
                    max_length=64,
                    num_steps=128,
                    temperature=0.4,
                    remasking_strategy="low_confidence"
                )

            print(f"Generated text: '{generated_text}'")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDemo completed!")
