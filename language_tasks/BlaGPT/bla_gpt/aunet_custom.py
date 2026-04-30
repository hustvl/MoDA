"""
This is a custom implementation of the AU-Net model that I tried to implement based on the paper's specifications.
"""

import math
import re
from coqpit import Coqpit
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


#
# Configuration class for AU-Net
#


@dataclass
class AUNetConfig(Coqpit):
    """Configuration class for AU-Net model following the paper's specifications"""

    # Basic model configuration
    block_size: int = 4096           # Maximum sequence length
    vocab_size: int = 256            # Raw bytes (0-255)

    # AU-Net specific configuration
    num_stages: int = 3              # Number of hierarchical stages (2-4)
    stage_dims: list = None    # Dimensions for each stage
    stage_layers: list = None   # Number of transformer layers per stage

    # Transformer configuration
    n_head: int = 12                  # Number of attention heads
    dropout: float = 0.0               # Dropout rate
    bias: bool = True                 # Whether to use bias in linear layers
    ffn_ratio: int = 4               # FFN expansion ratio (FFN_dim = ffn_ratio * stage_dim)

    # Window attention for Stage 1 (byte level) to keep computation tractable
    window_size: int = 256           # Window size for Stage 1 attention

    # Advanced options
    tie_weights: bool = False         # Whether to tie embedding and output weights

    # Overiding training parameters
    byte_level_training: bool = True  # Whether to use byte sequences

    def __post_init__(self):
        # Set default values for mutable fields
        if self.stage_dims is None:
            self.stage_dims = [384, 768, 1536]
        if self.stage_layers is None:
            self.stage_layers = [2, 2, 9]

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration parameters"""
        assert self.vocab_size == 256, "AU-Net operates on raw bytes (vocab_size must be 256)"
        assert 2 <= self.num_stages <= 4, "Number of stages should be between 2 and 4"
        assert len(self.stage_dims) >= self.num_stages, "Not enough stage dimensions specified"
        assert len(self.stage_layers) >= self.num_stages, "Not enough stage layers specified"

        # Ensure all stage dimensions are divisible by number of heads
        for i, dim in enumerate(self.stage_dims[:self.num_stages]):
            assert dim % self.n_head == 0, f"Stage {i+1} dimension {dim} not divisible by n_head {self.n_head}"

        # Validate FLOP distribution roughly matches paper for AU-Net-3
        if self.num_stages == 3 and len(self.stage_layers) >= 3:
            total_layers = sum(self.stage_layers[:3])
            stage3_ratio = self.stage_layers[2] / total_layers
            if not (0.6 <= stage3_ratio <= 0.8):  # Paper shows ~70% FLOP at Stage 3
                print(f"WARNING: Stage 3 layer ratio {stage3_ratio:.2f} differs from paper's ~70% FLOP distribution")


#
# Layers
#


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, stage_dim, use_window_attention=False, window_size=512):
        super().__init__()
        assert stage_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(stage_dim, 3 * stage_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(stage_dim, stage_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = stage_dim
        self.dropout = config.dropout

        # Window attention for byte-level stage (Stage 1) as mentioned in paper
        self.use_window_attention = use_window_attention
        self.window_size = window_size

        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply window attention for Stage 1 to keep computation tractable
        if self.use_window_attention and T > self.window_size:
            # Create windowed attention mask
            window_mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
            for i in range(T):
                start = max(0, i - self.window_size + 1)
                window_mask[i, start:i+1] = True
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            attention_mask = window_mask if attention_mask is None else (attention_mask & window_mask)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if attention_mask is not None:
                # Apply custom attention mask
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
                # Use causal mask
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config, stage_dim):
        super().__init__()
        # Use configurable FFN dimension or default 4x expansion
        ffn_ratio = getattr(config, 'ffn_ratio', 4)
        ffn_dim = ffn_ratio * stage_dim
        self.c_fc    = nn.Linear(stage_dim, ffn_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(ffn_dim, stage_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, stage_dim, use_window_attention=False, window_size=512):
        super().__init__()
        self.ln_1 = LayerNorm(stage_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config, stage_dim, use_window_attention, window_size)
        self.ln_2 = LayerNorm(stage_dim, bias=config.bias)
        self.mlp = MLP(config, stage_dim)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


#
# AUnet modules
#


class SplittingFunction:
    """
    Implements the regex-based splitting function from the paper.

    Note: The original paper uses Unicode regex: ( \p{L}{1,16})|\p{N}{1,3}|?([^\s\p{L}\p{N}]){1,3}+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
    This implementation uses a simplified pattern that works with Python's standard re module.
    For full Unicode support, consider using the regex library: pip install regex
    """

    def __init__(self):
        # Simplified regex pattern approximating the paper's Unicode pattern
        # Original: ( \p{L}{1,16})|\p{N}{1,3}|?([^\s\p{L}\p{N}]){1,3}+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
        # Simplified for ASCII/Latin scripts:
        self.word_pattern = re.compile(
            r'[ ]*[a-zA-ZÀ-ÿ]{1,16}|[0-9]{1,3}|[^\s\w]{1,3}[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+'
        )

    def get_word_boundaries(self, text):
        """Get word boundary positions in the text"""
        boundaries = []
        pos = 0
        for match in self.word_pattern.finditer(text):
            end_pos = match.end()
            if end_pos > pos:
                boundaries.append(end_pos - 1)  # Last position of word
                pos = end_pos

        # Ensure we have the final position
        if not boundaries or boundaries[-1] < len(text) - 1:
            boundaries.append(len(text) - 1)

        return boundaries

    def get_split_indices(self, seq_len, stage, byte_sequence=None):
        """
        Get splitting indices for different stages following the paper:
        - Stage 1: raw bytes (no pooling)
        - Stage 2: word boundaries
        - Stage 3: every two words
        - Stage 4: every four words
        """
        if stage == 1:
            # Stage 1: all positions (no pooling)
            return list(range(seq_len))
        elif stage == 2:
            # Stage 2: word boundaries
            if byte_sequence is not None:
                try:
                    # Convert bytes to text for regex processing
                    if isinstance(byte_sequence, torch.Tensor):
                        text = ''.join(chr(min(max(b.item(), 0), 255)) for b in byte_sequence.flatten())
                    else:
                        text = ''.join(chr(min(max(b, 0), 255)) for b in byte_sequence)
                    boundaries = self.get_word_boundaries(text)
                    # Filter boundaries to be within sequence length
                    return [b for b in boundaries if b < seq_len]
                except Exception:
                    # Fallback if text processing fails
                    pass
            # Fallback: every ~4 positions (approximate average word length in bytes)
            return list(range(3, seq_len, 4))
        elif stage == 3:
            # Stage 3: every two words (every ~8 positions)
            return list(range(7, seq_len, 8))
        elif stage == 4:
            # Stage 4: every four words (every ~16 positions)
            return list(range(15, seq_len, 16))
        else:
            raise ValueError(f"Unsupported stage: {stage}")



class MultiLinearUpsampling(nn.Module):
    """
    Multi-Linear Upsampling as described in the paper.

    "duplicate each coarse vector to match the length of the following segment,
    applying distinct, position-specific linear transformations to these duplicates"
    """

    def __init__(self, input_dim, output_dim, max_segment_length=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_segment_length = max_segment_length
        # Different linear layer for each position in segment
        self.position_linears = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(max_segment_length)
        ])

    def forward(self, pooled_vectors, pooling_indices, target_length):
        """
        Upsample pooled vectors back to target length using multi-linear approach.
        Each pooled vector reconstructs the segment that led to that pooled position.

        pooled_vectors: (B, num_pooled, input_dim) - the pooled representations
        pooling_indices: list of indices where pooling occurred
        target_length: desired output sequence length

        Returns: (B, target_length, output_dim)
        """
        B, num_pooled, _ = pooled_vectors.shape

        if num_pooled == 0 or target_length == 0:
            return torch.zeros(B, target_length, self.output_dim, device=pooled_vectors.device)

        # Initialize output
        upsampled = torch.zeros(B, target_length, self.output_dim, device=pooled_vectors.device)

        # Calculate segment boundaries: each pooled vector reconstructs the segment leading to it
        segment_starts = [0] + [idx + 1 for idx in pooling_indices[:-1]] if len(pooling_indices) > 1 else [0]
        segment_ends = [idx + 1 for idx in pooling_indices] if pooling_indices else [target_length]

        for i in range(min(num_pooled, len(segment_starts), len(segment_ends))):
            start_pos = segment_starts[i]
            end_pos = min(segment_ends[i], target_length)
            segment_length = end_pos - start_pos

            if segment_length <= 0:
                continue

            pooled_vec = pooled_vectors[:, i]  # (B, input_dim)

            # Apply position-specific transformations for each position in the segment
            for pos in range(segment_length):
                if start_pos + pos >= target_length:
                    break

                # Use position-specific linear layer (with wrapping for long segments)
                linear_idx = min(pos, self.max_segment_length - 1)
                transformed = self.position_linears[linear_idx](pooled_vec)  # (B, output_dim)
                upsampled[:, start_pos + pos] = transformed

        return upsampled


class AUNetStage(nn.Module):
    """Single stage of AU-Net with Transformer blocks"""

    def __init__(self, config, stage_dim, num_layers, use_window_attention=False, window_size=512):
        super().__init__()
        self.stage_dim = stage_dim
        self.blocks = nn.ModuleList([
            Block(config, stage_dim, use_window_attention, window_size)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(stage_dim, bias=config.bias)

    def forward(self, x, attention_mask=None):
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        return x


#
# AUNet model
#


class AUNet(nn.Module):
    """
    Autoregressive U-Net for language modeling on raw bytes
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size == 256, "AU-Net operates on raw bytes (vocab_size should be 256)"
        self.config = config

        # Configuration for different stages
        self.num_stages = getattr(config, 'num_stages', 3)
        self.stage_dims = getattr(config, 'stage_dims', [512, 2048, 3072])[:self.num_stages]
        self.stage_layers = getattr(config, 'stage_layers', [3, 3, 18])[:self.num_stages]

        # Byte embedding (simple embedding for 256 possible byte values)
        self.byte_embed = nn.Embedding(256, self.stage_dims[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, self.stage_dims[0]))

        # Splitting function
        self.splitter = SplittingFunction()

        # Contracting path stages
        self.contracting_stages = nn.ModuleList()
        for i in range(self.num_stages):
            # Use window attention only for Stage 1 (byte level) as mentioned in paper
            use_window = (i == 0)
            window_size = getattr(config, 'window_size', 512)
            stage = AUNetStage(config, self.stage_dims[i], self.stage_layers[i], use_window, window_size)
            self.contracting_stages.append(stage)

        # Pooling projections (for dimension changes between stages)
        self.pooling_projections = nn.ModuleList()
        for i in range(self.num_stages - 1):
            proj = nn.Linear(self.stage_dims[i], self.stage_dims[i + 1], bias=config.bias)
            self.pooling_projections.append(proj)

        # Expanding path stages (reverse order)
        self.expanding_stages = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):  # Skip the deepest stage
            stage = AUNetStage(config, self.stage_dims[i - 1], self.stage_layers[i - 1])
            self.expanding_stages.append(stage)

        # Upsampling modules
        self.upsamplers = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            upsampler = MultiLinearUpsampling(
                self.stage_dims[i],
                self.stage_dims[i - 1],
                max_segment_length=16  # Maximum segment length
            )
            self.upsamplers.append(upsampler)

        # Skip connection projections - U-Net uses concatenation, so we need to handle doubled dimensions
        self.skip_projections = nn.ModuleList()
        for i in range(self.num_stages - 1):
            # Project from concatenated (upsampled + skip) to target dimension
            # Input: stage_dims[i] + stage_dims[i] = 2 * stage_dims[i] (concatenation)
            # Output: stage_dims[i]
            proj = nn.Linear(2 * self.stage_dims[i], self.stage_dims[i], bias=config.bias)
            self.skip_projections.append(proj)

        # Final head for byte prediction
        self.lm_head = nn.Linear(self.stage_dims[0], 256, bias=False)

        # Weight tying (optional)
        if getattr(config, 'tie_weights', False):
            self.lm_head.weight = self.byte_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embed.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight'.
        # so let's manually remove 'lm_head.weight' from decay set if it's there.
        # This will include the shared weights in the no_decay set.
        if hasattr(self, 'lm_head') and hasattr(self.lm_head, 'weight'):
            decay.discard('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Byte embeddings
        x = self.byte_embed(idx)  # (B, T, stage_dims[0])
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.pos_embed[:, :t, :]  # (B, T, stage_dims[0])

        # Store activations and indices for skip connections and upsampling
        contracting_activations = []
        pooling_indices_per_stage = []

        # Contracting path - processes entire sequence in parallel during training
        current_x = x
        for stage_idx in range(self.num_stages):
            # Apply stage transformer blocks
            current_x = self.contracting_stages[stage_idx](current_x)

            if stage_idx < self.num_stages - 1:
                # Store for skip connection - save before pooling
                contracting_activations.append(current_x)

                # Get split indices for pooling to next stage
                split_indices = self.splitter.get_split_indices(
                    current_x.size(1),
                    stage_idx + 2,  # Stage numbering starts from 1 in the paper
                    byte_sequence=idx[0] if len(idx) > 0 else None  # Use first sequence for split calculation
                )
                pooling_indices_per_stage.append(split_indices)

                # Pool: select vectors at split indices (simple pooling from paper)
                if split_indices:
                    # Ensure indices are within bounds
                    valid_indices = [i for i in split_indices if i < current_x.size(1)]
                    if valid_indices:
                        pooled_x = current_x[:, valid_indices, :]  # (B, num_selected, dim)
                        # Project to next stage dimension
                        current_x = self.pooling_projections[stage_idx](pooled_x)
                    else:
                        # Fallback: take last position and project
                        pooled_x = current_x[:, -1:, :]
                        current_x = self.pooling_projections[stage_idx](pooled_x)
                        valid_indices = [current_x.size(1) - 1]
                    pooling_indices_per_stage[-1] = valid_indices
                else:
                    # No valid split points, take the last position
                    pooled_x = current_x[:, -1:, :]
                    current_x = self.pooling_projections[stage_idx](pooled_x)
                    pooling_indices_per_stage[-1] = [current_x.size(1) - 1]

        # Expanding path - reverses the contracting process with skip connections
        for expand_idx in range(self.num_stages - 1):
            contracting_stage_idx = self.num_stages - 2 - expand_idx  # Which contracting stage to connect to
            target_activation = contracting_activations[contracting_stage_idx]
            target_length = target_activation.size(1)

            # Get the pooling indices that were used for this stage
            if expand_idx < len(pooling_indices_per_stage):
                pooling_indices = pooling_indices_per_stage[-(expand_idx + 1)]
            else:
                pooling_indices = [target_length - 1]  # Fallback

            # Upsample using Multi-Linear Upsampling
            # "duplicate each coarse vector to match the length of the following segment,
            # applying distinct, position-specific linear transformations"
            upsampled_x = self.upsamplers[expand_idx](
                current_x,
                pooling_indices,
                target_length
            )

            # U-Net style skip connection: concatenate instead of add
            # This preserves both local details from contracting path and
            # high-level information from expanding path
            skip_connection = target_activation  # No projection needed, we'll project after concat

            # Concatenate upsampled signal with skip connection (U-Net style)
            concatenated = torch.cat([upsampled_x, skip_connection], dim=-1)  # (B, T, 2*dim)

            # Project concatenated features back to target dimension
            current_x = self.skip_projections[contracting_stage_idx](concatenated)

            # Apply expanding stage transformer blocks
            current_x = self.expanding_stages[expand_idx](current_x)

        # Final output projection to vocabulary
        logits = self.lm_head(current_x)  # (B, T, vocab_size)

        if targets is not None:
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation with cascading stage activation as described in paper.
        The byte-level stage is active at every step, while deeper stages activate
        less frequently according to the pooling pattern.
        """
        self.eval()

        for step in range(max_new_tokens):
            # Crop context if it grows too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get next token logits
            logits, _ = self(idx_cond)

            # Extract logits for the last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_efficient(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        More efficient generation implementing cascading activation as described in the paper.

        "At inference, generation is autoregressive: the byte-level stage is active at every step,
        while deeper stages activate less frequently according to the pooling pattern."

        This is a simplified version - full implementation would require:
        - Caching stage outputs at pooling boundaries
        - Only recomputing stages when their pooling pattern triggers
        - Proper propagation of cached states through skip connections
        """
        self.eval()

        # For now, use the standard generation approach
        # TODO: Implement true cascading activation for efficiency
        return self.generate(idx, max_new_tokens, temperature, top_k)


if __name__ == "__main__":
    # Example usage demonstrating AU-Net's key features

    # Test different AU-Net configurations as described in the paper
    configs = {
        'AU-Net-2': AUNetConfig(
            block_size=1024,
            num_stages=2,
            stage_dims=[512, 2048],
            stage_layers=[3, 25],  # Most computation at stage 2 to match ~96% FLOP
            n_head=8,
            dropout=0.1
        ),
        'AU-Net-3': AUNetConfig(
            block_size=1024,
            num_stages=3,
            stage_dims=[512, 2048, 3072],
            stage_layers=[3, 3, 18],  # 70% FLOP at stage 3 as per paper
            n_head=8,
            dropout=0.1
        ),
        'AU-Net-4': AUNetConfig(
            block_size=1024,
            num_stages=4,
            stage_dims=[512, 2048, 3072, 4608],
            stage_layers=[3, 3, 4, 10],  # Most computation at deepest stage
            n_head=8,
            dropout=0.1
        )
    }

    print("Testing AU-Net configurations from the paper:")
    print("=" * 50)

    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        model = AUNet(config)

        # Test with sample byte sequence (representing "Hello World!")
        text = "Hello World!"
        byte_sequence = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

        print(f"Input text: '{text}'")
        print(f"Byte sequence shape: {byte_sequence.shape}")

        # Forward pass
        with torch.no_grad():
            logits, _ = model(byte_sequence)
            print(f"Output logits shape: {logits.shape}")

            # Generate some new bytes
            generated = model.generate(byte_sequence[:, :5], max_new_tokens=10, temperature=0.8)
            generated_text = ''.join(chr(max(0, min(255, token.item()))) for token in generated[0])
            print(f"Generated text: '{generated_text}'")

        print("-" * 30)

    # Demonstrate byte-level processing advantage
    print("\nByte-level processing examples:")
    print("=" * 50)

    model = AUNet(configs['AU-Net-3'])

    # Test with different languages/scripts
    test_texts = [
        "Hello",           # English
        "Héllo",          # Accented characters
        "🤖",             # Emoji (will be multiple bytes in UTF-8)
        "123.45",         # Numbers and punctuation
    ]

    for text in test_texts:
        try:
            # Convert to UTF-8 bytes
            utf8_bytes = text.encode('utf-8')
            byte_tensor = torch.tensor(list(utf8_bytes), dtype=torch.long).unsqueeze(0)

            print(f"Text: '{text}'")
            print(f"UTF-8 bytes: {list(utf8_bytes)}")
            print(f"Tensor shape: {byte_tensor.shape}")

            with torch.no_grad():
                logits, _ = model(byte_tensor)
                print(f"Output shape: {logits.shape}")

            print("-" * 20)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
