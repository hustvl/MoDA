"""
Data loading utilities for BlaGPT training.

Supports:
- Binary shard loading (pre-tokenized .bin files)
- Byte-level shard loading
- HuggingFace streaming datasets with on-the-fly tokenization
"""

import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from coqpit import Coqpit

try:
    import torch.distributed as dist
except ImportError:
    dist = None


def print_rank0(*args, **kwargs):
    """Print only on rank 0 (master process) in distributed training."""
    if dist is None or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


class DataShard:
    """Handles loading and validation of BPE token data shards."""

    MAGIC_NUMBER = 20240520
    HEADER_SIZE = 256 * 4
    VERSION = 1

    @staticmethod
    def peek_data_shard(filename: str) -> int:
        """Read the token count from a shard file header."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)

        if header[0] != DataShard.MAGIC_NUMBER:
            print_rank0("ERROR: magic number mismatch in the data .bin file!")
            print_rank0("---> HINT: Are you passing in a correct file with --input_bin?")
            print_rank0(
                "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
            )
            exit(1)
        assert header[1] == DataShard.VERSION, "Unsupported version"
        return header[2]

    @staticmethod
    def load_data_shard(filename: str) -> np.ndarray:
        """Load tokens from a shard file."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)
            assert header[0] == DataShard.MAGIC_NUMBER, "Magic number mismatch"
            assert header[1] == DataShard.VERSION, "Unsupported version"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)

        assert len(tokens) == ntok, "Token count mismatch"
        return tokens


class ByteDataShard:
    """Handles loading and validation of byte-level data shards."""

    MAGIC_NUMBER = 20240520
    HEADER_SIZE = 256 * 4
    VERSION = 1

    @staticmethod
    def peek_byte_data_shard(filename: str) -> int:
        """Read the byte count from a shard file header."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(ByteDataShard.HEADER_SIZE), dtype=np.int32)

        if header[0] != ByteDataShard.MAGIC_NUMBER:
            print_rank0("ERROR: magic number mismatch in the byte data .bin file!")
            print_rank0("---> HINT: Are you passing in a correct byte-level data file?")
            print_rank0(
                "---> HINT: For byte-level training, use fineweb_bytes.py to generate data"
            )
            exit(1)
        assert header[1] == ByteDataShard.VERSION, "Unsupported version"
        return header[2]

    @staticmethod
    def load_byte_data_shard(filename: str) -> np.ndarray:
        """Load bytes from a shard file."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(ByteDataShard.HEADER_SIZE), dtype=np.int32)
            assert header[0] == ByteDataShard.MAGIC_NUMBER, "Magic number mismatch"
            assert header[1] == ByteDataShard.VERSION, "Unsupported version"
            nbytes = header[2]
            bytes_data = np.frombuffer(f.read(), dtype=np.uint8)

        assert len(bytes_data) == nbytes, "Byte count mismatch"
        return bytes_data


class DistributedDataLoader:
    """Distributed data loader for handling pre-tokenized binary shards."""

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_length: int,
        process_rank: int,
        num_processes: int,
        byte_level_training: bool = False,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.byte_level_training = byte_level_training

        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")

        self.ntok_total = self._validate_shards()
        self.reset()

    def _load_shard_data(self, filename: str) -> np.ndarray:
        """Load data from a shard file based on training mode."""
        if self.byte_level_training:
            if filename.endswith(".bin"):
                return ByteDataShard.load_byte_data_shard(filename)
            else:
                raise ValueError(
                    f"Byte-level training only supports .bin files, got: {filename}"
                )
        else:
            return DataShard.load_data_shard(filename)

    def _get_shard_size(self, filename: str) -> int:
        """Get the size of a shard file based on training mode."""
        if self.byte_level_training:
            if filename.endswith(".bin"):
                return ByteDataShard.peek_byte_data_shard(filename)
            else:
                raise ValueError(
                    f"Byte-level training only supports .bin files, got: {filename}"
                )
        else:
            return DataShard.peek_data_shard(filename)

    def _validate_shards(self) -> int:
        """Validate all shards and return total token count."""
        ntok_total = 0
        min_required = self.num_processes * self.batch_size * self.seq_length + 1

        for fname in self.files:
            shard_ntok = self._get_shard_size(fname)
            assert shard_ntok >= min_required, f"Shard {fname} too small"
            ntok_total += int(shard_ntok)
        return ntok_total

    def reset(self):
        """Reset loader to beginning of first shard."""
        self.current_shard = 0
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = self._load_shard_data(self.files[self.current_shard])

    def advance(self):
        """Move to next shard."""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = self._load_shard_data(self.files[self.current_shard])

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input, target) tensors."""
        B, T = self.batch_size, self.seq_length
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        if self.byte_level_training:
            assert x.max().item() < 256, f"Byte value out of range: {x.max().item()}"

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda()


@dataclass
class HFDatasetConfig(Coqpit):
    """Configuration for HuggingFace streaming datasets."""

    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: Optional[str] = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    streaming: bool = True
    shuffle_buffer_size: int = 10000
    seed: int = 42
    # For streaming datasets: skip first N samples (use for training to skip val samples)
    skip_samples: int = 0
    # For streaming datasets: take only first N samples (use for validation)
    take_samples: Optional[int] = None
    # Caching options for validation data
    cache_tokens: bool = False
    cache_dir: Optional[str] = None


def generate_cache_key(
    config: HFDatasetConfig, tokenizer_backend: str, tokenizer_name: str
) -> str:
    """Generate a deterministic cache key based on dataset and tokenizer config.

    The key includes all parameters that would affect the cached token output,
    ensuring different configurations produce different cache files.
    """
    key_parts = [
        config.dataset_name,
        config.dataset_config or "default",
        config.split,
        str(config.take_samples or "all"),
        tokenizer_backend,
        tokenizer_name,
    ]
    return hashlib.sha256("_".join(key_parts).encode()).hexdigest()[:16]


class HFStreamingDataLoader:
    """Distributed data loader for HuggingFace streaming datasets.

    Tokenizes on-the-fly and packs sequences efficiently.
    """

    def __init__(
        self,
        config: HFDatasetConfig,
        tokenizer,  # TokenizerWrapper
        batch_size: int,
        seq_length: int,
        process_rank: int,
        num_processes: int,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        self._token_buffer: List[int] = []
        self._iterator: Optional[Iterator] = None
        self._dataset = None

        # Track approximate tokens seen
        self.ntok_total = 0  # Will be updated during iteration

        self._init_dataset()

    def _init_dataset(self):
        """Initialize the HuggingFace dataset."""
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        # Disable progress bars on non-rank-0 processes to avoid cluttered output
        if self.process_rank != 0:
            from datasets import disable_progress_bar

            disable_progress_bar()

        try:
            self._dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split,
                streaming=self.config.streaming,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset '{self.config.dataset_name}' "
                f"with config '{self.config.dataset_config}': {e}"
            )

        # Apply take/skip for train/val splitting (streaming datasets don't support slice syntax)
        # For validation: take first N samples
        if self.config.take_samples is not None:
            self._dataset = self._dataset.take(self.config.take_samples)
        # For training: skip first N samples (those used for validation)
        if self.config.skip_samples > 0:
            self._dataset = self._dataset.skip(self.config.skip_samples)

        # Shard across processes for distributed training
        # For IterableDataset (streaming), use split_dataset_by_node
        if self.num_processes > 1:
            self._dataset = split_dataset_by_node(
                self._dataset,
                rank=self.process_rank,
                world_size=self.num_processes,
            )

        # Shuffle with buffer
        if self.config.shuffle_buffer_size > 0:
            self._dataset = self._dataset.shuffle(
                seed=self.config.seed, buffer_size=self.config.shuffle_buffer_size
            )

        self._iterator = iter(self._dataset)

    def reset(self):
        """Reset the data loader by reinitializing the iterator."""
        self._token_buffer = []
        self._init_dataset()

    def _fill_buffer(self, min_tokens: int):
        """Fill token buffer until we have at least min_tokens."""
        eos_id = self.tokenizer.eos_token_id

        while len(self._token_buffer) < min_tokens:
            try:
                example = next(self._iterator)
            except StopIteration:
                # Reinitialize iterator when exhausted
                self._iterator = iter(self._dataset)
                example = next(self._iterator)

            text = example.get(self.config.text_column, "")
            if not text:
                continue

            # Tokenize and add EOS
            tokens = self.tokenizer.encode(text)
            tokens.append(eos_id)
            self._token_buffer.extend(tokens)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input, target) tensors."""
        B, T = self.batch_size, self.seq_length
        needed_tokens = B * T + 1

        self._fill_buffer(needed_tokens)

        # Extract tokens for this batch
        batch_tokens = self._token_buffer[:needed_tokens]
        self._token_buffer = self._token_buffer[needed_tokens - 1 :]  # Keep last token for continuity

        # Convert to tensor
        buf = torch.tensor(batch_tokens, dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Track tokens seen
        self.ntok_total += B * T

        return x.cuda(), y.cuda()


class CachedHFValidationLoader:
    """Cached data loader for HuggingFace streaming validation datasets.

    Caches tokenized validation data to disk on first run.
    Subsequent validations load directly from cache, avoiding
    redundant network fetches and tokenization.
    """

    def __init__(
        self,
        config: HFDatasetConfig,
        tokenizer,  # TokenizerWrapper
        batch_size: int,
        seq_length: int,
        process_rank: int,
        num_processes: int,
        tokenizer_backend: str = "tiktoken",
        tokenizer_name: str = "gpt2",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Generate cache key and path
        self.cache_key = generate_cache_key(config, tokenizer_backend, tokenizer_name)
        cache_dir = config.cache_dir or "data/hf_cache"
        self.cache_path = os.path.join(cache_dir, f"hf_val_cache_{self.cache_key}.bin")

        # Token storage
        self._tokens: Optional[np.ndarray] = None
        self._current_position = 0
        self.ntok_total = 0

        # Load or create cache
        self._init_cache()

    def _init_cache(self):
        """Initialize cache: load existing or create new."""
        if os.path.exists(self.cache_path):
            self._load_from_cache()
        else:
            self._create_cache()

    def _load_from_cache(self):
        """Load cached tokens from disk."""
        self._tokens = DataShard.load_data_shard(self.cache_path)
        self.ntok_total = len(self._tokens)
        print_rank0(f"Loaded {self.ntok_total:,} validation tokens from cache: {self.cache_path}")

    def _create_cache(self):
        """Create cache by tokenizing validation data."""
        # Only rank 0 creates the cache
        if self.process_rank == 0:
            print_rank0(f"Creating validation token cache: {self.cache_path}")
            self._fetch_and_cache_tokens()

        # Synchronize across ranks
        if dist is not None and dist.is_initialized():
            dist.barrier()

        # All ranks load from cache
        if self.process_rank != 0:
            self._load_from_cache()

    def _fetch_and_cache_tokens(self):
        """Fetch all validation samples, tokenize, and write to cache."""
        from datasets import load_dataset

        # Load dataset (streaming)
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
            streaming=True,
        )

        # Take only N samples for validation
        if self.config.take_samples is not None:
            dataset = dataset.take(self.config.take_samples)

        # Tokenize all samples
        all_tokens: List[int] = []
        eos_id = self.tokenizer.eos_token_id
        sample_count = 0

        for example in dataset:
            text = example.get(self.config.text_column, "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text)
            tokens.append(eos_id)
            all_tokens.extend(tokens)
            sample_count += 1

            if sample_count % 100 == 0:
                print_rank0(f"  Tokenized {sample_count} samples, {len(all_tokens):,} tokens...")

        print_rank0(f"  Completed: {sample_count} samples, {len(all_tokens):,} tokens")

        # Convert to numpy array
        self._tokens = np.array(all_tokens, dtype=np.uint16)
        self.ntok_total = len(self._tokens)

        # Write to cache file
        self._write_cache()

    def _write_cache(self):
        """Write cached tokens to disk in DataShard format."""
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Create header (same format as DataShard)
        header = np.zeros(256, dtype=np.int32)
        header[0] = DataShard.MAGIC_NUMBER
        header[1] = DataShard.VERSION
        header[2] = len(self._tokens)

        with open(self.cache_path, "wb") as f:
            f.write(header.tobytes())
            f.write(self._tokens.tobytes())

        print_rank0(f"  Cache written to: {self.cache_path}")

    def reset(self):
        """Reset the data loader to the beginning."""
        self._current_position = self.process_rank * self.batch_size * self.seq_length

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input, target) tensors."""
        B, T = self.batch_size, self.seq_length
        needed_tokens = B * T + 1

        # Handle wrap-around if we reach end of tokens
        if self._current_position + needed_tokens > len(self._tokens):
            self._current_position = self.process_rank * self.batch_size * self.seq_length

        # Extract tokens for this batch
        buf = self._tokens[self._current_position : self._current_position + needed_tokens]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance position (stride by all processes' worth)
        self._current_position += B * T * self.num_processes

        return x.cuda(), y.cuda()
