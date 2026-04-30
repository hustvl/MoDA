import datetime
import glob
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_model
from optimizers import get_optimizer
from data_loaders import (
    DataShard,
    ByteDataShard,
    DistributedDataLoader,
    HFStreamingDataLoader,
    HFDatasetConfig,
)
from tokenizers_config import TokenizerConfig, TokenizerWrapper

# Set environment variables
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"


def print_rank0(*args, **kwargs):
    """Print only on rank 0 (master process) in distributed training."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


@dataclass
class Hyperparameters(Coqpit):
    """Training hyperparameters configuration."""

    run_name: str = field(
        default="nano_gpt+rms_norm+geglu+gqa+softcap",
        metadata={"help": "Name for this training run"}
    )
    compile_model: bool = field(
        default=False,
        metadata={"help": "Compile model with torch.compile for better performance"}
    )
    input_bin: str = field(
        default="../data/fineweb10B/fineweb_train_*.bin",
        metadata={"help": "Pattern for training data files (BPE tokenized)"}
    )
    input_val_bin: str = field(
        default="../data/fineweb10B/fineweb_val_*.bin",
        metadata={"help": "Pattern for validation data files (BPE tokenized)"}
    )
    byte_level_training: bool = field(
        default=False,
        metadata={"help": "Enable byte-level training mode"}
    )
    text_data_dir: str = field(
        default="../data/fineweb10B_text",
        metadata={"help": "Directory containing text data (.jsonl files)"}
    )
    byte_data_dir: str = field(
        default="../data/fineweb10B_bytes",
        metadata={"help": "Directory containing byte data (.bin files)"}
    )
    # HuggingFace streaming mode
    use_hf_streaming: bool = field(
        default=False,
        metadata={"help": "Use HuggingFace streaming datasets instead of binary shards"}
    )
    hf_dataset: str = field(
        default="HuggingFaceFW/fineweb",
        metadata={"help": "HuggingFace dataset name or path"}
    )
    hf_dataset_config: Optional[str] = field(
        default="sample-10BT",
        metadata={"help": "HuggingFace dataset configuration/subset name"}
    )
    hf_text_column: str = field(
        default="text",
        metadata={"help": "Name of the text column in the dataset"}
    )
    hf_val_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Validation dataset name (defaults to hf_dataset if None)"}
    )
    hf_val_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Validation dataset config (defaults to hf_dataset_config if None)"}
    )
    hf_shuffle_buffer: int = field(
        default=10000,
        metadata={"help": "Shuffle buffer size for streaming datasets"}
    )
    # Tokenizer configuration
    tokenizer_backend: str = field(
        default="tiktoken",
        metadata={"help": "Tokenizer backend: 'tiktoken' or 'huggingface'"}
    )
    tokenizer_name: str = field(
        default="gpt2",
        metadata={"help": "Tokenizer name (tiktoken encoding or HF model name)"}
    )
    batch_size: int = field(
        default=8 * 64,
        metadata={"help": "Total batch size across all devices"}
    )
    device_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per device"}
    )
    sequence_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for training"}
    )
    num_iterations: int = field(
        default=5100,
        metadata={"help": "Number of training iterations"}
    )
    learning_rate: float = field(
        default=0.001,
        metadata={"help": "Learning rate for optimizer"}
    )
    warmup_iters: int = field(
        default=250,
        metadata={"help": "Number of warmup iterations"}
    )
    warmdown_iters: int = field(
        default=2000,
        metadata={"help": "Number of warmdown iterations"}
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "Weight decay coefficient"}
    )
    val_loss_every: int = field(
        default=125,
        metadata={"help": "Validate every N iterations"}
    )
    val_tokens: int = field(
        default=10485760,
        metadata={"help": "Number of tokens for validation"}
    )
    save_every: int = field(
        default=5000,
        metadata={"help": "Save checkpoint every N iterations"}
    )
    keep_last_n_checkpoints: int = field(
        default=1,
        metadata={"help": "Number of recent checkpoints to keep"}
    )
    save_best_model: bool = field(
        default=True,
        metadata={"help": "Save best model based on validation loss"}
    )
    precision: str = field(
        default="bfloat16",
        metadata={"help": "Training precision (float32 or bfloat16)"}
    )
    optimizer_name: str = field(
        default="Adam",
        metadata={"help": "Optimizer name (Adam, AdamW, etc.)"}
    )
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        metadata={"help": "Optimizer arguments"}
    )


class TeeLogger:
    """Logger that writes to both terminal and file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


class Trainer:
    """Main trainer class handling the training loop and validation."""

    def __init__(
        self, args: Hyperparameters, model_name: str, config_path: Optional[str] = None
    ):
        self.args = args

        self.setup_distributed()
        self.setup_model(model_name, config_path)
        self.setup_optimization()
        self.setup_data_loaders()
        if self.is_master:
            self.setup_logging()



    def setup_distributed(self):
        """Initialize distributed training setup."""
        assert torch.cuda.is_available()
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30)
        )
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.local_rank}"
        self.is_master = self.rank == 0
        torch.cuda.set_device(self.device)

    def setup_model(self, model_name: str, config_path: Optional[str]):
        """Initialize model and move to GPU."""
        self.model_config, model_cls = get_model(model_name)

        if config_path:
            self.model_config.load_json(config_path)

        # Override Hyperparameters with matching model_config attributes
        overridden_params = []
        for key, value in self.model_config.to_dict().items():
            if hasattr(self.args, key):
                old_value = getattr(self.args, key)
                if old_value != value:
                    overridden_params.append((key, old_value, value))
                setattr(self.args, key, value)

        # Log overridden parameters (only on master process)
        if overridden_params and self.is_master:
            print_rank0("Hyperparameters overridden by model config:")
            for key, old_val, new_val in overridden_params:
                print_rank0(f"  {key}: {old_val} -> {new_val}")

        # Validate vocab size for byte-level training
        if self.args.byte_level_training:
            if hasattr(self.model_config, 'vocab_size') and self.model_config.vocab_size != 256:
                raise ValueError(
                    f"Byte-level training requires vocab_size=256, but model config has vocab_size={self.model_config.vocab_size}. "
                    f"Please update your model configuration for byte-level training."
                )
            elif hasattr(self.model_config, 'vocab_size'):
                print_rank0(f"✓ Vocab size validation passed: {self.model_config.vocab_size} (suitable for byte-level training)")

        self.model = model_cls(self.model_config).cuda()

        if self.args.compile_model:
            print_rank0("Compiling the model...")
            self.model = torch.compile(self.model)
            print_rank0("✓ Model compilation passed")

        self.model = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=False
        )
        self.raw_model = self.model.module
        self.ctx = autocast(
            device_type="cuda", dtype=torch.bfloat16 if self.args.precision == "bfloat16" else torch.float32
        )

    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = get_optimizer(
            self.args.optimizer_name,
            self.args.optimizer_args,
            self.args.learning_rate,
            self.raw_model
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.get_lr_schedule()
        )

    def get_lr_schedule(self):
        """Create learning rate schedule function."""

        def schedule(it):
            if it < self.args.warmup_iters:
                return (it + 1) / self.args.warmup_iters
            elif it < self.args.num_iterations - self.args.warmdown_iters:
                return 1.0
            else:
                return (self.args.num_iterations - it) / self.args.warmdown_iters

        return schedule

    def setup_data_loaders(self):
        """Initialize training and validation data loaders."""
        if self.args.use_hf_streaming:
            self._setup_hf_streaming_loaders()
        elif self.args.byte_level_training:
            self._setup_byte_level_loaders()
        else:
            self._setup_binary_shard_loaders()

        self.train_accumulation_steps = self.args.batch_size // (
            self.args.device_batch_size * self.world_size
        )

        self.val_steps = self.args.val_tokens // (
            self.args.device_batch_size * self.args.sequence_length * self.world_size
        )

        if self.is_master:
            print_rank0(f"Data loader setup complete:")
            print_rank0(f"  Training accumulation steps: {self.train_accumulation_steps}")
            print_rank0(f"  Validation steps: {self.val_steps}")

    def _setup_hf_streaming_loaders(self):
        """Set up HuggingFace streaming data loaders."""
        if self.is_master:
            print_rank0("Setting up HuggingFace streaming data loaders...")
            print_rank0(f"  Dataset: {self.args.hf_dataset}")
            print_rank0(f"  Config: {self.args.hf_dataset_config}")
            print_rank0(f"  Text column: {self.args.hf_text_column}")

        # Initialize tokenizer
        tokenizer_config = TokenizerConfig(
            backend=self.args.tokenizer_backend,
            name=self.args.tokenizer_name,
        )
        self.tokenizer = TokenizerWrapper(tokenizer_config)

        if self.is_master:
            print_rank0(f"  Tokenizer: {self.tokenizer}")

        # Validate vocab size against model config
        model_vocab_size = getattr(self.model_config, 'vocab_size', None)
        if model_vocab_size and model_vocab_size != self.tokenizer.vocab_size:
            if self.is_master:
                print_rank0(
                    f"  WARNING: Model vocab_size ({model_vocab_size}) != tokenizer vocab_size ({self.tokenizer.vocab_size})"
                )

        # Training dataset config
        train_config = HFDatasetConfig(
            dataset_name=self.args.hf_dataset,
            dataset_config=self.args.hf_dataset_config,
            split="train",
            text_column=self.args.hf_text_column,
            shuffle_buffer_size=self.args.hf_shuffle_buffer,
            seed=42,
        )

        # Validation dataset config
        val_dataset = self.args.hf_val_dataset or self.args.hf_dataset
        val_config_name = self.args.hf_val_dataset_config or self.args.hf_dataset_config

        val_config = HFDatasetConfig(
            dataset_name=val_dataset,
            dataset_config=val_config_name,
            split="train",  # Many datasets don't have a val split
            text_column=self.args.hf_text_column,
            shuffle_buffer_size=1000,  # Smaller buffer for validation
            seed=123,  # Different seed for validation
        )

        try:
            self.train_loader = HFStreamingDataLoader(
                train_config,
                self.tokenizer,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
            )

            self.val_loader = HFStreamingDataLoader(
                val_config,
                self.tokenizer,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
            )
        except Exception as e:
            if self.is_master:
                print_rank0(f"Error setting up HF streaming loaders: {e}")
                print_rank0("\nTroubleshooting:")
                print_rank0(f"- Check dataset name: {self.args.hf_dataset}")
                print_rank0(f"- Check dataset config: {self.args.hf_dataset_config}")
                print_rank0(f"- Check text column: {self.args.hf_text_column}")
                print_rank0("- Ensure you have internet access for streaming")
            raise

        if self.is_master:
            print_rank0(f"  HF streaming loaders initialized successfully")

    def _setup_byte_level_loaders(self):
        """Set up byte-level binary shard data loaders."""
        if self.is_master:
            print_rank0("Setting up byte-level training data loaders...")

        train_pattern = os.path.join(self.args.byte_data_dir, "fineweb_train_*.bin")
        val_pattern = os.path.join(self.args.byte_data_dir, "fineweb_val_*.bin")

        train_files = glob.glob(train_pattern)
        val_files = glob.glob(val_pattern)

        if not train_files:
            raise ValueError(
                f"No byte-level training files found at {train_pattern}. "
                f"Please generate byte data using fineweb_bytes.py first."
            )

        if not val_files:
            if self.is_master:
                print_rank0(f"No validation files found at {val_pattern}")
                print_rank0("Using training files for validation...")
            val_pattern = train_pattern

        if self.is_master:
            print_rank0(f"Using binary files for byte-level training")
            print_rank0(f"Training files: {len(glob.glob(train_pattern))}")
            print_rank0(f"Validation files: {len(glob.glob(val_pattern))}")

        self._create_distributed_loaders(train_pattern, val_pattern, byte_level=True)

    def _setup_binary_shard_loaders(self):
        """Set up BPE token binary shard data loaders."""
        train_pattern = self.args.input_bin
        val_pattern = self.args.input_val_bin

        if self.is_master:
            print_rank0("Setting up BPE token-level training data loaders...")
            print_rank0(f"Training pattern: {train_pattern}")
            print_rank0(f"Validation pattern: {val_pattern}")

        self._create_distributed_loaders(train_pattern, val_pattern, byte_level=False)

    def _create_distributed_loaders(self, train_pattern: str, val_pattern: str, byte_level: bool):
        """Create distributed data loaders from file patterns."""
        try:
            self.train_loader = DistributedDataLoader(
                train_pattern,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
                byte_level,
            )

            self.val_loader = DistributedDataLoader(
                val_pattern,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
                byte_level,
            )
        except Exception as e:
            if self.is_master:
                print_rank0(f"Error setting up data loaders: {e}")
                print_rank0("\nTroubleshooting:")
                if byte_level:
                    print_rank0("For byte-level training:")
                    print_rank0(f"- Ensure byte data directory exists: {self.args.byte_data_dir}")
                    print_rank0("- Run: cd ../data && python fineweb_bytes.py --version 10B")
                else:
                    print_rank0("For BPE token training:")
                    print_rank0(f"- Check input_bin path: {self.args.input_bin}")
                    print_rank0(f"- Check input_val_bin path: {self.args.input_val_bin}")
                    print_rank0("- Run: cd ../data && python fineweb.py --version 10B")
            raise

        if self.is_master:
            print_rank0(f"  Total training tokens: {self.train_loader.ntok_total:,}")
            print_rank0(f"  Total validation tokens: {self.val_loader.ntok_total:,}")
            if byte_level:
                print_rank0(f"  Vocabulary size: 256 (byte-level)")
            else:
                print_rank0(f"  Vocabulary size: {getattr(self.model_config, 'vocab_size', 'Unknown')}")

    def setup_logging(self):
        """Initialize logging directory and files."""
        run_num = 0
        self.run_id = f"{self.args.run_name}_{run_num}"
        self.logdir = f"logs/{self.run_id}/"

        while os.path.exists(self.logdir):
            run_num += 1
            self.run_id = f"{self.args.run_name}_{run_num}"
            self.logdir = f"logs/{self.run_id}/"

        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"

        sys.stdout = TeeLogger(self.logfile)
        self._log_initial_info()

    def _format_number(self, num):
        """Format a number in human-readable format (e.g., 1.2M, 345K, 1.5B)."""
        if num >= 1e9:
            return f"{num / 1e9:.1f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.1f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.1f}K"
        else:
            return str(num)

    def _count_parameters(self):
        """Count the total number of parameters in the model."""
        total_params = sum(p.numel() for p in self.raw_model.parameters())
        trainable_params = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def _log_initial_info(self):
        """Log initial information about the training run."""

        print_rank0(
            f"Running pytorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )

        # Log model parameter count
        total_params, trainable_params = self._count_parameters()
        total_formatted = self._format_number(total_params)
        trainable_formatted = self._format_number(trainable_params)
        print_rank0(f"Model parameters: {total_formatted} total ({total_params:,}), {trainable_formatted} trainable ({trainable_params:,})")

        print_rank0("=" * 100)
        if self.args.use_hf_streaming:
            print_rank0("Data mode: HuggingFace Streaming")
            print_rank0(f"  Dataset: {self.args.hf_dataset} ({self.args.hf_dataset_config})")
            print_rank0(f"  Tokenizer: {self.args.tokenizer_backend}/{self.args.tokenizer_name}")
            if hasattr(self, 'tokenizer'):
                print_rank0(f"  Vocab size: {self.tokenizer.vocab_size}")
        elif self.args.byte_level_training:
            print_rank0("Data mode: Byte-level (binary shards)")
            print_rank0(f"  Data directory: {self.args.byte_data_dir}")
            print_rank0(f"  Vocab size: 256 (bytes)")
        else:
            print_rank0("Data mode: BPE tokens (binary shards)")
            print_rank0(f"  Train pattern: {self.args.input_bin}")
            print_rank0(f"  Vocab size: {getattr(self.model_config, 'vocab_size', 'Unknown')}")
        print_rank0("=" * 100)

    def validate(self) -> float:
        """Run validation loop and return validation loss."""
        self.model.eval()
        self.val_loader.reset()
        val_loss = 0.0

        with torch.no_grad():
            for _ in range(self.val_steps):
                x_val, y_val = self.val_loader.next_batch()
                with self.ctx:
                    _, loss = self.model(x_val, y_val)

                    if isinstance(loss, dict):
                        loss = loss["total"]
                    val_loss += loss.detach()

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        return val_loss / self.val_steps

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        log = {
            "step": step,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if val_loss is not None:
            log["val_loss"] = val_loss

        checkpoint_path = f"{self.logdir}/state_step{step:06d}.pt"
        torch.save(log, checkpoint_path)

        # Cleanup old checkpoints
        if self.args.keep_last_n_checkpoints > 0:
            checkpoints = sorted(glob.glob(f"{self.logdir}/state_step*.pt"))
            if len(checkpoints) > self.args.keep_last_n_checkpoints:
                for checkpoint in checkpoints[: -self.args.keep_last_n_checkpoints]:
                    os.remove(checkpoint)

    def save_best_model(self, step: int, val_loss: float):
        """Save model if it has the best validation loss so far."""
        log = {
            "step": step,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        best_model_path = f"{self.logdir}/best_model_{step}.pt"
        torch.save(log, best_model_path)

        # Remove previous best model if exists
        for f in glob.glob(f"{self.logdir}/best_model_*.pt"):
            if f != best_model_path:
                os.remove(f)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[Dict]]:
        """Execute a single training step."""
        self.model.train()
        metrics = None

        for i in range(1, self.train_accumulation_steps + 1):
            with self.ctx:
                _, loss = self.model(x, y)
                if isinstance(loss, dict):
                    metrics = {k: v for k, v in loss.items() if k != "total"}
                    loss = loss["total"]
                train_loss = loss.detach()

            x, y = self.train_loader.next_batch()

            if i < self.train_accumulation_steps:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= self.train_accumulation_steps

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

        return train_loss, metrics

    def train(self):
        """Main training loop."""
        training_time_ms = 0
        best_val_loss = float("inf")
        torch.cuda.synchronize()
        t0 = time.time()

        self.train_loader.reset()
        x, y = self.train_loader.next_batch()

        for step in range(self.args.num_iterations + 1):
            last_step = step == self.args.num_iterations

            # Reset timing after first 10 steps
            if step == 10:
                training_time_ms = 0
                t0 = time.time()

            # Calculate timed steps, accounting for warm-up period
            timed_steps = max(1, step - 10) if step > 10 else 1

            # Validation step
            if last_step or (
                self.args.val_loss_every > 0 and step % self.args.val_loss_every == 0
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)

                val_loss = self.validate()

                if self.is_master:
                    self._log_validation_results(
                        step, val_loss, training_time_ms, timed_steps
                    )
                    if self.args.save_best_model and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_best_model(step, val_loss)

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                t0 = time.time()

            # Checkpoint saving
            if self.is_master and (
                last_step
                or (self.args.save_every > 0 and step % self.args.save_every == 0)
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                self.save_checkpoint(step)
                torch.cuda.synchronize()
                t0 = time.time()

            if last_step:
                break

            # Training step
            train_loss, metrics = self.train_step(x, y)

            if self.is_master:
                self._log_training_progress(
                    step, train_loss, training_time_ms, t0, timed_steps, metrics
                )

        # Report peak memory usage
        if self.is_master:
            print_rank0(
                f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
            )

    def _log_validation_results(
        self, step: int, val_loss: float, training_time_ms: float, timed_steps: float
    ):
        """Log validation results."""
        log_msg = f"step:{step}/{self.args.num_iterations} val_loss:{val_loss:.4f} "
        log_msg += f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms"
        print_rank0(log_msg)
        with open(self.logfile, "a") as f:
            f.write(log_msg + "\n")

    def _log_training_progress(
        self,
        step: int,
        train_loss: torch.Tensor,
        training_time_ms: float,
        t0: float,
        timed_steps: float,
        metrics: Optional[Dict] = None,
    ):
        """Log training progress."""
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        lr = self.optimizer.param_groups[0]["lr"]

        log_msg = f"step:{step+1}/{self.args.num_iterations} lr:{lr} train_loss:{train_loss.item():.4f} "
        log_msg += (
            f"train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
        )

        if metrics:
            log_msg += f" {metrics}"
        print_rank0(log_msg)


def main():
    """Entry point for training."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to model config file")
    parser.add_argument("--model_name", type=str, default="bla_gpt", help="Model architecture to use")

    # Get default hyperparameters and add them to argument parser
    hyperparams = Hyperparameters()

    # Add all hyperparameters to the argument parser
    for field_name, field_info in hyperparams.__dataclass_fields__.items():
        field_type = field_info.type
        default_value = getattr(hyperparams, field_name)

        # Get descriptive help message from metadata
        help_msg = field_info.metadata.get("help", f"Configure {field_name}")
        help_text = f"{help_msg} (default: {default_value})"

        # Handle different field types
        if field_type == bool:
            if default_value:
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
            else:
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
        elif field_type == str:
            parser.add_argument(f"--{field_name}", type=str, default=default_value,
                              help=help_text)
        elif field_type == int:
            parser.add_argument(f"--{field_name}", type=int, default=default_value,
                              help=help_text)
        elif field_type == float:
            parser.add_argument(f"--{field_name}", type=float, default=default_value,
                              help=help_text)
        else:
            # For other types, try to infer from default value
            if isinstance(default_value, bool):
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
            elif isinstance(default_value, str):
                parser.add_argument(f"--{field_name}", type=str, default=default_value,
                                  help=help_text)
            elif isinstance(default_value, int):
                parser.add_argument(f"--{field_name}", type=int, default=default_value,
                                  help=help_text)
            elif isinstance(default_value, float):
                parser.add_argument(f"--{field_name}", type=float, default=default_value,
                                  help=help_text)

    args = parser.parse_args()

    # Update hyperparameters with parsed arguments
    for field_name in hyperparams.__dataclass_fields__.keys():
        if hasattr(args, field_name):
            setattr(hyperparams, field_name, getattr(args, field_name))

    trainer = Trainer(hyperparams, args.model_name, args.config)
    trainer.train()


if __name__ == "__main__":
    main()
