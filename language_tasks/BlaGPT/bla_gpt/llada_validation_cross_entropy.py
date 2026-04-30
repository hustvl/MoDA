"""
Distributed Validation Cross-Entropy Evaluation Script for LLaDA

This script computes the cross-entropy loss of the LLaDA model on the same
FineWeb10B validation dataset used during training, using multi-GPU distributed evaluation.

This script for:
- Multiple choice tasks: Score each option and pick the highest likelihood
- Benchmarking: Compare model performance against other approaches
- Model selection: Choose between different checkpoints
"""

from types import new_class
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
import argparse
import os
import glob
import numpy as np
import datetime
from typing import Optional
from llada import (
    LLaDA,
    LLaDAConfig,
    load_checkpoint
)
from train import DistributedDataLoader, _load_data_shard

# Set up NCCL environment variables for better stability
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"

torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision("high")

def evaluate_cross_entropy_distributed(
    model: DDP,
    val_loader: DistributedDataLoader,
    num_monte_carlo: int = 128,
    val_steps: int = 100,
    device: str = 'cuda',
    master_process: bool = False
) -> dict:
    """Evaluate cross-entropy on validation data using distributed evaluation"""

    model.eval()
    val_loader.reset()

    total_cross_entropy = 0.0
    total_sequences = 0

    if master_process:
        print(f"Evaluating {val_steps} validation steps across {dist.get_world_size()} GPUs")
        print(f"Using {num_monte_carlo} Monte Carlo samples per batch")

    with torch.no_grad():
        for step in range(val_steps):
            # Get next batch from distributed data loader
            x_val, y_val = val_loader.next_batch()
            batch_size = x_val.size(0)

            # Compute cross-entropy for this batch using the model's evaluate_cross_entropy method
            batch_cross_entropy = model.module.evaluate_cross_entropy(
                x_val,  # x_val already contains both input and target
                num_monte_carlo=num_monte_carlo
            )

            # Accumulate locally
            total_cross_entropy += batch_cross_entropy.item() * batch_size
            total_sequences += batch_size

            if master_process and (step + 1) % 10 == 0:
                current_avg = total_cross_entropy / total_sequences
                print(f"  Processed {step + 1}/{val_steps} steps. "
                      f"Current local avg cross-entropy: {current_avg:.4f}")

    # Convert to tensors for distributed reduction
    local_cross_entropy = torch.tensor(total_cross_entropy, device=device)
    local_sequences = torch.tensor(total_sequences, device=device)

    # All-reduce across all processes
    dist.all_reduce(local_cross_entropy, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sequences, op=dist.ReduceOp.SUM)

    # Calculate global averages
    global_avg_cross_entropy = local_cross_entropy.item() / local_sequences.item()
    global_total_sequences = local_sequences.item()

    perplexity = torch.exp(torch.tensor(global_avg_cross_entropy))

    return {
        'cross_entropy': global_avg_cross_entropy,
        'perplexity': perplexity.item(),
        'num_sequences': int(global_total_sequences),
        'tokens_evaluated': int(global_total_sequences * x_val.size(1))
    }

def main():
    parser = argparse.ArgumentParser(description='Distributed LLaDA cross-entropy evaluation on FineWeb10B validation data')
    parser.add_argument('--model_size', choices=['1b', '8b'], default='1b',
                       help='Model size (1b or 8b)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (required for meaningful evaluation)')
    parser.add_argument('--val_pattern', type=str, default="../data/fineweb10B/fineweb_val_*.bin",
                       help='Pattern for validation files')
    parser.add_argument('--val_tokens', type=int, default=10485760,
                       help='Number of validation tokens to evaluate. Default is set according to the default training dataset.')
    parser.add_argument('--seq_len', type=int, default=1024,
                       help='Sequence length (should match training)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Per-device batch size for evaluation')
    parser.add_argument('--num_monte_carlo', type=int, default=128,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results (optional)')
    parser.add_argument('--compile_model', action='store_true',
                       help='Compile the model with torch.compile')

    args = parser.parse_args()

    # Set up distributed training - this must be done before any CUDA operations
    assert torch.cuda.is_available(), "CUDA is required for distributed evaluation"

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=30)
    )

    # Get distributed training parameters from environment
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

    master_process = ddp_rank == 0  # only rank 0 will do logging and saving

    if master_process:
        print("=" * 60)
        print("LLaDA Distributed Cross-Entropy Evaluation on FineWeb10B")
        print("=" * 60)
        print(f"Using {ddp_world_size} GPUs")
        print(f"Device: {device}")

    # Initialize tokenizer (same as training)
    tokenizer = tiktoken.get_encoding("gpt2")
    if master_process:
        print(f"Using GPT-2 tokenizer with vocab size: {tokenizer.n_vocab}")

    # Load model configuration and create model
    config = LLaDAConfig()
    model = LLaDA.from_config(config)

    if args.checkpoint:
        if master_process:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        new_state_dict = {}
        for key in checkpoint['model']:
            new_state_dict[key.replace('_orig_mod.', '')] = checkpoint['model'][key]
        model.load_state_dict(new_state_dict)

    model = model.to(device)

    if args.compile_model:
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    if master_process:
        print(f"Model parameters: {total_params:,}")

    # Wrap model in DDP
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    # Set up distributed data loader
    B, T = args.batch_size, args.seq_len

    # Calculate total validation tokens that will be processed
    val_steps = args.val_tokens // (B * T * ddp_world_size)

    val_loader = DistributedDataLoader(
        args.val_pattern, B, T, ddp_rank, ddp_world_size
    )

    if master_process:
        print(f"Validation DataLoader: total tokens available: {val_loader.ntok_total:,} across {len(val_loader.files)} files")
        print(f"Will evaluate {args.val_tokens:,} tokens ({val_steps} steps × {B} batch size × {T} seq len × {ddp_world_size} GPUs)")
        print(f"Sequence length: {args.seq_len}")
        print(f"Per-device batch size: {args.batch_size}")
        print(f"Monte Carlo samples: {args.num_monte_carlo}")

    # Run distributed cross-entropy evaluation
    if master_process:
        print(f"\nStarting distributed cross-entropy evaluation...")

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    results = evaluate_cross_entropy_distributed(
        model=model,
        val_loader=val_loader,
        num_monte_carlo=args.num_monte_carlo,
        val_steps=val_steps,
        device=device,
        master_process=master_process
    )

    end_time.record()
    torch.cuda.synchronize()
    eval_time_ms = start_time.elapsed_time(end_time)

    # Display results (only on master process)
    if master_process:
        print("\n" + "=" * 60)
        print("DISTRIBUTED VALIDATION RESULTS")
        print("=" * 60)
        print(f"Dataset:              FineWeb10B validation")
        print(f"Cross-entropy loss:   {results['cross_entropy']:.6f}")
        print(f"Perplexity:          {results['perplexity']:.2f}")
        print(f"Model size:          {args.model_size}")
        print(f"Sequences evaluated: {results['num_sequences']:,}")
        print(f"Tokens evaluated:    {results['tokens_evaluated']:,}")
        print(f"Sequence length:     {args.seq_len}")
        print(f"Per-device batch:    {args.batch_size}")
        print(f"Total GPUs:          {ddp_world_size}")
        print(f"Monte Carlo samples: {args.num_monte_carlo}")
        print(f"Evaluation time:     {eval_time_ms/1000:.2f}s")
        if args.checkpoint:
            print(f"Checkpoint:          {os.path.basename(args.checkpoint)}")
        else:
            print(f"Model state:         Random initialization")
        print("=" * 60)

        # Save results if requested
        if args.save_results:
            save_data = {
                'cross_entropy': results['cross_entropy'],
                'perplexity': results['perplexity'],
                'model_size': args.model_size,
                'num_sequences': results['num_sequences'],
                'tokens_evaluated': results['tokens_evaluated'],
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'num_gpus': ddp_world_size,
                'num_monte_carlo': args.num_monte_carlo,
                'eval_time_ms': eval_time_ms,
                'checkpoint': args.checkpoint,
                'val_pattern': args.val_pattern,
                'dataset': 'FineWeb10B',
                'distributed': True
            }
            torch.save(save_data, args.save_results)
            print(f"\nResults saved to: {args.save_results}")

    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
