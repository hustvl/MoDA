import datetime
import os
import sys

from coqpit import Coqpit
from optimizers import get_optimizer
from utils import get_model

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import glob
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"

torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision("high")


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# Hyperparameters


@dataclass
class Hyperparameters(Coqpit):
    run_name: str = "nano_gpt+rms_norm+geglu+gqa+softcap"
    compile_model: bool = True
    # data hyperparams
    input_bin: str = "../data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    input_val_bin: str = (
        "../data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    )

    # training
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 32  # batch size, in sequences, per device, grad_accum_steps = batch_size // num_devices /// device_batch_size
    sequence_length: int = 1024  # sequence length, in tokens
    num_iterations: int = 5100  # number of iterations to run

    # optimizer
    optimizer_name: str = "Adam"
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "use_cautious_weight_decay": False,
        }
    )
    learning_rate: float = 0.001
    warmup_iters: int = 250
    warmdown_iters: int = 2000  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule

    # evaluation and logging hyperparams
    val_loss_every: int = (
        125  # every how many steps to evaluate val loss? 0 for only at the end
    )
    val_tokens: int = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every: int = (
        5000  # every how many steps to save the checkpoint? 0 for only at the end
    )

    # checkpoint params
    keep_last_n_checkpoints: int = 1  # number of checkpoints to keep
    save_best_model: bool = True  # whether to save best model based on val loss


# -----------------------------------------------------------------------------
# int main

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bla_gpt")
    cli_args = parser.parse_args()

    args = Hyperparameters()
    model_config, model = get_model(cli_args.model_name)

    # Override Hyperparameters with matching model_config attributes
    for key, value in model_config.to_dict().items():
        if hasattr(args, key):
            setattr(args, key, value)

    if cli_args.run_name:
        args.run_name = cli_args.run_name

    if cli_args.config:
        model_config.load_json(cli_args.config)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30)
    )
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)
    if master_process:
        print(f"Accumulation steps: {train_accumulation_steps}")

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(
        args.input_val_bin, B, T, ddp_rank, ddp_world_size
    )
    if master_process:
        print(
            f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
        )
        print(
            f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files"
        )
    x, y = train_loader.next_batch()

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    torch.cuda.empty_cache()
    model = model(model_config)
    model = model.cuda()

    if args.compile_model:
        model = torch.compile(model)

    model_size = (
        sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    )  # size in MB

    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if master_process:
        print(f"Model size: {model_size:.2f} MB")
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of trainable parameters: {num_trainable_parameters}")

    # here we wrap model into DDP container
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    raw_model = model.module  # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # init the optimizer(s)
    # foreach == false to save VRAM

    optimizer1 = get_optimizer(
        args.optimizer_name, args.optimizer_args, args.learning_rate, raw_model
    )

    optimizers = [
        optimizer1,
    ]

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # begin logging
    if master_process:
        run_num = 0
        run_id = f"{args.run_name}_{run_num}"
        logdir = "logs/%s/" % run_id
        while os.path.exists(logdir):
            run_num += 1
            run_id = f"{args.run_name}_{run_num}"
            logdir = "logs/%s/" % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = "logs/%s.txt" % run_id
        print(f"Logging run in {logdir}")
        # create the log file and set up TeeLogger
        sys.stdout = TeeLogger(logfile)
        # begin the log by printing this file (the Python code)
        print("=" * 100)
        print(code)
        print("=" * 100)
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        print(
            f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:"
        )
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"{result.stdout}")
        print("=" * 100)

    training_time_ms = 0
    best_val_loss = float("inf")
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = (
            float("nan") if step <= 11 else (step - 10) + 1
        )  # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    with ctx:  # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                        _, loss = model(x_val, y_val)
                        metrics = None
                        if type(loss) is dict:
                            metrics = {k: v for k, v in loss.items() if k != "total"}
                            loss = loss["total"]
                        val_loss += loss.detach()
                        del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                if metrics is None:
                    print(
                        f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms"
                    )
                else:
                    print(
                        f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms {metrics}"
                    )
                with open(logfile, "a") as f:
                    f.write(
                        f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n"
                    )
            # start the clock again
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            t0 = time.time()

            # Save best model if validation loss improved
            if master_process and (args.save_best_model and val_loss < best_val_loss):
                best_val_loss = val_loss
                log = dict(
                    step=step,
                    code=code,
                    model_config=model_config.to_dict(),
                    train_config=args.to_dict(),
                    model=raw_model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                    val_loss=val_loss,
                )
                best_model_path = f"logs/{run_id}/best_model_{step}.pt"
                torch.save(log, best_model_path)

                # Remove previous best model if exists
                for f in glob.glob(f"logs/{run_id}/best_model_*.pt"):
                    if f != best_model_path:
                        os.remove(f)

        if master_process and (
            last_step or (args.save_every > 0 and step % args.save_every == 0)
        ):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(
                step=step,
                code=code,
                model_config=model_config.to_dict(),
                train_config=args.to_dict(),
                model=raw_model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            torch.save(log, "logs/%s/state_step%06d.pt" % (run_id, step))

            # Cleanup old checkpoints
            if args.keep_last_n_checkpoints > 0:
                checkpoints = sorted(glob.glob(f"logs/{run_id}/state_step*.pt"))
                if len(checkpoints) > args.keep_last_n_checkpoints:
                    for checkpoint in checkpoints[: -args.keep_last_n_checkpoints]:
                        os.remove(checkpoint)

            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                _, loss = model(x, y)
                metrics = None
                if type(loss) is dict:
                    metrics = {k: v for k, v in loss.items() if k != "total"}
                    loss = loss["total"]
                train_loss = loss.detach()

            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with (
                    model.no_sync()
                ):  # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward()  # just sync on the last step
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= train_accumulation_steps
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            lr = optimizers[0].param_groups[0]["lr"]
            if metrics is None:
                print(
                    f"step:{step+1}/{args.num_iterations} lr:{lr} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
                )
            else:
                print(
                    f"step:{step+1}/{args.num_iterations} lr:{lr} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms {metrics}"
                )

    if master_process:
        print(
            f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()
