import datetime
import os
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from matplotlib import pylab as plt
from optimizers import get_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from train import DistributedDataLoader, Hyperparameters, get_model


class LRFinder:
    """1. Learning Rate Range Test (LR Finder)
    •	Proposed by Leslie Smith in the “Cyclical Learning Rates for Training Neural Networks” paper.
    •	Start with a very small LR (e.g., 10^{-7}) and gradually increase it exponentially over a few thousand iterations.
    •	Plot the loss against the LR and identify the largest LR before the loss starts increasing sharply.
    •	The best LR is typically an order of magnitude lower than the LR where the loss starts diverging.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        start_lr=1e-7,
        end_lr=10,
        num_iterations=100,
        smooth_f=0.05,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = num_iterations
        self.smooth_f = smooth_f

        # Initialize tracking variables
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory = {}

    def range_test(self, train_loader, accumulation_steps=1):
        # Save the original state of the model and optimizer
        self.model.train()
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())

        # Calculate the multiplier for learning rate increase
        self.mult = (self.end_lr / self.start_lr) ** (1 / self.num_iterations)

        # Set initial learning rate
        self.optimizer.param_groups[0]["lr"] = self.start_lr

        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        is_finite = True

        while batch_num < self.num_iterations and is_finite:
            batch_num += 1
            accumulation_loss = 0

            # Training loop with gradient accumulation
            for i in range(accumulation_steps):
                x, y = train_loader.next_batch()

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                    if isinstance(loss, dict):
                        loss = loss["total"]

                loss = loss / accumulation_steps
                loss.backward()
                accumulation_loss += loss.item()

            # Update weights and learning rate
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Update loss
            avg_loss = (
                self.smooth_f * accumulation_loss + (1 - self.smooth_f) * avg_loss
                if batch_num > 1
                else accumulation_loss
            )
            self.history["loss"].append(avg_loss)

            # Check if the loss is getting too high
            if batch_num > 1 and avg_loss > 4 * best_loss:
                break

            if avg_loss < best_loss or batch_num == 1:
                best_loss = avg_loss

            # Update learning rate
            self.optimizer.param_groups[0]["lr"] = current_lr * self.mult

            if not np.isfinite(avg_loss):
                is_finite = False
                break

        # Restore the original state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

        return self.history

    def plot(self, skip_start=10, skip_end=5):
        """
        Plots the learning rate range test results.
        """
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary
        lrs = self.history["lr"]
        losses = self.history["loss"]

        # Plot loss as a function of the learning rate
        plt.figure(figsize=(10, 6))
        plt.plot(lrs[skip_start:-skip_end], losses[skip_start:-skip_end])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Learning rate range test")
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Save the plot
        plt.savefig("lr_finder_plot.png")
        plt.close()


def find_lr():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to test"
    )
    parser.add_argument(
        "--start_lr", type=float, default=1e-7, help="Starting learning rate"
    )
    parser.add_argument("--end_lr", type=float, default=10, help="Ending learning rate")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for the test",
    )
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30)
    )

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # Get model and config
    train_args = Hyperparameters()
    model_config, model = get_model(args.model_name)
    model_config = model_config()

    # Override config if provided
    if args.config:
        model_config.load_json(args.config)

    if args.run_name:
        train_args.run_name = args.run_name

    # Create output directory
    if master_process:
        run_name = f"lr_finder_{args.model_name}"
        if args.run_name:
            run_name = f"{run_name}_{args.run_name}"

        os.makedirs(f"logs/{run_name}", exist_ok=True)

    # Initialize model
    model = model(model_config).cuda()
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    if master_process:
        print(f"Model: {args.model_name}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Initialize optimizer
    optimizer = get_optimizer(
        train_args.optimizer_name,
        train_args.optimizer_args,
        train_args.learning_rate,
        model.module,
    )

    # Initialize data loader
    train_loader = DistributedDataLoader(
        train_args.input_bin,
        train_args.device_batch_size,
        train_args.sequence_length,
        ddp_rank,
        ddp_world_size,
    )

    if master_process:
        print("Starting learning rate finder test...")
        print(f"Start LR: {args.start_lr}")
        print(f"End LR: {args.end_lr}")
        print(f"Number of iterations: {args.num_iterations}")

    # Initialize LR Finder
    lr_finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=None,  # Not needed as loss is computed in the model forward pass
        device=device,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iterations=args.num_iterations,
    )

    # Run the range test
    history = lr_finder.range_test(
        train_loader,
        accumulation_steps=train_args.batch_size
        // (train_args.device_batch_size * ddp_world_size),
    )

    if master_process:
        # Save the plot
        plt.figure(figsize=(10, 6))
        plt.plot(history["lr"][10:-5], history["loss"][10:-5])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title(f"Learning rate range test for {args.model_name}")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(f"logs/{run_name}/lr_finder_plot.png")
        plt.close()

        # Find the learning rate with the steepest negative gradient
        losses = np.array(history["loss"])
        lrs = np.array(history["lr"])
        gradients = np.gradient(losses)
        steepest_idx = np.argmin(gradients)
        steepest_lr = lrs[steepest_idx]

        # Find the elbow point (point of maximum curvature)
        # Calculate the angle between consecutive segments
        dx = np.diff(np.log(lrs))  # Use log scale for x-axis
        dy = np.diff(losses)
        angles = np.unwrap(np.arctan2(dy, dx))

        # Calculate the curvature as the gradient of angles
        curvature = np.gradient(angles)
        # Find the point of maximum curvature (excluding boundaries)
        elbow_idx = np.argmax(np.abs(curvature[10:-5])) + 10
        elbow_lr = lrs[elbow_idx]
        suggested_lr = elbow_lr / 10

        print("\nResults:")
        print(f"Steepest gradient learning rate: {steepest_lr:.2e}")
        print(f"Elbow point learning rate: {elbow_lr:.2e}")
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        print(f"Plot saved to: logs/{run_name}/lr_finder_plot.png")

        # Mark both points on the plot
        plt.figure(figsize=(10, 6))
        plt.plot(history["lr"][10:-5], history["loss"][10:-5], label="Loss curve")
        plt.scatter(
            steepest_lr,
            losses[steepest_idx],
            color="red",
            marker="o",
            label="Steepest point",
            zorder=5,
        )
        plt.scatter(
            elbow_lr,
            losses[elbow_idx],
            color="green",
            marker="o",
            label="Elbow point",
            zorder=5,
        )
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title(f"Learning rate range test for {args.model_name}")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig(f"logs/{run_name}/lr_finder_plot_with_points.png")
        plt.close()

        # Save the results
        results = {
            "model_name": args.model_name,
            "learning_rates": lrs,
            "losses": losses,
            "suggested_lr": suggested_lr,
            "elbow_lr": elbow_lr,
            "steepest_lr": steepest_lr,
            "model_config": model_config.to_dict(),
        }
        torch.save(results, f"logs/{run_name}/lr_finder_results.pt")
        print(f"Results saved to: logs/{run_name}/lr_finder_results.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    find_lr()
