import itertools
import json
import os
import subprocess

from bla_gpt import GPTConfig


def run_training(config, run_name):
    # Save the configuration to a temporary file
    config_file = "temp_config.json"
    with open(config_file, "w") as f:
        json.dump(config.__dict__, f)

    # Prepare the command to run the training script
    command = [
        "torchrun",
        "--nproc_per_node=8",
        "train.py",
        f"--config={config_file}",
        f"--run_name={run_name}",
    ]

    # Run the training script and capture the output
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output, error = process.communicate()

    # Check if the process exited with an error
    if process.returncode != 0:
        print(f"Error occurred during training for run: {run_name}")
        print("Error output:")
        print(error)
        print("Standard output:")
        print(output)
        return None, None, None

    # Parse the output to extract relevant metrics
    val_loss = None
    memory_usage = None
    step_time = None

    for line in output.split("\n"):
        if "val_loss:" in line:
            val_loss = float(line.split("val_loss:")[1].split()[0])
        if "peak memory consumption:" in line:
            memory_usage = int(line.split(":")[1].split()[0])
        if "step_avg:" in line:
            try:
                step_time = float(line.split("step_avg:")[1].split()[0])
            except ValueError:
                pass

    # Clean up the temporary config file
    os.remove(config_file)

    return val_loss, memory_usage, step_time


def hyperparameter_search():
    # Define the hyperparameter search space
    search_space = {
        "norm_layer": [
            "rmsnorm",
        ],
        "attention": [
            "GQA",
        ],
        "activation": [
            "geglu",
            "swiglu",
        ],
        "tie_embed_weights": [
            True,
        ],
        "zero_init_proj_layers": [
            True,
        ],
        # "use_soft_logit_capping": [True, False], # when true OOM
        "use_rotary_emb": [
            True,
        ],
        "rmsnorm_before_qk": [
            True,
        ],
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*search_space.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    for config_dict in configurations:
        config = GPTConfig(**config_dict)

        run_name = (
            f"run_{"_".join([f"{key}_{value}" for key, value in config_dict.items()])}"
        )

        print(f"Training with configuration: {config_dict}")

        val_loss, memory_usage, step_time = run_training(config, run_name)

        results.append(
            {
                "params": config_dict,
                "config": config.to_dict(),
                "val_loss": val_loss,
                "memory_usage": memory_usage,
                "step_time": step_time,
            }
        )

        print(
            f"Results: val_loss={val_loss}, memory_usage={memory_usage}MB, step_time={step_time}ms"
        )
        print("-" * 50)

    # Sort results by validation loss
    results.sort(key=lambda x: x["val_loss"])

    # Save results to a file
    with open("hyperparameter_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        "Hyperparameter search completed. Results saved to hyperparameter_search_results.json"
    )


if __name__ == "__main__":
    hyperparameter_search()
