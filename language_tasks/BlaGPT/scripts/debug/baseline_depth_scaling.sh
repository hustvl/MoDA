#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# BlaGPT depth-scaling baseline (64-layer post-norm SwiGLU GPT, ~1.3B tokens).
#
# Trains the `blagpt_depth_scaling_baseline_512t_64l_post_norm_1p3b` model
# registered in `bla_gpt/model_registers.py`. The config
# (`DepthScalingBaseline512T64LPostNorm1p3BConfig` in
# `bla_gpt/bla_gpt_depth_scaling.py`) overrides the in-code defaults from
# `Hyperparameters` in `bla_gpt/train.py` for everything that has a matching
# attribute, so the CLI only needs to pass `--model_name` / `--run_name`.
# The 64L config inherits everything except ``n_layer`` from the 108L recipe.
#
# Mirrors the seed recipe at:
#   BlaGPT-seed/bla_gpt/mscripts/depth_scaling_train/
#       train_baseline_512t_12l_flex_post_norm_real_1B3_8node.sh
# (seed uses the FineWeb100B tokenization on 8 nodes x 8 GPUs; this debug
# script trains on FineWeb10B with 4 GPUs and the same per-step token budget.)
#
# Settings (overrides applied via the model config):
#   * Model      (DepthScalingBaseline512T64LPostNorm1p3BConfig)
#       - n_layer            = 64
#       - n_head             = 8
#       - n_embd             = 512
#       - n_kv_head          = 2                (GQA: 8 query / 2 KV heads)
#       - block_size         = 512
#       - mlp_multiplier     = 2.0              (SwiGLU hidden = 1024)
#       - attention          = "regular"        (full causal SDPA, RoPE)
#       - activation         = "swiglu"         (depth-scaling SwiGLU MLP)
#       - residual style     = post-norm        (BlockPostNorm)
#       - tie_embed_weights  = True
#       - zero_init_proj_layers = False
#   * Optimizer  (model config overrides Hyperparameters via GPTConfig defaults)
#       - optimizer_name     = "Muon"           (matrix params via Muon; bias/norm/
#                                                embed_tokens/lm_head fall back to
#                                                Muon's internal AdamW)
#       - optimizer_args     = {"betas": (0.9, 0.95), "eps": 1e-8,
#                               "weight_decay": 0.0}
#         (NOTE: `weight_decay` is consumed by Muon.__init__'s **kwargs and is
#          NOT applied; the internal AdamW uses Muon's `wd=0.1` default. This
#          mirrors the seed BlaGPT recipe exactly.)
#       - learning_rate      = 1e-3             (Hyperparameters default)
#       - schedule           = linear warmup 250 -> constant -> linear warmdown 2000
#   * Training (model config overrides)
#       - num_iterations     = 5100             (~1.3B tokens)
#       - batch_size         = 512 sequences (global, Hyperparameters default)
#       - device_batch_size  = 8 sequences/GPU
#         => grad_accum_steps = 512 / (8 * NPROC) = 16 with 4 GPUs
#       - sequence_length    = 512 tokens
#       - compile_model      = True             (torch.compile, ~minutes warmup)
#       - precision          = bf16 autocast
#   * Eval / logging (Hyperparameters defaults; not overridden)
#       - val_loss_every     = 125 steps
#       - val_tokens         = 10,485,760
#       - save_every         = 5000 steps
#       - keep_last_n_checkpoints = 1
#       - save_best_model    = True
#   * Data (FineWeb10B GPT-2 tokens, kjj0/fineweb10B-gpt2)
#       - input_bin          = ../data/fineweb10B/fineweb_train_*.bin
#       - input_val_bin      = ../data/fineweb10B/fineweb_val_*.bin
#
# Run:
#   bash language_tasks/BlaGPT/scripts/debug/baseline_depth_scaling.sh
#
# Logs/checkpoints land in `language_tasks/BlaGPT/bla_gpt/logs/<run_name>_<n>/`.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BLAGPT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_DIR="${BLAGPT_DIR}/bla_gpt"

PYTHON_BIN=/mnt/bn/ic-vlm/lianghuizhu/miniconda3/envs/blagpt/bin/python
DATA_SRC=/mnt/bn/ic-vlm/lianghuizhu/BlaGPT-seed/data/fineweb10B
DATA_LINK="${BLAGPT_DIR}/data/fineweb10B"

NPROC=4
MASTER_PORT=49503
MODEL_NAME=blagpt_depth_scaling_baseline_512t_64l_post_norm_1p3b
RUN_NAME=baseline_depth_scaling_512t_64l_post_norm_1p3b_4gpu

# Ensure the FineWeb10B shards are available where train.py expects them
# (../data/fineweb10B/ relative to bla_gpt/). We symlink to the existing
# pre-tokenized copy on /mnt/bn/ic-vlm to avoid re-downloading from HF.
if [ ! -e "${DATA_LINK}" ]; then
    ln -snf "${DATA_SRC}" "${DATA_LINK}"
fi

cd "${TRAIN_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NPROC} \
    --master_port ${MASTER_PORT} \
    train.py \
    --model_name "${MODEL_NAME}" \
    --run_name "${RUN_NAME}"
