#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# BlaGPT baseline (vanilla GPT-2 124M-style) on FineWeb10B.
#
# Trains the `blagpt` model registered in `bla_gpt/model_registers.py`
# (`GPTConfig` in `bla_gpt/bla_gpt.py`). The CLI does not pass any training
# overrides; the effective config is `Hyperparameters` (in `bla_gpt/train.py`)
# with `optimizer_name` / `optimizer_args` overridden by `GPTConfig` via
# train.py's `for k,v in model_config.to_dict(): setattr(args, k, v)` loop.
#
# Settings (model config overrides where noted; all other fields are the
# in-code defaults from `Hyperparameters` and are not touched by the CLI):
#   * Model      (GPTConfig)
#       - n_layer            = 12
#       - n_head             = 12
#       - n_embd             = 768
#       - block_size         = 1024
#       - vocab_size         = 50304            (GPT-2 50257 padded to /64)
#       - attention          = "regular"        (full causal attention, RoPE)
#       - activation         = "polynorm"
#       - norm_layer         = "rmsnorm"
#       - n_kv_head          = 4                (GQA: 12 query heads / 4 KV heads)
#       - tie_embed_weights  = True
#       - zero_init_proj_layers = True
#   * Optimizer  (model config overrides Hyperparameters via GPTConfig defaults)
#       - optimizer_name     = "Muon"           (matrix params via Muon; bias/norm/
#                                                embed_tokens/lm_head fall back to
#                                                Muon's internal AdamW)
#       - optimizer_args     = {"betas": (0.9, 0.95), "eps": 1e-8,
#                               "weight_decay": 0.0}
#         (NOTE: `weight_decay` is consumed by Muon.__init__'s **kwargs and is
#          NOT applied; the internal AdamW uses Muon's `wd=0.1` default.)
#       - learning_rate      = 1e-3             (Hyperparameters default)
#       - schedule           = linear warmup 250 -> constant -> linear warmdown 2000
#   * Training (Hyperparameters defaults; not overridden)
#       - num_iterations     = 5100             (~2.67B tokens)
#       - batch_size         = 512 sequences (global)
#       - device_batch_size  = 32  sequences/GPU
#         => grad_accum_steps = 512 / (32 * NPROC) = 4 with 4 GPUs (2 with 8 GPUs)
#       - sequence_length    = 1024 tokens
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
#   bash language_tasks/BlaGPT/scripts/debug/baseline.sh
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
MASTER_PORT=49502
MODEL_NAME=blagpt
RUN_NAME=baseline_blagpt_4gpu

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
