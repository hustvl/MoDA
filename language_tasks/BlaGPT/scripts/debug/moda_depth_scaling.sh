#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# BlaGPT MoDA depth-scaling recipe (64-layer post-norm GPT with depth +
# spatial attention via the MoDA Triton kernel `parallel_moda`, ~1.3B tokens).
#
# Trains the `blagpt_depth_scaling_moda_512t_64l_post_norm_1p3b`
# model registered in `bla_gpt/model_registers.py`. The config
# (`DepthScalingMoDA512T64LPostNorm1p3BConfig` in
# `bla_gpt/bla_gpt_moda_depth_scaling.py`) overrides the in-code defaults
# from `Hyperparameters` in `bla_gpt/train.py` for everything that has a
# matching attribute, so the CLI only needs to pass `--model_name` /
# `--run_name`. The 64L config inherits everything except ``n_layer`` from
# the 108L recipe.
#
# Mirrors the seed recipe at:
#   BlaGPT-seed/bla_gpt/mscripts/depth_scaling_train/
#       train_mergedv2dsav4_512t_12l_flex_post_norm_real_1B3_8node.sh
# (seed uses the FineWeb100B tokenization on a single node x 8 GPUs and the
# `fla.ops.dsa.parallel_dsa` kernel from `mfla`; this debug script trains on
# FineWeb10B with 4 GPUs and uses the equivalent MoDA kernel
# `fla.ops.moda.parallel_moda` from `libs/moda_triton`.)
#
# Architecture (defined in `bla_gpt/bla_gpt_moda_depth_scaling.py`):
#   Each transformer block is `BlockPostNormMoDA` and runs:
#       attn_out, k_list, v_list = attn(x, k_list, v_list)
#       x = ln_1(x + attn_out)
#       mlp_out, k_list, v_list = mlp(x, k_list, v_list)
#       x = ln_2(x + mlp_out)
#   - `attn` is `MixtureOfDepthsAttention`: standard causal SDPA at block 0
#     (cache empty), then `parallel_moda(q, k, v, cached_k=k_list,
#     cached_v=v_list, moda_group_num=n_head/n_kv_head)` from block 1 on.
#   - `mlp` is `SwiGLUMLPDepthScalingMoDA`: standard SwiGLU MLP plus an extra
#     `kv_proj` whose K/V are appended to the depth cache.
#   The cache `(k_list, v_list)` has shape `(B*T, n_kv_head, depth, head_dim)`
#   and grows by +2 (one from attn, one from mlp) per block.
#
# Settings (overrides applied via the model config):
#   * Model      (DepthScalingMoDA512T64LPostNorm1p3BConfig)
#       - n_layer            = 64
#       - n_head             = 8
#       - n_embd             = 512
#       - n_kv_head          = 2                (GQA: 8 query / 2 KV heads)
#       - block_size         = 512
#       - mlp_multiplier     = 2.0              (SwiGLU hidden = 1024)
#       - residual style     = post-norm        (BlockPostNormMoDA)
#       - tie_embed_weights  = True
#       - zero_init_proj_layers = False
#       - rmsnorm_before_qk  = True             (q_norm/k_norm on head_dim)
#       - pos_encoding       = "rotary"         (RoPE on attn Q/K)
#       - parallel_moda kernel resolved eagerly at MixtureOfDepthsAttention init
#       Note: the `attention="regular"` / `activation="swiglu"` placeholders in
#       the config exist only so the base `GPT.__init__` can construct
#       throwaway pre-norm blocks; `MoDADepthScalingGPT.__init__` then
#       replaces `transformer.h` with `BlockPostNormMoDA` instances and
#       wraps `transformer.ln_f` with a tuple-aware norm.
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
# Dependencies:
#   The MoDA Triton kernel package `fla` (from `libs/moda_triton`) must be
#   importable in the conda env. It is already pre-installed in
#   `/mnt/bn/ic-vlm/lianghuizhu/miniconda3/envs/blagpt`. To reinstall:
#       pip install /mnt/bn/ic-vlm/lianghuizhu/MoDA/libs/moda_triton
#
# Run:
#   bash language_tasks/BlaGPT/scripts/debug/moda_depth_scaling.sh
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
MASTER_PORT=49504
MODEL_NAME=blagpt_depth_scaling_moda_512t_64l_post_norm_1p3b
RUN_NAME=moda_depth_scaling_512t_64l_post_norm_1p3b_4gpu

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
