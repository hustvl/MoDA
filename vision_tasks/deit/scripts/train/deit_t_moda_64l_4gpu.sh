#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DeiT-Tiny MoDA depth-scaling recipe (64-layer MoDA ViT) on ImageNet-1k.
#
# Trains the `deit_tiny_moda_64l_patch16_224` model registered in
# `vision_tasks/deit/models.py`. The factory mirrors
# `deit_tiny_moda_patch16_224` (same ``embed_dim=256``, ``num_heads=4``,
# ``num_kv_heads=1``, ``mlp_ratio=4``, ``patch_size=16``, ``img_size=224``,
# default ``moda_backend="v17"`` and ``mlp_depth_kv_projection=True``) but
# stacks 64 transformer blocks. The depth-attention cache therefore grows
# to ``2 * depth = 128`` slots per token (dual-slot-per-layer BlaGPT
# layout; one slot from attention, one from MLP ``kv_proj``) and the
# ``parallel_moda_v17`` Triton kernel is the main reason for choosing the
# 64L depth: deeper stacks let the ``parallel_moda`` K1 kernel attend over
# up to 127 historical rows per block, which is what we're trying to
# evaluate.
#
# Companion BlaGPT MoDA depth-scaling recipe for reference:
#   language_tasks/BlaGPT/scripts/debug/moda_depth_scaling.sh
#   (64-layer post-norm GPT-2 MoDA baseline using the same
#   ``fla.ops.moda.parallel_moda`` kernel family from ``libs/moda_triton``.)
#
# Settings (CLI flags + main.py / timm defaults):
#   * Model      (deit_tiny_moda_64l_patch16_224)
#       - embed_dim              = 256
#       - num_heads              = 4          (MoDA queries)
#       - num_kv_heads           = 1          (MoDA KV, 4:1 query-to-KV ratio)
#       - depth                  = 64         (vs. 12 for deit_tiny_moda_patch16_224)
#       - mlp_ratio              = 4
#       - patch_size / img_size  = 16 / 224
#       - drop_path_rate         = 0.1        (main.py default, see note below)
#       - moda_backend           = "v17"      (default: parallel_moda_v17 + K1 kernel,
#                                              pre-allocated buffer, O(slots) cache build)
#       - mlp_depth_kv_projection= True       (default: dual-slot-per-layer cache)
#         -> cache depth = 128 slots (2 * 64) per token; block i reads slots 0..2i-1
#   * Optimizer  (main.py defaults, unchanged)
#       - opt                = adamw
#       - lr                 = 5e-4 (linearly scaled by batch_size*world/512)
#       - weight_decay       = 0.05
#       - warmup_epochs      = 5
#       - epochs             = 300 (cosine schedule)
#   * Training
#       - batch-size         = 256 sequences / GPU -> 1024 global on 4 GPUs
#         (MoDA with a 128-slot depth cache allocates a
#          ``[B, N*128, num_kv_heads, head_dim]`` activation buffer per
#          forward pass. If this OOMs, halve to ``--batch-size 128`` or
#          64, or switch to the single-slot layout with
#          ``--no-mlp-depth-kv-projection`` to drop the cache to 64 slots.)
#       - mixup / cutmix / RandAugment / random-erase -> main.py defaults
#       - precision          = fp32 (main.py does not enable amp/bf16)
#   * Data (ImageNet-1k)
#       - Resolved at runtime from the available storage mount:
#           /mnt/bn/ic-vlm-hl/public/cv_task/Imagenet1k/   (preferred)
#         or fallback /mnt/bn/ic-vlm/zilonghuang/Imagenet1k.
#
# Dependencies:
#   The MoDA Triton kernel package ``fla`` (from ``libs/moda_triton``) must
#   be importable. This script installs it via ``pip3 install .`` in the
#   setup block below. Triton 3.6.0 on Hopper (sm90) hits an MLIR pass
#   crash compiling ``parallel_attn_bwd_kernel_dkv_depth``, so we pin to
#   triton==3.3.0 as in ``legacy/deit_t_moda_4gpu.sh``.
#
# Note on drop_path_rate: with 64 blocks the classical DeiT default 0.1
# may be too mild to prevent saturation. DeiT-III / ViT-L-scale recipes
# typically use 0.45-0.55 for this depth range. Try adding
# ``--drop-path 0.5`` to the torchrun line below if training plateaus.
#
# Run:
#   bash vision_tasks/deit/scripts/train/deit_t_moda_64l_4gpu.sh
#
# Logs / checkpoints land in ``${OUTPUT_ROOT}/${MODEL_NAME}/``.
# -----------------------------------------------------------------------------

cd /opt/tiger/MoDA/vision_tasks/deit
pip3 install -r requirements.txt
pip3 install ipdb

cd /opt/tiger/MoDA/libs/moda_triton
pip3 install .

cd /opt/tiger/MoDA/vision_tasks/deit

# triton 3.6.0 on Hopper (sm90) GPU, when compiling the parallel_attn_bwd_kernel_dkv_depth kernel, the MLIR pass pipeline internally crashes.
# [fix] we downgrade to triton 3.3.0 to avoid the issue.
pip3 install triton==3.3.0

set -euo pipefail

# check the available storage mount in order
if [ -d "/mnt/bn/ic-vlm-hl" ]; then
    DATA_PATH=/mnt/bn/ic-vlm-hl/public/cv_task/Imagenet1k/
    OUTPUT_ROOT=/mnt/bn/ic-vlm-hl/personal/lianghuizhu/deit_output_dir
elif [ -d "/mnt/bn/ic-vlm" ]; then
    DATA_PATH=/mnt/bn/ic-vlm/zilonghuang/Imagenet1k
    OUTPUT_ROOT=/mnt/bn/ic-vlm/personal/lianghuizhu/deit_output_dir
else
    echo "Error: neither /mnt/bn/ic-vlm-hl nor /mnt/bn/ic-vlm exists." >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEIT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=python3
MODEL_NAME=deit_tiny_moda_64l_patch16_224
OUTPUT_DIR=${OUTPUT_ROOT}/${MODEL_NAME}


cd "${DEIT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 --master_port 49513 main.py --model "${MODEL_NAME}" --batch-size 256 --data-path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
