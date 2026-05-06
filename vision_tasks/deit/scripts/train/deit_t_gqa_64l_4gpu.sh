#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DeiT-Tiny GQA depth-scaling baseline (64-layer GQA ViT) on ImageNet-1k.
#
# Trains the `deit_tiny_gqa_64l_patch16_224` model registered in
# `vision_tasks/deit/models.py`. The factory mirrors
# `deit_tiny_gqa_patch16_224` (same ``embed_dim=256``, ``num_heads=4``,
# ``num_kv_heads=1``, ``mlp_ratio=4``, ``patch_size=16``, ``img_size=224``)
# but stacks 64 transformer blocks instead of 12. This run is the
# apples-to-apples GQA counterpart to the full-attention baseline in
# ``deit_t_64l_4gpu.sh`` and to the MoDA run in
# ``deit_t_moda_64l_4gpu.sh``: width / heads / KV heads / patching /
# augmentations / optimizer are all identical -- depth is the only knob.
#
# Settings (CLI flags + main.py / timm defaults):
#   * Model      (deit_tiny_gqa_64l_patch16_224)
#       - embed_dim          = 256
#       - num_heads          = 4          (GQA queries)
#       - num_kv_heads       = 1          (GQA KV, 4:1 query-to-KV ratio)
#       - depth              = 64         (vs. 12 for deit_tiny_gqa_patch16_224)
#       - mlp_ratio          = 4
#       - patch_size / img_size = 16 / 224
#       - drop_path_rate     = 0.1        (main.py default, see note below)
#   * Optimizer  (main.py defaults, unchanged)
#       - opt                = adamw
#       - lr                 = 5e-4 (linearly scaled by batch_size*world/512)
#       - weight_decay       = 0.05
#       - warmup_epochs      = 5
#       - epochs             = 300 (cosine schedule)
#   * Training
#       - batch-size         = 256 sequences / GPU -> 1024 global on 4 GPUs
#         (if the 64L model OOMs, halve this -- ``--batch-size 128`` gives
#         the classical 512 global batch DeiT recipe)
#       - mixup / cutmix / RandAugment / random-erase -> main.py defaults
#       - precision          = fp32 (main.py does not enable amp/bf16)
#   * Data (ImageNet-1k)
#       - Resolved at runtime from the available storage mount:
#           /mnt/bn/ic-vlm-hl/public/cv_task/Imagenet1k/   (preferred)
#         or fallback /mnt/bn/ic-vlm/zilonghuang/Imagenet1k.
#
# Note on drop_path_rate: with 64 blocks the classical DeiT default 0.1
# may be too mild to prevent saturation. DeiT-III / ViT-L-scale recipes
# typically use 0.45-0.55 for this depth range. Try adding
# ``--drop-path 0.5`` to the torchrun line below if training plateaus.
#
# Run:
#   bash vision_tasks/deit/scripts/train/deit_t_gqa_64l_4gpu.sh
#
# Logs / checkpoints land in ``${OUTPUT_ROOT}/${MODEL_NAME}/``.
# -----------------------------------------------------------------------------

cd /opt/tiger/MoDA/vision_tasks/deit
pip3 install -r requirements.txt
pip3 install ipdb

cd /opt/tiger/MoDA/libs/moda_triton
pip3 install .

cd /opt/tiger/MoDA/vision_tasks/deit

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
MODEL_NAME=deit_tiny_gqa_64l_patch16_224
OUTPUT_DIR=${OUTPUT_ROOT}/${MODEL_NAME}

cd "${DEIT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 --master_port 49512 main.py --model "${MODEL_NAME}" --batch-size 256 --data-path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
