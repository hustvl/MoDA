#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DeiT-Tiny GQA depth-scaling baseline (64-layer GQA ViT) -- LOCAL DEBUG
# variant of ``scripts/train/deit_t_gqa_64l_4gpu.sh``.
#
# Same model / CLI flags as the train script; the only differences are:
#   * No ``pip install`` setup block (assumes the deit conda env is ready).
#   * PYTHON_BIN / DATA_PATH / OUTPUT_DIR point at the developer's local
#     ``/mnt/bn/ic-vlm/lianghuizhu`` paths instead of trying
#     ``/mnt/bn/ic-vlm-hl`` first.
#
# See the train script for the full config table. Summary:
#   * Model: ``deit_tiny_gqa_64l_patch16_224`` (embed_dim=256, num_heads=4,
#     num_kv_heads=1, depth=64, mlp_ratio=4, patch16, img224)
#   * Training: batch-size=256/GPU, ``--drop-path 0.1`` (main.py default),
#     adamw lr=5e-4, cosine, 300 epochs
#   * 4 GPUs via ``torch.distributed.run --nproc_per_node=4``
#
# Run:
#   bash vision_tasks/deit/scripts/debug/deit_t_gqa_64l_4gpu.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEIT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=/mnt/bn/ic-vlm/lianghuizhu/miniconda3/envs/deit/bin/python
DATA_PATH=/mnt/bn/ic-vlm/zilonghuang/Imagenet1k
MODEL_NAME=deit_tiny_gqa_64l_patch16_224
OUTPUT_DIR=/mnt/bn/ic-vlm/lianghuizhu/MoDA/vision_tasks/deit/output_dir/${MODEL_NAME}

cd "${DEIT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 --master_port 49512 main.py --model "${MODEL_NAME}" --batch-size 256 --data-path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
