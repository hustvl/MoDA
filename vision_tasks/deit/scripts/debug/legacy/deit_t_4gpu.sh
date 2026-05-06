#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEIT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=/mnt/bn/ic-vlm/lianghuizhu/miniconda3/envs/deit/bin/python
DATA_PATH=/mnt/bn/ic-vlm/zilonghuang/Imagenet1k
MODEL_NAME=deit_tiny_patch16_224
OUTPUT_DIR=/mnt/bn/ic-vlm/lianghuizhu/MoDA/vision_tasks/deit/output_dir/${MODEL_NAME}

cd "${DEIT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 --master_port 49501 main.py --model "${MODEL_NAME}" --batch-size 256 --data-path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
