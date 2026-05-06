#!/usr/bin/env bash

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
MODEL_NAME=deit_tiny_gqa_patch16_224
OUTPUT_DIR=${OUTPUT_ROOT}/${MODEL_NAME}

cd "${DEIT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 --master_port 49501 main.py --model "${MODEL_NAME}" --batch-size 256 --data-path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
