#!/bin/bash

# 表层超分独立训练脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 参数设置
DEVICE_ID=${1:-0}
DEVICE_TARGET=${2:-Ascend}
CONFIG_FILE=${3:-./configs/surface_sr.yaml}
MODE=${4:-GRAPH}

echo "=========================================="
echo "Surface Super-Resolution Training"
echo "=========================================="
echo "Device ID: $DEVICE_ID"
echo "Device Target: $DEVICE_TARGET"
echo "Config File: $CONFIG_FILE"
echo "Mode: $MODE"
echo "=========================================="

# 运行训练
python main.py \
    --device_id "$DEVICE_ID" \
    --device_target "$DEVICE_TARGET" \
    --config_file_path "$CONFIG_FILE" \
    --mode "$MODE" \
    --batch_size 2 \
    --epochs 100

echo "Training completed!"
