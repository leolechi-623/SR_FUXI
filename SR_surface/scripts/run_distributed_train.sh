#!/bin/bash

# 分布式训练脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 参数
RANK_TABLE_FILE=${1:-./rank_table.json}
DEVICE_NUM=${2:-8}
DEVICE_START_ID=${3:-0}
CONFIG_FILE=${4:-./configs/surface_sr.yaml}

echo "=========================================="
echo "Surface Super-Resolution Distributed Training"
echo "=========================================="
echo "Rank Table: $RANK_TABLE_FILE"
echo "Device Num: $DEVICE_NUM"
echo "Device Start ID: $DEVICE_START_ID"
echo "Config: $CONFIG_FILE"
echo "=========================================="

# 启动分布式训练
mpirun --allow-run-as-root -n "$DEVICE_NUM" \
    --bind-to none -map-by slot \
    -x RANK_TABLE_FILE="$RANK_TABLE_FILE" \
    -x RANK_SIZE="$DEVICE_NUM" \
    python main.py \
        --config_file_path "$CONFIG_FILE" \
        --distribute True

echo "Distributed training completed!"
