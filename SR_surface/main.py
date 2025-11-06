"""
表层超分主训练脚本
"""

import sys
import os
import argparse
import time
import logging
import random
import numpy as np
from pathlib import Path

# 设置基础路径
sys.path.insert(0, str(Path(__file__).parent))

import mindspore as ms
from mindspore import context, set_seed
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.communication import init

from src.config import load_yaml_config, ConfigParser
from src.surface_sr_net import SurfaceSRNet
from src.surface_sr import (
    create_loss_fn,
    create_optimizer,
    CustomWithLossCell,
    SurfaceSRTrainer,
)
from src.data import create_mindspore_dataset, SurfaceDataLoader


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    """获取命令行参数"""
    parser = ConfigParser.get_train_parser()
    args = parser.parse_args()
    return args


def init_env(args):
    """初始化环境"""
    # 设置随机种子
    set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设置上下文
    if args.device_target == "Ascend":
        context.set_context(
            mode=context.GRAPH_MODE if args.mode == "GRAPH" else context.PYNATIVE_MODE,
            device_target="Ascend",
            device_id=args.device_id,
        )
    elif args.device_target == "GPU":
        context.set_context(
            mode=context.GRAPH_MODE if args.mode == "GRAPH" else context.PYNATIVE_MODE,
            device_target="GPU",
            device_id=args.device_id,
        )
    else:
        context.set_context(
            mode=context.GRAPH_MODE if args.mode == "GRAPH" else context.PYNATIVE_MODE,
            device_target="CPU",
        )

    logger.info(f"Device target: {args.device_target}, Device ID: {args.device_id}")


def build_model(args):
    """构建模型"""
    model = SurfaceSRNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        low_h=args.low_h,
        low_w=args.low_w,
        high_h=args.high_h,
        high_w=args.high_w,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        kernel_size=(4, 4),
        batch_size=args.batch_size,
    )

    logger.info(f"Model built: SurfaceSRNet")
    logger.info(f"  Input: {args.in_channels} channels, {args.low_h}x{args.low_w}")
    logger.info(f"  Output: {args.out_channels} channels, {args.high_h}x{args.high_w}")

    return model


def build_loss_and_optimizer(model, args):
    """构建损失函数和优化器"""
    loss_fn = create_loss_fn(args.loss_type)
    optimizer = create_optimizer(model, args.optimizer, args.learning_rate)

    logger.info(f"Loss function: {args.loss_type}")
    logger.info(f"Optimizer: {args.optimizer}, LR: {args.learning_rate}")

    return loss_fn, optimizer


def load_data(args):
    """加载数据"""
    logger.info("Loading training data...")

    # 这里需要根据实际数据格式加载
    # 示例代码：
    try:
        # 加载低分辨率和高分辨率数据
        train_low = np.random.randn(100, args.in_channels, args.low_h, args.low_w)
        train_high = np.random.randn(100, args.out_channels, args.high_h, args.high_w)

        val_low = np.random.randn(10, args.in_channels, args.low_h, args.low_w)
        val_high = np.random.randn(10, args.out_channels, args.high_h, args.high_w)

        logger.info(f"Train data shape: LR {train_low.shape}, HR {train_high.shape}")
        logger.info(f"Val data shape: LR {val_low.shape}, HR {val_high.shape}")

        return train_low, train_high, val_low, val_high

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def train(args):
    """训练函数"""

    # 初始化环境
    init_env(args)

    # 创建检查点目录
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型
    model = build_model(args)

    # 构建损失和优化器
    loss_fn, optimizer = build_loss_and_optimizer(model, args)

    # 加载数据
    train_low, train_high, val_low, val_high = load_data(args)

    # 创建数据集
    train_dataset = create_mindspore_dataset(
        train_low,
        train_high,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataset = create_mindspore_dataset(
        val_low, val_high, batch_size=1, shuffle=False, num_workers=0
    )

    # 包装损失函数
    loss_cell = CustomWithLossCell(model, loss_fn)

    # 创建模型
    mindspore_model = Model(loss_cell, optimizer=optimizer)

    # 配置检查点回调
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=len(train_dataset) * args.save_interval,
        keep_checkpoint_max=3,
    )

    ckpt_callback = ModelCheckpoint(
        prefix="surface_sr", directory=str(ckpt_dir), config=ckpt_config
    )

    loss_callback = LossMonitor(per_print_times=args.save_interval)

    # 训练
    logger.info("Starting training...")
    start_time = time.time()

    try:
        mindspore_model.train(
            epoch=args.epochs,
            train_dataset=train_dataset,
            callbacks=[ckpt_callback, loss_callback],
            dataset_sink_mode=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/3600:.2f} hours")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info(f"Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    args = get_args()

    # 如果提供了 yaml 配置文件，则加载
    if os.path.exists(args.config_file_path):
        config_dict = load_yaml_config(args.config_file_path)
        # 用 yaml 配置覆盖命令行参数
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    train(args)
