"""
表层超分配置模块
"""

import yaml
from pathlib import Path
import argparse


def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(config, config_path):
    """保存 YAML 配置文件"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class ConfigParser:
    """配置解析器"""

    @staticmethod
    def get_train_parser():
        """训练配置解析器"""
        parser = argparse.ArgumentParser(description="Surface SR Training")

        # 基础配置
        parser.add_argument(
            "--config_file_path",
            type=str,
            default="./configs/surface_sr.yaml",
            help="Config file path",
        )
        parser.add_argument(
            "--device_target",
            type=str,
            default="Ascend",
            choices=["Ascend", "GPU", "CPU"],
            help="Device target",
        )
        parser.add_argument("--device_id", type=int, default=0, help="Device ID")
        parser.add_argument(
            "--mode",
            type=str,
            default="GRAPH",
            choices=["GRAPH", "PYNATIVE"],
            help="Execution mode",
        )

        # 模型配置
        parser.add_argument("--in_channels", type=int, default=6, help="Input channels")
        parser.add_argument(
            "--out_channels", type=int, default=6, help="Output channels"
        )
        parser.add_argument(
            "--embed_dim", type=int, default=96, help="Embedding dimension"
        )
        parser.add_argument(
            "--depths", type=int, default=12, help="Number of Swin blocks"
        )
        parser.add_argument(
            "--num_heads", type=int, default=8, help="Number of attention heads"
        )

        # 数据配置
        parser.add_argument(
            "--low_h", type=int, default=256, help="Low resolution height"
        )
        parser.add_argument(
            "--low_w", type=int, default=256, help="Low resolution width"
        )
        parser.add_argument(
            "--high_h", type=int, default=1024, help="High resolution height"
        )
        parser.add_argument(
            "--high_w", type=int, default=1024, help="High resolution width"
        )

        # 训练配置
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument(
            "--learning_rate", type=float, default=1e-4, help="Learning rate"
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-5, help="Weight decay"
        )

        # 损失配置
        parser.add_argument(
            "--loss_type",
            type=str,
            default="mae",
            choices=["mae", "mse", "combined", "perceptual"],
            help="Loss function type",
        )

        # 优化器配置
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd", "adamw"],
            help="Optimizer type",
        )

        # 路径配置
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="./data/train",
            help="Training data path",
        )
        parser.add_argument(
            "--val_data_path",
            type=str,
            default="./data/val",
            help="Validation data path",
        )
        parser.add_argument(
            "--ckpt_dir", type=str, default="./checkpoints", help="Checkpoint directory"
        )
        parser.add_argument(
            "--log_dir", type=str, default="./logs", help="Log directory"
        )

        # 其他配置
        parser.add_argument(
            "--save_interval", type=int, default=10, help="Save checkpoint interval"
        )
        parser.add_argument(
            "--eval_interval", type=int, default=5, help="Evaluation interval"
        )
        parser.add_argument(
            "--mixed_precision", type=str, default="O0", help="Mixed precision level"
        )
        parser.add_argument(
            "--num_workers", type=int, default=4, help="Number of data loading workers"
        )

        return parser

    @staticmethod
    def get_eval_parser():
        """评估配置解析器"""
        parser = argparse.ArgumentParser(description="Surface SR Evaluation")

        parser.add_argument(
            "--config_file_path",
            type=str,
            default="./configs/surface_sr.yaml",
            help="Config file path",
        )
        parser.add_argument(
            "--device_target", type=str, default="Ascend", help="Device target"
        )
        parser.add_argument("--device_id", type=int, default=0, help="Device ID")
        parser.add_argument(
            "--ckpt_path", type=str, required=True, help="Checkpoint path"
        )
        parser.add_argument(
            "--test_data_path", type=str, required=True, help="Test data path"
        )
        parser.add_argument(
            "--output_path", type=str, default="./results", help="Output path"
        )

        return parser


class Config:
    """配置容器"""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None

    def to_dict(self):
        """转换为字典"""
        return self.__dict__

    @staticmethod
    def from_yaml(config_path):
        """从 YAML 文件创建配置"""
        config_dict = load_yaml_config(config_path)
        return Config(config_dict)
