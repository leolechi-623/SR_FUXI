"""
表层超分工具函数
"""

import os
import logging
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime


logger = logging.getLogger(__name__)


def create_exp_dir(exp_name, base_path="./experiments"):
    """创建实验目录"""
    exp_dir = Path(base_path) / exp_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    for subdir in ["checkpoints", "logs", "results", "plots"]:
        (exp_dir / subdir).mkdir(exist_ok=True)

    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def save_config(config, save_path):
    """保存配置文件"""
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Config saved to {save_path}")


def save_results(results, save_path):
    """保存结果"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {save_path}")


def load_results(load_path):
    """加载结果"""
    with open(load_path, "r") as f:
        results = json.load(f)
    return results


def setup_logger(log_dir="./logs", log_file="surface_sr.log"):
    """设置日志"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


def normalize_data(data, mean=None, std=None):
    """标准化数据"""
    if mean is None:
        mean = np.mean(data, axis=(0, 2, 3), keepdims=True)
    if std is None:
        std = np.std(data, axis=(0, 2, 3), keepdims=True)

    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std


def denormalize_data(data, mean, std):
    """反标准化数据"""
    return data * std + mean


def save_checkpoint(model, optimizer, epoch, save_path):
    """保存检查点"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict() if hasattr(model, "state_dict") else None,
        "optimizer_state": (
            optimizer.state_dict() if hasattr(optimizer, "state_dict") else None
        ),
    }

    # 对于 MindSpore，使用专用函数
    try:
        from mindspore import save_checkpoint

        save_checkpoint(model, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    except:
        logger.warning("Could not save checkpoint using MindSpore API")


def load_checkpoint(model, ckpt_path):
    """加载检查点"""
    try:
        from mindspore import load_checkpoint, load_param_into_net

        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(model, param_dict)
        logger.info(f"Checkpoint loaded from {ckpt_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return model


def get_model_size(model):
    """获取模型大小"""
    total_params = 0
    for param in model.trainable_params():
        total_params += np.prod(param.shape)

    size_mb = total_params * 4 / (1024**2)  # 假设 float32
    return total_params, size_mb


def count_parameters(model):
    """计算参数数量"""
    trainable_params = sum(p.numel() for p in model.trainable_params())
    non_trainable_params = sum(
        p.numel() for p in model.get_parameters() if p not in model.trainable_params()
    )
    total_params = trainable_params + non_trainable_params

    logger.info(f"Model Parameters:")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Non-trainable: {non_trainable_params:,}")
    logger.info(f"  Total: {total_params:,}")

    return total_params, trainable_params, non_trainable_params


def visualize_results(pred, target, save_path=None):
    """可视化结果"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 取第一个样本的第一个通道
    pred_vis = pred[0, 0] if len(pred.shape) == 4 else pred[0]
    target_vis = target[0, 0] if len(target.shape) == 4 else target[0]
    diff = np.abs(pred_vis - target_vis)

    # 预测
    im0 = axes[0].imshow(pred_vis, cmap="viridis")
    axes[0].set_title("Prediction")
    plt.colorbar(im0, ax=axes[0])

    # 目标
    im1 = axes[1].imshow(target_vis, cmap="viridis")
    axes[1].set_title("Target")
    plt.colorbar(im1, ax=axes[1])

    # 差异
    im2 = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute Difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")

    plt.close()


class Timer:
    """计时器"""

    def __init__(self):
        self.times = {}
        self.starts = {}

    def start(self, name):
        """开始计时"""
        import time

        self.starts[name] = time.time()

    def stop(self, name):
        """停止计时"""
        import time

        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(elapsed)
            return elapsed
        return None

    def get_average(self, name):
        """获取平均时间"""
        if name in self.times:
            return np.mean(self.times[name])
        return None

    def print_summary(self):
        """打印摘要"""
        logger.info("Timing Summary:")
        for name, times in self.times.items():
            avg = np.mean(times)
            logger.info(f"  {name}: {avg:.4f}s (avg)")


def set_random_seed(seed=42):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import mindspore as ms

        ms.set_seed(seed)
    except:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

    logger.info(f"Random seed set to {seed}")
