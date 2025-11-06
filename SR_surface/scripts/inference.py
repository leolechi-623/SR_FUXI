"""
表层超分推理示例
展示如何使用训练好的模型进行推理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from src.surface_sr_net import SurfaceSRNet
from src.surface_sr import SurfaceSRInference
from src.utils import setup_logger, Timer, visualize_results
import mindspore as ms


def main():
    """推理示例"""

    # 设置日志
    setup_logger(log_dir="./logs", log_file="inference.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Surface SR Inference Example")
    logger.info("=" * 50)

    # 创建模型
    model = SurfaceSRNet(
        in_channels=6,
        out_channels=6,
        low_h=256,
        low_w=256,
        high_h=1024,
        high_w=1024,
        embed_dim=96,
        depths=12,
        num_heads=8,
    )

    logger.info("Model created")

    # 模拟输入数据
    batch_size = 2
    low_res = np.random.randn(batch_size, 6, 256, 256).astype(np.float32)

    logger.info(f"Input shape: {low_res.shape}")

    # 推理
    timer = Timer()

    timer.start("inference")

    with ms.no_grad():
        input_tensor = ms.Tensor(low_res, ms.float32)
        output = model(input_tensor)
        output_np = output.asnumpy()

    elapsed = timer.stop("inference")

    logger.info(f"Output shape: {output_np.shape}")
    logger.info(f"Inference time: {elapsed:.4f}s")
    logger.info(f"Average time per sample: {elapsed/batch_size:.4f}s")

    # 统计
    logger.info(f"Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
    logger.info(f"Output mean: {output_np.mean():.4f}, std: {output_np.std():.4f}")

    # 可视化（如果有真值）
    high_res = np.random.randn(batch_size, 6, 1024, 1024).astype(np.float32)

    try:
        visualize_results(
            output_np, high_res, save_path="./results/inference_result.png"
        )
        logger.info("Visualization saved")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    logger.info("\nInference completed!")

    return output_np


if __name__ == "__main__":
    pred = main()
