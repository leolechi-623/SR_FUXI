"""
表层超分评估脚本
用于评估训练好的模型性能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import logging
from src.config import ConfigParser, load_yaml_config
from src.surface_sr_net import SurfaceSRNet
from src.eval import Evaluator, PerformanceMonitor
from src.utils import setup_logger
import mindspore as ms


def get_eval_args():
    """获取评估参数"""
    parser = ConfigParser.get_eval_parser()
    return parser.parse_args()


def main():
    args = get_eval_args()

    # 设置日志
    setup_logger(log_dir="./logs", log_file="eval.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Surface SR Evaluation")
    logger.info("=" * 50)

    # 加载配置
    if Path(args.config_file_path).exists():
        config_dict = load_yaml_config(args.config_file_path)
        logger.info(f"Loaded config from {args.config_file_path}")
    else:
        config_dict = {}

    # 构建模型
    model = SurfaceSRNet(
        in_channels=config_dict.get("in_channels", 6),
        out_channels=config_dict.get("out_channels", 6),
        low_h=256,
        low_w=256,
        high_h=1024,
        high_w=1024,
    )

    logger.info("Model built successfully")

    # 创建评估器
    evaluator = Evaluator(metrics_list=["mae", "mse", "rmse", "psnr", "ssim"])

    logger.info("Evaluator created")

    # 模拟测试数据
    test_low = np.random.randn(5, 6, 256, 256).astype(np.float32)
    test_high = np.random.randn(5, 6, 1024, 1024).astype(np.float32)

    logger.info(f"Test data shape: LR {test_low.shape}, HR {test_high.shape}")

    # 评估
    logger.info("Starting evaluation...")

    all_results = []
    for i in range(len(test_low)):
        # 推理
        lr = ms.Tensor(test_low[i : i + 1], ms.float32)
        with ms.no_grad():
            pred = model(lr).asnumpy()

        # 评估单个样本
        hr = test_high[i : i + 1]
        results = evaluator.evaluate(pred, hr)
        all_results.append(results)

        logger.info(
            f"Sample {i+1}: MAE={results['mae']:.6f}, PSNR={results['psnr']:.2f}"
        )

    # 计算平均指标
    avg_results = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        avg_results[key] = np.mean(values)

    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("Average Evaluation Results:")
    logger.info("=" * 50)
    evaluator.print_results(avg_results)

    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
