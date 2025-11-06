"""
表层超分评估模块
"""

import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import logging


logger = logging.getLogger(__name__)


class Metrics:
    """评估指标类"""

    @staticmethod
    def mae(pred, target):
        """平均绝对误差"""
        return np.mean(np.abs(pred - target))

    @staticmethod
    def mse(pred, target):
        """均方误差"""
        return np.mean((pred - target) ** 2)

    @staticmethod
    def rmse(pred, target):
        """均方根误差"""
        return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """峰值信噪比"""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return np.inf
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr

    @staticmethod
    def ssim(pred, target, max_val=1.0, window_size=11, sigma=1.5):
        """结构相似性指数"""
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        mean_pred = np.mean(pred)
        mean_target = np.mean(target)

        var_pred = np.var(pred)
        var_target = np.var(target)
        covar = np.mean((pred - mean_pred) * (target - mean_target))

        numerator = (2 * mean_pred * mean_target + c1) * (2 * covar + c2)
        denominator = (mean_pred**2 + mean_target**2 + c1) * (
            var_pred + var_target + c2
        )

        ssim = numerator / (denominator + 1e-8)
        return np.mean(ssim)

    @staticmethod
    def nrmse(pred, target):
        """标准化均方根误差"""
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        normalized = rmse / (np.max(target) - np.min(target) + 1e-8)
        return normalized

    @staticmethod
    def r2_score(pred, target):
        """R² 分数"""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2

    @staticmethod
    def correlation(pred, target):
        """相关系数"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        pred_mean = np.mean(pred_flat)
        target_mean = np.mean(target_flat)

        pred_std = np.std(pred_flat)
        target_std = np.std(target_flat)

        correlation = np.mean((pred_flat - pred_mean) * (target_flat - target_mean)) / (
            pred_std * target_std + 1e-8
        )
        return correlation


class Evaluator:
    """评估器"""

    def __init__(self, metrics_list=["mae", "mse", "rmse", "psnr", "ssim"]):
        self.metrics_list = metrics_list
        self.metrics_dict = {
            "mae": Metrics.mae,
            "mse": Metrics.mse,
            "rmse": Metrics.rmse,
            "psnr": Metrics.psnr,
            "ssim": Metrics.ssim,
            "nrmse": Metrics.nrmse,
            "r2": Metrics.r2_score,
            "correlation": Metrics.correlation,
        }

    def evaluate(self, pred, target):
        """评估"""
        results = {}

        # 转换为 numpy 数组
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        if isinstance(target, Tensor):
            target = target.asnumpy()

        for metric_name in self.metrics_list:
            if metric_name in self.metrics_dict:
                try:
                    metric_fn = self.metrics_dict[metric_name]
                    results[metric_name] = metric_fn(pred, target)
                except Exception as e:
                    logger.warning(f"Failed to compute {metric_name}: {e}")
                    results[metric_name] = np.nan
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return results

    def evaluate_batch(self, pred_batch, target_batch):
        """批量评估"""
        batch_results = []

        for pred, target in zip(pred_batch, target_batch):
            results = self.evaluate(pred, target)
            batch_results.append(results)

        # 计算平均值
        avg_results = {}
        for key in batch_results[0].keys():
            avg_results[key] = np.mean([r[key] for r in batch_results])

        return avg_results, batch_results

    def print_results(self, results, prefix=""):
        """打印结果"""
        logger.info(f"{prefix}")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.6f}")


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "metrics": {}}

    def record(self, epoch, train_loss, val_loss, metrics=None):
        """记录性能"""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        if metrics:
            for key, value in metrics.items():
                if key not in self.history["metrics"]:
                    self.history["metrics"][key] = []
                self.history["metrics"][key].append(value)

    def get_best_epoch(self, metric="val_loss"):
        """获取最佳 epoch"""
        if metric == "val_loss":
            best_idx = np.argmin(self.history["val_loss"])
        elif metric in self.history["metrics"]:
            best_idx = np.argmin(self.history["metrics"][metric])
        else:
            logger.warning(f"Unknown metric: {metric}")
            return None

        return self.history["epoch"][best_idx]

    def save_history(self, save_path):
        """保存历史记录"""
        import json

        with open(save_path, "w") as f:
            # 转换 numpy 类型
            history = {}
            for key, value in self.history.items():
                if isinstance(value, dict):
                    history[key] = {
                        k: [float(v) for v in val] for k, val in value.items()
                    }
                else:
                    history[key] = [float(v) for v in value]

            json.dump(history, f, indent=2)

    def load_history(self, load_path):
        """加载历史记录"""
        import json

        with open(load_path, "r") as f:
            self.history = json.load(f)


def compare_models(model1_results, model2_results):
    """比较两个模型的结果"""
    comparison = {}

    for key in model1_results.keys():
        if key in model2_results:
            diff = model1_results[key] - model2_results[key]
            comparison[key] = {
                "model1": model1_results[key],
                "model2": model2_results[key],
                "difference": diff,
                "ratio": model1_results[key] / (model2_results[key] + 1e-8),
            }

    return comparison
