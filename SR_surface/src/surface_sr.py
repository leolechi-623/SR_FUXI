"""
表层超分训练和推理模块
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import numpy as np


class SurfaceSRTrainer:
    """表层超分训练器"""

    def __init__(self, model, loss_fn, optimizer, args):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.args = args

    def train_step(self, low_res, high_res, epoch=0):
        """单步训练"""

        def forward_fn(lr, hr):
            pred = self.model(lr)
            loss = self.loss_fn(pred, hr)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters)
        loss, grads = grad_fn(low_res, high_res)

        self.optimizer(grads)
        return loss

    def eval_step(self, low_res, high_res):
        """验证步骤"""
        pred = self.model(low_res)
        loss = self.loss_fn(pred, high_res)
        return pred, loss


class SurfaceSRInference:
    """表层超分推理模块"""

    def __init__(self, model, ckpt_path=None):
        self.model = model
        if ckpt_path:
            self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        """加载检查点"""
        from mindspore import load_checkpoint, load_param_into_net

        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(self.model, param_dict)

    def infer(self, low_res):
        """推理"""
        self.model.set_train(False)
        with ms.no_grad():
            pred = self.model(low_res)
        return pred


class MAELoss(nn.Cell):
    """平均绝对误差损失 - 针对表层数据"""

    def __init__(self, mask_weight=1.0):
        super().__init__()
        self.mask_weight = mask_weight

    def construct(self, pred, target):
        """
        pred: 预测的高分辨率数据
        target: 目标高分辨率数据
        返回: MAE 损失
        """
        loss = ops.mean(ops.abs(pred - target))
        return loss


class MSELoss(nn.Cell):
    """均方误差损失 - 针对表层数据"""

    def __init__(self):
        super().__init__()

    def construct(self, pred, target):
        """
        pred: 预测的高分辨率数据
        target: 目标高分辨率数据
        返回: MSE 损失
        """
        loss = ops.mean((pred - target) ** 2)
        return loss


class CombinedLoss(nn.Cell):
    """组合损失 - MAE + MSE"""

    def __init__(self, mae_weight=0.7, mse_weight=0.3):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight

    def construct(self, pred, target):
        """
        pred: 预测的高分辨率数据
        target: 目标高分辨率数据
        返回: 组合损失
        """
        mae = ops.mean(ops.abs(pred - target))
        mse = ops.mean((pred - target) ** 2)
        loss = self.mae_weight * mae + self.mse_weight * mse
        return loss


class PerceptualLoss(nn.Cell):
    """感知损失 - 基于特征相似性"""

    def __init__(self, feature_weight=1.0):
        super().__init__()
        self.feature_weight = feature_weight

    def construct(self, pred, target, pred_features=None, target_features=None):
        """
        使用特征空间的差异计算感知损失
        """
        pixel_loss = ops.mean(ops.abs(pred - target))

        if pred_features is not None and target_features is not None:
            feature_loss = ops.mean((pred_features - target_features) ** 2)
            return pixel_loss + self.feature_weight * feature_loss

        return pixel_loss


class SurfaceSRModule(nn.Cell):
    """表层超分完整模块 - 包含模型、损失和优化"""

    def __init__(self, model, loss_fn, lr=1e-4, warmup_steps=1000):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.warmup_steps = warmup_steps

        # 优化器设置
        self.optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    def construct(self, low_res, high_res):
        """前向传播 - 用于 nn.DataParallel"""
        pred = self.model(low_res)
        loss = self.loss_fn(pred, high_res)
        return loss, pred


class CustomWithLossCell(nn.Cell):
    """自定义损失包装器"""

    def __init__(self, network, loss_fn):
        super().__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, *inputs):
        if len(inputs) == 2:
            low_res, high_res = inputs
            pred = self.network(low_res)
            loss = self.loss_fn(pred, high_res)
        else:
            loss = self.loss_fn(self.network(*inputs))
        return loss


def create_loss_fn(loss_type="mae", **kwargs):
    """创建损失函数"""
    if loss_type == "mae":
        return MAELoss(**kwargs)
    elif loss_type == "mse":
        return MSELoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "perceptual":
        return PerceptualLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_optimizer(model, optimizer_type="adam", lr=1e-4, **kwargs):
    """创建优化器"""
    if optimizer_type == "adam":
        return nn.Adam(model.trainable_params(), learning_rate=lr, **kwargs)
    elif optimizer_type == "sgd":
        return nn.SGD(model.trainable_params(), learning_rate=lr, **kwargs)
    elif optimizer_type == "adamw":
        return nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
