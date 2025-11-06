# SR_surface - 表层超分项目

简体中文 | [English](README_en.md)

## 项目概述

SR_surface 是一个基于深度学习的表层数据超分辨率项目。相比 FuXi 的三维超分，本项目专注于**二维表层数据**的超分辨率，移除了深度维度，优化了网络结构。

### 关键特性

✨ **继承 FuXi 的核心技术**
- Swin Transformer 架构
- 多尺度特征融合
- 跳跃连接（Skip Connection）

✨ **针对表层数据优化**
- 移除了深度（Z）维度处理
- 纯 2D 卷积和注意力机制
- 更轻量化的模型结构

✨ **灵活的超分倍数**
- 默认 4 倍超分（256×256 → 1024×1024）
- 支持自定义分辨率

✨ **多种损失函数**
- MAE（平均绝对误差）
- MSE（均方误差）
- 组合损失
- 感知损失（可扩展）

## 项目结构

```
SR_surface/
├── configs/
│   └── surface_sr.yaml          # 主配置文件
├── src/
│   ├── __init__.py
│   ├── surface_sr_net.py        # 核心模型定义
│   ├── surface_sr.py            # 训练和推理模块
│   ├── data.py                  # 数据加载和处理
│   ├── config.py                # 配置解析
│   ├── eval.py                  # 评估指标
│   └── utils.py                 # 工具函数
├── mindearth/                   # MindEarth 依赖库
│   ├── cell/
│   ├── core/
│   ├── data/
│   ├── module/
│   └── utils/
├── scripts/                     # 训练脚本
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_distributed.sh
├── main.py                      # 主训练脚本
└── README.md                    # 本文件
```

## 快速开始

### 环境要求

- Python >= 3.8
- MindSpore >= 2.0
- NumPy
- PyYAML

### 安装

```bash
# 克隆或复制项目到本地
cd SR_surface

# 安装依赖（如需要）
pip install -r requirements.txt
```

### 训练

```bash
# 使用默认配置训练
python main.py

# 使用自定义配置文件
python main.py --config_file_path ./configs/surface_sr.yaml

# 自定义参数
python main.py --in_channels 6 --out_channels 6 --batch_size 2 --epochs 100

# 分布式训练
bash ./scripts/run_distributed.sh
```

### 配置文件

编辑 `configs/surface_sr.yaml` 调整：

```yaml
model:
  in_channels: 6          # 输入通道数（表层变量）
  out_channels: 6         # 输出通道数
  depths: 12              # Transformer block 数量
  
data:
  low_resolution:
    height: 256
    width: 256
  high_resolution:
    height: 1024
    width: 1024
    
training:
  epochs: 100
  learning_rate: 1.0e-4
  loss_function:
    type: "mae"          # 损失函数类型
```

## 模型架构

```
输入 (B, C, H, W)
  ↓
2D Patch Embedding + Position Embedding
  ↓
跳连
  ↓
Down Sample (2D Conv)
  ↓
[Swin Transformer Block] × N
  ↓
Up Sample (2D ConvTranspose)
  ↓
+ 跳连
  ↓
Patch Recover (2D ConvTranspose + Interpolation)
  ↓
输出 (B, out_channels, H*4, W*4)
```

## 核心模块说明

### SurfaceSRNet

表层超分网络的主体，包含：

- **SurfaceEmbed**: 2D Patch 嵌入和位置编码
- **SurfaceSwinBlock**: 2D Swin Transformer 块
- **DownSample2D / UpSample2D**: 上下采样
- **PatchRecover2D**: 特征恢复到目标分辨率

### 损失函数

- **MAELoss**: 简单且稳定，适合多数情况
- **MSELoss**: 对大误差更敏感
- **CombinedLoss**: MAE 和 MSE 的加权组合
- **PerceptualLoss**: 基于特征空间相似性

### 数据处理

```python
from src.data import SurfaceDataLoader

# 加载 NumPy 数据
low_res, high_res = SurfaceDataLoader.load_from_npy(
    'low_res.npy', 'high_res.npy'
)

# 标准化
low_res, mean, std = SurfaceDataLoader.normalize(low_res)
```

### 评估指标

支持以下指标：
- MAE, MSE, RMSE
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NRMSE (Normalized RMSE)
- R² Score
- Correlation

```python
from src.eval import Evaluator

evaluator = Evaluator(metrics_list=['mae', 'mse', 'psnr', 'ssim'])
results = evaluator.evaluate(pred, target)
```

## 表层超分 vs FuXi 三维超分

| 特性 | SR_surface | FuXi |
|------|-----------|------|
| 维度 | 2D (H×W) | 3D (Z×H×W) |
| 输入分辨率 | 256×256 | 256×256×30 |
| 输出分辨率 | 1024×1024 | 1024×1024×60 |
| 模型复杂度 | 低 | 高 |
| 超分对象 | 表层 SST、风场等 | 全大气 |
| 计算效率 | 高 | 低 |

## 使用示例

### 训练完整流程

```python
import numpy as np
from src.surface_sr_net import SurfaceSRNet
from src.surface_sr import create_loss_fn, create_optimizer, CustomWithLossCell
from src.data import create_mindspore_dataset

# 1. 加载数据
train_low = np.random.randn(100, 6, 256, 256)  # (N, C, H, W)
train_high = np.random.randn(100, 6, 1024, 1024)

# 2. 创建模型
model = SurfaceSRNet(in_channels=6, out_channels=6)

# 3. 创建损失和优化器
loss_fn = create_loss_fn('mae')
optimizer = create_optimizer(model, 'adam', lr=1e-4)

# 4. 创建数据集
dataset = create_mindspore_dataset(train_low, train_high, batch_size=2)

# 5. 训练（使用 MindSpore API）
...
```

### 推理

```python
from src.surface_sr import SurfaceSRInference

# 加载模型
inference = SurfaceSRInference(model, ckpt_path='checkpoint.ckpt')

# 推理
pred = inference.infer(low_res_data)
```

## 关键技术要点

### 为什么移除深度维度？

1. **表层数据的特性**：表层数据（SST、风场等）通常只有单一高度，不需要垂直超分
2. **计算效率**：2D 运算远快于 3D，显存占用少
3. **模型设计**：2D Swin Transformer 更适合地理空间数据的纹理超分

### 2D Swin Transformer 的优势

- 局部窗口注意力降低计算复杂度
- 移位操作实现跨窗口交互
- 分层结构适合多尺度特征提取

### Patch Recover 的作用

- 恢复丢失的细节信息
- 通过反卷积和双线性插值进行分辨率转换
- 跳连融合保留低分辨率的细节

## 性能优化建议

1. **数据标准化**：对输入数据进行标准化，提升训练稳定性
2. **学习率调度**：使用 Cosine Annealing 或 Warm-up 策略
3. **梯度裁剪**：防止梯度爆炸
4. **混合精度训练**：加速训练并减少显存占用
5. **数据增强**：随机翻转、旋转等操作

## 扩展功能

### 添加自定义损失函数

```python
from src.surface_sr import PerceptualLoss

class MyCustomLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        
    def construct(self, pred, target):
        # 自定义损失计算
        return loss

# 在 create_loss_fn 中注册
```

### 支持不同的超分倍数

修改 `PatchRecover2D` 的 `kernel_size` 参数：
- 2 倍超分：`kernel_size=(2, 2)`
- 4 倍超分：`kernel_size=(4, 4)` （默认）
- 8 倍超分：`kernel_size=(8, 8)`

## 故障排除

**Q: 显存不足**
- 减小 `batch_size`
- 减少 `depths` 参数
- 启用混合精度训练 `mixed_precision: O2`

**Q: 模型收敛慢**
- 增大学习率
- 调整损失函数权重
- 检查数据标准化是否正确

**Q: 推理结果质量差**
- 确保训练数据的质量
- 尝试其他损失函数
- 增加训练 epoch 数

## 参考资料

- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [FuXi: Weather Forecasting](https://www.nature.com/articles/s41612-023-00512-1)
- [Super-Resolution Techniques](https://arxiv.org/abs/2104.10852)

## 许可证

Apache License 2.0

## 作者

基于 FuXi（Fudan University）框架开发，针对表层超分优化。

## 更新日志

### v1.0.0 (2025-11-06)
- 初始版本发布
- 支持 2D 表层超分
- 集成 Swin Transformer
- 支持多种损失函数和评估指标
