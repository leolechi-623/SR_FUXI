"""
SR_surface 项目快速入门指南
"""

# SR_surface - 表层超分项目快速入门

## 🎯 项目目标

SR_surface 是一个专注于**表层数据超分辨率**的深度学习项目。

- 基于 FuXi 的超分技术
- 针对 **2D 表层数据** 优化（移除深度维度）
- 使用 Swin Transformer 架构
- 支持 4 倍超分（256×256 → 1024×1024）

## 📁 项目结构详解

```
SR_surface/
├── src/                          # 核心代码
│   ├── surface_sr_net.py         # 模型定义
│   │   ├── SurfaceEmbed          # 2D Patch嵌入
│   │   ├── SurfaceSwinBlock      # Swin Transformer 块
│   │   ├── DownSample2D/UpSample2D  # 上下采样
│   │   └── SurfaceSRNet          # 主网络
│   │
│   ├── surface_sr.py             # 训练和推理
│   │   ├── SurfaceSRTrainer      # 训练器
│   │   ├── SurfaceSRInference    # 推理器
│   │   ├── MAELoss/MSELoss       # 损失函数
│   │   └── create_loss_fn        # 工厂函数
│   │
│   ├── data.py                   # 数据处理
│   │   ├── SurfaceDataLoader     # 数据加载
│   │   ├── SurfaceDataTransform  # 数据增强
│   │   └── create_mindspore_dataset  # 数据集创建
│   │
│   ├── config.py                 # 配置管理
│   │   ├── load_yaml_config      # 加载配置
│   │   └── ConfigParser          # 参数解析
│   │
│   ├── eval.py                   # 评估模块
│   │   ├── Metrics               # 评估指标
│   │   └── Evaluator             # 评估器
│   │
│   └── utils.py                  # 工具函数
│       ├── Timer                 # 计时器
│       ├── setup_logger          # 日志设置
│       └── visualize_results     # 结果可视化
│
├── mindearth/                    # MindEarth 库支持
│   ├── cell/                     # 神经网络组件
│   ├── core/                     # 核心模块
│   ├── data/                     # 数据处理
│   ├── module/                   # 训练模块
│   └── utils/                    # 工具函数
│
├── scripts/                      # 脚本
│   ├── run_standalone_train.sh   # 单卡训练
│   ├── run_distributed_train.sh  # 分布式训练
│   ├── eval.py                   # 评估脚本
│   └── inference.py              # 推理脚本
│
├── configs/
│   └── surface_sr.yaml           # 主配置文件
│
├── main.py                       # 主训练脚本
├── demo.ipynb                    # 演示 Notebook
├── README.md                     # 项目说明
├── requirements.txt              # 依赖列表
└── .gitignore                    # Git 忽略规则
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd SR_surface

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# Jupyter Notebook 演示
jupyter notebook demo.ipynb

# 或者运行 Python 脚本
python scripts/inference.py
```

### 3. 开始训练

```bash
# 使用默认配置
python main.py

# 使用自定义配置
python main.py --config_file_path ./configs/surface_sr.yaml

# 自定义参数
python main.py \
    --in_channels 6 \
    --out_channels 6 \
    --batch_size 2 \
    --epochs 100 \
    --learning_rate 1e-4
```

### 4. 分布式训练

```bash
bash scripts/run_distributed_train.sh rank_table.json 8 0 ./configs/surface_sr.yaml
```

## 📊 模型架构

### 2D 超分网络流程图

```
输入: (B, 6, 256, 256)
    ↓
[Patch Embedding] → (B, 96, 64, 64)
    ↓
跳连保存
    ↓
[Down Sample] → (B, 192, 32, 32)
    ↓
[12× Swin Block] → (B, 192, 32, 32)
    ↓
[Up Sample] → (B, 96, 64, 64)
    ↓
+ 跳连
    ↓
[Patch Recover] + [Bilinear Interpolation]
    ↓
输出: (B, 6, 1024, 1024)
```

### 为什么是 2D？

| 方面 | 3D (FuXi) | 2D (SR_surface) |
|------|----------|-----------------|
| 应用对象 | 全大气层 | 表层 |
| 维度处理 | Z×H×W | H×W |
| 计算量 | 很大 | 小 |
| 显存需求 | 高 | 低 |
| 超分速度 | 慢 | 快 |
| 表层精度 | 一般 | 优秀 |

## 🔧 配置文件详解

编辑 `configs/surface_sr.yaml`:

```yaml
model:
  type: "SurfaceSRNet"
  in_channels: 6              # 表层变量数
  out_channels: 6             # 输出通道
  embed_dim: 96               # 嵌入维度
  depths: 12                  # Swin Block 数量
  num_heads: 8                # 注意力头数
  kernel_size: [4, 4]         # 4倍超分

data:
  low_resolution:
    height: 256
    width: 256
  high_resolution:
    height: 1024
    width: 1024
  batch_size: 1               # 根据显存调整

training:
  epochs: 100
  learning_rate: 1.0e-4
  loss_function:
    type: "mae"               # 或 "mse", "combined"
  optimizer:
    type: "adam"              # 或 "sgd", "adamw"

device:
  target: "Ascend"            # "Ascend", "GPU", "CPU"
  device_id: 0
```

## 📈 支持的评估指标

| 指标 | 说明 | 范围 | 越高越好 |
|------|------|------|----------|
| MAE | 平均绝对误差 | [0, ∞) | ❌ |
| MSE | 均方误差 | [0, ∞) | ❌ |
| RMSE | 均方根误差 | [0, ∞) | ❌ |
| PSNR | 峰值信噪比 | (0, ∞) | ✅ |
| SSIM | 结构相似性 | [-1, 1] | ✅ |
| NRMSE | 标准化RMSE | [0, 1] | ❌ |
| R² | 决定系数 | (-∞, 1] | ✅ |
| Correlation | 相关系数 | [-1, 1] | ✅ |

## 💡 关键技术点

### 1. Swin Transformer 的优势

```
局部窗口注意力 → 降低计算复杂度 O(HW log(HW))
   ↓
移位操作 → 实现跨窗口交互
   ↓
分层结构 → 多尺度特征提取
   ↓
特别适合 2D 地理空间数据
```

### 2. Patch Recover 的作用

- **反卷积**: 恢复细节信息
- **双线性插值**: 精确调整到目标分辨率
- **跳连融合**: 保留低分辨率特征

### 3. 为什么使用 MAE 而不是 MSE？

- MAE: 对异常值不敏感，更适合气象数据
- MSE: 对大误差更敏感，可能被噪声主导
- 建议: 结合使用 (CombinedLoss)

## 🎓 使用示例

### 基础训练

```python
from src.surface_sr_net import SurfaceSRNet
from src.surface_sr import create_loss_fn, create_optimizer
import mindspore as ms

# 1. 创建模型
model = SurfaceSRNet(in_channels=6, out_channels=6)

# 2. 创建损失和优化器
loss_fn = create_loss_fn('mae')
optimizer = create_optimizer(model, 'adam', lr=1e-4)

# 3. 推理
input_data = ms.Tensor(...)  # (B, 6, 256, 256)
with ms.no_grad():
    output = model(input_data)  # (B, 6, 1024, 1024)
```

### 评估

```python
from src.eval import Evaluator

evaluator = Evaluator(
    metrics_list=['mae', 'mse', 'psnr', 'ssim']
)

results = evaluator.evaluate(pred, target)
# {'mae': 0.1234, 'mse': 0.0456, 'psnr': 25.34, 'ssim': 0.89}
```

## 🐛 常见问题

**Q: 模型如何支持不同的输入分辨率？**
- 修改 `low_h`, `low_w`, `high_h`, `high_w` 参数
- 确保高分辨率 = 低分辨率 × kernel_size

**Q: 如何实现不同倍数的超分？**
- 2 倍: `kernel_size=(2, 2)`
- 4 倍: `kernel_size=(4, 4)` (默认)
- 8 倍: `kernel_size=(8, 8)`

**Q: 推理速度如何优化？**
- 减小 `embed_dim` 或 `depths`
- 启用混合精度训练
- 批处理推理

**Q: 显存不足？**
- 减小 `batch_size`
- 使用 `mixed_precision: O2`
- 减少 `depths` 参数

## 📚 参考文献

1. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
   https://arxiv.org/abs/2103.14030

2. FuXi: Cascading Deep Transformer for 10-day Medium-range Weather Forecasting
   https://www.nature.com/articles/s41612-023-00512-1

3. Image Super-Resolution Using Very Deep Residual Channel Attention Networks
   https://arxiv.org/abs/1807.02758

## 📝 许可证

Apache License 2.0

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题，欢迎联系项目维护者。

---

**最后更新**: 2025-11-06
**版本**: 1.0.0
"""
