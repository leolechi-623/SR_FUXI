"""
Surface Super-Resolution Network (2D版本)
基于 FuXi 的超分技术，针对表层数据优化
- 移除了深度维度（Z）
- 保留了 Swin Transformer 的关键结构
- 使用 2D Patch Embedding 和超分模块
"""

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import nn, Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform, Normal

import math
import numpy as np


class SurfaceEmbed(nn.Cell):
    """2D 表层数据嵌入模块"""

    def __init__(
        self,
        in_channels=6,
        h_size=256,
        w_size=256,
        embed_dim=96,
        patch_size=4,
        batch_size=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.h_size = h_size
        self.w_size = w_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        # 2D Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            pad_mode="pad",
            padding=0,
            has_bias=True,
        )

        # 位置编码
        num_patches_h = h_size // patch_size
        num_patches_w = w_size // patch_size
        self.pos_embed = Parameter(
            initializer(Uniform(0.02), (1, embed_dim, num_patches_h, num_patches_w)),
            name="pos_embed",
        )

    def construct(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x + self.pos_embed
        return x


class SurfaceSwinBlock(nn.Cell):
    """2D Swin Transformer Block（表层优化版）"""

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Layer Norm
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))

        # Window Attention
        self.attn = SurfaceWindowAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.SequentialCell(
            nn.Dense(dim, mlp_hidden_dim, has_bias=True),
            act_layer(),
            nn.Dense(mlp_hidden_dim, dim, has_bias=True),
        )

        self.drop_path = DropPath(drop_path)

    def construct(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 平移操作
        if self.shift_size > 0:
            x_shifted = ops.roll(x, (-self.shift_size, -self.shift_size), axis=(2, 3))
        else:
            x_shifted = x

        # Window Attention
        x_attn = self.attn(x_shifted)
        x = x + self.drop_path(x_attn)

        # MLP
        x_flat = x.reshape(B, C, -1).transpose(0, 2, 1)  # (B, HW, C)
        x_mlp = self.mlp(x_flat).transpose(0, 2, 1).reshape(B, C, H, W)
        x = x + self.drop_path(x_mlp)

        if self.shift_size > 0:
            x = ops.roll(x, (self.shift_size, self.shift_size), axis=(2, 3))

        return x


class SurfaceWindowAttention(nn.Cell):
    """2D Window Attention"""

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)

    def construct(self, x):
        # 简化版：直接对整个特征图做注意力
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1).transpose(0, 2, 1)  # (B, HW, C)

        qkv = self.qkv(x_flat).reshape(
            B, -1, 3, self.num_heads, self.dim // self.num_heads
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * (self.dim // self.num_heads) ** (-0.5)
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x.transpose(0, 2, 1).reshape(B, C, H, W)


class DownSample2D(nn.Cell):
    """2D 下采样"""

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=scale,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

    def construct(self, x):
        return self.down(x)


class UpSample2D(nn.Cell):
    """2D 上采样"""

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.up = nn.Conv2dTranspose(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=scale,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

    def construct(self, x):
        return self.up(x)


class PatchRecover2D(nn.Cell):
    """2D Patch 恢复模块"""

    def __init__(self, in_channels, out_h, out_w, out_channels, kernel_size=(4, 4)):
        super().__init__()
        self.in_channels = in_channels
        self.out_h = out_h
        self.out_w = out_w
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 反卷积恢复
        self.recover = nn.Conv2dTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            pad_mode="pad",
            padding=0,
            has_bias=True,
        )

        self.interpolate = ops.ResizeBilinearV2()

    def construct(self, x):
        # x: (B, C, h, w) -> (B, out_channels, out_h, out_w)
        x = self.recover(x)  # 恢复到中间分辨率

        # 双线性插值到目标分辨率
        if x.shape[-2:] != (self.out_h, self.out_w):
            x = self.interpolate(x, size=(self.out_h, self.out_w))

        return x


class DropPath(nn.Cell):
    """Drop Path (Stochastic Depth)"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = ops.uniform(shape, Tensor(0, ms.float32), Tensor(1, ms.float32))
        random_tensor = (random_tensor < keep_prob).astype(x.dtype)
        return x * random_tensor / keep_prob


class SurfaceSRNet(nn.Cell):
    """
    表层超分网络 - 2D版本

    参数:
        in_channels: 输入通道数（表层变量数）
        out_channels: 输出通道数
        low_h, low_w: 低分辨率大小
        high_h, high_w: 高分辨率大小
        embed_dim: 嵌入维度
        depths: Swin Block 数量
        num_heads: 注意力头数
        kernel_size: Patch恢复的卷积核大小
    """

    def __init__(
        self,
        in_channels=6,
        out_channels=6,
        low_h=256,
        low_w=256,
        high_h=1024,
        high_w=1024,
        embed_dim=96,
        depths=12,
        num_heads=8,
        kernel_size=(4, 4),
        batch_size=1,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # 2D Embedding
        self.surface_embed = SurfaceEmbed(
            in_channels=in_channels,
            h_size=low_h,
            w_size=low_w,
            embed_dim=embed_dim,
            patch_size=4,
            batch_size=batch_size,
        )

        # 下采样
        self.down = DownSample2D(embed_dim, embed_dim * 2, scale=2)

        # Swin Transformer Blocks
        self.swin_blocks = nn.CellList(
            [
                SurfaceSwinBlock(
                    dim=embed_dim * 2,
                    num_heads=num_heads,
                    window_size=8,
                    shift_size=4 if i % 2 == 0 else 0,
                    mlp_ratio=4.0,
                    drop_path=0.1 * (i / depths),
                )
                for i in range(depths)
            ]
        )

        # 上采样
        self.up = UpSample2D(embed_dim * 2, embed_dim, scale=2)

        # Patch 恢复
        self.patch_recover = PatchRecover2D(
            in_channels=embed_dim,
            out_h=high_h,
            out_w=high_w,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def construct(self, x):
        """
        x: (B, C, H, W) - 输入表层数据
        返回: (B, out_channels, high_h, high_w) - 超分后的表层数据
        """
        # Embedding
        x = self.surface_embed(x)  # (B, embed_dim, H/4, W/4)

        # 跳连
        skip = x

        # 下采样
        x = self.down(x)  # (B, embed_dim*2, H/8, W/8)

        # Swin Transformer
        for block in self.swin_blocks:
            x = block(x)

        # 上采样
        x = self.up(x)  # (B, embed_dim, H/4, W/4)

        # 跳连融合
        x = x + skip

        # Patch 恢复到高分辨率
        x = self.patch_recover(x)  # (B, out_channels, high_h, high_w)

        return x
