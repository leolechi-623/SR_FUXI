# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The Substructure of FuXiNet"""
import math
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Uniform


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


def window_partition(x, window_size):
    batch_size, pressure_level_num, h_size, w_size, channel_size = x.shape
    x = x.reshape(batch_size, pressure_level_num // window_size[0], window_size[0], h_size // window_size[1],
                  window_size[1],
                  w_size // window_size[2],
                  window_size[2], channel_size)
    windows = x.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (pressure_level_num // window_size[0]) *
                                                          (h_size // window_size[1]),
                                                          window_size[0], window_size[1], window_size[2], channel_size)
    return windows


def window_reverse(windows, window_size, pressure_level_num, h_size, w_size):
    batch_size, _, _, _ = windows.shape
    batch_size = int(batch_size / (w_size // window_size[2]))
    x = windows.reshape(batch_size, w_size // window_size[2], pressure_level_num // window_size[0],
                        h_size // window_size[1],
                        window_size[0],
                        window_size[1], window_size[2], -1)
    x = x.transpose(0, 2, 4, 3, 5, 1, 6, 7).reshape(batch_size, pressure_level_num, h_size, w_size, -1)
    return x


class CustomConv3d(nn.Cell):
    """
    Applies a 3D convolution over an input tensor which is typically of shape (N, C, D, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode='same',
                 has_bias=False, dtype=ms.float16):
        super(CustomConv3d, self).__init__()
        self.out_channels = out_channels
        self.conv2d_blocks = nn.CellList(
            [nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size=(kernel_size[1], kernel_size[2]),
                       stride=(stride[1], stride[2]),
                       pad_mode=pad_mode,
                       dtype=dtype,
                       ) for _ in range(kernel_size[0])]
        )
        w = Tensor(np.identity(kernel_size[0]), dtype=dtype)
        self.conv2d_weight = ops.expand_dims(ops.expand_dims(w, axis=0), axis=0)
        self.k = kernel_size[0]
        self.stride = stride
        self.pad_mode = pad_mode
        self.conv2d_dtype = dtype
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(Uniform(), [1, out_channels, 1, 1, 1], dtype=dtype))

    def construct(self, x_):
        b, c, d, h, w = x_.shape
        x_ = x_.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        out = []
        for i in range(self.k):
            out.append(self.conv2d_blocks[i](x_))
        out = ops.stack(out, axis=-1)
        _, cnew, hnew, wnew, _ = out.shape
        out = out.reshape(b, d, cnew, hnew, wnew, self.k).transpose(0, 2, 3, 4, 1, 5).reshape(-1, 1, d, self.k)
        out = ops.conv2d(out, self.conv2d_weight, stride=(self.stride[0], 1), pad_mode='valid')
        out = out.reshape(b, cnew, hnew, wnew, -1).transpose(0, 1, 4, 2, 3)
        if self.has_bias:
            out += self.bias
        return out


class CubeEmbed(nn.Cell):
    """
    Cube Embedding for surface data (z=1)
    """

    def __init__(self, in_channels, h_size, w_size, level_feature_size, pressure_level_num, batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.h_size = h_size
        self.w_size = w_size
        self.batch_size = batch_size
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.conv3d_dtype = mstype.float16
        
        # 针对表层数据优化：kernel_size 在深度维度为1
        self.cube3d = CustomConv3d(level_feature_size, in_channels, kernel_size=(1, 4, 4),
                                   pad_mode="valid", stride=(1, 4, 4), has_bias=True, dtype=mstype.float32)

    def construct(self, x):
        """CubeEmbed forward function for surface data."""
        x = x.reshape(self.batch_size, self.h_size, self.w_size, self.level_feature_size, self.pressure_level_num)
        x = x.transpose(0, 3, 4, 1, 2)  # (batch_size, level_feature_size, pressure_level_num, h_size, w_size)
        x = self.cube3d(x)  # (batch_size, in_channels, pressure_level_num, h_size//4, w_size//4)
        output = self.layer_norm(x.transpose(0, 2, 3, 4, 1))
        output = output.transpose(0, 4, 1, 2, 3)
        return output


class ResidualBlock(nn.Cell):
    """
    Residual Block in down sample.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  pad_mode="pad",
                                  stride=1,
                                  has_bias=True)
        self.group_norm = nn.GroupNorm(8, 8)
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  pad_mode="pad",
                                  stride=1,
                                  has_bias=True)
        self.silu = nn.SiLU()

    def construct(self, x):
        """Residual Block forward function."""
        x1 = x
        x = self.conv2d_1(x)
        x = self.silu(x)
        x = self.conv2d_2(x)
        output = x + x1
        return output


class DownSample(nn.Cell):
    """Down Sample module."""
    def __init__(self, in_channels=96, out_channels=192):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                stride=2,
                                has_bias=True)
        self.residual_block = ResidualBlock(in_channels=out_channels,
                                            out_channels=out_channels)
        self.silu = nn.SiLU()

    def construct(self, x):
        """Down Sample forward function."""
        batch_size, channels, patch_size, h_size, w_size = x.shape
        x = x.transpose(0, 2, 1, 3, 4).reshape(batch_size * patch_size, channels, h_size, w_size)
        x = self.conv2d(x)
        x = self.residual_block(x)
        x = self.silu(x)
        x = x.transpose(0, 2, 3, 1)
        output = x.reshape(batch_size, patch_size, h_size // 2, w_size // 2, channels * 2)
        return output


class RelativeBias(nn.Cell):
    """Improved RelativeBias with continuous position encoding"""
    def __init__(self, type_windows, num_heads, window_size):
        super(RelativeBias, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.type_windows = type_windows
        
        # Continuous position bias
        # NOTE: relative_coords_table stores 3 coordinates (z, h, w) -> input dim must be 3
        self.cpb_mlp = nn.SequentialCell([
            nn.Dense(3, 512, has_bias=True),
            nn.ReLU(),
            nn.Dense(512, num_heads, has_bias=False)
        ])
        
        # 计算窗口内位置数量
        self.window_num_elements = window_size[0] * window_size[1] * window_size[2]
        
        # Relative coordinates table
        relative_coords_h = ops.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=ms.float32)
        relative_coords_w = ops.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=ms.float32)
        relative_coords_z = ops.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=ms.float32)
        
        # 3D 相对坐标
        relative_coords_table = ops.stack(ops.meshgrid(relative_coords_z, relative_coords_h, relative_coords_w, indexing='ij'))
        relative_coords_table = relative_coords_table.transpose(1, 2, 3, 0)  # (2*Wz-1, 2*Wh-1, 2*Ww-1, 3)
        relative_coords_table = ops.expand_dims(relative_coords_table, axis=0)  # (1, 2*Wz-1, 2*Wh-1, 2*Ww-1, 3)
        
        # 归一化
        relative_coords_table[:, :, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table[:, :, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = ops.sign(relative_coords_table) * ops.log2(ops.abs(relative_coords_table) + 1.0) / 3.0
        
        self.relative_coords_table = relative_coords_table
        
        # 3D 相对位置索引
        coords_z = ops.arange(self.window_size[0])
        coords_h = ops.arange(self.window_size[1])
        coords_w = ops.arange(self.window_size[2])
        
        coords = ops.stack(ops.meshgrid(coords_z, coords_h, coords_w, indexing='ij'))  # 3, Wz, Wh, Ww
        coords_flatten = ops.flatten(coords, start_dim=1)  # 3, Wz*Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wz*Wh*Ww, Wz*Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wz*Wh*Ww, Wz*Wh*Ww, 3
        
        # 将相对坐标转换为索引
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        
        relative_position_index = relative_coords.sum(-1)  # Wz*Wh*Ww, Wz*Wh*Ww
        self.relative_position_index = Parameter(relative_position_index.astype(mnp.int32), requires_grad=False)

    def construct(self):
        """Relative Bias forward function."""
        # 获取相对位置偏置表
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)  # (1, 2*Wz-1, 2*Wh-1, 2*Ww-1, num_heads)
        relative_position_bias_table = relative_position_bias_table.reshape(-1, self.num_heads)  # ((2*Wz-1)*(2*Wh-1)*(2*Ww-1), num_heads)
        
        # 根据索引获取相对位置偏置
        relative_position_bias = relative_position_bias_table[self.relative_position_index.reshape(-1)]  # (Wz*Wh*Ww * Wz*Wh*Ww, num_heads)
        relative_position_bias = relative_position_bias.reshape(
            self.window_num_elements, self.window_num_elements, -1)  # (Wz*Wh*Ww, Wz*Wh*Ww, num_heads)
        
        # 调整维度以匹配注意力分数: (1, 1, num_heads, Wz*Wh*Ww, Wz*Wh*Ww)
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)  # (num_heads, Wz*Wh*Ww, Wz*Wh*Ww)
        relative_position_bias = 16 * ops.sigmoid(relative_position_bias)
        relative_position_bias = ops.expand_dims(relative_position_bias, axis=0)  # (1, num_heads, Wz*Wh*Ww, Wz*Wh*Ww)
        relative_position_bias = ops.expand_dims(relative_position_bias, axis=0)  # (1, 1, num_heads, Wz*Wh*Ww, Wz*Wh*Ww)
        
        return relative_position_bias


class WindowAttention(nn.Cell):
    def __init__(self, dim, num_heads, window_size, input_shape):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_surface = input_shape[0] == 1
        self.q = nn.Dense(dim, dim)
        self.k = nn.Dense(dim, dim)
        self.v = nn.Dense(dim, dim)
        self.proj = nn.Dense(dim, dim)

    def construct(self, x, mask=None):
        # x expected shape: (w_nums, z_h_nums, nums, dim)
        w_nums, z_h_nums, nums, C = x.shape
        assert C % self.num_heads == 0, "dim must be divisible by num_heads"
        head_dim = C // self.num_heads

        # Project Q, K, V in-place (preserve 4D layout)
        q = self.q(x)  # (w_nums, z_h_nums, nums, C)
        k = self.k(x)
        v = self.v(x)

        # Flatten first two spatial dims into batch for batched matmul
        B_ = w_nums * z_h_nums
        q = q.reshape((B_, nums, self.num_heads, head_dim)).transpose(0, 2, 1, 3)  # (B_, heads, N, D)
        k = k.reshape((B_, nums, self.num_heads, head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((B_, nums, self.num_heads, head_dim)).transpose(0, 2, 1, 3)

        # Scaled dot-product
        q = q * self.scale
        attn = ops.BatchMatMul()(q, ops.transpose(k, (0, 1, 3, 2)))  # (B_, heads, N, N)

        # If mask provided: mask shape expected (nW, z_h_nums, nums, nums) where nW == w_nums
        if mask is not None:
            nW = mask.shape[0]
            # reshape attn to (z_h_nums, w_nums, heads, N, N)
            attn = attn.reshape((z_h_nums, w_nums, self.num_heads, nums, nums))
            # mask: (nW, z_h_nums, N, N) -> transpose to (z_h_nums, nW, N, N)
            mask_t = ops.transpose(mask, (1, 0, 2, 3))
            # expand mask to have head dim: (z_h_nums, nW, 1, N, N)
            mask_t = ops.expand_dims(mask_t, axis=2)
            attn = attn + mask_t  # broadcast over heads
            # reshape back to (B_, heads, N, N)
            attn = attn.reshape((B_, self.num_heads, nums, nums))

        attn = ops.softmax(attn, axis=-1)

        # attention * V
        out = ops.BatchMatMul()(attn, v)  # (B_, heads, N, D)
        out = out.transpose(0, 2, 1, 3).reshape((B_, nums, C))  # (B_, N, C)

        # restore 4D window layout to (w_nums, z_h_nums, nums, C)
        out = out.reshape((w_nums, z_h_nums, nums, C))

        # final projection applied per-last-dim
        out = self.proj(out)
        return out


class Mlp(nn.Cell):
    """MLP Layer."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act_layer = nn.GELU(approximate=False)
        self.fc2 = nn.Dense(hidden_features, out_features)

    def construct(self, x):
        """MLP Layer forward function"""
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Cell):
    """Enhanced Swin Transformer Block with DropPath"""
    def __init__(self,
                 shift_size,
                 window_size,
                 dim=192,
                 num_heads=6,
                 input_shape=None,
                 mlp_ratio=4.,
                 drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.window_size = window_size
        self.is_surface = input_shape[0] == 1  # Check if surface data (Z=1)
        
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-5)
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    input_shape=input_shape)
        
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        
        self.drop_path = DropPath(drop_path, ndim=1) if drop_path > 0. else nn.Identity()

    def construct(self, x, mask_matrix, pressure_level_num, h_size, w_size):
        """Enhanced Transformer Block forward with surface data handling"""
        batch_size = x.shape[0]
        channel_size = x.shape[2]
        shortcut = x
        x = x.reshape(batch_size, pressure_level_num, h_size, w_size, channel_size)

        # For surface data (Z=1), only shift in H,W dimensions
        if self.is_surface:
            if self.shift_size[1] > 0 or self.shift_size[2] > 0:
                shifted_x = mnp.roll(x,
                                   shift=(0, -self.shift_size[1], -self.shift_size[2]),
                                   axis=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None
        else:
            # Original 3D shifting
            if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
                shifted_x = mnp.roll(x,
                                   shift=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   axis=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None
            
        x_windows = window_partition(shifted_x, self.window_size)
        b_w, _, _, _, _, channel_size = x_windows.shape
        x_windows = x_windows.reshape(b_w,
                                    (pressure_level_num // self.window_size[0]) * (h_size // self.window_size[1]),
                                    self.window_size[0] * self.window_size[1] * self.window_size[2],
                                    self.dim)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, pressure_level_num, h_size, w_size)

        # Reverse shift - also handle surface case
        if self.is_surface:
            if mask_matrix is not None:
                x = mnp.roll(shifted_x,
                            shift=(0, self.shift_size[1], self.shift_size[2]),
                            axis=(1, 2, 3))
            else:
                x = shifted_x
        else:
            if mask_matrix is not None:
                x = mnp.roll(shifted_x,
                            shift=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                            axis=(1, 2, 3))
            else:
                x = shifted_x

        x = x.reshape(batch_size, pressure_level_num * h_size * w_size, channel_size)

        # FFN with DropPath
        x = shortcut + self.drop_path(self.norm1(x))
        output = x + self.drop_path(self.norm2(self.mlp(x)))
        return output


class BaseBlock(nn.Cell):
    """Base Block with enhanced configuration"""
    def __init__(self,
                 in_channels=192,
                 input_shape=None,
                 window_size=(2, 4, 8),
                 recompute=False):
        super().__init__()
        self.in_channels = in_channels
        self.is_surface = input_shape[0] == 1
        
        # 针对表层数据调整窗口和偏移大小
        if self.is_surface:
            self.window_size = (1, window_size[1], window_size[2])
            self.shift_size = (0, 2, 2)  # 表层数据只在 H,W 维度上偏移
        else:
            self.window_size = window_size
            self.shift_size = (1, 2, 4)

        # 创建 transformer blocks 时传入正确的配置
        self.blk1 = TransformerBlock(
            dim=in_channels,
            shift_size=(0, 0, 0),  # 第一个 block 不偏移
            window_size=self.window_size,
            input_shape=input_shape,
            mlp_ratio=4.0,
            drop_path=0.0
        )
        
        self.blk2 = TransformerBlock(
            dim=in_channels,
            shift_size=self.shift_size,
            window_size=self.window_size,
            input_shape=input_shape,
            mlp_ratio=4.0,
            drop_path=0.0
        )

        if recompute:
            self.blk1.recompute()
            self.blk2.recompute()
            
        # 添加数值保护
        self.eps = 1e-6
        self.clip = ops.clip_by_value

    def construct(self, x, batch_size, pressure_level_num, h_size, w_size):
        """Surface-aware BaseBlock forward function"""
        # 确保压力层数与表层数据一致
        if self.is_surface:
            pressure_level_num = 1
            
        # 创建掩码，注意表层数据的特殊处理
        img_mask = ops.zeros((batch_size, pressure_level_num, h_size, w_size, 1), mstype.float32)
        
        # 调整切片以适应表层数据
        if self.is_surface:
            h_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None)
            )
            w_slices = (
                slice(0, -self.window_size[2]),
                slice(-self.window_size[2], -self.shift_size[2]),
                slice(-self.shift_size[2], None)
            )
            
            # 只在 H,W 维度上计数
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w, :] = cnt
                    cnt += 1
        else:
            # 原有的3D掩码生成逻辑
            z_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None)
            )
            h_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None)
            )
            cnt = 0
            for z in z_slices:
                for h in h_slices:
                    img_mask[:, z, h, :, :] = cnt
                    cnt += 1
        
        # 重塑掩码以匹配注意力计算
        mask_windows = self._reshape_mask(img_mask, batch_size, pressure_level_num, h_size, w_size)
        
        # 生成注意力掩码
        attn_mask = mnp.expand_dims(mask_windows, axis=2) - mnp.expand_dims(mask_windows, axis=3)
        attn_mask = ops.masked_fill(
            ops.masked_fill(attn_mask, attn_mask != 0, float(-100.0)),
            attn_mask == 0,
            float(0.0)
        )
        
        # 应用 transformer blocks
        x = self.clip(x, -1e6, 1e6)
        x = self.blk1(x, None, pressure_level_num, h_size, w_size)  # 第一个 block 不需要 mask
        x = self.clip(x, -1e6, 1e6)
        x = self.blk2(x, attn_mask, pressure_level_num, h_size, w_size)
        x = self.clip(x, -1e6, 1e6)
        
        return x

    def _reshape_mask(self, img_mask, batch_size, pressure_level_num, h_size, w_size):
        """Safe mask reshaping for both surface and volumetric data"""
        mask_windows = img_mask.reshape(
            batch_size,
            pressure_level_num // self.window_size[0],
            self.window_size[0],
            h_size // self.window_size[1],
            self.window_size[1],
            w_size // self.window_size[2],
            self.window_size[2],
            1
        )
        
        mask_windows = mask_windows.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(
            -1,
            (pressure_level_num // self.window_size[0]) * (h_size // self.window_size[1]),
            self.window_size[0] * self.window_size[1] * self.window_size[2]
        )
        
        return mask_windows


# 保持原有的 UpSamplePS 和 PatchRecover 不变
class DepthToSpaceCell(nn.Cell):
    def __init__(self, block_size=2):
        super().__init__()
        self.op = ops.DepthToSpace(block_size=block_size)
        
    def construct(self, x):
        return self.op(x)


class UpSamplePS(nn.Cell):
    """x8 上采样：Conv(c -> 4*mid) -> PixelShuffle(2) -> Conv(mid->mid) * 3层，然后 tail 输出"""
    def __init__(self, in_channels=768, mid=384, out_channels=384, stages=3):
        super().__init__()
        self.stages = nn.CellList()
        c = in_channels
        for _ in range(stages):
            self.stages.append(
                nn.SequentialCell(
                    nn.Conv2d(c, mid * 4, kernel_size=3, padding=1, pad_mode='pad'),
                    DepthToSpaceCell(block_size=2),
                    nn.Conv2d(mid, mid, kernel_size=3, padding=1, pad_mode='pad'),
                    nn.SiLU()
                )
            )
            c = mid
        self.tail = nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, pad_mode='pad')

    def construct(self, x):
        B, P, H, W, C = x.shape
        x = x.transpose(0, 1, 4, 2, 3).reshape(B * P, C, H, W)
        for s in self.stages:
            x = s(x)
        x = self.tail(x)
        H2, W2 = x.shape[-2], x.shape[-1]
        x = x.reshape(B, P, -1, H2, W2).transpose(0, 1, 3, 4, 2)
        return x


class PatchRecover(nn.Cell):
    """Patch Recover module."""
    def __init__(self, channels, h_size, w_size, level_feature_size, pressure_level_num, kernel_size):
        super().__init__()
        self.channels = channels
        self.h_size = h_size
        self.w_size = w_size
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(channels,
                              level_feature_size * kernel_size[0] * kernel_size[1] * kernel_size[2],
                              kernel_size=1,
                              stride=1,
                              group=1,
                              has_bias=True)

    def construct(self, x):
        """Patch Recover forward function."""
        batch_size, _, _, _, _ = x.shape
        x = x.transpose(0, 4, 1, 2, 3)
        x_3d = x.reshape(batch_size, self.channels, -1)
        x_3d = self.proj(x_3d)
        pz = self.pressure_level_num // self.kernel_size[0]
        hh = self.h_size // self.kernel_size[1]
        ww = self.w_size // self.kernel_size[2]
        x_3d = x_3d.reshape(batch_size,
                            self.level_feature_size,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            pz, hh, ww).transpose(0, 1, 5, 2, 6, 3, 7, 4)

        x_3d = x_3d.reshape(batch_size, self.level_feature_size, self.pressure_level_num, self.h_size, self.w_size)
        output = x_3d.reshape([batch_size, self.level_feature_size, self.pressure_level_num, self.h_size, self.w_size])
        
        return output