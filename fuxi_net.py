import mindspore.numpy as mnp
from mindspore import nn, ops
from fuxi import PatchRecover, CubeEmbed
from fuxi import BaseBlock, UpSamplePS, DownSample


class FuXiNet(nn.Cell):
    """
    优化的 FuXiNet，用于海洋表层超分
    """

    def __init__(
        self,
        depths=6,
        in_channels=96,
        out_channels=192,
        low_h=256,
        low_w=256,
        low_z=1,
        high_h=1024,
        high_w=1024,
        high_z=1,
        out_h=785,
        out_w=625,
        in_feature_size=5,
        out_feature_size=1,
        batch_size=1,
        kernel_size=(1,4,4),
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_feature_size = in_feature_size
        self.batch_size = batch_size

        # CubeEmbed: 针对表层数据优化
        self.cube_embed = CubeEmbed(
            in_channels=in_channels,
            h_size=low_h,
            w_size=low_w,
            level_feature_size=in_feature_size,
            pressure_level_num=low_z,
            batch_size=batch_size,
        )
        
        # 下采样
        self.down_sample = DownSample(
            in_channels=in_channels,
            out_channels=out_channels
        )

        # Enhanced Swin Transformer Blocks - 修改窗口大小处理
        sh = max(1, low_h // 8)  # 确保不会为0
        sw = max(1, low_w // 8)
        # 表层数据特殊处理 - 深度维度固定为1，不参与分块
        ws = (1, min(sh, 8), min(sw, 8))  # 限制窗口最大尺寸
        
        self.swin_blocks = nn.CellList([
            BaseBlock(
                in_channels=out_channels,
                input_shape=[1, sh, sw],  # 固定深度为1
                window_size=ws,
                recompute=True
            ) for _ in range(depths)
        ])

        # 上采样部分
        self.up_sample = UpSamplePS(
            in_channels=out_channels * 2,
            out_channels=out_channels
        )
      
        # PatchRecover: 恢复到高分辨率网格
        self.patch_recover = PatchRecover(
            out_channels,
            low_h * kernel_size[1],
            low_w * kernel_size[2],
            out_feature_size,
            low_z * kernel_size[0],
            kernel_size
        )
        
        self.interpolate = ops.ResizeBilinearV2()
        self.out_size = (out_h, out_w)

        # 添加数值保护的 clip 操作
        self.clip = ops.clip_by_value
        self.eps = 1e-6

    def construct(self, inputs):
        # 添加输入保护
        inputs = self.clip(inputs, -1e6, 1e6)
        
        # 1. Cube Embedding
        out = self.cube_embed(inputs)  # (B,C,1,H,W)
        out = self.clip(out, -1e6, 1e6)
        
        # 2. 下采样
        out_ds = self.down_sample(out)  # (B,1,H/8,W/8,C)
        out_ds = self.clip(out_ds, -1e6, 1e6)
        B, Z, H, W, C = out_ds.shape

        # 3. Swin Block输入 reshape: (B, Z*H*W, C)
        x = ops.reshape(out_ds, (B, Z * H * W, C))
        out_skip = x  # skip connection
        
        # 4. Swin Blocks
        for block in self.swin_blocks:
            x = block(x, B, Z, H, W)
            x = self.clip(x, -1e6, 1e6)
        
        # 5. 跳跃连接: concat 在最后一维 (特征维度)
        x = ops.concat((out_skip, x), axis=-1)  # (B, Z*H*W, 2*C)
        x = ops.reshape(x, (B, Z, H, W, 2 * C))
        x = self.clip(x, -1e6, 1e6)
        
        # 6. Upsample
        out_us = self.up_sample(x)  # (B,1,H*8,W*8,C)
        out_us = self.clip(out_us, -1e6, 1e6)
        
        # 7. Patch recover
        out_pr = self.patch_recover(out_us)  # (B,C,1,H,W)
        out_pr = self.clip(out_pr, -1e6, 1e6)
        B2, C2, Z2, H2, W2 = out_pr.shape
        
        # 8. Final reshape and interpolate
        out_r = ops.reshape(out_pr, (B2, C2 * Z2, H2, W2))
        out_interp = self.interpolate(out_r, self.out_size)
        out_interp = self.clip(out_interp, -1e6, 1e6)
        
        # 9. Final output reshape
        out_final = ops.reshape(out_interp, (B2, C2, 1, self.out_size[0], self.out_size[1]))
        return out_final