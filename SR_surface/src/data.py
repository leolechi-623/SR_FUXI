"""
表层超分数据集模块
"""

import numpy as np
import mindspore as ms
from mindspore.dataset import Dataset
import os


class SurfaceSRDataset:
    """表层超分数据集"""

    def __init__(self, low_res_data, high_res_data, transform=None):
        """
        参数:
            low_res_data: (N, C, H, W) 低分辨率表层数据
            high_res_data: (N, C, H, W) 高分辨率表层数据
            transform: 数据变换函数
        """
        self.low_res = low_res_data
        self.high_res = high_res_data
        self.transform = transform
        self.num_samples = low_res_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        low_res = self.low_res[idx]
        high_res = self.high_res[idx]

        if self.transform:
            low_res, high_res = self.transform(low_res, high_res)

        return low_res, high_res


class SurfaceDataLoader:
    """表层数据加载器"""

    @staticmethod
    def load_from_npy(low_res_path, high_res_path):
        """从 npy 文件加载数据"""
        low_res = np.load(low_res_path)
        high_res = np.load(high_res_path)
        return low_res, high_res

    @staticmethod
    def load_from_nc(nc_files, variables, patch_size=256):
        """从 NetCDF 文件加载数据"""
        try:
            import netCDF4
        except ImportError:
            raise ImportError("netCDF4 is required to load NetCDF files")

        data_list = []
        for nc_file in nc_files:
            ds = netCDF4.Dataset(nc_file)
            patch = []
            for var in variables:
                if var in ds.variables:
                    patch.append(ds.variables[var][:])
                else:
                    raise ValueError(f"Variable {var} not found in {nc_file}")
            ds.close()
            data_list.append(np.stack(patch, axis=0))

        return np.stack(data_list, axis=0)

    @staticmethod
    def normalize(data, mean=None, std=None):
        """数据标准化"""
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std()

        normalized = (data - mean) / (std + 1e-6)
        return normalized, mean, std

    @staticmethod
    def denormalize(data, mean, std):
        """反标准化"""
        return data * std + mean


class SurfaceDataTransform:
    """数据变换"""

    @staticmethod
    def random_flip(lr, hr):
        """随机翻转"""
        if np.random.rand() > 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
        if np.random.rand() > 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        return lr, hr

    @staticmethod
    def random_rotation(lr, hr, angles=[0, 90, 180, 270]):
        """随机旋转"""
        angle = np.random.choice(angles)
        lr = np.rot90(lr, angle // 90)
        hr = np.rot90(hr, angle // 90)
        return lr, hr

    @staticmethod
    def random_noise(lr, hr, noise_level=0.01):
        """添加随机噪声"""
        noise_lr = np.random.randn(*lr.shape) * noise_level * lr.std()
        lr = lr + noise_lr
        return lr, hr

    @staticmethod
    def compose(transforms):
        """组合变换"""

        def combined_transform(lr, hr):
            for transform in transforms:
                lr, hr = transform(lr, hr)
            return lr, hr

        return combined_transform


def create_mindspore_dataset(
    low_res, high_res, batch_size=1, shuffle=True, num_workers=1
):
    """创建 MindSpore 数据集"""

    def data_generator():
        indices = np.arange(len(low_res))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            yield low_res[idx], high_res[idx]

    from mindspore.dataset import GeneratorDataset

    dataset = GeneratorDataset(
        data_generator,
        column_names=["low_res", "high_res"],
        num_parallel_workers=num_workers,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
