# -*- coding: utf-8 -*-Parameter
"""
train_geoalign_lossflex.py  — 物理项鲁棒化 + 内域/模糊差分 + 课程调度（完整改好版）

特性：
  1) Charbonnier（鲁棒 L1）用于导数/旋度/散度/拉普拉斯误差
  2) 先 3x3 平均模糊，再差分，所有物理算子统一到 (H-4, W-4) 内域
  3) 相对归一化（基于 GT 的二阶矩）避免权重开巨量
  4) 自动权重 + 课程式调度（grad 早期 30% 力量；curl 随 ramp 打开）
  5) 验证增加 curl RMS 比值（动力结构指标）

运行时常用示例：
  # 纯 MSE 对照
  USE_MSE=1 USE_GRAD=0 USE_LAP=0 USE_PHY_DIV=0 USE_PHY_CURL=0 python3 -u train_geoalign_lossflex.py

  # 只用 grad+curl（建议）
  USE_MSE=1 USE_GRAD=1 USE_LAP=0 USE_PHY_DIV=0 USE_PHY_CURL=1 \
  W_GRAD=2.0 W_PHY_CURL=0.5 AUTO_W_MAX=30 CURRIC_START=0.2 CURRIC_END=0.7 \
  python3 -u train_geoalign_lossflex.py
"""

import os, json, csv, math
import numpy as np
import xarray as xr

# ==================== 设置环境变量 ====================
os.environ["GLOG_v"] = "3"  # 完全关闭 GLOG 输出
os.environ["MS_DEV_CLOSE_PREACTIVATE"] = "1"
os.environ["MS_DEV_DISABLE_COPY_ACTOR_WARNING"] = "1"

from mindspore import Parameter
import mindspore as ms
from mindspore import context, nn, ops, Tensor, dtype as mstype, dataset as ds, save_checkpoint
from mindspore.ops import clip_by_global_norm

# ==================== 运行环境 ====================
device_id = int(os.environ.get("DEVICE_ID", "0"))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id)
ms.set_seed(42); np.random.seed(42)

# ==================== 数据路径 ====================
FILE_LR = os.environ.get("FILE_LR", "/data/home/lheaw/fuxi/data/SCS_avg_10km_yr03_yr05_256.nc")
HR_FILES = os.environ.get("HR_FILES", "").split() or [
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr03_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr03_p2.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr04_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr04_p2.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr05_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr05_p2.nc",
]
in_vars  = ['u','v','w','salt','z_rho','temp']
out_vars = ['u','v']  # 物理项只约束 u,v

# ==================== 训练超参 ====================
n_train = int(os.environ.get("N_TRAIN", "72"))
n_test  = int(os.environ.get("N_TEST",  "36"))
batch = int(os.environ.get("BATCH", "1"))
epochs = int(os.environ.get("EPOCHS", "150"))
base_lr = float(os.environ.get("BASE_LR", "1e-5"))
grad_clip_norm = float(os.environ.get("GRAD_CLIP", "0.5"))

# 输出
EXP_NAME  = os.environ.get("EXP_NAME",  "exp")
OUT_DIR   = os.environ.get("OUT_DIR",   f"./runs/{EXP_NAME}")
OUT_PREF  = os.environ.get("OUT_PREFIX", EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

# ==================== 损失开关与权重 ====================
def _flag(name, default):  return int(os.environ.get(name, str(default))) == 1
def _weight(name, default): return float(os.environ.get(name, str(default)))


# === 损失开关与权重 ===
USE_MSE     = _flag("USE_MSE",     1)
USE_MAE     = _flag("USE_MAE",     0)   # <--- 新增：是否用 MAE（与 USE_MSE 互斥，二选一）
MAE_WEIGHT_MODE = os.environ.get("MAE_WEIGHT_MODE", "curriculum").lower()  # "curriculum" | "uncertainty"

USE_GRAD    = _flag("USE_GRAD",    1)
USE_LAP     = _flag("USE_LAP",     0)
USE_PHY_DIV = _flag("USE_PHY_DIV", 0)
USE_PHY_CURL= _flag("USE_PHY_CURL",0)
USE_NORMED_LOSS = _flag("USE_NORMED_LOSS", 1)

USE_TS_GRAD = _flag("USE_TS_GRAD", 1)       # 新增
USE_TS_LAP  = _flag("USE_TS_LAP",  1)       # 新增
W_TS_GRAD   = _weight("W_TS_GRAD", 0.08)    # 建议 0.05~0.10
W_TS_LAP    = _weight("W_TS_LAP",  0.03)    # 建议 0.02~0.05


W_GRAD = _weight("W_GRAD", 0.0)
W_LAP  = _weight("W_LAP",  0.0)
W_DIV  = _weight("W_PHY_DIV", 0.0)
W_CURL = _weight("W_PHY_CURL",0.0)

AUTO_W_GRAD = _flag("AUTO_W_GRAD", 1)
AUTO_W_LAP  = _flag("AUTO_W_LAP",  1)
AUTO_W_DIV  = _flag("AUTO_W_DIV",  1)
AUTO_W_CURL = _flag("AUTO_W_CURL", 1)
AUTO_W_MIN  = float(os.environ.get("AUTO_W_MIN", "0.05"))
AUTO_W_MAX  = float(os.environ.get("AUTO_W_MAX", "50"))  # 建议 ≤ 50

# 课程式调度（按 epoch 百分比）
CURRIC_START = float(os.environ.get("CURRIC_START", "0.4"))
CURRIC_END   = float(os.environ.get("CURRIC_END",   "0.8"))

PRINT_LOSS_BREAKDOWN = _flag("PRINT_LOSS_BREAKDOWN", 1)

print(f"[loss switches] MSE:{USE_MSE} MAE:{USE_MAE} | GRAD:{USE_GRAD} | LAP:{USE_LAP} | DIV:{USE_PHY_DIV} | CURL:{USE_PHY_CURL}")
print(f"[normed] USE_NORMED_LOSS={USE_NORMED_LOSS}, AUTO_W in [{AUTO_W_MIN}, {AUTO_W_MAX}]")

# --- Added: ensure PHY_BLUR is defined early to avoid NameError during module import/compile ---
PHY_BLUR = int(os.environ.get("PHY_BLUR", "3"))  # allowed: 3 or 5
print(f"[PHY_BLUR] Using {PHY_BLUR}x{PHY_BLUR} blur kernel (PHY_BLUR={PHY_BLUR})")

# --- Added: define blur_flag Parameter immediately so graph compile can see it ---
# blur_flag = 1.0 -> use blur5; = 0.0 -> use blur3. It is non-trainable.
default_blur_flag = 1.0 if PHY_BLUR == 5 else 0.0
blur_flag = Parameter(Tensor(default_blur_flag, mstype.float32), name="blur_flag", requires_grad=False)
print(f"[PHY_BLUR] initial blur_flag set to {default_blur_flag} (1.0=blur5,0.0=blur3)")

# ==================== I/O 打开 NetCDF ====================
def _open_dataset_safe(path):
    # LR：单文件
    return xr.open_dataset(
        path,
        engine="netcdf4",        # ✅ 换成 netCDF4
        decode_times=False,
        chunks=None,  # 禁用分块
        lock=False    # 禁用锁
    )

def _open_mfdataset_safe(paths):
    # HR：多文件 - 逐个处理
    print(f"逐个处理 {len(paths)} 个 HR 文件")
    datasets = []
    for i, path in enumerate(paths):
        print(f"  处理文件 {i+1}/{len(paths)}: {os.path.basename(path)}")
        try:
            ds = xr.open_dataset(
                path,
                engine="netcdf4",
                decode_times=False,
                chunks=None,  # 禁用分块
                lock=False    # 禁用锁
            )
            datasets.append(ds)
        except Exception as e:
            print(f"  跳过文件 {path}: {e}")
            continue
    
    if not datasets:
        raise RuntimeError("没有成功加载任何 HR 文件")
    
    # 合并数据集
    combined = xr.concat(datasets, dim="ocean_time")
    return combined

_Z_CANDIDATES  = ("z","depth","lev","level","layer","s_rho","sigma")
_T_CANDIDATES  = ("time","ocean_time","record","t")

def _find_dim(da, keys):
    for d in da.dims:
        if any(k in d.lower() for k in keys): return d
    return None

SURFACE_ONLY=True
def _pick_surface(da):
    zdim=_find_dim(da,_Z_CANDIDATES)
    if zdim is None: return da
    # 使用索引 -1 作为表层
    surface_idx = -1
    print(f"提取表层数据: 变量 {da.name}, 垂直维度 {zdim}, 使用索引 {surface_idx}")
    return da.isel({zdim:surface_idx}).expand_dims({zdim:[surface_idx]})

def _to_NZHW(da):
    zdim=_find_dim(da,_Z_CANDIDATES)
    tdim=_find_dim(da,_T_CANDIDATES)
    order=[]
    if tdim: order.append(tdim)
    if zdim: order.append(zdim)
    order.extend([d for d in da.dims if d not in order])
    return da.transpose(*order)

def valid_mask_np(a, thr=100.0):
    if isinstance(a,np.ma.MaskedArray): a=a.filled(np.nan)
    a=a.astype(np.float32)
    a=np.where(np.abs(a)<=thr, a, np.nan)
    return a

print("Loading LR/HR ...")

# LR 数据加载保持不变
with _open_dataset_safe(FILE_LR) as ds_lr:
    arrs_lr=[]
    for v in in_vars:
        print(f"处理 LR 变量: {v}")
        da=ds_lr[v]
        if SURFACE_ONLY: da=_pick_surface(da)
        da=_to_NZHW(da)
        tdim=_find_dim(da,_T_CANDIDATES) or da.dims[0]
        da=da.isel({tdim: slice(0, n_train+n_test)})
        a=valid_mask_np(da.load().values)  # 立即加载
        if a.ndim==3: a=a[:,None,:,:]
        arrs_lr.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
    lr_all=np.stack(arrs_lr, axis=1)

# HR 数据：逐个文件处理，避免内存不足
print("逐个处理 HR 文件（只提取表层）...")
hr_files = [
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr03_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr03_p2.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr04_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr04_p2.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr05_p1.nc",
    "/data/home/lheaw/fuxi/data/SCS_avg_3km_yr05_p2.nc"
]

arrs_hr_all = []  # 存储所有文件的 HR 数据
arrs_mask_all = []  # 存储所有文件的 mask 数据

for i, hr_file in enumerate(hr_files):
    print(f"处理 HR 文件 {i+1}/{len(hr_files)}: {os.path.basename(hr_file)}")
    
    try:
        with _open_dataset_safe(hr_file) as ds_hr_single:
            arrs_hr = []
            arrs_mask = []
            
            for v in out_vars:
                print(f"  提取变量: {v}")
                da = ds_hr_single[v]
                
                # 立即提取表层数据
                if SURFACE_ONLY:
                    da = _pick_surface(da)
                da = _to_NZHW(da)
                
                # 只提取需要的時間步
                tdim = _find_dim(da, _T_CANDIDATES) or da.dims[0]
                total_time = da.sizes[tdim]
                time_slice = slice(0, min(n_train+n_test, total_time))
                da = da.isel({tdim: time_slice})
                
                # 立即加载数据到内存
                a = da.load().values
                if isinstance(a, np.ma.MaskedArray): 
                    a = a.filled(np.nan)
                if a.ndim == 3: 
                    a = a[:, None, :, :]
                
                arrs_hr.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
                arrs_mask.append(~np.isnan(a))
            
            # 堆叠当前文件的变量
            hr_single = np.stack(arrs_hr, axis=1)  # (N,4,1,H,W)
            mask_single = np.stack(arrs_mask, axis=1).astype(np.float32)
            
            arrs_hr_all.append(hr_single)
            arrs_mask_all.append(mask_single)
            
            print(f"  ✓ 完成: {hr_single.shape}")
            
    except Exception as e:
        print(f"  ✗ 文件 {hr_file} 处理失败: {e}")
        print("  跳过此文件继续...")
        continue

# 合并所有文件的数据
if arrs_hr_all:
    hr_all = np.concatenate(arrs_hr_all, axis=0)
    valid_all = np.concatenate(arrs_mask_all, axis=0)
    print(f"HR 数据合并完成: {hr_all.shape}")
else:
    print("错误: 没有成功加载任何 HR 文件")
    exit(1)

mask_all = valid_all.astype(np.float32)

# ======= 时间切分 =======
N_AVAIL = min(int(lr_all.shape[0]), int(hr_all.shape[0]))
n_test  = min(max(1, n_test), max(1, N_AVAIL - 1))
n_train = N_AVAIL - n_test

Xtr, Ytr, Mtr = lr_all[:n_train], hr_all[:n_train], mask_all[:n_train]
Xte, Yte, Mte = lr_all[-n_test:], hr_all[-n_test:], mask_all[-n_test:]

# ======= 标准化 =======
def zscore_fit(x):
    mean=np.nanmean(x, axis=(0,2,3,4), keepdims=True)
    std =np.nanstd( x, axis=(0,2,3,4), keepdims=True)
    std =np.where(std<1e-6,1e-6,std)
    return mean.astype(np.float32), std.astype(np.float32)
def zscore_apply(x, mean, std):
    x=np.nan_to_num(x, nan=0.0); return (x-mean)/std
def minmax_fit_multi(y, mask):
    C=y.shape[1]
    y_min=np.zeros((1,C,1,1,1), dtype=np.float32)
    y_max=np.ones( (1,C,1,1,1), dtype=np.float32)
    for c in range(C):
        valid=mask[:,c]>0.5
        y_valid=y[:,c][valid]
        if y_valid.size==0:
            y_min[0,c,0,0,0]=0.0; y_max[0,c,0,0,0]=1.0
        else:
            y_min[0,c,0,0,0]=np.percentile(y_valid,0.5)
            y_max[0,c,0,0,0]=np.percentile(y_valid,99.5)
    span=np.maximum(y_max-y_min,1e-6)
    return y_min,y_max,span
def minmax_apply_multi(y, y_min, y_span):
    return ((y-y_min)/y_span).astype(np.float32)

x_mean,x_std=zscore_fit(Xtr)
Xtr_n,Xte_n=zscore_apply(Xtr,x_mean,x_std), zscore_apply(Xte,x_mean,x_std)
y_min,y_max,y_span=minmax_fit_multi(Ytr, Mtr)
Ytr_n,Yte_n=minmax_apply_multi(Ytr,y_min,y_span), minmax_apply_multi(Yte,y_min,y_span)

np.savez(os.path.join(OUT_DIR, f"{OUT_PREF}_scalers.npz"),
         x_mean=x_mean, x_std=x_std, y_min=y_min, y_max=y_max, y_span=y_span)

# ======= Dataset =======
train_ds=ds.NumpySlicesDataset((Xtr_n,Ytr_n,Mtr), column_names=["lr","hr","mask"], shuffle=True).batch(batch)
test_ds =ds.NumpySlicesDataset((Xte_n,Yte_n,Mte), column_names=["lr","hr","mask"], shuffle=False).batch(batch)
steps_per_epoch=train_ds.get_dataset_size()

# ==================== 网络 ====================
from fuxi_net import FuXiNet
net = FuXiNet(
    depths=6, in_channels=96, out_channels=192,
    low_h=Xtr_n.shape[-2], low_w=Xtr_n.shape[-1], low_z=1,
    high_h=Ytr_n.shape[-2], high_w=Ytr_n.shape[-1], high_z=1,
    out_h=Ytr_n.shape[-2], out_w=Ytr_n.shape[-1],
    in_feature_size=len(in_vars),
    out_feature_size=len(out_vars),
    batch_size=batch, kernel_size=(1,4,4)
)
net.to_float(mstype.float32)
net.set_train(True)

# ===== resume from checkpoint (if provided) =====
from mindspore import load_checkpoint, load_param_into_net
ckpt_path = os.environ.get("CKPT_RESUME", "").strip()
if ckpt_path:
    import os as _os
    if not _os.path.isabs(ckpt_path):
        ckpt_path = _os.path.abspath(ckpt_path)
    if _os.path.exists(ckpt_path):
        print(f"[resume] loading checkpoint from: {ckpt_path}")
        params = load_checkpoint(ckpt_path)
        not_loaded, _ = load_param_into_net(net, params)
        if not_loaded:
            print("[resume][warn] not loaded:", not_loaded)
        else:
            print("[resume] all params loaded.")
    else:
        print(f"[resume][ERROR] ckpt not found: {ckpt_path}")


# ==================== 工具函数（模糊+中心差分+掩膜） ====================
logical_and = ops.LogicalAnd()
reduce_max_keep = ops.ReduceMax(keep_dims=True)

def _charbonnier(x, eps=1e-6):
    return ops.sqrt(x * x + Tensor(eps, mstype.float32))

# --- Replace slice-based blur/derivative with conv-based implementations ---
_pad2 = ops.Pad(((0,0),(0,0),(1,1),(1,1)))  # 在 H,W 维各 pad 1

# convolution kernels (will be applied per-channel by reshaping (N,C,H,W) -> (N*C,1,H,W))
_k_avg3 = Tensor(np.ones((1,1,3,3), dtype=np.float32) / 9.0, mstype.float32)
_k_avg5 = Tensor(np.ones((1,1,5,5), dtype=np.float32) / 25.0, mstype.float32)

# central difference kernels (3x3) that produce the same valid-domain reduction as original slicing
_k_gx = Tensor(np.array([[[[0.0, 0.0, 0.0],
                           [-0.5, 0.0, 0.5],
                           [0.0, 0.0, 0.0]]]], dtype=np.float32), mstype.float32)
_k_gy = Tensor(np.array([[[[0.0, -0.5, 0.0],
                           [0.0,  0.0, 0.0],
                           [0.0,  0.5, 0.0]]]], dtype=np.float32), mstype.float32)

# Laplacian kernel matching dxx + dyy used previously (valid conv produces H-4,W-4)
_k_lap = Tensor(np.array([[[[0.0, 1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0, 1.0, 0.0]]]], dtype=np.float32), mstype.float32)

def _conv_per_channel_valid(x, kernel):
    """Apply 2D conv 'valid' with kernel to each channel independently.
       x: Tensor (N,C,H,W); kernel: Tensor (1,1,kh,kw)
       returns Tensor (N,C,H-kh+1, W-kw+1)
    """
    # Graph-safe shape accesses
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[-2]
    W = x.shape[-1]

    # kernel sizes are constants (kernel is a Tensor created at module import)
    kh = int(kernel.shape[-2])
    kw = int(kernel.shape[-1])

    # reshape to (N*C, 1, H, W)
    x2 = x.reshape((N * C, 1, H, W))

    # use 'same' conv to avoid "input smaller than kernel" errors, then crop center to emulate 'valid'
    out_same = ops.conv2d(x2, kernel, stride=(1, 1), pad_mode='same')

    # compute valid output size (may be <= 1 if input smaller than kernel, clamp to at least 1)
    Ho = max(1, H - kh + 1)
    Wo = max(1, W - kw + 1)

    # center start indices for cropping
    sh = (kh - 1) // 2
    sw = (kw - 1) // 2

    # slice out center valid region: out_same[:, :, sh:sh+Ho, sw:sw+Wo]
    out_cropped = out_same[:, :, sh:sh + Ho, sw:sw + Wo]

    # reshape back to (N, C, Ho, Wo)
    out = out_cropped.reshape((N, C, Ho, Wo))
    return out

def _blur3(x):
    # x: (N,C,H,W) -> valid conv with 3x3 -> (N,C,H-2,W-2)
    return _conv_per_channel_valid(x, _k_avg3)

def _blur5(x):
    # x: (N,C,H,W) -> valid conv with 5x5 -> (N,C,H-4,W-4)
    b5 = _conv_per_channel_valid(x, _k_avg5)
    # pad to match 3x3's (H-2,W-2) as original code did
    return _pad2(b5)

# blur selector (only calls _blur5 when PHY_BLUR==5)
def _blurK(x):
    if PHY_BLUR == 5:
        b3 = _blur3(x)              # (..., H-2, W-2)
        b5 = _blur5(x)              # (..., H-2, W-2) after pad
        return blur_flag * b5 + (1.0 - blur_flag) * b3
    else:
        return _blur3(x)


# ↓↓↓ 关键：根据核大小，自适应“内域”裁剪宽度
# 总收缩 = (模糊收缩) + (中心差分额外收缩) = (PHY_BLUR-1) + 2
_TOTAL_SHRINK = (PHY_BLUR - 1) + 2          # k=3 -> 4, k=5 -> 6
_CROP = _TOTAL_SHRINK // 2                  # 每边裁掉像素数：k=3 -> 2, k=5 -> 3

# --- Ensure inner_mask_hwK is defined early so graph compilation can see it ---
def inner_mask_hwK(m_bool):
    """Crop mask edges with dimension checks to avoid zero-size dimensions"""
    if _CROP <= 0:
        return m_bool
    
    # Get shape safely
    H = m_bool.shape[-2]
    W = m_bool.shape[-1]
    
    # Ensure crop size doesn't exceed tensor dimensions
    h_crop = min(_CROP, H // 2)  # 不能裁剪超过一半
    w_crop = min(_CROP, W // 2)

    # Calculate end indices (must leave at least 1 element)
    h_end = max(H - h_crop, h_crop + 1)
    w_end = max(W - w_crop, w_crop + 1)
    
    # Use safe indices for cropping
    return m_bool[..., h_crop:h_end, w_crop:w_end]


# ======= 以 conv 实现的差分 / 散度 / 旋度 / 拉普拉斯（输出对齐 (H-4, W-4)）=======
def _grad_center_blur(x):
    x4 = x.squeeze(2) if x.ndim == 5 else x        # (N,C,H,W)
    xb = _blurK(x4)                                 # (N,C,H-2,W-2)
    gx = _conv_per_channel_valid(xb, _k_gx)         # (N,C,H-4,W-4)
    gy = _conv_per_channel_valid(xb, _k_gy)         # (N,C,H-4,W-4)
    return gx, gy

def _div_center_blur(u, v):
    u4 = u.squeeze(2) if u.ndim == 5 else u
    v4 = v.squeeze(2) if v.ndim == 5 else v
    ub = _blurK(u4); vb = _blurK(v4)
    ux = _conv_per_channel_valid(ub, _k_gx)
    vy = _conv_per_channel_valid(vb, _k_gy)
    return ux + vy                                   # (N,1,H-4,W-4)

def _curl_center_blur(u, v):
    u4 = u.squeeze(2) if u.ndim == 5 else u
    v4 = v.squeeze(2) if v.ndim == 5 else v
    ub = _blurK(u4); vb = _blurK(v4)
    vx = _conv_per_channel_valid(vb, _k_gx)
    uy = _conv_per_channel_valid(ub, _k_gy)
    return vx - uy                                    # (N,1,H-4,W-4)

def _lap_center_blur(x):
    x4 = x.squeeze(2) if x.ndim == 5 else x
    xb = _blurK(x4)
    lap = _conv_per_channel_valid(xb, _k_lap)
    return lap                                       # (N,C,H-4,W-4)

# ======= 参考尺度（相对归一化）=======
def _grad_ref(t_uv):  # t_uv: (N,2,1,H,W) 或 (N,2,H,W)
    t2 = t_uv[:, :2]
    gx, gy = _grad_center_blur(t2)
    return ((gx*gx).mean() + (gy*gy).mean())/2.0 + Tensor(1e-12, mstype.float32)

def _lap_ref(t_uv):
    lap = _lap_center_blur(t_uv[:,0:1]) + _lap_center_blur(t_uv[:,1:2])
    return (lap*lap).mean() + Tensor(1e-12, mstype.float32)

def _div_ref(ut, vt):
    div = _div_center_blur(ut, vt)
    return (div*div).mean() + Tensor(1e-12, mstype.float32)

def _curl_ref(ut, vt):
    curl = _curl_center_blur(ut, vt)
    return (curl*curl).mean() + Tensor(1e-12, mstype.float32)

# ======= 损失函数（Charbonnier + 内域/模糊）=======
def grad_loss(pred_n, tgt_n, mask):
    p = pred_n[:, :2]; t = tgt_n[:, :2]
    m_bool = (mask[:, :2] > 0.5)

    gx_p_u, gy_p_u = _grad_center_blur(p[:,0:1])
    gx_t_u, gy_t_u = _grad_center_blur(t[:,0:1])
    gx_p_v, gy_p_v = _grad_center_blur(p[:,1:2])
    gx_t_v, gy_t_v = _grad_center_blur(t[:,1:2])

    ex = ops.concat([gx_p_u - gx_t_u, gx_p_v - gx_t_v], axis=1)  # (N,2,H-4,W-4)
    ey = ops.concat([gy_p_u - gy_t_u, gy_p_v - gy_t_v], axis=1)

    if USE_NORMED_LOSS:
        ref = _grad_ref(t)
        sref = ops.sqrt(ref)
        ex = ex / sref; ey = ey / sref

    err = _charbonnier(ex) + _charbonnier(ey)        # (N,2,H-4,W-4)
    mm = inner_mask_hwK(m_bool)                      # (N,2,H-4,W-4)
    mm_any = reduce_max_keep(ops.cast(mm, mstype.float32), 1) > 0.5
    loss, cnt = _masked_mean(err, mm_any)
    full_cnt = _full_valid(mask[:, :2])
    return loss * (full_cnt / (cnt + Tensor(1e-6, mstype.float32)))

def lap_loss(pred_n, tgt_n, mask):
    p = pred_n[:, :2]; t = tgt_n[:, :2]
    m_bool = (mask[:, :2] > 0.5)
    lap_p = _lap_center_blur(p[:,0:1]) + _lap_center_blur(p[:,1:2])
    lap_t = _lap_center_blur(t[:,0:1]) + _lap_center_blur(t[:,1:2])
    raw = lap_p - lap_t
    if USE_NORMED_LOSS:
        raw = raw / ops.sqrt(_lap_ref(t))
    err = _charbonnier(raw)
    mm = inner_mask_hwK(m_bool)
    mm_any = reduce_max_keep(ops.cast(mm, mstype.float32), 1) > 0.5
    loss, cnt = _masked_mean(err, mm_any)
    full_cnt = _full_valid(mask[:, :2])
    return loss * (full_cnt / (cnt + Tensor(1e-6, mstype.float32)))

def ts_grad_loss(pred_n, tgt_n, mask):
    # 通道 2: temp, 3: salt（使用归一化空间，内部做相对归一化）
    p = pred_n[:, 2:4]; t = tgt_n[:, 2:4]
    m_bool = (mask[:, 2:4] > 0.5)

    gx_p_t, gy_p_t = _grad_center_blur(p[:,0:1])
    gx_t_t, gy_t_t = _grad_center_blur(t[:,0:1])
    gx_p_s, gy_p_s = _grad_center_blur(p[:,1:2])
    gx_t_s, gy_t_s = _grad_center_blur(t[:,1:2])

    ex = ops.concat([gx_p_t - gx_t_t, gx_p_s - gx_t_s], axis=1)
    ey = ops.concat([gy_p_t - gy_t_t, gy_p_s - gy_t_s], axis=1)

    # 相对归一化（参照 GT 的梯度能量）
    ref = ((gx_t_t*gx_t_t).mean() + (gy_t_t*gy_t_t).mean() +
           (gx_t_s*gx_t_s).mean() + (gy_t_s*gy_t_s).mean())/2.0 + Tensor(1e-12, mstype.float32)
    sref = ops.sqrt(ref); ex = ex / sref; ey = ey / sref

    err = _charbonnier(ex) + _charbonnier(ey)
    mm = inner_mask_hwK(m_bool)
    mm_any = reduce_max_keep(ops.cast(mm, mstype.float32), 1) > 0.5
    loss, cnt = _masked_mean(err, mm_any)
    full_cnt = _full_valid(mask[:, 2:4])
    return loss * (full_cnt / (cnt + Tensor(1e-6, mstype.float32)))

def ts_lap_loss(pred_n, tgt_n, mask):
    p = pred_n[:, 2:4]; t = tgt_n[:, 2:4]
    m_bool = (mask[:, 2:4] > 0.5)
    lap_p = _lap_center_blur(p[:,0:1]) + _lap_center_blur(p[:,1:2])
    lap_t = _lap_center_blur(t[:,0:1]) + _lap_center_blur(t[:,1:2])
    raw = lap_p - lap_t
    ref = (lap_t*lap_t).mean() + Tensor(1e-12, mstype.float32)
    raw = raw / ops.sqrt(ref)
    err = _charbonnier(raw)
    mm = inner_mask_hwK(m_bool)
    mm_any = reduce_max_keep(ops.cast(mm, mstype.float32), 1) > 0.5
    loss, cnt = _masked_mean(err, mm_any)
    full_cnt = _full_valid(mask[:, 2:4])
    return loss * (full_cnt / (cnt + Tensor(1e-6, mstype.float32)))

def phy_curl_loss(pred_n, tgt_n, mask):
    up = pred_n[:,0:1]; vp = pred_n[:,1:2]
    ut = tgt_n[:,0:1]; vt = tgt_n[:,1:2]
    m_uv = logical_and(mask[:,0:1] > 0.5, mask[:,1:2] > 0.5)

    curl_p = _curl_center_blur(up, vp)
    curl_t = _curl_center_blur(ut, vt)
    raw = curl_p - curl_t
    if USE_NORMED_LOSS:
        raw = raw / ops.sqrt(_curl_ref(ut, vt))
    err = _charbonnier(raw)
    mm = inner_mask_hwK(m_uv)
    loss, cnt = _masked_mean(err, mm)
    full_cnt = _full_valid(mask[:, :2])
    return loss * (full_cnt / (cnt + Tensor(1e-6, mstype.float32)))

EPS = 1e-12

# ======= 掩膜加权均值/计数 =======
def _masked_mean(x, mask_bool):
    x = ops.cast(x, mstype.float32)
    m = ops.cast(mask_bool, mstype.float32)
    # —— 新增：若掩膜比数据多一个 Z=1 维，去掉 —— 
    if len(m.shape) == len(x.shape) + 1 and m.shape[2] == 1:
        m = ops.squeeze(m, axis=2)
    cnt = m.sum()
    if (cnt <= 0):
        return ops.zeros((), mstype.float32), cnt
    mean_val = (x * m).sum() / (cnt + Tensor(1e-6, mstype.float32))
    return mean_val, cnt

def _full_valid(mask):
    return ops.reduce_sum(mask.astype(mstype.float32)) + Tensor(1e-6, mstype.float32)

# ==================== 组装损失 & 自动权重 ====================
# 顶部（或开关区）确保有：
MSE_WEIGHT_MODE = os.environ.get("MSE_WEIGHT_MODE", "curriculum").lower()
# === Uncertainty 学习策略（控制/冻结）===
LEARN_UNCERT = int(os.environ.get("LEARN_UNCERT", "0")) == 1  # 默认为 0：冻结
LS_UV_MIN = float(os.environ.get("LS_UV_MIN", "-1.2"))
LS_UV_MAX = float(os.environ.get("LS_UV_MAX", "0.3"))
LS_TS_MIN = float(os.environ.get("LS_TS_MIN", "0.3"))
LS_TS_MAX = float(os.environ.get("LS_TS_MAX", "1.5"))

# ======= 反归一化 / Pixel loss helpers (MSE / MAE) =======
# Ensure these are defined before WithLossCell so they are available at graph-compile time.
y_min_T  = Tensor(y_min,  mstype.float32)
y_span_T = Tensor(y_span, mstype.float32)

def unnormalize_y(y_n):
    return y_n * y_span_T + y_min_T

def mse_masked_phys(pred_n, tgt_n, mask):
    pred = unnormalize_y(pred_n); tgt = unnormalize_y(tgt_n)
    diff = (pred - tgt) * mask
    return (diff * diff).sum() / (mask.sum() + Tensor(1e-6, mstype.float32))

def mse_groups_phys(pred_n, tgt_n, mask):
    pred = unnormalize_y(pred_n); tgt = unnormalize_y(tgt_n)
    diff2 = (pred - tgt) * (pred - tgt)               # (N,C,1,H,W) or (N,C,H,W)
    m = ops.cast(mask, mstype.float32)
    def _mean_over(idx):
        x = diff2[:, idx, :, :, :]
        w = m[:, idx, :, :, :]
        num = (x * w).sum()
        den = w.sum() + Tensor(1e-6, mstype.float32)
        return num / den
    L_uv = _mean_over([0, 1])     # u,v
    L_ts = _mean_over([2, 3])     # temp,salt
    return L_uv, L_ts

def mae_masked_phys(pred_n, tgt_n, mask):
    pred = unnormalize_y(pred_n); tgt = unnormalize_y(tgt_n)
    diff = ops.abs((pred - tgt) * mask)
    return diff.sum() / (mask.sum() + Tensor(1e-6, mstype.float32))

def mae_groups_phys(pred_n, tgt_n, mask):
    pred = unnormalize_y(pred_n); tgt = unnormalize_y(tgt_n)
    m = ops.cast(mask, mstype.float32)
    def _mean_over(idx):
        x = ops.abs(pred[:, idx] - tgt[:, idx])
        w = m[:, idx]
        num = (x * w).sum()
        den = w.sum() + Tensor(1e-6, mstype.float32)
        return num / den
    L_uv = _mean_over([0, 1])
    L_ts = _mean_over([2, 3])
    return L_uv, L_ts

class WithLossCell(nn.Cell):
    def __init__(self, network, init_w, mse_mode=MSE_WEIGHT_MODE):
        super().__init__()
        self.network = network
        self.mse_mode = mse_mode
        self.mae_mode = MAE_WEIGHT_MODE   # <--- 新增
        self.w_mse  = Parameter(Tensor(1.0, mstype.float32),  name="w_mse",  requires_grad=False)
        self.w_mae  = Parameter(Tensor(1.0, mstype.float32),  name="w_mae",  requires_grad=False)  
        self.w_grad = Parameter(Tensor(init_w["grad"], mstype.float32), name="w_grad", requires_grad=False)
        self.w_lap  = Parameter(Tensor(init_w["lap"],  mstype.float32), name="w_lap",  requires_grad=False)
        self.w_div  = Parameter(Tensor(init_w["div"],  mstype.float32), name="w_div",  requires_grad=False)  # 保留参数，但不会用
        self.w_curl = Parameter(Tensor(init_w["curl"], mstype.float32), name="w_curl", requires_grad=False)
        # uncertainty（可用可不用）
        self.log_sigma_uv = Parameter(Tensor(0.0, mstype.float32), name="log_sigma_uv")
        self.log_sigma_ts = Parameter(Tensor(0.0, mstype.float32), name="log_sigma_ts")
        ls_uv_init = float(os.environ.get("LOG_SIGMA_UV_INIT", "-0.7"))
        ls_ts_init = float(os.environ.get("LOG_SIGMA_TS_INIT", "0.7"))
        self.log_sigma_uv.set_data(Tensor(ls_uv_init, mstype.float32))
        self.log_sigma_ts.set_data(Tensor(ls_ts_init, mstype.float32))

    def construct(self, lr, hr, mask):
        pred = self.network(ops.cast(lr, mstype.float32))
        # ----- SANITIZE network outputs to avoid NaN/Inf propagating to loss -----
        # replace NaN -> 0, clip extreme values
        pred = ops.masked_fill(pred, ops.isnan(pred), Tensor(0.0, mstype.float32))
        pred = ops.clip_by_value(pred, Tensor(-1e6, mstype.float32), Tensor(1e6, mstype.float32))
        hr   = ops.masked_fill(hr,   ops.isnan(hr),   Tensor(0.0, mstype.float32))
        mask = ops.masked_fill(mask, ops.isnan(mask), Tensor(0.0, mstype.float32))

        # record simple diagnostics (will be returned as scalars in last_terms)
        p_max = ops.reduce_max(pred)
        p_min = ops.reduce_min(pred)
        p_sum = ops.reduce_sum(ops.abs(pred))
        nan_cnt = ops.reduce_sum(ops.cast(ops.isnan(pred), mstype.float32))
        # attach small diagnostics so caller can inspect via loss_net.last_terms
        diag_terms = [("pred_max", p_max), ("pred_min", p_min), ("pred_abs_sum", p_sum), ("pred_nan_cnt", nan_cnt)]
        # ------------------------------------------------------------------------
        total = ops.zeros((), mstype.float32)
        log_terms = []

        # --- Pixel term: MSE or MAE（二选一） ---
        if USE_MSE and not USE_MAE:
            if self.mse_mode == "uncertainty":
                L_uv, L_ts = mse_groups_phys(pred, hr, mask)
                ls_uv = ops.clip_by_value(self.log_sigma_uv,
                                          Tensor(LS_UV_MIN, mstype.float32),
                                          Tensor(LS_UV_MAX, mstype.float32))
                ls_ts = ops.clip_by_value(self.log_sigma_ts,
                                          Tensor(LS_TS_MIN, mstype.float32),
                                          Tensor(LS_TS_MAX, mstype.float32))
                if not LEARN_UNCERT:  # 冻结：只用你设定/裁剪后的值，不反传
                    ls_uv = ops.stop_gradient(ls_uv)
                    ls_ts = ops.stop_gradient(ls_ts)
                
                inv_var_uv = ops.exp(-2.0 * ls_uv)
                inv_var_ts = ops.exp(-2.0 * ls_ts)
                l_uv = 0.5 * inv_var_uv * L_uv + ls_uv
                l_ts = 0.5 * inv_var_ts * L_ts + ls_ts

                l_pix = (l_uv + l_ts) * self.w_mse
                total += l_pix
                log_terms += [("pix_uv", l_uv), ("pix_ts", l_ts)]
            else:
                l_pix = mse_masked_phys(pred, hr, mask) * self.w_mse
                total += l_pix; log_terms.append(("pix", l_pix))

        elif USE_MAE and not USE_MSE:
            if self.mae_mode == "uncertainty":
                L_uv, L_ts = mae_groups_phys(pred, hr, mask)
                ls_uv = ops.clip_by_value(self.log_sigma_uv, Tensor(-3.0, mstype.float32), Tensor(3.0, mstype.float32))
                ls_ts = ops.clip_by_value(self.log_sigma_ts, Tensor(-3.0, mstype.float32), Tensor(3.0, mstype.float32))
                inv_var_uv = ops.exp(-2.0 * ls_uv); inv_var_ts = ops.exp(-2.0 * ls_ts)
                l_uv = inv_var_uv * L_uv + ls_uv              # MAE 没 0.5 系数
                l_ts = inv_var_ts * L_ts + ls_ts
                l_pix = 0.5 * (l_uv + l_ts) * self.w_mae      # 与 MSE 量纲对齐，乘 0.5
                total += l_pix
                log_terms += [("pix_uv", l_uv), ("pix_ts", l_ts)]
            else:
                l_pix = mae_masked_phys(pred, hr, mask) * self.w_mae
                total += l_pix; log_terms.append(("pix", l_pix))
        else:
            # 若两个都开或都关，默认回退到 MSE
            l_pix = mse_masked_phys(pred, hr, mask) * self.w_mse
            total += l_pix; log_terms.append(("pix", l_pix))

        # --- 物理项（保持不变） ---
        if USE_GRAD:
            l_g = grad_loss(pred, hr, mask) * self.w_grad
            total += l_g; log_terms.append(("grad", l_g))
        if USE_LAP:
            l_l = lap_loss(pred, hr, mask) * self.w_lap
            total += l_l; log_terms.append(("lap", l_l))
        if USE_PHY_CURL:
            l_c = phy_curl_loss(pred, hr, mask) * self.w_curl
            total += l_c; log_terms.append(("curl", l_c))

        if USE_TS_GRAD:
            l_tg = ts_grad_loss(pred, hr, mask) * Tensor(W_TS_GRAD, mstype.float32)
            total += l_tg; log_terms.append(("ts_grad", l_tg))
        if USE_TS_LAP:
            l_tl = ts_lap_loss(pred, hr, mask) * Tensor(W_TS_LAP, mstype.float32)
            total += l_tl; log_terms.append(("ts_lap", l_tl))

        # append diagnostics to last_terms for external inspection
        self.last_terms = tuple(log_terms) + tuple(diag_terms)
        self.last_pred = pred
        # Guard scalar loss (in case NaN still occurs)
        # If total is NaN or Inf, replace with zero (avoid optimizer crash)
        total = ops.masked_fill(total, ops.isnan(total), Tensor(0.0, mstype.float32))
        total = ops.clip_by_value(total, Tensor(-1e9, mstype.float32), Tensor(1e9, mstype.float32))
        return total



loss_net=WithLossCell(
    net, init_w={"grad": W_GRAD, "lap": W_LAP, "div": W_DIV, "curl": W_CURL}
)
# —— 非零“初始权重种子”，防止早期写回 0 锁死 —— 
INIT_W = {"grad": W_GRAD, "lap": W_LAP, "div": W_DIV, "curl": W_CURL}

# 学习率调度/优化器
steps_per_epoch = max(1, steps_per_epoch)
total_steps  = epochs * steps_per_epoch
warmup_steps = int(0.05*total_steps)
def lr_schedule(step):
    if step < warmup_steps:
        return base_lr * (step+1) / max(1,warmup_steps)
    t=(step-warmup_steps)/max(1,total_steps-warmup_steps)
    return 0.5*base_lr*(1+math.cos(math.pi*t))
lr_tensor=Tensor(np.array([lr_schedule(s) for s in range(total_steps)], dtype=np.float32))
optimizer = nn.Adam(net.trainable_params(), learning_rate=lr_tensor)

class TrainOneStepWithClip(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, max_norm=0.5):
        super().__init__(network, optimizer)
        self.max_norm=max_norm
        self.grad=ops.GradOperation(get_by_list=True, sens_param=False)
    def construct(self, *inputs):
        loss=self.network(*inputs)
        # if loss is NaN we still attempt to compute grads but will sanitize them
        grads=self.grad(self.network, self.weights)(*inputs)
        # sanitize each grad: NaN -> 0, large values clipped
        new_grads = []
        for g in grads:
            # replace nan with 0
            g = ops.masked_fill(g, ops.isnan(g), Tensor(0.0, mstype.float32))
            # clip extreme values to avoid inf propagation
            g = ops.clip_by_value(g, Tensor(-1e6, mstype.float32), Tensor(1e6, mstype.float32))
            new_grads.append(g)
        grads = tuple(new_grads)
        grads=clip_by_global_norm(grads, self.max_norm)
        self.optimizer(grads)
        return loss

train_net=TrainOneStepWithClip(loss_net, optimizer, max_norm=grad_clip_norm)
train_net.set_train(True)

# ==================== 训练 / 验证 ====================
log_csv = os.path.join(OUT_DIR, f"{OUT_PREF}_train_log.csv")
with open(log_csv, "w", newline="") as logf:
    # 增加 mse_uv / mse_ts（用于 uncertainty 模式的对齐参考）
    cols = [
        "epoch","train_loss","val_mse_phys","val_psnr_db","val_curl_rms_ratio",
        "pix","pix_uv","pix_ts","grad","lap","div","curl",
        "w_grad","w_lap","w_div","w_curl"
    ]

    logw = csv.writer(logf); logw.writerow(cols)

    print(f"Start training: epochs={epochs}, steps/epoch={steps_per_epoch}", flush=True)

    last_avg = {
        "pix": None, "pix_uv": None, "pix_ts": None,   # <--- 改这三行
        "grad": None, "lap": None, "div": None, "curl": None
    }


    for ep in range(1, epochs + 1):
        # —— 可选阶段切换：blur5 → blur3 ——
        stage_split = int(os.environ.get("STAGE_SPLIT", "0"))
        if stage_split > 0 and ep == stage_split + 1:
            blur_flag.set_data(Tensor(0.0, mstype.float32))   # 切到 blur3
            print(f"[stage] switched blur from 5 to 3 at epoch {ep}", flush=True)

        prog = ep / float(epochs)
        if   prog <= CURRIC_START: ramp = 0.0
        elif prog >= CURRIC_END:   ramp = 1.0
        else:
            ramp = (prog - CURRIC_START) / max(1e-6, (CURRIC_END - CURRIC_START))

        # —— 自动权重：与参考 MSE（uncertainty: uv+ts；其他: mse）对齐 —— 
        def _auto(use_flag, auto_flag, last_term, cur_w, init_w):
            if not use_flag:
                return cur_w
            # 冷启动：没历史或当前项为 0，用初始权作为种子
            if (not auto_flag) or (last_term is None) or (last_term <= 0):
                return max(init_w, AUTO_W_MIN)

            # 关键：uncertainty 用 uv+ts 的和作为参考；否则用 mse
            # 选用像素参考（pix 或 pix_uv+pix_ts）
            if USE_MAE and not USE_MSE:
                if MAE_WEIGHT_MODE == "uncertainty":
                    uv = last_avg.get("pix_uv"); ts = last_avg.get("pix_ts")
                    pix_ref = (uv or 0.0) + (ts or 0.0)
                else:
                    pix_ref = last_avg.get("pix")
            else:  # 默认当作 MSE
                if MSE_WEIGHT_MODE == "uncertainty":
                    uv = last_avg.get("pix_uv"); ts = last_avg.get("pix_ts")
                    pix_ref = (uv or 0.0) + (ts or 0.0)
                else:
                    pix_ref = last_avg.get("pix")
            
            if (pix_ref is None) or (pix_ref <= 0):
                return max(init_w, AUTO_W_MIN)
            
            eff = float(pix_ref / (last_term + 1e-12))
            eff = max(min(eff, AUTO_W_MAX), AUTO_W_MIN)
            return eff


        # —— 写回权重（grad 从一开始就有 30% 力量；curl 随 ramp 打开）——
        if USE_GRAD:
            prev = float(loss_net.w_grad.asnumpy())
            base = _auto(USE_GRAD, AUTO_W_GRAD, last_avg["grad"], prev, INIT_W["grad"])
            loss_net.w_grad.set_data(Tensor(base * (0.3 + 0.7 * ramp), mstype.float32))
        if USE_LAP:
            prev = float(loss_net.w_lap.asnumpy())
            base = _auto(USE_LAP, AUTO_W_LAP, last_avg["lap"], prev, INIT_W["lap"])
            loss_net.w_lap.set_data(Tensor(base * (0.2 + 0.8 * ramp), mstype.float32))
        if USE_PHY_DIV:
            prev = float(loss_net.w_div.asnumpy())
            base = _auto(USE_PHY_DIV, AUTO_W_DIV, last_avg["div"], prev, INIT_W["div"])
            loss_net.w_div.set_data(Tensor(base * (0.0 + 0.5 * ramp), mstype.float32))
        if USE_PHY_CURL:
            prev = float(loss_net.w_curl.asnumpy())
            base = _auto(USE_PHY_CURL, AUTO_W_CURL, last_avg["curl"], prev, INIT_W["curl"])
            loss_net.w_curl.set_data(Tensor(base * (0.2 + 0.8 * ramp), mstype.float32))

        # —— 训练 —— 
        tot = 0.0
        terms_sum = {
            "pix": 0.0, "pix_uv": 0.0, "pix_ts": 0.0,    # <--- 改这三行
            "grad": 0.0, "lap": 0.0, "div": 0.0, "curl": 0.0
        }

        for lr_b, hr_b, m_b in train_ds.create_tuple_iterator():
            loss = train_net(lr_b, hr_b, m_b)
            tot += float(loss.asnumpy())

            if hasattr(loss_net, "last_terms"):
                for name, val in loss_net.last_terms:
                    v = float(val.asnumpy())
                    if name == "pix":
                        terms_sum["pix"] += v
                    elif name == "pix_uv":
                        terms_sum["pix_uv"] += v
                    elif name == "pix_ts":
                        terms_sum["pix_ts"] += v
                    elif name == "grad":
                        wg = float(loss_net.w_grad.asnumpy())
                        if wg > 1e-12:
                            terms_sum["grad"] += v / wg
                    elif name == "lap":
                        wl = float(loss_net.w_lap.asnumpy())
                        if wl > 1e-12:
                            terms_sum["lap"] += v / wl
                    elif name == "div":
                        wd = float(loss_net.w_div.asnumpy())
                        if wd > 1e-12:
                            terms_sum["div"] += v / wd
                    elif name == "curl":
                        wc = float(loss_net.w_curl.asnumpy())
                        if wc > 1e-12:
                            terms_sum["curl"] += v / wc

        avg_loss = tot / max(1, steps_per_epoch)

        # —— 验证（物理空间 MSE + curl RMS 比值）——
        net.set_train(False)
        total_mse = 0.0; total_cnt = 0.0
        curl_sq_pred = 0.0; curl_sq_gt = 0.0; curl_cnt = 0.0
        for lr_b, hr_b, m_b in test_ds.create_tuple_iterator():
            sr = net(ops.cast(lr_b, mstype.float32))
            sr_phys = unnormalize_y(sr)
            hr_phys = unnormalize_y(hr_b)
            diff = (sr_phys - hr_phys) ** 2 * m_b
            total_mse += float(diff.sum().asnumpy())
            total_cnt += float(m_b.sum().asnumpy())

            # curl RMS ratio 指标（同一内域）
            up, vp = sr_phys[:, 0:1], sr_phys[:, 1:2]
            ut, vt = hr_phys[:, 0:1], hr_phys[:, 1:2]
            curl_p = _curl_center_blur(up, vp)
            curl_t = _curl_center_blur(ut, vt)
            m_uv_bool = logical_and(m_b[:, 0:1] > 0.5, m_b[:, 1:2] > 0.5)
            mm = inner_mask_hwK(m_uv_bool)
            mfloat = ops.cast(mm, mstype.float32)
            curl_sq_pred += float(((curl_p * curl_p) * mfloat).sum().asnumpy())
            curl_sq_gt   += float(((curl_t * curl_t) * mfloat).sum().asnumpy())
            curl_cnt     += float(mfloat.sum().asnumpy())
        mse_phys = total_mse / (total_cnt + 1e-6)
        psnr = 10 * np.log10(1.0 / (mse_phys + 1e-8))
        curl_rms_ratio = 0.0
        if curl_cnt > 0 and curl_sq_gt > 0:
            curl_rms_ratio = math.sqrt(curl_sq_pred / curl_cnt) / max(1e-12, math.sqrt(curl_sq_gt / curl_cnt))
        net.set_train(True)

        steps = max(1, steps_per_epoch)
        # 有的项可能没启用 -> 若从未出现则保持 None
        last_avg = {
            k: (terms_sum[k] / steps if terms_sum[k] > 0 else None)
            for k in terms_sum.keys()
        }

        def fmt(x):
            return (f"{x:.6e}" if (x is not None) else "0.000000e+00")
        print(
            f"Epoch {ep:02d}/{epochs} — train loss: {avg_loss:.6f} | "
            f"val mse(phys): {mse_phys:.6f}, psnr: {psnr:.2f} dB | "
            f"curl_rms_ratio: {curl_rms_ratio:.3f}",
            flush=True
        )
        if PRINT_LOSS_BREAKDOWN:
            lt = ", ".join([
                f"pix={fmt(last_avg['pix'])}",
                f"pix_uv={fmt(last_avg['pix_uv'])}",
                f"pix_ts={fmt(last_avg['pix_ts'])}",
                f"grad={fmt(last_avg['grad'])}",
                f"lap={fmt(last_avg['lap'])}",
                f"div={fmt(last_avg['div'])}",
                f"curl={fmt(last_avg['curl'])}",
            ])
            wt = (
                f" [w_grad={float(loss_net.w_grad.asnumpy()):.3f}, "
                f"w_lap={float(loss_net.w_lap.asnumpy()):.3f}, "
                f"w_div={float(loss_net.w_div.asnumpy()):.3f}, "
                f"w_curl={float(loss_net.w_curl.asnumpy()):.3f}]"
            )
            print("[loss terms] " + lt + wt, flush=True)

        row = [
            ep, avg_loss, mse_phys, psnr, curl_rms_ratio,
            last_avg["pix"] or 0.0,        # <--- 这三处
            last_avg["pix_uv"] or 0.0,
            last_avg["pix_ts"] or 0.0,
            last_avg["grad"] or 0.0,
            last_avg["lap"]  or 0.0,
            last_avg["div"]  or 0.0,
            last_avg["curl"] or 0.0,
            float(loss_net.w_grad.asnumpy()),
            float(loss_net.w_lap.asnumpy()),
            float(loss_net.w_div.asnumpy()),
            float(loss_net.w_curl.asnumpy()),
        ]

        logw.writerow(row); logf.flush()

        # —— 只在最后一轮把模型存为 best —— 
        if ep == epochs:
            ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
            ckpt_best = os.path.join(ckpt_dir, "fuxinet_geoalign_lossflex_best.ckpt")
            save_checkpoint(net, ckpt_best)
            with open(os.path.join(ckpt_dir, "val_best.json"), "w") as f:
                json.dump({
                    "policy": "last",
                    "val_mse_phys": float(mse_phys),
                    "psnr_db": float(psnr),
                    "curl_rms_ratio": float(curl_rms_ratio),
                    "epoch": ep
                }, f, indent=2)

# ===== 保存验证集预测 =====
print("\nSaving predictions and labels to npy files...")
net.set_train(False)
all_preds, all_labels, all_masks = [], [], []
for lr_b, hr_b, m_b in test_ds.create_tuple_iterator():
    sr = net(ops.cast(lr_b, mstype.float32))
    all_preds.append(sr.asnumpy()); all_labels.append(hr_b.asnumpy()); all_masks.append(m_b.asnumpy())
np.save(os.path.join(OUT_DIR, f"{OUT_PREF}_preds.npy"),  np.concatenate(all_preds, axis=0))
np.save(os.path.join(OUT_DIR, f"{OUT_PREF}_labels.npy"), np.concatenate(all_labels, axis=0))
np.save(os.path.join(OUT_DIR, f"{OUT_PREF}_masks.npy"),  np.concatenate(all_masks, axis=0))
print("Done.")