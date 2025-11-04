#!/usr/bin/env python3
"""Utilities for converting SCS ocean NetCDF files into FuXi training tensors.

This script mirrors the preprocessing logic used in the original project
notebooks.  It exposes two sub-commands:

* ``to-npy``: slice a NetCDF file into ``.npy`` tensors that match the FuXi
  training pipeline expectations.
* ``compute-stats``: compute the ``*_mean.npy`` and ``*_std.npy`` files that
  FuXi uses for normalisation.

Example workflow (paths are illustrative):

    # 1) Convert the low-resolution NetCDF into samples that include grid info.
    python preprocess_ocean_data.py to-npy \\
        --nc-path /data/raw/SCS_avg_10km_yr03_yr05_256.nc \\
        --out-dir /data/fuxi/10km/train \\
        --prefix SCS_avg_10km_step \\
        --start 0 \\
        --with-grid

    # 2) Convert every high-resolution NetCDF, splitting into train/valid/test.
    python preprocess_ocean_data.py to-npy \\
        --nc-path /data/raw/SCS_avg_3km_yr03_p1.nc \\
        --out-dir /data/fuxi/3km/train \\
        --prefix SCS_avg_3km_step \\
        --start 0

    python preprocess_ocean_data.py to-npy \\
        --nc-path /data/raw/SCS_avg_3km_yr05_p2.nc \\
        --out-dir /data/fuxi/3km/test \\
        --prefix SCS_avg_3km_step \\
        --start 90

    # 3) Generate mean/std files for every split (train/valid/test).
    python preprocess_ocean_data.py compute-stats \\
        --data-dir /data/fuxi/3km \\
        --prefix SCS_avg_3km_step \\
        --modes train valid test

Place the resulting directory tree under ``root_dir`` in your FuXi YAML config:

    root_dir/
      ├─ 10km/
      │   ├─ train_mean.npy / train_std.npy / …
      │   ├─ train/*.npy
      │   ├─ valid/*.npy
      │   └─ test/*.npy
      └─ 3km/
          ├─ train_mean.npy / train_std.npy / …
          ├─ train/*.npy
          ├─ valid/*.npy
          └─ test/*.npy
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        """Fallback progress bar when tqdm is not installed."""
        return iterable


DEFAULT_LEVEL_VARS: Sequence[str] = ("u", "v", "w", "temp", "salt")
GRID_VARS: Sequence[str] = ("lon_rho", "lat_rho")


@dataclass
class ConvertConfig:
    """Configuration bundle for a single to-npy conversion."""

    nc_path: str
    out_dir: str
    prefix: str
    start: int = 0
    with_grid: bool = False
    include_depth: bool = True
    level_vars: Sequence[str] = DEFAULT_LEVEL_VARS


def preprocess_ocean_to_npy(cfg: ConvertConfig) -> None:
    """Convert a NetCDF file into FuXi-ready ``.npy`` samples.

    Each saved tensor has shape ``(1, H, W, S, C)`` where ``C`` equals
    ``len(level_vars)`` plus optional grid channels (lon/lat) and depth.
    """
    if not os.path.exists(cfg.nc_path):
        raise FileNotFoundError(f"NetCDF file not found: {cfg.nc_path}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    with xr.open_dataset(cfg.nc_path, decode_times=False) as ds:
        level_vars = list(cfg.level_vars)
        arrays: List[np.ndarray] = []

        if cfg.with_grid:
            for var in GRID_VARS:
                if var not in ds:
                    raise KeyError(f"Grid variable '{var}' missing in {cfg.nc_path}")
            lon = ds[GRID_VARS[0]].values
            lat = ds[GRID_VARS[1]].values
            grid = np.stack([lon, lat], axis=-1)  # (eta, xi, 2)
            arrays.append(_expand_grid(grid, depth=len(ds["s_rho"])))

        if cfg.include_depth and "z_rho" in ds:
            level_vars = ["z_rho", *level_vars]

        missing = [var for var in level_vars if var not in ds.variables]
        if missing:
            raise KeyError(
                f"Variables {missing} are not present in NetCDF file {cfg.nc_path}"
            )

        times = ds["ocean_time"].values if "ocean_time" in ds else range(ds.dims["time"])

        for idx in tqdm(range(len(times)), desc=f"Exporting {os.path.basename(cfg.nc_path)}"):
            stacked = [
                ds[var].isel(ocean_time=idx).values if "ocean_time" in ds[var].dims
                else ds[var].isel(time=idx).values
                for var in level_vars
            ]
            arr = np.stack(stacked, axis=-1)  # (S, eta, xi, C) or similar
            arr = _ensure_layout(arr)  # (eta, xi, S, C)
            arr = arr[np.newaxis, ...]  # (1, eta, xi, S, C)

            if arrays:
                combined = np.concatenate([*arrays, arr], axis=-1)
            else:
                combined = arr

            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            step = idx + cfg.start
            filename = f"{cfg.prefix}_step{step:02d}.npy"
            np.save(os.path.join(cfg.out_dir, filename), combined.astype(np.float32))


def compute_ocean_statistics(
    data_dir: str,
    prefix: str,
    mode: str,
    epsilon: float = 1e-6,
) -> None:
    """Compute ``mean`` / ``std`` across (N, H, W) for a given split directory."""
    split_dir = os.path.join(data_dir, mode)
    pattern = os.path.join(split_dir, f"{prefix}*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    tensors = [np.load(path) for path in files]
    data = np.concatenate(tensors, axis=0)  # (N, H, W, S, C)

    mean = np.nanmean(data, axis=(0, 1, 2), keepdims=True)
    std = np.nanstd(data, axis=(0, 1, 2), keepdims=True)
    std = np.where(std < epsilon, epsilon, std)

    out_mean = os.path.join(data_dir, f"{mode}_mean.npy")
    out_std = os.path.join(data_dir, f"{mode}_std.npy")
    np.save(out_mean, mean.astype(np.float32))
    np.save(out_std, std.astype(np.float32))


def _expand_grid(grid: np.ndarray, depth: int) -> np.ndarray:
    """Expand a (H, W, 2) grid to shape (1, H, W, depth, 2)."""
    grid = grid[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 2)
    grid = np.repeat(grid, depth, axis=1)  # (1, depth, H, W, 2)
    grid = grid.transpose(0, 2, 3, 1, 4)
    return grid.astype(np.float32)


def _ensure_layout(array: np.ndarray) -> np.ndarray:
    """Ensure data is shaped as (H, W, S, C) regardless of source layout."""
    if array.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {array.shape}")
    # Heuristic: if first axis equals depth length (<= 128), treat layout as (S, H, W, C)
    s_axis = array.shape[0]
    if s_axis <= 128:
        return array.transpose(1, 2, 0, 3)
    return array


def _parse_modes(values: List[str]) -> List[str]:
    allowed = {"train", "valid", "test"}
    modes = [v.lower() for v in values]
    unknown = [m for m in modes if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}. Allowed: {sorted(allowed)}")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_npy = subparsers.add_parser(
        "to-npy",
        help="Convert a NetCDF file into FuXi-compatible .npy tensors.",
    )
    to_npy.add_argument("--nc-path", required=True, help="Input NetCDF file path.")
    to_npy.add_argument(
        "--out-dir",
        required=True,
        help="Directory where .npy tensors will be saved (created if missing).",
    )
    to_npy.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix (defaults to the NetCDF basename).",
    )
    to_npy.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index offset applied to the *_stepXX.npy filenames.",
    )
    to_npy.add_argument(
        "--with-grid",
        action="store_true",
        help="Append lon/lat grid channels (expected for 10km data).",
    )
    to_npy.add_argument(
        "--skip-depth",
        action="store_true",
        help="Do not include the z_rho depth variable even if present.",
    )
    to_npy.add_argument(
        "--vars",
        nargs="+",
        default=list(DEFAULT_LEVEL_VARS),
        help=f"Variables to extract (default: {', '.join(DEFAULT_LEVEL_VARS)}).",
    )

    stats = subparsers.add_parser(
        "compute-stats",
        help="Compute mean/std files for a directory containing split sub-folders.",
    )
    stats.add_argument(
        "--data-dir",
        required=True,
        help="Directory that contains train/valid/test sub-folders.",
    )
    stats.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix used when saving .npy samples.",
    )
    stats.add_argument(
        "--modes",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process (default: train valid test).",
    )
    stats.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Minimum std deviation to avoid division-by-zero (default: 1e-6).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "to-npy":
        prefix = args.prefix or os.path.splitext(os.path.basename(args.nc_path))[0]
        cfg = ConvertConfig(
            nc_path=args.nc_path,
            out_dir=args.out_dir,
            prefix=prefix,
            start=args.start,
            with_grid=args.with_grid,
            include_depth=not args.skip_depth,
            level_vars=args.vars,
        )
        preprocess_ocean_to_npy(cfg)
    elif args.command == "compute-stats":
        modes = _parse_modes(args.modes)
        for mode in modes:
            compute_ocean_statistics(
                data_dir=args.data_dir,
                prefix=args.prefix,
                mode=mode,
                epsilon=args.epsilon,
            )
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
