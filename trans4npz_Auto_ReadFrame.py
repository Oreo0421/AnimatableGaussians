#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_align_root.py

Auto-infer target frame count N from target motion (npz or folder),
then align/transfer root motion from src 00.npz to target.

Target formats supported:
  A) Folder: contains poses.npy (N,D>=3) + trans.npy (N,3)
  B) NPZ: contains poses (N,D>=3) + (trans or transl)

Source format supported (00.npz):
  - global_orient (N,3 or 3,) + (transl/trans/translation)
  - OR poses (Ns,D>=3) + (transl/trans/translation)

Modes:
  - align_first_frame (recommended): compute alignment using frame0 and apply to all target frames
  - copy: replace target root with source root (auto length-adjust to target N)

Usage:
  python auto_align_root.py --src 00.npz --tgt_dir /path/to/target_dir --out_dir /path/to/out --mode align_first_frame
  python auto_align_root.py --src 00.npz --tgt_npz target.npz --out_npz target_rooted.npz --mode copy
"""

import argparse
import shutil
from pathlib import Path
import numpy as np


# ---------- math ----------
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = _skew(k)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = w / (2.0 * np.sin(theta) + 1e-12)
    return axis * theta


# ---------- helpers ----------
def first_key(data_keys, candidates):
    for k in candidates:
        if k in data_keys:
            return k
    return None


def load_npz(path: str):
    return np.load(path, allow_pickle=True)


def normalize_root_arr(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (3,):
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} bad shape {arr.shape}, expected (N,3) or (3,)")
    return arr.astype(np.float64)


def adjust_len(arr: np.ndarray, N: int, mode: str) -> np.ndarray:
    """arr: (K,3) -> (N,3)"""
    K = arr.shape[0]
    if K == N:
        return arr
    if mode == "repeat_first":
        return np.tile(arr[0:1], (N, 1))
    if mode == "tile_or_crop":
        if K > N:
            return arr[:N]
        reps = int(np.ceil(N / K))
        tiled = np.tile(arr, (reps, 1))
        return tiled[:N]
    raise ValueError(f"Unknown src_len_mode: {mode}")


# ---------- load source root (00.npz) ----------
def load_src_root(src_npz_path: str):
    src = load_npz(src_npz_path)
    # orientation
    k_go = first_key(src.files, ["global_orient", "root_orient"])
    if k_go is not None:
        go = normalize_root_arr(src[k_go], f"src:{k_go}")
    else:
        if "poses" not in src.files:
            raise KeyError("Source missing global_orient/root_orient and poses.")
        poses = np.asarray(src["poses"])
        if poses.ndim != 2 or poses.shape[1] < 3:
            raise ValueError(f"src:poses bad shape {poses.shape}, expected (N,D>=3)")
        go = normalize_root_arr(poses[:, :3], "src:poses[:,:3]")
        k_go = "poses[:,:3]"

    # translation
    k_tr = first_key(src.files, ["transl", "trans", "translation"])
    if k_tr is None:
        raise KeyError("Source missing transl/trans/translation.")
    tr = normalize_root_arr(src[k_tr], f"src:{k_tr}")

    return go, tr, k_go, k_tr


# ---------- load target (auto infer N) ----------
def load_tgt_dir(tgt_dir: str):
    tgt_dir = Path(tgt_dir)
    poses_path = tgt_dir / "poses.npy"
    trans_path = tgt_dir / "trans.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Target dir missing {poses_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Target dir missing {trans_path}")

    poses = np.load(str(poses_path))
    trans = np.load(str(trans_path))

    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"tgt poses.npy bad shape {poses.shape}, expected (N,D>=3)")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"tgt trans.npy bad shape {trans.shape}, expected (N,3)")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"tgt length mismatch: poses {poses.shape[0]} vs trans {trans.shape[0]}")

    # load all npy for copying
    arrays = {p.name: np.load(str(p)) for p in tgt_dir.glob("*.npy")}
    N = poses.shape[0]
