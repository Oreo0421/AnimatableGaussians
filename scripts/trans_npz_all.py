#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_root_motion.py

Transfer pose00's root motion (global_orient + transl) to another SMPL/SMPL-X npz motion.

Modes:
  1) copy: directly replace target's global_orient/transl with source's (repeated / length-aligned)
  2) align_first_frame (recommended): compute alignment using first frame:
       R_align = R_src0 @ inv(R_tgt0)
       t_align = t_src0 - R_align @ t_tgt0
     then apply to every target frame:
       R_new = R_align @ R_tgt
       t_new = R_align @ t_tgt + t_align

Usage examples:
  python transfer_root_motion.py --src 00.npz --tgt 01.npz --out 01_rooted.npz --mode align_first_frame
  python transfer_root_motion.py --src 00.npz --tgt 05.npz --mode copy
"""

import argparse
import numpy as np


# -----------------------------
# Math utils: axis-angle <-> R
# -----------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    """aa: (3,) axis-angle (Rodrigues). Return R: (3,3)."""
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = _skew(k)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """R: (3,3) -> axis-angle (3,). Numerically stable enough for typical SMPL roots."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    # axis from skew-symmetric part
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = w / (2.0 * np.sin(theta) + 1e-12)
    aa = axis * theta
    return aa


# -----------------------------
# IO helpers
# -----------------------------
def to_Nx3(arr, N, name, mode="repeat_first"):
    """
    Ensure (N,3).
    If arr is (3,), repeat to (N,3).
    If arr is (K,3):
      - if K==N: return as-is
      - else:
          mode="repeat_first": take arr[0] and repeat
          mode="tile_or_crop": tile/crop to N
    """
    arr = np.asarray(arr)
    if arr.shape == (3,):
        return np.tile(arr[None, :], (N, 1))
    if arr.ndim == 2 and arr.shape[1] == 3:
        K = arr.shape[0]
        if K == N:
            return arr
        if mode == "repeat_first":
            return np.tile(arr[0:1, :], (N, 1))
        if mode == "tile_or_crop":
            if K > N:
                return arr[:N, :]
            reps = int(np.ceil(N / K))
            tiled = np.tile(arr, (reps, 1))
            return tiled[:N, :]
        raise ValueError(f"Unknown length_mismatch mode: {mode}")
    raise ValueError(f"{name} has unexpected shape {arr.shape}, expected (3,) or (K,3)")


def infer_N(npz_obj, prefer=("body_pose", "global_orient", "transl")):
    """Infer frame count from common keys."""
    for k in prefer:
        if k in npz_obj.files:
            v = npz_obj[k]
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                return v.shape[0]
    raise ValueError("Cannot infer frame count N from target npz (need body_pose/global_orient/transl as (N,*)).")


def load_root(npz_obj, N, length_mismatch="repeat_first"):
    """Return (N,3) global_orient and transl arrays from npz."""
    if "global_orient" not in npz_obj.files:
        raise KeyError("npz missing key: global_orient")
    if "transl" not in npz_obj.files:
        raise KeyError("npz missing key: transl")
    go = to_Nx3(npz_obj["global_orient"], N, "global_orient", mode=length_mismatch)
    tr = to_Nx3(npz_obj["transl"], N, "transl", mode=length_mismatch)
    return go.astype(np.float64), tr.astype(np.float64)


# -----------------------------
# Main operations
# -----------------------------
def mode_copy(src_go, src_tr, tgt_N, src_length_mode="tile_or_crop"):
    """
    Copy source root to target length.
    src_go/src_tr can be (Ns,3) already; we adjust to target N using mode.
    """
    new_go = to_Nx3(src_go, tgt_N, "src_global_orient", mode=src_length_mode)
    new_tr = to_Nx3(src_tr, tgt_N, "src_transl", mode=src_length_mode)
    return new_go, new_tr


def mode_align_first_frame(src_go, src_tr, tgt_go, tgt_tr):
    """
    Align target root to source using first frame.
    src_go/src_tr: (Ns,3) but only first frame used for alignment
    tgt_go/tgt_tr: (Nt,3) -> output (Nt,3)
    """
    R_src0 = axis_angle_to_R(src_go[0])
    R_tgt0 = axis_angle_to_R(tgt_go[0])

    R_align = R_src0 @ np.linalg.inv(R_tgt0)
    t_align = src_tr[0] - (R_align @ tgt_tr[0])

    out_go = np.zeros_like(tgt_go, dtype=np.float64)
    out_tr = np.zeros_like(tgt_tr, dtype=np.float64)

    for i in range(tgt_go.shape[0]):
        R_i = axis_angle_to_R(tgt_go[i])
        R_new = R_align @ R_i
        out_go[i] = R_to_axis_angle(R_new)
        out_tr[i] = (R_align @ tgt_tr[i]) + t_align

    return out_go, out_tr, R_align, t_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source npz (e.g. 00.npz)")
    ap.add_argument("--tgt", required=True, help="target npz (e.g. 01.npz)")
    ap.add_argument("--out", default=None, help="output npz path (default: <tgt>_rooted.npz)")
    ap.add_argument("--mode", choices=["copy", "align_first_frame"], default="align_first_frame",
                    help="how to transfer root motion")
    ap.add_argument("--src_length_mode", choices=["repeat_first", "tile_or_crop"], default="tile_or_crop",
                    help="when mode=copy and src length != tgt length, how to adjust src root")
    ap.add_argument("--tgt_infer_from", default="body_pose,global_orient,transl",
                    help="comma-separated keys preference to infer N from target")
    args = ap.parse_args()

    src = np.load(args.src, allow_pickle=True)
    tgt = np.load(args.tgt, allow_pickle=True)

    prefer = tuple([x.strip() for x in args.tgt_infer_from.split(",") if x.strip()])
    Nt = infer_N(tgt, prefer=prefer)

    # Load roots
    src_go, src_tr = load_root(src, N=max(1, infer_N(src, prefer=("body_pose", "global_orient", "transl"))),
                               length_mismatch="tile_or_crop")
    tgt_go, tgt_tr = load_root(tgt, N=Nt, length_mismatch="repeat_first")

    # Prepare new data from target
    new_data = {k: tgt[k] for k in tgt.files}

    if args.mode == "copy":
        new_go, new_tr = mode_copy(src_go, src_tr, tgt_N=Nt, src_length_mode=args.src_length_mode)
        new_data["global_orient"] = new_go.astype(np.float32)
        new_data["transl"] = new_tr.astype(np.float32)
        debug_msg = f"[copy] src -> tgt length: {src_go.shape[0]} -> {Nt} (src_length_mode={args.src_length_mode})"

    else:
        out_go, out_tr, R_align, t_align = mode_align_first_frame(src_go, src_tr, tgt_go, tgt_tr)
        new_data["global_orient"] = out_go.astype(np.float32)
        new_data["transl"] = out_tr.astype(np.float32)
        debug_msg = (
            "[align_first_frame] computed alignment using frame0\n"
            f"R_align=\n{R_align}\n"
            f"t_align={t_align}"
        )

    out_path = args.out
    if out_path is None:
        if args.tgt.lower().endswith(".npz"):
            out_path = args.tgt[:-4] + "_rooted.npz"
        else:
            out_path = args.tgt + "_rooted.npz"

    np.savez(out_path, **new_data)

    print("Saved:", out_path)
    print("Target N =", Nt)
    print("global_orient shape:", new_data["global_orient"].shape, "| transl shape:", new_data["transl"].shape)
    print(debug_msg)


if __name__ == "__main__":
    main()
