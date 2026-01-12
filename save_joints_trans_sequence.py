#!/usr/bin/env python3
"""
Process .npy joint files from a folder and apply a 4x4 transform.

Usage:
    python save_joints_trans.py \
        --input_dir /path/to/joints \
        --output_dir /path/to/save \
        [--no-transform]
"""

import os
import re
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def process_joints_folder(input_dir, output_dir, apply_transform=True):
    # Folder structure
    json_orig_dir = os.path.join(output_dir, "json", "original")
    json_trans_dir = os.path.join(output_dir, "json", "transformed")
    npy_orig_dir  = os.path.join(output_dir, "npy",  "original")
    npy_trans_dir = os.path.join(output_dir, "npy",  "transformed")

    os.makedirs(json_orig_dir, exist_ok=True)
    os.makedirs(npy_orig_dir,  exist_ok=True)
    if apply_transform:
        os.makedirs(json_trans_dir, exist_ok=True)
        os.makedirs(npy_trans_dir,  exist_ok=True)
 # First transform (the one you used earlier on the human position)
    transform = np.array([
        [1.417337, -3.719793,  5.757977, 20.600479],
        [-0.786601, -5.929179, -3.636770,  2.822414],
        [6.809730,  0.089329, -1.618520, 25.250027],
        [0.000000,  0.000000,  0.000000,  1.000000]
    ], dtype=np.float64)

    # 4x4 transformation matrix (edit as needed)
    trans_matrix_scene = np.array([
        [0.004506839905, -0.124592848122, 0.083404511213, -3.700955867767],
        [0.149711236358, 0.008269036189, 0.004262818955, -2.735711812973],
        [-0.008138610050, 0.083115860820, 0.124601446092, -4.244910240173],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ], dtype=np.float64)
    trans_matrix =  trans_matrix_scene @ transform



    # Collect 8-digit .npy files (e.g., 00000000.npy)
    pattern = re.compile(r"^\d{8}\.npy$")
    files = sorted([f for f in os.listdir(input_dir) if pattern.match(f)])

    if not files:
        print(f" No 8-digit .npy files found in {input_dir}")
        return

    all_original = []
    all_transformed = []

    for fname in tqdm(files, desc="Processing joints"):
        frame_idx = int(os.path.splitext(fname)[0])
        src_path = os.path.join(input_dir, fname)

        joints_3d = np.load(src_path)  # expected shape (J, 3)
        if joints_3d.ndim != 2 or joints_3d.shape[1] != 3:
            print(f"Skipping {fname}: expected shape (J,3), got {joints_3d.shape}")
            continue

        # ---- Save original ----
        # NPY
        np.save(os.path.join(npy_orig_dir, f"{frame_idx:08d}.npy"), joints_3d)
        # JSON
        with open(os.path.join(json_orig_dir, f"{frame_idx:08d}.json"), "w") as f:
            json.dump({
                "frame_idx": frame_idx,
                "joints_3d": joints_3d.tolist()
            }, f, indent=2)

        all_original.append(joints_3d)

        # ---- Save transformed (optional) ----
        if apply_transform:
            joints_h = np.hstack([joints_3d, np.ones((joints_3d.shape[0], 1), dtype=joints_3d.dtype)])
            transformed = (trans_matrix @ joints_h.T).T[:, :3]

            # NPY
            np.save(os.path.join(npy_trans_dir, f"{frame_idx:08d}.npy"), transformed)
            # JSON
            with open(os.path.join(json_trans_dir, f"{frame_idx:08d}.json"), "w") as f:
                json.dump({
                    "frame_idx": frame_idx,
                    "joints_3d": transformed.tolist(),
                    "transformation_matrix": trans_matrix.tolist()
                }, f, indent=2)

            all_transformed.append(transformed)

    # ---- Combined arrays (NPZ) ----
    # Put combined files under npy/ since they’re array bundles
    if len(all_original) > 0:
        np.savez(
            os.path.join(output_dir, "npy", "all_original_joints.npz"),
            joints_3d=np.array(all_original, dtype=np.float32)
        )

    if apply_transform and len(all_transformed) > 0:
        np.savez(
            os.path.join(output_dir, "npy", "all_transformed_joints.npz"),
            joints_3d=np.array(all_transformed, dtype=np.float32),
            transformation_matrix=trans_matrix
        )

    print(f" Processed {len(files)} frames.")
    print(f"Outputs:")
    print(f"  JSON: {os.path.join(output_dir, 'json')}")
    print(f"  NPY : {os.path.join(output_dir, 'npy')}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Transform .npy joint files in a folder")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing joint .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save processed joints")
    parser.add_argument("--no-transform", action="store_true", help="Skip transformation")
    args = parser.parse_args()

    process_joints_folder(args.input_dir, args.output_dir, apply_transform=not args.no_transform)
