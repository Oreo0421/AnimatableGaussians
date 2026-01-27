#!/usr/bin/env python3
"""
Process per-frame .npy joint files from a folder and apply a 4x4 transform
loaded from scene_transforms.py.

Keeps the SAME output structure:
  output_dir/
    json/original/00000000.json
    json/transformed/00000000.json
    npy/original/00000000.npy
    npy/transformed/00000000.npy
    npy/{pose}_{scene}_all_original_joints.npz
    npy/{pose}_{scene}_all_transformed_joints.npz

pose_name priority:
  1) --pose_name (explicit)
  2) --motion basename (without .npz)
  3) auto infer from input_dir by going up N parents (default N=4) -> folder name

Usage examples:
  # auto pose_name
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /path/to/out \
    --scene djr_p1

  # explicit pose_name
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /path/to/out \
    --scene djr_p1 \
    --pose_name 06_13_poses_inplace_stride5

  # use motion basename as pose_name
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /path/to/out \
    --scene djr_p1 \
    --motion 06_13_poses.npz
"""

import os
import re
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from scene_transforms import SCENE_TRANSFORMS


def infer_pose_name_from_input_dir(input_dir: str, up: int = 4) -> str:
    """
    Auto infer pose_name from input_dir.
    Go up N parent directories and use that folder name.
    """
    p = os.path.abspath(input_dir)
    for _ in range(max(0, up)):
        p = os.path.dirname(p)
    name = os.path.basename(p)
    return name if name else "unknown_pose"


def get_pose_name(args) -> str:
    # Priority 1: explicit pose_name
    if getattr(args, "pose_name", None):
        return args.pose_name

    # Priority 2: motion basename
    if getattr(args, "motion", None):
        base = os.path.basename(args.motion)
        return os.path.splitext(base)[0]

    # Priority 3: auto infer from input_dir parents
    return infer_pose_name_from_input_dir(args.input_dir, up=args.auto_up)


def process_joints_folder(input_dir, output_dir, scene_name, pose_name, apply_transform=True):
    # Folder structure (unchanged)
    json_orig_dir = os.path.join(output_dir, "json", "original")
    json_trans_dir = os.path.join(output_dir, "json", "transformed")
    npy_orig_dir = os.path.join(output_dir, "npy", "original")
    npy_trans_dir = os.path.join(output_dir, "npy", "transformed")
    npy_bundle_dir = os.path.join(output_dir, "npy")

    os.makedirs(json_orig_dir, exist_ok=True)
    os.makedirs(npy_orig_dir, exist_ok=True)
    os.makedirs(npy_bundle_dir, exist_ok=True)

    if apply_transform:
        os.makedirs(json_trans_dir, exist_ok=True)
        os.makedirs(npy_trans_dir, exist_ok=True)

    # Load transform from registry
    if scene_name not in SCENE_TRANSFORMS:
        raise KeyError(
            f"Scene '{scene_name}' not found in scene_transforms.py. "
            f"Available keys example: {list(SCENE_TRANSFORMS.keys())[:10]}"
        )
    trans_matrix = np.array(SCENE_TRANSFORMS[scene_name], dtype=np.float64)

    # Collect 8-digit .npy files (e.g., 00000000.npy)
    pattern = re.compile(r"^\d{8}\.npy$")
    files = sorted([f for f in os.listdir(input_dir) if pattern.match(f)])

    if not files:
        print(f"No 8-digit .npy files found in {input_dir}")
        return

    all_original = []
    all_transformed = []

    for fname in tqdm(files, desc="Processing joints"):
        frame_idx = int(os.path.splitext(fname)[0])
        src_path = os.path.join(input_dir, fname)

        joints_3d = np.load(src_path)  # expected shape (J, 3)
        joints_3d = np.asarray(joints_3d)

        if joints_3d.ndim != 2 or joints_3d.shape[1] != 3:
            print(f"Skipping {fname}: expected shape (J,3), got {joints_3d.shape}")
            continue

        # ---- Save original ----
        # NPY
        np.save(os.path.join(npy_orig_dir, f"{frame_idx:08d}.npy"), joints_3d.astype(np.float32))
        # JSON (structure unchanged)
        with open(os.path.join(json_orig_dir, f"{frame_idx:08d}.json"), "w") as f:
            json.dump(
                {
                    "frame_idx": frame_idx,
                    "joints_3d": joints_3d.tolist(),
                },
                f,
                indent=2
            )

        all_original.append(joints_3d)

        # ---- Save transformed (optional) ----
        if apply_transform:
            joints_h = np.hstack(
                [
                    joints_3d,
                    np.ones((joints_3d.shape[0], 1), dtype=joints_3d.dtype),
                ]
            )  # (J,4)

            transformed = (trans_matrix @ joints_h.T).T[:, :3]

            # NPY
            np.save(os.path.join(npy_trans_dir, f"{frame_idx:08d}.npy"), transformed.astype(np.float32))
            # JSON (structure unchanged)
            with open(os.path.join(json_trans_dir, f"{frame_idx:08d}.json"), "w") as f:
                json.dump(
                    {
                        "frame_idx": frame_idx,
                        "joints_3d": transformed.tolist(),
                        "transformation_matrix": trans_matrix.tolist(),
                    },
                    f,
                    indent=2
                )

            all_transformed.append(transformed)

    # ---- Combined arrays (NPZ) ----
    prefix = f"{pose_name}_{scene_name}"

    if len(all_original) > 0:
        np.savez(
            os.path.join(npy_bundle_dir, f"{prefix}_all_original_joints.npz"),
            joints_3d=np.array(all_original, dtype=np.float32),
        )

    if apply_transform and len(all_transformed) > 0:
        np.savez(
            os.path.join(npy_bundle_dir, f"{prefix}_all_transformed_joints.npz"),
            joints_3d=np.array(all_transformed, dtype=np.float32),
            transformation_matrix=trans_matrix,
        )

    print(f"Processed {len(files)} frames (saved {len(all_original)} valid).")
    print("Outputs:")
    print(f"  JSON: {os.path.join(output_dir, 'json')}")
    print(f"  NPY : {os.path.join(output_dir, 'npy')}")
    print(f"  NPZ : {os.path.join(output_dir, 'npy', f'{prefix}_all_*.npz')}")
    if apply_transform:
        print(f"  using scene transform: {scene_name}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Transform per-frame .npy joints in a folder (keeps original json structure)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing 8-digit joint .npy files (00000000.npy...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save processed joints",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene key in scene_transforms.py (e.g., djr_p1)",
    )

    # Optional: pose name prefix
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Pose/motion npz path (prefix uses its basename)",
    )
    group.add_argument(
        "--pose_name",
        type=str,
        default=None,
        help="Pose name string to use as prefix",
    )

    # Only used when neither --motion nor --pose_name is provided
    parser.add_argument(
        "--auto_up",
        type=int,
        default=4,
        help="Auto-infer pose_name from input_dir: go up N parents and use folder name (default: 4)",
    )

    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip transformation (only save original)",
    )

    args = parser.parse_args()

    pose_name = get_pose_name(args)
    process_joints_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_name=args.scene,
        pose_name=pose_name,
        apply_transform=not args.no_transform,
    )
