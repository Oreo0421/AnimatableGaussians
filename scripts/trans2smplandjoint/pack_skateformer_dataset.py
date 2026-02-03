#!/usr/bin/env python3
"""
Pack joints data into SkateFormer-style dataset_22j.npz

Supports two input formats:
1. Our generated SMPL joints: .../smpl/{camera}/*_joints.npz (contains joints_3d_world, joints_3d_cam)
2. Original format: *_all_transformed_joints.npz (contains joints_3d)

Usage:
    # Use our generated data (world coordinates)
    python pack_skateformer_dataset.py \
        --input_glob "/mnt/data_hdd/fzhi/output/101010/**/smpl/*/*_joints.npz" \
        --out_npz /mnt/data_hdd/fzhi/skateformer_data/101010/dataset_22j.npz \
        --coord_type world \
        --save_label_map

    # Use camera coordinates
    python pack_skateformer_dataset.py \
        --input_glob "/mnt/data_hdd/fzhi/output/101010/**/smpl/*/*_joints.npz" \
        --out_npz /mnt/data_hdd/fzhi/skateformer_data/101010/dataset_22j.npz \
        --coord_type camera \
        --save_label_map
"""

import os
import re
import json
import glob
import argparse
import numpy as np


def parse_action_from_path(path: str):
    """
    Parse action name from path.
    
    Supports:
    - .../101010/brooming/djr/p1/smpl/top/101010_brooming_joints.npz -> brooming
    - .../101010_brooming_djr_p1_all_transformed_joints.npz -> brooming
    """
    # Try to extract from directory structure first
    # Pattern: .../subject/ACTION/scene/position/...
    parts = path.replace("\\", "/").split("/")
    
    # Look for pattern: subject_name/action_name/scene/position
    for i, part in enumerate(parts):
        if re.match(r'^\d{6}$', part):  # Found subject like 101010
            if i + 1 < len(parts):
                action = parts[i + 1]
                # Validate it's not a system folder
                if action not in ['smpl', 'joint', 'image', 'npy', 'transformed']:
                    return action
    
    # Fallback: parse from filename
    name = os.path.basename(path)
    # Remove common suffixes
    name = re.sub(r'_joints\.npz$', '', name)
    name = re.sub(r'_all_transformed_joints\.npz$', '', name)
    name = re.sub(r'_smpl\.npz$', '', name)
    
    parts = name.split("_")
    if len(parts) >= 2:
        # Assume format: subject_action_... 
        return parts[1]
    
    raise ValueError(f"Cannot parse action from path: {path}")


def get_joints_from_npz(npz_path: str, coord_type: str = 'world', require_key: str = None):
    """
    Load joints array from npz file.
    
    Args:
        npz_path: path to npz file
        coord_type: 'world' or 'camera'
        require_key: specific key to use (overrides coord_type)
    
    Returns:
        joints: (T, J, 3) array
    """
    z = np.load(npz_path, allow_pickle=True)
    
    # Priority order for finding joints
    if require_key and require_key in z:
        return z[require_key]
    
    key_priority = []
    if coord_type == 'world':
        key_priority = ['joints_3d_world', 'joints_3d', 'joints_3d_cam']
    else:  # camera
        key_priority = ['joints_3d_cam', 'joints_3d', 'joints_3d_world']
    
    for key in key_priority:
        if key in z:
            return z[key]
    
    raise KeyError(f"No joints key found in {npz_path}. Available: {z.files}")


def pad_or_trunc(j: np.ndarray, T: int):
    """j: (t, J, 3) -> (T, J, 3)"""
    t = j.shape[0]
    if t == T:
        return j
    if t > T:
        # Uniform sampling instead of truncation
        indices = np.linspace(0, t - 1, T).astype(int)
        return j[indices]
    # Pad with last frame
    pad_n = T - t
    last = j[-1:, :, :]
    pad = np.repeat(last, pad_n, axis=0)
    return np.concatenate([j, pad], axis=0)


def normalize_joints(joints: np.ndarray, method: str = 'root_relative'):
    """
    Normalize joints.
    
    Args:
        joints: (T, J, 3)
        method: 
            'root_relative' - subtract root joint (pelvis) position
            'first_frame' - subtract first frame root position
            'none' - no normalization
    
    Returns:
        normalized joints: (T, J, 3)
    """
    if method == 'none':
        return joints
    elif method == 'root_relative':
        # Subtract root (pelvis, joint 0) at each frame
        return joints - joints[:, 0:1, :]
    elif method == 'first_frame':
        # Subtract first frame's root position
        return joints - joints[0:1, 0:1, :]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def one_hot(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def main():
    ap = argparse.ArgumentParser("Pack joints into SkateFormer-style dataset")
    ap.add_argument("--input_glob", required=True,
                    help="Glob for joints npz files, e.g. '/mnt/.../**/smpl/*/*_joints.npz'")
    ap.add_argument("--out_npz", required=True, help="Output dataset npz path")
    ap.add_argument("--T", type=int, default=100, help="Fixed temporal length. Default=100")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio. Default=0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coord_type", choices=['world', 'camera'], default='world',
                    help="Coordinate type to use. Default=world")
    ap.add_argument("--require_key", default=None, 
                    help="Specific key to read (overrides coord_type)")
    ap.add_argument("--normalize", choices=['root_relative', 'first_frame', 'none'], 
                    default='root_relative',
                    help="Normalization method. Default=root_relative")
    ap.add_argument("--save_label_map", action="store_true",
                    help="Save label_map.json")
    ap.add_argument("--exclude_actions", nargs='*', default=[],
                    help="Actions to exclude")
    ap.add_argument("--min_frames", type=int, default=10,
                    help="Minimum frames required. Default=10")
    args = ap.parse_args()

    print("=" * 60)
    print("Pack SkateFormer Dataset")
    print("=" * 60)
    print(f"Input glob: {args.input_glob}")
    print(f"Output: {args.out_npz}")
    print(f"Coord type: {args.coord_type}")
    print(f"Normalize: {args.normalize}")
    print(f"T (frames): {args.T}")
    print()

    # Find all files
    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    print(f"[1] Found {len(files)} files")

    # Build label map
    print("\n[2] Parsing actions...")
    actions = []
    valid_files = []
    
    for f in files:
        try:
            act = parse_action_from_path(f)
            if act in args.exclude_actions:
                continue
            actions.append(act)
            valid_files.append(f)
        except ValueError as e:
            print(f"  [Skip] {e}")

    if not actions:
        raise SystemExit("No valid actions found!")

    uniq_actions = sorted(set(actions))
    label_map = {a: i for i, a in enumerate(uniq_actions)}
    num_classes = len(uniq_actions)

    print(f"  Num classes: {num_classes}")
    print("  Label map:")
    for a, i in label_map.items():
        count = actions.count(a)
        print(f"    {i:2d}: {a} ({count} samples)")

    # Load all data
    print(f"\n[3] Loading joints data...")
    X = []
    Y = []
    sample_info = []
    skipped = 0

    for f, act in zip(valid_files, actions):
        try:
            j = get_joints_from_npz(f, args.coord_type, args.require_key)
        except KeyError as e:
            print(f"  [Skip] {e}")
            skipped += 1
            continue

        # Validate shape
        if j.ndim != 3 or j.shape[2] != 3:
            print(f"  [Skip] Bad shape {j.shape} in {f}")
            skipped += 1
            continue

        if j.shape[0] < args.min_frames:
            print(f"  [Skip] Too few frames ({j.shape[0]}) in {f}")
            skipped += 1
            continue

        num_joints = j.shape[1]
        
        # Normalize
        j = normalize_joints(j.astype(np.float32), args.normalize)
        
        # Pad/truncate to fixed length
        j = pad_or_trunc(j, args.T)

        lab = label_map[act]
        X.append(j)
        Y.append(lab)
        sample_info.append({'file': f, 'action': act, 'label': lab})

    if not X:
        raise SystemExit("No valid samples loaded!")

    # Stack - handle variable joint numbers by padding to max
    max_joints = max(x.shape[1] for x in X)
    print(f"  Max joints: {max_joints}")
    
    X_padded = []
    for x in X:
        if x.shape[1] < max_joints:
            pad = np.zeros((x.shape[0], max_joints - x.shape[1], 3), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        X_padded.append(x)

    X = np.stack(X_padded, axis=0)  # (N, T, J, 3)
    Y = np.array(Y, dtype=np.int64)

    print(f"\n[4] Dataset statistics:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Skipped: {skipped}")
    print(f"  X shape: {X.shape}")
    print(f"  X dtype: {X.dtype}")

    # Shuffle and split
    print(f"\n[5] Splitting train/test (ratio={args.test_ratio})...")
    rng = np.random.default_rng(args.seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n_test = int(round(X.shape[0] * args.test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    x_train = X[train_idx]
    x_test = X[test_idx]

    y_train = one_hot(Y[train_idx], num_classes)
    y_test = one_hot(Y[test_idx], num_classes)

    print(f"  Train: {x_train.shape[0]} samples")
    print(f"  Test: {x_test.shape[0]} samples")

    # Save dataset
    print(f"\n[6] Saving dataset...")
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    
    np.savez_compressed(
        args.out_npz,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        class_names=np.array(uniq_actions, dtype=object),
        num_joints=np.array(max_joints),
        num_frames=np.array(args.T),
        coord_type=np.array(args.coord_type),
        normalize=np.array(args.normalize),
    )
    print(f"  Saved: {args.out_npz}")

    # Save label map
    if args.save_label_map:
        map_path = os.path.splitext(args.out_npz)[0] + "_label_map.json"
        with open(map_path, "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"  Saved: {map_path}")

    # Save sample info for debugging
    info_path = os.path.splitext(args.out_npz)[0] + "_samples.json"
    with open(info_path, "w") as f:
        json.dump({
            'train_indices': train_idx.tolist(),
            'test_indices': test_idx.tolist(),
            'samples': sample_info,
        }, f, indent=2)
    print(f"  Saved: {info_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Classes: {num_classes}")
    print(f"Total samples: {X.shape[0]}")
    print(f"Train: {x_train.shape}")
    print(f"Test: {x_test.shape}")
    print(f"Output: {args.out_npz}")


if __name__ == "__main__":
    main()
