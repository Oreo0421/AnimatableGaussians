#!/usr/bin/env python3
"""
Consolidate all SMPL data and copy images to a unified directory.

Output structure:
    /mnt/data_hdd/fzhi/hybrik_data/{subject}/
    ├── images/
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   └── ...
    ├── train.json
    ├── val.json
    ├── train.pkl
    ├── val.pkl
    └── stats.json

Usage:
    python consolidate_hybrik_data_with_images.py \
        --input_base /mnt/data_hdd/fzhi/output \
        --subject 101010 \
        --output_dir /mnt/data_hdd/fzhi/hybrik_data \
        --val_ratio 0.1
"""

import os
import json
import glob
import pickle
import shutil
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def compute_bbox_from_joints(joints_2d, img_shape, padding=1.2):
    """Compute bounding box from 2D joints."""
    H, W = img_shape
    
    valid_joints = joints_2d[~np.isnan(joints_2d).any(axis=1)]
    if len(valid_joints) == 0:
        return [0, 0, W, H]
    
    x_min, y_min = valid_joints.min(axis=0)
    x_max, y_max = valid_joints.max(axis=0)
    
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * padding
    
    x = max(0, cx - size / 2)
    y = max(0, cy - size / 2)
    w = min(W - x, size)
    h = min(H - y, size)
    
    return [float(x), float(y), float(w), float(h)]


def smplx_to_smpl_pose(global_orient, body_pose):
    """Convert SMPL-X pose to SMPL 72-dim pose."""
    if len(body_pose) == 63:
        body_pose_padded = np.concatenate([body_pose, np.zeros(6)])
    elif len(body_pose) == 69:
        body_pose_padded = body_pose
    else:
        body_pose_padded = np.zeros(69)
        body_pose_padded[:min(len(body_pose), 69)] = body_pose[:69]
    
    smpl_pose = np.concatenate([global_orient, body_pose_padded])
    return smpl_pose


def find_all_smpl_dirs(input_base, subject):
    """Find all SMPL directories for a subject."""
    subject_dir = os.path.join(input_base, subject)
    smpl_dirs = []
    
    for root, dirs, files in os.walk(subject_dir):
        if 'smpl' in dirs:
            smpl_base = os.path.join(root, 'smpl')
            for camera in os.listdir(smpl_base):
                camera_dir = os.path.join(smpl_base, camera)
                if os.path.isdir(camera_dir):
                    if glob.glob(os.path.join(camera_dir, '*_smpl.npz')):
                        image_dir = os.path.join(root, 'image', camera)
                        if os.path.exists(image_dir):
                            smpl_dirs.append({
                                'smpl_dir': camera_dir,
                                'image_dir': image_dir,
                                'root': root,
                                'camera': camera,
                            })
    
    return smpl_dirs


def process_single_sequence(smpl_dir, image_dir, output_images_dir, start_idx):
    """
    Process a single sequence and copy images.
    
    Returns:
        annotations: list of annotation dicts
        next_idx: next available image index
        copied_images: list of (src, dst) tuples
    """
    annotations = []
    copied_images = []
    
    smpl_files = glob.glob(os.path.join(smpl_dir, '*_smpl.npz'))
    if not smpl_files:
        return [], start_idx, []
    
    smpl_file = smpl_files[0]
    joints_file = smpl_file.replace('_smpl.npz', '_joints.npz')
    camera_file = smpl_file.replace('_smpl.npz', '_camera.npz')
    
    if not os.path.exists(joints_file) or not os.path.exists(camera_file):
        return [], start_idx, []
    
    # Load data
    smpl_data = np.load(smpl_file, allow_pickle=True)
    joints_data = np.load(joints_file, allow_pickle=True)
    camera_data = np.load(camera_file, allow_pickle=True)
    
    global_orient = smpl_data['global_orient']
    body_pose = smpl_data['body_pose']
    betas = smpl_data['betas']
    trans = smpl_data['trans']
    
    joints_3d_cam = joints_data['joints_3d_cam']
    joints_2d = joints_data['joints_2d']
    visibility = joints_data.get('visibility', np.ones((len(joints_2d), joints_2d.shape[1])))
    
    fx = float(camera_data['fx'])
    fy = float(camera_data['fy'])
    cx = float(camera_data['cx'])
    cy = float(camera_data['cy'])
    img_w = int(camera_data['image_width'])
    img_h = int(camera_data['image_height'])
    
    N = len(global_orient)
    
    # Find image pattern
    image_patterns = [
        (os.path.join(image_dir, '{:08d}.png'), '.png'),
        (os.path.join(image_dir, '{:08d}.jpg'), '.jpg'),
        (os.path.join(image_dir, 'frame_{:08d}.png'), '.png'),
        (os.path.join(image_dir, 'frame_{:08d}.jpg'), '.jpg'),
    ]
    
    working_pattern = None
    ext = '.png'
    for pattern, e in image_patterns:
        if os.path.exists(pattern.format(0)):
            working_pattern = pattern
            ext = e
            break
    
    if working_pattern is None:
        return [], start_idx, []
    
    current_idx = start_idx
    
    for i in range(N):
        src_img_path = working_pattern.format(i)
        if not os.path.exists(src_img_path):
            continue
        
        # New image name
        new_img_name = f"{current_idx:08d}{ext}"
        new_img_path = os.path.join(output_images_dir, new_img_name)
        
        # Record copy operation
        copied_images.append((src_img_path, new_img_path))
        
        # Convert pose
        smpl_pose = smplx_to_smpl_pose(global_orient[i], body_pose[i])
        
        # Root position
        root_cam = joints_3d_cam[i, 0]
        
        # Relative joints
        joint_relative = joints_3d_cam[i] - root_cam
        
        # 2D joints with confidence
        joint_img = np.zeros((joints_2d.shape[1], 3))
        joint_img[:, :2] = joints_2d[i]
        joint_img[:, 2] = visibility[i]
        
        # Bbox
        bbox = compute_bbox_from_joints(joints_2d[i], [img_h, img_w])
        
        annot = {
            'img_path': f"images/{new_img_name}",
            'img_shape': [img_h, img_w],
            'bbox': bbox,
            'joint_img': joint_img.tolist(),
            'joint_cam': joints_3d_cam[i].tolist(),
            'joint_relative': joint_relative.tolist(),
            'smpl_pose': smpl_pose.tolist(),
            'smpl_shape': betas[:10].tolist(),
            'root_cam': root_cam.tolist(),
            'f': [fx, fy],
            'c': [cx, cy],
            'frame_idx': i,
            'global_idx': current_idx,
        }
        
        annotations.append(annot)
        current_idx += 1
    
    return annotations, current_idx, copied_images


def consolidate_data_with_images(
    input_base,
    subject,
    output_dir,
    val_ratio=0.1,
    copy_images=True,
):
    """Consolidate all SMPL data and copy images."""
    
    # Output directory with subject
    output_dir = os.path.join(output_dir, subject)
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Consolidating HybrIK Training Data (with Images)")
    print(f"{'='*60}")
    print(f"Input: {input_base}")
    print(f"Subject: {subject}")
    print(f"Output: {output_dir}")
    
    # Find all SMPL directories
    print("\n[1] Finding SMPL directories...")
    smpl_dirs = find_all_smpl_dirs(input_base, subject)
    print(f"    Found {len(smpl_dirs)} sequences")
    
    if len(smpl_dirs) == 0:
        print("No SMPL data found!")
        return
    
    # Process all sequences
    print("\n[2] Processing sequences...")
    all_annotations = []
    all_copies = []
    current_idx = 1  # Start from 1
    
    for item in tqdm(smpl_dirs, desc="Processing"):
        annots, current_idx, copies = process_single_sequence(
            smpl_dir=item['smpl_dir'],
            image_dir=item['image_dir'],
            output_images_dir=images_dir,
            start_idx=current_idx,
        )
        all_annotations.extend(annots)
        all_copies.extend(copies)
    
    print(f"    Total samples: {len(all_annotations)}")
    print(f"    Images to copy: {len(all_copies)}")
    
    if len(all_annotations) == 0:
        print("No annotations generated!")
        return
    
    # Copy images
    if copy_images:
        print("\n[3] Copying images...")
        for src, dst in tqdm(all_copies, desc="Copying"):
            shutil.copy2(src, dst)
        print(f"    Copied {len(all_copies)} images")
    else:
        print("\n[3] Skipping image copy (dry run)")
    
    # Shuffle and split
    print("\n[4] Splitting train/val...")
    np.random.seed(42)
    indices = np.random.permutation(len(all_annotations))
    
    val_size = int(len(all_annotations) * val_ratio)
    train_size = len(all_annotations) - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_annots = [all_annotations[i] for i in train_indices]
    val_annots = [all_annotations[i] for i in val_indices]
    
    print(f"    Train: {len(train_annots)}")
    print(f"    Val: {len(val_annots)}")
    
    # Save as JSON
    print("\n[5] Saving JSON files...")
    
    train_json_path = os.path.join(output_dir, 'train.json')
    val_json_path = os.path.join(output_dir, 'val.json')
    all_json_path = os.path.join(output_dir, 'all.json')
    
    with open(train_json_path, 'w') as f:
        json.dump(train_annots, f)
    print(f"    Saved: {train_json_path}")
    
    with open(val_json_path, 'w') as f:
        json.dump(val_annots, f)
    print(f"    Saved: {val_json_path}")
    
    with open(all_json_path, 'w') as f:
        json.dump(all_annotations, f)
    print(f"    Saved: {all_json_path}")
    
    # Save as PKL
    print("\n[6] Saving PKL files...")
    
    train_pkl_path = os.path.join(output_dir, 'train.pkl')
    val_pkl_path = os.path.join(output_dir, 'val.pkl')
    all_pkl_path = os.path.join(output_dir, 'all.pkl')
    
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_annots, f)
    print(f"    Saved: {train_pkl_path}")
    
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_annots, f)
    print(f"    Saved: {val_pkl_path}")
    
    with open(all_pkl_path, 'wb') as f:
        pickle.dump(all_annotations, f)
    print(f"    Saved: {all_pkl_path}")
    
    # Save statistics
    print("\n[7] Saving statistics...")
    
    stats = {
        'subject': subject,
        'total_samples': len(all_annotations),
        'train_samples': len(train_annots),
        'val_samples': len(val_annots),
        'num_sequences': len(smpl_dirs),
        'num_images': len(all_copies),
    }
    
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"    Saved: {stats_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject: {subject}")
    print(f"Total samples: {len(all_annotations)}")
    print(f"Train samples: {len(train_annots)}")
    print(f"Val samples: {len(val_annots)}")
    print(f"Images copied: {len(all_copies)}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── 00000001.png")
    print(f"  │   ├── ...")
    print(f"  │   └── {current_idx-1:08d}.png")
    print(f"  ├── train.json")
    print(f"  ├── val.json")
    print(f"  ├── train.pkl")
    print(f"  ├── val.pkl")
    print(f"  └── stats.json")
    
    # Sample annotation
    print("\n[Sample Annotation]")
    sample = all_annotations[0]
    print(f"  img_path: {sample['img_path']}")
    print(f"  img_shape: {sample['img_shape']}")
    print(f"  bbox: {[f'{x:.1f}' for x in sample['bbox']]}")
    print(f"  smpl_pose length: {len(sample['smpl_pose'])}")
    print(f"  smpl_shape length: {len(sample['smpl_shape'])}")


def main():
    parser = ArgumentParser(description="Consolidate SMPL data and copy images")
    
    parser.add_argument('--input_base', type=str, required=True,
                        help='Base input directory')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject ID (e.g., 101010)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--no_copy', action='store_true',
                        help='Do not copy images (dry run)')
    
    args = parser.parse_args()
    
    consolidate_data_with_images(
        input_base=args.input_base,
        subject=args.subject,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        copy_images=not args.no_copy,
    )


if __name__ == '__main__':
    main()
