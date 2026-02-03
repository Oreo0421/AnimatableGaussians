#!/usr/bin/env python3
"""
Consolidate all SMPL data and convert to HybrIK training format.

HybrIK annotation format (per sample):
{
    'img_path': str,              # relative path to image
    'img_shape': [H, W],          # image size
    'bbox': [x, y, w, h],         # bounding box (can be computed from joints)
    'joint_img': (J, 3),          # 2D joints [x, y, conf] in image coords
    'joint_cam': (J, 3),          # 3D joints in camera coords (meters)
    'joint_relative': (J, 3),     # 3D joints relative to root (pelvis)
    'smpl_pose': (72,),           # SMPL pose (global_orient + body_pose)
    'smpl_shape': (10,),          # SMPL betas
    'twist_phi': (23, 2),         # twist angle (sin, cos) - optional
    'twist_weight': (23,),        # twist weight - optional
    'root_cam': (3,),             # root position in camera coords
    'f': [fx, fy],                # focal length
    'c': [cx, cy],                # principal point
}

Usage:
    python consolidate_hybrik_data.py \
        --input_base /mnt/data_hdd/fzhi/output \
        --subject 101010 \
        --output_dir /mnt/data_hdd/fzhi/hybrik_data \
        --image_base /mnt/data_hdd/fzhi/output

    # With train/val split
    python consolidate_hybrik_data.py \
        --input_base /mnt/data_hdd/fzhi/output \
        --subject 101010 \
        --output_dir /mnt/data_hdd/fzhi/hybrik_data \
        --val_ratio 0.1
"""

import os
import json
import glob
import pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict


def compute_bbox_from_joints(joints_2d, img_shape, padding=1.2):
    """
    Compute bounding box from 2D joints.
    
    Args:
        joints_2d: (J, 2) 2D joint coordinates
        img_shape: [H, W] image size
        padding: bbox padding ratio
    
    Returns:
        bbox: [x, y, w, h] in XYWH format
    """
    H, W = img_shape
    
    valid_joints = joints_2d[~np.isnan(joints_2d).any(axis=1)]
    if len(valid_joints) == 0:
        return [0, 0, W, H]
    
    x_min, y_min = valid_joints.min(axis=0)
    x_max, y_max = valid_joints.max(axis=0)
    
    # Add padding
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * padding
    
    x = max(0, cx - size / 2)
    y = max(0, cy - size / 2)
    w = min(W - x, size)
    h = min(H - y, size)
    
    return [float(x), float(y), float(w), float(h)]


def smplx_to_smpl_pose(global_orient, body_pose):
    """
    Convert SMPL-X pose to SMPL 72-dim pose.
    
    SMPL pose: 72 dims = global_orient(3) + body_pose(69) = 24 joints * 3
    SMPL-X body_pose: 63 dims = 21 joints * 3
    
    We need to pad to 69 dims (23 joints * 3)
    """
    # global_orient: (3,)
    # body_pose: (63,) for SMPL-X or (69,) for SMPL
    
    if len(body_pose) == 63:
        # SMPL-X: pad with zeros for hand joints
        # SMPL has 23 body joints, SMPL-X has 21
        # Padding 2 joints (6 dims) at the end
        body_pose_padded = np.concatenate([body_pose, np.zeros(6)])
    elif len(body_pose) == 69:
        body_pose_padded = body_pose
    else:
        # Just use what we have and pad/truncate
        body_pose_padded = np.zeros(69)
        body_pose_padded[:min(len(body_pose), 69)] = body_pose[:69]
    
    smpl_pose = np.concatenate([global_orient, body_pose_padded])
    return smpl_pose  # (72,)


def process_single_sequence(smpl_dir, image_dir, image_base_for_path):
    """
    Process a single sequence (one camera view).
    
    Args:
        smpl_dir: directory containing SMPL npz files
        image_dir: directory containing rendered images
        image_base_for_path: base path to make relative image paths
    
    Returns:
        list of annotation dicts
    """
    annotations = []
    
    # Find SMPL files
    smpl_files = glob.glob(os.path.join(smpl_dir, '*_smpl.npz'))
    if not smpl_files:
        return []
    
    smpl_file = smpl_files[0]
    joints_file = smpl_file.replace('_smpl.npz', '_joints.npz')
    camera_file = smpl_file.replace('_smpl.npz', '_camera.npz')
    
    if not os.path.exists(joints_file) or not os.path.exists(camera_file):
        print(f"Missing joints or camera file for {smpl_file}")
        return []
    
    # Load data
    smpl_data = np.load(smpl_file, allow_pickle=True)
    joints_data = np.load(joints_file, allow_pickle=True)
    camera_data = np.load(camera_file, allow_pickle=True)
    
    global_orient = smpl_data['global_orient']  # (N, 3)
    body_pose = smpl_data['body_pose']          # (N, 63)
    betas = smpl_data['betas']                  # (10,)
    trans = smpl_data['trans']                  # (N, 3)
    
    joints_3d_cam = joints_data['joints_3d_cam']  # (N, J, 3)
    joints_2d = joints_data['joints_2d']          # (N, J, 2)
    visibility = joints_data.get('visibility', np.ones((len(joints_2d), joints_2d.shape[1])))
    
    fx = float(camera_data['fx'])
    fy = float(camera_data['fy'])
    cx = float(camera_data['cx'])
    cy = float(camera_data['cy'])
    img_w = int(camera_data['image_width'])
    img_h = int(camera_data['image_height'])
    
    N = len(global_orient)
    
    # Find image files
    image_patterns = [
        os.path.join(image_dir, '{:08d}.png'),
        os.path.join(image_dir, '{:08d}.jpg'),
        os.path.join(image_dir, 'frame_{:08d}.png'),
        os.path.join(image_dir, 'frame_{:08d}.jpg'),
    ]
    
    working_pattern = None
    for pattern in image_patterns:
        if os.path.exists(pattern.format(0)):
            working_pattern = pattern
            break
    
    if working_pattern is None:
        print(f"No images found in {image_dir}")
        return []
    
    for i in range(N):
        img_path_full = working_pattern.format(i)
        if not os.path.exists(img_path_full):
            continue
        
        # Make relative path
        img_path_rel = os.path.relpath(img_path_full, image_base_for_path)
        
        # Convert pose to SMPL 72-dim format
        smpl_pose = smplx_to_smpl_pose(global_orient[i], body_pose[i])
        
        # Root (pelvis) position
        root_cam = joints_3d_cam[i, 0]  # pelvis is joint 0
        
        # Joint positions relative to root
        joint_relative = joints_3d_cam[i] - root_cam
        
        # 2D joints with confidence
        joint_img = np.zeros((joints_2d.shape[1], 3))
        joint_img[:, :2] = joints_2d[i]
        joint_img[:, 2] = visibility[i]
        
        # Compute bbox
        bbox = compute_bbox_from_joints(joints_2d[i], [img_h, img_w])
        
        annot = {
            'img_path': img_path_rel,
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
        }
        
        annotations.append(annot)
    
    return annotations


def find_all_smpl_dirs(input_base, subject):
    """
    Find all SMPL directories for a subject.
    
    Expected structure:
    input_base/{subject}/{pose}/{scene}/{position}/smpl/{camera}/
    """
    subject_dir = os.path.join(input_base, subject)
    smpl_dirs = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(subject_dir):
        if 'smpl' in dirs:
            smpl_base = os.path.join(root, 'smpl')
            for camera in os.listdir(smpl_base):
                camera_dir = os.path.join(smpl_base, camera)
                if os.path.isdir(camera_dir):
                    # Check if it has SMPL files
                    if glob.glob(os.path.join(camera_dir, '*_smpl.npz')):
                        # Find corresponding image directory
                        image_dir = os.path.join(root, 'image', camera)
                        if os.path.exists(image_dir):
                            smpl_dirs.append({
                                'smpl_dir': camera_dir,
                                'image_dir': image_dir,
                                'root': root,
                                'camera': camera,
                            })
    
    return smpl_dirs


def consolidate_data(
    input_base,
    subject,
    output_dir,
    image_base_for_path=None,
    val_ratio=0.1,
):
    """
    Consolidate all SMPL data for a subject into HybrIK format.
    """
    # Add subject subdirectory to output
    output_dir = os.path.join(output_dir, subject)
    os.makedirs(output_dir, exist_ok=True)
    
    if image_base_for_path is None:
        image_base_for_path = input_base
    
    print(f"\n{'='*60}")
    print("Consolidating HybrIK Training Data")
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
    
    for item in tqdm(smpl_dirs, desc="Processing"):
        annots = process_single_sequence(
            smpl_dir=item['smpl_dir'],
            image_dir=item['image_dir'],
            image_base_for_path=image_base_for_path,
        )
        all_annotations.extend(annots)
    
    print(f"    Total samples: {len(all_annotations)}")
    
    if len(all_annotations) == 0:
        print("No annotations generated!")
        return
    
    # Shuffle and split
    print("\n[3] Splitting train/val...")
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
    print("\n[4] Saving JSON files...")
    
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
    
    # Save as PKL (HybrIK often uses pickle)
    print("\n[5] Saving PKL files...")
    
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
    print("\n[6] Saving statistics...")
    
    stats = {
        'subject': subject,
        'total_samples': len(all_annotations),
        'train_samples': len(train_annots),
        'val_samples': len(val_annots),
        'num_sequences': len(smpl_dirs),
        'image_base': image_base_for_path,
        'sequences': [
            {
                'smpl_dir': item['smpl_dir'],
                'image_dir': item['image_dir'],
                'camera': item['camera'],
            }
            for item in smpl_dirs
        ]
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
    print(f"\nOutput files:")
    print(f"  - {train_json_path}")
    print(f"  - {val_json_path}")
    print(f"  - {train_pkl_path}")
    print(f"  - {val_pkl_path}")
    print(f"  - {stats_path}")
    
    # Print sample annotation
    print("\n[Sample Annotation]")
    sample = all_annotations[0]
    print(f"  img_path: {sample['img_path']}")
    print(f"  img_shape: {sample['img_shape']}")
    print(f"  bbox: {sample['bbox']}")
    print(f"  joint_img shape: {len(sample['joint_img'])} x 3")
    print(f"  joint_cam shape: {len(sample['joint_cam'])} x 3")
    print(f"  smpl_pose shape: {len(sample['smpl_pose'])}")
    print(f"  smpl_shape shape: {len(sample['smpl_shape'])}")
    print(f"  f: {sample['f']}")
    print(f"  c: {sample['c']}")


def main():
    parser = ArgumentParser(description="Consolidate SMPL data for HybrIK training")
    
    parser.add_argument('--input_base', type=str, required=True,
                        help='Base input directory (e.g., /mnt/data_hdd/fzhi/output)')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject ID (e.g., 101010)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HybrIK data')
    parser.add_argument('--image_base', type=str, default=None,
                        help='Base path for relative image paths (default: same as input_base)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    consolidate_data(
        input_base=args.input_base,
        subject=args.subject,
        output_dir=args.output_dir,
        image_base_for_path=args.image_base,
        val_ratio=args.val_ratio,
    )


if __name__ == '__main__':
    main()
