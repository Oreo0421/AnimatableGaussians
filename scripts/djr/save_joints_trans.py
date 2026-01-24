#!/usr/bin/env python3
"""
Standalone script to save joints from avatar dataset
Usage: python save_joints.py -c configs/avatarrex_zzr/avatar.yaml -o output_joints_dir
"""

import os
import json
import numpy as np
import torch
import importlib
from argparse import ArgumentParser
from tqdm import tqdm

import config
from utils.net_util import to_cuda

def save_joints_from_dataset(config_path, output_dir, apply_transform=True):
    """
    Save joints from dataset with optional transformation
    """
    config.load_global_opt(config_path)

    # Your transformation matrix
    trans_matrix = np.array([
        [ 1.417337, -3.719793,  5.757977, 20.600479],
        [-0.786601, -5.929179, -3.636770,  2.822414],
        [ 6.809730,  0.089329, -1.618520, 25.250027],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ])

    # Initialize dataset
    if 'pose_data' in config.opt['test']:
        from dataset.dataset_pose import PoseDataset
        dataset_module = config.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        training_dataset = MvRgbDataset(**config.opt['train']['data'], training=False)

        dataset = PoseDataset(**config.opt['test']['pose_data'],
                             smpl_shape=training_dataset.smpl_data['betas'][0])
        dataset_name = f"{dataset.dataset_name}_{dataset.seq_name}"
    else:
        dataset_module = config.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        dataset = MvRgbDataset(**config.opt['test']['data'], training=False)
        dataset_name = "training_data"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/original", exist_ok=True)
    if apply_transform:
        os.makedirs(f"{output_dir}/transformed", exist_ok=True)

    print(f"Saving joints from {len(dataset)} frames...")
    print(f"Output directory: {output_dir}")

    # Process all frames
    all_original_joints = []
    all_transformed_joints = []

    for idx in tqdm(range(len(dataset)), desc="Processing frames"):
        try:
            # Get item from dataset
            getitem_func = dataset.getitem_fast if hasattr(dataset, 'getitem_fast') else dataset.getitem
            item = getitem_func(idx, training=False)

            # Extract only joints data
            joints_tensor = item['joints']
            if isinstance(joints_tensor, torch.Tensor):
                joints_3d = joints_tensor.detach().cpu().numpy()  # Always detach first
            else:
                joints_3d = np.array(joints_tensor)

            # Save original joints
            original_data = {
                'frame_idx': item.get('data_idx', idx),
                'joints_3d': joints_3d.tolist(),
                'dataset_name': dataset_name
            }

            # Save individual frame files
            frame_idx = item.get('data_idx', idx)

            # JSON format
            with open(f"{output_dir}/original/joints_{frame_idx:08d}.json", 'w') as f:
                json.dump(original_data, f, indent=2)

            # NumPy format
            np.savez(f"{output_dir}/original/joints_{frame_idx:08d}.npz",
                    joints_3d=joints_3d,
                    frame_idx=frame_idx)

            all_original_joints.append(original_data)

            # Apply transformation if requested
            if apply_transform:
                # Convert to homogeneous coordinates and transform
                joints_homogeneous = np.hstack([joints_3d, np.ones((joints_3d.shape[0], 1))])
                transformed_joints = (trans_matrix @ joints_homogeneous.T).T
                transformed_joints_3d = transformed_joints[:, :3]

                transformed_data = {
                    'frame_idx': frame_idx,
                    'joints_3d': transformed_joints_3d.tolist(),
                    'transformation_matrix': trans_matrix.tolist(),
                    'dataset_name': dataset_name
                }

                # Save transformed individual frame files
                with open(f"{output_dir}/transformed/joints_{frame_idx:08d}.json", 'w') as f:
                    json.dump(transformed_data, f, indent=2)

                np.savez(f"{output_dir}/transformed/joints_{frame_idx:08d}.npz",
                        joints_3d=transformed_joints_3d,
                        transformation_matrix=trans_matrix,
                        frame_idx=frame_idx)

                all_transformed_joints.append(transformed_data)

        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            continue

    # Save consolidated files
    print("Saving consolidated files...")

    if len(all_original_joints) == 0:
        print("❌ No frames were successfully processed!")
        return

    # Save all original joints in one file
    with open(f"{output_dir}/all_original_joints.json", 'w') as f:
        json.dump({
            'dataset_name': dataset_name,
            'total_frames': len(all_original_joints),
            'joints_data': all_original_joints
        }, f, indent=2)

    # Save as numpy array
    all_joints_array = np.array([data['joints_3d'] for data in all_original_joints])
    np.savez(f"{output_dir}/all_original_joints.npz",
             joints_3d=all_joints_array,
             frame_indices=[data['frame_idx'] for data in all_original_joints])

    if apply_transform and len(all_transformed_joints) > 0:
        # Save all transformed joints
        with open(f"{output_dir}/all_transformed_joints.json", 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'total_frames': len(all_transformed_joints),
                'transformation_matrix': trans_matrix.tolist(),
                'joints_data': all_transformed_joints
            }, f, indent=2)

        all_transformed_array = np.array([data['joints_3d'] for data in all_transformed_joints])
        np.savez(f"{output_dir}/all_transformed_joints.npz",
                 joints_3d=all_transformed_array,
                 transformation_matrix=trans_matrix,
                 frame_indices=[data['frame_idx'] for data in all_transformed_joints])

    print(f" Successfully saved {len(all_original_joints)} frames of joint data")
    if len(all_original_joints) > 0:
        print(f" Output directory: {output_dir}")
        print(f" Joint dimensions: {np.array(all_original_joints[0]['joints_3d']).shape}")

    failed_frames = len(dataset) - len(all_original_joints)
    if failed_frames > 0:
        print(f"{failed_frames} frames failed to process")

def main():
    parser = ArgumentParser(description="Save joints from avatar dataset")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Output directory for saved joints')
    parser.add_argument('--no-transform', action='store_true',
                       help='Skip transformation, save only original joints')

    args = parser.parse_args()

    apply_transform = not args.no_transform
    save_joints_from_dataset(args.config_path, args.output_dir, apply_transform)

if __name__ == '__main__':
    main()
