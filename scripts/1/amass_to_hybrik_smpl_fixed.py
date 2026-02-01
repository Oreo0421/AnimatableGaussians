#!/usr/bin/env python3
"""
Convert AMASS motion sequence NPZ to HybrIK-compatible SMPL training data.
FIXED VERSION: Correctly computes trans after global_orient transformation.

The key insight is that SMPL's trans is NOT the pelvis position:
    pelvis = trans + body_offset
    
where body_offset depends on global_orient and body_pose.

When we transform global_orient, body_offset changes. So we must:
1. Compute target pelvis position (by transforming joints)
2. Compute new body_offset with transformed global_orient (but trans=0)
3. trans_new = pelvis_target - body_offset_new

Usage:
    python amass_to_hybrik_smpl_fixed.py \
        --amass_npz /path/to/motion.npz \
        --scene djr_p1 \
        --camera_json /path/to/top.json \
        --output_dir /path/to/output \
        --smpl_model_path /path/to/smpl_files
"""

import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from tqdm import tqdm
import torch


# ============================================================
# Scene Transforms
# ============================================================
def load_scene_transforms(transform_file=None):
    if transform_file is None:
        try:
            from scene_transforms import SCENE_TRANSFORMS
            return SCENE_TRANSFORMS
        except ImportError:
            raise ImportError("Could not import scene_transforms.py")
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("scene_transforms", transform_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SCENE_TRANSFORMS


# ============================================================
# Camera Utils
# ============================================================
def parse_3dgs_camera(json_path, frame_idx=0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    W2C = np.array(data['world_view_transform'][frame_idx], dtype=np.float64).T
    R_cam = W2C[:3, :3]
    T_cam = W2C[:3, 3]
    
    H = data['image_height'][frame_idx]
    W = data['image_width'][frame_idx]
    
    fovx = data.get('fovx', [None])[frame_idx]
    fovy = data.get('fovy', [None])[frame_idx]
    
    if 'intrinsics' in data:
        K = np.array(data['intrinsics'][frame_idx], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    elif fovx is not None and fovy is not None:
        fx = W / (2 * np.tan(fovx / 2))
        fy = H / (2 * np.tan(fovy / 2))
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError("Need 'intrinsics' or 'fovx'/'fovy'")
    
    return {
        'W2C': W2C, 'R_cam': R_cam, 'T_cam': T_cam,
        'K': K, 'H': H, 'W': W,
        'fovx': fovx, 'fovy': fovy,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
    }


# ============================================================
# Transform Utils
# ============================================================
def decompose_transform(T_4x4):
    R = T_4x4[:3, :3]
    t = T_4x4[:3, 3]
    
    scale = np.array([
        np.linalg.norm(R[:, 0]),
        np.linalg.norm(R[:, 1]),
        np.linalg.norm(R[:, 2])
    ])
    
    R_pure = R.copy()
    R_pure[:, 0] /= scale[0]
    R_pure[:, 1] /= scale[1]
    R_pure[:, 2] /= scale[2]
    
    det = np.linalg.det(R_pure)
    if det < 0:
        R_pure[:, 0] *= -1
        scale[0] *= -1
    
    return R_pure, t, scale


def transform_points(points, T_4x4):
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1))])
    return (T_4x4 @ points_h.T).T[:, :3]


def transform_rotation(rot_vec, R):
    """Transform axis-angle rotation by rotation matrix R."""
    R_body = Rotation.from_rotvec(rot_vec).as_matrix()
    R_new = R @ R_body
    return Rotation.from_matrix(R_new).as_rotvec()


def world_to_camera(points, R_cam, T_cam):
    return (R_cam @ points.T).T + T_cam


def camera_to_pixel(points_cam, K):
    points_proj = (K @ points_cam.T).T
    return points_proj[:, :2] / points_proj[:, 2:3]


# ============================================================
# SMPL Utils
# ============================================================
def load_amass_npz(npz_path):
    data = dict(np.load(npz_path, allow_pickle=True))
    
    result = {
        'poses': data['poses'],
        'trans': data['trans'],
        'betas': data['betas'],
    }
    
    if 'gender' in data:
        gender = data['gender']
        if isinstance(gender, np.ndarray):
            gender = str(gender)
        result['gender'] = gender
    else:
        result['gender'] = 'neutral'
    
    return result


def get_body_model(smpl_model_path, gender='neutral', model_type='smplx'):
    import smplx
    
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    if gender.startswith("b'"):
        gender = gender[2:-1]
    if gender not in ['male', 'female', 'neutral']:
        gender = 'neutral'
    
    return smplx.create(
        smpl_model_path,
        model_type=model_type,
        gender=gender,
        batch_size=1
    )


def smpl_forward(body_model, global_orient, body_pose, betas, trans):
    """Run SMPL forward pass, return joints."""
    with torch.no_grad():
        output = body_model(
            global_orient=torch.tensor(global_orient).float().unsqueeze(0),
            body_pose=torch.tensor(body_pose).float().unsqueeze(0),
            betas=torch.tensor(betas[:10]).float().unsqueeze(0),
            transl=torch.tensor(trans).float().unsqueeze(0)
        )
        return output.joints[0, :24].numpy()


def compute_correct_trans(body_model, global_orient_new, body_pose, betas, pelvis_target):
    """
    Compute the correct trans value so that SMPL produces the target pelvis position.
    
    SMPL: pelvis = trans + body_offset
    where body_offset = SMPL(global_orient, body_pose, trans=0).pelvis
    
    So: trans = pelvis_target - body_offset
    """
    # Get body_offset with trans=0
    joints_zero_trans = smpl_forward(body_model, global_orient_new, body_pose, betas, np.zeros(3))
    body_offset = joints_zero_trans[0]  # pelvis position when trans=0
    
    # Compute correct trans
    trans_correct = pelvis_target - body_offset
    
    return trans_correct


# ============================================================
# Main Processing
# ============================================================
def process_amass_to_hybrik(
    amass_npz,
    scene_transform,
    camera_json,
    output_dir,
    smpl_model_path,
    subject_name='unknown',
    pose_name='unknown',
):
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Load AMASS data ==========
    print("\n[1] Loading AMASS data...")
    amass_data = load_amass_npz(amass_npz)
    
    poses = amass_data['poses']
    trans = amass_data['trans']
    betas = amass_data['betas']
    gender = amass_data['gender']
    
    N = poses.shape[0]
    pose_dim = poses.shape[1]
    
    print(f"    Frames: {N}")
    print(f"    Pose dim: {pose_dim}")
    print(f"    Gender: {gender}")
    
    # Determine body pose end index
    if pose_dim >= 156:
        body_pose_end = 66  # SMPL-X
        model_type = 'smplx'
    else:
        body_pose_end = 72  # SMPL
        model_type = 'smpl'
    
    print(f"    Model type: {model_type}")
    
    # ========== 2. Load SMPL model ==========
    print("\n[2] Loading SMPL model...")
    body_model = get_body_model(smpl_model_path, gender, model_type)
    
    # ========== 3. Decompose transforms ==========
    print("\n[3] Analyzing transforms...")
    R_scene, t_scene, scale_scene = decompose_transform(scene_transform)
    print(f"    Scene scale: {scale_scene}")
    
    camera = parse_3dgs_camera(camera_json, frame_idx=0)
    R_cam = camera['R_cam']
    T_cam = camera['T_cam']
    print(f"    Camera: {camera['W']}x{camera['H']}, fx={camera['fx']:.2f}")
    
    # Combined rotation: original → world → camera
    R_total = R_cam @ R_scene
    
    # ========== 4. Process each frame ==========
    print("\n[4] Processing frames...")
    
    # Output arrays
    global_orient_cam_all = []
    body_pose_all = []
    trans_cam_all = []
    
    joints_original_all = []
    joints_world_all = []
    joints_cam_all = []
    joints_2d_all = []
    
    for i in tqdm(range(N), desc="Processing"):
        # Original SMPL params
        global_orient_orig = poses[i, :3]
        body_pose = poses[i, 3:body_pose_end]
        trans_orig = trans[i]
        
        # ----- Step 1: Generate original joints -----
        joints_orig = smpl_forward(body_model, global_orient_orig, body_pose, betas, trans_orig)
        
        # ----- Step 2: Transform joints to world coords -----
        joints_world = transform_points(joints_orig, scene_transform)
        
        # ----- Step 3: Transform joints to camera coords -----
        joints_cam = world_to_camera(joints_world, R_cam, T_cam)
        
        # ----- Step 4: Compute new global_orient -----
        # global_orient transforms from body frame to world frame
        # We need new global_orient that goes: body frame → camera frame
        global_orient_cam = transform_rotation(global_orient_orig, R_total)
        
        # ----- Step 5: Compute correct trans -----
        # Target pelvis in camera coords
        pelvis_target = joints_cam[0]
        
        # Compute trans that produces this pelvis
        trans_cam = compute_correct_trans(
            body_model, global_orient_cam, body_pose, betas, pelvis_target
        )
        
        # ----- Step 6: Verify -----
        joints_verify = smpl_forward(body_model, global_orient_cam, body_pose, betas, trans_cam)
        error = np.linalg.norm(joints_verify - joints_cam, axis=1).mean()
        if error > 0.001 and i == 0:
            print(f"    Warning: Frame {i} verification error = {error:.6f} m")
        
        # ----- Step 7: Project to 2D -----
        joints_2d = camera_to_pixel(joints_cam, camera['K'])
        
        # Store results
        global_orient_cam_all.append(global_orient_cam)
        body_pose_all.append(body_pose)
        trans_cam_all.append(trans_cam)
        
        joints_original_all.append(joints_orig)
        joints_world_all.append(joints_world)
        joints_cam_all.append(joints_cam)
        joints_2d_all.append(joints_2d)
    
    # Convert to arrays
    global_orient_cam_all = np.array(global_orient_cam_all, dtype=np.float32)
    body_pose_all = np.array(body_pose_all, dtype=np.float32)
    trans_cam_all = np.array(trans_cam_all, dtype=np.float32)
    
    joints_original_all = np.array(joints_original_all, dtype=np.float32)
    joints_world_all = np.array(joints_world_all, dtype=np.float32)
    joints_cam_all = np.array(joints_cam_all, dtype=np.float32)
    joints_2d_all = np.array(joints_2d_all, dtype=np.float32)
    
    # Visibility
    H, W = camera['H'], camera['W']
    visibility = np.ones((N, 24), dtype=np.float32)
    for i in range(N):
        for j in range(24):
            x, y = joints_2d_all[i, j]
            if x < 0 or x >= W or y < 0 or y >= H or joints_cam_all[i, j, 2] <= 0:
                visibility[i, j] = 0.0
    
    print(f"    Avg visibility: {visibility.mean():.2%}")
    
    # ========== 5. Final verification ==========
    print("\n[5] Final verification...")
    
    # Regenerate joints from saved params and compare
    errors = []
    for i in range(min(10, N)):
        joints_regen = smpl_forward(
            body_model, 
            global_orient_cam_all[i], 
            body_pose_all[i], 
            betas, 
            trans_cam_all[i]
        )
        error = np.linalg.norm(joints_regen - joints_cam_all[i], axis=1).mean()
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"    Mean regeneration error: {mean_error:.6f} m ({mean_error*1000:.3f} mm)")
    
    if mean_error < 0.001:
        print("    ✅ SMPL parameters are correct!")
    else:
        print("    ❌ Warning: Regeneration error is high!")
    
    # ========== 6. Save results ==========
    print("\n[6] Saving results...")
    
    prefix = f"{subject_name}_{pose_name}"
    
    # SMPL parameters
    smpl_output = {
        'global_orient': global_orient_cam_all,
        'body_pose': body_pose_all,
        'betas': betas[:10].astype(np.float32),
        'trans': trans_cam_all,
        
        # World coordinates (for reference)
        'global_orient_world': np.array([
            transform_rotation(poses[i, :3], R_scene) for i in range(N)
        ], dtype=np.float32),
        'trans_world': transform_points(trans, scene_transform).astype(np.float32),
        
        # Original
        'poses_original': poses.astype(np.float32),
        'trans_original': trans.astype(np.float32),
        
        # Metadata
        'gender': np.array(gender),
        'num_frames': np.array(N),
        'scene_transform': scene_transform.astype(np.float64),
        'scene_scale': scale_scene.astype(np.float32),
    }
    
    smpl_path = os.path.join(output_dir, f"{prefix}_smpl.npz")
    np.savez(smpl_path, **smpl_output)
    print(f"    SMPL: {smpl_path}")
    
    # Joints
    joints_output = {
        'joints_3d_cam': joints_cam_all,
        'joints_3d_world': joints_world_all,
        'joints_3d_original': joints_original_all,
        'joints_2d': joints_2d_all,
        'visibility': visibility,
    }
    
    joints_path = os.path.join(output_dir, f"{prefix}_joints.npz")
    np.savez(joints_path, **joints_output)
    print(f"    Joints: {joints_path}")
    
    # Camera
    camera_output = {
        'K': camera['K'].astype(np.float64),
        'R': R_cam.astype(np.float64),
        'T': T_cam.astype(np.float64),
        'W2C': camera['W2C'].astype(np.float64),
        'image_width': np.array(W),
        'image_height': np.array(H),
        'fx': np.array(camera['fx']),
        'fy': np.array(camera['fy']),
        'cx': np.array(camera['cx']),
        'cy': np.array(camera['cy']),
    }
    
    camera_path = os.path.join(output_dir, f"{prefix}_camera.npz")
    np.savez(camera_path, **camera_output)
    print(f"    Camera: {camera_path}")
    
    # Per-frame annotations (JSON)
    print("\n[7] Saving per-frame annotations...")
    
    annot_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)
    
    annotations = []
    for i in range(N):
        annot = {
            'frame_idx': i,
            'image_path': f"images/{i:08d}.png",
            'smpl': {
                'global_orient': global_orient_cam_all[i].tolist(),
                'body_pose': body_pose_all[i].tolist(),
                'betas': betas[:10].tolist(),
                'trans': trans_cam_all[i].tolist(),
            },
            'joints_3d': joints_cam_all[i].tolist(),
            'joints_2d': joints_2d_all[i].tolist(),
            'visibility': visibility[i].tolist(),
            'camera': {
                'fx': float(camera['fx']),
                'fy': float(camera['fy']),
                'cx': float(camera['cx']),
                'cy': float(camera['cy']),
            },
        }
        annotations.append(annot)
    
    annot_path = os.path.join(annot_dir, f"{prefix}_annotations.json")
    with open(annot_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"    Annotations: {annot_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject: {subject_name}")
    print(f"Pose: {pose_name}")
    print(f"Frames: {N}")
    print(f"Regeneration error: {mean_error*1000:.3f} mm")
    print(f"Output: {output_dir}")


def main():
    parser = ArgumentParser(description="Convert AMASS NPZ to HybrIK training data (FIXED)")
    
    parser.add_argument('--amass_npz', type=str, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--camera_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--smpl_model_path', type=str, required=True)
    
    parser.add_argument('--transform_file', type=str, default=None)
    parser.add_argument('--subject_name', type=str, default='unknown')
    parser.add_argument('--pose_name', type=str, default='unknown')
    
    args = parser.parse_args()
    
    SCENE_TRANSFORMS = load_scene_transforms(args.transform_file)
    
    if args.scene not in SCENE_TRANSFORMS:
        print(f"Error: Scene '{args.scene}' not found!")
        print(f"Available: {list(SCENE_TRANSFORMS.keys())}")
        return
    
    scene_transform = np.array(SCENE_TRANSFORMS[args.scene], dtype=np.float64)
    
    process_amass_to_hybrik(
        amass_npz=args.amass_npz,
        scene_transform=scene_transform,
        camera_json=args.camera_json,
        output_dir=args.output_dir,
        smpl_model_path=args.smpl_model_path,
        subject_name=args.subject_name,
        pose_name=args.pose_name,
    )


if __name__ == '__main__':
    main()
