#!/usr/bin/env python3
"""
Generate HybrIK SMPL training data using pre-rendered joints from 3DGS.

Since the 3DGS rendering already saved joints that align with the rendered images,
we use those joints directly and compute the correct SMPL trans to match.

Usage:
    python amass_to_hybrik_with_rendered_joints.py \
        --amass_npz /path/to/motion.npz \
        --rendered_joints_dir /path/to/3dgs/output/joints \
        --camera_json /path/to/top.json \
        --output_dir /path/to/output \
        --smpl_model_path /path/to/smpl_files \
        --scene djr_p1
"""

import os
import json
import glob
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


def transform_rotation(rot_vec, R):
    """Transform axis-angle rotation by rotation matrix R."""
    R_body = Rotation.from_rotvec(rot_vec).as_matrix()
    R_new = R @ R_body
    return Rotation.from_matrix(R_new).as_rotvec()


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


def compute_trans_from_pelvis(body_model, global_orient, body_pose, betas, pelvis_target):
    """
    Compute trans so that SMPL produces the target pelvis position.
    
    SMPL: pelvis = trans + body_offset
    So: trans = pelvis_target - body_offset
    """
    joints_zero_trans = smpl_forward(body_model, global_orient, body_pose, betas, np.zeros(3))
    body_offset = joints_zero_trans[0]
    trans = pelvis_target - body_offset
    return trans


def load_rendered_joints(joints_dir, max_frames=None):
    """Load pre-rendered joints from 3DGS output.
    
    Returns:
        joints: (N, J, 3) array
        num_frames: actual number of frames loaded
    """
    # Try different filename patterns
    patterns = [
        os.path.join(joints_dir, "{:08d}.npy"),
        os.path.join(joints_dir, "frame_{:08d}.npy"),
        os.path.join(joints_dir, "{:06d}.npy"),
        os.path.join(joints_dir, "frame_{:06d}.npy"),
    ]
    
    # Find which pattern works
    working_pattern = None
    for pattern in patterns:
        if os.path.exists(pattern.format(0)):
            working_pattern = pattern
            break
    
    if working_pattern is None:
        files = glob.glob(os.path.join(joints_dir, "*.npy"))
        if files:
            print(f"Found .npy files but pattern doesn't match. Examples: {files[:3]}")
        raise FileNotFoundError(f"No joint files found in {joints_dir}")
    
    print(f"Using pattern: {working_pattern}")
    
    # Count available frames
    all_joints = []
    i = 0
    while True:
        if max_frames is not None and i >= max_frames:
            break
        filepath = working_pattern.format(i)
        if os.path.exists(filepath):
            joints = np.load(filepath)
            all_joints.append(joints)
            i += 1
        else:
            break
    
    if len(all_joints) == 0:
        raise FileNotFoundError(f"No joint files found starting from frame 0")
    
    print(f"Loaded {len(all_joints)} frames")
    return np.array(all_joints), len(all_joints)


# ============================================================
# Main Processing
# ============================================================
def process_with_rendered_joints(
    amass_npz,
    rendered_joints_dir,
    camera_json,
    output_dir,
    smpl_model_path,
    scene_transform=None,
    subject_name='unknown',
    pose_name='unknown',
):
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Load rendered joints FIRST to determine frame count ==========
    print("\n[1] Loading rendered joints...")
    joints_rendered, num_rendered_frames = load_rendered_joints(rendered_joints_dir)
    print(f"    Loaded joints shape: {joints_rendered.shape}")
    print(f"    Available frames: {num_rendered_frames}")
    
    # ========== 2. Load AMASS data ==========
    print("\n[2] Loading AMASS data...")
    amass_data = load_amass_npz(amass_npz)
    
    poses = amass_data['poses']
    trans_orig = amass_data['trans']
    betas = amass_data['betas']
    gender = amass_data['gender']
    
    N_amass = poses.shape[0]
    pose_dim = poses.shape[1]
    
    print(f"    AMASS frames: {N_amass}")
    print(f"    Pose dim: {pose_dim}")
    print(f"    Gender: {gender}")
    
    # Use minimum of both
    N = min(N_amass, num_rendered_frames)
    print(f"    Using {N} frames")
    
    # Trim data to N frames
    poses = poses[:N]
    trans_orig = trans_orig[:N]
    joints_rendered = joints_rendered[:N]
    
    # Determine body pose end index
    if pose_dim >= 156:
        body_pose_end = 66
        model_type = 'smplx'
    else:
        body_pose_end = 72
        model_type = 'smpl'
    
    print(f"    Model type: {model_type}")
    
    # ========== 3. Load SMPL model ==========
    print("\n[3] Loading SMPL model...")
    body_model = get_body_model(smpl_model_path, gender, model_type)
    
    # ========== 4. Load camera and scene transform ==========
    print("\n[4] Loading camera parameters...")
    camera = parse_3dgs_camera(camera_json, frame_idx=0)
    R_cam = camera['R_cam']
    T_cam = camera['T_cam']
    print(f"    Camera: {camera['W']}x{camera['H']}, fx={camera['fx']:.2f}")
    print(f"    R_cam:\n{R_cam}")
    print(f"    T_cam: {T_cam}")
    
    if scene_transform is not None:
        R_scene, t_scene, scale_scene = decompose_transform(scene_transform)
        R_total = R_cam @ R_scene
        print(f"    Scene scale: {scale_scene}")
    else:
        R_scene = np.eye(3)
        R_total = R_cam
        scale_scene = np.array([1., 1., 1.])
    
    # ========== 5. Transform joints from world to camera coordinates ==========
    print("\n[5] Transforming joints to camera coordinates...")
    
    # The rendered joints are in WORLD coordinates (after scene transform)
    # We need to transform them to CAMERA coordinates
    joints_world = joints_rendered  # (N, J, 3) in world coords
    
    joints_cam = np.zeros_like(joints_world)
    for i in range(N):
        # World to camera: P_cam = R_cam @ P_world + T_cam
        joints_cam[i] = (R_cam @ joints_world[i].T).T + T_cam
    
    print(f"    Joints world (frame 0, pelvis): {joints_world[0, 0]}")
    print(f"    Joints cam (frame 0, pelvis): {joints_cam[0, 0]}")
    
    # ========== 6. Compute SMPL parameters ==========
    print("\n[6] Computing SMPL parameters...")
    
    global_orient_cam_all = []
    body_pose_all = []
    trans_cam_all = []
    
    for i in tqdm(range(N), desc="Processing"):
        # Original SMPL params
        global_orient_orig = poses[i, :3]
        body_pose = poses[i, 3:body_pose_end]
        
        # Transform global_orient to camera space
        global_orient_cam = transform_rotation(global_orient_orig, R_total)
        
        # Target pelvis from rendered joints (now in camera coords)
        pelvis_target = joints_cam[i, 0]
        
        # Compute trans to match the target pelvis
        trans_cam = compute_trans_from_pelvis(
            body_model, global_orient_cam, body_pose, betas, pelvis_target
        )
        
        global_orient_cam_all.append(global_orient_cam)
        body_pose_all.append(body_pose)
        trans_cam_all.append(trans_cam)
    
    global_orient_cam_all = np.array(global_orient_cam_all, dtype=np.float32)
    body_pose_all = np.array(body_pose_all, dtype=np.float32)
    trans_cam_all = np.array(trans_cam_all, dtype=np.float32)
    
    # ========== 7. Verify ==========
    print("\n[7] Verifying SMPL parameters...")
    
    errors = []
    for i in range(min(10, N)):
        joints_regen = smpl_forward(
            body_model,
            global_orient_cam_all[i],
            body_pose_all[i],
            betas,
            trans_cam_all[i]
        )
        
        # Compare with camera-space joints (first min joints)
        num_joints = min(joints_regen.shape[0], joints_cam[i].shape[0])
        error = np.linalg.norm(joints_regen[:num_joints] - joints_cam[i, :num_joints], axis=1).mean()
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"    Mean error: {mean_error:.6f} m ({mean_error*1000:.3f} mm)")
    
    if mean_error < 0.01:
        print("    ✅ SMPL parameters match rendered joints!")
    else:
        print("    ⚠️  Warning: Some mismatch detected")
    
    # ========== 8. Project to 2D ==========
    print("\n[8] Projecting to 2D...")
    
    joints_2d = np.zeros((N, joints_cam.shape[1], 2), dtype=np.float32)
    for i in range(N):
        joints_2d[i] = camera_to_pixel(joints_cam[i], camera['K'])
    
    # Visibility
    H, W = camera['H'], camera['W']
    visibility = np.ones((N, joints_cam.shape[1]), dtype=np.float32)
    for i in range(N):
        for j in range(joints_cam.shape[1]):
            x, y = joints_2d[i, j]
            if x < 0 or x >= W or y < 0 or y >= H or joints_cam[i, j, 2] <= 0:
                visibility[i, j] = 0.0
    
    print(f"    Avg visibility: {visibility.mean():.2%}")
    print(f"    Sample 2D joint (frame 0, pelvis): {joints_2d[0, 0]}")
    
    # ========== 9. Save results ==========
    print("\n[9] Saving results...")
    
    prefix = f"{subject_name}_{pose_name}"
    
    # SMPL parameters
    smpl_output = {
        'global_orient': global_orient_cam_all,
        'body_pose': body_pose_all,
        'betas': betas[:10].astype(np.float32),
        'trans': trans_cam_all,
        
        # Original
        'poses_original': poses.astype(np.float32),
        'trans_original': trans_orig.astype(np.float32),
        
        # Metadata
        'gender': np.array(gender),
        'num_frames': np.array(N),
    }
    
    if scene_transform is not None:
        smpl_output['scene_transform'] = scene_transform.astype(np.float64)
        smpl_output['scene_scale'] = scale_scene.astype(np.float32)
    
    smpl_path = os.path.join(output_dir, f"{prefix}_smpl.npz")
    np.savez(smpl_path, **smpl_output)
    print(f"    SMPL: {smpl_path}")
    
    # Joints
    joints_output = {
        'joints_3d_cam': joints_cam.astype(np.float32),
        'joints_3d_world': joints_world.astype(np.float32),
        'joints_2d': joints_2d,
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
    
    # Per-frame annotations
    print("\n[10] Saving per-frame annotations...")
    
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
            'joints_3d': joints_cam[i].tolist(),
            'joints_2d': joints_2d[i].tolist(),
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
    print(f"Verification error: {mean_error*1000:.3f} mm")
    print(f"Output: {output_dir}")


def main():
    parser = ArgumentParser(description="Generate HybrIK data using rendered joints")
    
    parser.add_argument('--amass_npz', type=str, required=True,
                        help='Input AMASS motion NPZ file')
    parser.add_argument('--rendered_joints_dir', type=str, required=True,
                        help='Directory containing rendered joints (.npy files)')
    parser.add_argument('--camera_json', type=str, required=True,
                        help='Path to 3DGS camera JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--smpl_model_path', type=str, required=True,
                        help='Path to SMPL model files')
    
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene key in scene_transforms.py (optional)')
    parser.add_argument('--transform_file', type=str, default=None,
                        help='Path to scene_transforms.py')
    parser.add_argument('--subject_name', type=str, default='unknown')
    parser.add_argument('--pose_name', type=str, default='unknown')
    
    args = parser.parse_args()
    
    # Load scene transform if specified
    scene_transform = None
    if args.scene:
        SCENE_TRANSFORMS = load_scene_transforms(args.transform_file)
        if args.scene not in SCENE_TRANSFORMS:
            print(f"Error: Scene '{args.scene}' not found!")
            print(f"Available: {list(SCENE_TRANSFORMS.keys())}")
            return
        scene_transform = np.array(SCENE_TRANSFORMS[args.scene], dtype=np.float64)
    
    process_with_rendered_joints(
        amass_npz=args.amass_npz,
        rendered_joints_dir=args.rendered_joints_dir,
        camera_json=args.camera_json,
        output_dir=args.output_dir,
        smpl_model_path=args.smpl_model_path,
        scene_transform=scene_transform,
        subject_name=args.subject_name,
        pose_name=args.pose_name,
    )


if __name__ == '__main__':
    main()
