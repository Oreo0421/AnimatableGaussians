#!/usr/bin/env python3
"""
Verify SMPL parameters by:
1. Regenerating joints from SMPL params
2. Comparing with saved joints
3. Visualizing 2D projections on images (if available)
4. Rendering SMPL mesh (optional)

Usage:
    # Basic verification (compare joints)
    python verify_smpl.py \
        --smpl_npz /path/to/101010_brooming_smpl.npz \
        --joints_npz /path/to/101010_brooming_joints.npz \
        --camera_npz /path/to/101010_brooming_camera.npz \
        --smpl_model_path /path/to/smpl_files

    # With image visualization
    python verify_smpl.py \
        --smpl_npz /path/to/smpl.npz \
        --joints_npz /path/to/joints.npz \
        --camera_npz /path/to/camera.npz \
        --smpl_model_path /path/to/smpl_files \
        --image_dir /path/to/rendered_images \
        --output_dir /path/to/visualization

    # Render SMPL mesh
    python verify_smpl.py \
        --smpl_npz /path/to/smpl.npz \
        --joints_npz /path/to/joints.npz \
        --camera_npz /path/to/camera.npz \
        --smpl_model_path /path/to/smpl_files \
        --render_mesh \
        --output_dir /path/to/visualization
"""

import os
import numpy as np
import json
from argparse import ArgumentParser
from tqdm import tqdm


def load_data(smpl_npz, joints_npz, camera_npz):
    """Load all NPZ files."""
    smpl_data = dict(np.load(smpl_npz, allow_pickle=True))
    joints_data = dict(np.load(joints_npz, allow_pickle=True))
    camera_data = dict(np.load(camera_npz, allow_pickle=True))
    
    return smpl_data, joints_data, camera_data


def regenerate_joints_from_smpl(smpl_data, smpl_model_path):
    """Regenerate joints from SMPL parameters."""
    try:
        import torch
        import smplx
    except ImportError:
        raise ImportError("Please install: pip install smplx torch")
    
    global_orient = smpl_data['global_orient']
    body_pose = smpl_data['body_pose']
    betas = smpl_data['betas']
    trans = smpl_data['trans']
    
    N = global_orient.shape[0]
    body_pose_dim = body_pose.shape[1]
    
    # Determine model type
    if body_pose_dim <= 63:
        model_type = 'smplx'
    else:
        model_type = 'smpl'
    
    # Get gender
    gender = smpl_data.get('gender', 'neutral')
    if isinstance(gender, np.ndarray):
        gender = str(gender)
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    if gender.startswith("b'"):
        gender = gender[2:-1]
    if gender not in ['male', 'female', 'neutral']:
        gender = 'neutral'
    
    print(f"  Model type: {model_type}")
    print(f"  Gender: {gender}")
    print(f"  Body pose dim: {body_pose_dim}")
    
    # Load model
    body_model = smplx.create(
        smpl_model_path,
        model_type=model_type,
        gender=gender,
        batch_size=1
    )
    
    # Generate joints
    all_joints = []
    all_vertices = []
    
    for i in tqdm(range(N), desc="Regenerating joints"):
        with torch.no_grad():
            output = body_model(
                global_orient=torch.tensor(global_orient[i:i+1]).float(),
                body_pose=torch.tensor(body_pose[i:i+1]).float(),
                betas=torch.tensor(betas[:10]).float().unsqueeze(0),
                transl=torch.tensor(trans[i:i+1]).float()
            )
            
            joints = output.joints[0, :24].numpy()
            vertices = output.vertices[0].numpy()
            
            all_joints.append(joints)
            all_vertices.append(vertices)
    
    return np.array(all_joints), np.array(all_vertices)


def compare_joints(joints_regenerated, joints_saved, tolerance=0.01):
    """Compare regenerated joints with saved joints."""
    print("\n" + "=" * 60)
    print("JOINT COMPARISON")
    print("=" * 60)
    
    # Ensure same number of joints
    num_joints = min(joints_regenerated.shape[1], joints_saved.shape[1])
    
    joints_regen = joints_regenerated[:, :num_joints]
    joints_save = joints_saved[:, :num_joints]
    
    # Per-frame error
    frame_errors = np.linalg.norm(joints_regen - joints_save, axis=2).mean(axis=1)
    
    # Overall statistics
    mean_error = frame_errors.mean()
    max_error = frame_errors.max()
    min_error = frame_errors.min()
    
    print(f"  Frames: {len(frame_errors)}")
    print(f"  Joints compared: {num_joints}")
    print(f"  Mean error: {mean_error:.6f} m ({mean_error * 1000:.3f} mm)")
    print(f"  Max error:  {max_error:.6f} m ({max_error * 1000:.3f} mm)")
    print(f"  Min error:  {min_error:.6f} m ({min_error * 1000:.3f} mm)")
    
    if mean_error < tolerance:
        print(f"\n  ✅ PASSED: Mean error < {tolerance} m")
        return True
    else:
        print(f"\n  ❌ FAILED: Mean error >= {tolerance} m")
        
        # Find worst frames
        worst_frames = np.argsort(frame_errors)[-5:][::-1]
        print(f"\n  Worst frames:")
        for idx in worst_frames:
            print(f"    Frame {idx}: error = {frame_errors[idx]:.6f} m")
        
        return False


def project_joints_to_2d(joints_3d, camera_data):
    """Project 3D joints to 2D."""
    K = camera_data['K']
    
    N, J, _ = joints_3d.shape
    joints_2d = np.zeros((N, J, 2))
    
    for i in range(N):
        pts = joints_3d[i]  # (J, 3)
        pts_proj = (K @ pts.T).T  # (J, 3)
        joints_2d[i] = pts_proj[:, :2] / pts_proj[:, 2:3]
    
    return joints_2d


def visualize_joints_on_image(image, joints_2d, output_path, 
                              color_regen=(0, 255, 0), color_saved=(255, 0, 0)):
    """Draw joints on image."""
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not installed, skipping visualization")
        return
    
    img = image.copy()
    
    # Draw joints
    for j in range(joints_2d.shape[0]):
        x, y = int(joints_2d[j, 0]), int(joints_2d[j, 1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 3, color_regen, -1)
            cv2.putText(img, str(j), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, color_regen, 1)
    
    cv2.imwrite(output_path, img)


def visualize_comparison(joints_2d_regen, joints_2d_saved, image, output_path):
    """Draw both regenerated and saved joints for comparison."""
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not installed, skipping visualization")
        return
    
    img = image.copy()
    
    # Draw saved joints (red)
    for j in range(joints_2d_saved.shape[0]):
        x, y = int(joints_2d_saved[j, 0]), int(joints_2d_saved[j, 1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    
    # Draw regenerated joints (green)
    for j in range(joints_2d_regen.shape[0]):
        x, y = int(joints_2d_regen[j, 0]), int(joints_2d_regen[j, 1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    
    # Add legend
    cv2.putText(img, "Red: Saved", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Green: Regenerated", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, img)


def create_blank_image(H, W):
    """Create blank white image."""
    return np.ones((H, W, 3), dtype=np.uint8) * 255


def draw_skeleton(img, joints_2d, color=(0, 255, 0), thickness=2):
    """Draw skeleton connections."""
    try:
        import cv2
    except ImportError:
        return img
    
    # SMPL skeleton connections (24 joints)
    connections = [
        (0, 1), (0, 2), (0, 3),      # pelvis to legs and spine
        (1, 4), (2, 5), (3, 6),      # upper legs, spine
        (4, 7), (5, 8), (6, 9),      # lower legs, spine
        (7, 10), (8, 11), (9, 12),   # feet, neck
        (12, 13), (12, 14),          # neck to head and shoulders
        (13, 16), (14, 17),          # shoulders to elbows  
        (16, 18), (17, 19),          # elbows to wrists
        (18, 20), (19, 21),          # wrists to hands
        (12, 15),                     # neck to nose
    ]
    
    for (i, j) in connections:
        if i < len(joints_2d) and j < len(joints_2d):
            x1, y1 = int(joints_2d[i, 0]), int(joints_2d[i, 1])
            x2, y2 = int(joints_2d[j, 0]), int(joints_2d[j, 1])
            
            if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and
                0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def render_mesh_to_image(vertices, faces, camera_data, output_path):
    """Render SMPL mesh to image using pyrender."""
    try:
        import trimesh
        import pyrender
    except ImportError:
        print("Warning: trimesh/pyrender not installed, skipping mesh render")
        print("Install with: pip install trimesh pyrender")
        return
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # Create scene
    scene = pyrender.Scene()
    scene.add(mesh)
    
    # Camera
    K = camera_data['K']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = int(camera_data['image_height']), int(camera_data['image_width'])
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    
    # Camera pose (identity since joints are already in camera coords)
    camera_pose = np.eye(4)
    camera_pose[1, 1] = -1  # Flip Y
    camera_pose[2, 2] = -1  # Flip Z
    
    scene.add(camera, pose=camera_pose)
    
    # Light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(W, H)
    color, depth = renderer.render(scene)
    
    # Save
    try:
        import cv2
        cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    except:
        from PIL import Image
        Image.fromarray(color).save(output_path)
    
    renderer.delete()


def print_data_summary(smpl_data, joints_data, camera_data):
    """Print summary of loaded data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print("\nSMPL Data:")
    for key, val in smpl_data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
    
    print("\nJoints Data:")
    for key, val in joints_data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
    
    print("\nCamera Data:")
    for key, val in camera_data.items():
        if isinstance(val, np.ndarray):
            if val.size < 20:
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: shape={val.shape}")
        else:
            print(f"  {key}: {val}")


def main():
    parser = ArgumentParser(description="Verify SMPL parameters")
    
    parser.add_argument('--smpl_npz', type=str, required=True)
    parser.add_argument('--joints_npz', type=str, required=True)
    parser.add_argument('--camera_npz', type=str, required=True)
    parser.add_argument('--smpl_model_path', type=str, required=True)
    
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing rendered images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--render_mesh', action='store_true',
                        help='Render SMPL mesh')
    parser.add_argument('--num_vis_frames', type=int, default=10,
                        help='Number of frames to visualize')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Error tolerance in meters')
    
    args = parser.parse_args()
    
    # Load data
    print("\n[1] Loading data...")
    smpl_data, joints_data, camera_data = load_data(
        args.smpl_npz, args.joints_npz, args.camera_npz
    )
    
    # Print summary
    print_data_summary(smpl_data, joints_data, camera_data)
    
    # Regenerate joints
    print("\n[2] Regenerating joints from SMPL...")
    joints_regen, vertices = regenerate_joints_from_smpl(smpl_data, args.smpl_model_path)
    
    # Compare with saved joints
    print("\n[3] Comparing joints...")
    joints_saved = joints_data['joints_3d_cam']
    passed = compare_joints(joints_regen, joints_saved, tolerance=args.tolerance)
    
    # Compare 2D projections
    print("\n[4] Comparing 2D projections...")
    joints_2d_regen = project_joints_to_2d(joints_regen, camera_data)
    joints_2d_saved = joints_data['joints_2d']
    
    # Use minimum number of joints
    num_joints_2d = min(joints_2d_regen.shape[1], joints_2d_saved.shape[1])
    error_2d = np.linalg.norm(
        joints_2d_regen[:, :num_joints_2d] - joints_2d_saved[:, :num_joints_2d], 
        axis=2
    ).mean()
    print(f"  Mean 2D error: {error_2d:.2f} pixels")
    
    # Visualizations
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"\n[5] Creating visualizations in {args.output_dir}...")
        
        try:
            import cv2
            
            H = int(camera_data['image_height'])
            W = int(camera_data['image_width'])
            N = len(joints_regen)
            
            # Select frames to visualize
            frame_indices = np.linspace(0, N-1, args.num_vis_frames, dtype=int)
            
            for i, frame_idx in enumerate(tqdm(frame_indices, desc="Visualizing")):
                # Load or create image
                img = None
                if args.image_dir:
                    # Try multiple filename patterns
                    patterns = [
                        f"{frame_idx:08d}.png",
                        f"{frame_idx:08d}.jpg",
                        f"frame_{frame_idx:08d}.png",
                        f"frame_{frame_idx:08d}.jpg",
                        f"{frame_idx:06d}.png",
                        f"{frame_idx:06d}.jpg",
                        f"frame_{frame_idx:06d}.png",
                        f"frame_{frame_idx:06d}.jpg",
                    ]
                    
                    for pattern in patterns:
                        img_path = os.path.join(args.image_dir, pattern)
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            break
                
                if img is None:
                    img = create_blank_image(H, W)
                
                # Draw comparison
                img_compare = img.copy()
                
                # Draw saved (red)
                num_joints_vis = min(24, joints_2d_saved.shape[1])
                img_compare = draw_skeleton(img_compare, joints_2d_saved[frame_idx, :num_joints_vis], 
                                           color=(0, 0, 255), thickness=2)
                for j in range(num_joints_vis):
                    x, y = int(joints_2d_saved[frame_idx, j, 0]), int(joints_2d_saved[frame_idx, j, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(img_compare, (x, y), 5, (0, 0, 255), -1)
                
                # Draw regenerated (green)
                img_compare = draw_skeleton(img_compare, joints_2d_regen[frame_idx],
                                           color=(0, 255, 0), thickness=1)
                for j in range(joints_2d_regen.shape[1]):
                    x, y = int(joints_2d_regen[frame_idx, j, 0]), int(joints_2d_regen[frame_idx, j, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(img_compare, (x, y), 3, (0, 255, 0), -1)
                
                # Add legend
                cv2.putText(img_compare, f"Frame {frame_idx}", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(img_compare, "Red: Saved joints", (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img_compare, "Green: Regenerated from SMPL", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                out_path = os.path.join(args.output_dir, f"compare_{frame_idx:08d}.png")
                cv2.imwrite(out_path, img_compare)
            
            print(f"  Saved {len(frame_indices)} comparison images")
            
        except ImportError:
            print("  Warning: cv2 not installed, skipping 2D visualization")
        
        # Render mesh
        if args.render_mesh:
            print("\n[6] Rendering SMPL mesh...")
            try:
                import smplx
                
                # Get faces from SMPL model
                gender = smpl_data.get('gender', 'neutral')
                if isinstance(gender, np.ndarray):
                    gender = str(gender)
                if isinstance(gender, bytes):
                    gender = gender.decode('utf-8')
                if gender.startswith("b'"):
                    gender = gender[2:-1]
                if gender not in ['male', 'female', 'neutral']:
                    gender = 'neutral'
                
                body_model = smplx.create(
                    args.smpl_model_path,
                    model_type='smplx',
                    gender=gender
                )
                faces = body_model.faces
                
                for i, frame_idx in enumerate(tqdm(frame_indices[:3], desc="Rendering mesh")):
                    out_path = os.path.join(args.output_dir, f"mesh_{frame_idx:08d}.png")
                    render_mesh_to_image(vertices[frame_idx], faces, camera_data, out_path)
                
            except Exception as e:
                print(f"  Mesh rendering failed: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    if passed:
        print("✅ SMPL parameters are CORRECT!")
        print("   Regenerated joints match saved joints within tolerance.")
    else:
        print("❌ SMPL parameters may have issues.")
        print("   Check the visualization outputs for details.")
    
    return passed


if __name__ == '__main__':
    main()
