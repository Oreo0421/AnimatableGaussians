#!/usr/bin/env python3
"""
smplx_model.py ── single‑frame SMPL‑X ➜ transformed Gaussian point cloud
──────────────────────────────────────────────────────────────────────────────
Outputs **three** files in the same folder as your param files:

* `smpl_mesh.ply`      – transformed mesh (full topology)
* `smpl_pointcloud.ply`– 10 k‑sampled point cloud (PLY)
* `smpl_pointcloud.pt` – dict that matches Animatable Gaussians format:

      xyz   : (N, 3)  – world‑space positions
      scales: (N, 3)  – isotropic scale σ along x,y,z  (here: 0.01 m default)
      rotq  : (N, 4)  – quaternion [w, x, y, z]  (identity)
      shs   : (N,16,3)– SH coeffs (degree‑3 ⇒ 16) for RGB  (zero‑init)
      opacity:(N, 1)  – α per point             (1.0 default)
      active_sh_degree: 3

Edit the hard‑coded paths below (`SMPL_PARAMS_PATH`, `POSE_PATH`, `SMPL_MODEL_DIR`) if needed. Adjust `TRANSFORM` to place the avatar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import torch

# ───────────────────────────────
# USER CONFIGURATION
# ───────────────────────────────
SMPL_PARAMS_PATH = Path("/home/zhiyw/Desktop/EasyMocap/mydata/output_0165/smpl_undis/smpl/000000.json")
POSE_PATH        = Path("/home/zhiyw/Desktop/EasyMocap/mydata/output_0165/smpl_undis/smpl/000000.json")
SMPL_MODEL_DIR   = Path("/home/zhiyw/Desktop/AnimatableGaussians/smpl_files")  # parent dir containing smplx/

OUT_DIR          = SMPL_PARAMS_PATH.parent
OUT_MESH_PLY     = OUT_DIR / "smpl_mesh.ply"
OUT_CLOUD_PLY    = OUT_DIR / "smpl_pointcloud.ply"
OUT_CLOUD_PT     = OUT_DIR / "smpl_pointcloud.pt"

# World‑space 4×4 transform
TRANSFORM = np.array([
    [1.417337, -3.719793,  5.757977, 20.600479],
    [-0.786601, -5.929179, -3.636770,  2.822414],
    [6.809730,   0.089329, -1.618520, 25.250027],
    [0.0,        0.0,       0.0,       1.0      ]
], dtype=np.float64)

# Gaussian defaults
DEFAULT_SCALE   = 0.01          # metres
DEFAULT_OPACITY = 1.0
SH_DEGREE       = 3             # ⇒ 16 coeffs per colour channel


# ───────────────────────────────
# Utility helpers
# ───────────────────────────────

def first_frame(arr: np.ndarray):
    arr = np.asarray(arr)
    return arr[0] if arr.ndim > 1 else arr


def apply_transform(verts: np.ndarray, T: np.ndarray):
    N = verts.shape[0]
    verts_h = np.hstack([verts, np.ones((N, 1), dtype=verts.dtype)])  # (N,4)
    return (T @ verts_h.T).T[:, :3]


def save_ply_pointcloud(points: np.ndarray, path: Path):
    vertex_data = np.array([(x, y, z) for x, y, z in points],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(path)
    print(f"Point cloud saved to: {path}")


def save_ply_mesh(verts: np.ndarray, faces: np.ndarray | None, path: Path):
    if faces is None:
        save_ply_pointcloud(verts, path)
        return
    vdat = np.array([(x, y, z) for x, y, z in verts],
                    dtype=[('x','f4'),('y','f4'),('z','f4')])
    fdat = np.array([(faces[i].tolist(),) for i in range(len(faces))],
                    dtype=[('vertex_indices','i4', (3,))])
    PlyData([PlyElement.describe(vdat,'vertex'), PlyElement.describe(fdat,'face')]).write(path)
    print(f"Mesh saved to: {path}")


def mesh_to_pointcloud(verts: np.ndarray, faces: np.ndarray | None, n: int = 10000):
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts = mesh.sample(n)
    else:
        pts = verts[:n] if len(verts) >= n else verts
    return pts


# ───────────────────────────────
# Main
# ───────────────────────────────

def main():
    print("Loading SMPL parameters…")
    smpl_data = np.load(SMPL_PARAMS_PATH, allow_pickle=True)
    pose_data = np.load(POSE_PATH, allow_pickle=True)
    print("SMPL keys:", list(smpl_data.keys()))

    try:
        import smplx
    except ImportError as err:
        sys.exit(f"smplx not installed: {err}")

    # ── build SMPL‑X model ────────────────────────────────────────────────
    print("Creating SMPL‑X model…")
    model = smplx.create(
        SMPL_MODEL_DIR,
        model_type='smplx',
        gender='neutral',
        ext='npz',
        use_pca=False,
    )

    # First frame params
    betas         = first_frame(smpl_data.get('betas',           np.zeros(10)))
    body_pose     = first_frame(pose_data.get('body_pose',       np.zeros(63)))
    global_orient = first_frame(pose_data.get('global_orient',   np.zeros(3)))
    jaw_pose      = first_frame(pose_data.get('jaw_pose',        np.zeros(3)))
    expression    = first_frame(pose_data.get('expression',      np.zeros(10)))
    lhand_pose    = first_frame(pose_data.get('left_hand_pose',  np.zeros(45)))
    rhand_pose    = first_frame(pose_data.get('right_hand_pose', np.zeros(45)))
    transl        = first_frame(pose_data.get('transl',          np.zeros(3)))

    t = lambda a: torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)

    output = model(
        betas=t(betas), body_pose=t(body_pose), global_orient=t(global_orient),
        jaw_pose=t(jaw_pose), expression=t(expression),
        left_hand_pose=t(lhand_pose), right_hand_pose=t(rhand_pose), transl=t(transl))

    vertices = output.vertices.detach().cpu().numpy()[0]
    faces    = model.faces if hasattr(model,'faces') else None

    # ── world transform & IO ───────────────────────────────────────────────
    verts_w = apply_transform(vertices, TRANSFORM)

    print("Saving PLY files…")
    save_ply_mesh(verts_w, faces, OUT_MESH_PLY)

    pts = mesh_to_pointcloud(verts_w, faces, 10000)
    save_ply_pointcloud(pts, OUT_CLOUD_PLY)

    # ── build Gaussian dict ───────────────────────────────────────────────
    N = pts.shape[0]
    xyz     = torch.from_numpy(pts.astype(np.float32))                         # (N,3)
    scales  = torch.full((N,3), DEFAULT_SCALE, dtype=torch.float32)            # (N,3)
    rotq    = torch.zeros((N,4), dtype=torch.float32); rotq[:,0] = 1.0         # (N,4) identity quat
    shs     = torch.zeros((N, 16, 3), dtype=torch.float32)                     # degree‑3 SH coeffs
    opacity = torch.full((N,1), DEFAULT_OPACITY, dtype=torch.float32)          # (N,1)

    torch.save({
        'xyz'               : xyz,
        'scales'            : scales,
        'rotq'              : rotq,
        'shs'               : shs,
        'opacity'           : opacity,
        'active_sh_degree'  : SH_DEGREE,
    }, OUT_CLOUD_PT)

    print(f"PyTorch dict saved to: {OUT_CLOUD_PT}\nDone!")


if __name__ == "__main__":
    main()
