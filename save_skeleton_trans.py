#!/usr/bin/env python3
"""
Example
-------
python save_skeleton_trans.py skeleton.ply skeleton_xf.ply
"""

import argparse
import numpy as np
import trimesh


# --- the transform you supplied ------------------------------------------------
TRANS_MATRIX = np.array([
    [ 1.417337, -3.719793,  5.757977, 20.600479],
    [-0.786601, -5.929179, -3.636770,  2.822414],
    [ 6.809730,  0.089329, -1.618520, 25.250027],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
], dtype=np.float32)


def apply_transform(mesh: trimesh.Trimesh, M: np.ndarray) -> trimesh.Trimesh:
    """Return a *new* trimesh with vertices transformed by M (4×4)."""
    v = mesh.vertices                       # (N,3)
    v_h = np.column_stack((v, np.ones(len(v), dtype=v.dtype)))   # -> (N,4)
    v_xf = (M @ v_h.T).T[:, :3]             # back to (N,3)
    mesh_tf = mesh.copy()
    mesh_tf.vertices = v_xf
    return mesh_tf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_ply",  help="input PLY (skeleton.ply)")
    ap.add_argument("out_ply", help="output PLY with transform applied")
    args = ap.parse_args()

    mesh = trimesh.load(args.in_ply, process=False)
    mesh_tf = apply_transform(mesh, TRANS_MATRIX)

    mesh_tf.export(args.out_ply)
    print(f" wrote transformed mesh → {args.out_ply}")


if __name__ == "__main__":
    main()
