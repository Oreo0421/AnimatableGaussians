'''
python skeleton_ply2pt.py \       ./output_joints_skeleton/transformed/skeleton/ply/00000000_trans_skeleton.ply \
       ./output_joints_skeleton/transformed/skeleton/pt//00000000_trans_skeleton.pt
'''

import argparse, math, pathlib, torch, trimesh
from typing import Tuple

def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] → degree-0 SH (divide by √π)."""
    return rgb / math.sqrt(math.pi)

def load_mesh(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mesh = trimesh.load(path, process=False)  # keep vertices unaltered
    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)             # (N,3)

    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors):
        vc = mesh.visual.vertex_colors[:, :3]           # RGBA -> RGB
        colors = torch.as_tensor(vc, dtype=torch.float32) / 255.0           # (N,3) in [0,1]
    else:
        colors = torch.ones_like(verts)                 # default white
    return verts, colors

def make_gaussian_dict(xyz: torch.Tensor,
                       rgb: torch.Tensor,
                       radius: float,
                       sh_degree: int = 3) -> dict:
    N = xyz.shape[0]
    scales = torch.full((N, 3), radius, dtype=torch.float32)
    rotq   = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).repeat(N, 1)
    shs    = torch.zeros((N, (sh_degree + 1) ** 2, 3), dtype=torch.float32)
    shs[:, 0, :] = rgb_to_sh_dc(rgb)   # fill DC; higher coeffs stay zero
    opacity = torch.ones((N, 1), dtype=torch.float32)

    return {
        "xyz": xyz,
        "scales": scales,
        "rotq": rotq,
        "shs": shs,
        "opacity": opacity,
        "active_sh_degree": sh_degree,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_ply",  type=str, help="input skeleton.ply")
    p.add_argument("out_pt",  type=str, help="output skeleton.pt")
    p.add_argument("--radius", type=float, default=0.01,
                   help="Gaussian radius in *world units* (default 0.01 m)")
    args = p.parse_args()

    xyz, rgb = load_mesh(args.in_ply)
    gs_dict  = make_gaussian_dict(xyz, rgb, args.radius)

    torch.save(gs_dict, args.out_pt)
    print(f" wrote {xyz.shape[0]} Gaussians → {args.out_pt}")

if __name__ == "__main__":
    main()
