#!/usr/bin/env python3
"""
stickfigure_gaussians.py
Turn joints.npy into a colour-coded Gaussian stick-figure (.pt).

Colours
-------
  • Legs   : red
  • Arms   : green
  • Spine+head : blue
  • Joints : light grey

Parameters
----------
--steps       Gaussians per edge  (default 40)
--joint_rad   radius of joint blobs   (default 0.03 m)
--stick_rad   radius of sticks        (default 0.02 m)
"""

import argparse, math, numpy as np, torch

# ─────────────────────  joint indices & edges  ──────────────────────────
# 0 = pelvis root
EDGE_LIST = {
    "right_leg": [(0, 1), (1, 4), (4, 7), (7, 10)],
    "left_leg" : [(0, 2), (2, 5), (5, 8), (8, 11)],
    "spine"    : [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)],
    "right_arm": [(9, 13), (13, 16), (16, 18), (18, 20)],
    "left_arm" : [(9, 14), (14, 17), (17, 19), (19, 21)],
}

PART_COLOUR = {
    "right_leg": torch.tensor([1.0, 0.0, 0.0]),   # red
    "left_leg" : torch.tensor([1.0, 0.0, 0.0]),   # red
    "spine"    : torch.tensor([0.0, 0.0, 1.0]),   # blue
    "right_arm": torch.tensor([0.0, 1.0, 0.0]),   # green
    "left_arm" : torch.tensor([0.0, 1.0, 0.0]),   # green
}

# ─────────────────────────────────────────────────────────────────────────
def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """RGB ∈ [0,1] → degree-0 SH coeff (divide by √π)."""
    return rgb / math.sqrt(math.pi)

def build_stickfigure(joints: np.ndarray,
                      steps: int = 40,
                      joint_rad: float = 0.03,
                      stick_rad: float = 0.02):
    joints_t = torch.from_numpy(joints).float()        # (J,3)

    xyz_list, col_list, rad_list = [], [], []

    # 1) joints (light grey)
    xyz_list.append(joints_t)
    col_list.append(torch.full_like(joints_t, 0.7))
    rad_list.append(torch.full((len(joints_t),), joint_rad))

    # 2) sticks per body part
    for part, edges in EDGE_LIST.items():
        colour = PART_COLOUR[part]
        for i, j in edges:
            v0, v1 = joints_t[i], joints_t[j]
            t = torch.linspace(0, 1, steps + 2)[1:-1].unsqueeze(1)  # skip ends
            pts = v0 + t * (v1 - v0)
            xyz_list.append(pts)
            col_list.append(colour.expand_as(pts))
            rad_list.append(torch.full((steps,), stick_rad))

    # concat everything
    xyz   = torch.cat(xyz_list, 0)
    rgb   = torch.cat(col_list, 0)
    radii = torch.cat(rad_list, 0)

    N = xyz.shape[0]
    scales = radii.unsqueeze(1).repeat(1, 3)           # (N,3)
    rotq   = torch.tensor([1., 0., 0., 0.]).repeat(N,1)
    shs    = torch.zeros((N, 16, 3))
    shs[:, 0, :] = rgb_to_sh_dc(rgb)

    return {
        "xyz": xyz,
        "scales": scales,
        "rotq": rotq,
        "shs": shs,
        "opacity": torch.ones((N,1)),
        "active_sh_degree": 3,
    }

# ──────────────────────────────  CLI  ────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", required=True, help="N×3 joints.npy (metres)")
    ap.add_argument("--out",    required=True, help="output .pt path")
    ap.add_argument("--steps",  type=int, default=40,   help="Gaussians per edge")
    ap.add_argument("--joint_rad", type=float, default=0.03)
    ap.add_argument("--stick_rad", type=float, default=0.02)
    args = ap.parse_args()

    j_np = np.load(args.joints)
    if j_np.shape[-1] != 3:
        raise ValueError("joints.npy must be N×3 XYZ positions")

    gs_dict = build_stickfigure(j_np,
                                steps=args.steps,
                                joint_rad=args.joint_rad,
                                stick_rad=args.stick_rad)

    torch.save(gs_dict, args.out)
    print(f" saved {gs_dict['xyz'].shape[0]} Gaussians → {args.out}")
