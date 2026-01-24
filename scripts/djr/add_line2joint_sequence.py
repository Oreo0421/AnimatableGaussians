#!/usr/bin/env python3
"""
stickfigure_gaussians_batch.py
Turn a sequence of joints .npy files (e.g., 00000000.npy–00000099.npy)
into colour-coded Gaussian stick-figures (.pt) with matching names.

Example:
  python stickfigure_gaussians_batch.py \
      --in_dir path/to/joints \
      --out_dir path/to/skeletons \
      --start 0 --end 99 \
      --steps 40 --joint_rad 0.03 --stick_rad 0.02
"""

import argparse, math, os, numpy as np, torch
from pathlib import Path

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  required=True, help="Directory containing frame .npy files (e.g., 00000000.npy).")
    ap.add_argument("--out_dir", required=True, help="Directory to write matching .pt files (e.g., 00000000.pt).")
    ap.add_argument("--start", type=int, default=0,  help="Start frame index (default 0).")
    ap.add_argument("--end",   type=int, default=99, help="End frame index inclusive (default 99).")
    ap.add_argument("--steps",  type=int, default=40,   help="Gaussians per edge.")
    ap.add_argument("--joint_rad", type=float, default=0.03)
    ap.add_argument("--stick_rad", type=float, default=0.02)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for idx in range(args.start, args.end + 1):
        stem = f"{idx:08d}"
        in_path  = in_dir / f"{stem}.npy"
        out_path = out_dir / f"{stem}.pt"

        if not in_path.exists():
            print(f"[skip] missing {in_path}")
            continue

        j_np = np.load(in_path)
        if j_np.shape[-1] != 3:
            raise ValueError(f"{in_path} must be N×3 XYZ positions (got {j_np.shape}).")

        gs_dict = build_stickfigure(j_np,
                                    steps=args.steps,
                                    joint_rad=args.joint_rad,
                                    stick_rad=args.stick_rad)
        torch.save(gs_dict, out_path)
        print(f"[ok] {in_path.name} → {out_path.name}  ({gs_dict['xyz'].shape[0]} Gaussians)")
        total_saved += 1

    print(f"Done. Saved {total_saved} file(s) to {out_dir}")

if __name__ == "__main__":
    main()
