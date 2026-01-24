
#!/usr/bin/env python3
"""
Convert multiple joint .npy files into .pt Gaussian files.

Usage:
    python joint_npy2pt.py --input_dir /path/to/npy --output_dir /path/to/save

- Processes only files with 8-digit numeric names (e.g., 00000000.npy)
- Each output .pt file has the same basename.
"""

import os
import re
import math
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

def convert_npy_to_pt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Match files like 00000000.npy
    pattern = re.compile(r"^\d{8}\.npy$")
    files = sorted([f for f in os.listdir(input_dir) if pattern.match(f)])

    if not files:
        print(f"❌ No matching .npy files found in {input_dir}")
        return

    for fname in tqdm(files, desc="Converting .npy to .pt"):
        npy_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # Load joint centres (J×3, metres)
        xyz = torch.from_numpy(np.load(npy_path)).float().cuda()
        J = xyz.shape[0]

        # --- Gaussian geometry ---
        scales  = torch.ones(J, 3, device='cuda') * 0.02  # 2 cm
        rotq    = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                               device='cuda').repeat(J, 1)
        opacity = torch.ones(J, 1, device='cuda')

        # --- SH colour for degree-3 ---
        active_sh_degree = 3
        n_coeffs = (active_sh_degree + 1) ** 2  # 16
        shs = torch.zeros(J, n_coeffs, 3, device='cuda')

        # Set only the DC term (index 0) → pure red dots.
        shs[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32,
                                    device='cuda') / math.sqrt(math.pi)

        # --- Pack and save ---
        gs = {
            "xyz":     xyz,
            "scales":  scales,
            "rotq":    rotq,
            "shs":     shs,
            "opacity": opacity,
            "active_sh_degree": active_sh_degree,
        }

        pt_path = os.path.join(output_dir, f"{base_name}.pt")
        torch.save(gs, pt_path)
        print(f"Saved {pt_path} with {J} Gaussians, SH degree {active_sh_degree}")

    print(f" Converted {len(files)} files from {input_dir} to {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Batch convert joint .npy to .pt files")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .npy joint files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save .pt files")
    args = parser.parse_args()

    convert_npy_to_pt(args.input_dir, args.output_dir)
