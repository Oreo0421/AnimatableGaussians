import numpy as np, torch, math

# --- load joint centres (J×3, metres) ---
xyz = torch.from_numpy(np.load("output_joints_directory/transformed/joints_trans_00.npy")).float().cuda()   # (J,3)
J = xyz.shape[0]

# --- Gaussian geometry ---
scales  = torch.ones(J, 3, device='cuda') * 0.02               # 2 cm
rotq    = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                       device='cuda').repeat(J, 1)
opacity = torch.ones(J, 1, device='cuda')

# --- SH colour for degree-3 ---
active_sh_degree = 3
n_coeffs = (active_sh_degree + 1) ** 2                         # 16
shs = torch.zeros(J, n_coeffs, 3, device='cuda')

# Set only the DC term (index 0) → pure red dots.
# Each channel’s DC gets divided by √π.
shs[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32,
                            device='cuda') / math.sqrt(math.pi)

# --- pack and save ---
gs = {
    "xyz":     xyz,
    "scales":  scales,
    "rotq":    rotq,
    "shs":     shs,
    "opacity": opacity,
    "active_sh_degree": active_sh_degree,
}

torch.save(gs, "joints_degree3.pt")
print(f"saved joints_degree3.pt with {J} Gaussians, SH degree {active_sh_degree}")
