import numpy as np

src_npz = "pose_00.npz"
tgt_npz = "pose_01.npz"
out_npz = "01.npz"

src = np.load(src_npz)
tgt = np.load(tgt_npz)

def to_Nx3(arr, N, name):
    arr = np.asarray(arr)

    # case 1: already (3,)
    if arr.shape == (3,):
        return np.tile(arr[None, :], (N, 1))

    # case 2: already (N,3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if arr.shape[0] == N:
            return arr
        # mismatch: use first frame and repeat
        return np.tile(arr[0:1, :], (N, 1))

    raise ValueError(f"{name} has unexpected shape {arr.shape}, expected (3,) or (K,3)")

# infer target length N from target file (prefer body_pose if present)
if "body_pose" in tgt.files and tgt["body_pose"].ndim == 2:
    N = tgt["body_pose"].shape[0]
elif "global_orient" in tgt.files and tgt["global_orient"].ndim == 2:
    N = tgt["global_orient"].shape[0]
else:
    raise ValueError("Cannot infer frame count N from target npz (need body_pose or global_orient as (N,*)).")

new_data = {k: tgt[k] for k in tgt.files}

new_data["global_orient"] = to_Nx3(src["global_orient"], N, "global_orient")
new_data["transl"]        = to_Nx3(src["transl"],        N, "transl")

np.savez(out_npz, **new_data)

print("Saved:", out_npz)
print("N =", N, "| global_orient shape:", new_data["global_orient"].shape, "| transl shape:", new_data["transl"].shape)