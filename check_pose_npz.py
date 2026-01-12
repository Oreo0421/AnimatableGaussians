 
import numpy as np
import sys

path = sys.argv[1]
d = np.load(path, allow_pickle=True)

print("Keys:", d.files)
for k in d.files:
    v = d[k]
    if hasattr(v, "shape"):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")

# 常见字段名尝试推断帧数
for cand in ["poses", "pose_body", "body_pose", "full_pose", "thetas"]:
    if cand in d.files:
        print("Inferred num_frames from", cand, "=", d[cand].shape[0])
        break
