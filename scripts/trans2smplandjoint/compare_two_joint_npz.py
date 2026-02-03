#!/usr/bin/env python3
import argparse
import numpy as np
from collections import deque


def load_joints(npz_path: str, key: str = None):
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.files)

    if key is None:
        # auto-detect common 3D keys (your case included)
        for k in ["joints_3d", "joints3d", "joints",
                  "joints_3d_world", "joints_3d_cam"]:
            if k in keys:
                key = k
                break

    if key is None:
        raise KeyError(f"No joints key found in {npz_path}. keys={keys}")

    if key not in keys:
        raise KeyError(f"Key '{key}' is not in {npz_path}. keys={keys}")

    j = z[key]
    if j.ndim != 3 or j.shape[2] != 3:
        raise ValueError(f"{npz_path}: {key} must be (T,J,3), got {j.shape}")

    Tmat = None
    for tk in ["transformation_matrix", "T", "T_align", "T_total"]:
        if tk in keys:
            t = z[tk]
            if isinstance(t, np.ndarray) and t.shape == (4, 4):
                Tmat = t
                break

    return j.astype(np.float32), key, Tmat, keys


def stats_basic(j):
    flat = j.reshape(-1, 3)
    nan = np.isnan(flat).any()
    inf = np.isinf(flat).any()
    mn = flat.min(0)
    mx = flat.max(0)
    mean = flat.mean(0)
    std = flat.std(0)

    frame_energy = np.linalg.norm(j, axis=2).sum(axis=1)  # (T,)
    zero_frames = int(np.sum(frame_energy < 1e-8))

    return {
        "nan": bool(nan), "inf": bool(inf),
        "min": mn, "max": mx,
        "mean": mean, "std": std,
        "zero_frames": zero_frames
    }


def choose_root_idx_auto(j):
    std_xyz = j.std(axis=0)  # (J,3)
    std_norm = np.linalg.norm(std_xyz, axis=1)  # (J,)
    return int(std_norm.argmin()), std_norm


def temporal_motion_metrics(j):
    v = j[1:] - j[:-1]   # (T-1,J,3)
    a = v[1:] - v[:-1]   # (T-2,J,3)
    vmag = np.linalg.norm(v, axis=2)  # (T-1,J)
    amag = np.linalg.norm(a, axis=2)  # (T-2,J)
    return {
        "vel_mean": float(vmag.mean()),
        "vel_p95": float(np.percentile(vmag, 95)),
        "acc_mean": float(amag.mean()) if amag.size else 0.0,
        "acc_p95": float(np.percentile(amag, 95)) if amag.size else 0.0,
    }


def bone_length_metrics(j, kin_parent):
    kp = np.asarray(kin_parent).astype(int).reshape(-1)
    J = j.shape[1]
    if kp.shape[0] != J:
        raise ValueError(f"kin_parent has {kp.shape[0]} joints but data has J={J}")

    edges = [(i, int(kp[i])) for i in range(J) if int(kp[i]) >= 0 and int(kp[i]) != i]
    if not edges:
        return {"bone_mean_std": None, "bone_p95_std": None, "num_edges": 0}

    bl = []
    for (i, p) in edges:
        d = j[:, i] - j[:, p]            # (T,3)
        bl.append(np.linalg.norm(d, axis=1))  # (T,)
    bl = np.stack(bl, axis=1)            # (T,E)
    std_per_bone = bl.std(axis=0)        # (E,)

    return {
        "num_edges": len(edges),
        "bone_mean_std": float(std_per_bone.mean()),
        "bone_p95_std": float(np.percentile(std_per_bone, 95)),
    }


def root_center_quality(j, root_idx):
    root = j[:, root_idx:root_idx + 1, :]  # (T,1,3)
    jc = j - root
    flat = jc.reshape(-1, 3)
    mean = flat.mean(0)
    std = flat.std(0)
    return {"rc_mean": mean, "rc_std": std}


def score_for_training(metrics):
    s = 0.0
    if metrics["basic"]["nan"] or metrics["basic"]["inf"]:
        s -= 1e6
    s -= metrics["basic"]["zero_frames"] * 100.0

    # jitter penalty
    s -= metrics["motion"]["acc_p95"] * 200.0
    s -= metrics["motion"]["vel_p95"] * 50.0

    # bone stability reward (lower std is better)
    b = metrics["bone"]
    if b["bone_mean_std"] is not None:
        s -= b["bone_mean_std"] * 5000.0
        s -= b["bone_p95_std"] * 2000.0

    return s


def pretty_vec(v):
    return np.array2string(np.asarray(v), precision=6, suppress_small=True)


def analyze(npz_path, kin_parent_path=None, joints_key=None):
    j, key, Tmat, keys = load_joints(npz_path, joints_key)
    basic = stats_basic(j)
    root_idx, _ = choose_root_idx_auto(j)
    motion = temporal_motion_metrics(j)
    rc = root_center_quality(j, root_idx)

    bone = {"bone_mean_std": None, "bone_p95_std": None, "num_edges": 0}
    if kin_parent_path:
        kp = np.load(kin_parent_path)
        bone = bone_length_metrics(j, kp)

    metrics = {
        "path": npz_path,
        "key": key,
        "shape": j.shape,
        "has_T": Tmat is not None,
        "basic": basic,
        "root_idx_auto": root_idx,
        "motion": motion,
        "root_center": rc,
        "bone": bone
    }
    metrics["score"] = score_for_training(metrics)
    return metrics


def report(m, name="A"):
    print("\n" + "=" * 80)
    print(f"[{name}] {m['path']}")
    print(f"  joints key: {m['key']}")
    print(f"  shape    : {m['shape']}   (T={m['shape'][0]}, J={m['shape'][1]})")
    print(f"  has transform matrix: {m['has_T']}")
    b = m["basic"]
    print(f"  NaN/Inf  : {b['nan']}/{b['inf']}   zero_frames: {b['zero_frames']}")
    print(f"  min xyz  : {pretty_vec(b['min'])}")
    print(f"  max xyz  : {pretty_vec(b['max'])}")
    print(f"  mean xyz : {pretty_vec(b['mean'])}")
    print(f"  std xyz  : {pretty_vec(b['std'])}")

    print(f"  auto root_idx (most stable): {m['root_idx_auto']}")
    rc = m["root_center"]
    print(f"  root-centered mean xyz: {pretty_vec(rc['rc_mean'])}  (should be near 0)")
    print(f"  root-centered std  xyz: {pretty_vec(rc['rc_std'])}")

    mo = m["motion"]
    print(f"  vel_mean / vel_p95: {mo['vel_mean']:.6f} / {mo['vel_p95']:.6f}")
    print(f"  acc_mean / acc_p95: {mo['acc_mean']:.6f} / {mo['acc_p95']:.6f}")

    bo = m["bone"]
    if bo["bone_mean_std"] is not None:
        print(f"  bones(E={bo['num_edges']}): mean_std={bo['bone_mean_std']:.6f}  p95_std={bo['bone_p95_std']:.6f}")
    else:
        print("  bones: (skipped, no kin_parent provided)")

    print(f"  TRAIN SCORE (higher better): {m['score']:.2f}")


def main():
    ap = argparse.ArgumentParser("Compare two joints npz files for SkateFormer training quality")
    ap.add_argument("--a", required=True, help="npz file A")
    ap.add_argument("--b", required=True, help="npz file B")
    ap.add_argument("--kin_parent", default=None, help="kin_parent.npy (recommended)")

    ap.add_argument("--key_a", default=None, help="joints key override for file A")
    ap.add_argument("--key_b", default=None, help="joints key override for file B")

    args = ap.parse_args()

    ma = analyze(args.a, args.kin_parent, args.key_a)
    mb = analyze(args.b, args.kin_parent, args.key_b)

    report(ma, "A")
    report(mb, "B")

    print("\n" + "-" * 80)
    if ma["score"] > mb["score"]:
        print("✅ Recommendation: Use file A (better for skeleton training overall).")
    elif mb["score"] > ma["score"]:
        print("✅ Recommendation: Use file B (better for skeleton training overall).")
    else:
        print("⚠️ Both look similar by metrics. Either is fine; choose the one with correct coordinate convention.")


if __name__ == "__main__":
    main()
