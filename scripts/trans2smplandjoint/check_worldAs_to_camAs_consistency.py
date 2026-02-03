#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, numpy as np
import torch, smplx
from scipy.spatial.transform import Rotation as Rot

def normalize_T4(T):
    T = np.array(T, dtype=np.float32)
    if T.shape != (4,4):
        raise ValueError(f"Expect 4x4, got {T.shape}")
    # if translation appears in last row -> transpose
    if abs(T[3,3]-1.0)<1e-4 and np.linalg.norm(T[:3,3])<1e-4 and np.linalg.norm(T[3,:3])>1e-4:
        T = T.T
    return T.astype(np.float32)

def load_w2c(camera_json):
    cam = json.load(open(camera_json, "r"))
    # Prefer c2w (more explicit), invert it to w2c
    if "c2w" in cam:
        C = cam["c2w"][0] if isinstance(cam["c2w"], list) else cam["c2w"]
        C = normalize_T4(C)
        return np.linalg.inv(C).astype(np.float32)

    # Otherwise try world_view_transform as raw w2c-like
    if "world_view_transform" in cam:
        M = cam["world_view_transform"][0] if isinstance(cam["world_view_transform"], list) else cam["world_view_transform"]
        return normalize_T4(M)

    for k in ["w2c","extrinsic","world_to_camera"]:
        if k in cam:
            M = cam[k][0] if isinstance(cam[k], list) else cam[k]
            return normalize_T4(M)

    raise KeyError(f"no usable extrinsic key in {camera_json}. keys={list(cam.keys())}")

def apply_T(joints, T4):
    # joints: (T,J,3)
    Tn,J,_ = joints.shape
    ones = np.ones((Tn,J,1), np.float32)
    jh = np.concatenate([joints, ones], -1)            # (T,J,4)
    out = (T4[None,None,:,:] @ jh[...,None]).squeeze(-1)
    return out[...,:3].astype(np.float32)

def make_betas_batch(betas, T, num_betas):
    if betas is None:
        betas=np.zeros((num_betas,), np.float32)
    betas=betas.astype(np.float32)
    if betas.ndim==1:
        b=betas[:num_betas]
        return np.repeat(b[None,:], T, 0)
    b=betas[:T,:num_betas]
    if b.shape[0]!=T:
        b=np.repeat(b[:1], T, 0)
    return b

def split_pose_smplx(poses_t):
    T,D=poses_t.shape
    if D==156:
        pad=torch.zeros((T,9), device=poses_t.device, dtype=poses_t.dtype)
        poses_t=torch.cat([poses_t,pad],1)
        D=165
    if D!=165:
        raise ValueError(f"need 165 or 156->pad, got {D}")
    return dict(
        global_orient=poses_t[:,0:3],
        body_pose=poses_t[:,3:66],
        left_hand_pose=poses_t[:,66:111],
        right_hand_pose=poses_t[:,111:156],
        jaw_pose=poses_t[:,156:159],
        leye_pose=poses_t[:,159:162],
        reye_pose=poses_t[:,162:165],
    )

def forward_smplx_joints(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
    d=np.load(npz_path, allow_pickle=True)
    poses=d["poses"].astype(np.float32)
    trans=d["trans"].astype(np.float32)
    betas=d["betas"].astype(np.float32) if "betas" in d.files else None
    T=poses.shape[0]

    poses_t=torch.from_numpy(poses).to(device)
    trans_t=torch.from_numpy(trans).to(device)
    betas_t=torch.from_numpy(make_betas_batch(betas,T,num_betas)).to(device)

    kwargs=split_pose_smplx(poses_t)
    kwargs["transl"]=trans_t
    kwargs["betas"]=betas_t

    model=smplx.create(
        model_path=model_path, model_type="smplx", gender=gender,
        num_betas=num_betas, use_pca=False, batch_size=T
    ).to(device)
    model.eval()
    with torch.no_grad():
        out=model(**kwargs, return_verts=False)
        joints=out.joints.detach().cpu().numpy().astype(np.float32)  # (T,127,3)
    return joints

def stats(a,b):
    l2=np.linalg.norm(a-b, axis=-1)  # (T,J)
    return float(l2.mean()), float(np.percentile(l2,99)), float(l2.max())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--world_as_npz", required=True)
    ap.add_argument("--cam_as_npz", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--model_path", required=True)  # should be .../smpl_files/smplx
    ap.add_argument("--gender", default="neutral", choices=["male","female","neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    W2C = load_w2c(args.camera_json)

    jw = forward_smplx_joints(args.world_as_npz, args.model_path, args.gender, args.num_betas, args.device)
    jc = forward_smplx_joints(args.cam_as_npz,   args.model_path, args.gender, args.num_betas, args.device)
    Tmin=min(jw.shape[0], jc.shape[0])
    jw=jw[:Tmin]; jc=jc[:Tmin]

    jw_to_cam = apply_T(jw, W2C)
    mean,p99,mx = stats(jw_to_cam, jc)

    print("\n==== self-consistency: W2C * joints(world_As) vs joints(cam_As) ====")
    print("mean=", mean, "p99=", p99, "max=", mx)
    print("If this is ~1e-6~1e-4, exporter camera application is consistent.")
    print("If this is big (cm/m), then exporter uses different convention than this W2C.")

if __name__=="__main__":
    main()
