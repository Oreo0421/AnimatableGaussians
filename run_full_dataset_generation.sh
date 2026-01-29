#!/usr/bin/env bash
set -euo pipefail

# =========================
# Usage:
#   ./run_full_dataset_generation.sh 101223
# Optional env overrides:
#   POSE_DIR=... CONFIG_SRC=... PCA_DIR=... FRAME_START=0 FRAME_END=99 ./run_full_dataset_generation.sh 101223
# =========================

SUBJECT="${1:-}"
if [[ -z "${SUBJECT}" ]]; then
  echo "Usage: $0 <SUBJECT_ID e.g. 101223>"
  exit 1
fi

# ---------- Fixed paths (edit if needed) ----------
ANIM_ROOT="${ANIM_ROOT:-/home/fzhi/fzt/3dgs_pipeline/animatable_3DGS/AnimatableGaussians}"
HUGS_ROOT="${HUGS_ROOT:-/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs}"

POSE_DIR="${POSE_DIR:-/home/fzhi/fzt/3dgs_pipeline/animatable_3DGS/AnimatableGaussians/Data/Pose/AMASS/CMU/test}"

# default config path rule: configs/mvhn_xxxx/avatar.yaml
CONFIG_SRC="${CONFIG_SRC:-${ANIM_ROOT}/configs/mvhn_${SUBJECT}/avatar.yaml}"

HUMAN_DATA_ROOT="${HUMAN_DATA_ROOT:-/mnt/data_hdd/fzhi/human_data}"
MID_ROOT="${MID_ROOT:-/mnt/data_hdd/fzhi/mid}"
OUT_ROOT="${OUT_ROOT:-/mnt/data_hdd/fzhi/output}"

SCENE_PT_ROOT="${SCENE_PT_ROOT:-/home/fzhi/fzt/3dgs_pipeline/animatable_dataset/scenefixed}"
CAMERA_ROOT="${CAMERA_ROOT:-${HUGS_ROOT}/camera}"

# PCA output folder name (matches your example)
PCA_DIR="${PCA_DIR:-pca_20_sigma_2.00}"

FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-99}"

RENDER_MODE="${RENDER_MODE:-human_scene}"

# ---------- Concurrency knobs (keep 1 for safety) ----------
# If你GPU够强，可以自己改大；但大量渲染建议先保证稳定再并行
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# ---------- Helpers ----------
log() { echo -e "[`date '+%F %T'`] $*"; }

require_path() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "ERROR: path not found: $p"
    exit 1
  fi
}
# ---- conda safe wrappers: avoid "unbound variable" from activate scripts under set -u ----
conda_activate_safe() {
  local env="$1"
  set +u
  conda activate "$env"
  set -u
}

conda_deactivate_safe() {
  set +u
  conda deactivate
  set -u
}

# Create a temp avatar yaml for a given pose (no PyYAML dependency; pure text-state machine)
make_temp_avatar_yaml() {
  local src="$1"
  local dst="$2"
  local pose_npz="$3"
  local out_dir="$4"

  python3 - "$src" "$dst" "$pose_npz" "$out_dir" <<'PY'
import sys, os

src, dst, pose_npz, out_dir = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_test = False
in_pose_data = False

def indent_level(s: str) -> int:
    return len(s) - len(s.lstrip(' '))

out = []
for line in lines:
    raw = line.rstrip("\n")
    # enter/exit top-level blocks
    if raw.startswith("test:") and indent_level(raw) == 0:
        in_test = True
        in_pose_data = False
        out.append(line); continue
    if indent_level(raw) == 0 and raw and not raw.startswith("test:"):
        # leaving test block
        in_test = False
        in_pose_data = False

    # pose_data block inside test
    if in_test and raw.strip() == "pose_data:":
        in_pose_data = True
        out.append(line); continue
    if in_pose_data:
        # leaving pose_data: when indentation <= 2 and line is a key (non-empty)
        if raw.strip() and indent_level(raw) <= 2 and raw.strip() != "pose_data:":
            in_pose_data = False

    # replace pose_data.data_path
    if in_test and in_pose_data:
        if raw.lstrip().startswith("data_path:") and not raw.lstrip().startswith("#"):
            # keep original indentation
            prefix = raw.split("data_path:")[0] + "data_path:"
            out.append(f"{prefix} {pose_npz}\n")
            continue

    # replace test.output_dir
    if in_test and raw.lstrip().startswith("output_dir:") and indent_level(raw) == 2:
        prefix = raw.split("output_dir:")[0] + "output_dir:"
        out.append(f"{prefix} {out_dir}\n")
        continue

    out.append(line)

os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(out)
print(dst)
PY
}

# For a camera json that may contain multiple cameras (arrays),
# slice it into per-camera temp json so your "first camera only" renderer can render ALL cameras. :contentReference[oaicite:3]{index=3}
render_all_cameras_in_json() {
  local human_pt_dir="$1"
  local scene_pt="$2"
  local camera_json="$3"
  local output_base="$4"

  require_path "$human_pt_dir"
  require_path "$scene_pt"
  require_path "$camera_json"

  local ncam
  ncam="$(python3 - "$camera_json" <<'PY'
import json, sys
p=sys.argv[1]
j=json.load(open(p,'r'))
# try common key
k='fovx'
v=j.get(k, [])
print(len(v) if isinstance(v, list) else 1)
PY
)"

  if [[ "$ncam" == "0" ]]; then
    log "WARN: camera_json has 0 cameras? skip: $camera_json"
    return 0
  fi

  # If only 1 camera, render directly into output_base
  if [[ "$ncam" == "1" ]]; then
    mkdir -p "$output_base"
    python "${HUGS_ROOT}/hugs/renderer/render_sequence_from1camera.py" \
      --human_pt_dir "$human_pt_dir" \
      --scene_pt "$scene_pt" \
      --output_dir "$output_base" \
      --start_frame "$FRAME_START" \
      --end_frame "$FRAME_END" \
      --camera_json "$camera_json" \
      --render_mode "$RENDER_MODE"
    return 0
  fi

  # Multiple cameras: slice each camera to a temp json, render to output_base/cam_XX
  for ((i=0; i<ncam; i++)); do
    local cam_out="${output_base}/cam_$(printf '%02d' "$i")"
    mkdir -p "$cam_out"

    local tmp_json
    tmp_json="$(mktemp --suffix=".json")"

    python3 - "$camera_json" "$tmp_json" "$i" <<'PY'
import json, sys
src, dst, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
j = json.load(open(src,'r'))

def slice_value(v):
    # if list and long enough -> keep single element list
    if isinstance(v, list):
        if len(v) > idx:
            return [v[idx]]
        return v
    return v

out = {}
for k, v in j.items():
    out[k] = slice_value(v)

json.dump(out, open(dst,'w'), indent=2)
print(dst)
PY

    python "${HUGS_ROOT}/hugs/renderer/render_sequence_from1camera.py" \
      --human_pt_dir "$human_pt_dir" \
      --scene_pt "$scene_pt" \
      --output_dir "$cam_out" \
      --start_frame "$FRAME_START" \
      --end_frame "$FRAME_END" \
      --camera_json "$tmp_json" \
      --render_mode "$RENDER_MODE"

    rm -f "$tmp_json"
  done
}

# ---------- Preconditions ----------
require_path "$ANIM_ROOT"
require_path "$HUGS_ROOT"
require_path "$POSE_DIR"
require_path "$CONFIG_SRC"

# conda init
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- Enumerate motions ----------
shopt -s nullglob
mapfile -t POSE_FILES < <(ls -1 "${POSE_DIR}"/*.npz 2>/dev/null | sort)
if [[ "${#POSE_FILES[@]}" -eq 0 ]]; then
  echo "No .npz found in: ${POSE_DIR}"
  exit 1
fi

TMP_CFG_DIR="/tmp/avatar_cfg_${SUBJECT}"
mkdir -p "$TMP_CFG_DIR"

log "Found ${#POSE_FILES[@]} motions in ${POSE_DIR}"
log "SUBJECT=${SUBJECT}"
log "CONFIG_SRC=${CONFIG_SRC}"

# ---------- Stage A: AnimatableGaussians (anim3dgs) ----------
conda_activate_safe anim3dgs
log "Activated conda env: anim3dgs"

cd "$ANIM_ROOT"

POSE_NAMES=()

for pose_npz in "${POSE_FILES[@]}"; do
  pose_name="$(basename "$pose_npz" .npz)"
  POSE_NAMES+=("$pose_name")

  out_dir="${HUMAN_DATA_ROOT}/${SUBJECT}/${pose_name}/test_results"
  tmp_yaml="${TMP_CFG_DIR}/avatar_${pose_name}.yaml"

  # Skip if already exists (posed_gaussians has something)
  if [[ -d "${out_dir}/${PCA_DIR}/posed_gaussians" ]] && compgen -G "${out_dir}/${PCA_DIR}/posed_gaussians/*" > /dev/null; then
    log "[SKIP anim] ${pose_name} -> ${out_dir} (already has posed_gaussians)"
    continue
  fi

  log "[anim] pose=${pose_name}"
  make_temp_avatar_yaml "$CONFIG_SRC" "$tmp_yaml" "$pose_npz" "$out_dir" >/dev/null

  # main run
  python main_avatar_joint.py -c "$tmp_yaml"
done

conda_deactivate_safe
log "Deactivated conda env: anim3dgs"

# ---------- Stage B/C/D: HUGS (transform, render, joints) ----------
conda_activate_safe
log "Activated conda env: hugs"

cd "$HUGS_ROOT"

# Find all transform scripts under human_trans
mapfile -t TRANSFORM_SCRIPTS < <(find scripts/human_trans -type f -name "transform_human_sequence.py" | sort)
if [[ "${#TRANSFORM_SCRIPTS[@]}" -eq 0 ]]; then
  echo "ERROR: No transform_human_sequence.py found under ${HUGS_ROOT}/scripts/human_trans"
  exit 1
fi
log "Found ${#TRANSFORM_SCRIPTS[@]} human_trans scripts"

for pose_name in "${POSE_NAMES[@]}"; do
  anim_out_dir="${HUMAN_DATA_ROOT}/${SUBJECT}/${pose_name}/test_results"
  posed_gaussians="${anim_out_dir}/${PCA_DIR}/posed_gaussians"
  joints_dir="${anim_out_dir}/${PCA_DIR}/joints"

  if [[ ! -d "$posed_gaussians" ]]; then
    log "WARN: posed_gaussians not found, skip pose: ${pose_name} -> ${posed_gaussians}"
    continue
  fi

  log "=============================="
  log "[pose] ${pose_name}"
  log "posed_gaussians=${posed_gaussians}"
  log "joints_dir=${joints_dir}"
  log "=============================="

  for trans_script in "${TRANSFORM_SCRIPTS[@]}"; do
    # Expected: scripts/human_trans/<scene>/<pos>/transform_human_sequence.py
    rel="${trans_script#scripts/human_trans/}"
    scene="$(echo "$rel" | cut -d'/' -f1)"
    pos="$(echo "$rel" | cut -d'/' -f2)"
    scene_pos_key="${scene}_${pos}"

    mid_out="${MID_ROOT}/${SUBJECT}/${pose_name}/${scene}/${pos}"
    out_scene_pos="${OUT_ROOT}/${SUBJECT}/${pose_name}/${scene}/${pos}"

    # 1) transform human sequence (ply -> pt/both)
    if [[ -f "${mid_out}/pt/00000000.pt" ]]; then
      log "[SKIP trans] ${scene}/${pos} already has pt/00000000.pt"
    else
      log "[trans] ${scene}/${pos} -> ${mid_out}"
      python "${trans_script}" \
        --input_dir "${posed_gaussians}/" \
        --output_dir "${mid_out}" \
        --output_format both \
        --start_frame "${FRAME_START}" \
        --end_frame "${FRAME_END}"
    fi

    # 2) render all scenes + all cameras
    scene_pt="${SCENE_PT_ROOT}/${scene}/${scene}.pt"
    cam_dir="${CAMERA_ROOT}/${scene}/${pos}"

    if [[ ! -f "$scene_pt" ]]; then
      log "WARN: scene_pt not found, skip render: ${scene_pt}"
    elif [[ ! -d "$cam_dir" ]]; then
      log "WARN: camera dir not found, skip render: ${cam_dir}"
    else
      human_pt_dir="${mid_out}/pt"
      if [[ ! -d "$human_pt_dir" ]]; then
        log "WARN: human pt dir missing, skip render: ${human_pt_dir}"
      else
        shopt -s nullglob
        cam_jsons=( "${cam_dir}"/*.json )
        if [[ "${#cam_jsons[@]}" -eq 0 ]]; then
          log "WARN: no camera json under ${cam_dir}"
        else
          for cam_json in "${cam_jsons[@]}"; do
            cam_base="$(basename "$cam_json" .json)"
            img_out="${out_scene_pos}/image/${cam_base}"

            # skip if already rendered at least one frame
            if compgen -G "${img_out}/frame_*.png" > /dev/null || compgen -G "${img_out}/cam_*/frame_*.png" > /dev/null; then
              log "[SKIP render] ${scene}/${pos}/${cam_base} already has images"
              continue
            fi

            log "[render] ${scene}/${pos}/${cam_base} -> ${img_out}"
            render_all_cameras_in_json "$human_pt_dir" "$scene_pt" "$cam_json" "$img_out"
          done
        fi
      fi
    fi

    # 3) save joints with scene transform
    if [[ ! -d "$joints_dir" ]]; then
      log "WARN: joints dir not found, skip joint_trans: ${joints_dir}"
    else
      # if joint npz already exists, skip
      if compgen -G "${out_scene_pos}/joint/npz/${SUBJECT}_${pose_name}_${scene_pos_key}_all_*.npz" > /dev/null; then
        log "[SKIP joints] ${scene}/${pos} already has joint npz"
      else
        log "[joints] ${scene}/${pos} -> ${out_scene_pos}"
        python scripts/joint_trans/save_joints_trans_sequence_v2.py \
          --input_dir "$joints_dir" \
          --output_dir "$out_scene_pos" \
          --scene "$scene_pos_key" \
          --pose_name "$pose_name" \
          --subject_name "$SUBJECT"
      fi
    fi

  done
done

conda_deactivate_safe
log "Deactivated conda env: hugs"

log "ALL DONE."
