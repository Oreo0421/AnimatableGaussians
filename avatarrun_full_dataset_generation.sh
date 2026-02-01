#!/usr/bin/env bash
# ==============================================================================
# run_full_dataset_generation.sh
#
# 目标：一个 json 就是一个相机（不做 cam_00/cam_01 切片）
#
# 流程：
#   A) anim3dgs: POSE_DIR/*.npz -> main_avatar_joint.py --mode=test
#   B) hugs: scripts/human_trans/**/transform_human_sequence.py
#   C) hugs: camera/<scene>/<pos>/*.json 逐个渲染到 image/<view_name>/
#   D) hugs: save_joints_trans_sequence_v2.py
#
# 用法：
#   chmod +x run_full_dataset_generation.sh
#   ./run_full_dataset_generation.sh 101223
#
# 可选覆盖：
#   MAX_PARALLEL=2 FRAME_START=0 FRAME_END=99 ./run_full_dataset_generation.sh 101223
# ==============================================================================

set -euo pipefail

# -----------------------
# Args
# -----------------------
SUBJECT="${1:-}"
if [[ -z "${SUBJECT}" ]]; then
  echo "Usage: $0 <SUBJECT_ID e.g. 101223>"
  exit 1
fi

# -----------------------
# Paths (可用环境变量覆盖)
# -----------------------
BASE_DIR="${BASE_DIR:-/home/fzhi/fzt/3dgs_pipeline}"

ANIM_ROOT="${ANIM_ROOT:-${BASE_DIR}/animatable_3DGS/AnimatableGaussians}"
HUGS_ROOT="${HUGS_ROOT:-${BASE_DIR}/ml_hug/ml-hugs}"

POSE_DIR="${POSE_DIR:-${ANIM_ROOT}/Data/Pose/AMASS/CMU/omilabpose}"

# subject config（作为模板，不会写回；脚本会生成临时 yaml）
CONFIG_SRC="${CONFIG_SRC:-${ANIM_ROOT}/configs/avatarrex_${SUBJECT}/avatar.yaml}"

# main_avatar_joint 输出目录名（必须匹配你的工程）
PCA_DIR="${PCA_DIR:-pca_20_sigma_2.00}"

# 输出根目录
HDD_BASE="${HDD_BASE:-/mnt/data_hdd/fzhi}"
HUMAN_DATA_ROOT="${HUMAN_DATA_ROOT:-${HDD_BASE}/human_data}"
MID_ROOT="${MID_ROOT:-${HDD_BASE}/mid}"
OUT_ROOT="${OUT_ROOT:-${HDD_BASE}/output}"

# scene pt 根目录：scenefixed/<scene>/<scene>.pt
SCENE_PT_ROOT="${SCENE_PT_ROOT:-${BASE_DIR}/animatable_dataset/scenefixed}"

# camera 根目录：ml-hugs/camera/<scene>/<pos>/*.json
CAMERA_ROOT="${CAMERA_ROOT:-${HUGS_ROOT}/camera}"

# hugs transform 脚本根目录：scripts/human_trans/**/transform_human_sequence.py
TRANS_SCRIPT_ROOT="${TRANS_SCRIPT_ROOT:-${HUGS_ROOT}/scripts/human_trans}"

# 帧范围（你需求固定 0-99）
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-99}"

RENDER_MODE="${RENDER_MODE:-human_scene}"

# 并发：只用于“同一 scene/pos 下多个 view(json) 并行渲染”
# A6400 建议 1~2
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# -----------------------
# Helpers
# -----------------------
log() { echo -e "[`date '+%F %T'`] $*"; }

require_path() {
  local p="${1:-}"
  if [[ -z "$p" ]]; then
    echo "ERROR: require_path called with empty arg" >&2
    exit 1
  fi
  if [[ ! -e "$p" ]]; then
    echo "ERROR: path not found: $p" >&2
    exit 1
  fi
}

# ---- conda init + safe wrappers (避免 set -u 下 activate.d 脚本 unbound variable) ----
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda_activate_safe() {
  local env="${1:-}"
  if [[ -z "$env" ]]; then
    echo "ERROR: conda_activate_safe requires env name" >&2
    exit 1
  fi
  set +u
  conda activate "$env"
  set -u
}

conda_deactivate_safe() {
  set +u
  conda deactivate || true
  set -u
}

# 简单并发：保持最多 MAX_PARALLEL 个后台任务；任何失败会让脚本直接退出（set -e）
BG_PIDS=()
run_bg_limited() {
  "$@" &
  BG_PIDS+=($!)
  while [[ ${#BG_PIDS[@]} -gt $MAX_PARALLEL ]]; do
    wait "${BG_PIDS[0]}"
    BG_PIDS=("${BG_PIDS[@]:1}")
  done
}
wait_all_bg() {
  local pid
  for pid in "${BG_PIDS[@]:-}"; do
    wait "$pid"
  done
  BG_PIDS=()
}

# ------------------------------------------------------------------------------
# 生成临时 avatar yaml：只做文本级替换，避免 PyYAML dump 破坏 anchor/alias/注释
# ------------------------------------------------------------------------------
make_temp_avatar_yaml() {
  local src="$1"
  local dst="$2"
  local pose_npz="$3"
  local out_dir="$4"
  local subject="$5"

  python3 - "$src" "$dst" "$pose_npz" "$out_dir" "$subject" <<'PY'
import re, sys, os
src, dst, pose_npz, out_dir, subject = sys.argv[1:6]
lines = open(src,'r',encoding='utf-8').read().splitlines(True)

top_key_re = re.compile(r'^[A-Za-z0-9_]+\s*:\s*$')

def is_comment_or_empty(s: str) -> bool:
    t = s.strip()
    return (not t) or t.startswith("#")

out = []
in_test = False
in_pose_data = False

for line in lines:
    raw = line.rstrip("\n")
    s = raw.strip()

    # 进入/离开 test（忽略注释/空行）
    if not is_comment_or_empty(raw):
        if raw.startswith("test:"):
            in_test = True
            in_pose_data = False
        elif top_key_re.match(raw) and not raw.startswith("test:"):
            in_test = False
            in_pose_data = False

    # pose_data block
    if in_test and raw.startswith("  pose_data:") and not is_comment_or_empty(raw):
        in_pose_data = True
        out.append(line)
        continue

    # pose_data 结束：遇到 test 下其他二级 key
    if in_pose_data and not is_comment_or_empty(raw):
        if raw.startswith("  ") and (not raw.startswith("    ")) and (not raw.startswith("  pose_data:")):
            in_pose_data = False

    # test.data.subject_name（如果存在）
    if in_test and raw.startswith("    subject_name:") and not is_comment_or_empty(raw):
        prefix = raw.split("subject_name:")[0] + "subject_name:"
        out.append(f"{prefix} {subject}\n")
        continue

    # pose_data.data_path
    if in_test and in_pose_data and raw.startswith("    data_path:") and not is_comment_or_empty(raw):
        prefix = raw.split("data_path:")[0] + "data_path:"
        out.append(f"{prefix} {pose_npz}\n")
        continue

    # test.output_dir
    if in_test and raw.startswith("  output_dir:") and not is_comment_or_empty(raw):
        prefix = raw.split("output_dir:")[0] + "output_dir:"
        out.append(f"{prefix} {out_dir}\n")
        continue

    out.append(line)

os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst,'w',encoding='utf-8').writelines(out)
print(dst)
PY
}

# ------------------------------------------------------------------------------
# 渲染：一个 json 就是一个相机（完全按你给的指令）
# 输出：.../image/<view_name>/
# ------------------------------------------------------------------------------
render_one_camera_json() {
  local human_pt_dir="$1"
  local scene_pt="$2"
  local cam_json="$3"
  local out_dir="$4"

  mkdir -p "$out_dir"
  python "${HUGS_ROOT}/hugs/renderer/render_sequence_from1camera.py" \
    --human_pt_dir "$human_pt_dir" \
    --scene_pt "$scene_pt" \
    --output_dir "$out_dir" \
    --start_frame "$(printf '%08d' "$FRAME_START")" \
    --end_frame   "$(printf '%08d' "$FRAME_END")" \
    --camera_json "$cam_json" \
    --render_mode "$RENDER_MODE"
}

# ==============================================================================
# Preconditions
# ==============================================================================
require_path "$ANIM_ROOT"
require_path "$HUGS_ROOT"
require_path "$POSE_DIR"
require_path "$CONFIG_SRC"
require_path "$SCENE_PT_ROOT"
require_path "$CAMERA_ROOT"
require_path "$TRANS_SCRIPT_ROOT"

# ==============================================================================
# Enumerate poses
# ==============================================================================
shopt -s nullglob
mapfile -t POSE_FILES < <(ls -1 "${POSE_DIR}"/*.npz 2>/dev/null | sort)
if [[ "${#POSE_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: No .npz found in: ${POSE_DIR}"
  exit 1
fi

log "Found ${#POSE_FILES[@]} motions in ${POSE_DIR}"
log "SUBJECT=${SUBJECT}"
log "CONFIG_SRC=${CONFIG_SRC}"
log "FRAME_START=${FRAME_START} FRAME_END=${FRAME_END} MAX_PARALLEL=${MAX_PARALLEL}"

TMP_CFG_DIR="/tmp/avatar_cfg_${SUBJECT}"
mkdir -p "$TMP_CFG_DIR"

POSE_NAMES=()
for f in "${POSE_FILES[@]}"; do
  POSE_NAMES+=( "$(basename "$f" .npz)" )
done

# ==============================================================================
# Stage A: AnimatableGaussians (anim3dgs)
# ==============================================================================
conda_activate_safe anim3dgs
log "Activated conda env: anim3dgs"

cd "$ANIM_ROOT"

for pose_npz in "${POSE_FILES[@]}"; do
  pose_name="$(basename "$pose_npz" .npz)"
  out_dir="${HUMAN_DATA_ROOT}/${SUBJECT}/${pose_name}/test_results"
  tmp_yaml="${TMP_CFG_DIR}/avatar_${pose_name}.yaml"

  # 断点续跑：如果 posed_gaussians 已存在且非空，跳过
  if [[ -d "${out_dir}/${PCA_DIR}/posed_gaussians" ]] && compgen -G "${out_dir}/${PCA_DIR}/posed_gaussians/*" > /dev/null; then
    log "[SKIP anim] ${pose_name} -> ${out_dir} (already has posed_gaussians)"
    continue
  fi

  log "[anim] pose=${pose_name}"
  make_temp_avatar_yaml "$CONFIG_SRC" "$tmp_yaml" "$pose_npz" "$out_dir" "$SUBJECT" >/dev/null

  # 关键：加 --mode=test
  python main_avatar_joint.py -c "$tmp_yaml" --mode=test
done

conda_deactivate_safe
log "Deactivated conda env: anim3dgs"

# ==============================================================================
# Stage B/C/D: HUGS (transform, render, joints)
# ==============================================================================
conda_activate_safe hugs
log "Activated conda env: hugs"

cd "$HUGS_ROOT"

# 找到所有 transform 脚本
mapfile -t TRANSFORM_SCRIPTS < <(find "$TRANS_SCRIPT_ROOT" -type f -name "transform_human_sequence.py" | sort)
if [[ "${#TRANSFORM_SCRIPTS[@]}" -eq 0 ]]; then
  echo "ERROR: No transform_human_sequence.py found under ${TRANS_SCRIPT_ROOT}"
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
    # 期望结构：scripts/human_trans/<scene>/<pos>/transform_human_sequence.py
    rel="${trans_script#${TRANS_SCRIPT_ROOT}/}"
    scene="$(echo "$rel" | cut -d'/' -f1)"
    pos="$(echo "$rel" | cut -d'/' -f2)"
    if [[ -z "$scene" || -z "$pos" ]]; then
      log "WARN: cannot parse scene/pos from ${trans_script}, skip"
      continue
    fi

    scene_key="${scene}_${pos}"

    mid_out="${MID_ROOT}/${SUBJECT}/${pose_name}/${scene}/${pos}"
    out_scene_pos="${OUT_ROOT}/${SUBJECT}/${pose_name}/${scene}/${pos}"

    # --------------------------
    # Step 2: transform human
    # --------------------------
    if [[ -f "${mid_out}/pt/00000000.pt" ]]; then
      log "[SKIP trans] ${scene}/${pos} already has pt/00000000.pt"
    else
      log "[trans] ${scene}/${pos} -> ${mid_out}"
      mkdir -p "$mid_out"
      python "${trans_script}" \
        --input_dir "${posed_gaussians}/" \
        --output_dir "${mid_out}" \
        --output_format both \
        --start_frame "$(printf '%08d' "$FRAME_START")" \
        --end_frame   "$(printf '%08d' "$FRAME_END")"
    fi

    # --------------------------
    # Step 3: render (一个 json 一个相机)
    # 输出：.../image/<view_name>/
    # --------------------------
    scene_pt="${SCENE_PT_ROOT}/${scene}/${scene}.pt"
    cam_dir="${CAMERA_ROOT}/${scene}/${pos}"
    human_pt_dir="${mid_out}/pt"

    if [[ ! -f "$scene_pt" ]]; then
      log "WARN: scene_pt not found, skip render: ${scene_pt}"
    elif [[ ! -d "$cam_dir" ]]; then
      log "WARN: camera dir not found, skip render: ${cam_dir}"
    elif [[ ! -d "$human_pt_dir" ]]; then
      log "WARN: human pt dir missing, skip render: ${human_pt_dir}"
    else
      shopt -s nullglob
      mapfile -t cam_jsons < <(ls -1 "${cam_dir}"/*.json 2>/dev/null | sort)
      if [[ "${#cam_jsons[@]}" -eq 0 ]]; then
        log "WARN: no camera json under ${cam_dir}"
      else
        log "[render] ${scene}/${pos} view_count=${#cam_jsons[@]} (parallel=${MAX_PARALLEL})"
        for cam_json in "${cam_jsons[@]}"; do
          view="$(basename "$cam_json" .json)"           # top / forward / backward / left / right ...
          img_out="${out_scene_pos}/image/${view}"

          # 断点续跑：如果已有图片，跳过
          if compgen -G "${img_out}/frame_*.png" > /dev/null; then
            log "[SKIP render] ${scene}/${pos}/${view} already has images"
            continue
          fi

          run_bg_limited render_one_camera_json "$human_pt_dir" "$scene_pt" "$cam_json" "$img_out"
        done
        wait_all_bg
      fi
    fi

    # --------------------------
    # Step 4: save joints trans
    # --------------------------
    if [[ ! -d "$joints_dir" ]]; then
      log "WARN: joints dir not found, skip joint_trans: ${joints_dir}"
    else
      # 断点续跑：如果 joint npz 已存在，跳过
      if compgen -G "${out_scene_pos}/joint/npz/${SUBJECT}_${pose_name}_${scene_key}_all_*.npz" > /dev/null; then
        log "[SKIP joints] ${scene}/${pos} already has joint npz"
      else
        log "[joints] ${scene}/${pos} -> ${out_scene_pos}"
        mkdir -p "$out_scene_pos"
        python scripts/joint_trans/save_joints_trans_sequence_v2.py \
          --input_dir "$joints_dir" \
          --output_dir "$out_scene_pos" \
          --scene "$scene_key" \
          --pose_name "$pose_name" \
          --subject_name "$SUBJECT"
      fi
    fi

  done
done

conda_deactivate_safe
log "Deactivated conda env: hugs"
log "ALL DONE."
