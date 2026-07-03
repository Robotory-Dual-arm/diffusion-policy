#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

COMMON_DATA_ROOT="${COMMON_DATA_ROOT:-data/baetae/260628_erase_board}"
SLOW_CKPT="${SLOW_CKPT:-data/outputs/2026.07.02_residual_policy/slow/no_force/epoch=0900-train_loss=0.000.ckpt}"
SLOW_TRAIN_DATASET="${SLOW_TRAIN_DATASET:-data/baetae/260628_erase_board/diffusion_data_erase_board_actual_action.hdf5}"
REFERENCE_VIRTUAL_DATASET="${REFERENCE_VIRTUAL_DATASET:-data/baetae/260628_erase_board/diffusion_data_erase_board_desired_action.hdf5}"

DATA_ROOT="${DATA_ROOT:-data/outputs/2026.07.02_residual_policy/data}"
ACTUAL_BASE_DATASET="${ACTUAL_BASE_DATASET:-$DATA_ROOT/actual_base_residual.hdf5}"
PRED_DATASET="${PRED_DATASET:-$DATA_ROOT/no_force_pred_base_residual.hdf5}"
VALIDATION_LOG="${VALIDATION_LOG:-$DATA_ROOT/actual_base_residual_validation.log}"
ACTION_COMPARE_LOG="${ACTION_COMPARE_LOG:-$DATA_ROOT/actual_action_compare.log}"
PRED_CHECK_LOG="${PRED_CHECK_LOG:-$DATA_ROOT/pred_base_check.log}"

ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/2026.07.02_residual_policy/fast/no_force_pred_base_${RUN_STAMP}}"
DEVICE="${DEVICE:-cuda:0}"
DATA_DEVICE="${DATA_DEVICE:-$DEVICE}"
VIS_DEVICE="${VIS_DEVICE:-$DEVICE}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-20260702_no_force_pred_base_fast2_${RUN_STAMP}}"

RUN_CONVERT="${RUN_CONVERT:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_CREATE_PRED="${RUN_CREATE_PRED:-1}"
OVERWRITE_PRED_DATA="${OVERWRITE_PRED_DATA:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_VIS="${RUN_VIS:-1}"

VIRTUAL_ROTATION_FORMAT="${VIRTUAL_ROTATION_FORMAT:-euler_ZYX_deg}"
ALLOW_ROTATION_OVERRIDE="${ALLOW_ROTATION_OVERRIDE:-0}"
VIRTUAL_POSITION_SCALE="${VIRTUAL_POSITION_SCALE:-0.001}"
ROBOT_DOWNSAMPLE="${ROBOT_DOWNSAMPLE:-2}"
CONVERT_MAX_DEMOS="${CONVERT_MAX_DEMOS:-}"

MAX_RESIDUAL_ROTATION_DEG="${MAX_RESIDUAL_ROTATION_DEG:-30.0}"
MAX_REFERENCE_ROTATION_DEG="${MAX_REFERENCE_ROTATION_DEG:-0.1}"
ACTION_COMPARE_MAX_POS_MM="${ACTION_COMPARE_MAX_POS_MM:-0.05}"
ACTION_COMPARE_MAX_ROT_DEG="${ACTION_COMPARE_MAX_ROT_DEG:-0.05}"

PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-16}"
PRED_NUM_INFERENCE_STEPS="${PRED_NUM_INFERENCE_STEPS:-16}"
PRED_DEMO_LIMIT="${PRED_DEMO_LIMIT:-}"
SLOW_ACTION_INDEX="${SLOW_ACTION_INDEX:-0}"
TARGET_SHIFT="${TARGET_SHIFT:-1}"

NUM_DEMOS="${NUM_DEMOS:-3}"
DEMOS="${DEMOS:-}"
WINDOW_START="${WINDOW_START:-0}"
WINDOW_STEP="${WINDOW_STEP:-20}"
WINDOW_COUNT="${WINDOW_COUNT:-6}"
WINDOW_STARTS="${WINDOW_STARTS:-}"
CHUNK_START="${CHUNK_START:-0}"
CHUNK_STARTS="${CHUNK_STARTS:-}"
CHUNK_TOTAL_STEPS="${CHUNK_TOTAL_STEPS:--1}"
CHUNK_EXEC_STEPS="${CHUNK_EXEC_STEPS:-8}"
SAVE_NPZ="${SAVE_NPZ:-0}"

CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

hydra_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "'%s'" "$value"
}

# Optional Hydra / visualizer overrides.
# Examples:
#   TRAIN_OVERRIDES='training.num_epochs=300 dataloader.num_workers=4'
#   VIS_OVERRIDES='--window-count 10 --batch-size 8'
# shellcheck disable=SC2206
EXTRA_TRAIN_OVERRIDES=(${TRAIN_OVERRIDES:-})
# shellcheck disable=SC2206
EXTRA_VIS_OVERRIDES=(${VIS_OVERRIDES:-})

if [[ "$VIRTUAL_ROTATION_FORMAT" != "euler_ZYX_deg" && "$ALLOW_ROTATION_OVERRIDE" != "1" ]]; then
  echo "Refusing VIRTUAL_ROTATION_FORMAT=$VIRTUAL_ROTATION_FORMAT." >&2
  echo "260628 erase-board desired_pose is xyz(mm) + Euler ZYX(deg)." >&2
  echo "Set ALLOW_ROTATION_OVERRIDE=1 only if you are intentionally testing another representation." >&2
  exit 1
fi

mapfile -t COMMON_DATA_FILES < <(find "$COMMON_DATA_ROOT" -mindepth 2 -maxdepth 2 -type f -name common_data.hdf5 | sort)
if [[ "${#COMMON_DATA_FILES[@]}" -ne 5 ]]; then
  echo "Expected 5 common_data.hdf5 files under $COMMON_DATA_ROOT, found ${#COMMON_DATA_FILES[@]}:" >&2
  printf '  %s\n' "${COMMON_DATA_FILES[@]}" >&2
  exit 1
fi
if [[ ! -s "$SLOW_CKPT" ]]; then
  echo "Missing slow checkpoint: $SLOW_CKPT" >&2
  exit 1
fi
if [[ ! -s "$SLOW_TRAIN_DATASET" ]]; then
  echo "Missing slow actual-action dataset: $SLOW_TRAIN_DATASET" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" "$ROOT_OUTPUT"

echo "========== 2026.07.02 no-force pred-base fast2 =========="
echo "Python command:        ${PYTHON_CMD[*]}"
echo "Common data files:"
printf '  %s\n' "${COMMON_DATA_FILES[@]}"
echo "Slow ckpt:             $SLOW_CKPT"
echo "Slow train dataset:    $SLOW_TRAIN_DATASET"
echo "Reference virtual:     $REFERENCE_VIRTUAL_DATASET"
echo "Actual-base dataset:   $ACTUAL_BASE_DATASET"
echo "Pred-base dataset:     $PRED_DATASET"
echo "Root output:           $ROOT_OUTPUT"
echo "Rotation format:       $VIRTUAL_ROTATION_FORMAT"
echo "Position scale:        $VIRTUAL_POSITION_SCALE"
echo "Device:                $DEVICE"
echo "Data device:           $DATA_DEVICE"
echo "Logging mode:          $LOGGING_MODE"

if [[ "$RUN_CONVERT" == "1" ]]; then
  echo
  echo "========== CONVERT COMMON -> ACTUAL-BASE RESIDUAL =========="
  if [[ -s "$ACTUAL_BASE_DATASET" && "$OVERWRITE_DATA" != "1" ]]; then
    echo "Keep existing actual-base dataset: $ACTUAL_BASE_DATASET"
  else
    CONVERT_ARGS=(
      --input "${COMMON_DATA_FILES[@]}"
      --output "$ACTUAL_BASE_DATASET"
      --virtual-key desired_pose
      --virtual-position-scale "$VIRTUAL_POSITION_SCALE"
      --virtual-rotation-format "$VIRTUAL_ROTATION_FORMAT"
      --robot-downsample "$ROBOT_DOWNSAMPLE"
    )
    if [[ "$OVERWRITE_DATA" == "1" ]]; then
      CONVERT_ARGS+=(--overwrite)
    fi
    if [[ -n "$CONVERT_MAX_DEMOS" ]]; then
      CONVERT_ARGS+=(--max-demos "$CONVERT_MAX_DEMOS")
    fi
    "${PYTHON_CMD[@]}" diffusion_policy/residual_policy/convert_common_to_slow_dataset.py "${CONVERT_ARGS[@]}"
  fi
else
  echo
  echo "========== SKIP CONVERT =========="
fi

if [[ ! -s "$ACTUAL_BASE_DATASET" ]]; then
  echo "Missing actual-base residual dataset: $ACTUAL_BASE_DATASET" >&2
  exit 1
fi

if [[ "$RUN_VALIDATE" == "1" ]]; then
  echo
  echo "========== VALIDATE ACTUAL-BASE RESIDUAL =========="
  VALIDATE_CMD=(
    "${PYTHON_CMD[@]}"
    diffusion_policy/residual_policy/validate_residual_dataset.py
    --dataset "$ACTUAL_BASE_DATASET"
    --max-residual-rotation-deg "$MAX_RESIDUAL_ROTATION_DEG"
  )
  if [[ -s "$REFERENCE_VIRTUAL_DATASET" ]]; then
    VALIDATE_CMD+=(
      --reference-dataset "$REFERENCE_VIRTUAL_DATASET"
      --max-reference-rotation-deg "$MAX_REFERENCE_ROTATION_DEG"
    )
  else
    echo "Reference virtual dataset not found; skipping virtual action comparison: $REFERENCE_VIRTUAL_DATASET"
  fi
  "${VALIDATE_CMD[@]}" 2>&1 | tee "$VALIDATION_LOG"

  echo
  echo "========== COMPARE ACTUAL ACTIONS WITH SLOW TRAIN DATA =========="
  "${PYTHON_CMD[@]}" - \
    "$ACTUAL_BASE_DATASET" \
    "$SLOW_TRAIN_DATASET" \
    "$ACTION_COMPARE_MAX_POS_MM" \
    "$ACTION_COMPARE_MAX_ROT_DEG" \
    <<'PY' 2>&1 | tee "$ACTION_COMPARE_LOG"
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_policy.pose_util import pose9_to_mat


actual_base_path, slow_path, max_pos_mm, max_rot_deg = sys.argv[1:5]
max_pos_mm = float(max_pos_mm)
max_rot_deg = float(max_rot_deg)


def sorted_demo_keys(group):
    return sorted(group.keys(), key=lambda name: int(name.split("_")[-1]))


def pose_error(a, b):
    mat_a = pose9_to_mat(a)
    mat_b = pose9_to_mat(b)
    pos_mm = np.linalg.norm(mat_a[..., :3, 3] - mat_b[..., :3, 3], axis=-1) * 1000.0
    rel = np.linalg.inv(mat_a) @ mat_b
    rot_deg = Rotation.from_matrix(rel[..., :3, :3]).magnitude() * 180.0 / np.pi
    return pos_mm, rot_deg


with h5py.File(actual_base_path, "r") as actual_file, h5py.File(slow_path, "r") as slow_file:
    actual_data = actual_file["data"]
    slow_data = slow_file["data"]
    actual_demos = sorted_demo_keys(actual_data)
    slow_demos = sorted_demo_keys(slow_data)
    if actual_demos != slow_demos:
        raise SystemExit(
            f"Demo key mismatch: actual_base={len(actual_demos)} slow={len(slow_demos)}"
        )

    pos_all = []
    rot_all = []
    frame_count = 0
    for demo_name in actual_demos:
        actual_action = np.asarray(actual_data[demo_name]["actions"])
        slow_action = np.asarray(slow_data[demo_name]["actions"])
        if actual_action.shape != slow_action.shape:
            raise SystemExit(
                f"{demo_name} action shape mismatch: {actual_action.shape} vs {slow_action.shape}"
            )
        pos_mm, rot_deg = pose_error(actual_action, slow_action)
        pos_all.append(pos_mm)
        rot_all.append(rot_deg)
        frame_count += len(actual_action)

    pos_all = np.concatenate(pos_all)
    rot_all = np.concatenate(rot_all)
    print(f"demos: {len(actual_demos)}")
    print(f"frames: {frame_count}")
    print(f"actual action pos error mm: mean={pos_all.mean():.9g} max={pos_all.max():.9g}")
    print(f"actual action rot error deg: mean={rot_all.mean():.9g} max={rot_all.max():.9g}")
    if pos_all.max() > max_pos_mm:
        raise SystemExit(
            f"Actual action position mismatch {pos_all.max():.9g} mm > {max_pos_mm} mm"
        )
    if rot_all.max() > max_rot_deg:
        raise SystemExit(
            f"Actual action rotation mismatch {rot_all.max():.9g} deg > {max_rot_deg} deg"
        )
PY
else
  echo
  echo "========== SKIP VALIDATE =========="
fi

if [[ "$RUN_CREATE_PRED" == "1" ]]; then
  echo
  echo "========== CREATE SLOW-PRED BASE DATASET =========="
  if [[ -s "$PRED_DATASET" && "$OVERWRITE_PRED_DATA" != "1" ]]; then
    echo "Keep existing pred-base dataset: $PRED_DATASET"
  else
    PRED_ARGS=(
      --input "$ACTUAL_BASE_DATASET"
      --output "$PRED_DATASET"
      --slow-ckpt "$SLOW_CKPT"
      --device "$DATA_DEVICE"
      --batch-size "$PRED_BATCH_SIZE"
      --target-shift "$TARGET_SHIFT"
      --slow-action-index "$SLOW_ACTION_INDEX"
      --num-inference-steps "$PRED_NUM_INFERENCE_STEPS"
      --full-action-steps
    )
    if [[ "$OVERWRITE_PRED_DATA" == "1" ]]; then
      PRED_ARGS+=(--overwrite)
    fi
    if [[ -n "$PRED_DEMO_LIMIT" ]]; then
      PRED_ARGS+=(--demo-limit "$PRED_DEMO_LIMIT")
    fi
    "${PYTHON_CMD[@]}" diffusion_policy/residual_policy/create_slow_pred_fast_dataset.py "${PRED_ARGS[@]}"
  fi
else
  echo
  echo "========== SKIP CREATE SLOW-PRED BASE DATASET =========="
fi

if [[ ! -s "$PRED_DATASET" ]]; then
  echo "Missing pred-base residual dataset: $PRED_DATASET" >&2
  exit 1
fi

echo
echo "========== CHECK SLOW-PRED BASE DATASET =========="
"${PYTHON_CMD[@]}" - "$PRED_DATASET" "$SLOW_CKPT" <<'PY' 2>&1 | tee "$PRED_CHECK_LOG"
import sys

import h5py
import numpy as np

dataset_path, slow_ckpt = sys.argv[1:3]
required = [
    "slow_pred_target_abs",
    "slow_pred_action_rel",
    "residual_delta6_slow_pred_to_virtual",
    "residual_delta6_slow_pred_to_actual",
]

with h5py.File(dataset_path, "r") as f:
    print("slow_pred_ckpt:", f.attrs.get("slow_pred_ckpt"))
    if str(f.attrs.get("slow_pred_ckpt")) != slow_ckpt:
        raise SystemExit("slow_pred_ckpt attr does not match requested slow checkpoint")
    data = f["data"]
    demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))
    if not demo_names:
        raise SystemExit("pred-base dataset has no demos")
    lengths = []
    residual_norms = []
    for demo_name in demo_names:
        obs = data[demo_name]["obs"]
        for key in required:
            if key not in obs:
                raise SystemExit(f"{demo_name} missing obs/{key}")
        length = len(data[demo_name]["actions"])
        lengths.append(length)
        residual = np.asarray(obs["residual_delta6_slow_pred_to_virtual"])
        residual_norms.append(np.linalg.norm(residual[:, :3], axis=-1))

    residual_norms = np.concatenate(residual_norms) * 1000.0
    print("demos:", len(demo_names))
    print("frames:", int(sum(lengths)))
    print("min/max frames per demo:", min(lengths), max(lengths))
    print(
        "slow_pred_to_virtual residual translation mm: "
        f"mean={residual_norms.mean():.9g} p99={np.percentile(residual_norms, 99):.9g} "
        f"max={residual_norms.max():.9g}"
    )
PY

declare -a MODELS=(mlp gru)
declare -A CKPT_BY_MODEL

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo
  echo "========== TRAIN FAST POLICIES =========="
else
  echo
  echo "========== SKIP TRAIN =========="
fi

for MODEL in "${MODELS[@]}"; do
  COMBO="no_force_pred_base_${MODEL}"
  RUN_DIR="$ROOT_OUTPUT/$COMBO"
  TRAIN_LOG="$RUN_DIR/train.log"
  mkdir -p "$RUN_DIR"

  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo
    echo "========== TRAIN $COMBO =========="
    echo "Run dir:   $RUN_DIR"
    echo "Slow ckpt: $SLOW_CKPT"
    echo "Dataset:   $PRED_DATASET"

    SLOW_CKPT_HYDRA="$(hydra_quote "$SLOW_CKPT")"
    HYDRA_FULL_ERROR=1 "${PYTHON_CMD[@]}" train.py \
      --config-name="residual_policy/$MODEL" \
      "residual_policy/task=no_force_pred_base" \
      "hydra.run.dir=$RUN_DIR" \
      "hydra.sweep.dir=$RUN_DIR" \
      "multi_run.run_dir=$RUN_DIR" \
      "task.dataset_path=$PRED_DATASET" \
      "task.slow_ckpt_path=$SLOW_CKPT_HYDRA" \
      "slow_ckpt_path=$SLOW_CKPT_HYDRA" \
      "policy.slow_ckpt_path=$SLOW_CKPT_HYDRA" \
      "training.device=$DEVICE" \
      "logging.mode=$LOGGING_MODE" \
      "logging.group=$WANDB_GROUP" \
      "logging.name=${RUN_STAMP}_${COMBO}" \
      "${EXTRA_TRAIN_OVERRIDES[@]}" \
      2>&1 | tee "$TRAIN_LOG"
  fi

  CKPT="$RUN_DIR/checkpoints/latest.ckpt"
  if [[ "$RUN_TRAIN" == "1" || "$RUN_VIS" == "1" ]]; then
    if [[ ! -s "$CKPT" ]]; then
      echo "Missing latest checkpoint: $CKPT" >&2
      exit 1
    fi
  fi
  CKPT_BY_MODEL["$MODEL"]="$CKPT"
done

if [[ "$RUN_VIS" == "1" ]]; then
  echo
  echo "========== VISUALIZE FAST POLICIES =========="
  for MODEL in "${MODELS[@]}"; do
    COMBO="no_force_pred_base_${MODEL}"
    RUN_DIR="$ROOT_OUTPUT/$COMBO"
    CKPT="${CKPT_BY_MODEL[$MODEL]}"
    VIS_DIR="$RUN_DIR/visualization_world"
    VIS_LOG="$RUN_DIR/visualization_world.log"
    mkdir -p "$VIS_DIR"

    VIS_CMD=(
      "${PYTHON_CMD[@]}"
      diffusion_policy/residual_policy/test/visualize_step_residual_predictions.py
      --dataset "$PRED_DATASET"
      --slow-ckpt "$SLOW_CKPT"
      --fast-ckpt "$CKPT"
      --output-dir "$VIS_DIR"
      --device "$VIS_DEVICE"
      --organized-output
      --world-frame
      --no-continuous-plots
      --window-plots
      --chunked-plots
      --chunk-total-steps "$CHUNK_TOTAL_STEPS"
      --chunk-exec-steps "$CHUNK_EXEC_STEPS"
    )

    if [[ -n "$DEMOS" ]]; then
      IFS=',' read -ra DEMO_LIST <<< "$DEMOS"
      for DEMO in "${DEMO_LIST[@]}"; do
        DEMO="${DEMO#"${DEMO%%[![:space:]]*}"}"
        DEMO="${DEMO%"${DEMO##*[![:space:]]}"}"
        if [[ -n "$DEMO" ]]; then
          VIS_CMD+=(--demo "$DEMO")
        fi
      done
    else
      VIS_CMD+=(--num-demos "$NUM_DEMOS")
    fi

    if [[ -n "$WINDOW_STARTS" ]]; then
      VIS_CMD+=(--window-starts "$WINDOW_STARTS")
    else
      VIS_CMD+=(--window-start "$WINDOW_START" --window-step "$WINDOW_STEP" --window-count "$WINDOW_COUNT")
    fi

    if [[ -n "$CHUNK_STARTS" ]]; then
      VIS_CMD+=(--chunk-starts "$CHUNK_STARTS")
    else
      VIS_CMD+=(--chunk-start "$CHUNK_START")
    fi

    if [[ "$SAVE_NPZ" == "1" ]]; then
      VIS_CMD+=(--save-npz)
    fi

    VIS_CMD+=("${EXTRA_VIS_OVERRIDES[@]}")

    echo
    echo "========== VIS $COMBO =========="
    echo "Fast ckpt: $CKPT"
    echo "Vis dir:   $VIS_DIR"
    "${VIS_CMD[@]}" 2>&1 | tee "$VIS_LOG"
  done

  "${PYTHON_CMD[@]}" - "$ROOT_OUTPUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []


def mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def add_row(combo, demo, view, case, metric):
    slow = metric.get("slow_vs_gt_actual", {})
    fast = metric.get("fast_vs_gt_virtual", {})
    slow_pos = slow.get("pos_mm_mean")
    fast_pos = fast.get("pos_mm_mean")
    slow_rot = slow.get("rot_deg_mean")
    fast_rot = fast.get("rot_deg_mean")
    rows.append({
        "combo": combo,
        "demo": demo,
        "view": view,
        "case": str(case),
        "slow_pos": slow_pos,
        "fast_pos": fast_pos,
        "pos_gain": None if slow_pos is None or fast_pos is None else slow_pos - fast_pos,
        "slow_rot": slow_rot,
        "fast_rot": fast_rot,
        "rot_gain": None if slow_rot is None or fast_rot is None else slow_rot - fast_rot,
    })


for metrics_path in sorted(root.glob("*/visualization_world/metrics.json")):
    combo = metrics_path.parent.parent.name
    data = json.loads(metrics_path.read_text())
    for demo, sections in data.items():
        for anchor, metric in sections.get("windows", {}).items():
            add_row(combo, demo, "window16", anchor, metric)
        for start, metric in sections.get("chunked", {}).items():
            add_row(combo, demo, f"chunked{metric.get('exec_steps', '')}", start, metric)

summary_json = root / "visualization_error_summary.json"
summary_md = root / "visualization_error_summary.md"
summary_json.write_text(json.dumps(rows, indent=2))

groups = defaultdict(list)
for row in rows:
    groups[(row["combo"], row["view"])].append(row)

lines = [
    "# 2026.07.02 No-Force Pred-Base Fast Summary",
    "",
    "| combo | view | n | slow pos mm | fast pos mm | pos gain mm | slow rot deg | fast rot deg | rot gain deg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for (combo, view), group_rows in sorted(groups.items()):
    vals = {
        "slow_pos": mean([r["slow_pos"] for r in group_rows]),
        "fast_pos": mean([r["fast_pos"] for r in group_rows]),
        "pos_gain": mean([r["pos_gain"] for r in group_rows]),
        "slow_rot": mean([r["slow_rot"] for r in group_rows]),
        "fast_rot": mean([r["fast_rot"] for r in group_rows]),
        "rot_gain": mean([r["rot_gain"] for r in group_rows]),
    }
    lines.append(
        f"| {combo} | {view} | {len(group_rows)} | "
        f"{fmt(vals['slow_pos'])} | {fmt(vals['fast_pos'])} | {fmt(vals['pos_gain'])} | "
        f"{fmt(vals['slow_rot'])} | {fmt(vals['fast_rot'])} | {fmt(vals['rot_gain'])} |"
    )

summary_md.write_text("\n".join(lines) + "\n")
print("Wrote", summary_json)
print("Wrote", summary_md)
PY
else
  echo
  echo "========== SKIP VISUALIZE =========="
fi

echo
echo "Done."
echo "Actual-base dataset: $ACTUAL_BASE_DATASET"
echo "Pred-base dataset:   $PRED_DATASET"
echo "Validation log:      $VALIDATION_LOG"
echo "Action compare log:  $ACTION_COMPARE_LOG"
echo "Pred check log:      $PRED_CHECK_LOG"
echo "Training output:     $ROOT_OUTPUT"
if [[ "$RUN_VIS" == "1" ]]; then
  echo "Vis summary:         $ROOT_OUTPUT/visualization_error_summary.md"
fi
