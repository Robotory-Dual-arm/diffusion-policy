#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

RAW_COMMON="${RAW_COMMON:-/home/baetae/Downloads/common_data_height.hdf5}"
REFERENCE_DIFFUSION="${REFERENCE_DIFFUSION:-data/baetae/260602/diffusion_data.hdf5}"

CORRECTED_DATASET="${CORRECTED_DATASET:-data/outputs/residual_policy/data/original/common_data_height_euler_zyx_residual.hdf5}"
ACTIVE_DATASET="${ACTIVE_DATASET:-data/outputs/residual_policy/data/fast/actual_base_residual.hdf5}"
VALIDATION_LOG="${VALIDATION_LOG:-data/outputs/residual_policy/data/original/common_data_height_euler_zyx_validation.log}"

ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/residual_policy/fast/corrected_actual_base_${RUN_STAMP}}"

CLEAN_BAD="${CLEAN_BAD:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-1}"
RUN_FAST4="${RUN_FAST4:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

VIRTUAL_ROTATION_FORMAT="${VIRTUAL_ROTATION_FORMAT:-euler_ZYX_deg}"
VIRTUAL_POSITION_SCALE="${VIRTUAL_POSITION_SCALE:-0.001}"
ROBOT_DOWNSAMPLE="${ROBOT_DOWNSAMPLE:-2}"
CONVERT_MAX_DEMOS="${CONVERT_MAX_DEMOS:-}"
MAX_RESIDUAL_ROTATION_DEG="${MAX_RESIDUAL_ROTATION_DEG:-30.0}"
MAX_REFERENCE_ROTATION_DEG="${MAX_REFERENCE_ROTATION_DEG:-1.0}"

DEVICE="${DEVICE:-cuda:0}"
VIS_DEVICE="${VIS_DEVICE:-$DEVICE}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-corrected_residual_fast4_${RUN_STAMP}}"

NUM_DEMOS="${NUM_DEMOS:-3}"
WINDOW_START="${WINDOW_START:-0}"
WINDOW_STEP="${WINDOW_STEP:-20}"
WINDOW_COUNT="${WINDOW_COUNT:-6}"
CHUNK_START="${CHUNK_START:-0}"
CHUNK_TOTAL_STEPS="${CHUNK_TOTAL_STEPS:--1}"
CHUNK_EXEC_STEPS="${CHUNK_EXEC_STEPS:-8}"

FORCE_SLOW_CKPT="${FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/force/slow_force.ckpt}"
NO_FORCE_SLOW_CKPT="${NO_FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt}"

echo "Raw common:          $RAW_COMMON"
echo "Reference diffusion: $REFERENCE_DIFFUSION"
echo "Corrected dataset:   $CORRECTED_DATASET"
echo "Active dataset:      $ACTIVE_DATASET"
echo "Root output:         $ROOT_OUTPUT"
echo "Rotation format:     $VIRTUAL_ROTATION_FORMAT"
echo "Run convert:         $RUN_CONVERT"
echo "Run fast4:           $RUN_FAST4"
echo "Run train:           $RUN_TRAIN"

if [[ ! -s "$RAW_COMMON" ]]; then
  echo "Missing raw common dataset: $RAW_COMMON" >&2
  exit 1
fi
if [[ ! -s "$FORCE_SLOW_CKPT" ]]; then
  echo "Missing force slow checkpoint: $FORCE_SLOW_CKPT" >&2
  exit 1
fi
if [[ ! -s "$NO_FORCE_SLOW_CKPT" ]]; then
  echo "Missing no-force slow checkpoint: $NO_FORCE_SLOW_CKPT" >&2
  exit 1
fi

if [[ "$CLEAN_BAD" == "1" ]]; then
  echo
  echo "========== CLEAN BAD RESIDUAL OUTPUTS =========="
  shopt -s nullglob
  BAD_PATHS=(
    data/outputs/residual_policy/data/fast/actual_base_residual.hdf5
    data/outputs/residual_policy/data/original/slow_erase_board_virtual_m.hdf5
    data/outputs/residual_policy/data/fast/slow_pred_base/force_pred_base_residual.hdf5
    data/outputs/residual_policy/data/fast/slow_pred_base/no_force_pred_base_residual.hdf5
    data/outputs/residual_policy/fast/compare_actual_vs_pred_base_summary.md
    data/baetae/260618/slow_erase_board_virtual_m.hdf5
  )
  BAD_GLOBS=(
    data/outputs/residual_policy/fast/20260625_fast
    data/outputs/residual_policy/fast/train4_20260625_*
    data/outputs/residual_policy/fast/pred_base_20260625_*
  )
  for path in "${BAD_PATHS[@]}" "${BAD_GLOBS[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
      echo "remove $path"
      rm -rf "$path"
    fi
  done
  shopt -u nullglob
fi

mkdir -p "$(dirname "$CORRECTED_DATASET")" "$(dirname "$ACTIVE_DATASET")" "$(dirname "$VALIDATION_LOG")"

if [[ "$RUN_CONVERT" == "1" ]]; then
  echo
  echo "========== CONVERT COMMON DATA =========="
  CONVERT_ARGS=()
  if [[ "$OVERWRITE_DATA" == "1" ]]; then
    CONVERT_ARGS+=(--overwrite)
  fi
  if [[ -n "$CONVERT_MAX_DEMOS" ]]; then
    CONVERT_ARGS+=(--max-demos "$CONVERT_MAX_DEMOS")
  fi
  python diffusion_policy/residual_policy/convert_common_to_slow_dataset.py \
    --input "$RAW_COMMON" \
    --output "$CORRECTED_DATASET" \
    --virtual-key desired_pose \
    --virtual-position-scale "$VIRTUAL_POSITION_SCALE" \
    --virtual-rotation-format "$VIRTUAL_ROTATION_FORMAT" \
    --robot-downsample "$ROBOT_DOWNSAMPLE" \
    "${CONVERT_ARGS[@]}"
else
  echo
  echo "========== SKIP CONVERT =========="
fi

if [[ ! -s "$CORRECTED_DATASET" ]]; then
  echo "Missing corrected dataset after conversion: $CORRECTED_DATASET" >&2
  exit 1
fi

echo
echo "========== VALIDATE CORRECTED DATASET =========="
VALIDATE_CMD=(
  python diffusion_policy/residual_policy/validate_residual_dataset.py
  --dataset "$CORRECTED_DATASET"
  --max-residual-rotation-deg "$MAX_RESIDUAL_ROTATION_DEG"
)
if [[ -s "$REFERENCE_DIFFUSION" ]]; then
  VALIDATE_CMD+=(
    --reference-dataset "$REFERENCE_DIFFUSION"
    --max-reference-rotation-deg "$MAX_REFERENCE_ROTATION_DEG"
  )
else
  echo "Reference diffusion dataset not found; skipping reference action comparison: $REFERENCE_DIFFUSION"
fi
"${VALIDATE_CMD[@]}" 2>&1 | tee "$VALIDATION_LOG"

echo
echo "========== ACTIVATE CORRECTED DATASET =========="
rm -f "$ACTIVE_DATASET"
ln -s "$(realpath --relative-to="$(dirname "$ACTIVE_DATASET")" "$CORRECTED_DATASET")" "$ACTIVE_DATASET"
echo "active -> $(readlink -f "$ACTIVE_DATASET")"

echo
echo "========== TRAIN + VISUALIZE FAST4 =========="
if [[ "$RUN_FAST4" == "1" ]]; then
  DATASET="$ACTIVE_DATASET" \
  ROOT_OUTPUT="$ROOT_OUTPUT" \
  RUN_TRAIN="$RUN_TRAIN" \
  DEVICE="$DEVICE" \
  VIS_DEVICE="$VIS_DEVICE" \
  LOGGING_MODE="$LOGGING_MODE" \
  WANDB_GROUP="$WANDB_GROUP" \
  NUM_DEMOS="$NUM_DEMOS" \
  WINDOW_START="$WINDOW_START" \
  WINDOW_STEP="$WINDOW_STEP" \
  WINDOW_COUNT="$WINDOW_COUNT" \
  CHUNK_START="$CHUNK_START" \
  CHUNK_TOTAL_STEPS="$CHUNK_TOTAL_STEPS" \
  CHUNK_EXEC_STEPS="$CHUNK_EXEC_STEPS" \
  FORCE_SLOW_CKPT="$FORCE_SLOW_CKPT" \
  NO_FORCE_SLOW_CKPT="$NO_FORCE_SLOW_CKPT" \
  ./scripts/train_residual_fast4_and_visualize.sh
else
  echo "Skipping fast4 train/visualize because RUN_FAST4=$RUN_FAST4"
fi

echo
echo "Done."
echo "Corrected dataset: $CORRECTED_DATASET"
echo "Validation log:    $VALIDATION_LOG"
echo "Active dataset:    $ACTIVE_DATASET -> $(readlink -f "$ACTIVE_DATASET")"
echo "Training output:   $ROOT_OUTPUT"
echo "Vis summary:       $ROOT_OUTPUT/visualization_error_summary.md"
