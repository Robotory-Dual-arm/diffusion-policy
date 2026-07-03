#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/residual_policy/fast/pred_base_${RUN_STAMP}}"

SOURCE_DATASET="${SOURCE_DATASET:-data/outputs/residual_policy/data/fast/actual_base_residual.hdf5}"
PRED_DATA_ROOT="${PRED_DATA_ROOT:-data/outputs/residual_policy/data/fast/slow_pred_base}"
FORCE_PRED_DATASET="${FORCE_PRED_DATASET:-$PRED_DATA_ROOT/force_pred_base_residual.hdf5}"
NO_FORCE_PRED_DATASET="${NO_FORCE_PRED_DATASET:-$PRED_DATA_ROOT/no_force_pred_base_residual.hdf5}"

FORCE_SLOW_CKPT="${FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/force/slow_force.ckpt}"
NO_FORCE_SLOW_CKPT="${NO_FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt}"

DEVICE="${DEVICE:-cuda:0}"
DATA_DEVICE="${DATA_DEVICE:-$DEVICE}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-residual_pred_base_fast4_${RUN_STAMP}}"

CREATE_DATA="${CREATE_DATA:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-16}"
PRED_NUM_INFERENCE_STEPS="${PRED_NUM_INFERENCE_STEPS:-16}"

# Optional extra Hydra overrides, e.g.
# TRAIN_OVERRIDES='training.num_epochs=300 dataloader.num_workers=4'
# shellcheck disable=SC2206
EXTRA_TRAIN_OVERRIDES=(${TRAIN_OVERRIDES:-})

mkdir -p "$ROOT_OUTPUT" "$PRED_DATA_ROOT"

echo "Root output:     $ROOT_OUTPUT"
echo "Source dataset:  $SOURCE_DATASET"
echo "Force dataset:   $FORCE_PRED_DATASET"
echo "No-force dataset:$NO_FORCE_PRED_DATASET"
echo "Device:          $DEVICE"
echo "Data device:     $DATA_DEVICE"
echo "Create data:     $CREATE_DATA"
echo "Run train:       $RUN_TRAIN"

make_pred_dataset() {
  local slow_name="$1"
  local slow_ckpt="$2"
  local output_dataset="$3"

  if [[ "$CREATE_DATA" != "1" ]]; then
    echo
    echo "========== SKIP DATA $slow_name =========="
    return
  fi

  if [[ -s "$output_dataset" && "$OVERWRITE_DATA" != "1" ]]; then
    echo
    echo "========== KEEP DATA $slow_name =========="
    echo "Dataset already exists: $output_dataset"
    return
  fi

  echo
  echo "========== CREATE DATA $slow_name =========="
  echo "Slow ckpt: $slow_ckpt"
  echo "Output:    $output_dataset"

  local overwrite_args=()
  if [[ "$OVERWRITE_DATA" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  python diffusion_policy/residual_policy/create_slow_pred_fast_dataset.py \
    --input "$SOURCE_DATASET" \
    --output "$output_dataset" \
    --slow-ckpt "$slow_ckpt" \
    --device "$DATA_DEVICE" \
    --batch-size "$PRED_BATCH_SIZE" \
    --target-shift 1 \
    --slow-action-index 0 \
    --num-inference-steps "$PRED_NUM_INFERENCE_STEPS" \
    --full-action-steps \
    "${overwrite_args[@]}"
}

make_pred_dataset "force_pred_base" "$FORCE_SLOW_CKPT" "$FORCE_PRED_DATASET"
make_pred_dataset "no_force_pred_base" "$NO_FORCE_SLOW_CKPT" "$NO_FORCE_PRED_DATASET"

declare -a COMBOS=(
  "force_pred_base mlp $FORCE_SLOW_CKPT $FORCE_PRED_DATASET"
  "force_pred_base gru $FORCE_SLOW_CKPT $FORCE_PRED_DATASET"
  "no_force_pred_base mlp $NO_FORCE_SLOW_CKPT $NO_FORCE_PRED_DATASET"
  "no_force_pred_base gru $NO_FORCE_SLOW_CKPT $NO_FORCE_PRED_DATASET"
)

for entry in "${COMBOS[@]}"; do
  read -r TASK MODEL SLOW_CKPT DATASET <<< "$entry"
  COMBO="${TASK}_${MODEL}"
  RUN_DIR="$ROOT_OUTPUT/$COMBO"
  TRAIN_LOG="$RUN_DIR/train.log"

  mkdir -p "$RUN_DIR"

  if [[ ! -s "$DATASET" ]]; then
    echo "Missing predicted-base dataset: $DATASET" >&2
    exit 1
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo
    echo "========== TRAIN $COMBO =========="
    echo "Run dir:   $RUN_DIR"
    echo "Slow ckpt: $SLOW_CKPT"
    echo "Dataset:   $DATASET"

    HYDRA_FULL_ERROR=1 python train.py \
      --config-name="residual_policy/$MODEL" \
      "residual_policy/task=$TASK" \
      "hydra.run.dir=$RUN_DIR" \
      "hydra.sweep.dir=$RUN_DIR" \
      "multi_run.run_dir=$RUN_DIR" \
      "task.dataset_path=$DATASET" \
      "task.slow_ckpt_path=$SLOW_CKPT" \
      "slow_ckpt_path=$SLOW_CKPT" \
      "policy.slow_ckpt_path=$SLOW_CKPT" \
      "training.device=$DEVICE" \
      "logging.mode=$LOGGING_MODE" \
      "logging.group=$WANDB_GROUP" \
      "logging.name=${RUN_STAMP}_${COMBO}" \
      "${EXTRA_TRAIN_OVERRIDES[@]}" \
      2>&1 | tee "$TRAIN_LOG"
  else
    echo
    echo "========== SKIP TRAIN $COMBO =========="
    echo "Run dir: $RUN_DIR"
  fi

  CKPT="$RUN_DIR/checkpoints/latest.ckpt"
  if [[ "$RUN_TRAIN" == "1" && ! -s "$CKPT" ]]; then
    echo "Missing latest checkpoint after training: $CKPT" >&2
    exit 1
  fi
done

echo
echo "Done."
echo "Root output: $ROOT_OUTPUT"
echo "Datasets:"
echo "  $FORCE_PRED_DATASET"
echo "  $NO_FORCE_PRED_DATASET"
