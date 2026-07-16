#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK_CONFIG="${TASK_CONFIG:-insert_box_hand_force_pred_base}"
MODEL="${MODEL:-gru}"

DATA_ROOT="${DATA_ROOT:-data/baetae/260715_insert_box_hand}"
ACTUAL_DATASET="${ACTUAL_DATASET:-$DATA_ROOT/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5}"
VIRTUAL_DATASET="${VIRTUAL_DATASET:-$DATA_ROOT/diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5}"
SLOW_CKPT="${SLOW_CKPT:-data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt}"

PREP_ROOT="${PREP_ROOT:-data/outputs/2026.07.15_residual_policy/data}"
ACTUAL_BASE_DATASET="${ACTUAL_BASE_DATASET:-$PREP_ROOT/insert_box_hand_actual_base_residual.hdf5}"
PRED_DATASET="${PRED_DATASET:-$PREP_ROOT/insert_box_hand_pred_base_residual.hdf5}"
ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/2026.07.15_residual_policy/fast/${TASK_CONFIG}_${MODEL}_${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-$ROOT_OUTPUT/${TASK_CONFIG}_${MODEL}}"

DEVICE="${DEVICE:-cuda:0}"
DATA_DEVICE="${DATA_DEVICE:-$DEVICE}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${TASK_CONFIG}_${MODEL}_${RUN_STAMP}}"
RUN_PAIR_DATA="${RUN_PAIR_DATA:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_CREATE_PRED="${RUN_CREATE_PRED:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
OVERWRITE_PAIR_DATA="${OVERWRITE_PAIR_DATA:-0}"
OVERWRITE_PRED_DATA="${OVERWRITE_PRED_DATA:-0}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-16}"
PRED_NUM_INFERENCE_STEPS="${PRED_NUM_INFERENCE_STEPS:-16}"
DEMO_LIMIT="${DEMO_LIMIT:-}"

CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

if [[ "$MODEL" != "gru" && "$MODEL" != "mlp" ]]; then
  echo "MODEL must be gru or mlp, got: $MODEL" >&2
  exit 1
fi

hydra_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "'%s'" "$value"
}

# Example: TRAIN_OVERRIDES='training.num_epochs=300 dataloader.num_workers=4'
# shellcheck disable=SC2206
EXTRA_TRAIN_OVERRIDES=(${TRAIN_OVERRIDES:-})

for required in "$ACTUAL_DATASET" "$VIRTUAL_DATASET" "$SLOW_CKPT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
mkdir -p "$PREP_ROOT" "$RUN_DIR"

echo "========== 2026.07.15 insert-box-hand residual =========="
echo "Model:              $MODEL"
echo "Actual data:        $ACTUAL_DATASET"
echo "Virtual data:       $VIRTUAL_DATASET"
echo "Slow checkpoint:    $SLOW_CKPT"
echo "Actual-base data:   $ACTUAL_BASE_DATASET"
echo "Pred-base data:     $PRED_DATASET"
echo "Fast correction:    EE delta6 only"
echo "Slow passthrough:    hand7"
echo "Run dir:            $RUN_DIR"

if [[ "$RUN_PAIR_DATA" == "1" ]]; then
  if [[ -s "$ACTUAL_BASE_DATASET" && "$OVERWRITE_PAIR_DATA" != "1" ]]; then
    echo "Keeping existing actual-base dataset: $ACTUAL_BASE_DATASET"
  else
    PAIR_ARGS=(
      --actual "$ACTUAL_DATASET"
      --virtual "$VIRTUAL_DATASET"
      --output "$ACTUAL_BASE_DATASET"
    )
    if [[ "$OVERWRITE_PAIR_DATA" == "1" ]]; then
      PAIR_ARGS+=(--overwrite)
    fi
    if [[ -n "$DEMO_LIMIT" ]]; then
      PAIR_ARGS+=(--demo-limit "$DEMO_LIMIT")
    fi
    "${PYTHON_CMD[@]}" diffusion_policy/residual_policy/create_residual_dataset_from_action_pair.py "${PAIR_ARGS[@]}"
  fi
fi

if [[ ! -s "$ACTUAL_BASE_DATASET" ]]; then
  echo "Missing actual-base residual dataset: $ACTUAL_BASE_DATASET" >&2
  exit 1
fi

if [[ "$RUN_VALIDATE" == "1" ]]; then
  "${PYTHON_CMD[@]}" diffusion_policy/residual_policy/validate_residual_dataset.py \
    --dataset "$ACTUAL_BASE_DATASET" \
    --reference-dataset "$VIRTUAL_DATASET" \
    --max-reference-rotation-deg 0.001 \
    --max-residual-rotation-deg 30.0
fi

if [[ "$RUN_CREATE_PRED" == "1" ]]; then
  if [[ -s "$PRED_DATASET" && "$OVERWRITE_PRED_DATA" != "1" ]]; then
    echo "Keeping existing slow-pred-base dataset: $PRED_DATASET"
  else
    PRED_ARGS=(
      --input "$ACTUAL_BASE_DATASET"
      --output "$PRED_DATASET"
      --slow-ckpt "$SLOW_CKPT"
      --device "$DATA_DEVICE"
      --batch-size "$PRED_BATCH_SIZE"
      --target-shift 1
      --slow-action-index 0
      --num-inference-steps "$PRED_NUM_INFERENCE_STEPS"
      --full-action-steps
    )
    if [[ "$OVERWRITE_PRED_DATA" == "1" ]]; then
      PRED_ARGS+=(--overwrite)
    fi
    "${PYTHON_CMD[@]}" diffusion_policy/residual_policy/create_slow_pred_fast_dataset.py "${PRED_ARGS[@]}"
  fi
fi

if [[ ! -s "$PRED_DATASET" ]]; then
  echo "Missing slow-pred-base residual dataset: $PRED_DATASET" >&2
  exit 1
fi

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "RUN_TRAIN=0, data preparation complete."
  exit 0
fi

SLOW_CKPT_HYDRA="$(hydra_quote "$SLOW_CKPT")"
HYDRA_FULL_ERROR=1 "${PYTHON_CMD[@]}" train.py \
  --config-name="residual_policy/$MODEL" \
  "residual_policy/task=$TASK_CONFIG" \
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
  "logging.name=${RUN_STAMP}_${TASK_CONFIG}_${MODEL}" \
  "${EXTRA_TRAIN_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Training output: $RUN_DIR"
