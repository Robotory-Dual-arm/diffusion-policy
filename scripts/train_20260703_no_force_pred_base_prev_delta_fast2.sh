#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DATE="${OUTPUT_DATE:-$(date +%Y.%m.%d)}"

SLOW_CKPT="${SLOW_CKPT:-data/outputs/2026.07.02_residual_policy/slow/no_force/epoch=0900-train_loss=0.000.ckpt}"
PRED_DATASET="${PRED_DATASET:-data/outputs/2026.07.02_residual_policy/data/no_force_pred_base_residual.hdf5}"
ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/${OUTPUT_DATE}_residual_policy/fast/no_force_pred_base_prev_delta_${RUN_STAMP}}"

DEVICE="${DEVICE:-cuda:0}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-no_force_pred_base_prev_delta_${RUN_STAMP}}"
WANDB_RESUME="${WANDB_RESUME:-false}"
WANDB_ID="${WANDB_ID:-null}"
RESET_WANDB_RESUME="${RESET_WANDB_RESUME:-1}"
RUN_DATA_PREP="${RUN_DATA_PREP:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"

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

# Optional Hydra overrides.
# Example:
#   TRAIN_OVERRIDES='training.num_epochs=300 dataloader.num_workers=4'
# shellcheck disable=SC2206
EXTRA_TRAIN_OVERRIDES=(${TRAIN_OVERRIDES:-})

if [[ "$RUN_DATA_PREP" == "1" && ! -s "$PRED_DATASET" ]]; then
  echo "Pred-base dataset not found; building data with previous 2026.07.02 data-prep script."
  CONDA_ENV="$CONDA_ENV" \
  SLOW_CKPT="$SLOW_CKPT" \
  PRED_DATASET="$PRED_DATASET" \
  RUN_TRAIN=0 \
  RUN_VIS=0 \
  ./scripts/train_20260702_no_force_pred_base_fast2.sh
fi

if [[ ! -s "$SLOW_CKPT" ]]; then
  echo "Missing slow checkpoint: $SLOW_CKPT" >&2
  exit 1
fi
if [[ ! -s "$PRED_DATASET" ]]; then
  echo "Missing pred-base residual dataset: $PRED_DATASET" >&2
  echo "Set RUN_DATA_PREP=1 to let this script call the data-prep script." >&2
  exit 1
fi

mkdir -p "$ROOT_OUTPUT"

echo "========== no-force pred-base prev-delta fast =========="
echo "Python command:    ${PYTHON_CMD[*]}"
echo "Slow ckpt:         $SLOW_CKPT"
echo "Pred-base dataset: $PRED_DATASET"
echo "Output date:       $OUTPUT_DATE"
echo "Root output:       $ROOT_OUTPUT"
echo "Device:            $DEVICE"
echo "Logging mode:      $LOGGING_MODE"
echo "W&B group:         $WANDB_GROUP"
echo "W&B resume:        $WANDB_RESUME"

# Optional model subset.
# Example:
#   MODELS='mlp gru' ./scripts/train_20260703_no_force_pred_base_prev_delta_fast2.sh
# shellcheck disable=SC2206
MODELS=(${MODELS:-gru})

if [[ "$RUN_TRAIN" == "1" ]]; then
  for MODEL in "${MODELS[@]}"; do
    COMBO="no_force_pred_base_prev_delta_${MODEL}"
    RUN_DIR="$ROOT_OUTPUT/$COMBO"
    TRAIN_LOG="$RUN_DIR/train.log"
    mkdir -p "$RUN_DIR"

    if [[ "$RESET_WANDB_RESUME" == "1" && -f "$RUN_DIR/wandb/wandb-resume.json" ]]; then
      mv "$RUN_DIR/wandb/wandb-resume.json" "$RUN_DIR/wandb/wandb-resume.json.disabled.${RUN_STAMP}"
    fi

    echo
    echo "========== TRAIN $COMBO =========="
    echo "Run dir:   $RUN_DIR"

    SLOW_CKPT_HYDRA="$(hydra_quote "$SLOW_CKPT")"
    HYDRA_FULL_ERROR=1 "${PYTHON_CMD[@]}" train.py \
      --config-name="residual_policy/$MODEL" \
      "residual_policy/task=no_force_pred_base_prev_delta" \
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
      "logging.resume=$WANDB_RESUME" \
      "logging.id=$WANDB_ID" \
      "logging.name=${RUN_STAMP}_${COMBO}" \
      "${EXTRA_TRAIN_OVERRIDES[@]}" \
      2>&1 | tee "$TRAIN_LOG"

    CKPT="$RUN_DIR/checkpoints/latest.ckpt"
    if [[ ! -s "$CKPT" ]]; then
      echo "Missing latest checkpoint: $CKPT" >&2
      exit 1
    fi
  done
else
  echo "RUN_TRAIN=0, skipping training."
fi

echo
echo "Done."
echo "Training output: $ROOT_OUTPUT"
