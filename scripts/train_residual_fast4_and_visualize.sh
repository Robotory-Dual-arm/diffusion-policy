#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUTPUT="${ROOT_OUTPUT:-data/outputs/residual_policy/fast/train4_${RUN_STAMP}}"
DATASET="${DATASET:-data/outputs/residual_policy/data/fast/actual_base_residual.hdf5}"

DEVICE="${DEVICE:-cuda:0}"
VIS_DEVICE="${VIS_DEVICE:-$DEVICE}"
LOGGING_MODE="${LOGGING_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-residual_fast4_${RUN_STAMP}}"

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
RUN_TRAIN="${RUN_TRAIN:-1}"

FORCE_SLOW_CKPT="${FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/force/slow_force.ckpt}"
NO_FORCE_SLOW_CKPT="${NO_FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt}"

# Optional extra Hydra overrides, e.g.
# TRAIN_OVERRIDES='training.num_epochs=300 dataloader.num_workers=4'
# VIS_OVERRIDES='--window-count 10 --batch-size 8'
# shellcheck disable=SC2206
EXTRA_TRAIN_OVERRIDES=(${TRAIN_OVERRIDES:-})
# shellcheck disable=SC2206
EXTRA_VIS_OVERRIDES=(${VIS_OVERRIDES:-})

mkdir -p "$ROOT_OUTPUT"

declare -a COMBOS=(
  "force mlp $FORCE_SLOW_CKPT"
  "force gru $FORCE_SLOW_CKPT"
  "no_force mlp $NO_FORCE_SLOW_CKPT"
  "no_force gru $NO_FORCE_SLOW_CKPT"
)

declare -A RUN_DIR_BY_COMBO
declare -A CKPT_BY_COMBO
declare -A SLOW_CKPT_BY_COMBO

echo "Root output: $ROOT_OUTPUT"
echo "Dataset:     $DATASET"
echo "Device:      $DEVICE"
echo "Vis device:  $VIS_DEVICE"
echo "Run train:   $RUN_TRAIN"

for entry in "${COMBOS[@]}"; do
  read -r TASK MODEL SLOW_CKPT <<< "$entry"
  COMBO="${TASK}_${MODEL}"
  RUN_DIR="$ROOT_OUTPUT/$COMBO"
  TRAIN_LOG="$RUN_DIR/train.log"

  mkdir -p "$RUN_DIR"
  RUN_DIR_BY_COMBO["$COMBO"]="$RUN_DIR"
  SLOW_CKPT_BY_COMBO["$COMBO"]="$SLOW_CKPT"

  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo
    echo "========== TRAIN $COMBO =========="
    echo "Run dir:   $RUN_DIR"
    echo "Slow ckpt: $SLOW_CKPT"

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
  if [[ ! -s "$CKPT" ]]; then
    echo "Missing latest checkpoint: $CKPT" >&2
    exit 1
  fi
  CKPT_BY_COMBO["$COMBO"]="$CKPT"
done

echo
echo "========== VISUALIZE =========="

for entry in "${COMBOS[@]}"; do
  read -r TASK MODEL SLOW_CKPT <<< "$entry"
  COMBO="${TASK}_${MODEL}"
  RUN_DIR="${RUN_DIR_BY_COMBO[$COMBO]}"
  CKPT="${CKPT_BY_COMBO[$COMBO]}"
  VIS_DIR="$RUN_DIR/visualization_world"
  VIS_LOG="$RUN_DIR/visualization_world.log"

  mkdir -p "$VIS_DIR"

  VIS_CMD=(
    python diffusion_policy/residual_policy/test/visualize_step_residual_predictions.py
    --dataset "$DATASET"
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
      DEMO="$(echo "$DEMO" | xargs)"
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

python - "$ROOT_OUTPUT" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []

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
        "num_points": metric.get("num_points", ""),
        "slow_pos_mm_mean": slow_pos,
        "fast_pos_mm_mean": fast_pos,
        "pos_improvement_mm": (None if slow_pos is None or fast_pos is None else slow_pos - fast_pos),
        "slow_rot_deg_mean": slow_rot,
        "fast_rot_deg_mean": fast_rot,
        "rot_improvement_deg": (None if slow_rot is None or fast_rot is None else slow_rot - fast_rot),
    })

for metrics_path in sorted(root.glob("*/visualization_world/metrics.json")):
    combo = metrics_path.parent.parent.name
    data = json.loads(metrics_path.read_text())
    for demo, sections in data.items():
        if "continuous" in sections:
            add_row(combo, demo, "continuous", "all", sections["continuous"])
        for anchor, metric in sections.get("windows", {}).items():
            add_row(combo, demo, "window16", anchor, metric)
        for start, metric in sections.get("chunked", {}).items():
            add_row(combo, demo, f"chunked{metric.get('exec_steps', '')}_all", start, metric)

summary_json = root / "visualization_error_summary.json"
summary_csv = root / "visualization_error_summary.csv"
summary_md = root / "visualization_error_summary.md"

summary_json.write_text(json.dumps(rows, indent=2))

fieldnames = [
    "combo", "demo", "view", "case", "num_points",
    "slow_pos_mm_mean", "fast_pos_mm_mean", "pos_improvement_mm",
    "slow_rot_deg_mean", "fast_rot_deg_mean", "rot_improvement_deg",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["combo"], row["view"])].append(row)

def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None

lines = [
    "# Residual Fast Visualization Error Summary",
    "",
    f"Root output: `{root}`",
    "",
    "| combo | view | n | slow pos mm | fast pos mm | pos gain mm | slow rot deg | fast rot deg | rot gain deg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for (combo, view), group_rows in sorted(groups.items()):
    vals = {
        "slow_pos": mean([r["slow_pos_mm_mean"] for r in group_rows]),
        "fast_pos": mean([r["fast_pos_mm_mean"] for r in group_rows]),
        "pos_gain": mean([r["pos_improvement_mm"] for r in group_rows]),
        "slow_rot": mean([r["slow_rot_deg_mean"] for r in group_rows]),
        "fast_rot": mean([r["fast_rot_deg_mean"] for r in group_rows]),
        "rot_gain": mean([r["rot_improvement_deg"] for r in group_rows]),
    }
    def fmt(x):
        return "" if x is None else f"{x:.4f}"
    lines.append(
        f"| {combo} | {view} | {len(group_rows)} | "
        f"{fmt(vals['slow_pos'])} | {fmt(vals['fast_pos'])} | {fmt(vals['pos_gain'])} | "
        f"{fmt(vals['slow_rot'])} | {fmt(vals['fast_rot'])} | {fmt(vals['rot_gain'])} |"
    )
summary_md.write_text("\n".join(lines) + "\n")

print(summary_json)
print(summary_csv)
print(summary_md)
PY

echo
echo "Done."
echo "Root output: $ROOT_OUTPUT"
echo "Summary:     $ROOT_OUTPUT/visualization_error_summary.md"
