#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${ROOT_OUTPUT:-}" ]]; then
  ROOT_OUTPUT="$(find data/outputs/residual_policy/fast -maxdepth 1 -type d -name 'pred_base_*' | sort | tail -n 1)"
fi
if [[ -z "$ROOT_OUTPUT" || ! -d "$ROOT_OUTPUT" ]]; then
  echo "Could not find pred-base output root. Set ROOT_OUTPUT=..." >&2
  exit 1
fi

FORCE_DATASET="${FORCE_DATASET:-data/outputs/residual_policy/data/fast/slow_pred_base/force_pred_base_residual.hdf5}"
NO_FORCE_DATASET="${NO_FORCE_DATASET:-data/outputs/residual_policy/data/fast/slow_pred_base/no_force_pred_base_residual.hdf5}"
FORCE_SLOW_CKPT="${FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/force/slow_force.ckpt}"
NO_FORCE_SLOW_CKPT="${NO_FORCE_SLOW_CKPT:-data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt}"

VIS_DEVICE="${VIS_DEVICE:-cuda:0}"

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

# Optional extra visualize args, e.g.
# VIS_OVERRIDES='--window-count 10 --batch-size 8'
# shellcheck disable=SC2206
EXTRA_VIS_OVERRIDES=(${VIS_OVERRIDES:-})

declare -a COMBOS=(
  "force_pred_base_mlp $FORCE_SLOW_CKPT $FORCE_DATASET"
  "force_pred_base_gru $FORCE_SLOW_CKPT $FORCE_DATASET"
  "no_force_pred_base_mlp $NO_FORCE_SLOW_CKPT $NO_FORCE_DATASET"
  "no_force_pred_base_gru $NO_FORCE_SLOW_CKPT $NO_FORCE_DATASET"
)

echo "Root output:      $ROOT_OUTPUT"
echo "Force dataset:    $FORCE_DATASET"
echo "No-force dataset: $NO_FORCE_DATASET"
echo "Vis device:       $VIS_DEVICE"

echo
echo "========== VISUALIZE PRED-BASE FAST4 =========="

for entry in "${COMBOS[@]}"; do
  read -r COMBO SLOW_CKPT DATASET <<< "$entry"
  RUN_DIR="$ROOT_OUTPUT/$COMBO"
  CKPT="$RUN_DIR/checkpoints/latest.ckpt"
  VIS_DIR="$RUN_DIR/visualization_world"
  VIS_LOG="$RUN_DIR/visualization_world.log"

  if [[ ! -s "$CKPT" ]]; then
    echo "Missing latest checkpoint: $CKPT" >&2
    exit 1
  fi
  if [[ ! -s "$DATASET" ]]; then
    echo "Missing visualization dataset: $DATASET" >&2
    exit 1
  fi

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
  echo "Dataset:   $DATASET"
  echo "Slow ckpt: $SLOW_CKPT"
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
    "# Residual Pred-Base Fast Visualization Error Summary",
    "",
    f"Root output: `{root}`",
    "",
    "| combo | view | n | slow pos mm | fast pos mm | pos gain mm | slow rot deg | fast rot deg | rot gain deg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for (combo, view), group_rows in sorted(groups.items()):
    slow_pos = mean([r["slow_pos_mm_mean"] for r in group_rows])
    fast_pos = mean([r["fast_pos_mm_mean"] for r in group_rows])
    pos_gain = mean([r["pos_improvement_mm"] for r in group_rows])
    slow_rot = mean([r["slow_rot_deg_mean"] for r in group_rows])
    fast_rot = mean([r["fast_rot_deg_mean"] for r in group_rows])
    rot_gain = mean([r["rot_improvement_deg"] for r in group_rows])
    lines.append(
        f"| {combo} | {view} | {len(group_rows)} | "
        f"{slow_pos:.4f} | {fast_pos:.4f} | {pos_gain:.4f} | "
        f"{slow_rot:.4f} | {fast_rot:.4f} | {rot_gain:.4f} |"
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
