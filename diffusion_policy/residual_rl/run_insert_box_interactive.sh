#!/usr/bin/env bash
set -euo pipefail

# Required values are deliberately not guessed: copy the force/torque and
# freshness limits approved for the robot into these environment variables.
if [[ -z "${MAX_FORCE_N:-}" || -z "${MAX_TORQUE_NM:-}" || -z "${MAX_OBS_AGE_S:-}" ]]; then
  echo "Set MAX_FORCE_N, MAX_TORQUE_NM, and MAX_OBS_AGE_S before running." >&2
  echo "Example: MAX_FORCE_N=... MAX_TORQUE_NM=... MAX_OBS_AGE_S=... bash $0" >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-robodiff}"
DEVICE="${DEVICE:-cuda:0}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-data/outputs/2026.07.15_residual_rl/fast_bc_16step/checkpoints/best.pt}"
SESSION_OUTPUT="${SESSION_OUTPUT:-data/results/residual_rl_insert_box_interactive}"
TRAIN_AFTER_EPISODES="${TRAIN_AFTER_EPISODES:-5}"
MIN_REPLAY_TRANSITIONS="${MIN_REPLAY_TRANSITIONS:-256}"
UTD_RATIO="${UTD_RATIO:-4}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-20000}"
OFFLINE_ACTUAL_DATASET="${OFFLINE_ACTUAL_DATASET:-data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5}"
OFFLINE_VIRTUAL_DATASET="${OFFLINE_VIRTUAL_DATASET:-data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5}"
OFFLINE_BASE_PREDICTIONS="${OFFLINE_BASE_PREDICTIONS:-data/outputs/2026.07.15_residual_rl/base_predictions.hdf5}"

args=(
  --actor-checkpoint "$ACTOR_CHECKPOINT"
  --base-checkpoint "$BASE_CHECKPOINT"
  --output "$SESSION_OUTPUT"
  --device "$DEVICE"
  --training-device "$DEVICE"
  --control-mode impedance
  --base-inference-steps 16
  --slow-action-start-index 1
  --fast-steps-per-slow 6
  --max-force-norm-n "$MAX_FORCE_N"
  --max-torque-norm-nm "$MAX_TORQUE_NM"
  --max-observation-age-s "$MAX_OBS_AGE_S"
  --train-after-episodes "$TRAIN_AFTER_EPISODES"
  --min-replay-transitions "$MIN_REPLAY_TRANSITIONS"
  --utd-ratio "$UTD_RATIO"
  --n-step 3
  --offline-ratio 0.5
  --offline-actual-dataset "$OFFLINE_ACTUAL_DATASET"
  --offline-virtual-dataset "$OFFLINE_VIRTUAL_DATASET"
  --offline-base-predictions "$OFFLINE_BASE_PREDICTIONS"
  --replay-capacity "$REPLAY_CAPACITY"
)

# Optional reproducibility/debug override. Normally update count is
# UTD_RATIO * newly collected online transitions after the robot is closed.
if [[ -n "${TD3_UPDATES:-}" ]]; then
  args+=(--td3-updates "$TD3_UPDATES")
fi

# When ACTOR_CHECKPOINT is a TD3 checkpoint this must point to the original
# frozen BC checkpoint. It is intentionally omitted for a first BC-start run.
if [[ -n "${BC_CHECKPOINT:-}" ]]; then
  args+=(--bc-checkpoint "$BC_CHECKPOINT")
fi

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python -m diffusion_policy.residual_rl.interactive "${args[@]}" "$@"
