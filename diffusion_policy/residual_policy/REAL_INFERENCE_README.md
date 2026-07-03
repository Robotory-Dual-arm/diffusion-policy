# Corrected Slow-Fast Residual Inference Notes

This note is for moving the corrected slow-fast residual policy to another
robot computer. Checkpoint paths may change; use the command-line overrides
below instead of relying on the original paths embedded in the checkpoint.

## What Was Trained

Raw source dataset:

```text
/home/baetae/Downloads/common_data_height.hdf5
```

Important conversion facts:

```text
common_data_height observations/desired_pose:
  xyz: millimeters
  rotation: Euler ZYX degrees

corrected residual dataset:
  xyz: meters
  rotation: pose9, i.e. position + rotation_6d
```

The old bad residual dataset interpreted `desired_pose[:, 3:6]` as a rotvec in
radians. That created actual-to-virtual residual rotations around 120 degrees.
The corrected dataset uses `--virtual-rotation-format euler_ZYX_deg`.

Corrected active dataset used for training:

```text
data/outputs/residual_policy/data/original/common_data_height_euler_zyx_residual.hdf5
data/outputs/residual_policy/data/fast/actual_base_residual.hdf5 -> ../original/common_data_height_euler_zyx_residual.hdf5
```

Validation after conversion:

```text
checked_frames: 20605
actual_to_virtual_rotation_deg: mean=6.39514 p50=6.32783 p99=12.8746 max=15.6951
virtual_vs_reference_actions_rot_deg: max=1.62767e-05
validation: ok
```

Fast models trained:

```text
force_mlp
force_gru
no_force_mlp
no_force_gru
```

Latest run on the training machine:

```text
data/outputs/residual_policy/fast/corrected_actual_base_20260627_001230/
```

Recommended first robot candidates:

```text
1. no_force_gru/checkpoints/latest.ckpt
2. no_force_mlp/checkpoints/latest.ckpt
```

Why:

```text
no_force_gru: lowest validation loss and best chunked position error
no_force_mlp: best window16 average visualized error, useful non-recurrent comparison
```

## Files To Copy

Copy at least:

```text
fast checkpoint:
  corrected_actual_base_20260627_001230/no_force_gru/checkpoints/latest.ckpt

matching slow checkpoint:
  data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt
```

For force variants, copy the corresponding force slow checkpoint:

```text
data/outputs/residual_policy/slow/force/slow_force.ckpt
```

The dataset is not required for real inference. It is only referenced in the
checkpoint config for metadata. If the slow checkpoint path changes on the robot
computer, pass `--slow_ckpt_path`.

## Real Robot Eval Command

The real run needs the robot/ROS2 environment, including `rclpy`, camera access,
and the Doosan controller topics. The CLI help can be checked without hardware:

```bash
python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug --help
```

Minimal template:

```bash
python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
  --input /path/to/fast/latest.ckpt \
  --slow_ckpt_path /path/to/slow/slow_no_force.ckpt \
  --output /path/to/save/real_rollout
```

Current defaults:

```text
robot_ip: 192.168.111.50 placeholder, kept only for compatibility
steps_per_inference: 6
slow_action_start_offset: 1
frequency: 10 Hz
command_latency: 0.01 s
wrench_frame: auto
device: cuda:0
max_duration: 60 s
```

`--wrench_frame auto` should resolve to sensor frame for these corrected models.
They were trained with the original sensor/EEF-style `wrench_wrist_R` history
from `common_data_height`, not with a world-frame wrench dataset. If the startup
log does not show `fast wrench frame: sensor`, explicitly add:

```bash
--wrench_frame sensor
```

If the robot computer has a different CUDA device:

```bash
--device cuda:1
```

CPU inference is supported syntactically:

```bash
--device cpu
```

but it is not recommended for real-time robot control.

## Expected Startup Log

For `no_force_gru`, expect:

```text
policy target diffusion_policy.residual_policy.temporal_step_policy.FastResidualTemporalPolicy
task no_force
fast_action_target_shift: 1
slow obs keys: ['image0', 'robot_pose_R', 'robot_quat_R']
fast obs keys: ['image0', 'robot_pose_R', 'robot_quat_R', 'wrench_wrist_R', 'base_action_rel']
env obs keys: ['image0', 'robot_pose_R', 'robot_quat_R', 'wrench_wrist_R']
env action shape: [9]
fast wrench frame: sensor
slow obs/action repr: abs relative
fast obs repr: abs
device: cuda:0
```

During running, expect:

```text
New slow seq: ... input_steps: (0, 5) target_steps: (1, 6)
Fast inference latency: ... input_step: k target_step: k+1 cmd_lead: ...
Submitted 1 residual step. input_step=k/... target_step=k+1 cmd_lead=...
```

The first slow action `a_t` is not executed. Fast step 0 uses the slow anchor
observation and base action `a_{t+1}`, predicts `delta_{t+1}`, and schedules
the corrected command for timestamp `t+1`.

## Timing And Action Semantics

Training alignment:

```text
input at t:
  image from slow anchor t
  current pose/force at t+k
  base_action_rel for target t+k+1

target:
  residual_delta6_gt_actual_to_virtual at t+k+1
```

Real inference:

```text
slow_abs_target = slow_abs_action_seq[target_step]
base_action_rel = relative(slow_abs_target, current/input pose)
residual_delta6 = fast(base_action_rel, current force/pose, slow context image)
final_abs_action = slow_abs_target @ residual_delta6
```

The final command sent to the robot is pose9:

```text
xyz + rotation_6d
```

The robot controller converts rotation_6d to rotvec before executing.

Controller mode note:

```text
diffusion_policy/real_world/rightarm_hand_insert_plug_interpolation_controller.py
USE_IMPEDANCE_CONTROLLER = False
```

With the current flag, the controller uses the joint-position path through
`servoJ`. If the robot computer should use the task-space impedance topic
instead, change that flag intentionally and verify the corresponding ROS2
topic/controller is running.

## Safety Checks Before Real Run

1. Keep the wrench unloaded during startup calibration.
2. Confirm `fast wrench frame: sensor`.
3. Confirm `fast_action_target_shift: 1`.
4. Confirm logs show `target_steps: (1, 6)` for the default `steps_per_inference=6`.
5. Confirm `cmd_lead` is positive. If repeated outdated warnings appear, increase
   `--command_latency` or reduce runtime load.
6. Start with a short `--max_duration`, for example:

```bash
--max_duration 10
```

7. Prefer testing `no_force_gru` first, then `no_force_mlp`.

## Training And Visualization Results

Latest visualization summary:

```text
data/outputs/residual_policy/fast/corrected_actual_base_20260627_001230/visualization_error_summary.md
```

Fast-vs-virtual averages:

```text
window16:
  no_force_mlp  pos 1.7518 mm, rot 0.2242 deg
  no_force_gru  pos 1.8443 mm, rot 0.2542 deg
  force_gru     pos 1.9174 mm, rot 0.2688 deg
  force_mlp     pos 2.0420 mm, rot 0.2697 deg

chunked8_all:
  no_force_gru  pos 0.8663 mm, rot 0.2085 deg
  no_force_mlp  pos 0.8714 mm, rot 0.2177 deg
  force_gru     pos 0.9884 mm, rot 0.1818 deg
  force_mlp     pos 1.1249 mm, rot 0.2352 deg
```

The summary file's `pos_gain` and `rot_gain` columns compare different targets:
`slow` is measured against GT actual, while `fast` is measured against GT
virtual. Use `fast_vs_gt_virtual` for comparing fast models.

## Code Paths

Core files:

```text
diffusion_policy/residual_policy/eval_real_robot_rightarm_insert_plug.py
diffusion_policy/residual_policy/pose_util.py
diffusion_policy/residual_policy/mlp_policy.py
diffusion_policy/residual_policy/temporal_step_policy.py
diffusion_policy/real_world/bae_real_env_rightarm_hand_insert_plug.py
diffusion_policy/real_world/rightarm_hand_insert_plug_interpolation_controller.py
```

Data conversion and validation:

```text
diffusion_policy/residual_policy/convert_common_to_slow_dataset.py
diffusion_policy/residual_policy/validate_residual_dataset.py
scripts/rebuild_corrected_residual_and_train_fast4.sh
scripts/train_residual_fast4_and_visualize.sh
```
