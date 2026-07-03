# Residual Policy Notes

The active residual setup is intentionally limited to:

```text
slow: force, no_force
fast: mlp, gru
total: 4 training runs
```

Legacy `mlp_seq`, `gru_seq`, `*_slow`, pose9, image residual, slow-pred fast
dataset, and one-step policy files were removed from the active tree to keep
new training unambiguous.

Active code entry points:

```text
diffusion_policy/config/residual_policy/mlp.yaml
diffusion_policy/config/residual_policy/gru.yaml
diffusion_policy/config/residual_policy/task/force.yaml
diffusion_policy/config/residual_policy/task/no_force.yaml
diffusion_policy/residual_policy/step_dataset.py::FastResidualContextStepDataset
diffusion_policy/residual_policy/mlp_policy.py::FastResidualMLPPolicy
diffusion_policy/residual_policy/temporal_step_policy.py::FastResidualTemporalPolicy
```

## Slow Checkpoints

The slow checkpoints are exposed through stable residual-policy paths:

```text
data/outputs/residual_policy/slow/force/slow_force.ckpt
data/outputs/residual_policy/slow/no_force/slow_no_force.ckpt
```

These are the slow checkpoints used by the active fast training configs.

## Fast Training Data

Both active task configs train on the same actual-base residual dataset:

```text
data/outputs/residual_policy/data/fast/actual_base_residual.hdf5
```

For `common_data_height.hdf5`, rebuild this dataset through:

```bash
./scripts/rebuild_corrected_residual_and_train_fast4.sh
```

Important conversion detail:

```text
raw observations/desired_pose:
  xyz: mm
  rotation: Euler ZYX degrees
converted obs/virtual_target_abs:
  xyz: meters
  rotation: pose9 rotation_6d
```

Do not convert `desired_pose[:, 3:6]` as a rotvec in radians. The rebuild script
uses `--virtual-rotation-format euler_ZYX_deg`, then validates that
`actual->virtual` residual rotations stay in a normal range and that
`virtual_target_abs` matches `data/baetae/260602/diffusion_data.hdf5/actions`.

The base and target are:

```text
base target: obs/actual_target_abs shifted by 1, converted to rel from current pose
target:      obs/residual_delta6_gt_actual_to_virtual shifted by 1
shift:       action_target_shift = base_action_target_shift = 1
```

For training only, the dataset's actual action is used as a proxy for the slow
base action. In other words, fast learns a one-step-ahead residual:

```text
input at t:  obs_t, actual_action_{t+1} expressed relative to pose_t
target:      residual_delta6_{t+1}
```

At inference, that same learned correction is applied on top of the slow
policy's next predicted action.

## Action Representation

Slow policy output is relative pose9. For a slow chunk at anchor time `t`, the
slow policy predicts:

```text
a_t, a_{t+1}, ..., a_{t+15}
```

Each slow relative action is converted to an absolute target using the robot
pose at the slow anchor `t`:

```text
slow_abs_{t+k} = current_pose_abs_t @ slow_rel_{t+k}
```

Before feeding fast at real fast step `t+k`, the slow target that will receive
the residual is converted to a relative base action from the current pose at
`t+k`:

```text
base_rel_{t+k+1} = inv(current_pose_abs_{t+k}) @ slow_abs_{t+k+1}
```

The fast policy is trained to predict `residual_delta6_gt_actual_to_virtual`;
at inference we interpret that output as the same kind of correction, but
applied to the slow action:

```text
training target: delta6_train_{t+k+1} = inv(actual_action_abs_{t+k+1}) @ virtual_target_abs_{t+k+1}
inference pred:  delta6_pred_{t+k+1}  ~= correction from slow_abs_{t+k+1} to desired fast_abs_{t+k+1}
```

At inference:

```text
fast_abs_{t+k+1} = slow_abs_{t+k+1} @ predicted_delta6_{t+k+1}
```

## Slow/Fast Timing

Clean receding-horizon timing should be interpreted as:

```text
observe latest state at t
run slow once on obs_t
slow outputs nominal actions: a_t, a_{t+1}, ..., a_{t+15}
```

The first returned action `a_t` is usually already too close to the current
time by the time slow inference finishes. Treat it as stale for execution, but
do not erase it from the fast policy's time alignment:

```text
do not execute a_t
MLP: feed current obs plus base action a_{t+1}; this output corrects a_{t+1}
GRU: run step 0 with base action a_{t+1} to advance hidden and correct a_{t+1}
execute candidates a_{t+1}, a_{t+2}, ...
```

The real-robot residual eval script now defaults to this behavior with:

```text
--slow_action_start_offset 1
```

The eval loop binds every slow action index to a timestamp from the slow anchor:

```text
timestamp(a_{t+k}) = slow_anchor_timestamp + k * dt
```

For each fast loop:

```text
1. read the latest available obs; for input step 0, reuse the slow anchor obs_t
2. choose the earliest future timestamp that can still be commanded
3. select the matching target slow step k for that timestamp
4. feed fast with latest obs from step k-1 and base action a_k relative to pose_{k-1}
5. predict delta_k before timestamp(a_k)
6. schedule fast_abs_k = slow_abs_k @ predicted_delta6_k
```

If a residual prediction finishes too late for its chosen timestamp, that output
is not executed. The fast state is already advanced through its input step, so
the loop immediately tries the next slow target.

After a small accepted prefix, for example 6 or 8 fast ticks, re-observe and
run slow again. The unused tail of the old slow chunk is dropped.

## MLP

Config:

```text
diffusion_policy/config/residual_policy/mlp.yaml
```

Policy:

```text
diffusion_policy/residual_policy/mlp_policy.py::FastResidualMLPPolicy
```

The MLP is a non-recurrent sequence model. It uses one fixed slow-context image
for the whole 16-step chunk, but per-step current low-dim, force, slow action,
and step encoding:

```text
fixed:    image0[t]
per step: robot_pose_R[t+k], robot_quat_R[t+k],
          wrench_wrist_R[t+k], base_rel_for_target[t+k+1], step_encoding(k)
output:   residual_delta6[t+k+1]
```

Force handling:

```text
force slow:    reuse frozen slow force encoder
no_force slow: train a new causalconv force encoder inside fast
```

## GRU

Config:

```text
diffusion_policy/config/residual_policy/gru.yaml
```

The GRU initializes hidden state from the slow-context observation, then rolls
forward one fast step at a time:

```text
h0:       image0[t], robot_pose_R[t], robot_quat_R[t]
per step: wrench_wrist_R[t+k], base_rel_for_target[t+k+1]
output:   residual_delta6[t+k+1]
```

`include_initial_wrench` is set to `False`; force enters through the recurrent
step input.

## Training Commands

```bash
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/mlp residual_policy/task=force
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/gru residual_policy/task=force
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/mlp residual_policy/task=no_force
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/gru residual_policy/task=no_force
```

Outputs go under:

```text
data/outputs/residual_policy/fast/<task>_<fast>/<timestamp>/
```

The batch runner for all four active trainings plus world-frame visualization is:

```bash
./scripts/train_residual_fast4_and_visualize.sh
```

## Window Sampling

Training uses 16-step overlapping windows with stride 1:

```text
t     ... t+15
t+1   ... t+16
t+2   ... t+17
```

At episode starts the sampler still pads the missing previous frames because
`pad_before = n_obs_steps - 1`.
