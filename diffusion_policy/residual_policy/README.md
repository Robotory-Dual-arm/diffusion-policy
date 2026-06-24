# Residual Policy Notes

This folder keeps the CR-Dagger-style slow/fast residual experiments separate
from the original single diffusion policy code.

## Slow Policy

The slow policy is the existing diffusion policy trained in the usual way. It
predicts the slow/base action trajectory. The residual policies load this
checkpoint and keep it frozen.

Typical checkpoint:

```bash
data/outputs/2026.06.18/slow/slow.ckpt
```

## Fast Residual Variants

Main no-force-slow train configs:

```text
diffusion_policy/config/residual_policy/mlp.yaml
diffusion_policy/config/residual_policy/mlp_seq.yaml
diffusion_policy/config/residual_policy/gru.yaml
diffusion_policy/config/residual_policy/gru_seq.yaml
```

Older force-slow / legacy train configs:

```text
diffusion_policy/config/residual_policy/mlp_slow.yaml
diffusion_policy/config/residual_policy/mlp_seq_slow.yaml
diffusion_policy/config/residual_policy/gru_slow.yaml
diffusion_policy/config/residual_policy/image.yaml
```

Current task configs:

```text
diffusion_policy/config/residual_policy/task/nfslow_actual.yaml
diffusion_policy/config/residual_policy/task/nfslow_pred.yaml
diffusion_policy/config/residual_policy/task/slow_actual.yaml
diffusion_policy/config/residual_policy/task/slow_pred.yaml
diffusion_policy/config/residual_policy/task/slow_actual_pose9.yaml
diffusion_policy/config/residual_policy/task/image.yaml
```

Naming:

```text
mlp       = one-step MLP fast policy, default no-force slow
mlp_seq   = context-step MLP fast policy, default no-force slow
gru       = temporal GRU fast policy, default no-force slow
gru_seq   = temporal GRU fast policy with fixed slow context, default no-force slow
*_slow    = same structure but using the older force slow-policy task config
```

### One-step MLP

Config:

```bash
diffusion_policy/config/residual_policy/mlp.yaml
```

Policy:

```bash
diffusion_policy/residual_policy/step_policy.py
```

Input per fast step:

```text
current image/proprio/wrench + current slow/base action -> residual_delta6
```

### Temporal GRU

Config:

```bash
diffusion_policy/config/residual_policy/gru.yaml
```

Policy:

```bash
diffusion_policy/residual_policy/temporal_step_policy.py
```

Input per slow chunk:

```text
h0 from first image/proprio/wrench
then each recurrent step gets current wrench + current slow/base action
```

The GRU hidden state resets whenever the slow policy is re-run.

### Context-step MLP

Config:

```bash
diffusion_policy/config/residual_policy/mlp_seq.yaml
```

Policy:

```bash
diffusion_policy/residual_policy/context_step_policy.py
```

Dataset:

```bash
diffusion_policy/residual_policy/step_dataset.py::FastResidualContextStepDataset
```

Input per slow chunk:

```text
fixed context: first image + first proprio
per-step input: wrench[t] + slow/base_action[t]
output: residual_delta6[t]
```

This variant learns all 16 residual steps without a recurrent hidden state. The
context dataset intentionally loads only one image/proprio frame per sample, so
batch size 64 remains practical.

## Common Training Commands

Actual-to-virtual residual target:

```bash
HYDRA_FULL_ERROR=1 python train.py \
  --config-name=residual_policy/mlp_seq \
  task.dataset_path=data/baetae/260618/slow_erase_board_virtual_m.hdf5 \
  task.slow_ckpt_path=data/outputs/2026.06.18/slow/slow.ckpt
```

Slow-predicted residual target:

```bash
HYDRA_FULL_ERROR=1 python train.py \
  --config-name=residual_policy/mlp_seq \
  residual_policy/task=slow_pred
```

No-force slow 4-way run names:

```bash
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/mlp
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/mlp_seq
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/gru
HYDRA_FULL_ERROR=1 python train.py --config-name=residual_policy/gru_seq
```

## Visualization

Script:

```bash
python -m diffusion_policy.residual_policy.test.visualize_step_residual_predictions
```

Useful output layout:

```text
plots/rotvec_eval/<model_name>/window16
plots/rotvec_eval/<model_name>/chunked80
plots/rotvec_eval/<model_name>/world_frame
```

`--world-frame` applies the same right-arm robot-base to world-frame rotation
used by the older plotting scripts.
