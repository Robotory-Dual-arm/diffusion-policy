# Insert-box feed-forward residual TD3

이 폴더는 기존 `diffusion_policy/residual_policy`와 분리된 residual RL v1이다.
기존 slow/base diffusion policy는 항상 고정하고, 새 fast actor만 paired
demonstration으로 BC한 뒤 실제 로봇의 terminal 성공/실패 label로 TD3 학습한다.

## 실행 전 필수 입력 체크리스트

필요한 파일은 실행 모드에 따라 다르다.

| 실행 모드 | 반드시 필요한 파일 | 필요하지 않은 파일 |
|---|---|---|
| exploration 없는 collect/evaluate | frozen base checkpoint, 실행할 fast BC/TD3 actor checkpoint | actual/virtual HDF5, base prediction cache |
| 최초 interactive collect + 자동 TD3 | frozen base checkpoint, fast BC checkpoint, actual/virtual HDF5, base prediction cache | 기존 TD3 checkpoint |
| TD3 학습 후 interactive 재시작 | frozen base checkpoint, 마지막 TD3 checkpoint, 원본 frozen fast BC checkpoint, actual/virtual HDF5, base prediction cache, 기존 online episode directory | 없음 |
| fast BC를 처음부터 다시 학습 | actual/virtual HDF5, frozen base checkpoint, 생성한 base prediction cache | online episode, TD3 checkpoint |
| 저장된 episode로 TD3만 독립 실행 | fast BC checkpoint, actual/virtual HDF5, base prediction cache, online episode sidecar | 실로봇 연결, base checkpoint |

각 파일의 역할은 다음과 같다.

- **Frozen base checkpoint**: 이미지와 robot state에서 nominal actual target chunk를
  예측한다. 실로봇 inference에서 매번 직접 실행하며 RL 동안 절대 학습하지 않는다.
- **Fast BC checkpoint**: actual-to-virtual residual actor의 시작점이자 TD3 동안
  고정할 BC prior이다. 첫 interactive 실행에서는 이 파일 하나가 `actor`와
  `frozen prior` 두 역할을 한다.
- **마지막 TD3 checkpoint**: critic/target/optimizer까지 포함한 이어서 학습할
  checkpoint이다. 이것으로 재시작할 때에도 원본 fast BC checkpoint를 별도로
  반드시 보관해야 한다.
- **Actual/virtual HDF5**: fast BC label과 TD3 batch의 offline 성공 demonstration
  50%를 제공한다. 순수 실로봇 inference에는 필요하지 않다.
- **Base prediction cache**: offline sample마다 frozen base가 예측한
  `base_action16`을 제공한다. Base model weight를 대신하는 파일이 아니며 순수
  실로봇 inference에서는 사용하지 않는다.
- **Online episode sidecar**: 실로봇에서 실제 실행한 residual과 성공/실패
  terminal reward를 저장한다. TD3 online batch 50%의 원본이다.

현재 preset이 사용하는 기본 경로는 다음과 같다.

```text
frozen base:
  data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt

original fast BC / frozen prior:
  data/outputs/2026.07.15_residual_rl/fast_bc_16step/checkpoints/best.pt

base prediction cache:
  data/outputs/2026.07.15_residual_rl/base_predictions.hdf5

actual demonstration:
  data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5

virtual demonstration:
  data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5

online replay와 TD3 round 출력:
  data/results/residual_rl_insert_box_interactive/
```

파일 외에도 다음 항목이 반드시 필요하다.

- `robodiff` Python/CUDA 환경과 base policy용 PyTorch3D
- robot 및 camera 연결, 올바른 robot IP와 camera serial
- impedance controller와 arm/hand feedback 및 command topic
- 로봇에서 승인한 `MAX_FORCE_N`, `MAX_TORQUE_NM`, `MAX_OBS_AGE_S`
- replay와 checkpoint를 저장할 충분한 RAM/disk 및 쓰기 가능한 session directory

Force/torque/freshness limit은 repository 안에 authoritative default가 없으므로
launcher가 추측하지 않는다. 세 값을 주지 않으면 robot environment를 열기 전에
실행을 거부한다.

## 모델과 시간축

fast tick마다 최신 observation을 새로 읽는다.

- `image0`: `[3, 224, 224]`
- `robot_pose_R`: `[3]`
- `robot_quat_R`: `[4]`, XYZW quaternion
- `hand_pose_R`: `[7]`
- `wrench_wrist_R`: `[6, 32]`
- `base_action`: `[16]` = 현재 actual pose 기준 nominal relative pose9 + base hand target7
- actor action: `[6]` = local XYZ translation(m) + rotation vector(rad)

이전 residual action, GRU hidden state, action history는 넣지 않는다. Actor는
`obs + base_action16 -> residual6`, twin critic은
`obs + base_action16 + 실제 실행 residual6 -> Q1,Q2`이다.

base는 한 번 추론해 action chunk의 index `1:7`을 여섯 fast tick에 사용한다.
그동안 fast actor의 이미지·pose·hand·wrench는 매 tick 갱신한다. 여섯 action을
소진하면 base를 동기식으로 다시 추론한다. 현재는 다음 base inference가 늦을 때
이전 chunk의 index `7:`을 사용하는 fallback이 없다. Cache, BC, collect,
evaluate 모두 base diffusion sampling 16 steps와 이 `1:7` schedule을 동일하게
사용하며, checkpoint metadata가
다르면 실로봇 launcher가 시작을 거부한다.

Pose 합성은 기존 구현과 같은 right composition이다.

```text
T_virtual = T_base @ T_delta
```

BC label은 base prediction과 virtual의 차이가 아니라 paired demonstration의
`actual_target -> virtual_target` delta6이다. 따라서 actor는 0부터 시작하지 않고
demonstration의 actual-to-virtual 보정에서 시작한다.

## 최초 1회: base prediction cache

Base checkpoint가 이미 학습되어 있어도 fast BC sample마다 넣을 nominal
`base_action16`은 별도로 필요하다. 다음 명령은 실제 online slow/fast schedule과
같은 방식으로 frozen base prediction을 한 번 계산해 작은 HDF5 cache로 저장한다.

```bash
conda run --no-capture-output -n robodiff \
  python -m diffusion_policy.residual_rl.cache_base_predictions \
  --actual-dataset data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5 \
  --base-checkpoint data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt \
  --output data/outputs/2026.07.15_residual_rl/base_predictions.hdf5 \
  --target-offset 1 \
  --slow-action-start-index 1 \
  --fast-steps-per-slow 6 \
  --inference-steps 16 \
  --batch-size 4 \
  --seed 42 \
  --device cuda:0
```

Cache에는 actual HDF5, base checkpoint, EMA/model 선택, schedule의 SHA-256과
metadata가 저장된다. 다른 컴퓨터에서 경로가 달라도 파일 내용이 같으면 사용할
수 있고, 내용이 달라지면 BC가 시작 전에 거부한다.

## 최초 1회: fast BC 100 epoch

```bash
conda run --no-capture-output -n robodiff \
  python -m diffusion_policy.residual_rl.train_bc \
  --actual-dataset data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5 \
  --virtual-dataset data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5 \
  --base-predictions data/outputs/2026.07.15_residual_rl/base_predictions.hdf5 \
  --base-action-key actions \
  --target-shift 1 \
  --slow-action-start-index 1 \
  --fast-steps-per-slow 6 \
  --base-inference-steps 16 \
  --residual-bound-scale 1.3 \
  --epochs 100 \
  --batch-size 64 \
  --stats-batch-size 256 \
  --num-workers 4 \
  --seed 42 \
  --output data/outputs/2026.07.15_residual_rl/fast_bc_16step \
  --device cuda:0
```

Residual limit은 각 delta6 축마다
`limit_i = 1.3 * max(abs(demonstration_delta_i))`로 계산하며 범위는
`[-limit_i, +limit_i]`이다. 현재 20,140 sample에서 계산된 값은 다음과 같다.
앞의 3축은 m, 뒤의 3축은 rad이다.

```text
residual_min =
[-0.0351267792, -0.0313787833, -0.0283588432,
 -0.1836365312, -0.1261037141, -0.1770352721]

residual_max =
[ 0.0351267792,  0.0313787833,  0.0283588432,
  0.1836365312,  0.1261037141,  0.1770352721]
```

100 epoch 동안 `latest.pt`와 `best.pt`만 덮어써서 checkpoint는 두 개만 남는다.
실로봇 시작 checkpoint는 `fast_bc_16step/checkpoints/best.pt`이다.

## 권장 사용법: 한 프로세스에서 rollout과 TD3 반복

실제 robot에서 승인한 세 safety 값만 넣으면 preset script가 현재 경로와
16-step/1:7 schedule을 채운다. 값은 repository에 authoritative default가 없어
임의로 넣지 않았다.

### 1. 최초 deterministic smoke test

처음에는 noise와 TD3 없이 한 episode만 짧게 실행해 pose 합성, hand publish,
camera/robot observation, F/T stop을 확인한다.

```bash
cd /home/baetae/diffusion-policy

MAX_FORCE_N=APPROVED_FORCE_NORM \
MAX_TORQUE_NM=APPROVED_TORQUE_NORM \
MAX_OBS_AGE_S=APPROVED_MAX_AGE \
bash diffusion_policy/residual_rl/run_insert_box_interactive.sh \
  --exploration-mode none \
  --episodes 1 \
  --train-after-episodes 999999 \
  --max-duration 20
```

### 2. Interactive residual RL

Smoke test가 정상일 때 기본 ResFiT exploration과 자동 TD3를 켠다.

```bash
MAX_FORCE_N=APPROVED_FORCE_NORM \
MAX_TORQUE_NM=APPROVED_TORQUE_NORM \
MAX_OBS_AGE_S=APPROVED_MAX_AGE \
bash diffusion_policy/residual_rl/run_insert_box_interactive.sh
```

메뉴와 rollout 조작:

- 메뉴에서 `Enter` 또는 `r`: 한 episode 시작
- rollout 중 terminal에서 `s`, `q`, `Space`, 또는 `Enter`: command 생성 종료
- terminal이 정상 모드로 복구된 뒤 `1` 성공 / `0` 실패 입력
- 메뉴에서 `t`: 지금까지 저장된 성공·실패 episode 전부로 즉시 TD3
- 메뉴에서 `q`: session 종료

기본은 새 episode 5개와 최소 256 online transition이 모이면 자동 TD3를 시작한다.
이때 robot environment/controller/camera recording을 먼저 닫고 base/actor GPU
객체를 해제한 다음 optimizer를 만든다. TD3가 끝나면 새 checkpoint를 episode
사이에만 load하고 다시 rollout 메뉴로 돌아온다. 한 episode 도중 checkpoint가
바뀌거나 robot command와 optimizer update가 동시에 실행되는 경로는 없다.

학습 update 수는 기본적으로 `UTD=4`를 사용한다. 예를 들어 마지막 학습 이후
새 command transition이 2,000개면 robot을 닫은 뒤 `2,000 × 4 = 8,000`번
gradient update한다. 재현/debug 목적으로만 `TD3_UPDATES`를 주면 고정 update
수로 덮어쓸 수 있다.

UTD는 update가 robot step과 같은 시각에 실행되어야 한다는 뜻이 아니라, 새로
수집한 environment transition 수에 대한 optimizer update 수의 비율이다.
ResFiT 실로봇 시스템은 actor와 learner를 별도 process로 두고, 한 trajectory가
끝난 뒤 learner가 최신 replay로 학습하는 동안 actor 쪽에서 scene reset과 다음
trajectory 수집을 진행해 wall-clock 시간을 줄였다. 이 구현은 한 GPU에서 base
inference와 image TD3 backward가 경쟁해 control latency나 OOM을 일으키지 않도록
`collect -> robot close -> UTD updates -> actor reload`를 직렬화한다. 동일한 replay와
동일한 총 update 수를 사용하면 TD3의 목적함수 자체는 같지만, 병렬 방식은 수집
actor가 learner보다 잠시 오래된 checkpoint를 사용할 수 있어 데이터 순서와
분포가 완전히 동일하지는 않다.

값을 바꾸려면 environment variable 또는 뒤쪽 CLI 옵션을 사용한다.

```bash
TRAIN_AFTER_EPISODES=3 UTD_RATIO=4 REPLAY_CAPACITY=10000 \
MAX_FORCE_N=... MAX_TORQUE_NM=... MAX_OBS_AGE_S=... \
bash diffusion_policy/residual_rl/run_insert_box_interactive.sh \
  --max-duration 45
```

Exploration은 ResFiT schedule이 기본이며 base action에는 절대 noise를 넣지
않는다. 우리 actor는 0 residual이 아니라 actual-to-virtual BC에서 시작하므로,
noise는 그 BC residual을 대체하지 않고 주변에만 더한다.

```text
online transition < 10,000:
  residual + Uniform(-0.2, 0.2) * per-axis physical half-range

그 이후:
  residual + Normal(0, 0.025) * per-axis physical half-range
```

모든 결과는 checkpoint residual limit으로 다시 clip된다. Deterministic 비교가
필요할 때만 exploration을 끈다.

원 ResFiT은 residual actor가 0에서 시작하므로 warmup에서 `base + noise`를
실행한다. 여기서는 이미 학습된 actual-to-virtual residual을 버리지 않도록
`base + frozen/learned fast residual + noise`로 바꾼 것이다. 10,000은 exploration
phase 전환점이며, interactive TD3 시작 조건은 별도 `--min-replay-transitions`로
조절한다.

```bash
MAX_FORCE_N=... MAX_TORQUE_NM=... MAX_OBS_AGE_S=... \
bash diffusion_policy/residual_rl/run_insert_box_interactive.sh \
  --exploration-mode none
```

기존 물리 단위 Gaussian을 별도로 시험하려면
`--exploration-mode gaussian --exploration-std 0.001`을 사용한다.

### Session 결과와 재시작

기본 session root는 `data/results/residual_rl_insert_box_interactive`이다.

```text
residual_rl_episodes/episode_*.hdf5
residual_rl_training/round_0001/checkpoints/latest.pt
residual_rl_training/round_0002/checkpoints/latest.pt
residual_rl_training/session_state.json
```

각 TD3 round에는 `latest.pt` 하나만 저장한다. 중단 후에는 마지막 TD3 actor와
원래 frozen BC prior를 둘 다 명시하고 같은 session root로 재실행한다.

```bash
ACTOR_CHECKPOINT=data/results/residual_rl_insert_box_interactive/residual_rl_training/round_0002/checkpoints/latest.pt \
BC_CHECKPOINT=data/outputs/2026.07.15_residual_rl/fast_bc_16step/checkpoints/best.pt \
SESSION_OUTPUT=data/results/residual_rl_insert_box_interactive \
MAX_FORCE_N=... MAX_TORQUE_NM=... MAX_OBS_AGE_S=... \
bash diffusion_policy/residual_rl/run_insert_box_interactive.sh
```

Resume은 actor/target/critic/optimizer/global update를 이어간다. Gamma, tau,
learning rate, policy delay, target noise, `lambda_bc`가 이전 round와 다르면
optimizer 의미가 바뀌므로 시작 전에 거부한다.

## TD3 정의와 replay

성공과 실패 episode를 모두 저장한다. Episode reward/done은 다음과 같다.

```text
success: reward [0, ..., 0, 1], done [0, ..., 0, 1]
failure: reward [0, ..., 0, 0], done [0, ..., 0, 1]
```

Actor loss는 다음과 같다.

```text
-Q1(obs, base_action, actor(obs, base_action))
+ lambda_bc * MSE(
    normalize_residual(actor(obs, base_action)),
    normalize_residual(frozen_BC_prior(obs, base_action))
  )
```

Prior MSE는 translation(m)과 rotation(rad)의 물리 크기 차이가 loss 가중치로
섞이지 않도록 BC와 동일한 per-axis `[-1,1]` 공간에서 계산한다. Frozen prior는
reference만 제공하며 gradient나 optimizer update를 받지 않는다.

이 loss는 **최소화**한다. 따라서 `-Q1`은 actor가 더 큰 Q를 갖는 action을
선택하게 하고, `+ lambda_bc * MSE`는 frozen BC action에서 멀어질수록 비용을
더한다. 부호가 `- lambda_bc * MSE`이면 오히려 BC prior에서 멀어지도록
학습되므로 여기서는 반드시 `+`가 맞다. `lambda_bc=0`이면 순수 TD3 actor
objective이고, 값을 키울수록 기존 actual-to-virtual BC residual을 더 보수적으로
유지한다.

Twin critic, delayed actor update, target policy smoothing, Polyak update와
critic warmup을 사용한다. Base policy와 frozen BC prior는 학습하지 않는다.
이번 v1은 visual encoder도 RL에서 고정한다.

TD target은 기본 3-step return이다. Raw episode의 terminal reward 표기는 그대로
보존하고 replay load 시에만 마지막 성공 신호를 최대 세 transition 앞으로
전파한다.

매 batch는 기본적으로 정확히 50:50으로 섞는다.

```text
offline 50%: paired 성공 demonstration
  obs + cached frozen-base action + actual-to-virtual residual

online 50%: 실제 robot의 성공 및 실패 episode 전체
```

Offline demo HDF5는 lazy sampling하므로 20k 이미지 transition 전체를 별도 RAM
replay에 복사하지 않는다.

Replay 기본 cap 20,000은 canonical obs/next_obs 기준 약 5.8 GiB host RAM이다.
실제 저장 transition 수가 더 적으면 그 수만큼만 할당하고, cap을 넘으면 최신
transition을 유지한다. 60초 10 Hz episode는 이미지 압축 전 약 181 MB이므로
장시간 수집 시 disk 공간을 확인한다.

## 독립 collect / train / evaluate

Interactive 대신 단계를 따로 실행할 수도 있다. `collect`와 `evaluate`는 같은
필수 safety 인자를 받는다. `evaluate`는 exploration을 항상 0으로 강제한다.

```bash
conda run --no-capture-output -n robodiff \
  python -m diffusion_policy.residual_rl.collect \
  --actor-checkpoint data/outputs/2026.07.15_residual_rl/fast_bc_16step/checkpoints/best.pt \
  --base-checkpoint data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt \
  --output data/results/residual_rl_collect \
  --episodes 10 \
  --max-force-norm-n APPROVED_FORCE_NORM \
  --max-torque-norm-nm APPROVED_TORQUE_NORM \
  --max-observation-age-s APPROVED_MAX_AGE \
  --control-mode impedance --device cuda:0
```

```bash
conda run --no-capture-output -n robodiff \
  python -m diffusion_policy.residual_rl.train \
  --bc-checkpoint data/outputs/2026.07.15_residual_rl/fast_bc_16step/checkpoints/best.pt \
  --episodes data/results/residual_rl_collect/residual_rl_episodes \
  --output data/outputs/2026.07.15_residual_rl/td3_round_0001 \
  --offline-actual-dataset data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5 \
  --offline-virtual-dataset data/baetae/260715_insert_box_hand/diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5 \
  --offline-base-predictions data/outputs/2026.07.15_residual_rl/base_predictions.hdf5 \
  --offline-ratio 0.5 --n-step 3 \
  --lambda-bc 1.0 --save-every 0 --device cuda:0
```

```bash
conda run --no-capture-output -n robodiff \
  python -m diffusion_policy.residual_rl.evaluate \
  --actor-checkpoint data/outputs/2026.07.15_residual_rl/td3_round_0001/checkpoints/latest.pt \
  --base-checkpoint data/outputs/2026.07.15_residual_policy/insert_box_slow_policy/epoch=0900-train_loss=0.003.ckpt \
  --output data/results/residual_rl_eval \
  --episodes 10 \
  --max-force-norm-n APPROVED_FORCE_NORM \
  --max-torque-norm-nm APPROVED_TORQUE_NORM \
  --max-observation-age-s APPROVED_MAX_AGE \
  --control-mode impedance --device cuda:0
```

## 다른 컴퓨터로 옮기기

필수로 복사할 항목:

- 이 repository의 현재 코드
- actual/virtual HDF5 두 파일
- frozen base checkpoint
- 생성된 `base_predictions.hdf5`와 `fast_bc_16step/checkpoints/best.pt`
- 이어서 학습할 경우 기존 session의 `residual_rl_episodes`와 마지막 TD3 checkpoint

현재 검증 환경은 Python 3.10.13, PyTorch 2.9.1+cu128, torchvision
0.24.1+cu128, NumPy 1.23.5, h5py 3.7.0이다. 기존
`conda_environment_real.yaml`의 Python 3.9 / Torch 1.12 / CUDA 11.6 조합은
RTX 5070과 현재 residual code의 기준 환경이 아니므로 그대로 재생성하지 않는다.
기존 real-robot dependency 환경을 Python 3.10으로 만든 뒤 Torch/torchvision을
위 CUDA 12.8 build로 맞추고, base policy가 사용하는 PyTorch3D도 해당 Torch
버전에 맞는 build인지 확인한다.

설치 직후 먼저 확인한다.

```bash
conda run -n robodiff python -c \
  "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

conda run -n robodiff python -m compileall -q diffusion_policy/residual_rl
conda run -n robodiff python -m unittest discover \
  -s diffusion_policy/residual_rl/test -p 'test_*.py' -q
conda run -n robodiff python -m diffusion_policy.residual_rl.tests.test_runtime
```

## 실로봇 전 필수 확인

코드는 NaN/Inf, stale camera/robot observation, force/torque 초과, malformed
action을 command 전에 fail-closed 처리하고 safety violation 시 controller
`stop(wait=False)`를 호출한다. 그렇더라도 ROS/hardware가 없는 개발 환경에서는
실제 정지와 topic 배선까지 검증할 수 없다.

- arm feedback `/dsr01/joint_states`, hand feedback `/joint_states` 확인
- hand command `/aidin_dualarm_joint_controller/joint_state_command` 확인
- hand7 순서 `[thumb1, thumb2, thumb3, index2, index3, middle2, middle3]` 확인
- full-30 hand command에서 미선택/왼손에 0을 보내는 것이 현재 controller에서 안전한지 확인
- 승인된 force norm, torque norm, observation age 값을 저속/무부하 시험으로 확인
- demonstration의 1.3배 envelope가 robot/controller의 실제 물리 limit보다 넓지 않은지 확인

실제 robot에서는 첫 episode를 exploration 없이 짧은 `--max-duration`으로 시작하고,
안전 정지 동작과 hand publish를 확인한 뒤 반복 수집으로 넘어간다.
