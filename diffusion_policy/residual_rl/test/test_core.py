import copy

import numpy as np
import torch

from diffusion_policy.residual_rl.models import (
    ConvImageEncoder,
    ResidualActor,
    StructuredObservationEncoder,
    TwinQCritic,
)
from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer
from diffusion_policy.residual_rl.replay_buffer import ResidualReplayBuffer
from diffusion_policy.residual_rl.episode_io import make_n_step_episode
from diffusion_policy.residual_rl.td3 import ResidualTD3, TD3Config


OBS_SHAPES = {
    "image0": (3, 16, 16),
    "robot_pose_R": (3,),
    "robot_quat_R": (4,),
    "hand_pose_R": (7,),
    "wrench_wrist_R": (6, 32),
}
RESIDUAL_MIN = np.array([-0.02, -0.02, -0.01, -0.1, -0.1, -0.1], dtype=np.float32)
RESIDUAL_MAX = -RESIDUAL_MIN


def make_obs(batch_size=4, uint8_image=False):
    image = torch.rand(batch_size, *OBS_SHAPES["image0"])
    if uint8_image:
        image = torch.round(image * 255).to(torch.uint8)
    return {
        "image0": image,
        "robot_pose_R": torch.randn(batch_size, 3),
        "robot_quat_R": torch.randn(batch_size, 4),
        "hand_pose_R": torch.randn(batch_size, 7),
        "wrench_wrist_R": torch.randn(batch_size, 6, 32),
    }


def make_encoder(freeze_image=True):
    return StructuredObservationEncoder(
        image_encoder=ConvImageEncoder(output_dim=8),
        image_feature_dim=8,
        wrench_feature_dim=8,
        freeze_image_encoder=freeze_image,
    )


def make_actor():
    return ResidualActor(
        state_encoder=make_encoder(freeze_image=True),
        residual_min=RESIDUAL_MIN,
        residual_max=RESIDUAL_MAX,
        hidden_dims=(32, 16),
    )


def make_critics():
    return TwinQCritic(
        state_encoder=make_encoder(freeze_image=True),
        hidden_dims=(32, 16),
    )


def numpy_episode(length, success, seed):
    rng = np.random.default_rng(seed)
    reward = np.zeros(length, dtype=np.float32)
    reward[-1] = float(success)
    done = np.zeros(length, dtype=np.float32)
    done[-1] = 1.0
    return {
        "obs": {
            "image0": rng.integers(0, 256, (length,) + OBS_SHAPES["image0"], dtype=np.uint8),
            "robot_pose_R": rng.normal(size=(length, 3)).astype(np.float32),
            "robot_quat_R": rng.normal(size=(length, 4)).astype(np.float32),
            "hand_pose_R": rng.normal(size=(length, 7)).astype(np.float32),
            "wrench_wrist_R": rng.normal(size=(length, 6, 32)).astype(np.float32),
        },
        "next_obs": {
            "image0": rng.integers(0, 256, (length,) + OBS_SHAPES["image0"], dtype=np.uint8),
            "robot_pose_R": rng.normal(size=(length, 3)).astype(np.float32),
            "robot_quat_R": rng.normal(size=(length, 4)).astype(np.float32),
            "hand_pose_R": rng.normal(size=(length, 7)).astype(np.float32),
            "wrench_wrist_R": rng.normal(size=(length, 6, 32)).astype(np.float32),
        },
        "base_action": rng.normal(size=(length, 16)).astype(np.float32),
        "next_base_action": rng.normal(size=(length, 16)).astype(np.float32),
        "action": rng.uniform(RESIDUAL_MIN, RESIDUAL_MAX, size=(length, 6)).astype(np.float32),
        "reward": reward,
        "done": done,
    }


def test_actor_critic_shapes_bounds_uint8_and_visual_freeze():
    actor = make_actor()
    actor.train()
    assert not actor.state_encoder.image_encoder.training
    assert not any(
        parameter.requires_grad
        for parameter in actor.state_encoder.image_encoder.parameters()
    )

    obs = make_obs(batch_size=4, uint8_image=True)
    base_action = torch.randn(4, 16)
    action = actor(obs, base_action)
    assert action.shape == (4, 6)
    assert torch.all(action >= torch.as_tensor(RESIDUAL_MIN))
    assert torch.all(action <= torch.as_tensor(RESIDUAL_MAX))

    critics = make_critics()
    q1, q2 = critics(obs, base_action, action)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)


def test_normalizer_broadcasts_vector_and_wrench_fields():
    normalizer = StructuredAffineNormalizer({
        "robot_pose_R": {"scale": [2, 3, 4], "offset": [1, 1, 1]},
        "wrench_wrist_R": {
            "scale": np.full(6 * 32, 2.0, dtype=np.float32),
            "offset": np.zeros(6 * 32, dtype=np.float32),
        },
    })
    pose = torch.ones(2, 5, 3)
    wrench = torch.ones(2, 5, 6, 32)
    assert torch.equal(
        normalizer.normalize_field("robot_pose_R", pose)[0, 0],
        torch.tensor([3.0, 4.0, 5.0]),
    )
    assert torch.all(normalizer.normalize_field("wrench_wrist_R", wrench) == 2)


def test_replay_canonical_episodes_keep_success_and_failure_and_roundtrip():
    replay = ResidualReplayBuffer(
        capacity=8,
        observation_shapes=OBS_SHAPES,
        seed=7,
    )
    replay.add_episode(numpy_episode(length=3, success=True, seed=1))
    replay.add_episode(numpy_episode(length=2, success=False, seed=2))
    assert len(replay) == 5
    state = replay.state_dict()
    assert float(state["data"]["reward"].sum()) == 1.0
    assert float(state["data"]["done"].sum()) == 2.0

    batch = replay.sample(4)
    assert set(batch) == {
        "obs",
        "next_obs",
        "base_action",
        "next_base_action",
        "action",
        "reward",
        "done",
    }
    assert batch["obs"]["image0"].dtype == torch.float32
    assert torch.all((batch["obs"]["image0"] >= 0) & (batch["obs"]["image0"] <= 1))
    assert batch["base_action"].shape == (4, 16)
    assert batch["action"].shape == (4, 6)

    restored = ResidualReplayBuffer(
        capacity=8,
        observation_shapes=OBS_SHAPES,
        seed=99,
    )
    restored.load_state_dict(state)
    restored_state = restored.state_dict()
    assert len(restored) == len(replay)
    np.testing.assert_array_equal(
        restored_state["data"]["action"], state["data"]["action"]
    )
    np.testing.assert_array_equal(
        restored_state["data"]["reward"], state["data"]["reward"]
    )


def test_labeled_episode_assigns_only_terminal_reward():
    replay = ResidualReplayBuffer(4, OBS_SHAPES)
    rng = np.random.default_rng(3)
    observations = {
        "image0": rng.random((4,) + OBS_SHAPES["image0"], dtype=np.float32),
        "robot_pose_R": rng.normal(size=(4, 3)).astype(np.float32),
        "robot_quat_R": rng.normal(size=(4, 4)).astype(np.float32),
        "hand_pose_R": rng.normal(size=(4, 7)).astype(np.float32),
        "wrench_wrist_R": rng.normal(size=(4, 6, 32)).astype(np.float32),
    }
    replay.add_labeled_episode(
        observations=observations,
        base_actions=rng.normal(size=(4, 16)).astype(np.float32),
        actions=rng.uniform(RESIDUAL_MIN, RESIDUAL_MAX, size=(3, 6)).astype(np.float32),
        success=1,
    )
    data = replay.state_dict()["data"]
    np.testing.assert_array_equal(data["reward"].reshape(-1), [0, 0, 1])
    np.testing.assert_array_equal(data["done"].reshape(-1), [0, 0, 1])


def test_n_step_episode_propagates_terminal_reward_without_crossing_boundary():
    episode = numpy_episode(length=5, success=True, seed=11)
    for key in episode["next_obs"]:
        shape = episode["next_obs"][key].shape
        episode["next_obs"][key] = np.broadcast_to(
            np.arange(5, dtype=np.float32).reshape((5,) + (1,) * (len(shape) - 1)),
            shape,
        ).copy()
    episode["next_base_action"] = np.broadcast_to(
        np.arange(5, dtype=np.float32)[:, None],
        (5, 16),
    ).copy()

    transformed = make_n_step_episode(episode, n_step=3, gamma=0.9)
    np.testing.assert_allclose(
        transformed["reward"],
        [0.0, 0.0, 0.9**2, 0.9, 1.0],
    )
    np.testing.assert_array_equal(transformed["done"], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        transformed["next_base_action"][:, 0],
        [2, 3, 4, 4, 4],
    )


def test_td3_delayed_actor_update_prior_freeze_polyak_and_checkpoint(tmp_path):
    torch.manual_seed(5)
    actor = make_actor()
    prior = copy.deepcopy(actor)
    critics = make_critics()
    td3 = ResidualTD3(
        actor=actor,
        critics=critics,
        bc_prior=prior,
        config=TD3Config(
            gamma=0.95,
            tau=0.5,
            actor_lr=1e-3,
            critic_lr=1e-3,
            policy_delay=2,
            target_policy_noise=[0.002] * 6,
            target_noise_clip=[0.003] * 6,
            lambda_bc=0.25,
        ),
    )
    replay = ResidualReplayBuffer(16, OBS_SHAPES, seed=4)
    replay.add_episode(numpy_episode(length=8, success=True, seed=8))
    batch = replay.sample(6)

    prior_before = {
        key: value.detach().clone() for key, value in td3.bc_prior.state_dict().items()
    }
    actor_before = {
        key: value.detach().clone() for key, value in td3.actor.state_dict().items()
    }
    target_before = {
        key: value.detach().clone()
        for key, value in td3.target_actor.state_dict().items()
    }
    first_metrics = td3.update(batch)
    assert first_metrics["actor_updated"] is False
    second_metrics = td3.update(batch)
    assert second_metrics["actor_updated"] is True
    assert second_metrics["bc_loss"] >= 0.0
    assert any(
        not torch.equal(actor_before[key], value)
        for key, value in td3.actor.state_dict().items()
        if value.is_floating_point()
    )
    assert any(
        not torch.equal(target_before[key], value)
        for key, value in td3.target_actor.state_dict().items()
        if value.is_floating_point()
    )
    for key, value in td3.bc_prior.state_dict().items():
        assert torch.equal(value, prior_before[key])
    assert all(not parameter.requires_grad for parameter in td3.bc_prior.parameters())

    noisy_action = td3.act(
        make_obs(batch_size=3),
        torch.randn(3, 16),
        exploration_std=[10.0] * 6,
    )
    assert torch.all(noisy_action >= torch.as_tensor(RESIDUAL_MIN))
    assert torch.all(noisy_action <= torch.as_tensor(RESIDUAL_MAX))

    checkpoint_path = tmp_path / "td3.pt"
    td3.save_checkpoint(checkpoint_path, extra={"round": 2})
    restored = ResidualTD3(
        actor=make_actor(),
        critics=make_critics(),
        config=copy.deepcopy(td3.config),
    )
    extra = restored.load_checkpoint(checkpoint_path)
    assert extra == {"round": 2}
    assert restored.total_updates == td3.total_updates
    for key, value in td3.actor.state_dict().items():
        assert torch.equal(restored.actor.state_dict()[key], value)
