"""Standard-library runner for environments where pytest is not installed."""

import tempfile
import unittest
from pathlib import Path

from diffusion_policy.residual_rl.cache_base_predictions import _chunk_assignments
from diffusion_policy.residual_rl.test.test_core import (
    test_actor_critic_shapes_bounds_uint8_and_visual_freeze,
    test_labeled_episode_assigns_only_terminal_reward,
    test_normalizer_broadcasts_vector_and_wrench_fields,
    test_n_step_episode_propagates_terminal_reward_without_crossing_boundary,
    test_replay_canonical_episodes_keep_success_and_failure_and_roundtrip,
    test_td3_delayed_actor_update_prior_freeze_polyak_and_checkpoint,
)


class ResidualRLCoreTest(unittest.TestCase):
    def test_chunked_base_cache_schedule(self):
        assignments = _chunk_assignments(
            15,
            target_offset=1,
            slow_action_start_index=1,
            fast_steps_per_slow=6,
        )
        self.assertEqual(
            assignments[:6],
            [
                (0, 1, 1),
                (0, 2, 2),
                (0, 3, 3),
                (0, 4, 4),
                (0, 5, 5),
                (0, 6, 6),
            ],
        )
        self.assertEqual(assignments[6], (6, 7, 1))
        self.assertEqual(assignments[-1], (12, 14, 2))

    def test_models(self):
        test_actor_critic_shapes_bounds_uint8_and_visual_freeze()

    def test_normalizer(self):
        test_normalizer_broadcasts_vector_and_wrench_fields()

    def test_replay(self):
        test_replay_canonical_episodes_keep_success_and_failure_and_roundtrip()

    def test_terminal_reward(self):
        test_labeled_episode_assigns_only_terminal_reward()

    def test_n_step_return(self):
        test_n_step_episode_propagates_terminal_reward_without_crossing_boundary()

    def test_td3(self):
        with tempfile.TemporaryDirectory() as directory:
            test_td3_delayed_actor_update_prior_freeze_polyak_and_checkpoint(
                Path(directory)
            )


if __name__ == "__main__":
    unittest.main()
