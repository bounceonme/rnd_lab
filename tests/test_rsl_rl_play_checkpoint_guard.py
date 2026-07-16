from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPTS_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
PLAY_PATH = RSL_RL_SCRIPTS_DIR / "play.py"
sys.path.insert(0, str(RSL_RL_SCRIPTS_DIR))

from play_checkpoint_guard import CheckpointExperimentError, validate_checkpoint_experiment


class RslRlPlayCheckpointGuardTest(unittest.TestCase):
    def _make_checkpoint(self, root: Path, agent_yaml: str | None) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        checkpoint_path = run_dir / "model_100.pt"
        checkpoint_path.touch()
        if agent_yaml is not None:
            params_dir = run_dir / "params"
            params_dir.mkdir()
            (params_dir / "agent.yaml").write_text(agent_yaml, encoding="utf-8")
        return checkpoint_path

    def test_matching_saved_experiment_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = self._make_checkpoint(
                Path(temporary_directory),
                "experiment_name: rnd_step/flat_actuator\n",
            )

            saved_experiment = validate_checkpoint_experiment(
                checkpoint_path,
                "rnd_step/flat_actuator",
            )

        self.assertEqual(saved_experiment, "rnd_step/flat_actuator")

    def test_mismatched_saved_experiment_fails_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = self._make_checkpoint(
                Path(temporary_directory),
                "experiment_name: rnd_step/flat\n",
            )

            with self.assertRaisesRegex(
                CheckpointExperimentError,
                "trained for 'rnd_step/flat'.*selected task uses 'rnd_step/flat_actuator'",
            ):
                validate_checkpoint_experiment(checkpoint_path, "rnd_step/flat_actuator")

    def test_missing_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = self._make_checkpoint(Path(temporary_directory), None)

            with self.assertRaisesRegex(CheckpointExperimentError, "saved agent configuration is missing"):
                validate_checkpoint_experiment(checkpoint_path, "rnd_step/flat_actuator")

    def test_explicit_mismatch_opt_out_bypasses_metadata_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = self._make_checkpoint(Path(temporary_directory), None)

            saved_experiment = validate_checkpoint_experiment(
                checkpoint_path,
                "rnd_step/flat_actuator",
                allow_task_mismatch=True,
            )

        self.assertIsNone(saved_experiment)

    def test_play_runs_guard_before_creating_environment(self):
        play_source = PLAY_PATH.read_text(encoding="utf-8")
        guard_call_offset = play_source.index("    validate_checkpoint_experiment(\n")
        gym_make_offset = play_source.index("    env = gym.make(")

        self.assertLess(guard_call_offset, gym_make_offset)
        self.assertIn('"--allow_task_mismatch"', play_source)

    def test_observation_corruption_and_stateful_sensor_randomization_are_opt_in(self):
        play_source = PLAY_PATH.read_text(encoding="utf-8")

        self.assertIn('"--enable_observation_corruption"', play_source)
        self.assertIn(
            "env_cfg.observations.policy.enable_corruption = args_cli.enable_observation_corruption",
            play_source,
        )
        self.assertIn(
            'term_params["sample_randomization"] = args_cli.enable_observation_corruption',
            play_source,
        )


if __name__ == "__main__":
    unittest.main()
