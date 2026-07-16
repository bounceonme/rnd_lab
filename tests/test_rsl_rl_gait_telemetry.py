from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPTS_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
PLAY_PATH = RSL_RL_SCRIPTS_DIR / "play.py"
sys.path.insert(0, str(RSL_RL_SCRIPTS_DIR))

from gait_telemetry import GaitTelemetryError, GaitTelemetryLogger, resolve_ordered_foot_names


class _FakeCommandManager:
    def __init__(self, command: np.ndarray) -> None:
        self.command = command

    def get_command(self, name: str) -> np.ndarray:
        if name != "base_velocity":
            raise KeyError(name)
        return self.command


def _make_fake_env() -> SimpleNamespace:
    num_envs = 2
    body_names = ["base", "L_Leg_foot", "R_Leg_foot"]
    sensor_body_names = ["base", "R_Leg_foot", "L_Leg_foot"]
    joint_names = ["left_joint", "right_joint"]

    robot_data = SimpleNamespace(
        root_pos_w=np.arange(num_envs * 3, dtype=np.float32).reshape(num_envs, 3),
        root_quat_w=np.arange(num_envs * 4, dtype=np.float32).reshape(num_envs, 4),
        root_lin_vel_w=np.full((num_envs, 3), 3.0, dtype=np.float32),
        root_ang_vel_w=np.full((num_envs, 3), 4.0, dtype=np.float32),
        body_pos_w=np.arange(num_envs * len(body_names) * 3, dtype=np.float32).reshape(num_envs, len(body_names), 3),
        body_quat_w=np.arange(num_envs * len(body_names) * 4, dtype=np.float32).reshape(
            num_envs, len(body_names), 4
        ),
        joint_pos=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        joint_vel=np.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32),
        applied_torque=np.asarray([[2.1, 2.2], [2.3, 2.4]], dtype=np.float32),
        computed_torque=np.asarray([[3.1, 3.2], [3.3, 3.4]], dtype=np.float32),
    )
    robot = SimpleNamespace(body_names=body_names, joint_names=joint_names, data=robot_data)

    net_forces_w = np.zeros((num_envs, len(sensor_body_names), 3), dtype=np.float32)
    net_forces_w[1, 1] = [0.0, 0.0, 10.0]
    contact_data = SimpleNamespace(
        net_forces_w=net_forces_w,
        current_air_time=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]], dtype=np.float32),
        last_air_time=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.3, 0.4]], dtype=np.float32),
        current_contact_time=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
        last_contact_time=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.6]], dtype=np.float32),
    )
    contact_sensor = SimpleNamespace(
        body_names=sensor_body_names,
        cfg=SimpleNamespace(force_threshold=1.0),
        data=contact_data,
    )
    return SimpleNamespace(
        num_envs=num_envs,
        scene={"robot": robot, "contact_forces": contact_sensor},
        command_manager=_FakeCommandManager(np.asarray([[0.0, 0.0, 0.0], [0.1, -0.5, 0.2]], dtype=np.float32)),
    )


class RslRlGaitTelemetryTest(unittest.TestCase):
    def test_foot_resolution_is_right_then_left_despite_scene_order(self):
        self.assertEqual(
            resolve_ordered_foot_names(["base", "L_Leg_foot", "R_Leg_foot"]),
            ("R_Leg_foot", "L_Leg_foot"),
        )

    def test_logger_records_ordered_post_step_state_and_pickle_free_npz(self):
        env = _make_fake_env()
        logger = GaitTelemetryLogger(
            env,
            task="RND-Step",
            checkpoint="model_5000.pt",
            env_index=1,
            step_dt=0.02,
        )
        actions = np.asarray([[0.0, 0.0], [0.7, -0.8]], dtype=np.float32)
        logger.record(actions, np.asarray([False, True]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "gait.telemetry"
            saved_path = logger.save(output_path)
            with np.load(saved_path, allow_pickle=False) as gait_log:
                self.assertEqual(gait_log["task"].item(), "RND-Step")
                self.assertEqual(gait_log["step_dt"].item(), 0.02)
                self.assertEqual(gait_log["env_index"].item(), 1)
                np.testing.assert_array_equal(gait_log["foot_order"], ["right", "left"])
                np.testing.assert_array_equal(gait_log["foot_body_names"], ["R_Leg_foot", "L_Leg_foot"])
                np.testing.assert_allclose(gait_log["command"], [[0.1, -0.5, 0.2]])
                np.testing.assert_allclose(gait_log["actions"], [[0.7, -0.8]])
                np.testing.assert_array_equal(gait_log["foot_contact"], [[True, False]])
                np.testing.assert_allclose(gait_log["foot_current_air_time"], [[0.0, 0.2]])
                np.testing.assert_allclose(gait_log["foot_current_contact_time"], [[0.1, 0.0]])
                self.assertTrue(gait_log["done"].item())

            self.assertEqual(saved_path, output_path.resolve())
            self.assertTrue(output_path.is_file())

    def test_logger_rejects_invalid_environment_index(self):
        env = _make_fake_env()
        for env_index in (-1, env.num_envs):
            with self.subTest(env_index=env_index):
                with self.assertRaisesRegex(GaitTelemetryError, "outside"):
                    GaitTelemetryLogger(
                        env,
                        task="RND-Step",
                        checkpoint="model.pt",
                        env_index=env_index,
                        step_dt=0.02,
                    )

    def test_ambiguous_foot_names_fail_closed(self):
        with self.assertRaisesRegex(GaitTelemetryError, "Could not resolve"):
            resolve_ordered_foot_names(["base", "foot_a", "foot_b"])

    def test_play_wires_post_step_record_and_saves_before_close(self):
        play_source = PLAY_PATH.read_text(encoding="utf-8")
        step_offset = play_source.index("            obs, _, dones, _ = env.step(actions)\n")
        record_offset = play_source.index("                gait_logger.record(actions, dones)\n")
        save_offset = play_source.index("        gait_log_path = gait_logger.save(args_cli.gait_log)\n")
        close_offset = play_source.index("    env.close()\n")

        self.assertLess(step_offset, record_offset)
        self.assertLess(record_offset, save_offset)
        self.assertLess(save_offset, close_offset)
        self.assertIn('"--gait_log"', play_source)
        self.assertIn('"--gait_log_env"', play_source)


if __name__ == "__main__":
    unittest.main()
