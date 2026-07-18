from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source"
    / "robot_lab"
    / "robot_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "imu_observations.py"
)


def _load_module_without_isaac_app():
    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    isaaclab_stub = types.ModuleType("isaaclab")
    managers_stub = types.ModuleType("isaaclab.managers")
    utils_stub = types.ModuleType("isaaclab.utils")
    math_stub = types.ModuleType("isaaclab.utils.math")

    def quat_apply(quat, vector):
        xyz = quat[..., 1:]
        intermediate = 2.0 * torch.cross(xyz, vector, dim=-1)
        return vector + quat[..., :1] * intermediate + torch.cross(xyz, intermediate, dim=-1)

    def quat_apply_inverse(quat, vector):
        inverse = quat.clone()
        inverse[..., 1:] *= -1.0
        return quat_apply(inverse, vector)

    managers_stub.ManagerTermBase = ManagerTermBase
    math_stub.quat_apply = quat_apply
    math_stub.quat_apply_inverse = quat_apply_inverse
    utils_stub.math = math_stub
    isaaclab_stub.managers = managers_stub
    isaaclab_stub.utils = utils_stub
    module_names = ("isaaclab", "isaaclab.managers", "isaaclab.utils", "isaaclab.utils.math")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules["isaaclab"] = isaaclab_stub
    sys.modules["isaaclab.managers"] = managers_stub
    sys.modules["isaaclab.utils"] = utils_stub
    sys.modules["isaaclab.utils.math"] = math_stub
    try:
        module_name = "_test_rnd_imu_observations"
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


imu = _load_module_without_isaac_app()


def _runtime_model() -> dict:
    return {
        "schema_version": 1,
        "model_type": "rnd_cmp10a_policy_observation",
        "integration_enabled": True,
        "policy_hz": 50.0,
        "policy_angular_velocity_scale": 0.25,
        "policy_observation": {"policy_hz": 50.0, "angular_velocity_scale": 0.25},
        "sensor_to_base_matrix": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        "runtime_transform": {
            "source_frame": "sensor",
            "target_frame": "base_link",
            "sensor_to_base_matrix": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "quality_gates": {
            "promotion_pass": True,
            "static_runtime_gate_pass": True,
            "static_mount_axis_gate_pass": True,
            "dynamic_communication_gate_pass": True,
            "dynamic_consistency_gate_pass": True,
        },
        "assumed_simulation_envelopes": {
            "gyro_sample_age_delay": {"range_s": [0.0, 0.005]},
            "residual_gyro_episode_bias": {
                "distribution": "uniform",
                "range_rad_s_per_axis": [-0.01, 0.01],
            },
            "gyro_white_noise": {
                "distribution": "zero_mean_gaussian",
                "sigma_range_rad_s": [0.0003, 0.003],
            },
            "orientation_delay": {"range_s": [0.0, 0.020]},
            "projected_gravity_tangent_angle_noise": {
                "distribution": "zero_mean_tangent_plane_gaussian",
                "sigma_range_rad": [0.00005, 0.002],
            },
        },
    }


class RndImuObservationModelTest(unittest.TestCase):
    def test_fractional_delay_interpolates_current_first_history(self):
        history = torch.tensor([
            [[10.0, 20.0], [6.0, 12.0], [2.0, 4.0]],
            [[-1.0, 3.0], [-5.0, 7.0], [-9.0, 11.0]],
        ])

        result = imu.fractional_delay_interpolate(history, torch.tensor([0.25, 1.5]))

        expected = torch.tensor([[9.0, 18.0], [-7.0, 9.0]])
        torch.testing.assert_close(result, expected)

    def test_partial_reset_refills_only_selected_history_and_state(self):
        torch.manual_seed(3)
        state = imu.Cmp10aObservationState(
            num_envs=2,
            channel="gyro",
            step_dt=0.02,
            delay_range_s=(0.0, 0.02),
            bias_range=(-0.01, 0.01),
            noise_sigma_range=(0.0003, 0.003),
            sample_randomization=True,
        )
        initial = torch.zeros((2, 3))
        state.reset(initial)
        state.observe(initial, 0)
        state.observe(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)
        env_zero_history = state.history[0].clone()
        env_zero_delay = state.delay_s[0].clone()
        env_zero_bias = state.bias[0].clone()
        env_zero_sigma = state.noise_sigma[0].clone()
        env_zero_cache = state.cached_result[0].clone()

        reset_raw = torch.tensor([[100.0, 100.0, 100.0], [9.0, 8.0, 7.0]])
        state.reset(reset_raw, torch.tensor([1]))

        torch.testing.assert_close(state.history[0], env_zero_history)
        torch.testing.assert_close(state.delay_s[0], env_zero_delay)
        torch.testing.assert_close(state.bias[0], env_zero_bias)
        torch.testing.assert_close(state.noise_sigma[0], env_zero_sigma)
        torch.testing.assert_close(state.history[1], reset_raw[1].view(1, 3).expand_as(state.history[1]))
        self.assertEqual(int(state.last_step_counter[0]), 1)
        self.assertEqual(int(state.last_step_counter[1]), imu._UNSEEN_STEP)

        result = state.observe(reset_raw, 1)
        torch.testing.assert_close(result[0], env_zero_cache)
        torch.testing.assert_close(state.history[0], env_zero_history)

    def test_deterministic_mode_uses_midpoint_delay_and_zero_noise(self):
        state = imu.Cmp10aObservationState(
            num_envs=1,
            channel="gyro",
            step_dt=0.02,
            delay_range_s=(0.0, 0.02),
            bias_range=(-1.0, 1.0),
            noise_sigma_range=(0.0003, 0.003),
            sample_randomization=False,
        )
        zeros = torch.zeros((1, 3))
        state.reset(zeros)
        state.observe(zeros, 0)

        result = state.observe(torch.full((1, 3), 2.0), 1)

        torch.testing.assert_close(state.delay_s, torch.tensor([0.01]))
        torch.testing.assert_close(state.bias, torch.zeros((1, 3)))
        torch.testing.assert_close(state.noise_sigma, torch.zeros(1))
        torch.testing.assert_close(result, torch.ones((1, 3)))

    def test_gravity_noise_is_tangent_plane_rotation_and_output_is_unit_norm(self):
        vectors = torch.tensor([[0.0, 0.0, -2.0], [2.0, 0.0, 0.0]])
        angular_noise = torch.tensor([[0.1, 0.2, 50.0], [40.0, 0.3, 0.4]])

        result = imu.apply_tangent_plane_angular_noise(vectors, angular_noise)

        torch.testing.assert_close(torch.linalg.vector_norm(result, dim=-1), torch.ones(2), atol=1.0e-7, rtol=0.0)
        expected_cosines = torch.cos(torch.tensor([(0.1**2 + 0.2**2) ** 0.5, (0.3**2 + 0.4**2) ** 0.5]))
        input_unit = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        torch.testing.assert_close(torch.sum(result * input_unit, dim=-1), expected_cosines, atol=1.0e-6, rtol=0.0)
        radial_only = imu.apply_tangent_plane_angular_noise(
            torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[0.0, 0.0, 100.0]])
        )
        torch.testing.assert_close(radial_only, torch.tensor([[0.0, 0.0, -1.0]]))

    def test_duplicate_step_is_idempotent_and_returns_a_clone(self):
        torch.manual_seed(7)
        state = imu.Cmp10aObservationState(
            num_envs=2,
            channel="gyro",
            step_dt=0.02,
            delay_range_s=(0.0, 0.005),
            bias_range=(-0.01, 0.01),
            noise_sigma_range=(0.0003, 0.003),
            sample_randomization=True,
        )
        raw = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        state.reset(raw)
        first = state.observe(raw, 42)
        expected = first.clone()
        history = state.history.clone()
        first.fill_(1234.0)

        duplicate = state.observe(raw + 100.0, 42)

        torch.testing.assert_close(duplicate, expected)
        torch.testing.assert_close(state.cached_result, expected)
        torch.testing.assert_close(state.history, history)

    def test_loader_accepts_clear_generated_aliases_and_rejects_bad_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmp10a.json"
            path.write_text(json.dumps(_runtime_model()), encoding="utf-8")

            model = imu.load_rnd_cmp10a_observation_model(path)

            self.assertEqual(model.policy_hz, 50.0)
            self.assertEqual(model.policy_angular_velocity_scale, 0.25)
            self.assertEqual(model.gyro_delay_range_s, (0.0, 0.005))
            self.assertEqual(model.gravity_delay_range_s, (0.0, 0.020))
            self.assertEqual(
                model.sensor_to_base_matrix,
                ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            )

            invalid = _runtime_model()
            invalid["assumed_simulation_envelopes"]["gyro_sample_age_delay"]["range_s"][1] = 0.006
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(imu.RndCmp10aObservationModelError, "supported gyro delay envelope"):
                imu.load_rnd_cmp10a_observation_model(path)

            failed_gate = _runtime_model()
            failed_gate["quality_gates"]["promotion_pass"] = False
            path.write_text(json.dumps(failed_gate), encoding="utf-8")
            with self.assertRaisesRegex(imu.RndCmp10aObservationModelError, "promotion_pass"):
                imu.load_rnd_cmp10a_observation_model(path)

    def test_default_loader_accepts_promoted_runtime_artifact(self):
        model = imu.load_rnd_cmp10a_observation_model()

        self.assertEqual(model.path.name, "rnd_cmp10a_runtime.json")
        self.assertEqual(model.policy_hz, 50.0)

    def test_manager_term_reads_articulation_data_and_validates_policy_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmp10a.json"
            path.write_text(json.dumps(_runtime_model()), encoding="utf-8")
            data = SimpleNamespace(
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]),
                body_ang_vel_w=torch.zeros((1, 2, 3)),
                GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]]),
            )
            asset = SimpleNamespace(data=data, body_names=["Upper_Body", "imu"])
            asset.find_bodies = lambda body_name, preserve_order=False: (
                ([1], ["imu"]) if body_name == "imu" else ([], [])
            )
            env = SimpleNamespace(
                num_envs=1,
                device="cpu",
                step_dt=0.02,
                common_step_counter=0,
                scene={"robot": asset},
            )
            params = {
                "channel": "gyro",
                "model_path": str(path),
                "sample_randomization": False,
                "body_name": "imu",
            }
            term = imu.RndCmp10aObservation(SimpleNamespace(params=params, scale=0.25), env)
            term(env, **params)
            data.body_ang_vel_w[:, 1].fill_(8.0)
            env.common_step_counter = 1

            result = term(env, **params)

            torch.testing.assert_close(result, torch.full((1, 3), 7.0))
            self.assertEqual(term.body_name, "imu")
            self.assertEqual(term.body_id, 1)

            env.step_dt = 0.01
            with self.assertRaisesRegex(ValueError, "1/env.step_dt=100.0"):
                imu.RndCmp10aObservation(SimpleNamespace(params=params, scale=0.25), env)


if __name__ == "__main__":
    unittest.main()
