from __future__ import annotations

import copy
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
    / "encoder_observations.py"
)


def _load_module_without_isaac_app():
    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    isaaclab_stub = types.ModuleType("isaaclab")
    managers_stub = types.ModuleType("isaaclab.managers")
    managers_stub.ManagerTermBase = ManagerTermBase
    isaaclab_stub.managers = managers_stub
    module_names = ("isaaclab", "isaaclab.managers")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules["isaaclab"] = isaaclab_stub
    sys.modules["isaaclab.managers"] = managers_stub
    try:
        module_name = "_test_rnd_encoder_observations"
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


encoder = _load_module_without_isaac_app()
JOINT_COUNT = 12


def _state(*, num_envs: int = 1, sample_randomization: bool = False):
    return encoder.DynamixelEncoderObservationState(
        num_envs=num_envs,
        num_joints=JOINT_COUNT,
        step_dt=0.02,
        position_quantum_rad=encoder.DYNAMIXEL_POSITION_QUANTUM_RAD,
        velocity_quantum_rad_s=encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S,
        velocity_scale=0.05,
        sample_age_range_s=(0.0, 0.005),
        zero_offset_range_rad=(-0.005, 0.005),
        sample_randomization=sample_randomization,
    )


def _zeros(num_envs: int = 1):
    return torch.zeros((num_envs, JOINT_COUNT), dtype=torch.float32)


class RndEncoderObservationModelTest(unittest.TestCase):
    def test_default_model_schema_and_path_are_strict(self):
        model = encoder.load_rnd_dynamixel_encoder_observation_model()
        short_name_model = encoder.load_rnd_encoder_observation_model()

        self.assertEqual(model.path, encoder.RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH.resolve())
        self.assertEqual(short_name_model, model)
        self.assertEqual(model.joint_order, encoder.RND_DYNAMIXEL_ENCODER_JOINT_ORDER)
        self.assertEqual(model.output_dimension, 24)
        self.assertEqual(model.sample_age_range_s, (0.0, 0.005))
        self.assertEqual(model.zero_offset_range_rad, (-0.005, 0.005))
        self.assertLess(max(abs(value) for value in model.zero_offset_range_rad), 0.01)

        payload = json.loads(model.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["zero_offset"]["source"], "training_prior_not_measured")
        self.assertEqual(payload["zero_offset"]["quality"], "assumed_for_training_only")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.json"
            unknown = copy.deepcopy(payload)
            unknown["unexpected"] = True
            path.write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(encoder.RndDynamixelEncoderObservationModelError, "unexpected"):
                encoder.load_rnd_dynamixel_encoder_observation_model(path)

            widened = copy.deepcopy(payload)
            widened["zero_offset"]["range_rad"] = [-0.01, 0.01]
            path.write_text(json.dumps(widened), encoding="utf-8")
            with self.assertRaisesRegex(encoder.RndDynamixelEncoderObservationModelError, "zero_offset.range_rad"):
                encoder.load_rnd_dynamixel_encoder_observation_model(path)

            measured_claim = copy.deepcopy(payload)
            measured_claim["zero_offset"]["source"] = "measured"
            path.write_text(json.dumps(measured_claim), encoding="utf-8")
            with self.assertRaisesRegex(
                encoder.RndDynamixelEncoderObservationModelError, "training_prior_not_measured"
            ):
                encoder.load_rnd_dynamixel_encoder_observation_model(path)

    def test_core_validates_shape_and_returns_24_dimensions(self):
        state = _state(num_envs=2)
        zeros = _zeros(2)
        state.reset(zeros, zeros, zeros)

        result = state.observe(zeros, zeros, zeros, 0)

        self.assertEqual(result.shape, (2, 24))
        with self.assertRaisesRegex(ValueError, "shape"):
            state.observe(torch.zeros((2, 11)), zeros, zeros, 1)

    def test_duplicate_policy_step_returns_cache_without_advancing_state(self):
        state = _state(num_envs=2)
        zeros = _zeros(2)
        state.reset(zeros, zeros, zeros)
        first_position = torch.arange(24, dtype=torch.float32).reshape(2, JOINT_COUNT) * 0.01
        first_velocity = first_position * 2.0
        first = state.observe(first_position, first_velocity, zeros, 42)
        snapshots = {
            "previous_position": state.previous_position.clone(),
            "current_position": state.current_position.clone(),
            "previous_velocity": state.previous_velocity.clone(),
            "current_velocity": state.current_velocity.clone(),
            "cached_output": state.cached_output.clone(),
            "last_step_counter": state.last_step_counter.clone(),
        }
        first.fill_(1234.0)

        duplicate = state.observe(first_position + 100.0, first_velocity - 100.0, zeros, 42)

        torch.testing.assert_close(duplicate, snapshots["cached_output"])
        for name, expected in snapshots.items():
            torch.testing.assert_close(getattr(state, name), expected)

    def test_partial_reset_is_isolated_and_prefills_current_measurement(self):
        torch.manual_seed(19)
        state = _state(num_envs=2, sample_randomization=True)
        initial_position = torch.stack((torch.full((JOINT_COUNT,), 0.2), torch.full((JOINT_COUNT,), -0.3)))
        initial_velocity = torch.stack((torch.full((JOINT_COUNT,), 0.4), torch.full((JOINT_COUNT,), -0.5)))
        defaults = _zeros(2)
        state.reset(initial_position, initial_velocity, defaults)
        sampled_offset = state.zero_offset_rad.clone()
        sampled_age = state.sample_age_s.clone()
        state.observe(initial_position, initial_velocity, defaults, 7)
        torch.testing.assert_close(state.zero_offset_rad, sampled_offset)
        torch.testing.assert_close(state.sample_age_s, sampled_age)
        env_zero = {
            "previous_position": state.previous_position[0].clone(),
            "current_position": state.current_position[0].clone(),
            "previous_velocity": state.previous_velocity[0].clone(),
            "current_velocity": state.current_velocity[0].clone(),
            "zero_offset_rad": state.zero_offset_rad[0].clone(),
            "sample_age_s": state.sample_age_s[0].clone(),
            "cached_output": state.cached_output[0].clone(),
            "last_step_counter": state.last_step_counter[0].clone(),
        }
        reset_position = torch.stack((torch.full((JOINT_COUNT,), 99.0), torch.full((JOINT_COUNT,), 1.25)))
        reset_velocity = torch.stack((torch.full((JOINT_COUNT,), 99.0), torch.full((JOINT_COUNT,), -0.75)))

        state.reset(reset_position, reset_velocity, defaults, torch.tensor([1]))

        for name, expected in env_zero.items():
            torch.testing.assert_close(getattr(state, name)[0], expected)
        torch.testing.assert_close(state.previous_position[1], reset_position[1])
        torch.testing.assert_close(state.current_position[1], reset_position[1])
        torch.testing.assert_close(state.previous_velocity[1], reset_velocity[1])
        torch.testing.assert_close(state.current_velocity[1], reset_velocity[1])
        expected_prefill = torch.cat((
            encoder.quantize_to_increment(
                reset_position[1] + state.zero_offset_rad[1], encoder.DYNAMIXEL_POSITION_QUANTUM_RAD
            ),
            encoder.quantize_to_increment(reset_velocity[1], encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S) * 0.05,
        ))
        torch.testing.assert_close(state.cached_output[1], expected_prefill)
        self.assertTrue(bool(torch.all(state.sample_age_s[1] >= 0.0)))
        self.assertTrue(bool(torch.all(state.sample_age_s[1] <= 0.005)))
        self.assertFalse(bool(torch.all(state.cached_output[1] == 0.0)))
        self.assertFalse(torch.equal(state.zero_offset_rad[1], sampled_offset[1]))
        self.assertFalse(torch.equal(state.sample_age_s[1], sampled_age[1]))

        same_step = state.observe(reset_position, reset_velocity, defaults, 7)
        torch.testing.assert_close(same_step[0], env_zero["cached_output"])
        torch.testing.assert_close(state.current_position[0], env_zero["current_position"])

    def test_position_and_velocity_quantization_use_dynamixel_units(self):
        state = _state()
        zeros = _zeros()
        state.reset(zeros, zeros, zeros)
        state.sample_age_s.zero_()
        state.zero_offset_rad.zero_()
        position = torch.full_like(zeros, 1.51 * encoder.DYNAMIXEL_POSITION_QUANTUM_RAD)
        velocity = torch.full_like(zeros, -1.51 * encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S)

        result = state.observe(position, velocity, zeros, 0)

        expected_position = torch.full_like(zeros, 2.0 * encoder.DYNAMIXEL_POSITION_QUANTUM_RAD)
        expected_velocity = torch.full_like(zeros, -2.0 * encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S * 0.05)
        torch.testing.assert_close(result[:, :JOINT_COUNT], expected_position)
        torch.testing.assert_close(result[:, JOINT_COUNT:], expected_velocity)

    def test_q_and_dq_share_one_sample_age_for_interpolation(self):
        state = _state()
        zeros = _zeros()
        state.reset(zeros, zeros, zeros)
        state.sample_age_s.fill_(0.005)
        state.zero_offset_rad.zero_()
        current_position = torch.full_like(zeros, 8.0 * encoder.DYNAMIXEL_POSITION_QUANTUM_RAD)
        current_velocity = torch.full_like(zeros, 8.0 * encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S)

        result = state.observe(current_position, current_velocity, zeros, 0)

        expected_position = torch.full_like(zeros, 6.0 * encoder.DYNAMIXEL_POSITION_QUANTUM_RAD)
        expected_velocity = torch.full_like(zeros, 6.0 * encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S * 0.05)
        torch.testing.assert_close(result[:, :JOINT_COUNT], expected_position)
        torch.testing.assert_close(result[:, JOINT_COUNT:], expected_velocity)

    def test_manager_term_preserves_joint_order_and_zoh_by_common_step_counter(self):
        model = encoder.load_rnd_dynamixel_encoder_observation_model()
        storage_names = list(reversed(model.joint_order))
        joint_ids = [storage_names.index(name) for name in model.joint_order]
        data = SimpleNamespace(
            joint_pos=torch.zeros((1, JOINT_COUNT)),
            joint_vel=torch.zeros((1, JOINT_COUNT)),
            default_joint_pos=torch.zeros((1, JOINT_COUNT)),
        )
        asset = SimpleNamespace(data=data, joint_names=storage_names)
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            common_step_counter=10,
            scene={"robot": asset},
        )
        asset_cfg = SimpleNamespace(
            name="robot",
            joint_names=list(model.joint_order),
            joint_ids=joint_ids,
            preserve_order=True,
        )
        params = {
            "asset_cfg": asset_cfg,
            "model_path": str(encoder.RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH),
            "sample_randomization": False,
        }
        term = encoder.RndDynamixelEncoderObservation(SimpleNamespace(params=params, scale=1.0, noise=None), env)
        term.state.sample_age_s.zero_()
        first = term(env, **params)
        canonical_position = torch.arange(1, JOINT_COUNT + 1, dtype=torch.float32)
        canonical_position *= 2.0 * encoder.DYNAMIXEL_POSITION_QUANTUM_RAD
        canonical_velocity = torch.arange(1, JOINT_COUNT + 1, dtype=torch.float32)
        canonical_velocity *= 3.0 * encoder.DYNAMIXEL_VELOCITY_QUANTUM_RAD_S
        data.joint_pos[0] = canonical_position.flip(0)
        data.joint_vel[0] = canonical_velocity.flip(0)

        held = term(env, **params)
        torch.testing.assert_close(held, first)
        torch.testing.assert_close(term.state.current_position, torch.zeros_like(term.state.current_position))

        env.common_step_counter = 11
        updated = term(env, **params)

        self.assertEqual(updated.shape, (1, 24))
        torch.testing.assert_close(updated[0, :JOINT_COUNT], canonical_position)
        torch.testing.assert_close(updated[0, JOINT_COUNT:], canonical_velocity * 0.05)
        self.assertEqual(term.joint_names, model.joint_order)

        fixed_offset = torch.linspace(-0.004, 0.004, JOINT_COUNT)
        fixed_age = torch.linspace(0.0005, 0.0045, JOINT_COUNT)
        term.set_fixed_episode_parameters(zero_offset_rad=fixed_offset, sample_age_s=fixed_age)
        torch.testing.assert_close(term.state.zero_offset_rad[0], fixed_offset)
        torch.testing.assert_close(term.state.sample_age_s[0], fixed_age)
        data.joint_pos[0] = (canonical_position + 0.02).flip(0)
        term.reset()
        torch.testing.assert_close(term.state.zero_offset_rad[0], fixed_offset)
        torch.testing.assert_close(term.state.sample_age_s[0], fixed_age)
        torch.testing.assert_close(term.state.previous_position[0], canonical_position + 0.02)
        torch.testing.assert_close(term.state.current_position[0], canonical_position + 0.02)

        unordered_cfg = SimpleNamespace(**vars(asset_cfg))
        unordered_cfg.preserve_order = False
        unordered_params = dict(params, asset_cfg=unordered_cfg)
        with self.assertRaisesRegex(ValueError, "preserve_order=True"):
            encoder.RndDynamixelEncoderObservation(SimpleNamespace(params=unordered_params, scale=1.0, noise=None), env)


if __name__ == "__main__":
    unittest.main()
