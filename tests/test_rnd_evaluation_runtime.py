from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


RSL_RL_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPTS))

from evaluation_runtime import (  # noqa: E402
    CommandScheduleError,
    EvaluationArtifactWriter,
    FixedDomainError,
    FixedDomainSettings,
    FixedSingletonDomainApplicator,
    MassScaledPhysicalPulse,
    PhysicalPulseError,
    SafeCommandTargetAdapter,
    command_schedule_from_scenario,
    equally_weighted_summary,
    joint_order_permutation,
    physical_pulse_spec_from_scenario,
    reject_legacy_disturbances,
    stand_start_stop_schedule,
    straight_turn_straight_stop_schedule,
    validate_checkpoint_actor_observation_dimension,
    validate_split_checkpoint,
    yaw_rotate_body_vector,
)


def push_by_setting_velocity():
    pass


def apply_external_force_torque():
    pass


class _FakeComposer:
    def __init__(self, num_envs: int, num_bodies: int):
        self.composed_force_as_torch = torch.zeros(num_envs, num_bodies, 3)
        self.composed_torque_as_torch = torch.zeros_like(self.composed_force_as_torch)
        self.set_calls: list[dict[str, object]] = []
        self.reset_count = 0

    def set_forces_and_torques(self, *, forces, torques, body_ids, env_ids, is_global):
        self.set_calls.append({"body_ids": list(body_ids), "env_ids": env_ids.clone(), "is_global": is_global})
        for source_index, env_id in enumerate(env_ids.tolist()):
            for body_offset, body_id in enumerate(body_ids):
                self.composed_force_as_torch[env_id, body_id] = forces[source_index, body_offset]
                self.composed_torque_as_torch[env_id, body_id] = torques[source_index, body_offset]

    def reset(self, env_ids=None):
        if env_ids is None:
            self.composed_force_as_torch.zero_()
            self.composed_torque_as_torch.zero_()
        else:
            self.composed_force_as_torch[env_ids] = 0.0
            self.composed_torque_as_torch[env_ids] = 0.0
        self.reset_count += 1


class _FakeRootPhysxView:
    def __init__(self, masses: torch.Tensor):
        self._masses = masses

    def get_masses(self):
        return self._masses.clone()


class _FakeRobot:
    def __init__(self):
        self.device = "cpu"
        self.num_instances = 2
        self.num_bodies = 3
        self.body_names = ["base", "left", "right"]
        self.permanent_wrench_composer = _FakeComposer(self.num_instances, self.num_bodies)
        self.root_physx_view = _FakeRootPhysxView(torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
        half_yaw = math.pi / 4.0
        self.data = SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
        )

    def find_bodies(self, expression: str, preserve_order: bool = False):
        del preserve_order
        if expression == "base":
            return [0], ["base"]
        return [], []


class _RecordingDomainBackend:
    def __init__(self):
        self.calls: list[str] = []

    def _record(self, name: str):
        self.calls.append(name)
        return {"applied": name}

    def apply_material(self, settings):
        del settings
        return self._record("material")

    def apply_mass(self, base_add_kg, other_scale):
        del base_add_kg, other_scale
        return self._record("mass")

    def apply_base_com(self, offset_m):
        del offset_m
        return self._record("base_com")

    def apply_actuator(self, settings):
        del settings
        return self._record("actuator")

    def apply_encoder(self, settings):
        del settings
        return self._record("encoder")

    def apply_imu(self, settings):
        del settings
        return self._record("imu")


def _fixed_domain_mapping():
    return {
        "material": {"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
        "base_mass_add_kg": 0.25,
        "other_mass_scale": 1.1,
        "base_com_offset_m": {"x": 0.01, "y": -0.02, "z": 0.0},
        "actuator": {"stiffness_scale": 1.0, "damping_scale": 0.9},
        "encoder": {
            "zero_offset_rad": [0.0] * 12,
            "sample_age_s": [0.0025] * 12,
        },
        "imu": {"gyro": {"delay_s": 0.01, "noise_sigma": 0.0, "bias": 0.0}},
    }


class CommandScheduleTests(unittest.TestCase):
    def test_stand_start_stop_half_open_boundaries_and_straight_axis(self):
        schedule = stand_start_stop_schedule(-0.4)

        self.assertEqual(schedule.phase_at(0.0).name, "stand")
        self.assertEqual(schedule.phase_at(1.999999).name, "stand")
        self.assertEqual(schedule.phase_at(2.0).name, "straight")
        self.assertEqual(schedule.phase_at(7.999999).name, "straight")
        self.assertEqual(schedule.phase_at(8.0).name, "stop")
        self.assertEqual(schedule.phase_at(20.0).name, "stop")
        self.assertEqual(schedule.target_at(4.0), (0.0, -0.4, 0.0))
        self.assertTrue(all(phase.target_b[0] == 0.0 for phase in schedule.phases))

    def test_straight_turn_straight_stop_boundaries(self):
        schedule = straight_turn_straight_stop_schedule(0.5, -0.7)

        expected = {
            0.0: "stand",
            2.0: "straight-1",
            6.0: "turn",
            10.0: "straight-2",
            14.0: "stop",
            20.0: "stop",
        }
        self.assertEqual({time_s: schedule.phase_at(time_s).name for time_s in expected}, expected)
        self.assertEqual(schedule.target_at(6.0), (0.0, 0.5, -0.7))
        self.assertEqual(schedule.target_at(10.0), (0.0, 0.5, 0.0))

    def test_target_adapter_preserves_ramped_command_and_disables_heading(self):
        term = SimpleNamespace(
            cfg=SimpleNamespace(zero_velocity_threshold=0.05),
            vel_command_target_b=torch.zeros(3, 3),
            vel_command_b=torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.0, 0.3, 0.0]]),
            is_heading_env=torch.ones(3, dtype=torch.bool),
            is_standing_env=torch.ones(3, dtype=torch.bool),
            is_pure_yaw_env=torch.ones(3, dtype=torch.bool),
            is_straight_env=torch.zeros(3, dtype=torch.bool),
        )
        current_before = term.vel_command_b.clone()

        readback = SafeCommandTargetAdapter(term).inject((0.0, -0.6, 0.0))

        torch.testing.assert_close(term.vel_command_target_b, torch.tensor([[0.0, -0.6, 0.0]]).repeat(3, 1))
        torch.testing.assert_close(term.vel_command_b, current_before)
        self.assertFalse(bool(term.is_heading_env.any()))
        self.assertFalse(bool(term.is_standing_env.any()))
        self.assertTrue(bool(term.is_straight_env.all()))
        self.assertEqual(readback["command_b"], current_before.tolist())
        with self.assertRaises(CommandScheduleError):
            SafeCommandTargetAdapter(term).inject((0.01, 0.0, 0.0))
        term.cfg.transition_sequence_probabilities = (0.15, 0.15)
        with self.assertRaisesRegex(CommandScheduleError, "transition_sequence_probabilities"):
            SafeCommandTargetAdapter(term)

    def test_generic_suite_segments_are_contiguous_and_keep_body_x_zero(self):
        scenario = {
            "id": "generic",
            "horizon_steps": 1000,
            "segments": [
                {
                    "id": "stand",
                    "start_step": 0,
                    "end_step": 125,
                    "command": {"lin_vel_x_m_s": 0.0, "lin_vel_y_m_s": 0.0, "ang_vel_z_rad_s": 0.0},
                },
                {
                    "id": "move",
                    "start_step": 125,
                    "end_step": 875,
                    "command": {"lin_vel_x_m_s": 0.0, "lin_vel_y_m_s": 0.4, "ang_vel_z_rad_s": 0.1},
                },
                {
                    "id": "stop",
                    "start_step": 875,
                    "end_step": 1000,
                    "command": {"lin_vel_x_m_s": 0.0, "lin_vel_y_m_s": 0.0, "ang_vel_z_rad_s": 0.0},
                },
            ],
            "pulses": [],
        }

        schedule = command_schedule_from_scenario(scenario, 50.0)

        self.assertEqual(schedule.phase_at(2.5).name, "move")
        self.assertEqual(schedule.target_at(10.0), (0.0, 0.4, 0.1))
        invalid = {**scenario, "segments": [dict(segment) for segment in scenario["segments"]]}
        invalid["segments"][1]["start_step"] = 126
        with self.assertRaisesRegex(CommandScheduleError, "gap"):
            command_schedule_from_scenario(invalid, 50.0)


class JointOrderTests(unittest.TestCase):
    def test_runtime_arrays_are_reordered_into_the_metric_contract(self):
        runtime = ("L_hip", "R_hip", "L_knee", "R_knee")
        metric = ("R_hip", "R_knee", "L_hip", "L_knee")

        permutation = joint_order_permutation(runtime, metric)

        np.testing.assert_array_equal(permutation, np.asarray([1, 3, 0, 2], dtype=np.int64))
        runtime_values = np.asarray([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(runtime_values[permutation], np.asarray([20.0, 40.0, 10.0, 30.0]))

    def test_joint_order_mismatch_and_duplicates_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            joint_order_permutation(("left", "right"), ("left", "ankle"))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            joint_order_permutation(("left", "left"), ("left", "right"))


class PhysicalPulseTests(unittest.TestCase):
    def _pulse(self, robot=None, *, onset_s=0.04):
        return MassScaledPhysicalPulse(
            robot or _FakeRobot(),
            base_body_name="base",
            onset_s=onset_s,
            delta_velocity_body_m_s=(1.0, 0.0, 0.0),
            physics_dt=0.005,
            decimation=4,
        )

    def test_yaw_rotation_helper(self):
        rotated = yaw_rotate_body_vector((1.0, 0.0, 0.0), math.pi / 2.0)
        np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=1.0e-7)

    def test_mass_yaw_impulse_and_exact_tick_clear(self):
        robot = _FakeRobot()
        pulse = self._pulse(robot)
        active_before_after: list[bool] = []
        fake_env = SimpleNamespace(scene={"robot": robot})

        for step in range(8):
            pulse.before_policy_step(step)
            active_before_after.append(pulse.active)
            for substep in range(4):
                pulse.on_post_scene_update(fake_env, None, step, substep)
            pulse.after_policy_step(step)

        self.assertEqual(active_before_after, [False, False, True, True, True, True, True, True])
        self.assertEqual(pulse.duration_ticks, 24)
        self.assertEqual(pulse.observed_physics_ticks, 24)
        self.assertTrue(pulse.complete)
        self.assertFalse(pulse.active)
        self.assertEqual(len(robot.permanent_wrench_composer.set_calls), 24)
        self.assertTrue(all(call["is_global"] for call in robot.permanent_wrench_composer.set_calls))
        self.assertGreaterEqual(robot.permanent_wrench_composer.reset_count, 1)
        self.assertFalse(bool(robot.permanent_wrench_composer.composed_force_as_torch.any()))
        self.assertFalse(bool(robot.permanent_wrench_composer.composed_torque_as_torch.any()))

        readback = pulse.readback
        self.assertIsNotNone(readback)
        np.testing.assert_allclose(readback["mass_kg"], [6.0, 9.0])
        np.testing.assert_allclose(
            readback["delta_velocity_world_m_s"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1.0e-6
        )
        np.testing.assert_allclose(readback["force_world_n"], [[50.0, 0.0, 0.0], [0.0, 75.0, 0.0]], atol=1.0e-5)
        impulse_velocity = np.asarray(readback["force_world_n"]) * 0.12 / np.asarray(readback["mass_kg"])[:, None]
        np.testing.assert_allclose(impulse_velocity, readback["delta_velocity_world_m_s"], atol=1.0e-6)
        np.testing.assert_allclose(readback["composer_torque_body_nm"], 0.0)
        self.assertEqual(readback["status"], "complete-cleared")
        self.assertEqual(readback["observed_physics_ticks_by_env"], [24, 24])

    def test_per_environment_tick_readback_preserves_partial_delivery(self):
        robot = _FakeRobot()
        pulse = self._pulse(robot, onset_s=0.0)
        fake_env = SimpleNamespace(scene={"robot": robot})

        for step in range(6):
            pulse.before_policy_step(step)
            for substep in range(4):
                pulse.on_post_scene_update(fake_env, None, step, substep)
                if step == 0 and substep == 1:
                    pulse.on_pre_reset(
                        fake_env,
                        torch.tensor([0]),
                        torch.tensor([True, False]),
                        torch.tensor([False, False]),
                    )
            pulse.after_policy_step(step)

        readback = pulse.readback
        self.assertIsNotNone(readback)
        self.assertEqual(readback["observed_physics_ticks_by_env"], [2, 24])
        self.assertEqual(readback["status"], "complete-cleared-with-partial-episodes")

    def test_reset_clears_active_wrench_and_restarts_ordering(self):
        robot = _FakeRobot()
        pulse = self._pulse(robot, onset_s=0.0)
        pulse.before_policy_step(0)
        self.assertTrue(pulse.active)

        pulse.reset()

        self.assertFalse(pulse.active)
        self.assertFalse(pulse.complete)
        self.assertIsNone(pulse.readback)
        self.assertFalse(bool(robot.permanent_wrench_composer.composed_force_as_torch.any()))
        pulse.before_policy_step(0)
        self.assertTrue(pulse.active)

    def test_rejects_preexisting_wrench_and_step_order_violation(self):
        robot = _FakeRobot()
        robot.permanent_wrench_composer.composed_force_as_torch[0, 0, 0] = 1.0
        with self.assertRaises(PhysicalPulseError):
            self._pulse(robot)

        pulse = self._pulse()
        with self.assertRaises(PhysicalPulseError):
            pulse.before_policy_step(1)

    def test_pulse_suite_contract_rejects_short_or_angular_impulses(self):
        scenario = {
            "id": "pulse",
            "horizon_steps": 1000,
            "segments": [],
            "pulses": [
                {
                    "id": "side",
                    "start_step": 300,
                    "end_step": 306,
                    "root_velocity": {
                        "lin_vel_x_m_s": 0.0,
                        "lin_vel_y_m_s": 0.3,
                        "ang_vel_z_rad_s": 0.0,
                    },
                }
            ],
        }
        resolved = physical_pulse_spec_from_scenario(scenario, 50.0)
        self.assertEqual(resolved["delta_velocity_body_m_s"], (0.0, 0.3, 0.0))
        self.assertAlmostEqual(resolved["onset_s"], 6.0)

        scenario["pulses"][0]["end_step"] = 301
        with self.assertRaisesRegex(PhysicalPulseError, "6 policy steps"):
            physical_pulse_spec_from_scenario(scenario, 50.0)
        scenario["pulses"][0]["end_step"] = 306
        scenario["pulses"][0]["root_velocity"]["ang_vel_z_rad_s"] = 0.2
        with self.assertRaisesRegex(PhysicalPulseError, "purely translational"):
            physical_pulse_spec_from_scenario(scenario, 50.0)


class GuardAndDomainTests(unittest.TestCase):
    def test_legacy_disturbance_guard(self):
        zero_reset = SimpleNamespace(
            func=lambda: None,
            mode="reset",
            params={"velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}},
        )
        readback = reject_legacy_disturbances(SimpleNamespace(reset_base=zero_reset))
        self.assertEqual(readback["reset_velocity_range"]["yaw"], [0.0, 0.0])

        zero_joint_reset = SimpleNamespace(
            func=lambda: None,
            mode="reset",
            params={"velocity_range": (0.0, 0.0)},
        )
        reject_legacy_disturbances(SimpleNamespace(reset_joints=zero_joint_reset))

        push = SimpleNamespace(func=push_by_setting_velocity, mode="interval", params={})
        with self.assertRaises(PhysicalPulseError):
            reject_legacy_disturbances(SimpleNamespace(push=push))

        wrench = SimpleNamespace(func=apply_external_force_torque, mode="interval", params={})
        with self.assertRaises(PhysicalPulseError):
            reject_legacy_disturbances(SimpleNamespace(wrench=wrench))

        nonzero_reset = SimpleNamespace(
            func=lambda: None,
            mode="reset",
            params={"velocity_range": {"x": (0.0, 0.1)}},
        )
        with self.assertRaises(PhysicalPulseError):
            reject_legacy_disturbances(SimpleNamespace(reset_base=nonzero_reset))

        nonzero_joint_reset = SimpleNamespace(
            func=lambda: None,
            mode="reset",
            params={"velocity_range": (-0.1, 0.1)},
        )
        with self.assertRaises(PhysicalPulseError):
            reject_legacy_disturbances(SimpleNamespace(reset_joints=nonzero_joint_reset))

    def test_fixed_domain_rejects_ranges(self):
        domain = _fixed_domain_mapping()
        domain["material"]["static_friction"] = [0.7, 0.9]
        with self.assertRaises(FixedDomainError):
            FixedDomainSettings.from_mapping(domain)

    def test_fixed_domain_rejects_encoder_values_outside_training_prior(self):
        domain = _fixed_domain_mapping()
        domain["encoder"]["sample_age_s"][3] = 0.0051
        with self.assertRaisesRegex(FixedDomainError, "sample_age_s"):
            FixedDomainSettings.from_mapping(domain)

    def test_fixed_domain_application_order_and_readback(self):
        settings = FixedDomainSettings.from_mapping(_fixed_domain_mapping())
        backend = _RecordingDomainBackend()

        readback = FixedSingletonDomainApplicator(backend).apply(settings)

        expected = ["material", "mass", "base_com", "actuator", "encoder", "imu"]
        self.assertEqual(backend.calls, expected)
        self.assertEqual(list(readback), expected)
        self.assertEqual(readback["mass"], {"applied": "mass"})


class EvaluationContractTests(unittest.TestCase):
    def test_test_split_requires_matching_frozen_checkpoint_sha(self):
        validate_split_checkpoint("validation", None, "a" * 64)
        with self.assertRaisesRegex(Exception, "freeze"):
            validate_split_checkpoint("test", None, "a" * 64)
        with self.assertRaisesRegex(Exception, "mismatch"):
            validate_split_checkpoint("test", "b" * 64, "a" * 64)
        validate_split_checkpoint("test", "a" * 64, "a" * 64)

    def test_checkpoint_actor_dimension_fails_before_runner_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(512, 45)}}, checkpoint)
            with self.assertRaisesRegex(Exception, "45.*171"):
                validate_checkpoint_actor_observation_dimension(checkpoint, 171)
            torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(512, 171)}}, checkpoint)
            self.assertEqual(validate_checkpoint_actor_observation_dimension(checkpoint, 171), 171)

    def test_artifacts_and_equal_episode_then_case_weighting(self):
        summary = equally_weighted_summary({"a": [{"score": 0.0}, {"score": 2.0}], "b": [{"score": 10.0}]})
        self.assertEqual(summary["case_means"]["a"]["score"], 1.0)
        self.assertEqual(summary["overall"]["score"], 5.5)

        with tempfile.TemporaryDirectory() as directory:
            writer = EvaluationArtifactWriter(directory)
            artifact = writer.write_case(
                "case-1",
                resolved_config={"case": "case-1", "seed": 7},
                raw={"time_s": np.asarray([0.0, 0.02])},
                metrics={"score": 1.0},
            )
            writer.write_summary(summary)
            self.assertEqual(len(artifact["resolved_config_sha256"]), 64)
            self.assertTrue(Path(artifact["resolved_config"]).is_file())
            self.assertTrue(Path(artifact["raw_npz"]).is_file())
            self.assertTrue(Path(artifact["metrics_json"]).is_file())
            self.assertTrue((Path(directory) / "summary.json").is_file())

    def test_evaluate_cli_surface_omits_interactive_and_export_modes(self):
        source = (RSL_RL_SCRIPTS / "evaluate.py").read_text(encoding="utf-8")
        for flag in (
            "--checkpoint",
            "--task",
            "--suite",
            "--cases",
            "--split",
            "--num_envs",
            "--output",
            "--observation-corruption",
        ):
            self.assertIn(flag, source)
        self.assertNotIn("--video", source)
        self.assertNotIn("keyboard", source.lower())
        self.assertNotIn("export_policy", source)


if __name__ == "__main__":
    unittest.main()
