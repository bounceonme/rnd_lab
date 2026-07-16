from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
ROBOT_PACKAGE_DIR = REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import (
    RndActuatorModelError,
    StatefulCommandPath,
    compute_explicit_pd_effort,
    load_rnd_actuator_model,
    validate_rnd_actuator_model,
)
from rnd_actuator_build import DEFAULT_ASSET_CFG, DEFAULT_BASELINE, build_actuator_model
from rnd_actuator_promote import DEFAULT_FALLBACK_JOINTS, ActuatorPromotionError, promote_actuator_model
from rnd_actuator_sweep import PDSweepError, build_pd_candidates, parse_positive_scales, select_pd_candidate


RUNTIME_MODEL_PATH = TOOLS_DIR / "config" / "rnd_actuator_model.json"
CANDIDATE_MODEL_PATH = TOOLS_DIR / "config" / "rnd_actuator_model_candidate.json"
INTEGRATION_MODEL_PATH = TOOLS_DIR / "config" / "rnd_actuator_model_runtime.json"


def _minimal_model(
    *,
    delay_range: list[float] | None = None,
    position_bias_range: list[float] | None = None,
    thresholds: list[float] | None = None,
    weights: list[float] | None = None,
    linear_weight: float = 1.0,
) -> dict:
    return {
        "schema_version": 1,
        "model_type": "rnd_stateful_equivalent_actuator",
        "application_status": "requires_sim_replay_validation",
        "integration_enabled": False,
        "physics_hz": 10.0,
        "policy_hz": 5.0,
        "joint_order": ["joint"],
        "torque_calibration": {"available": False},
        "joints": {
            "joint": {
                "command_path": {
                    "residual_delay_s_range": [0.0, 0.0] if delay_range is None else delay_range,
                    "residual_position_bias_rad_range": (
                        [0.0, 0.0] if position_bias_range is None else position_bias_range
                    ),
                    "play_thresholds_rad": [] if thresholds is None else thresholds,
                    "play_weights": [] if weights is None else weights,
                    "linear_weight": linear_weight,
                    "play_threshold_scale_range": [1.0, 1.0],
                },
                "quality": {
                    "command_path_seed_usable": True,
                    "sim_replay_validated": False,
                    "integration_allowed": False,
                },
            }
        },
    }


class RndActuatorBuildTest(unittest.TestCase):
    def test_build_preserves_gates_and_current_domain(self):
        model = build_actuator_model(DEFAULT_BASELINE, DEFAULT_ASSET_CFG)
        validate_rnd_actuator_model(model)

        self.assertEqual(len(model["joints"]), 12)
        self.assertFalse(model["integration_enabled"])
        self.assertEqual(model["application_status"], "requires_sim_replay_validation")
        self.assertFalse(model["torque_calibration"]["available"])
        self.assertIsNone(model["torque_calibration"]["current_to_joint_torque_nm_per_a"])
        self.assertEqual(model["quality_summary"]["unresolved_joints"], [])
        self.assertEqual(model["quality_summary"]["sim_replay_validated_joints"], ["L_Leg_ankle_roll"])

        left_roll = model["joints"]["L_Leg_ankle_roll"]
        self.assertTrue(left_roll["quality"]["command_path_seed_usable"])
        self.assertTrue(left_roll["quality"]["sim_replay_validated"])
        self.assertFalse(left_roll["quality"]["integration_allowed"])
        self.assertEqual(len(left_roll["command_path"]["play_thresholds_rad"]), 1)
        self.assertGreater(left_roll["command_path"]["linear_weight"], 0.0)
        self.assertAlmostEqual(
            left_roll["command_path"]["linear_weight"] + sum(left_roll["command_path"]["play_weights"]),
            1.0,
        )
        self.assertEqual(left_roll["measured"]["command_path_seed_source"], "multi_amplitude_generalized_play")
        self.assertIsNotNone(left_roll["measured"]["multi_amplitude_model"])
        self.assertEqual(left_roll["controller_seed"]["stiffness"], 26.25)
        self.assertEqual(left_roll["controller_seed"]["damping"], 1.08)
        self.assertGreater(left_roll["command_path"]["residual_delay_s_range"][0], 0.0)
        self.assertFalse(left_roll["friction"]["enabled"])
        self.assertIsNone(left_roll["friction"]["coulomb_torque_nm"])

        right_knee = model["joints"]["R_Leg_knee"]
        measured_backlash = right_knee["measured"]["effective_backlash_rad"]["median"]
        self.assertAlmostEqual(right_knee["command_path"]["play_thresholds_rad"][0], 0.5 * measured_backlash)
        self.assertEqual(right_knee["command_path"]["residual_delay_s_range"], [0.0, 0.0])
        self.assertEqual(right_knee["command_path"]["residual_position_bias_rad_range"], [0.0, 0.0])
        self.assertIsNotNone(right_knee["measured"]["command_minus_position_center_bias_rad"])
        self.assertEqual(right_knee["controller_seed"]["stiffness"], 24.0)
        self.assertEqual(model["joints"]["R_Leg_ankle_pitch"]["controller_seed"]["stiffness"], 21.0)

    def test_checked_in_model_matches_current_inputs_except_timestamp(self):
        generated = build_actuator_model(DEFAULT_BASELINE, DEFAULT_ASSET_CFG)
        checked_in = json.loads(RUNTIME_MODEL_PATH.read_text(encoding="utf-8"))
        generated.pop("created_utc")
        checked_in.pop("created_utc")
        self.assertEqual(generated, checked_in)

    def test_loader_fails_closed_before_sim_replay(self):
        with self.assertRaisesRegex(RndActuatorModelError, "not passed simulator replay"):
            load_rnd_actuator_model(RUNTIME_MODEL_PATH, require_sim_replay_validation=True)
        loaded = load_rnd_actuator_model(
            RUNTIME_MODEL_PATH,
            ("L_Leg_ankle_roll",),
            require_command_path_seed=True,
        )
        self.assertTrue(loaded["joints"]["L_Leg_ankle_roll"]["quality"]["command_path_seed_usable"])

    def test_invalid_weight_sum_is_rejected(self):
        model = _minimal_model(thresholds=[0.1], weights=[0.6], linear_weight=0.6)
        with self.assertRaisesRegex(RndActuatorModelError, "must sum to 1.0"):
            validate_rnd_actuator_model(model)

    def test_reversed_signed_position_bias_range_is_rejected(self):
        model = _minimal_model(position_bias_range=[0.1, -0.1])
        with self.assertRaisesRegex(RndActuatorModelError, "lower bound exceeds upper bound"):
            validate_rnd_actuator_model(model)

    def test_builder_output_round_trips_loader(self):
        model = build_actuator_model(DEFAULT_BASELINE, DEFAULT_ASSET_CFG)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            loaded = load_rnd_actuator_model(path)
        self.assertEqual(loaded["source_baseline_sha256"], model["source_baseline_sha256"])


class RndActuatorPromotionTest(unittest.TestCase):
    @staticmethod
    def _candidate() -> dict:
        return json.loads(CANDIDATE_MODEL_PATH.read_text(encoding="utf-8"))

    def test_partial_runtime_enables_only_replay_validated_joints(self):
        candidate = self._candidate()
        candidate["joints"]["L_Leg_ankle_roll"]["quality"]["sim_replay_validated"] = False
        runtime = promote_actuator_model(candidate, DEFAULT_FALLBACK_JOINTS)
        integrated = tuple(runtime["integration_joint_names"])

        self.assertTrue(runtime["integration_enabled"])
        self.assertEqual(runtime["application_status"], "sim_replay_validated_partial")
        self.assertEqual(runtime["fallback_joint_names"], ["L_Leg_ankle_roll"])
        self.assertEqual(len(integrated), 11)
        limitations = " ".join(runtime["limitations"])
        self.assertNotIn("integration_enabled remains false", limitations)
        self.assertIn("fallback_joint_names use plain explicit PD", limitations)
        validate_rnd_actuator_model(
            runtime,
            integrated,
            require_sim_replay_validation=True,
            require_command_path_seed=True,
        )
        with self.assertRaisesRegex(RndActuatorModelError, "not enabled"):
            validate_rnd_actuator_model(
                runtime,
                ("L_Leg_ankle_roll",),
                require_sim_replay_validation=True,
            )

    def test_partial_runtime_does_not_hide_a_validated_fallback(self):
        with self.assertRaisesRegex(ActuatorPromotionError, "already replay validated"):
            promote_actuator_model(self._candidate(), DEFAULT_FALLBACK_JOINTS)

    def test_full_promotion_accepts_all_replay_validated_joints(self):
        runtime = promote_actuator_model(self._candidate(), ())
        self.assertEqual(runtime["application_status"], "sim_replay_validated")
        self.assertEqual(len(runtime["integration_joint_names"]), 12)
        self.assertEqual(runtime["fallback_joint_names"], [])
        self.assertTrue(runtime["quality_summary"]["integration_ready"])
        limitations = " ".join(runtime["limitations"])
        self.assertNotIn("integration_enabled remains false", limitations)
        self.assertNotIn("fallback_joint_names use plain explicit PD", limitations)
        self.assertIn("Changing or replacing any integrated motor", limitations)

    def test_checked_in_runtime_is_full_promotion_snapshot(self):
        checked_in = json.loads(INTEGRATION_MODEL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(checked_in["integration_enabled"])
        self.assertEqual(checked_in["application_status"], "sim_replay_validated")
        self.assertEqual(checked_in["fallback_joint_names"], [])
        self.assertEqual(len(checked_in["integration_joint_names"]), 12)
        self.assertTrue(checked_in["joints"]["L_Leg_ankle_roll"]["quality"]["sim_replay_validated"])
        self.assertTrue(checked_in["joints"]["L_Leg_ankle_roll"]["quality"]["integration_allowed"])
        validate_rnd_actuator_model(
            checked_in,
            checked_in["integration_joint_names"],
            require_sim_replay_validation=True,
            require_command_path_seed=True,
        )


class StatefulCommandPathTest(unittest.TestCase):
    def test_identity_path_has_no_reset_spike(self):
        model = _minimal_model()
        path = StatefulCommandPath(model, ("joint",), 2, "cpu", step_hz=10.0)
        initial = torch.tensor([[1.2], [-0.4]])
        path.reset(initial)
        result = path.transform(initial)
        torch.testing.assert_close(result, initial)

    def test_fractional_delay_interpolates_history(self):
        model = _minimal_model(delay_range=[0.15, 0.15])
        path = StatefulCommandPath(model, ("joint",), 1, "cpu", step_hz=10.0)
        path.reset(torch.zeros((1, 1)))
        first = path.transform(torch.tensor([[1.0]]))
        second = path.transform(torch.tensor([[2.0]]))
        torch.testing.assert_close(first, torch.tensor([[0.0]]))
        torch.testing.assert_close(second, torch.tensor([[0.5]]))
        torch.testing.assert_close(path.sampled_delay_s, torch.tensor([[0.15]]))

    def test_play_operator_tracks_reversals(self):
        model = _minimal_model(thresholds=[0.1], weights=[1.0], linear_weight=0.0)
        path = StatefulCommandPath(model, ("joint",), 1, "cpu")
        path.reset(torch.zeros((1, 1)))
        small = path.transform(torch.tensor([[0.05]]))
        rising = path.transform(torch.tensor([[0.20]]))
        falling = path.transform(torch.tensor([[-0.20]]))
        torch.testing.assert_close(small, torch.tensor([[0.0]]))
        torch.testing.assert_close(rising, torch.tensor([[0.1]]))
        torch.testing.assert_close(falling, torch.tensor([[-0.1]]))

    def test_position_bias_shifts_transformed_target(self):
        model = _minimal_model(position_bias_range=[-0.02, -0.02])
        path = StatefulCommandPath(model, ("joint",), 1, "cpu", sample_randomization=False)
        path.reset(torch.tensor([[0.5]]))
        torch.testing.assert_close(path.transform(torch.tensor([[0.5]])), torch.tensor([[0.48]]))
        torch.testing.assert_close(path.sampled_position_bias_rad, torch.tensor([[-0.02]]))

    def test_position_bias_override_persists_across_reset(self):
        path = StatefulCommandPath(_minimal_model(), ("joint",), 1, "cpu", sample_randomization=False)
        path.set_position_bias_override(-0.03)
        path.reset(torch.tensor([[0.2]]))
        torch.testing.assert_close(path.transform(torch.tensor([[0.2]])), torch.tensor([[0.17]]))

    def test_partial_reset_keeps_other_environment_history(self):
        model = _minimal_model(delay_range=[0.1, 0.1])
        path = StatefulCommandPath(model, ("joint",), 2, "cpu", step_hz=10.0)
        path.reset(torch.zeros((2, 1)))
        path.transform(torch.tensor([[1.0], [2.0]]))
        path.reset(torch.tensor([[7.0]]), env_ids=torch.tensor([0]))
        output = path.transform(torch.tensor([[7.0], [4.0]]))
        torch.testing.assert_close(output, torch.tensor([[7.0], [2.0]]))

    def test_transform_requires_reset(self):
        path = StatefulCommandPath(_minimal_model(), ("joint",), 1, "cpu")
        with self.assertRaisesRegex(RndActuatorModelError, "Call reset"):
            path.transform(torch.zeros((1, 1)))

    def test_explicit_pd_effort_clips_symmetrically(self):
        target = torch.tensor([[1.0, -1.0]])
        position = torch.zeros_like(target)
        velocity = torch.tensor([[0.5, -0.5]])
        effort = compute_explicit_pd_effort(
            target,
            position,
            velocity,
            stiffness=10.0,
            damping=2.0,
            effort_limit_nm=3.0,
        )
        torch.testing.assert_close(effort, torch.tensor([[3.0, -3.0]]))

    def test_randomization_is_reproducible(self):
        model = _minimal_model(
            delay_range=[0.0, 0.2],
            position_bias_range=[-0.02, 0.01],
            thresholds=[0.1],
            weights=[1.0],
            linear_weight=0.0,
        )
        model["joints"]["joint"]["command_path"]["play_threshold_scale_range"] = [0.8, 1.2]
        first = StatefulCommandPath(copy.deepcopy(model), ("joint",), 4, "cpu", seed=17)
        second = StatefulCommandPath(copy.deepcopy(model), ("joint",), 4, "cpu", seed=17)
        initial = torch.zeros((4, 1))
        first.reset(initial)
        second.reset(initial)
        torch.testing.assert_close(first.sampled_delay_s, second.sampled_delay_s)
        torch.testing.assert_close(first.sampled_position_bias_rad, second.sampled_position_bias_rad)
        torch.testing.assert_close(first.sampled_play_thresholds_rad, second.sampled_play_thresholds_rad)

    def test_nominal_mode_uses_delay_midpoint_and_unit_threshold_scale(self):
        model = _minimal_model(
            delay_range=[0.0, 0.2],
            position_bias_range=[-0.03, 0.01],
            thresholds=[0.1],
            weights=[1.0],
            linear_weight=0.0,
        )
        model["joints"]["joint"]["command_path"]["play_threshold_scale_range"] = [0.8, 1.3]
        path = StatefulCommandPath(model, ("joint",), 2, "cpu", sample_randomization=False)
        path.reset(torch.zeros((2, 1)))
        torch.testing.assert_close(path.sampled_delay_s, torch.full((2, 1), 0.1))
        torch.testing.assert_close(path.sampled_position_bias_rad, torch.full((2, 1), -0.01))
        torch.testing.assert_close(path.sampled_play_thresholds_rad, torch.full((2, 1, 1), 0.1))

    def test_delay_override_supports_fractional_physics_steps(self):
        path = StatefulCommandPath(_minimal_model(), ("joint",), 1, "cpu", step_hz=200.0)
        path.set_delay_override(0.00475)
        path.reset(torch.zeros((1, 1)))

        torch.testing.assert_close(path.sampled_delay_s, torch.full((1, 1), 0.00475))
        path.transform(torch.ones((1, 1)))
        output = path.transform(torch.full((1, 1), 2.0))
        torch.testing.assert_close(output, torch.full((1, 1), 1.05))

    def test_delay_override_rejects_history_growth_after_reset(self):
        path = StatefulCommandPath(_minimal_model(), ("joint",), 1, "cpu", step_hz=200.0)
        path.reset(torch.zeros((1, 1)))
        with self.assertRaisesRegex(RndActuatorModelError, "before reset"):
            path.set_delay_override(0.02)


class PDSweepTest(unittest.TestCase):
    @staticmethod
    def _candidate(
        delay_error: float,
        *,
        stiffness: float = 30.0,
        damping: float = 1.5,
        gain_error: float = 0.01,
        normalized_rmse: float = 0.02,
        max_abs_effort_nm: float = 0.4,
    ) -> dict:
        return {
            "stiffness": stiffness,
            "damping": damping,
            "delay_error_s": delay_error,
            "gain_relative_error": gain_error,
            "normalized_rmse": normalized_rmse,
            "max_abs_effort_nm": max_abs_effort_nm,
            "valid": True,
        }

    def test_scale_parser_and_grid_are_deterministic(self):
        stiffness_scales = parse_positive_scales("1.0, 1.5,1", "--stiffness-scales")
        damping_scales = parse_positive_scales("0.5,1.0", "--damping-scales")
        self.assertEqual(stiffness_scales, [1.0, 1.5])
        self.assertEqual(
            build_pd_candidates(24.0, 1.8, stiffness_scales, damping_scales),
            [(24.0, 0.9), (24.0, 1.8), (36.0, 0.9), (36.0, 1.8)],
        )

    def test_scale_parser_rejects_non_positive_values(self):
        with self.assertRaisesRegex(PDSweepError, "finite and positive"):
            parse_positive_scales("1.0,0.0", "--stiffness-scales")

    def test_selector_prefers_minimum_change_candidate_already_within_gate(self):
        near_seed = self._candidate(0.003, stiffness=24.0, damping=1.44, max_abs_effort_nm=0.3)
        high_gain = self._candidate(-0.001, stiffness=48.0, damping=2.52, max_abs_effort_nm=0.4)
        seed_outside_gate = self._candidate(0.006, stiffness=24.0, damping=1.8, max_abs_effort_nm=0.25)
        result = select_pd_candidate(
            [near_seed, high_gain, seed_outside_gate],
            seed_stiffness=24.0,
            seed_damping=1.8,
            maximum_delay_error_s=0.005,
        )
        self.assertIs(result["selected"], near_seed)
        self.assertEqual(result["selection_mode"], "within_gate_minimum_change")
        self.assertTrue(result["selected_within_delay_gate"])
        self.assertFalse(result["positive_residual_compensation_available"])
        self.assertEqual(result["within_delay_gate_candidate_count"], 2)

    def test_selector_uses_residual_only_when_no_candidate_passes_delay_gate(self):
        slower = self._candidate(0.008, stiffness=24.0, damping=1.8)
        faster_far = self._candidate(-0.020, stiffness=24.0, damping=1.44)
        faster_close = self._candidate(-0.010, stiffness=36.0, damping=1.8, gain_error=0.03)
        result = select_pd_candidate(
            [slower, faster_far, faster_close],
            seed_stiffness=24.0,
            seed_damping=1.8,
            maximum_delay_error_s=0.005,
        )
        self.assertIs(result["selected"], faster_close)
        self.assertEqual(result["selection_mode"], "non_negative_residual_compensable")
        self.assertTrue(result["positive_residual_compensation_available"])
        self.assertFalse(result["selected_within_delay_gate"])

    def test_selector_marks_uncompensable_fallback(self):
        slow_far = self._candidate(0.010)
        slow_close = self._candidate(0.006)
        result = select_pd_candidate(
            [slow_far, slow_close, {"valid": False}],
            seed_stiffness=24.0,
            seed_damping=1.8,
            maximum_delay_error_s=0.005,
        )
        self.assertIs(result["selected"], slow_close)
        self.assertEqual(result["selection_mode"], "least_slow_uncompensable_fallback")
        self.assertFalse(result["positive_residual_compensation_available"])


if __name__ == "__main__":
    unittest.main()
