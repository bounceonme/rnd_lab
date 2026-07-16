from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "scripts" / "tools"
ROBOT_PACKAGE_DIR = ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROBOT_PACKAGE_DIR))

from actuators.rnd_torque_randomization import (
    EpisodeTorqueRandomizer,
    RndTorqueRandomizationError,
    load_rnd_torque_randomization,
    validate_rnd_torque_randomization,
)
from rnd_real2sim.config import RND_LEG_JOINT_NAMES
from rnd_torque_randomization_build import build_torque_randomization


CONFIG_PATH = TOOLS_DIR / "config" / "rnd_torque_randomization.json"


def _minimal_model() -> dict:
    return {
        "schema_version": 2,
        "model_type": "rnd_joint_torque_randomization",
        "integration_enabled": True,
        "sample_per_episode": True,
        "sample_bilateral_pairs_with_shared_quantile": True,
        "viscous_friction_enabled": False,
        "static_breakaway_enabled": False,
        "joint_order": ["measured", "prior"],
        "motor_strength_scale_range": [0.8, 1.2],
        "friction_transition_velocity_rad_s_range": [0.05, 0.15],
        "joints": {
            "measured": {
                "evidence_status": "measured_quality_pass",
                "source_quality_pass": True,
                "measured_coulomb_torque_nm": 0.2,
                "coulomb_torque_range_nm": [0.1, 0.3],
            },
            "prior": {
                "evidence_status": "unidentified_prior",
                "source_quality_pass": False,
                "measured_coulomb_torque_nm": None,
                "coulomb_torque_range_nm": [0.0, 0.3],
            },
        },
    }


def _bilateral_model() -> dict:
    model = _minimal_model()
    model["joint_order"] = ["R_Leg_knee", "L_Leg_knee"]
    model["joints"] = {
        "R_Leg_knee": {
            "evidence_status": "measured_quality_pass",
            "source_quality_pass": True,
            "measured_coulomb_torque_nm": 0.2,
            "coulomb_torque_range_nm": [0.1, 0.3],
        },
        "L_Leg_knee": {
            "evidence_status": "unidentified_prior",
            "source_quality_pass": False,
            "measured_coulomb_torque_nm": None,
            "coulomb_torque_range_nm": [0.0, 0.5],
        },
    }
    return model


def _synthetic_report() -> dict:
    measured = set(RND_LEG_JOINT_NAMES[:4])
    joints = {}
    for index, joint_name in enumerate(RND_LEG_JOINT_NAMES):
        quality_pass = joint_name in measured
        joints[joint_name] = {
            "low_current_torque_calibration": {
                "quality": {"pass": quality_pass, "reasons": [] if quality_pass else ["unobservable"]},
                "coulomb_torque_nm": 0.15 + 0.01 * index,
                "coulomb_current_a": 0.1,
                "torque_per_amp_nm": 1.5 + 0.1 * index,
                "bootstrap_90pct_nm_per_a": [1.4 + 0.1 * index, 1.6 + 0.1 * index],
            }
        }
    return {
        "schema_version": 1,
        "model_type": "rnd_real2sim_all_joint_torque_calibration",
        "analysis_only": True,
        "integration_enabled": False,
        "source_dataset": "dataset.npz",
        "source_dataset_sha256": "d" * 64,
        "source_dynamics_trace": "trace.npz",
        "source_dynamics_trace_sha256": "t" * 64,
        "summary": {"calibration_passed_joints": list(RND_LEG_JOINT_NAMES[:4])},
        "joints": joints,
    }


class TorqueRandomizationConfigTest(unittest.TestCase):
    def test_checked_in_config_preserves_four_measured_and_eight_priors(self):
        model = load_rnd_torque_randomization(CONFIG_PATH, RND_LEG_JOINT_NAMES)
        summary = model["quality_summary"]
        self.assertEqual(summary["measured_joint_count"], 4)
        self.assertEqual(summary["unidentified_prior_joint_count"], 8)
        self.assertEqual(model["joints"]["R_Leg_hip_yaw"]["coulomb_torque_range_nm"], [0.0, 0.3])
        self.assertEqual(model["joints"]["R_Leg_hip_pitch"]["evidence_status"], "measured_quality_pass")
        self.assertTrue(model["sample_bilateral_pairs_with_shared_quantile"])
        self.assertFalse(Path(model["source_dataset"]).is_absolute())
        self.assertFalse(Path(model["source_dynamics_trace"]).is_absolute())

    def test_builder_uses_only_passing_values(self):
        model = build_torque_randomization(
            _synthetic_report(),
            report_path="report.json",
            report_sha256="a" * 64,
        )
        validate_rnd_torque_randomization(model, RND_LEG_JOINT_NAMES)
        measured = model["joints"][RND_LEG_JOINT_NAMES[0]]
        rejected = model["joints"][RND_LEG_JOINT_NAMES[-1]]
        self.assertEqual(measured["evidence_status"], "measured_quality_pass")
        self.assertLess(measured["coulomb_torque_range_nm"][0], measured["measured_coulomb_torque_nm"])
        self.assertGreater(measured["coulomb_torque_range_nm"][1], measured["measured_coulomb_torque_nm"])
        self.assertEqual(rejected["evidence_status"], "unidentified_prior")
        self.assertIsNone(rejected["measured_coulomb_torque_nm"])
        self.assertEqual(rejected["coulomb_torque_range_nm"], [0.0, 0.3])

    def test_validator_rejects_failed_fit_claimed_as_measured(self):
        model = _minimal_model()
        model["joints"]["measured"]["source_quality_pass"] = False
        with self.assertRaisesRegex(RndTorqueRandomizationError, "must have passed quality"):
            validate_rnd_torque_randomization(model)

    def test_validator_requires_bilateral_pair_sampling(self):
        model = _minimal_model()
        model["sample_bilateral_pairs_with_shared_quantile"] = False
        with self.assertRaisesRegex(RndTorqueRandomizationError, "sample_bilateral_pairs_with_shared_quantile"):
            validate_rnd_torque_randomization(model)


class EpisodeTorqueRandomizerTest(unittest.TestCase):
    def test_midpoint_mode_scales_then_clips_and_opposes_velocity(self):
        randomizer = EpisodeTorqueRandomizer(
            _minimal_model(),
            ("measured", "prior"),
            num_envs=1,
            device="cpu",
            sample_randomization=False,
        )
        torch.testing.assert_close(randomizer.sampled_coulomb_torque_nm, torch.tensor([[0.2, 0.15]]))
        torch.testing.assert_close(randomizer.sampled_motor_strength_scale, torch.ones((1, 2)))
        torch.testing.assert_close(randomizer.sampled_transition_velocity_rad_s, torch.full((1, 2), 0.1))

        output = randomizer.apply(
            torch.tensor([[6.0, -6.0]]),
            torch.tensor([[1.0, -1.0]]),
            effort_limit_nm=torch.tensor([[5.0, 5.0]]),
        )
        torch.testing.assert_close(output, torch.tensor([[4.8, -4.85]]), atol=1.0e-4, rtol=0.0)

    def test_zero_velocity_adds_no_coulomb_effort(self):
        randomizer = EpisodeTorqueRandomizer(
            _minimal_model(),
            ("measured", "prior"),
            num_envs=1,
            device="cpu",
            sample_randomization=False,
        )
        output = randomizer.apply(torch.tensor([[1.0, -1.0]]), torch.zeros((1, 2)), effort_limit_nm=5.0)
        torch.testing.assert_close(output, torch.tensor([[1.0, -1.0]]))
        torch.testing.assert_close(randomizer.last_friction_effort_nm, torch.zeros((1, 2)))

    def test_random_samples_are_reproducible_and_bounded(self):
        model = _minimal_model()
        first = EpisodeTorqueRandomizer(copy.deepcopy(model), ("measured", "prior"), 32, "cpu", seed=71)
        second = EpisodeTorqueRandomizer(copy.deepcopy(model), ("measured", "prior"), 32, "cpu", seed=71)
        torch.testing.assert_close(first.sampled_coulomb_torque_nm, second.sampled_coulomb_torque_nm)
        torch.testing.assert_close(first.sampled_motor_strength_scale, second.sampled_motor_strength_scale)
        self.assertTrue(bool((first.sampled_coulomb_torque_nm[:, 0] >= 0.1).all().item()))
        self.assertTrue(bool((first.sampled_coulomb_torque_nm[:, 0] <= 0.3).all().item()))
        self.assertTrue(bool((first.sampled_coulomb_torque_nm[:, 1] >= 0.0).all().item()))
        self.assertTrue(bool((first.sampled_coulomb_torque_nm[:, 1] <= 0.3).all().item()))
        self.assertTrue(bool((first.sampled_motor_strength_scale >= 0.8).all().item()))
        self.assertTrue(bool((first.sampled_motor_strength_scale <= 1.2).all().item()))

    def test_bilateral_pairs_share_quantiles_but_keep_per_side_friction_ranges(self):
        model = _bilateral_model()
        randomizer = EpisodeTorqueRandomizer(
            model,
            ("R_Leg_knee", "L_Leg_knee"),
            num_envs=64,
            device="cpu",
            seed=19,
        )

        torch.testing.assert_close(
            randomizer.sampled_motor_strength_scale[:, 0],
            randomizer.sampled_motor_strength_scale[:, 1],
        )
        torch.testing.assert_close(
            randomizer.sampled_transition_velocity_rad_s[:, 0],
            randomizer.sampled_transition_velocity_rad_s[:, 1],
        )
        right_quantile = (randomizer.sampled_coulomb_torque_nm[:, 0] - 0.1) / 0.2
        left_quantile = randomizer.sampled_coulomb_torque_nm[:, 1] / 0.5
        torch.testing.assert_close(right_quantile, left_quantile, atol=1.0e-6, rtol=0.0)

    def test_separate_actuator_groups_keep_mirrored_joint_samples_synchronized(self):
        model = _bilateral_model()
        right = EpisodeTorqueRandomizer(model, ("R_Leg_knee",), 8, "cpu", seed=23)
        left = EpisodeTorqueRandomizer(model, ("L_Leg_knee",), 8, "cpu", seed=23)

        torch.testing.assert_close(right.sampled_motor_strength_scale, left.sampled_motor_strength_scale)
        torch.testing.assert_close(
            (right.sampled_coulomb_torque_nm - 0.1) / 0.2,
            left.sampled_coulomb_torque_nm / 0.5,
            atol=1.0e-6,
            rtol=0.0,
        )
        right.reset(torch.tensor([1, 5]))
        left.reset(torch.tensor([1, 5]))
        torch.testing.assert_close(
            right.sampled_motor_strength_scale[[1, 5]],
            left.sampled_motor_strength_scale[[1, 5]],
        )

    def test_checked_in_split_actuator_layout_shares_all_bilateral_quantiles(self):
        model = load_rnd_torque_randomization(CONFIG_PATH, RND_LEG_JOINT_NAMES)
        fallback_names = ("L_Leg_ankle_roll",)
        primary_names = tuple(name for name in RND_LEG_JOINT_NAMES if name not in fallback_names)
        primary = EpisodeTorqueRandomizer(model, primary_names, 16, "cpu", seed=1_000_003)
        fallback = EpisodeTorqueRandomizer(model, fallback_names, 16, "cpu", seed=1_000_003)

        primary_indices = {name: index for index, name in enumerate(primary_names)}
        for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            right_name = f"R_Leg_{suffix}"
            left_name = f"L_Leg_{suffix}"
            right_index = primary_indices[right_name]
            left_randomizer = fallback if left_name in fallback_names else primary
            left_index = 0 if left_randomizer is fallback else primary_indices[left_name]

            torch.testing.assert_close(
                primary.sampled_motor_strength_scale[:, right_index],
                left_randomizer.sampled_motor_strength_scale[:, left_index],
            )
            torch.testing.assert_close(
                primary.sampled_transition_velocity_rad_s[:, right_index],
                left_randomizer.sampled_transition_velocity_rad_s[:, left_index],
            )
            right_range = model["joints"][right_name]["coulomb_torque_range_nm"]
            left_range = model["joints"][left_name]["coulomb_torque_range_nm"]
            right_quantile = (primary.sampled_coulomb_torque_nm[:, right_index] - right_range[0]) / (
                right_range[1] - right_range[0]
            )
            left_quantile = (left_randomizer.sampled_coulomb_torque_nm[:, left_index] - left_range[0]) / (
                left_range[1] - left_range[0]
            )
            torch.testing.assert_close(right_quantile, left_quantile, atol=1.0e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
