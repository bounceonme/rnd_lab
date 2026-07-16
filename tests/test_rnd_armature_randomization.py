from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "scripts" / "tools"
ROBOT_PACKAGE_DIR = ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROBOT_PACKAGE_DIR))

from actuators.rnd_armature_randomization import (
    RND_ARMATURE_JOINT_NAMES,
    RND_ARMATURE_MEASURED_JOINT_NAMES,
    RndArmatureRandomizationError,
    load_rnd_armature_randomization,
    sample_rnd_armatures,
    validate_rnd_armature_randomization,
)
from rnd_armature_randomization_build import (
    ArmatureRandomizationBuildError,
    build_armature_randomization,
)


CONFIG_PATH = TOOLS_DIR / "config" / "rnd_armature_randomization.json"
PRIMARY_REPORT_PATH = ROOT / "logs" / "rnd_real2sim" / "all_joints_armature_01_all_joint_armature.json"
FAILED_REPEAT_REPORT_PATH = (
    ROOT / "logs" / "rnd_real2sim" / "l_hip_pitch_armature_02_all_joint_armature.json"
)
EXPECTED_MEASURED_RANGES = {
    "R_Leg_hip_pitch": [0.02529998208456699, 0.04216663680761165],
    "R_Leg_knee": [0.011807110038759314, 0.019678516731265524],
    "L_Leg_knee": [0.01627245565552772, 0.027120759425879534],
}
EXPECTED_PRIOR_RANGE = [0.005, 0.04]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(report: dict | None = None, failed_repeat_report: dict | None = None) -> dict:
    return build_armature_randomization(
        _json(PRIMARY_REPORT_PATH) if report is None else report,
        _json(FAILED_REPEAT_REPORT_PATH) if failed_repeat_report is None else failed_repeat_report,
        report_path=PRIMARY_REPORT_PATH,
        report_sha256=_sha256(PRIMARY_REPORT_PATH),
        failed_repeat_report_path=FAILED_REPEAT_REPORT_PATH,
        failed_repeat_report_sha256=_sha256(FAILED_REPEAT_REPORT_PATH),
    )


class ArmatureRandomizationConfigTest(unittest.TestCase):
    def test_checked_in_contract_has_exact_three_measured_and_nine_priors(self):
        model = load_rnd_armature_randomization(CONFIG_PATH, RND_ARMATURE_JOINT_NAMES)

        self.assertEqual(model["joint_order"], list(RND_ARMATURE_JOINT_NAMES))
        self.assertEqual(model["quality_summary"]["measured_joint_count"], 3)
        self.assertEqual(model["quality_summary"]["unidentified_prior_joint_count"], 9)
        self.assertEqual(
            model["quality_summary"]["measured_joint_names"],
            list(RND_ARMATURE_MEASURED_JOINT_NAMES),
        )
        for joint_name, expected_range in EXPECTED_MEASURED_RANGES.items():
            joint = model["joints"][joint_name]
            self.assertEqual(joint["evidence_status"], "measured_quality_pass")
            self.assertEqual(joint["armature_range_kg_m2"], expected_range)

        for joint_name in set(RND_ARMATURE_JOINT_NAMES) - set(RND_ARMATURE_MEASURED_JOINT_NAMES):
            joint = model["joints"][joint_name]
            self.assertEqual(joint["evidence_status"], "unidentified_prior")
            self.assertEqual(joint["armature_range_kg_m2"], EXPECTED_PRIOR_RANGE)
            self.assertIsNone(joint["measured_armature_kg_m2"])
            self.assertIsNone(joint["bootstrap_90pct_kg_m2"])

        self.assertEqual(model["joints"]["L_Leg_hip_pitch"]["armature_range_kg_m2"], EXPECTED_PRIOR_RANGE)
        self.assertEqual(model["correlation_mode"], "global_shared_quantile")
        self.assertTrue(model["sample_on_startup"])
        self.assertFalse(model["sample_per_episode"])

    def test_checked_in_provenance_is_relative_hashed_and_non_promoting(self):
        model = load_rnd_armature_randomization(CONFIG_PATH)

        self.assertEqual(model["source_report"], PRIMARY_REPORT_PATH.relative_to(ROOT).as_posix())
        self.assertEqual(model["source_report_sha256"], _sha256(PRIMARY_REPORT_PATH))
        self.assertEqual(
            model["failed_repeat_report"],
            FAILED_REPEAT_REPORT_PATH.relative_to(ROOT).as_posix(),
        )
        self.assertEqual(model["failed_repeat_report_sha256"], _sha256(FAILED_REPEAT_REPORT_PATH))
        self.assertFalse(Path(model["source_report"]).is_absolute())
        self.assertFalse(Path(model["failed_repeat_report"]).is_absolute())
        self.assertEqual(model["integration_mode"], "opt_in_rl_training_randomization")
        self.assertFalse(model["physical_parameter_promotion"])
        self.assertFalse(model["failed_repeat_evidence"]["used_for_range"])
        self.assertTrue(any("does not promote" in item for item in model["limitations"]))

    def test_validator_rejects_invalid_measured_quality_claim(self):
        model = _json(CONFIG_PATH)
        model["joints"]["L_Leg_hip_pitch"].update(
            {
                "evidence_status": "measured_quality_pass",
                "source_quality_pass": True,
                "source_quality_reasons": [],
                "measured_armature_kg_m2": 0.007,
                "bootstrap_90pct_kg_m2": [0.006, 0.008],
                "armature_range_kg_m2": [0.00525, 0.00875],
                "range_method": "estimate +/-25%, expanded to contain the bootstrap 90% interval",
            }
        )
        with self.assertRaisesRegex(RndArmatureRandomizationError, "not approved"):
            validate_rnd_armature_randomization(model)

    def test_validator_rejects_measured_claim_without_source_quality(self):
        model = _json(CONFIG_PATH)
        model["joints"]["R_Leg_knee"]["source_quality_pass"] = False
        with self.assertRaisesRegex(RndArmatureRandomizationError, "source quality gates"):
            validate_rnd_armature_randomization(model)

    def test_validator_rejects_bounds_and_episode_resampling(self):
        model = _json(CONFIG_PATH)
        model["joints"]["R_Leg_hip_yaw"]["armature_range_kg_m2"] = [0.005, 0.11]
        with self.assertRaisesRegex(RndArmatureRandomizationError, "upper bound exceeds"):
            validate_rnd_armature_randomization(model)

        model = _json(CONFIG_PATH)
        model["sample_per_episode"] = True
        with self.assertRaisesRegex(RndArmatureRandomizationError, "sample_per_episode=false"):
            validate_rnd_armature_randomization(model)

    def test_validator_rejects_non_global_correlation_and_absolute_provenance(self):
        model = _json(CONFIG_PATH)
        model["correlation_mode"] = "bilateral_shared_quantile"
        with self.assertRaisesRegex(RndArmatureRandomizationError, "global_shared_quantile"):
            validate_rnd_armature_randomization(model)

        model = _json(CONFIG_PATH)
        model["source_report"] = str(PRIMARY_REPORT_PATH)
        with self.assertRaisesRegex(RndArmatureRandomizationError, "relative path"):
            validate_rnd_armature_randomization(model)


class ArmatureSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_rnd_armature_randomization(CONFIG_PATH)

    def test_random_samples_are_bounded_reproducible_and_have_expected_shape(self):
        first = sample_rnd_armatures(
            self.model,
            RND_ARMATURE_JOINT_NAMES,
            64,
            "cpu",
            seed=73,
        )
        second = sample_rnd_armatures(
            copy.deepcopy(self.model),
            RND_ARMATURE_JOINT_NAMES,
            64,
            "cpu",
            seed=73,
        )
        different = sample_rnd_armatures(
            self.model,
            RND_ARMATURE_JOINT_NAMES,
            64,
            "cpu",
            seed=74,
        )

        self.assertEqual(first.shape, (64, 12))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, different))
        ranges = torch.tensor(
            [self.model["joints"][name]["armature_range_kg_m2"] for name in RND_ARMATURE_JOINT_NAMES]
        )
        self.assertTrue(bool((first >= ranges[:, 0].unsqueeze(0)).all().item()))
        self.assertTrue(bool((first <= ranges[:, 1].unsqueeze(0)).all().item()))

    def test_one_normalized_quantile_is_shared_globally_per_environment(self):
        samples = sample_rnd_armatures(
            self.model,
            RND_ARMATURE_JOINT_NAMES,
            128,
            "cpu",
            seed=19,
            dtype=torch.float64,
        )
        ranges = torch.tensor(
            [self.model["joints"][name]["armature_range_kg_m2"] for name in RND_ARMATURE_JOINT_NAMES],
            dtype=torch.float64,
        )
        quantiles = (samples - ranges[:, 0].unsqueeze(0)) / (
            ranges[:, 1] - ranges[:, 0]
        ).unsqueeze(0)
        torch.testing.assert_close(
            quantiles,
            quantiles[:, :1].expand_as(quantiles),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_split_joint_calls_share_the_same_global_quantile(self):
        right = sample_rnd_armatures(
            self.model,
            ("R_Leg_hip_pitch",),
            32,
            "cpu",
            seed=31,
            dtype=torch.float64,
        )
        left = sample_rnd_armatures(
            self.model,
            ("L_Leg_knee",),
            32,
            "cpu",
            seed=31,
            dtype=torch.float64,
        )
        right_range = EXPECTED_MEASURED_RANGES["R_Leg_hip_pitch"]
        left_range = EXPECTED_MEASURED_RANGES["L_Leg_knee"]
        right_quantile = (right[:, 0] - right_range[0]) / (right_range[1] - right_range[0])
        left_quantile = (left[:, 0] - left_range[0]) / (left_range[1] - left_range[0])
        torch.testing.assert_close(right_quantile, left_quantile, atol=1.0e-12, rtol=0.0)

    def test_midpoint_mode_returns_per_joint_range_midpoints(self):
        names = ("R_Leg_hip_pitch", "R_Leg_knee", "L_Leg_hip_pitch")
        samples = sample_rnd_armatures(
            self.model,
            names,
            3,
            "cpu",
            seed=999,
            sample_randomization=False,
            dtype=torch.float64,
        )
        expected_row = torch.tensor(
            [
                sum(EXPECTED_MEASURED_RANGES["R_Leg_hip_pitch"]) / 2.0,
                sum(EXPECTED_MEASURED_RANGES["R_Leg_knee"]) / 2.0,
                sum(EXPECTED_PRIOR_RANGE) / 2.0,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(samples, expected_row.unsqueeze(0).expand(3, -1))

    def test_sampling_rejects_invalid_environment_counts_and_joint_names(self):
        for invalid_count in (0, -1, True, 1.5):
            with self.subTest(num_envs=invalid_count):
                with self.assertRaisesRegex(RndArmatureRandomizationError, "num_envs"):
                    sample_rnd_armatures(self.model, ("R_Leg_knee",), invalid_count, "cpu")
        with self.assertRaisesRegex(RndArmatureRandomizationError, "duplicates"):
            sample_rnd_armatures(self.model, ("R_Leg_knee", "R_Leg_knee"), 1, "cpu")
        with self.assertRaisesRegex(RndArmatureRandomizationError, "missing joints"):
            sample_rnd_armatures(self.model, ("not_a_joint",), 1, "cpu")


class ArmatureRandomizationBuilderTest(unittest.TestCase):
    def test_builder_is_reproducible_and_matches_checked_in_json(self):
        first = _build()
        second = _build()
        self.assertEqual(first, second)
        expected_text = json.dumps(first, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.assertEqual(CONFIG_PATH.read_text(encoding="utf-8"), expected_text)
        self.assertNotIn("created_utc", first)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "rnd_armature_randomization.json"
            completed = subprocess.run(
                [sys.executable, str(TOOLS_DIR / "rnd_armature_randomization_build.py"), "--output", str(output)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            self.assertEqual(output.read_bytes(), CONFIG_PATH.read_bytes())

    def test_builder_never_uses_failed_repeat_fit_values(self):
        failed_repeat = _json(FAILED_REPEAT_REPORT_PATH)
        failed_repeat["joints"]["L_Leg_hip_pitch"]["fit"].update(
            {
                "armature_kg_m2": 0.099,
                "bootstrap_90pct_kg_m2": [0.09, 0.1],
            }
        )
        model = _build(failed_repeat_report=failed_repeat)
        joint = model["joints"]["L_Leg_hip_pitch"]
        self.assertEqual(joint["evidence_status"], "unidentified_prior")
        self.assertEqual(joint["armature_range_kg_m2"], EXPECTED_PRIOR_RANGE)
        self.assertIsNone(joint["measured_armature_kg_m2"])
        self.assertIsNone(joint["bootstrap_90pct_kg_m2"])

    def test_builder_rejects_repeat_promoted_to_quality_pass(self):
        failed_repeat = _json(FAILED_REPEAT_REPORT_PATH)
        failed_repeat["joints"]["L_Leg_hip_pitch"]["quality"] = {"pass": True, "reasons": []}
        failed_repeat["summary"]["armature_passed_joints"] = ["L_Leg_hip_pitch"]
        failed_repeat["summary"]["armature_failed_joints"] = []
        with self.assertRaisesRegex(ArmatureRandomizationBuildError, "cannot be promoted"):
            _build(failed_repeat_report=failed_repeat)

    def test_builder_rejects_primary_summary_quality_disagreement(self):
        report = _json(PRIMARY_REPORT_PATH)
        report["summary"]["armature_passed_joints"].remove("R_Leg_knee")
        report["summary"]["armature_failed_joints"].append("R_Leg_knee")
        with self.assertRaisesRegex(ArmatureRandomizationBuildError, "disagree"):
            _build(report=report)


if __name__ == "__main__":
    unittest.main()
