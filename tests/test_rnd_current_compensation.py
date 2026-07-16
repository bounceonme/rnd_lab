from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_current_compensation_build import DEFAULT_BASELINE, build_current_compensation_model
from rnd_real2sim.current_compensation import (
    CurrentCompensationError,
    evaluate_compensation_current_a,
    evaluate_compensation_current_raw,
    load_current_compensation_model,
    validate_current_compensation_model,
)


CANDIDATE_PATH = TOOLS_DIR / "config" / "rnd_current_compensation_candidate.json"


class CurrentCompensationBuildTest(unittest.TestCase):
    def test_build_is_analysis_only_and_preserves_unavailable_joint(self):
        model = build_current_compensation_model(DEFAULT_BASELINE)
        validate_current_compensation_model(model)

        self.assertFalse(model["integration_enabled"])
        self.assertFalse(model["hardware_write_enabled"])
        self.assertFalse(model["torque_conversion"]["available"])
        self.assertEqual(model["quality_summary"]["candidate_usable_joint_count"], 11)
        self.assertEqual(model["quality_summary"]["unavailable_joints"], ["R_Leg_ankle_pitch"])
        self.assertIsNone(model["joints"]["R_Leg_ankle_pitch"]["current_model"])
        self.assertTrue(model["joints"]["L_Leg_ankle_roll"]["quality"]["candidate_usable"])

    def test_checked_in_candidate_matches_current_baseline_except_timestamp(self):
        generated = build_current_compensation_model(DEFAULT_BASELINE)
        checked_in = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
        generated.pop("created_utc")
        checked_in.pop("created_utc")
        self.assertEqual(generated, checked_in)

    def test_source_hash_change_fails_closed(self):
        baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
        baseline["joints"]["R_Leg_knee"]["coulomb_current"]["provenance"][0]["model_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "baseline.json"
            path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after aggregation"):
                build_current_compensation_model(path)


class CurrentCompensationEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_current_compensation_model(DEFAULT_BASELINE)

    def test_zero_velocity_is_zero_and_law_is_odd(self):
        joint = "R_Leg_knee"
        positive = evaluate_compensation_current_a(self.model, joint, 0.1, gain=0.5)
        negative = evaluate_compensation_current_a(self.model, joint, -0.1, gain=0.5)
        zero = evaluate_compensation_current_a(self.model, joint, 0.0, gain=0.5)
        self.assertEqual(zero, 0.0)
        self.assertAlmostEqual(positive, -negative)

    def test_transition_velocity_reaches_nearly_full_coulomb_current(self):
        joint = "R_Leg_knee"
        current_model = self.model["joints"][joint]["current_model"]
        output = evaluate_compensation_current_a(
            self.model,
            joint,
            current_model["transition_velocity_rad_s"],
            gain=1.0,
        )
        expected = current_model["nominal_coulomb_current_a"] * math.tanh(4.0)
        self.assertAlmostEqual(output, expected)

    def test_raw_output_uses_mx_current_quantization(self):
        joint = "R_Leg_knee"
        current_model = self.model["joints"][joint]["current_model"]
        raw = evaluate_compensation_current_raw(
            self.model,
            joint,
            10.0 * current_model["transition_velocity_rad_s"],
            gain=1.0,
        )
        self.assertEqual(raw, current_model["nominal_goal_current_raw"])

    def test_unavailable_joint_and_excess_gain_fail_closed(self):
        with self.assertRaisesRegex(CurrentCompensationError, "no quality-gated"):
            evaluate_compensation_current_a(self.model, "R_Leg_ankle_pitch", 0.1, gain=0.25)
        with self.assertRaisesRegex(CurrentCompensationError, "gain must be"):
            evaluate_compensation_current_a(self.model, "R_Leg_knee", 0.1, gain=1.1)

    def test_validator_rejects_hardware_enabled_candidate(self):
        invalid = copy.deepcopy(self.model)
        invalid["hardware_write_enabled"] = True
        with self.assertRaisesRegex(CurrentCompensationError, "fail closed"):
            validate_current_compensation_model(invalid)

    def test_checked_in_candidate_loads(self):
        model = load_current_compensation_model(CANDIDATE_PATH)
        self.assertEqual(model["quality_summary"]["candidate_usable_joint_count"], 11)
