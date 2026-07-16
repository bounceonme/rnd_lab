from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_real2sim.torque_identification import (
    TorqueIdentificationError,
    current_to_output_torque,
    estimate_joint_kinematics,
    fit_armature_residual,
    fit_friction_residual,
    fit_harmonic_armature_cycles,
    fit_quasistatic_gravity_calibration,
    load_torque_lut,
    validate_torque_lut,
)


LUT_PATH = TOOLS_DIR / "config" / "mx106_performance_lut.json"


class TorqueLutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_torque_lut(LUT_PATH)

    def test_checked_in_lut_has_official_source_provenance(self):
        validate_torque_lut(self.model)
        self.assertTrue(self.model["analysis_only"])
        self.assertEqual(
            self.model["source"]["image_sha256"],
            "a09370f6c5c5ed7d13070d9bc2233ebc323f0d5dc6db8b4d44487ade30c76291",
        )
        self.assertEqual(self.model["source"]["image_width_px"], 389)
        self.assertEqual(self.model["source"]["image_height_px"], 381)
        self.assertGreaterEqual(len(self.model["digitization"]["native_curve"]["torque_nm"]), 200)
        self.assertLessEqual(self.model["digitization"]["output_torque_step_nm"], 0.05)

    def test_conversion_is_odd_and_marks_unobserved_domains(self):
        first_current = self.model["curve"]["current_a"][0]
        last_current = self.model["curve"]["current_a"][-1]
        values = np.asarray([0.0, 0.5 * first_current, first_current, -first_current, 2.0 * last_current])
        converted = current_to_output_torque(values, self.model)
        self.assertEqual(converted.torque_nm[0], 0.0)
        self.assertAlmostEqual(converted.torque_nm[2], -converted.torque_nm[3])
        self.assertTrue(converted.below_observed_curve[1])
        self.assertFalse(converted.below_observed_curve[2])
        self.assertTrue(converted.above_observed_curve[-1])
        self.assertAlmostEqual(converted.torque_nm[-1], self.model["curve"]["torque_nm"][-1])


class DynamicResidualTest(unittest.TestCase):
    def test_savgol_derivatives_recover_slow_sine(self):
        sample_hz = 50.0
        time_s = np.arange(500) / sample_hz
        omega = 2.0 * math.pi * 0.5
        position = (0.2 * np.sin(omega * time_s))[:, None]
        velocity, acceleration = estimate_joint_kinematics(position, sample_hz, window_length=11, polynomial_order=3)
        interior = slice(10, -10)
        expected_velocity = 0.2 * omega * np.cos(omega * time_s)
        expected_acceleration = -0.2 * omega**2 * np.sin(omega * time_s)
        self.assertLess(float(np.sqrt(np.mean((velocity[interior, 0] - expected_velocity[interior]) ** 2))), 0.002)
        self.assertLess(
            float(np.sqrt(np.mean((acceleration[interior, 0] - expected_acceleration[interior]) ** 2))),
            0.02,
        )

    def test_robust_fit_recovers_coulomb_viscous_and_bias(self):
        velocity = np.linspace(-1.0, 1.0, 1001)
        transition = 0.08
        expected = {
            "coulomb_nm": 0.18,
            "viscous_nm_per_rad_s": 0.035,
            "bias_nm": -0.012,
        }
        residual = (
            expected["coulomb_nm"] * np.tanh(velocity / transition)
            + expected["viscous_nm_per_rad_s"] * velocity
            + expected["bias_nm"]
        )
        residual[500] += 2.0
        result = fit_friction_residual(
            residual,
            velocity,
            np.ones_like(velocity, dtype=np.bool_),
            transition_velocity_rad_s=transition,
            minimum_speed_rad_s=0.02,
        )
        self.assertAlmostEqual(result["coulomb_nm"], expected["coulomb_nm"], places=3)
        self.assertAlmostEqual(result["viscous_nm_per_rad_s"], expected["viscous_nm_per_rad_s"], places=3)
        self.assertAlmostEqual(result["bias_nm"], expected["bias_nm"], places=3)
        self.assertTrue(result["optimizer_success"])

    def test_multifrequency_fit_recovers_residual_armature(self):
        sample_hz = 50.0
        amplitude = math.radians(5.0)
        frequencies = (0.5, 1.0, 1.5)
        velocity_blocks = []
        acceleration_blocks = []
        for frequency_hz in frequencies:
            time_s = np.arange(500) / sample_hz
            omega = 2.0 * math.pi * frequency_hz
            velocity_blocks.append(amplitude * omega * np.cos(omega * time_s))
            acceleration_blocks.append(-amplitude * omega**2 * np.sin(omega * time_s))
        velocity = np.concatenate(velocity_blocks)
        acceleration = np.concatenate(acceleration_blocks)
        expected_armature = 0.012
        transition = math.radians(4.0)
        residual = (
            expected_armature * acceleration
            + 0.08 * np.tanh(velocity / transition)
            + 0.015 * velocity
            - 0.01
        )
        residual[100] += 1.5
        result = fit_armature_residual(
            residual,
            velocity,
            acceleration,
            np.ones_like(residual, dtype=np.bool_),
            transition_velocity_rad_s=transition,
        )
        self.assertAlmostEqual(result["armature_kg_m2"], expected_armature, places=4)
        self.assertAlmostEqual(result["coulomb_nm"], 0.08, places=3)
        self.assertAlmostEqual(result["viscous_nm_per_rad_s"], 0.015, places=3)
        self.assertGreater(result["rmse_improvement_over_friction_only"], 0.25)
        self.assertTrue(result["optimizer_success"])

    def test_cycle_harmonics_separate_armature_from_position_error(self):
        rng = np.random.default_rng(7)
        frequency = np.repeat((0.5, 1.0, 1.5), (5, 10, 15)).astype(np.float64)
        omega = 2.0 * math.pi * frequency
        acceleration = np.repeat((0.85, 3.35, 7.45), (5, 10, 15)) * (1.0 + 0.01 * rng.normal(size=30))
        position = -acceleration / np.square(omega)
        expected_armature = 0.021
        position_error = 0.28
        residual = expected_armature * acceleration + position_error * position + rng.normal(0.0, 0.002, size=30)

        result = fit_harmonic_armature_cycles(
            residual,
            acceleration,
            position,
            frequency,
            bootstrap_samples=300,
            seed=3,
        )

        self.assertAlmostEqual(result["armature_kg_m2"], expected_armature, places=3)
        self.assertGreater(result["bootstrap_90pct_kg_m2"][0], 0.0)
        self.assertLess(result["bootstrap_relative_interval_width"], 0.5)
        self.assertEqual(result["reliable_frequency_count"], 2)
        self.assertLess(result["reliable_frequency_relative_span"], 0.1)
        self.assertGreater(result["r2"], 0.9)

    def test_cycle_harmonics_expose_frequency_inconsistency(self):
        frequency = np.repeat((0.5, 1.0, 1.5), (5, 10, 15)).astype(np.float64)
        omega = 2.0 * math.pi * frequency
        acceleration = np.repeat((0.85, 3.35, 7.45), (5, 10, 15)).astype(np.float64)
        position = -acceleration / np.square(omega)
        frequency_dependent_armature = np.choose(
            np.searchsorted((0.5, 1.0, 1.5), frequency),
            (0.02, 0.0, 0.1),
        )
        residual = frequency_dependent_armature * acceleration + 0.2 * position

        result = fit_harmonic_armature_cycles(
            residual,
            acceleration,
            position,
            frequency,
            bootstrap_samples=200,
        )

        self.assertGreater(result["reliable_frequency_relative_span"], 0.5)

    def test_cycle_harmonics_reject_non_integer_bootstrap_count(self):
        values = np.ones(9, dtype=np.float64)
        frequency = np.repeat((0.5, 1.0, 1.5), 3)
        with self.assertRaisesRegex(TorqueIdentificationError, "bootstrap_samples"):
            fit_harmonic_armature_cycles(
                values,
                values,
                values,
                frequency,
                bootstrap_samples=100.5,
            )

    def test_lut_json_is_strict_json(self):
        model = json.loads(LUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(model["curve"]["torque_nm"][0], 0.8)
        self.assertEqual(model["curve"]["torque_nm"][-1], 5.55)

    def test_quasistatic_gravity_calibration_recovers_torque_constant(self):
        sample_count = 5000
        time_s = np.arange(sample_count) / 50.0
        position = 0.35 * np.sin(2.0 * math.pi * 0.02 * time_s)
        velocity = np.gradient(position, 1.0 / 50.0)
        gravity = 0.42 * np.sin(position + 0.3)
        torque_per_amp = 1.18
        coulomb_current = 0.045
        current = (gravity - 0.012) / torque_per_amp + coulomb_current * np.sign(velocity)
        result = fit_quasistatic_gravity_calibration(
            position,
            current,
            velocity,
            gravity,
            np.ones(sample_count, dtype=np.bool_),
            minimum_speed_rad_s=math.radians(0.2),
            bootstrap_samples=200,
        )
        self.assertAlmostEqual(result["torque_per_amp_nm"], torque_per_amp, places=2)
        self.assertAlmostEqual(result["coulomb_current_a"], coulomb_current, places=3)
        self.assertTrue(result["quality"]["pass"])


if __name__ == "__main__":
    unittest.main()
