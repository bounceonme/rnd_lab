from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_real2sim.armature_batch import (
    ArmatureDynamicsTrace,
    TorqueCalibrationReport,
    analyze_armature_joints,
)
from rnd_real2sim.dataset import Real2SimDataset


class ArmatureBatchTest(unittest.TestCase):
    def test_batch_fits_only_joint_with_quality_gated_torque_calibration(self):
        sample_hz = 50.0
        samples_per_phase = 500
        joint_names = ("joint_a", "joint_b")
        frequencies = (0.5, 1.0, 1.5)
        phase_count = len(joint_names) * len(frequencies)
        total_samples = phase_count * samples_per_phase
        phase_id = np.repeat(np.arange(phase_count), samples_per_phase)
        excitation_joint_id = np.repeat(
            np.repeat(np.arange(len(joint_names)), len(frequencies)), samples_per_phase
        )
        position = np.zeros((total_samples, len(joint_names)))
        goal = np.zeros_like(position)
        velocity = np.zeros_like(position)
        acceleration = np.zeros_like(position)
        current = np.zeros_like(position)
        modeled_torque = np.zeros_like(position)
        amplitude = math.radians(5.0)
        armature = (0.012, 0.02)
        torque_per_amp = (1.8, 1.6)
        calibration_bias = (-0.02, 0.01)
        phase_metadata = {}

        phase_index = 0
        for joint_index, joint_name in enumerate(joint_names):
            for frequency_hz in frequencies:
                start = phase_index * samples_per_phase
                stop = start + samples_per_phase
                time_s = np.arange(samples_per_phase) / sample_hz
                omega = 2.0 * math.pi * frequency_hz
                q = amplitude * np.sin(omega * time_s)
                qd = amplitude * omega * np.cos(omega * time_s)
                qdd = -amplitude * omega**2 * np.sin(omega * time_s)
                residual = (
                    armature[joint_index] * qdd
                    + 0.08 * np.tanh(qd / math.radians(4.0))
                    + 0.01 * qd
                    - 0.005
                )
                position[start:stop, joint_index] = q
                goal[start:stop, joint_index] = q
                velocity[start:stop, joint_index] = qd
                acceleration[start:stop, joint_index] = qdd
                current[start:stop, joint_index] = (
                    residual - calibration_bias[joint_index]
                ) / torque_per_amp[joint_index]
                phase_metadata[str(phase_index)] = {
                    "joint_name": joint_name,
                    "profile_name": f"armature_sine_{frequency_hz:.1f}hz",
                    "waveform": "sine",
                    "amplitude_rad": amplitude,
                    "frequency_hz": frequency_hz,
                    "cycles": round(10.0 * frequency_hz),
                }
                phase_index += 1

        dataset = Real2SimDataset(
            Path("synthetic_armature.npz"),
            {
                "joint_names": list(joint_names),
                "excitation_joint_names": list(joint_names),
                "phase_metadata": phase_metadata,
                "sample_hz": sample_hz,
            },
            {
                "time_s": np.arange(total_samples, dtype=np.float64) / sample_hz,
                "phase_id": phase_id,
                "excitation_joint_id": excitation_joint_id,
                "goal_position_rad": goal,
                "position_rad": position,
                "current_a": current,
            },
            "synthetic-dataset-sha",
        )
        trace = ArmatureDynamicsTrace(
            Path("synthetic_trace.npz"),
            {},
            {
                "smoothed_velocity_rad_s": velocity,
                "smoothed_acceleration_rad_s2": acceleration,
                "modeled_urdf_torque_nm": modeled_torque,
            },
            "synthetic-trace-sha",
        )
        calibration = TorqueCalibrationReport(
            Path("synthetic_calibration.json"),
            {
                "joints": {
                    "joint_a": {
                        "low_current_torque_calibration": {
                            "torque_per_amp_nm": torque_per_amp[0],
                            "bias_nm": calibration_bias[0],
                            "quality": {"pass": True, "reasons": []},
                        }
                    },
                    "joint_b": {
                        "low_current_torque_calibration": {
                            "quality": {"pass": False, "reasons": ["unobservable gravity span"]}
                        }
                    },
                }
            },
            "synthetic-calibration-sha",
        )

        results, summary = analyze_armature_joints(dataset, trace, calibration, joint_names)
        self.assertTrue(results["joint_a"]["quality"]["pass"])
        self.assertEqual(results["joint_a"]["selected_estimator"], "cycle_harmonic_acceleration_projection")
        self.assertAlmostEqual(results["joint_a"]["fit"]["armature_kg_m2"], armature[0], places=4)
        self.assertAlmostEqual(results["joint_a"]["time_domain_fit"]["armature_kg_m2"], armature[0], places=4)
        self.assertFalse(results["joint_b"]["quality"]["pass"])
        self.assertIn("Current-to-torque calibration is not valid", results["joint_b"]["quality"]["reasons"][0])
        self.assertEqual(summary["armature_passed_joints"], ["joint_a"])
        self.assertEqual(summary["armature_failed_joints"], ["joint_b"])
        self.assertFalse(summary["automatic_integration_allowed"])


if __name__ == "__main__":
    unittest.main()
