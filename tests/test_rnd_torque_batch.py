from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_real2sim.dataset import Real2SimDataset
from rnd_real2sim.torque_batch import DynamicsTrace, analyze_joints


class TorqueBatchTest(unittest.TestCase):
    def test_each_joint_uses_only_its_own_phase(self):
        sample_hz = 50.0
        phase_samples = 5000
        joint_names = ("joint_a", "joint_b")
        phase_id = np.repeat(np.arange(2), phase_samples)
        excitation_joint_id = phase_id.copy()
        position = np.zeros((2 * phase_samples, 2))
        velocity = np.zeros_like(position)
        current = np.zeros_like(position)
        gravity = np.zeros_like(position)
        residual = np.zeros_like(position)
        expected_kt = (1.18, 1.46)
        for joint_index in range(2):
            phase_slice = slice(joint_index * phase_samples, (joint_index + 1) * phase_samples)
            time_s = np.arange(phase_samples) / sample_hz
            q = 0.35 * np.sin(2.0 * math.pi * 0.02 * time_s)
            qd = np.gradient(q, 1.0 / sample_hz)
            tau_g = 0.42 * np.sin(q + 0.3)
            position[phase_slice, joint_index] = q
            velocity[phase_slice, joint_index] = qd
            gravity[phase_slice, joint_index] = tau_g
            current[phase_slice, joint_index] = (tau_g - 0.012) / expected_kt[joint_index] + 0.04 * np.sign(qd)
            residual[phase_slice, joint_index] = 0.12 * np.tanh(qd / math.radians(4.0)) + 0.01 * qd

        metadata = {
            "joint_names": list(joint_names),
            "phase_metadata": {
                str(index): {
                    "joint_name": name,
                    "profile_name": "quasistatic_gravity_sine",
                    "waveform": "sine",
                    "amplitude_rad": math.radians(20.0),
                    "frequency_hz": 0.02,
                }
                for index, name in enumerate(joint_names)
            },
        }
        dataset = Real2SimDataset(
            Path("synthetic.npz"),
            metadata,
            {
                "time_s": np.arange(2 * phase_samples, dtype=np.float64) / sample_hz,
                "phase_id": phase_id,
                "excitation_joint_id": excitation_joint_id,
                "position_rad": position,
                "current_a": current,
            },
            "synthetic-dataset-sha",
        )
        trace = DynamicsTrace(
            Path("synthetic_trace.npz"),
            {},
            {
                "smoothed_velocity_rad_s": velocity,
                "gravity_torque_nm": gravity,
                "friction_residual_torque_nm": residual,
                "low_current_extrapolation_mask": np.zeros_like(position, dtype=np.bool_),
                "high_current_clipping_mask": np.zeros_like(position, dtype=np.bool_),
            },
            "synthetic-trace-sha",
        )
        results, summary = analyze_joints(dataset, trace, joint_names)
        for joint_index, joint_name in enumerate(joint_names):
            calibration = results[joint_name]["low_current_torque_calibration"]
            self.assertAlmostEqual(calibration["torque_per_amp_nm"], expected_kt[joint_index], places=2)
            self.assertTrue(calibration["quality"]["pass"])
        self.assertEqual(summary["calibration_passed_joints"], list(joint_names))
        self.assertFalse(summary["automatic_integration_allowed"])


if __name__ == "__main__":
    unittest.main()
