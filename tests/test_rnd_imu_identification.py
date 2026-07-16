from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_imu.config import load_imu_identification_config
from rnd_imu.collector import BaudProbeResult, ImuCollectionError, select_probe_result
from rnd_imu.identification import identify_imu_dataset, load_imu_dataset, save_imu_dataset


CONFIG_PATH = TOOLS_DIR / "config" / "rnd_cmp10a.toml"


def _record(timestamp_ns: int, stage: str, frame_type: int, **values):
    return {
        "timestamp_ns": timestamp_ns,
        "stage": stage,
        "frame_type": frame_type,
        "raw": bytes([0x55, frame_type, 0, 0, 0, 0, 0, 0, 0, 0, (0x55 + frame_type) & 0xFF]),
        **values,
    }


class RndImuIdentificationTest(unittest.TestCase):
    def test_config_matches_policy_rate(self):
        config = load_imu_identification_config(CONFIG_PATH)
        self.assertEqual(config.experiment.policy_hz, 50.0)
        self.assertGreaterEqual(config.quality.minimum_required_rate_hz, config.experiment.policy_hz)
        self.assertIn(921600, config.serial.baud_candidates)
        self.assertIn(9600, config.serial.baud_candidates)

    def test_baud_selection_requires_a_clear_checksum_valid_stream(self):
        selected = select_probe_result(
            (
                BaudProbeResult(9600, 0, 4, 100, {}),
                BaudProbeResult(921600, 150, 0, 0, {0x51: 50, 0x52: 50, 0x53: 50}),
            ),
            minimum_valid_frames=5,
        )
        self.assertEqual(selected.baudrate, 921600)
        with self.assertRaises(ImuCollectionError):
            select_probe_result((BaudProbeResult(9600, 2, 0, 0, {0x52: 2}),), minimum_valid_frames=5)

    def test_dataset_round_trip_and_axis_identification(self):
        config = load_imu_identification_config(CONFIG_PATH)
        records = []
        dt_ns = 10_000_000
        gyro_bias = np.array([0.01, -0.02, 0.005])
        yaw = np.deg2rad(20.0)
        sensor_to_base = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        accel_sensor = sensor_to_base.T @ np.array([0.0, 0.0, config.experiment.gravity_mps2])

        for index in range(200):
            timestamp = index * dt_ns
            records.append(_record(timestamp, "static_upright", 0x51, accel_mps2=accel_sensor))
            records.append(_record(timestamp + 1, "static_upright", 0x52, gyro_rad_s=gyro_bias))
            records.append(_record(timestamp + 2, "static_upright", 0x53, euler_rad=np.zeros(3)))

        start = 3_000_000_000
        for base_axis, stage in enumerate(("axis_pos_x", "axis_pos_y", "axis_pos_z")):
            gyro_sensor = sensor_to_base.T @ np.eye(3)[base_axis]
            for index in range(101):
                records.append(_record(start + index * dt_ns, stage, 0x52, gyro_rad_s=gyro_sensor + gyro_bias))
            start += 2_000_000_000

        metadata = {
            "parser_stats": {"valid_frames": len(records), "checksum_errors": 0},
            "upper_body_mount": True,
            "mount_location": "top_of_Upper_Body",
            "mount_translation_m": None,
            "base_link_to_upper_body_joint": "fixed",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imu.npz"
            save_imu_dataset(path, records, metadata)
            data = load_imu_dataset(path)

        report = identify_imu_dataset(data, config)
        np.testing.assert_allclose(report["static_gyro"]["bias_rad_s"], gyro_bias, atol=1.0e-12)
        np.testing.assert_allclose(
            report["mount_axis_identification"]["sensor_to_base_matrix"], sensor_to_base, atol=1.0e-12
        )
        np.testing.assert_allclose(report["static_projected_gravity_b_from_accel"], [0.0, 0.0, -1.0], atol=1.0e-12)
        self.assertEqual(report["mount_context"]["location"], "top_of_Upper_Body")
        self.assertFalse(report["mount_context"]["translation_used_for_this_identification"])
        self.assertEqual(report["runtime_gate"]["orientation_source"], "euler_angle")
        self.assertTrue(report["runtime_gate"]["quality_pass"])


if __name__ == "__main__":
    unittest.main()
