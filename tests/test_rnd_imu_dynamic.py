from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_imu.dynamic import DYNAMIC_AXIS_STAGES, analyze_dynamic_imu_dataset


def _trajectory(timestamp_s: np.ndarray, motion_axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    frequency_hz = 0.8
    phase = 2.0 * np.pi * frequency_hz * timestamp_s
    angular_rate = 2.0 * np.pi * frequency_hz
    euler = np.column_stack((
        np.full_like(timestamp_s, 0.30),
        np.full_like(timestamp_s, -0.30),
        np.full_like(timestamp_s, np.pi - 0.12),
    ))
    euler_rate = np.zeros_like(euler)
    if motion_axis is not None:
        amplitudes = (0.55, 0.48, 0.52)
        euler[:, motion_axis] += amplitudes[motion_axis] * np.sin(phase)
        euler_rate[:, motion_axis] = amplitudes[motion_axis] * angular_rate * np.cos(phase)
    return euler, euler_rate


def _body_angular_velocity(euler: np.ndarray, euler_rate: np.ndarray) -> np.ndarray:
    roll = euler[:, 0]
    pitch = euler[:, 1]
    return np.column_stack((
        euler_rate[:, 0] - np.sin(pitch) * euler_rate[:, 2],
        np.cos(roll) * euler_rate[:, 1] + np.sin(roll) * np.cos(pitch) * euler_rate[:, 2],
        -np.sin(roll) * euler_rate[:, 1] + np.cos(roll) * np.cos(pitch) * euler_rate[:, 2],
    ))


def _dynamic_dataset(
    delays_ms: tuple[float, float, float] = (20.0, 30.0, 40.0),
    motion_axes: tuple[int | None, int | None, int | None] = (0, 1, 2),
) -> dict:
    rows = []
    duration_s = 6.0
    gyro_timestamp_s = np.arange(0.0, duration_s, 0.01)
    euler_timestamp_s = np.arange(0.003, duration_s, 0.0125)
    for stage_index, stage in enumerate(DYNAMIC_AXIS_STAGES):
        offset_ns = stage_index * 10_000_000_000
        motion_axis = motion_axes[stage_index]
        gyro_euler, gyro_euler_rate = _trajectory(gyro_timestamp_s, motion_axis)
        gyro = _body_angular_velocity(gyro_euler, gyro_euler_rate)
        delayed_euler, _ = _trajectory(euler_timestamp_s - delays_ms[stage_index] * 1.0e-3, motion_axis)
        wrapped_euler = (delayed_euler + np.pi) % (2.0 * np.pi) - np.pi

        for timestamp_s, value in zip(gyro_timestamp_s, gyro):
            rows.append((
                offset_ns + int(round(timestamp_s * 1.0e9)),
                stage,
                0x52,
                value,
                np.full(3, np.nan),
            ))
        for timestamp_s, value in zip(euler_timestamp_s, wrapped_euler):
            rows.append((
                offset_ns + int(round(timestamp_s * 1.0e9)),
                stage,
                0x53,
                np.full(3, np.nan),
                value,
            ))

    rows.sort(key=lambda row: row[0])
    return {
        "timestamp_ns": np.asarray([row[0] for row in rows], dtype=np.int64),
        "stage": np.asarray([row[1] for row in rows], dtype="U32"),
        "frame_type": np.asarray([row[2] for row in rows], dtype=np.uint8),
        "gyro_rad_s": np.stack([row[3] for row in rows]),
        "euler_rad": np.stack([row[4] for row in rows]),
        "metadata": {"sensor": "synthetic CMP10A"},
    }


class RndImuDynamicTest(unittest.TestCase):
    def test_recovers_relative_delay_for_all_axes_with_yaw_wrap(self):
        delays_ms = (20.0, 30.0, 40.0)
        data = _dynamic_dataset(delays_ms=delays_ms)
        yaw_mask = (data["stage"] == "dynamic_axis_z") & (data["frame_type"] == 0x53)
        self.assertGreater(np.ptp(data["euler_rad"][yaw_mask, 2]), 6.0)

        report = analyze_dynamic_imu_dataset(data)

        self.assertTrue(report["quality_pass"])
        self.assertFalse(report["absolute_usb_latency"])
        self.assertIn("not absolute USB latency", report["latency_note"])
        for stage, expected_delay_ms in zip(DYNAMIC_AXIS_STAGES, delays_ms):
            result = report["stages"][stage]
            self.assertTrue(result["quality_pass"], result["reason"])
            self.assertGreater(result["correlation"], 0.995)
            self.assertGreater(result["dominant_axis_rms_ratio"], 2.0)
            self.assertAlmostEqual(result["gain_ratio"], 1.0, delta=0.02)
            self.assertLessEqual(
                abs(result["delay_ms"] - expected_delay_ms),
                result["sample_period_ms"] + 1.0e-9,
            )
        self.assertAlmostEqual(report["median_relative_delay_ms"], 30.0, delta=10.0)
        json.dumps(report, allow_nan=False)

    def test_rejects_no_motion_and_cross_axis_motion_from_aggregate(self):
        data = _dynamic_dataset(motion_axes=(None, 0, 2))

        report = analyze_dynamic_imu_dataset(data)

        no_motion = report["stages"]["dynamic_axis_x"]
        cross_axis = report["stages"]["dynamic_axis_y"]
        passing = report["stages"]["dynamic_axis_z"]
        self.assertFalse(no_motion["quality_pass"])
        self.assertLess(no_motion["expected_axis_rms_rad_s"], 0.05)
        self.assertFalse(cross_axis["quality_pass"])
        self.assertLess(cross_axis["dominant_axis_rms_ratio"], 2.0)
        self.assertTrue(passing["quality_pass"], passing["reason"])
        self.assertEqual(report["passing_stages"], ["dynamic_axis_z"])
        self.assertAlmostEqual(report["median_relative_delay_ms"], passing["delay_ms"])
        self.assertFalse(report["quality_pass"])

    def test_nonfinite_and_insufficient_data_fail_without_non_json_values(self):
        data = _dynamic_dataset()
        euler_indices = np.flatnonzero((data["stage"] == "dynamic_axis_x") & (data["frame_type"] == 0x53))
        data["euler_rad"][euler_indices[10], 0] = np.nan

        report = analyze_dynamic_imu_dataset(data)

        self.assertFalse(report["stages"]["dynamic_axis_x"]["quality_pass"])
        self.assertIn("non-finite", report["stages"]["dynamic_axis_x"]["reason"])
        self.assertTrue(report["stages"]["dynamic_axis_y"]["quality_pass"])
        json.dumps(report, allow_nan=False)

        empty_report = analyze_dynamic_imu_dataset({
            "timestamp_ns": np.asarray([], dtype=np.int64),
            "stage": np.asarray([], dtype="U32"),
            "frame_type": np.asarray([], dtype=np.uint8),
            "gyro_rad_s": np.empty((0, 3)),
            "euler_rad": np.empty((0, 3)),
        })
        self.assertIsNone(empty_report["median_relative_delay_ms"])
        self.assertTrue(all(not result["quality_pass"] for result in empty_report["stages"].values()))
        json.dumps(empty_report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
