from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "scripts" / "tools"
HARDWARE_DIR = ROOT / "source" / "robot_lab" / "robot_lab" / "hardware"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(HARDWARE_DIR))

from cmp10a_runtime import validate_cmp10a_runtime_model

from rnd_imu.runtime_config import (
    CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD,
    CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S,
    CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S,
    CMP10A_ORIENTATION_DELAY_RANGE_S,
    CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S,
    CMP10A_RUNTIME_SIGNED_MAPPING,
    Cmp10aRuntimeConfigError,
    build_cmp10a_runtime_config,
    build_cmp10a_runtime_config_from_files,
    validate_cmp10a_runtime_config,
)


CONFIG_PATH = TOOLS_DIR / "config" / "rnd_cmp10a_runtime.json"
STATIC_REPORT_PATH = ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_20260716_035739_report.json"
DYNAMIC_REPORT_PATH = ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_dynamic_20260716_042138_report.json"
DYNAMIC_DATASET_PATH = ROOT / "logs" / "rnd_imu" / "rnd_cmp10a_dynamic_20260716_042138.npz"

EXPECTED_STATIC_SHA256 = "6c8f95e3052c9eb6ef1ee19afd50bc5563a05009debabf2d3db8e3a5a74e5317"
EXPECTED_DYNAMIC_SHA256 = "f83b1414172d685e523a08c135a57e340c060e12a379d404d8925821ecdf7867"
EXPECTED_DATASET_SHA256 = "22000017005a43331bcf304f11bcf0d0c116a87d5d5784f8e8d0a79b61f9928c"


def _checked_in() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _synthetic_fixture() -> dict:
    static_sha256 = "1" * 64
    mapping = [list(row) for row in CMP10A_RUNTIME_SIGNED_MAPPING]
    static_report = {
        "schema_version": 1,
        "policy_hz": 50.0,
        "runtime_gate": {
            "quality_pass": True,
            "gyro_rate_hz": 200.5,
            "orientation_rate_hz": 200.4,
            "orientation_source": "euler_angle",
        },
        "mount_axis_identification": {
            "quality_pass": True,
            "signed_axis_approximation": mapping,
            "sensor_to_base_matrix": [
                [-0.999, 0.02, 0.03],
                [-0.02, -0.999, -0.03],
                [0.03, -0.03, 0.999],
            ],
            "axis_fit_error_deg": [4.0, 5.0, 4.5],
            "determinant": 1.0,
            "source_direction_condition_number": 1.2,
        },
    }
    axis_metrics = {
        "dynamic_axis_x": (-5.0, 0.94),
        "dynamic_axis_y": (10.0, 0.95),
        "dynamic_axis_z": (10.0, 1.01),
    }
    dynamic_report = {
        "schema_version": 1,
        "policy_hz": 50.0,
        "quality_pass": True,
        "absolute_usb_latency": False,
        "delay_definition": "Positive delay_ms means Euler lags gyro.",
        "median_relative_delay_ms": 10.0,
        "passing_stages": list(axis_metrics),
        "passing_stage_count": 3,
        "communication": {
            "quality_pass": True,
            "valid_frames": 10,
            "checksum_failures": 0,
            "garbage_bytes": 0,
        },
        "source": {
            "experiment": "cmp10a_dynamic_consistency",
            "device": "/dev/synthetic-cmp10a",
            "baudrate": 921600,
            "identification_report_sha256": static_sha256,
            "sensor_to_base_signed_mapping": mapping,
            "mount_location": "top_of_Upper_Body",
        },
        "stages": {
            stage: {
                "quality_pass": True,
                "delay_ms": delay,
                "gain_ratio": gain,
                "correlation": 0.99,
            }
            for stage, (delay, gain) in axis_metrics.items()
        },
    }

    sample_timestamps = np.arange(5, dtype=np.int64) * 500_000_000
    timestamps = np.repeat(sample_timestamps, 2)
    frame_types = np.tile(np.asarray([0x52, 0x53], dtype=np.uint8), sample_timestamps.size)
    rows = timestamps.shape[0]
    gyro = np.column_stack((
        np.linspace(-0.002, 0.002, rows),
        np.linspace(0.001, -0.001, rows),
        np.linspace(0.0, 0.0009, rows),
    ))
    euler = np.column_stack((
        np.linspace(0.01, 0.011, rows),
        np.linspace(-0.02, -0.018, rows),
        np.linspace(0.20, 0.201, rows),
    ))
    dynamic_dataset = {
        "timestamp_ns": timestamps,
        "stage": np.full(rows, "dynamic_baseline", dtype="U32"),
        "frame_type": frame_types,
        "gyro_rad_s": gyro,
        "euler_rad": euler,
        "metadata": {
            "schema_version": 1,
            "experiment": "cmp10a_dynamic_consistency",
            "device": "/dev/synthetic-cmp10a",
            "baudrate": 921600,
            "identification_report_sha256": static_sha256,
            "sensor_to_base_signed_mapping": mapping,
            "mount_location": "top_of_Upper_Body",
            "parser_stats": {
                "valid_frames": 10,
                "checksum_failures": 0,
                "garbage_bytes": 0,
            },
        },
    }
    return {
        "static_report": static_report,
        "dynamic_report": dynamic_report,
        "dynamic_dataset": dynamic_dataset,
        "static_report_path": "logs/rnd_imu/synthetic_static_report.json",
        "static_report_sha256": static_sha256,
        "dynamic_report_path": "logs/rnd_imu/synthetic_dynamic_report.json",
        "dynamic_report_sha256": "2" * 64,
        "dynamic_dataset_path": "logs/rnd_imu/synthetic_dynamic.npz",
        "dynamic_dataset_sha256": "3" * 64,
    }


def _build_synthetic(fixture: dict | None = None, *, created_utc: str = "2026-07-16T00:00:00+00:00") -> dict:
    values = _synthetic_fixture() if fixture is None else fixture
    return build_cmp10a_runtime_config(**values, created_utc=created_utc)


class RndImuCheckedInRuntimeConfigTest(unittest.TestCase):
    def test_checked_in_model_validates_without_local_provenance_files(self):
        model = _checked_in()

        validate_cmp10a_runtime_config(model)
        validate_cmp10a_runtime_model(model)

        self.assertEqual(model["schema_version"], 1)
        self.assertEqual(model["model_type"], "rnd_cmp10a_policy_observation")
        self.assertTrue(model["integration_enabled"])
        self.assertEqual(model["source_static_report_sha256"], EXPECTED_STATIC_SHA256)
        self.assertEqual(model["source_dynamic_report_sha256"], EXPECTED_DYNAMIC_SHA256)
        self.assertEqual(model["source_dynamic_dataset_sha256"], EXPECTED_DATASET_SHA256)
        for key in ("source_static_report", "source_dynamic_report", "source_dynamic_dataset"):
            self.assertFalse(Path(model[key]).is_absolute())
            self.assertNotIn("..", Path(model[key]).parts)

    def test_runtime_matrix_and_assumed_ranges_are_exact(self):
        model = _checked_in()

        self.assertEqual(
            model["sensor_to_base_matrix"],
            [list(row) for row in CMP10A_RUNTIME_SIGNED_MAPPING],
        )
        self.assertEqual(
            model["runtime_transform"]["sensor_to_base_matrix"],
            [list(row) for row in CMP10A_RUNTIME_SIGNED_MAPPING],
        )
        fit = model["measured"]["mount_fit_evidence"]
        self.assertFalse(fit["applied_at_runtime"])
        self.assertNotEqual(fit["sensor_to_base_fitted_matrix"], model["runtime_transform"]["sensor_to_base_matrix"])

        assumed = model["assumed_simulation_envelopes"]
        self.assertEqual(assumed["evidence_status"], "assumed_not_measured")
        self.assertEqual(
            assumed["residual_gyro_episode_bias"]["range_rad_s_per_axis"],
            list(CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S),
        )
        self.assertEqual(
            assumed["gyro_white_noise"]["sigma_range_rad_s"],
            list(CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S),
        )
        self.assertEqual(
            assumed["projected_gravity_tangent_angle_noise"]["sigma_range_rad"],
            list(CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD),
        )
        self.assertEqual(
            assumed["gyro_sample_age_delay"]["range_s"],
            list(CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S),
        )
        self.assertEqual(
            assumed["orientation_delay"]["range_s"],
            list(CMP10A_ORIENTATION_DELAY_RANGE_S),
        )

    def test_measured_baseline_timing_gains_and_unmeasured_fields_are_preserved(self):
        model = _checked_in()
        measured = model["measured"]
        baseline = measured["held_baseline"]

        self.assertEqual(model["runtime"], {"max_frame_age_ms": 30.0, "max_pair_skew_ms": 20.0})
        self.assertEqual(model["policy_hz"], 50.0)
        self.assertEqual(model["policy_angular_velocity_scale"], 0.25)
        self.assertEqual(model["sensor_baudrate"], 921600)
        self.assertEqual(baseline["discard_initial_s"], 1.0)
        self.assertEqual(baseline["statistics_frame"], "sensor")
        self.assertEqual(baseline["gyro"]["samples"], 803)
        self.assertEqual(baseline["euler"]["samples"], 803)
        np.testing.assert_allclose(
            baseline["gyro"]["mean_rad_s"],
            [-0.00010480185609776325, -0.0005094166169815332, 7.031010598963865e-05],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(model["gyro_bias_rad_s"], baseline["gyro"]["mean_rad_s"])
        np.testing.assert_allclose(
            baseline["gyro"]["std_rad_s"],
            [0.0014574496391202708, 0.002595317743444557, 0.0007649076106342686],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            baseline["euler"]["std_rad"],
            [0.00021981419900484112, 0.000407468754338021, 0.00013526299693944198],
            rtol=0.0,
            atol=0.0,
        )
        self.assertAlmostEqual(measured["packet_rates"]["gyro_rate_hz"], 200.9936362436779)
        self.assertAlmostEqual(measured["packet_rates"]["orientation_rate_hz"], 200.97696979791795)
        dynamic = measured["dynamic_consistency"]
        self.assertAlmostEqual(dynamic["median_relative_orientation_to_gyro_delay_ms"], 9.99993199999949)
        self.assertEqual(
            [dynamic["axis_results"][axis]["euler_derived_angular_velocity_to_gyro_gain_ratio"] for axis in "xyz"],
            [0.9413184026977318, 0.9420725493982783, 1.007452844828143],
        )
        self.assertFalse(dynamic["static_gravity_gain_measured"])
        self.assertIsNone(model["unmeasured_quantities"]["absolute_transport_latency_s"])
        self.assertIsNone(model["unmeasured_quantities"]["level_offset_rad"])
        self.assertFalse(model["stress_only_provenance"]["applied_to_runtime_model"])
        self.assertEqual(model["stress_only_provenance"]["gyro_component_range_rad_s"], [-0.2, 0.2])
        self.assertEqual(model["stress_only_provenance"]["projected_gravity_component_range"], [-0.05, 0.05])

    def test_local_regeneration_matches_checked_in_model_except_timestamp(self):
        inputs = (STATIC_REPORT_PATH, DYNAMIC_REPORT_PATH, DYNAMIC_DATASET_PATH)
        if not all(path.is_file() for path in inputs):
            self.skipTest("gitignored CMP10A provenance inputs are not present in this clone")

        generated = build_cmp10a_runtime_config_from_files(*inputs)
        checked_in = _checked_in()
        generated.pop("created_utc")
        checked_in.pop("created_utc")
        self.assertEqual(generated, checked_in)


class RndImuRuntimeConfigBuilderTest(unittest.TestCase):
    def test_builder_is_deterministic_except_created_timestamp(self):
        first = _build_synthetic(created_utc="2026-07-16T00:00:00+00:00")
        second = _build_synthetic(created_utc="2026-07-16T00:00:01+00:00")
        self.assertNotEqual(first["created_utc"], second["created_utc"])
        first.pop("created_utc")
        second.pop("created_utc")
        self.assertEqual(first, second)

    def test_static_and_dynamic_quality_gates_fail_closed(self):
        fixture = _synthetic_fixture()
        fixture["static_report"]["runtime_gate"]["quality_pass"] = False
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "static_report runtime quality gate"):
            _build_synthetic(fixture)

        fixture = _synthetic_fixture()
        fixture["dynamic_report"]["quality_pass"] = False
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "dynamic_report quality gate"):
            _build_synthetic(fixture)

        fixture = _synthetic_fixture()
        fixture["dynamic_report"]["stages"]["dynamic_axis_y"]["quality_pass"] = False
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "stage dynamic_axis_y"):
            _build_synthetic(fixture)

    def test_static_report_hash_chain_mismatch_fails_closed(self):
        fixture = _synthetic_fixture()
        fixture["dynamic_report"]["source"]["identification_report_sha256"] = "4" * 64
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "different static report SHA-256"):
            _build_synthetic(fixture)

        fixture = _synthetic_fixture()
        fixture["dynamic_dataset"]["metadata"]["identification_report_sha256"] = "4" * 64
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "different static report SHA-256"):
            _build_synthetic(fixture)

    def test_validator_rejects_runtime_matrix_and_envelope_drift(self):
        model = _build_synthetic()
        model["runtime_transform"]["sensor_to_base_matrix"] = np.eye(3).tolist()
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, r"diag\(-1, -1, \+1\)"):
            validate_cmp10a_runtime_config(model)

        model = _build_synthetic()
        model["assumed_simulation_envelopes"]["gyro_white_noise"]["sigma_range_rad_s"] = [0.0003, 0.03]
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "gyro white sigma range"):
            validate_cmp10a_runtime_config(model)

        model = _build_synthetic()
        model["measured"]["mount_fit_evidence"]["applied_at_runtime"] = True
        with self.assertRaisesRegex(Cmp10aRuntimeConfigError, "evidence-only"):
            validate_cmp10a_runtime_config(model)


if __name__ == "__main__":
    unittest.main()
