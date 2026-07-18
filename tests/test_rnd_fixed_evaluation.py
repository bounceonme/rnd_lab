from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPTS_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
SUITE_PATH = RSL_RL_SCRIPTS_DIR / "config" / "rnd_step_actuator_imu_eval_v1.json"
sys.path.insert(0, str(RSL_RL_SCRIPTS_DIR))

from evaluation_runtime import (
    PULSE_PHYSICS_TICKS,
    FixedDomainSettings,
    stand_start_stop_schedule,
    straight_turn_straight_stop_schedule,
    validate_split_checkpoint,
)
from evaluation_schema import (
    EvaluationSchemaError,
    canonical_json_bytes,
    canonical_json_sha256,
    load_evaluation_suite,
    validate_evaluation_suite,
    verify_artifact_hashes,
)
from gait_metrics import (
    GaitMetricError,
    aggregate_evaluation_results,
    evaluate_episode_metrics,
    gait_symmetry_metrics,
    push_recovery_metrics,
    root_tilt_metrics,
    survival_fall_metrics,
    torque_saturation_metrics,
    touchdown_metrics,
    yaw_frame_linear_velocity_rmse,
    yaw_rate_rmse,
)


class FixedEvaluationSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_evaluation_suite(SUITE_PATH)

    def _suite_copy(self) -> dict:
        return copy.deepcopy(self.suite)

    def test_committed_suite_and_artifact_hashes_are_valid(self):
        validate_evaluation_suite(self.suite)
        verify_artifact_hashes(self.suite, REPO_ROOT)
        self.assertEqual(
            canonical_json_sha256(self.suite),
            "4e6769129ef5e2a228a3e6683d03319ed4c6fbec55d77e1726460a0fba996e58",
        )

    def test_canonical_json_has_stable_bytes_and_sha256(self):
        document = {"b": 1, "a": [2]}

        self.assertEqual(canonical_json_bytes(document), b'{"a":[2],"b":1}')
        self.assertEqual(
            canonical_json_sha256(document),
            "63c9663de90ee828bbda6cd9acf02d0c653986c1ec25aa239920641edc9a1de5",
        )

    def test_loader_rejects_duplicate_json_object_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

            with self.assertRaisesRegex(EvaluationSchemaError, "Duplicate JSON object key"):
                load_evaluation_suite(path)

    def test_duplicate_ids_and_undefined_references_are_rejected(self):
        duplicate = self._suite_copy()
        duplicate["domains"][1]["id"] = duplicate["domains"][0]["id"]
        with self.assertRaisesRegex(EvaluationSchemaError, "Duplicate ID"):
            validate_evaluation_suite(duplicate)

        undefined = self._suite_copy()
        undefined["cases"][0]["domain_id"] = "validation-undefined"
        with self.assertRaisesRegex(EvaluationSchemaError, "undefined domain"):
            validate_evaluation_suite(undefined)

        undefined_pulse = self._suite_copy()
        undefined_pulse["cases"][2]["recovery_pulse_id"] = "validation-missing-pulse"
        with self.assertRaisesRegex(EvaluationSchemaError, "undefined pulse"):
            validate_evaluation_suite(undefined_pulse)

    def test_cross_split_references_and_duplicate_domain_content_are_rejected(self):
        cross_reference = self._suite_copy()
        cross_reference["cases"][3]["domain_id"] = "validation-center"
        with self.assertRaisesRegex(EvaluationSchemaError, "split leakage"):
            validate_evaluation_suite(cross_reference)

        duplicate_content = self._suite_copy()
        duplicate_content["domains"][2]["resolved"] = copy.deepcopy(duplicate_content["domains"][0]["resolved"])
        with self.assertRaisesRegex(EvaluationSchemaError, "split leakage"):
            validate_evaluation_suite(duplicate_content)

    def test_resolved_domains_forbid_ranges_lists_nulls_and_unknown_fields(self):
        list_value = self._suite_copy()
        list_value["domains"][0]["resolved"]["material"]["static_friction"] = [0.85, 0.85]
        with self.assertRaisesRegex(EvaluationSchemaError, "singleton"):
            validate_evaluation_suite(list_value)

        range_key = self._suite_copy()
        range_key["domains"][0]["resolved"]["material"]["friction_range"] = 0.85
        with self.assertRaisesRegex(EvaluationSchemaError, "range fields"):
            validate_evaluation_suite(range_key)

        null_value = self._suite_copy()
        null_value["domains"][0]["resolved"]["material"]["static_friction"] = None
        with self.assertRaisesRegex(EvaluationSchemaError, "not null"):
            validate_evaluation_suite(null_value)

        unknown = self._suite_copy()
        unknown["cases"][0]["note"] = "not allowed"
        with self.assertRaisesRegex(EvaluationSchemaError, "unknown keys"):
            validate_evaluation_suite(unknown)

    def test_segment_overlap_gap_horizon_and_command_range_are_rejected(self):
        overlap = self._suite_copy()
        overlap["scenarios"][0]["segments"][1]["start_step"] = 99
        with self.assertRaisesRegex(EvaluationSchemaError, "overlap"):
            validate_evaluation_suite(overlap)

        gap = self._suite_copy()
        gap["scenarios"][0]["segments"][1]["start_step"] = 101
        with self.assertRaisesRegex(EvaluationSchemaError, "gap"):
            validate_evaluation_suite(gap)

        invalid_horizon = self._suite_copy()
        invalid_horizon["scenarios"][0]["horizon_steps"] = 0
        with self.assertRaisesRegex(EvaluationSchemaError, "horizon_steps"):
            validate_evaluation_suite(invalid_horizon)

        out_of_range = self._suite_copy()
        out_of_range["scenarios"][0]["segments"][0]["command"]["lin_vel_x_m_s"] = 0.11
        with self.assertRaisesRegex(EvaluationSchemaError, "outside the closed training range"):
            validate_evaluation_suite(out_of_range)

        runtime_incompatible = self._suite_copy()
        runtime_incompatible["scenarios"][0]["segments"][1]["command"]["lin_vel_x_m_s"] = 0.05
        with self.assertRaisesRegex(EvaluationSchemaError, "fixed STEP runtime"):
            validate_evaluation_suite(runtime_incompatible)

    def test_committed_scenarios_match_runtime_timelines_and_force_pulse_contract(self):
        policy_hz = self.suite["rates"]["policy_hz"]
        decimation = round(self.suite["rates"]["physics_hz"] / policy_hz)
        pulse_count = 0
        command_axes = ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s")

        for scenario in self.suite["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                segments = scenario["segments"]
                boundaries = (segments[0]["start_step"], *(segment["end_step"] for segment in segments))
                if len(segments) == 3:
                    self.assertEqual(boundaries, (0, 100, 400, 1000))
                    schedule = stand_start_stop_schedule(segments[1]["command"]["lin_vel_y_m_s"])
                elif len(segments) == 5:
                    self.assertEqual(boundaries, (0, 100, 300, 500, 700, 1000))
                    schedule = straight_turn_straight_stop_schedule(
                        segments[1]["command"]["lin_vel_y_m_s"],
                        segments[2]["command"]["ang_vel_z_rad_s"],
                    )
                else:
                    self.fail(f"Unsupported checked-in runtime timeline: {boundaries}")

                for segment, phase in zip(segments, schedule.phases, strict=True):
                    self.assertEqual(segment["start_step"] / policy_hz, phase.start_s)
                    self.assertEqual(segment["end_step"] / policy_hz, phase.end_s)
                    self.assertEqual(tuple(segment["command"][axis] for axis in command_axes), phase.target_b)

                pulses = scenario["pulses"]
                self.assertLessEqual(len(pulses), 1)
                if not pulses:
                    continue
                pulse_count += 1
                pulse = pulses[0]
                self.assertEqual(set(pulse), {"id", "start_step", "end_step", "root_velocity"})
                self.assertEqual(pulse["end_step"] - pulse["start_step"], 6)
                self.assertEqual(6 * decimation, PULSE_PHYSICS_TICKS)
                delta = pulse["root_velocity"]
                self.assertEqual(delta["ang_vel_z_rad_s"], 0.0)
                self.assertNotEqual((delta["lin_vel_x_m_s"], delta["lin_vel_y_m_s"]), (0.0, 0.0))

        self.assertEqual(pulse_count, 2)

    def test_schema_rejects_runtime_incompatible_force_pulses(self):
        too_short = self._suite_copy()
        pulse = too_short["scenarios"][2]["pulses"][0]
        pulse["end_step"] = pulse["start_step"] + 1
        with self.assertRaisesRegex(EvaluationSchemaError, "exactly 6 policy steps"):
            validate_evaluation_suite(too_short)

        angular = self._suite_copy()
        angular["scenarios"][2]["pulses"][0]["root_velocity"]["ang_vel_z_rad_s"] = 0.1
        with self.assertRaisesRegex(EvaluationSchemaError, "zero-torque force pulse"):
            validate_evaluation_suite(angular)

        zero = self._suite_copy()
        velocity = zero["scenarios"][2]["pulses"][0]["root_velocity"]
        velocity["lin_vel_x_m_s"] = 0.0
        velocity["lin_vel_y_m_s"] = 0.0
        with self.assertRaisesRegex(EvaluationSchemaError, "non-zero translational impulse"):
            validate_evaluation_suite(zero)

    def test_committed_domain_singletons_stay_inside_current_training_envelopes(self):
        for domain in self.suite["domains"]:
            with self.subTest(domain=domain["id"]):
                resolved = domain["resolved"]
                settings = FixedDomainSettings.from_mapping(resolved)
                material = resolved["material"]
                self.assertTrue(0.55 <= material["static_friction"] <= 1.15)
                self.assertTrue(0.40 <= material["dynamic_friction"] <= 0.90)
                self.assertLessEqual(material["dynamic_friction"], material["static_friction"])
                self.assertTrue(0.0 <= material["restitution"] <= 0.15)
                self.assertTrue(0.1952887 <= resolved["base_mass_add_kg"] <= 0.4152887)
                self.assertTrue(0.95 <= resolved["other_mass_scale"] <= 1.05)
                for axis in ("x", "y"):
                    self.assertTrue(-0.015 <= resolved["base_com_offset_m"][axis] <= 0.015)
                self.assertTrue(-0.010 <= resolved["base_com_offset_m"]["z"] <= 0.010)

                encoder = resolved["encoder"]
                self.assertEqual(len(encoder["zero_offset_rad"]), 12)
                self.assertEqual(len(encoder["sample_age_s"]), 12)
                self.assertTrue(all(-0.005 <= value <= 0.005 for value in encoder["zero_offset_rad"]))
                self.assertTrue(all(0.0 <= value <= 0.005 for value in encoder["sample_age_s"]))

                actuator = resolved["actuator"]
                self.assertEqual(
                    set(actuator),
                    {
                        "stiffness_scale",
                        "damping_scale",
                        "motor_strength_scale",
                        "coulomb_torque_nm",
                        "friction_transition_velocity_rad_s",
                    },
                )
                self.assertEqual(actuator["stiffness_scale"], 1.0)
                self.assertEqual(actuator["damping_scale"], 1.0)
                self.assertTrue(0.8 <= actuator["motor_strength_scale"] <= 1.25)
                self.assertTrue(0.17186861675963908 <= actuator["coulomb_torque_nm"] <= 0.17405707995827602)
                self.assertTrue(
                    0.03490658503988659 <= actuator["friction_transition_velocity_rad_s"] <= 0.13962634015954636
                )

                imu = resolved["imu"]
                self.assertTrue(0.0 <= imu["gyro"]["delay_s"] <= 0.005)
                self.assertTrue(0.0003 <= imu["gyro"]["noise_sigma"] <= 0.003)
                self.assertTrue(-0.01 <= imu["gyro"]["bias"] <= 0.01)
                self.assertTrue(0.0 <= imu["gravity"]["delay_s"] <= 0.02)
                self.assertTrue(0.00005 <= imu["gravity"]["noise_sigma"] <= 0.002)
                self.assertEqual(imu["gravity"]["bias"], 0.0)

                self.assertEqual(settings.base_mass_add_kg, resolved["base_mass_add_kg"])
                self.assertEqual(settings.actuator.motor_strength_scale, actuator["motor_strength_scale"])
                self.assertEqual(settings.encoder.zero_offset_rad, tuple(encoder["zero_offset_rad"]))
                self.assertEqual(settings.imu.gyro.delay_s, imu["gyro"]["delay_s"])

    def test_checked_in_test_split_stays_locked_until_a_171d_checkpoint_is_frozen(self):
        self.assertNotIn("policy-checkpoint", {artifact["id"] for artifact in self.suite["artifacts"]})
        validate_split_checkpoint("validation", None, "a" * 64)
        with self.assertRaisesRegex(Exception, "locked"):
            validate_split_checkpoint("test", None, "a" * 64)

    def test_bad_artifact_hash_fails_verification(self):
        document = self._suite_copy()
        document["artifacts"][0]["sha256"] = "0" * 64

        with self.assertRaisesRegex(EvaluationSchemaError, "SHA-256 mismatch"):
            verify_artifact_hashes(document, REPO_ROOT)


class GaitMetricsTest(unittest.TestCase):
    def test_survival_fall_timeout_and_censor_semantics(self):
        early = survival_fall_metrics(
            np.asarray([False, False, True], dtype=bool),
            horizon_steps=5,
            step_dt=0.02,
        )
        self.assertTrue(early["fell"])
        self.assertEqual(early["fall_step"], 2)
        self.assertEqual(early["survived_steps"], 3)
        self.assertAlmostEqual(early["survival_fraction"], 0.6)

        horizon_done = survival_fall_metrics(
            np.asarray([False, False, False, False, True], dtype=bool),
            horizon_steps=5,
            step_dt=0.02,
        )
        self.assertFalse(horizon_done["fell"])
        self.assertTrue(horizon_done["timed_out"])
        self.assertTrue(horizon_done["completed_horizon"])

        gymnasium_horizon = survival_fall_metrics(
            np.asarray([False, False, False, False, False], dtype=bool),
            horizon_steps=5,
            step_dt=0.02,
            timeout=np.asarray([False, False, False, False, True], dtype=bool),
        )
        self.assertFalse(gymnasium_horizon["fell"])
        self.assertTrue(gymnasium_horizon["timed_out"])
        self.assertTrue(gymnasium_horizon["completed_horizon"])

        censored = survival_fall_metrics(
            np.asarray([False, False, False], dtype=bool),
            horizon_steps=5,
            step_dt=0.02,
        )
        self.assertTrue(censored["censored"])
        self.assertAlmostEqual(censored["survival_time_s"], 0.06)

    def test_yaw_frame_linear_velocity_and_yaw_rate_rmse_formulas(self):
        half_yaw = math.pi / 4.0
        quaternion = np.asarray([
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        ])
        velocity_w = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        command = np.asarray([[1.0, 0.0, 0.2], [1.0, 0.0, -0.2]])
        angular_velocity_w = np.asarray([[0.0, 0.0, 0.3], [0.0, 0.0, -0.4]])

        self.assertAlmostEqual(
            yaw_frame_linear_velocity_rmse(velocity_w, quaternion, command),
            math.sqrt(0.5),
        )
        self.assertAlmostEqual(
            yaw_rate_rmse(angular_velocity_w, command),
            math.sqrt((0.1**2 + (-0.2) ** 2) / 2.0),
        )
        self.assertAlmostEqual(
            yaw_frame_linear_velocity_rmse(
                velocity_w,
                quaternion,
                command,
                valid_mask=np.asarray([True, False]),
            ),
            0.0,
        )

    def test_root_tilt_metrics_separate_step_lateral_and_sagittal_axes(self):
        ten_degrees = math.radians(10.0)
        half_angle = 0.5 * ten_degrees
        quaternion = np.asarray([
            [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0],
            [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        command = np.asarray([
            [0.0, -0.4, 0.0],
            [0.0, -0.4, 0.0],
            [0.0, 0.0, 0.0],
        ])

        metrics = root_tilt_metrics(
            quaternion,
            command,
            command_speed_threshold_m_s=0.1,
        )

        self.assertEqual(metrics["moving_sample_count"], 2)
        self.assertAlmostEqual(metrics["lateral_abs_mean_deg"], 5.0)
        self.assertAlmostEqual(metrics["sagittal_abs_mean_deg"], 5.0)
        self.assertAlmostEqual(metrics["sagittal_signed_mean_deg"], -5.0)

    def test_gait_symmetry_covers_air_stance_duty_count_and_same_foot(self):
        contact = np.asarray(
            [
                [True, True],
                [False, True],
                [False, True],
                [True, True],
                [True, False],
                [True, False],
                [True, True],
                [False, True],
                [False, True],
            ],
            dtype=bool,
        )

        metrics = gait_symmetry_metrics(contact, step_dt=0.1)

        self.assertAlmostEqual(metrics["right"]["air_time_mean_s"], 0.2)
        self.assertAlmostEqual(metrics["left"]["air_time_mean_s"], 0.2)
        self.assertAlmostEqual(metrics["right"]["stance_time_mean_s"], 0.25)
        self.assertAlmostEqual(metrics["left"]["stance_time_mean_s"], 0.4)
        self.assertAlmostEqual(metrics["right"]["duty_factor"], 5.0 / 9.0)
        self.assertAlmostEqual(metrics["left"]["duty_factor"], 7.0 / 9.0)
        self.assertEqual(metrics["touchdown_count_abs_difference"], 0)
        self.assertEqual(metrics["consecutive_same_foot_count"], 0)
        self.assertEqual(metrics["consecutive_same_foot_fraction"], 0.0)

        repeated_right = np.asarray(
            [[True, True], [False, True], [True, True], [False, True], [True, True]],
            dtype=bool,
        )
        repeated_metrics = gait_symmetry_metrics(repeated_right, step_dt=0.1)
        self.assertEqual(repeated_metrics["consecutive_same_foot_count"], 1)
        self.assertEqual(repeated_metrics["consecutive_same_foot_fraction"], 1.0)

    def test_touchdown_progress_alternation_tap_and_simultaneous_formulas(self):
        contact = np.asarray(
            [
                [True, True],
                [False, True],
                [False, True],
                [True, True],
                [True, False],
                [True, True],
                [False, False],
                [True, True],
            ],
            dtype=bool,
        )
        positions = np.zeros((contact.shape[0], 2, 3), dtype=np.float64)
        positions[3, 0, 0] = 0.1
        positions[5, 1, 0] = 0.2
        quaternion = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (contact.shape[0], 1))
        command = np.tile(np.asarray([1.0, 0.0, 0.0]), (contact.shape[0], 1))

        metrics = touchdown_metrics(
            contact,
            positions,
            quaternion,
            command,
            step_dt=0.1,
            minimum_progress_m=0.04,
            tap_max_air_time_s=0.15,
            command_speed_threshold_m_s=0.1,
        )

        self.assertEqual(metrics["touchdown_frame_count"], 3)
        self.assertEqual(metrics["touchdown_event_count"], 4)
        self.assertEqual(metrics["single_touchdown_count"], 2)
        self.assertEqual(metrics["simultaneous_touchdown_count"], 1)
        self.assertAlmostEqual(metrics["simultaneous_touchdown_fraction"], 1.0 / 3.0)
        self.assertEqual(metrics["tap_count"], 1)
        self.assertEqual(metrics["tap_fraction"], 0.5)
        self.assertEqual(metrics["alternation_fraction"], 1.0)
        self.assertEqual(metrics["progress_event_count"], 1)
        self.assertAlmostEqual(metrics["progress_mean_m"], 0.1)
        self.assertEqual(metrics["progress_below_minimum_count"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_episode_metrics_consume_exact_200hz_touchdown_events(self):
        policy_samples = 2
        physics_samples = 8
        telemetry = {
            "termination": np.zeros(policy_samples, dtype=bool),
            "timeout": np.asarray([False, True], dtype=bool),
            "command": np.tile(np.asarray([1.0, 0.0, 0.0]), (policy_samples, 1)),
            "root_lin_vel_w": np.zeros((policy_samples, 3)),
            "root_ang_vel_w": np.zeros((policy_samples, 3)),
            "root_quat_w": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (policy_samples, 1)),
            "foot_contact": np.ones((policy_samples, 2), dtype=bool),
            "foot_pos_w": np.zeros((policy_samples, 2, 3)),
            "applied_torque": np.zeros((policy_samples, 2)),
            "physics_command": np.tile(np.asarray([1.0, 0.0, 0.0]), (physics_samples, 1)),
            "physics_root_quat_w": np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0]), (physics_samples, 1)
            ),
            "physics_foot_contact": np.zeros((physics_samples, 2), dtype=bool),
            "physics_foot_pos_w": np.zeros((physics_samples, 2, 3)),
            "physics_touchdown_event": np.zeros((physics_samples, 2), dtype=bool),
            "physics_touchdown_air_time_s": np.zeros((physics_samples, 2)),
            "physics_touchdown_preimpact_speed_m_s": np.zeros((physics_samples, 2)),
        }
        telemetry["physics_touchdown_event"][3, 0] = True
        telemetry["physics_touchdown_air_time_s"][3, 0] = 0.2
        telemetry["physics_touchdown_preimpact_speed_m_s"][3, 0] = 0.45
        telemetry["physics_foot_pos_w"][3, 0, 0] = 0.1

        metrics = evaluate_episode_metrics(
            telemetry,
            horizon_steps=policy_samples,
            step_dt=0.02,
            touchdown_step_dt=0.005,
            effort_limits=np.asarray([5.4, 5.4]),
            minimum_touchdown_progress_m=0.04,
            tap_max_air_time_s=0.15,
            command_speed_threshold_m_s=0.1,
            torque_saturation_threshold_fraction=0.98,
            joint_names=("right", "left"),
            timeout=telemetry["timeout"],
        )

        self.assertEqual(metrics["touchdown"]["touchdown_event_count"], 1)
        self.assertEqual(metrics["touchdown"]["progress_event_count"], 1)
        self.assertAlmostEqual(metrics["touchdown"]["preimpact_speed_p95_m_s"], 0.45)
        self.assertFalse(metrics["push_recovery"]["applicable"])
        json.dumps(metrics, allow_nan=False)

    def test_incomplete_pulse_censors_recovery(self):
        telemetry = {
            "termination": np.asarray([False, True], dtype=bool),
            "timeout": np.zeros(2, dtype=bool),
            "command": np.zeros((2, 3)),
            "root_lin_vel_w": np.zeros((2, 3)),
            "root_ang_vel_w": np.zeros((2, 3)),
            "root_quat_w": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "foot_contact": np.ones((2, 2), dtype=bool),
            "foot_pos_w": np.zeros((2, 2, 3)),
            "applied_torque": np.zeros((2, 2)),
        }

        metrics = evaluate_episode_metrics(
            telemetry,
            horizon_steps=10,
            step_dt=0.02,
            effort_limits=np.asarray([5.4, 5.4]),
            minimum_touchdown_progress_m=0.04,
            tap_max_air_time_s=0.15,
            command_speed_threshold_m_s=0.1,
            torque_saturation_threshold_fraction=0.98,
            joint_names=("right", "left"),
            push_end_step=2,
            push_delivery_complete=False,
            linear_velocity_error_threshold_m_s=0.15,
            yaw_rate_error_threshold_rad_s=0.1,
            recovery_dwell_s=0.3,
        )

        self.assertTrue(metrics["push_recovery"]["applicable"])
        self.assertFalse(metrics["push_recovery"]["delivery_complete"])
        self.assertTrue(metrics["push_recovery"]["censored"])
        self.assertFalse(metrics["push_recovery"]["recovered"])

    def test_torque_saturation_fraction_events_and_longest_dwell(self):
        torque = np.asarray([
            [0.0, 0.0],
            [9.8, 0.0],
            [10.0, -9.9],
            [0.0, -9.8],
            [0.0, 0.0],
        ])

        metrics = torque_saturation_metrics(
            torque,
            np.asarray([10.0, 10.0]),
            step_dt=0.02,
            threshold_fraction=0.98,
            joint_names=("right", "left"),
        )

        self.assertAlmostEqual(metrics["sample_joint_fraction"], 0.4)
        self.assertAlmostEqual(metrics["any_joint_step_fraction"], 0.6)
        self.assertEqual(metrics["system_event_count"], 1)
        self.assertEqual(metrics["system_longest_dwell_steps"], 3)
        self.assertAlmostEqual(metrics["system_longest_dwell_s"], 0.06)
        self.assertEqual(metrics["joint_event_count"], 2)
        self.assertEqual(metrics["per_joint"]["right"]["longest_dwell_steps"], 2)
        self.assertEqual(metrics["per_joint"]["left"]["longest_dwell_steps"], 2)

    def test_push_recovery_requires_threshold_dwell_and_reports_censoring(self):
        linear_error = np.asarray([0.0, 0.0, 0.5, 0.3, 0.1, 0.1, 0.1, 0.4, 0.1])
        yaw_error = np.asarray([0.0, 0.0, 0.3, 0.2, 0.05, 0.05, 0.05, 0.0, 0.0])

        recovered = push_recovery_metrics(
            linear_error,
            yaw_error,
            push_end_step=2,
            step_dt=0.1,
            linear_velocity_threshold_m_s=0.15,
            yaw_rate_threshold_rad_s=0.1,
            dwell_s=0.3,
        )
        self.assertTrue(recovered["recovered"])
        self.assertFalse(recovered["censored"])
        self.assertEqual(recovered["recovery_step"], 4)
        self.assertEqual(recovered["confirmation_step"], 6)
        self.assertAlmostEqual(recovered["recovery_time_s"], 0.2)
        self.assertAlmostEqual(recovered["confirmation_time_s"], 0.5)

        interrupted_linear = linear_error.copy()
        interrupted_linear[6] = 0.4
        censored = push_recovery_metrics(
            interrupted_linear,
            yaw_error,
            push_end_step=2,
            step_dt=0.1,
            linear_velocity_threshold_m_s=0.15,
            yaw_rate_threshold_rad_s=0.1,
            dwell_s=0.3,
        )
        self.assertFalse(censored["recovered"])
        self.assertTrue(censored["censored"])
        self.assertIsNone(censored["recovery_time_s"])
        self.assertAlmostEqual(censored["censor_time_s"], 0.7)

    def test_episode_then_case_aggregation_is_case_balanced(self):
        records = [
            {"case_id": "case-a", "split": "validation", "metrics": {"score": 0.0, "passed": False}},
            {"case_id": "case-b", "split": "validation", "metrics": {"score": 10.0, "passed": True}},
            {"case_id": "case-b", "split": "validation", "metrics": {"score": 10.0, "passed": True}},
            {"case_id": "case-b", "split": "validation", "metrics": {"score": 10.0, "passed": True}},
        ]

        aggregate = aggregate_evaluation_results(records)

        self.assertEqual(aggregate["cases"]["case-a"]["episode_count"], 1)
        self.assertEqual(aggregate["cases"]["case-b"]["episode_count"], 3)
        self.assertEqual(aggregate["metrics"]["score"], 5.0)
        self.assertEqual(aggregate["metrics"]["passed"], 0.5)
        self.assertEqual(aggregate["splits"]["validation"]["metrics"]["score"], 5.0)
        json.dumps(aggregate, allow_nan=False)

    def test_nonfinite_inputs_and_nonpositive_effort_limits_fail_closed(self):
        quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(GaitMetricError, "NaN or infinity"):
            yaw_frame_linear_velocity_rmse(
                np.asarray([[math.nan, 0.0, 0.0]]),
                quaternion,
                np.asarray([[0.0, 0.0, 0.0]]),
            )
        with self.assertRaisesRegex(GaitMetricError, "strictly positive"):
            torque_saturation_metrics(
                np.zeros((1, 1)),
                np.asarray([0.0]),
                step_dt=0.02,
                threshold_fraction=0.98,
            )


if __name__ == "__main__":
    unittest.main()
