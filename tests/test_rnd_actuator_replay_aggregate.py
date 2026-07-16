from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
ROBOT_PACKAGE_DIR = REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import RndActuatorModelError, validate_rnd_actuator_model
from rnd_actuator_aggregate_replays import (
    ReplayAggregationError,
    aggregate_replay_reports,
    build_candidate_model,
)
from rnd_actuator_build import DEFAULT_ASSET_CFG, DEFAULT_BASELINE, build_actuator_model
from rnd_real2sim.config import RND_LEG_JOINT_NAMES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayAggregateFixture:
    unresolved_joint = "L_Leg_ankle_roll"

    def __init__(self, root: Path):
        self.root = root
        self.report_directory = root / "reports"
        self.report_directory.mkdir()
        self.base_model_path = root / "rnd_actuator_model.json"
        self.base_model = build_actuator_model(DEFAULT_BASELINE, DEFAULT_ASSET_CFG)
        self.base_model_path.write_text(json.dumps(self.base_model), encoding="utf-8")
        self.report_paths: dict[str, Path] = {}
        manifest_lines = [
            "schema_version = 1",
            "analysis_only = true",
            'purpose = "test"',
            "",
        ]
        for index, joint_name in enumerate(RND_LEG_JOINT_NAMES):
            manifest_lines.extend(["[[joints]]", f'name = "{joint_name}"'])
            if joint_name == self.unresolved_joint:
                manifest_lines.extend(["target_models = []", "coulomb_models = []", ""])
                continue
            dataset_path = root / f"dataset_{index}.npz"
            dataset_path.write_bytes(f"dataset-{joint_name}".encode())
            target_model_path = root / f"dataset_{index}_model.json"
            target_model = {
                "schema_version": 2,
                "source_dataset": str(dataset_path),
                "source_dataset_sha256": _sha256(dataset_path),
                "source_dataset_dry_run": False,
                "joints": {
                    joint_name: {
                        "quality": {
                            "target_randomization_usable": True,
                        }
                    }
                },
            }
            target_model_path.write_text(json.dumps(target_model), encoding="utf-8")
            manifest_lines.extend([
                f'target_models = ["{target_model_path}"]',
                "coulomb_models = []",
                "",
            ])
            report_path = self.report_directory / f"dataset_{index}_sim_replay.json"
            report_path.write_text(
                json.dumps(
                    self._report(
                        joint_name,
                        dataset_path,
                        stiffness=24.0 + index,
                        residual_delay=0.00025 * index,
                    )
                ),
                encoding="utf-8",
            )
            self.report_paths[joint_name] = report_path
        self.manifest_path = root / "manifest.toml"
        self.manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    def _report(
        self,
        joint_name: str,
        dataset_path: Path,
        *,
        stiffness: float,
        residual_delay: float,
    ) -> dict:
        phases = []
        for profile, frequency in (("micro_triangle", 0.1), ("slow_sine", 0.2), ("medium_sine", 0.5)):
            phase = {
                "profile_name": profile,
                "frequency_hz": frequency,
                "sample_count": 100,
                "hardware_vs_simulation": {
                    "r2": 0.99,
                    "normalized_rmse": 0.02,
                    "rmse_rad": 0.001,
                },
            }
            if profile == "medium_sine":
                phase["delay_error_s"] = -residual_delay
                phase["gain_relative_error"] = 0.01
            phases.append(phase)
        return {
            "schema_version": 1,
            "validation_type": "fixed_base_isaac_explicit_pd_replay",
            "dataset": str(dataset_path),
            "joint": joint_name,
            "model": str(self.base_model_path),
            "physics_hz": 200.0,
            "sample_hz": 50.0,
            "reference_profile": "medium_sine",
            "reference_hardware_delay_s": 0.05,
            "reference_simulation_delay_s": 0.05 - residual_delay,
            "recommended_residual_delay_s": residual_delay,
            "controller_settings": {
                "stiffness": stiffness,
                "damping": 1.08,
                "residual_position_bias_rad": 0.002,
            },
            "gate_thresholds": {
                "minimum_phase_r2": 0.95,
                "maximum_phase_normalized_rmse": 0.1,
                "maximum_reference_delay_error_s": 0.005,
                "maximum_reference_gain_relative_error": 0.1,
            },
            "phases": phases,
            "sim_replay_gate_satisfied": True,
            "automatic_model_update_performed": False,
        }


class RndActuatorReplayAggregateTest(unittest.TestCase):
    def test_aggregate_and_candidate_remain_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = ReplayAggregateFixture(Path(temporary_directory))
            summary = aggregate_replay_reports(
                fixture.manifest_path,
                fixture.base_model_path,
                fixture.report_directory,
            )
            candidate = build_candidate_model(
                fixture.base_model,
                summary,
                Path(temporary_directory) / "summary.json",
                "a" * 64,
            )

        self.assertEqual(summary["quality_summary"]["accepted_target_dataset_count"], 11)
        self.assertEqual(summary["quality_summary"]["sim_replay_validated_joint_count"], 11)
        self.assertEqual(summary["quality_summary"]["unresolved_joints"], [fixture.unresolved_joint])
        self.assertFalse(candidate["integration_enabled"])
        self.assertEqual(candidate["application_status"], "sim_replay_aggregated_not_enabled")
        self.assertTrue(candidate["joints"]["R_Leg_hip_yaw"]["quality"]["sim_replay_validated"])
        self.assertFalse(candidate["joints"][fixture.unresolved_joint]["quality"]["sim_replay_validated"])
        self.assertEqual(candidate["joints"]["R_Leg_hip_yaw"]["controller_seed"]["stiffness"], 24.0)
        self.assertEqual(
            candidate["joints"]["R_Leg_hip_yaw"]["command_path"]["residual_position_bias_rad_range"],
            [0.002, 0.002],
        )
        validate_rnd_actuator_model(candidate)
        with self.assertRaisesRegex(RndActuatorModelError, "not passed simulator replay"):
            validate_rnd_actuator_model(
                candidate,
                ("R_Leg_hip_yaw",),
                require_sim_replay_validation=True,
            )

    def test_missing_selected_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = ReplayAggregateFixture(Path(temporary_directory))
            fixture.report_paths["R_Leg_knee"].unlink()
            with self.assertRaisesRegex(ReplayAggregationError, "missing replay reports"):
                aggregate_replay_reports(
                    fixture.manifest_path,
                    fixture.base_model_path,
                    fixture.report_directory,
                )

    def test_failed_selected_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = ReplayAggregateFixture(Path(temporary_directory))
            report_path = fixture.report_paths["R_Leg_knee"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["sim_replay_gate_satisfied"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ReplayAggregationError, "did not pass"):
                aggregate_replay_reports(
                    fixture.manifest_path,
                    fixture.base_model_path,
                    fixture.report_directory,
                )

    def test_repeated_controller_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ReplayAggregateFixture(root)
            manifest = fixture.manifest_path.read_text(encoding="utf-8")
            first_model = root / "dataset_0_model.json"
            second_dataset = root / "dataset_repeat.npz"
            second_dataset.write_bytes(b"repeat")
            second_model = root / "dataset_repeat_model.json"
            model = json.loads(first_model.read_text(encoding="utf-8"))
            model["source_dataset"] = str(second_dataset)
            model["source_dataset_sha256"] = _sha256(second_dataset)
            second_model.write_text(json.dumps(model), encoding="utf-8")
            manifest = manifest.replace(
                f'target_models = ["{first_model}"]',
                f'target_models = ["{first_model}", "{second_model}"]',
                1,
            )
            fixture.manifest_path.write_text(manifest, encoding="utf-8")
            second_report = fixture._report("R_Leg_hip_yaw", second_dataset, stiffness=30.0, residual_delay=0.001)
            (fixture.report_directory / "dataset_repeat_sim_replay.json").write_text(
                json.dumps(second_report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ReplayAggregationError, "disagree on R_Leg_hip_yaw stiffness"):
                aggregate_replay_reports(
                    fixture.manifest_path,
                    fixture.base_model_path,
                    fixture.report_directory,
                )

    def test_command_path_model_selects_its_exact_replay_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ReplayAggregateFixture(root)
            sources = []
            selected_reports = []
            for index, residual_delay in enumerate((0.0046, 0.0051)):
                dataset = root / f"left_roll_{index}.npz"
                dataset.write_bytes(f"left-roll-{index}".encode())
                report_path = fixture.report_directory / f"left_roll_{index}_final_sim_replay.json"
                report = fixture._report(
                    fixture.unresolved_joint,
                    dataset,
                    stiffness=26.25,
                    residual_delay=residual_delay,
                )
                report["applied_residual_delay_s"] = residual_delay
                report["recommended_total_residual_delay_s"] = residual_delay
                next(phase for phase in report["phases"] if phase["profile_name"] == "medium_sine")["delay_error_s"] = (
                    0.0002
                )
                report_path.write_text(json.dumps(report), encoding="utf-8")
                sources.append({"dataset": str(dataset), "dataset_sha256": _sha256(dataset)})
                selected_reports.append({
                    "dataset": str(dataset),
                    "report": str(report_path),
                    "report_sha256": _sha256(report_path),
                })
                superseded = fixture.report_directory / f"left_roll_{index}_superseded_sim_replay.json"
                superseded.write_text(json.dumps(report), encoding="utf-8")

            command_path_model = root / "left_roll_multiamplitude.json"
            command_path_model.write_text(
                json.dumps({
                    "schema_version": 1,
                    "model_type": "rnd_multi_amplitude_generalized_play",
                    "joint": fixture.unresolved_joint,
                    "analysis_only": True,
                    "source_datasets": sources,
                    "quality": {
                        "cross_amplitude_usable": True,
                        "sim_replay_validated": True,
                        "integration_allowed": False,
                    },
                    "sim_replay": {"reports": selected_reports},
                }),
                encoding="utf-8",
            )
            manifest = fixture.manifest_path.read_text(encoding="utf-8")
            manifest = manifest.replace(
                f'name = "{fixture.unresolved_joint}"\ntarget_models = []\ncoulomb_models = []',
                f'name = "{fixture.unresolved_joint}"\ntarget_models = []\n'
                f'command_path_model = "{command_path_model}"\ncoulomb_models = []',
            )
            fixture.manifest_path.write_text(manifest, encoding="utf-8")

            summary = aggregate_replay_reports(
                fixture.manifest_path,
                fixture.base_model_path,
                fixture.report_directory,
            )
            candidate = build_candidate_model(
                fixture.base_model,
                summary,
                root / "summary.json",
                "a" * 64,
            )

        self.assertEqual(summary["quality_summary"]["accepted_target_dataset_count"], 13)
        self.assertEqual(summary["quality_summary"]["sim_replay_validated_joint_count"], 12)
        self.assertEqual(summary["quality_summary"]["unresolved_joints"], [])
        self.assertEqual(
            candidate["joints"][fixture.unresolved_joint]["command_path"]["residual_delay_s_range"],
            [0.0046, 0.0051],
        )
        self.assertNotIn("left ankle-roll command path remains unresolved", " ".join(candidate["limitations"]))


if __name__ == "__main__":
    unittest.main()
