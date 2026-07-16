from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_real2sim.config import RND_LEG_JOINT_NAMES
from rnd_real2sim_aggregate import AggregationError, aggregate_manifest


def _model(joint_name: str, *, dry_run: bool = False) -> dict:
    return {
        "schema_version": 2,
        "source_dataset": f"/tmp/{joint_name}.npz",
        "source_dataset_dry_run": dry_run,
        "source_dataset_sha256": "a" * 64,
        "joints": {
            joint_name: {
                "command_delay": {"seconds": 0.05, "reference_profile": "medium_sine"},
                "effective_backlash": {
                    "median_rad": 0.01,
                    "cycles": [
                        {"center_bias_rad": 0.002},
                        {"center_bias_rad": 0.004},
                        {"center_bias_rad": 0.003},
                    ],
                },
                "frequency_response": [
                    {
                        "profile_name": "medium_sine",
                        "gain": {"median": 0.98},
                        "full_output_fit": {"r2": 0.999},
                    }
                ],
                "friction_current_model": {"coulomb_current_a": 0.04},
                "quality": {
                    "status": "usable_target_and_coulomb",
                    "target_randomization_usable": True,
                    "coulomb_randomization_usable": True,
                },
            }
        },
    }


def _command_path_model(
    directory: Path,
    joint_name: str,
    *,
    usable: bool = True,
    sim_replay_validated: bool = False,
) -> Path:
    sources = []
    reports = []
    for index, amplitude in enumerate((0.02, 0.04)):
        dataset_path = directory / f"command_path_source_{index}.npz"
        identification_path = directory / f"command_path_source_{index}_model.json"
        dataset_path.write_bytes(f"dataset-{index}".encode())
        identification_path.write_text(f"identification-{index}")
        sources.append({
            "dataset": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "identification_model": str(identification_path),
            "identification_model_sha256": hashlib.sha256(identification_path.read_bytes()).hexdigest(),
        })
        reports.append({
            "label": str(dataset_path),
            "amplitude_rad": amplitude,
            "validation": {"r2": 0.99, "normalized_rmse": 0.02},
            "validation_pass": usable,
        })
    model = {
        "schema_version": 1,
        "model_type": "rnd_multi_amplitude_generalized_play",
        "analysis_only": True,
        "joint": joint_name,
        "amplitudes_rad": [0.02, 0.04],
        "amplitudes_deg": [1.1459155903, 2.2918311805],
        "source_datasets": sources,
        "measured": {
            "command_delay_s": {"count": 2, "median": 0.05, "minimum": 0.049, "maximum": 0.051},
            "effective_hysteresis_rad": {
                "count": 2,
                "median": 0.012,
                "minimum": 0.01,
                "maximum": 0.014,
            },
            "reference_sine_gain": {"count": 2, "median": 0.97, "minimum": 0.96, "maximum": 0.98},
        },
        "command_path": {
            "residual_delay_s_range": [0.0, 0.0],
            "residual_position_bias_rad_range": [0.0, 0.0],
            "play_thresholds_rad": [0.015],
            "play_weights": [0.8],
            "linear_weight": 0.2,
            "play_threshold_scale_range": [1.0, 1.0],
        },
        "fit": {
            "all_validation_gates_pass": usable,
            "minimum_validation_r2": 0.99,
            "maximum_validation_normalized_rmse": 0.02,
            "datasets": reports,
        },
        "quality": {
            "cross_amplitude_usable": usable,
            "minimum_validation_r2_required": 0.95,
            "maximum_normalized_rmse_allowed": 0.1,
        },
    }
    if sim_replay_validated:
        replay_reports = []
        for index, source in enumerate(sources):
            report_path = directory / f"command_path_source_{index}_sim_replay.json"
            report_path.write_text(json.dumps({"dataset": source["dataset"], "passed": True}))
            replay_reports.append({
                "dataset": source["dataset"],
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            })
        model["application_status"] = "sim_replay_validated_not_integrated"
        model["quality"]["sim_replay_validated"] = True
        model["quality"]["integration_allowed"] = False
        model["command_path"]["residual_delay_s_range"] = [0.0046, 0.0051]
        model["sim_replay"] = {
            "selected_controller": {"stiffness": 26.25, "damping": 1.08},
            "residual_delay_s_range": [0.0046, 0.0051],
            "residual_position_bias_rad_range": [0.0, 0.0],
            "reports": replay_reports,
        }
    path = directory / "multi_amplitude.json"
    path.write_text(json.dumps(model))
    return path


class RndReal2SimAggregateTest(unittest.TestCase):
    def _write_manifest(
        self,
        directory: Path,
        *,
        dry_run_joint: str | None = None,
        command_path_joint: str | None = None,
        command_path_usable: bool = True,
        command_path_sim_replay_validated: bool = False,
    ) -> Path:
        command_path_path = (
            _command_path_model(
                directory,
                command_path_joint,
                usable=command_path_usable,
                sim_replay_validated=command_path_sim_replay_validated,
            )
            if command_path_joint is not None
            else None
        )
        lines = [
            "schema_version = 1",
            "analysis_only = true",
            'purpose = "test baseline"',
        ]
        for joint_name in RND_LEG_JOINT_NAMES:
            model_path = directory / f"{joint_name}.json"
            model_path.write_text(json.dumps(_model(joint_name, dry_run=joint_name == dry_run_joint)))
            path_literal = json.dumps(str(model_path))
            lines.extend([
                "",
                "[[joints]]",
                f"name = {json.dumps(joint_name)}",
                f"target_models = {[] if joint_name == command_path_joint else [str(model_path)]!r}",
                f"coulomb_models = [{path_literal}]",
            ])
            if joint_name == command_path_joint:
                lines.append(f"command_path_model = {json.dumps(str(command_path_path))}")
        manifest_path = directory / "manifest.toml"
        manifest_path.write_text("\n".join(lines) + "\n")
        return manifest_path

    def test_aggregates_all_explicitly_selected_joints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(Path(temporary_directory))
            baseline = aggregate_manifest(manifest_path)

        self.assertTrue(baseline["analysis_only"])
        self.assertEqual(baseline["application_status"], "not_integrated_with_rl_or_simulation")
        self.assertEqual(baseline["quality_summary"]["joint_count"], 12)
        self.assertEqual(baseline["quality_summary"]["command_path_usable_joint_count"], 12)
        self.assertEqual(baseline["quality_summary"]["fully_usable_joint_count"], 12)
        self.assertEqual(baseline["joints"]["R_Leg_knee"]["target"]["command_delay_s"]["median"], 0.05)
        self.assertEqual(
            baseline["joints"]["R_Leg_knee"]["target"]["command_minus_position_center_bias_rad"]["median"],
            0.003,
        )

    def test_rejects_synthetic_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(Path(temporary_directory), dry_run_joint="R_Leg_knee")
            with self.assertRaisesRegex(AggregationError, "Synthetic model"):
                aggregate_manifest(manifest_path)

    def test_aggregates_cross_amplitude_command_path_without_promoting_constant_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(Path(temporary_directory), command_path_joint="L_Leg_ankle_roll")
            baseline = aggregate_manifest(manifest_path)

        joint = baseline["joints"]["L_Leg_ankle_roll"]
        self.assertFalse(joint["target"]["usable"])
        self.assertTrue(joint["command_path_model"]["usable"])
        self.assertEqual(joint["command_path_model"]["command_path"]["linear_weight"], 0.2)
        self.assertEqual(baseline["quality_summary"]["target_usable_joint_count"], 11)
        self.assertEqual(baseline["quality_summary"]["command_path_usable_joint_count"], 12)

    def test_rejects_failed_cross_amplitude_gate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(
                Path(temporary_directory),
                command_path_joint="L_Leg_ankle_roll",
                command_path_usable=False,
            )
            with self.assertRaisesRegex(AggregationError, "Cross-amplitude quality gate failed"):
                aggregate_manifest(manifest_path)

    def test_aggregates_finalized_cross_amplitude_replay_calibration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(
                Path(temporary_directory),
                command_path_joint="L_Leg_ankle_roll",
                command_path_sim_replay_validated=True,
            )
            baseline = aggregate_manifest(manifest_path)

        command_path = baseline["joints"]["L_Leg_ankle_roll"]["command_path_model"]
        self.assertTrue(command_path["validation"]["sim_replay_validated"])
        self.assertEqual(command_path["command_path"]["residual_delay_s_range"], [0.0046, 0.0051])
        self.assertEqual(command_path["sim_replay"]["selected_controller"]["stiffness"], 26.25)

    def test_rejects_residual_delay_without_replay_gate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest_path = self._write_manifest(directory, command_path_joint="L_Leg_ankle_roll")
            command_path_path = directory / "multi_amplitude.json"
            model = json.loads(command_path_path.read_text())
            model["command_path"]["residual_delay_s_range"] = [0.004, 0.005]
            command_path_path.write_text(json.dumps(model))
            with self.assertRaisesRegex(AggregationError, "must keep residual calibration at zero"):
                aggregate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
