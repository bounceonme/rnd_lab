from __future__ import annotations

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
                "effective_backlash": {"median_rad": 0.01},
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


class RndReal2SimAggregateTest(unittest.TestCase):
    def _write_manifest(self, directory: Path, *, dry_run_joint: str | None = None) -> Path:
        lines = [
            "schema_version = 1",
            "analysis_only = true",
            'purpose = "test baseline"',
        ]
        for joint_name in RND_LEG_JOINT_NAMES:
            model_path = directory / f"{joint_name}.json"
            model_path.write_text(json.dumps(_model(joint_name, dry_run=joint_name == dry_run_joint)))
            path_literal = json.dumps(str(model_path))
            lines.extend(
                [
                    "",
                    "[[joints]]",
                    f"name = {json.dumps(joint_name)}",
                    f"target_models = [{path_literal}]",
                    f"coulomb_models = [{path_literal}]",
                ]
            )
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
        self.assertEqual(baseline["quality_summary"]["fully_usable_joint_count"], 12)
        self.assertEqual(baseline["joints"]["R_Leg_knee"]["target"]["command_delay_s"]["median"], 0.05)

    def test_rejects_synthetic_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(Path(temporary_directory), dry_run_joint="R_Leg_knee")
            with self.assertRaisesRegex(AggregationError, "Synthetic model"):
                aggregate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
