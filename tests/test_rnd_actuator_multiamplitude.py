from __future__ import annotations

import math
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rnd_actuator_multiamplitude import (
    MultiAmplitudeModelError,
    TriangleTrace,
    finalize_with_sim_replays,
    fit_generalized_play,
    play_transform,
)


def _triangle(amplitude_rad: float, samples_per_cycle: int = 100, cycles: int = 6) -> np.ndarray:
    phase = np.arange(samples_per_cycle, dtype=np.float64) / samples_per_cycle
    cycle = amplitude_rad * (4.0 * np.abs(phase - 0.5) - 1.0)
    return np.tile(cycle, cycles)


def _trace(label: str, amplitude_rad: float, *, invert: bool = False) -> TriangleTrace:
    command = _triangle(amplitude_rad)
    position = 0.2 * command + 0.8 * play_transform(command, 0.02) + 0.007
    if invert:
        position = -position
    return TriangleTrace(
        label=label,
        command_rad=command,
        position_rad=position,
        amplitude_rad=amplitude_rad,
        samples_per_cycle=100,
        train_cycle_count=4,
    )


class GeneralizedPlayFitTest(unittest.TestCase):
    def test_recovers_common_linear_and_play_components(self):
        result = fit_generalized_play(
            [_trace("small", 0.03), _trace("large", 0.06)],
            [0.01, 0.02, 0.03, 0.04],
            max_play_branches=2,
        )

        self.assertTrue(result["quality"]["cross_amplitude_usable"])
        self.assertEqual(result["fit"]["selected_play_branch_count"], 1)
        self.assertAlmostEqual(result["command_path"]["play_thresholds_rad"][0], 0.02)
        self.assertAlmostEqual(result["command_path"]["play_weights"][0], 0.8, places=6)
        self.assertAlmostEqual(result["command_path"]["linear_weight"], 0.2, places=6)
        self.assertTrue(all(dataset["validation_pass"] for dataset in result["fit"]["datasets"]))

    def test_requires_two_distinct_amplitudes(self):
        with self.assertRaisesRegex(MultiAmplitudeModelError, "distinct triangle amplitudes"):
            fit_generalized_play(
                [_trace("first", 0.03), _trace("second", 0.03)],
                [0.01, 0.02, 0.03],
            )

    def test_failed_amplitude_is_not_hidden_by_other_trace(self):
        result = fit_generalized_play(
            [_trace("small", 0.03), _trace("inverted-large", 0.06, invert=True)],
            [0.01, 0.02, 0.03, 0.04],
            max_play_branches=2,
        )

        self.assertFalse(result["quality"]["cross_amplitude_usable"])
        self.assertFalse(result["fit"]["all_validation_gates_pass"])
        self.assertTrue(any(not dataset["validation_pass"] for dataset in result["fit"]["datasets"]))

    def test_runtime_weight_contract_is_preserved(self):
        result = fit_generalized_play(
            [_trace("small", 0.03), _trace("large", 0.06)],
            np.linspace(math.radians(0.2), math.radians(2.0), 10),
            max_play_branches=2,
        )
        command_path = result["command_path"]
        self.assertAlmostEqual(command_path["linear_weight"] + sum(command_path["play_weights"]), 1.0)
        self.assertEqual(command_path["residual_delay_s_range"], [0.0, 0.0])
        self.assertEqual(command_path["play_threshold_scale_range"], [1.0, 1.0])


class MultiAmplitudeReplayFinalizationTest(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[dict, list[Path]]:
        model = {
            "model_type": "rnd_multi_amplitude_generalized_play",
            "joint": "L_Leg_ankle_roll",
            "quality": {"cross_amplitude_usable": True},
            "command_path": {
                "residual_delay_s_range": [0.0, 0.0],
                "residual_position_bias_rad_range": [0.0, 0.0],
                "play_thresholds_rad": [0.018],
                "play_weights": [0.85],
                "linear_weight": 0.15,
                "play_threshold_scale_range": [1.0, 1.0],
            },
            "limitations": [
                "The hardware response delay aligns hysteresis during fitting; runtime delay remains zero until simulator replay.",
                "Passing this trace gate does not authorize RL integration; fixed-base Isaac replay is still required.",
            ],
            "source_datasets": [],
        }
        reports = []
        for index, total_delay in enumerate((0.0046, 0.0051)):
            dataset = directory / f"dataset_{index}.npz"
            dataset.write_bytes(f"dataset-{index}".encode())
            model["source_datasets"].append({
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            })
            report = {
                "schema_version": 1,
                "validation_type": "fixed_base_isaac_explicit_pd_replay",
                "joint": "L_Leg_ankle_roll",
                "dataset": str(dataset),
                "sim_replay_gate_satisfied": True,
                "applied_residual_delay_s": 0.0047,
                "recommended_total_residual_delay_s": total_delay,
                "reference_hardware_delay_s": 0.067,
                "reference_simulation_delay_s": 0.0671,
                "controller_settings": {
                    "stiffness": 26.25,
                    "damping": 1.08,
                    "residual_position_bias_rad": 0.0,
                },
                "phases": [
                    {
                        "hardware_vs_simulation": {
                            "r2": 0.98,
                            "normalized_rmse": 0.04,
                        }
                    }
                ],
            }
            report_path = directory / f"report_{index}.json"
            report_path.write_text(json.dumps(report))
            reports.append(report_path)
        return model, reports

    def test_finalizes_common_controller_and_delay_range(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model, reports = self._fixture(Path(temporary_directory))
            result = finalize_with_sim_replays(model, reports)

        self.assertEqual(result["application_status"], "sim_replay_validated_not_integrated")
        self.assertTrue(result["quality"]["sim_replay_validated"])
        self.assertFalse(result["quality"]["integration_allowed"])
        self.assertEqual(result["command_path"]["residual_delay_s_range"], [0.0047, 0.0047])
        self.assertEqual(result["sim_replay"]["recommended_total_residual_delay_s_range"], [0.0046, 0.0051])
        self.assertEqual(result["sim_replay"]["selected_controller"], {"stiffness": 26.25, "damping": 1.08})

    def test_rejects_failed_replay(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model, reports = self._fixture(Path(temporary_directory))
            report = json.loads(reports[0].read_text())
            report["sim_replay_gate_satisfied"] = False
            reports[0].write_text(json.dumps(report))
            with self.assertRaisesRegex(MultiAmplitudeModelError, "gate failed"):
                finalize_with_sim_replays(model, reports)


if __name__ == "__main__":
    unittest.main()
