#!/usr/bin/env python3
"""Fit one generalized-play command path across multiple excitation amplitudes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
sys.path.insert(0, str(_TOOL_DIR))

from rnd_real2sim.dataset import DatasetError, Real2SimDataset, load_dataset


MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "rnd_multi_amplitude_generalized_play"


class MultiAmplitudeModelError(ValueError):
    """Raised when the selected datasets cannot support a common model."""


@dataclass(frozen=True)
class TriangleTrace:
    """One delay-compensated triangle trace used by the common fit."""

    label: str
    command_rad: np.ndarray
    position_rad: np.ndarray
    amplitude_rad: float
    samples_per_cycle: int
    train_cycle_count: int

    @property
    def cycle_count(self) -> int:
        return int(self.command_rad.size // self.samples_per_cycle)

    @property
    def train_sample_count(self) -> int:
        return self.train_cycle_count * self.samples_per_cycle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the smallest convex generalized-play model that passes held-out triangle-cycle gates at every "
            "selected excitation amplitude. This produces a command-path seed, not an RL-ready actuator model."
        )
    )
    parser.add_argument("datasets", nargs="+", help="Complete real2sim NPZ datasets for one joint.")
    parser.add_argument("--joint", help="Excited joint; inferred when every dataset excites the same single joint.")
    parser.add_argument("--output", required=True, help="Output multi-amplitude analysis JSON.")
    parser.add_argument("--max-play-branches", type=int, default=3)
    parser.add_argument("--threshold-grid-size", type=int, default=35)
    parser.add_argument("--minimum-threshold-deg", type=float)
    parser.add_argument("--maximum-threshold-deg", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-validation-r2", type=float, default=0.95)
    parser.add_argument("--maximum-normalized-rmse", type=float, default=0.10)
    parser.add_argument(
        "--sim-replay-report",
        action="append",
        default=[],
        help=(
            "Passing fixed-base Isaac replay report for one source dataset. Repeat once per dataset to finalize "
            "the controller and residual-delay calibration without enabling RL integration."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MultiAmplitudeModelError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise MultiAmplitudeModelError(f"{label} must be finite, got {result!r}.")
    if minimum is not None and result < minimum:
        raise MultiAmplitudeModelError(f"{label} must be >= {minimum}, got {result}.")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MultiAmplitudeModelError(f"Identification model does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise MultiAmplitudeModelError(f"Identification model is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise MultiAmplitudeModelError(f"Identification model must be a JSON object: {path}")
    return value


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    sample_index = np.arange(signal.size, dtype=np.float64)
    return np.interp(
        sample_index - delay_samples,
        sample_index,
        signal,
        left=float(signal[0]),
        right=float(signal[-1]),
    )


def play_transform(command: np.ndarray, half_width_rad: float) -> np.ndarray:
    """Apply the same scalar play operator used by the runtime Torch kernel."""

    if command.ndim != 1 or command.size < 2 or not np.all(np.isfinite(command)):
        raise MultiAmplitudeModelError("Play input must be a finite one-dimensional trace.")
    if not math.isfinite(half_width_rad) or half_width_rad < 0.0:
        raise MultiAmplitudeModelError("Play half width must be finite and non-negative.")
    output = np.empty_like(command, dtype=np.float64)
    output[0] = command[0]
    for index in range(1, command.size):
        output[index] = min(
            max(output[index - 1], command[index] - half_width_rad),
            command[index] + half_width_rad,
        )
    return output


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    span = float(np.ptp(reference))
    denominator = float(np.sum(np.square(reference - float(np.mean(reference)))))
    return {
        "rmse_rad": rmse,
        "mae_rad": mae,
        "normalized_rmse": None if span <= 1.0e-12 else rmse / span,
        "r2": None if denominator <= 1.0e-15 else 1.0 - float(np.sum(np.square(error))) / denominator,
    }


def _simplex_least_squares(
    gram: np.ndarray,
    correlation: np.ndarray,
    target_energy: float,
    columns: tuple[int, ...],
) -> tuple[float, np.ndarray] | None:
    """Solve a small non-negative least-squares problem with weights summing to one."""

    best: tuple[float, np.ndarray] | None = None
    column_count = len(columns)
    for active_mask in range(1, 1 << column_count):
        active_local = [index for index in range(column_count) if active_mask & (1 << index)]
        active_global = [columns[index] for index in active_local]
        active_gram = gram[np.ix_(active_global, active_global)]
        active_correlation = correlation[active_global]
        kkt = np.block([
            [active_gram, np.ones((len(active_local), 1), dtype=np.float64)],
            [np.ones((1, len(active_local)), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
        ])
        right_hand_side = np.concatenate((active_correlation, np.ones(1, dtype=np.float64)))
        try:
            active_weights = np.linalg.solve(kkt, right_hand_side)[:-1]
        except np.linalg.LinAlgError:
            active_weights = np.linalg.lstsq(kkt, right_hand_side, rcond=None)[0][:-1]
        if np.min(active_weights) < -1.0e-9:
            continue

        weights = np.zeros(column_count, dtype=np.float64)
        weights[active_local] = np.maximum(active_weights, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            continue
        weights /= weight_sum
        global_weights = np.zeros(gram.shape[0], dtype=np.float64)
        global_weights[list(columns)] = weights
        residual_energy = float(
            target_energy - 2.0 * np.dot(global_weights, correlation) + np.dot(global_weights, gram @ global_weights)
        )
        candidate = (max(0.0, residual_energy), weights)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _trace_report(
    trace: TriangleTrace,
    design: np.ndarray,
    selected_columns: tuple[int, ...],
    weights: np.ndarray,
    *,
    minimum_validation_r2: float,
    maximum_normalized_rmse: float,
) -> dict[str, Any]:
    train_stop = trace.train_sample_count
    transformed = design[:, selected_columns] @ weights
    fit_offset = float(np.mean(trace.position_rad[:train_stop] - transformed[:train_stop]))
    transformed += fit_offset
    training = _metrics(trace.position_rad[:train_stop], transformed[:train_stop])
    validation = _metrics(trace.position_rad[train_stop:], transformed[train_stop:])
    cycle_metrics = []
    for cycle_index in range(trace.train_cycle_count, trace.cycle_count):
        start = cycle_index * trace.samples_per_cycle
        stop = start + trace.samples_per_cycle
        cycle_metrics.append({
            "cycle": cycle_index,
            **_metrics(trace.position_rad[start:stop], transformed[start:stop]),
        })
    validation_pass = bool(
        validation["r2"] is not None
        and validation["normalized_rmse"] is not None
        and float(validation["r2"]) >= minimum_validation_r2
        and float(validation["normalized_rmse"]) <= maximum_normalized_rmse
    )
    return {
        "label": trace.label,
        "amplitude_rad": trace.amplitude_rad,
        "amplitude_deg": math.degrees(trace.amplitude_rad),
        "cycle_count": trace.cycle_count,
        "training_cycle_count": trace.train_cycle_count,
        "validation_cycle_count": trace.cycle_count - trace.train_cycle_count,
        "nuisance_fit_offset_rad": fit_offset,
        "training": training,
        "validation": validation,
        "validation_cycles": cycle_metrics,
        "validation_pass": validation_pass,
    }


def fit_generalized_play(
    traces: Sequence[TriangleTrace],
    threshold_grid_rad: Sequence[float],
    *,
    max_play_branches: int = 3,
    minimum_validation_r2: float = 0.95,
    maximum_normalized_rmse: float = 0.10,
) -> dict[str, Any]:
    """Fit the smallest common convex generalized-play model across all traces."""

    if len(traces) < 2:
        raise MultiAmplitudeModelError("At least two complete datasets are required.")
    amplitudes = sorted({round(float(trace.amplitude_rad), 12) for trace in traces})
    if len(amplitudes) < 2:
        raise MultiAmplitudeModelError("At least two distinct triangle amplitudes are required.")
    if max_play_branches < 1:
        raise MultiAmplitudeModelError("max_play_branches must be positive.")
    if not 0.0 < minimum_validation_r2 <= 1.0:
        raise MultiAmplitudeModelError("minimum_validation_r2 must be in (0, 1].")
    if not 0.0 < maximum_normalized_rmse < 1.0:
        raise MultiAmplitudeModelError("maximum_normalized_rmse must be in (0, 1).")

    threshold_grid = np.asarray(threshold_grid_rad, dtype=np.float64)
    if (
        threshold_grid.ndim != 1
        or threshold_grid.size < max_play_branches
        or not np.all(np.isfinite(threshold_grid))
        or np.any(threshold_grid <= 0.0)
        or np.any(np.diff(threshold_grid) <= 0.0)
    ):
        raise MultiAmplitudeModelError(
            "Threshold grid must be finite, positive, strictly increasing, and large enough."
        )

    designs: list[np.ndarray] = []
    centered_designs: list[np.ndarray] = []
    centered_targets: list[np.ndarray] = []
    for trace in traces:
        if (
            trace.command_rad.ndim != 1
            or trace.position_rad.shape != trace.command_rad.shape
            or trace.command_rad.size != trace.samples_per_cycle * trace.cycle_count
            or trace.train_cycle_count < 1
            or trace.train_cycle_count >= trace.cycle_count
            or not np.all(np.isfinite(trace.command_rad))
            or not np.all(np.isfinite(trace.position_rad))
        ):
            raise MultiAmplitudeModelError(f"Trace {trace.label!r} has an invalid cycle layout or non-finite data.")
        columns = [trace.command_rad]
        columns.extend(play_transform(trace.command_rad, float(threshold)) for threshold in threshold_grid)
        design = np.column_stack(columns)
        designs.append(design)
        train = design[: trace.train_sample_count]
        target = trace.position_rad[: trace.train_sample_count]
        centered_designs.append(train - np.mean(train, axis=0))
        centered_targets.append(target - float(np.mean(target)))

    combined_design = np.vstack(centered_designs)
    combined_target = np.concatenate(centered_targets)
    gram = combined_design.T @ combined_design
    correlation = combined_design.T @ combined_target
    target_energy = float(np.dot(combined_target, combined_target))

    attempted_candidates = 0
    best_failed: tuple[tuple[float, float, tuple[int, ...]], dict[str, Any]] | None = None
    selected: dict[str, Any] | None = None
    for branch_count in range(1, max_play_branches + 1):
        passing: list[tuple[tuple[float, float, tuple[int, ...]], dict[str, Any]]] = []
        for play_columns in itertools.combinations(range(1, threshold_grid.size + 1), branch_count):
            selected_columns = (0, *play_columns)
            solution = _simplex_least_squares(gram, correlation, target_energy, selected_columns)
            attempted_candidates += 1
            if solution is None:
                continue
            training_error, weights = solution
            if not np.any(weights[1:] > 1.0e-8):
                continue
            reports = [
                _trace_report(
                    trace,
                    design,
                    selected_columns,
                    weights,
                    minimum_validation_r2=minimum_validation_r2,
                    maximum_normalized_rmse=maximum_normalized_rmse,
                )
                for trace, design in zip(traces, designs, strict=True)
            ]
            maximum_validation_nrmse = max(float(report["validation"]["normalized_rmse"]) for report in reports)
            minimum_validation_r2_value = min(float(report["validation"]["r2"]) for report in reports)
            key = (maximum_validation_nrmse, training_error, selected_columns)
            candidate = {
                "selected_columns": selected_columns,
                "weights": weights,
                "training_residual_energy": training_error,
                "maximum_validation_normalized_rmse": maximum_validation_nrmse,
                "minimum_validation_r2": minimum_validation_r2_value,
                "datasets": reports,
                "all_validation_gates_pass": all(report["validation_pass"] for report in reports),
            }
            if best_failed is None or key < best_failed[0]:
                best_failed = (key, candidate)
            if candidate["all_validation_gates_pass"]:
                passing.append((key, candidate))
        if passing:
            selected = min(passing, key=lambda item: item[0])[1]
            break

    if selected is None:
        if best_failed is None:
            raise MultiAmplitudeModelError("No numerically valid generalized-play candidate was found.")
        selected = best_failed[1]

    selected_columns = selected.pop("selected_columns")
    selected_weights = selected.pop("weights")
    active_thresholds: list[float] = []
    active_weights: list[float] = []
    for local_index, global_index in enumerate(selected_columns[1:], start=1):
        weight = float(selected_weights[local_index])
        if weight > 1.0e-8:
            active_thresholds.append(float(threshold_grid[global_index - 1]))
            active_weights.append(weight)
    linear_weight = float(selected_weights[0])
    total_weight = linear_weight + sum(active_weights)
    linear_weight /= total_weight
    active_weights = [weight / total_weight for weight in active_weights]

    return {
        "command_path": {
            "residual_delay_s_range": [0.0, 0.0],
            "residual_position_bias_rad_range": [0.0, 0.0],
            "play_thresholds_rad": active_thresholds,
            "play_weights": active_weights,
            "linear_weight": linear_weight,
            "play_threshold_scale_range": [1.0, 1.0],
        },
        "fit": {
            "method": "convex generalized play with per-dataset training offset",
            "selection_rule": "fewest play branches passing every held-out amplitude gate, then minimum worst NRMSE",
            "selected_play_branch_count": len(active_thresholds),
            "attempted_candidate_count": attempted_candidates,
            **selected,
        },
        "quality": {
            "minimum_validation_r2_required": minimum_validation_r2,
            "maximum_normalized_rmse_allowed": maximum_normalized_rmse,
            "cross_amplitude_usable": bool(selected["all_validation_gates_pass"]),
        },
    }


def _select_joint(datasets: Sequence[Real2SimDataset], requested: str | None) -> str:
    excited_sets = []
    for dataset in datasets:
        excited = dataset.metadata.get("excitation_joint_names")
        if not isinstance(excited, list) or not excited:
            raise MultiAmplitudeModelError(f"Dataset has no excitation_joint_names: {dataset.path}")
        excited_sets.append(set(str(value) for value in excited))
    if requested is not None:
        if any(requested not in names for names in excited_sets):
            raise MultiAmplitudeModelError(f"Requested joint {requested!r} is not excited in every dataset.")
        return requested
    common = set.intersection(*excited_sets)
    if len(common) != 1 or any(len(names) != 1 for names in excited_sets):
        raise MultiAmplitudeModelError("Use --joint unless every dataset excites the same single joint.")
    return common.pop()


def _reference_response(joint_model: dict[str, Any]) -> dict[str, Any]:
    name = joint_model.get("command_delay", {}).get("reference_profile")
    for response in joint_model.get("frequency_response", []):
        if response.get("profile_name") == name:
            return response
    raise MultiAmplitudeModelError(f"Reference response {name!r} is missing from an identification model.")


def _triangle_phase(dataset: Real2SimDataset, joint_name: str) -> tuple[int, np.ndarray, dict[str, Any]]:
    joint_index = dataset.joint_names.index(joint_name)
    phase_table = dataset.metadata.get("phase_metadata")
    if not isinstance(phase_table, dict):
        raise MultiAmplitudeModelError(f"Dataset is missing phase_metadata: {dataset.path}")
    candidates = []
    for phase_key, metadata in phase_table.items():
        if not isinstance(metadata, dict) or metadata.get("waveform") != "triangle":
            continue
        if metadata.get("joint_name") != joint_name:
            continue
        phase_id = int(phase_key)
        indices = np.flatnonzero(
            (dataset.arrays["phase_id"] == phase_id) & (dataset.arrays["excitation_joint_id"] == joint_index)
        )
        if indices.size:
            peak_velocity = 4.0 * float(metadata["frequency_hz"]) * float(metadata["amplitude_rad"])
            candidates.append((peak_velocity, phase_id, indices, metadata))
    if not candidates:
        raise MultiAmplitudeModelError(f"No triangle phase exists for {joint_name} in {dataset.path}.")
    _, phase_id, indices, metadata = min(candidates, key=lambda value: value[0])
    return phase_id, indices, metadata


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def fit_multi_amplitude_model(
    dataset_paths: Sequence[str | Path],
    *,
    joint_name: str | None = None,
    max_play_branches: int = 3,
    threshold_grid_size: int = 35,
    minimum_threshold_rad: float | None = None,
    maximum_threshold_rad: float | None = None,
    validation_fraction: float = 0.25,
    minimum_validation_r2: float = 0.95,
    maximum_normalized_rmse: float = 0.10,
) -> dict[str, Any]:
    """Load hardware datasets and build a provenance-complete cross-amplitude model."""

    if len(dataset_paths) < 2:
        raise MultiAmplitudeModelError("At least two dataset paths are required.")
    if threshold_grid_size < max_play_branches or threshold_grid_size < 2:
        raise MultiAmplitudeModelError("threshold_grid_size must be at least max_play_branches and at least two.")
    if not 0.0 < validation_fraction < 0.5:
        raise MultiAmplitudeModelError("validation_fraction must be in (0, 0.5).")
    try:
        datasets = [load_dataset(path) for path in dataset_paths]
    except DatasetError as error:
        raise MultiAmplitudeModelError(str(error)) from error
    selected_joint = _select_joint(datasets, joint_name)
    sample_rates = {round(float(dataset.metadata.get("sample_hz", 0.0)), 9) for dataset in datasets}
    if len(sample_rates) != 1 or next(iter(sample_rates)) <= 0.0:
        raise MultiAmplitudeModelError("Every dataset must use the same positive sample_hz.")
    sample_hz = next(iter(sample_rates))

    traces: list[TriangleTrace] = []
    provenance: list[dict[str, Any]] = []
    delays: list[float] = []
    backlashes: list[float] = []
    gains: list[float] = []
    for dataset in datasets:
        model_path = dataset.path.with_name(f"{dataset.path.stem}_model.json")
        model = _load_json(model_path)
        if model.get("schema_version") != 2 or model.get("source_dataset_dry_run") is not False:
            raise MultiAmplitudeModelError(f"Only schema-2 non-dry-run hardware models are accepted: {model_path}")
        if model.get("source_dataset_sha256") != dataset.sha256:
            raise MultiAmplitudeModelError(f"Dataset hash does not match its identification model: {dataset.path}")
        if model.get("timing", {}).get("quality_pass") is not True:
            raise MultiAmplitudeModelError(f"Timing quality gate failed: {model_path}")
        joints = model.get("joints")
        if not isinstance(joints, dict) or set(joints) != {selected_joint}:
            raise MultiAmplitudeModelError(f"Identification model must contain only {selected_joint}: {model_path}")
        joint_model = joints[selected_joint]
        reference_response = _reference_response(joint_model)
        if reference_response.get("quality_pass") is not True:
            raise MultiAmplitudeModelError(f"Reference sine quality gate failed: {model_path}")
        delay_s = float(joint_model["command_delay"]["seconds"])
        if not math.isfinite(delay_s) or delay_s < 0.0:
            raise MultiAmplitudeModelError(f"Invalid command delay in {model_path}")

        phase_id, indices, phase_metadata = _triangle_phase(dataset, selected_joint)
        frequency_hz = float(phase_metadata["frequency_hz"])
        cycle_count = int(phase_metadata["cycles"])
        samples_per_cycle_float = sample_hz / frequency_hz
        samples_per_cycle = round(samples_per_cycle_float)
        if abs(samples_per_cycle_float - samples_per_cycle) > 1.0e-6:
            raise MultiAmplitudeModelError(f"Triangle phase has a non-integral cycle length: {dataset.path}")
        if indices.size != samples_per_cycle * cycle_count:
            raise MultiAmplitudeModelError(
                f"Triangle phase has {indices.size} samples; expected {samples_per_cycle * cycle_count}: {dataset.path}"
            )
        validation_cycles = max(1, math.ceil(cycle_count * validation_fraction))
        train_cycle_count = cycle_count - validation_cycles
        if train_cycle_count < 1:
            raise MultiAmplitudeModelError(f"Triangle phase has too few cycles: {dataset.path}")

        joint_index = dataset.joint_names.index(selected_joint)
        command = dataset.arrays["goal_position_rad"][indices, joint_index].astype(np.float64)
        delayed_command = _fractional_delay(command, delay_s * sample_hz)
        position = dataset.arrays["position_rad"][indices, joint_index].astype(np.float64)
        amplitude_rad = float(phase_metadata["amplitude_rad"])
        traces.append(
            TriangleTrace(
                label=_relative(dataset.path),
                command_rad=delayed_command,
                position_rad=position,
                amplitude_rad=amplitude_rad,
                samples_per_cycle=samples_per_cycle,
                train_cycle_count=train_cycle_count,
            )
        )
        delays.append(delay_s)
        backlashes.append(float(joint_model["effective_backlash"]["median_rad"]))
        gains.append(float(reference_response["gain"]["median"]))
        provenance.append({
            "dataset": _relative(dataset.path),
            "dataset_sha256": dataset.sha256,
            "identification_model": _relative(model_path),
            "identification_model_sha256": _sha256(model_path),
            "triangle_phase_id": phase_id,
            "triangle_amplitude_rad": amplitude_rad,
            "triangle_amplitude_deg": math.degrees(amplitude_rad),
            "triangle_cycle_count": cycle_count,
            "training_cycle_count": train_cycle_count,
            "validation_cycle_count": validation_cycles,
            "delay_compensation_s": delay_s,
            "single_play_validation_r2": joint_model["effective_backlash"]["play_model"]["validation"]["r2"],
            "single_play_quality_pass": joint_model["effective_backlash"]["quality_pass"],
        })

    amplitudes = sorted({trace.amplitude_rad for trace in traces})
    if len(amplitudes) < 2:
        raise MultiAmplitudeModelError("The selected hardware datasets contain fewer than two triangle amplitudes.")
    minimum_threshold = min(amplitudes) / 20.0 if minimum_threshold_rad is None else minimum_threshold_rad
    maximum_threshold = max(amplitudes) * 0.95 if maximum_threshold_rad is None else maximum_threshold_rad
    if (
        not math.isfinite(minimum_threshold)
        or not math.isfinite(maximum_threshold)
        or minimum_threshold <= 0.0
        or maximum_threshold <= minimum_threshold
    ):
        raise MultiAmplitudeModelError("Require finite 0 < minimum_threshold_rad < maximum_threshold_rad.")
    threshold_grid = np.linspace(minimum_threshold, maximum_threshold, threshold_grid_size)
    result = fit_generalized_play(
        traces,
        threshold_grid,
        max_play_branches=max_play_branches,
        minimum_validation_r2=minimum_validation_r2,
        maximum_normalized_rmse=maximum_normalized_rmse,
    )
    usable = result["quality"]["cross_amplitude_usable"]
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "application_status": (
            "cross_amplitude_trace_validated_requires_sim_replay" if usable else "rejected_cross_amplitude_fit"
        ),
        "joint": selected_joint,
        "sample_hz": sample_hz,
        "amplitudes_rad": amplitudes,
        "amplitudes_deg": [math.degrees(value) for value in amplitudes],
        "source_datasets": provenance,
        "measured": {
            "command_delay_s": _summary(delays),
            "effective_hysteresis_rad": _summary(backlashes),
            "reference_sine_gain": _summary(gains),
        },
        "fit_configuration": {
            "max_play_branches": max_play_branches,
            "threshold_grid_size": threshold_grid_size,
            "minimum_threshold_rad": minimum_threshold,
            "maximum_threshold_rad": maximum_threshold,
            "validation_fraction": validation_fraction,
        },
        **result,
        "friction": {
            "enabled": False,
            "reason": "Current-domain friction evidence is not a calibrated joint-torque model.",
        },
        "limitations": [
            "The common model is validated on held-out suspended triangle cycles, not ground-contact motion.",
            "Per-dataset affine offsets are nuisance terms used only during fitting and are not exported to runtime.",
            "The hardware response delay aligns hysteresis during fitting; runtime delay remains zero until simulator replay.",
            "Threshold randomization remains disabled until repeatable cross-amplitude uncertainty is measured.",
            "Passing this trace gate does not authorize RL integration; fixed-base Isaac replay is still required.",
        ],
    }


def finalize_with_sim_replays(model: dict[str, Any], report_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Attach passing per-dataset Isaac replays to a cross-amplitude command-path model."""

    if model.get("model_type") != MODEL_TYPE or model.get("quality", {}).get("cross_amplitude_usable") is not True:
        raise MultiAmplitudeModelError("Only a passing multi-amplitude trace model can be finalized.")
    sources = model.get("source_datasets")
    if not isinstance(sources, list) or not sources:
        raise MultiAmplitudeModelError("Multi-amplitude model has no source dataset provenance.")
    expected: dict[Path, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("dataset"), str):
            raise MultiAmplitudeModelError("Multi-amplitude source dataset provenance is malformed.")
        dataset_path = _resolve_repo_path(source["dataset"])
        if dataset_path in expected:
            raise MultiAmplitudeModelError(f"Source dataset is duplicated: {dataset_path}")
        if _sha256(dataset_path) != source.get("dataset_sha256"):
            raise MultiAmplitudeModelError(f"Source dataset changed after fitting: {dataset_path}")
        expected[dataset_path] = source
    if len(report_paths) != len(expected):
        raise MultiAmplitudeModelError(
            f"Require exactly one replay report per source dataset: expected {len(expected)}, got {len(report_paths)}."
        )

    selected_controller: tuple[float, float] | None = None
    selected_bias: float | None = None
    selected_applied_delay: float | None = None
    calibrated_delays: list[float] = []
    reports: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for report_value in report_paths:
        report_path = _resolve_repo_path(report_value)
        report = _load_json(report_path)
        if report.get("schema_version") != 1 or report.get("validation_type") != "fixed_base_isaac_explicit_pd_replay":
            raise MultiAmplitudeModelError(f"Unsupported simulator replay report: {report_path}")
        if report.get("joint") != model.get("joint"):
            raise MultiAmplitudeModelError(f"Simulator replay joint mismatch: {report_path}")
        if report.get("sim_replay_gate_satisfied") is not True:
            raise MultiAmplitudeModelError(f"Simulator replay gate failed: {report_path}")
        dataset_path = _resolve_repo_path(str(report.get("dataset", "")))
        if dataset_path not in expected:
            raise MultiAmplitudeModelError(f"Simulator replay used an unselected dataset: {report_path}")
        if dataset_path in seen:
            raise MultiAmplitudeModelError(f"More than one replay report targets {dataset_path}.")
        seen.add(dataset_path)

        controller = report.get("controller_settings")
        if not isinstance(controller, dict):
            raise MultiAmplitudeModelError(f"Simulator replay has no controller settings: {report_path}")
        stiffness = _finite(controller.get("stiffness"), "replay stiffness", minimum=1.0e-9)
        damping = _finite(controller.get("damping"), "replay damping", minimum=0.0)
        bias = _finite(controller.get("residual_position_bias_rad", 0.0), "replay position bias")
        controller_pair = (stiffness, damping)
        if selected_controller is None:
            selected_controller = controller_pair
            selected_bias = bias
        elif controller_pair != selected_controller or not math.isclose(
            bias, float(selected_bias), rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise MultiAmplitudeModelError("Replay reports disagree on controller gains or residual position bias.")

        applied_delay = _finite(report.get("applied_residual_delay_s"), "applied residual delay", minimum=0.0)
        if selected_applied_delay is None:
            selected_applied_delay = applied_delay
        elif not math.isclose(applied_delay, selected_applied_delay, rel_tol=0.0, abs_tol=1.0e-9):
            raise MultiAmplitudeModelError("Replay reports disagree on the applied residual delay.")
        total_delay = _finite(
            report.get("recommended_total_residual_delay_s"),
            "recommended total residual delay",
            minimum=0.0,
        )
        hardware_delay = _finite(report.get("reference_hardware_delay_s"), "hardware delay", minimum=0.0)
        simulation_delay = _finite(report.get("reference_simulation_delay_s"), "simulation delay", minimum=0.0)
        phases = report.get("phases")
        if not isinstance(phases, list) or not phases:
            raise MultiAmplitudeModelError(f"Simulator replay has no phase metrics: {report_path}")
        r2_values = [_finite(phase.get("hardware_vs_simulation", {}).get("r2"), "phase R2") for phase in phases]
        nrmse_values = [
            _finite(
                phase.get("hardware_vs_simulation", {}).get("normalized_rmse"),
                "phase normalized RMSE",
                minimum=0.0,
            )
            for phase in phases
        ]
        calibrated_delays.append(total_delay)
        reports.append({
            "dataset": _relative(dataset_path),
            "report": _relative(report_path),
            "report_sha256": _sha256(report_path),
            "applied_residual_delay_s": applied_delay,
            "recommended_total_residual_delay_s": total_delay,
            "reference_hardware_delay_s": hardware_delay,
            "reference_simulation_delay_s": simulation_delay,
            "minimum_phase_r2": min(r2_values),
            "maximum_phase_normalized_rmse": max(nrmse_values),
        })
    if seen != set(expected):
        missing = sorted(str(path) for path in set(expected) - seen)
        raise MultiAmplitudeModelError(f"Source datasets are missing replay reports: {missing}")
    assert selected_controller is not None and selected_bias is not None and selected_applied_delay is not None

    result = dict(model)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["application_status"] = "sim_replay_validated_not_integrated"
    result["command_path"] = dict(model["command_path"])
    result["command_path"]["residual_delay_s_range"] = [selected_applied_delay, selected_applied_delay]
    result["command_path"]["residual_position_bias_rad_range"] = [selected_bias, selected_bias]
    result["quality"] = dict(model["quality"])
    result["quality"]["sim_replay_validated"] = True
    result["quality"]["integration_allowed"] = False
    result["sim_replay"] = {
        "validation_type": "fixed_base_isaac_explicit_pd_replay",
        "selected_controller": {
            "stiffness": selected_controller[0],
            "damping": selected_controller[1],
        },
        "residual_delay_s_range": [selected_applied_delay, selected_applied_delay],
        "recommended_total_residual_delay_s_range": [min(calibrated_delays), max(calibrated_delays)],
        "residual_position_bias_rad_range": [selected_bias, selected_bias],
        "report_count": len(reports),
        "reports": sorted(reports, key=lambda value: value["dataset"]),
    }
    result["limitations"] = [
        limitation
        for limitation in model.get("limitations", [])
        if "runtime delay remains zero" not in limitation
        and "fixed-base Isaac replay is still required" not in limitation
    ] + [
        "Fixed-base Isaac replay passed at every selected amplitude, but ground-contact dependence is unknown.",
        "This artifact remains analysis-only and is not enabled in reinforcement-learning training.",
    ]
    return result


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    args = _parser().parse_args()
    try:
        model = fit_multi_amplitude_model(
            args.datasets,
            joint_name=args.joint,
            max_play_branches=args.max_play_branches,
            threshold_grid_size=args.threshold_grid_size,
            minimum_threshold_rad=(
                None if args.minimum_threshold_deg is None else math.radians(args.minimum_threshold_deg)
            ),
            maximum_threshold_rad=(
                None if args.maximum_threshold_deg is None else math.radians(args.maximum_threshold_deg)
            ),
            validation_fraction=args.validation_fraction,
            minimum_validation_r2=args.minimum_validation_r2,
            maximum_normalized_rmse=args.maximum_normalized_rmse,
        )
        if args.sim_replay_report:
            model = finalize_with_sim_replays(model, args.sim_replay_report)
        output = Path(args.output).expanduser().resolve()
        _atomic_write_json(output, model)
        command_path = model["command_path"]
        print(f"Saved multi-amplitude actuator analysis: {output}")
        print(
            f"joint={model['joint']}, amplitudes_deg={model['amplitudes_deg']}, "
            f"play_thresholds_deg={[math.degrees(value) for value in command_path['play_thresholds_rad']]}, "
            f"play_weights={command_path['play_weights']}, linear_weight={command_path['linear_weight']:.6f}"
        )
        for dataset in model["fit"]["datasets"]:
            validation = dataset["validation"]
            print(
                f"{dataset['label']}: R2={validation['r2']:.6f}, "
                f"normalized_RMSE={validation['normalized_rmse']:.6f}, pass={dataset['validation_pass']}"
            )
        if not model["quality"]["cross_amplitude_usable"]:
            print("[ERROR] No candidate passed every cross-amplitude trace gate.", file=sys.stderr)
            return 2
        if model["quality"].get("sim_replay_validated"):
            replay = model["sim_replay"]
            print(
                "Cross-amplitude trace and Isaac replay gates passed. "
                f"Kp={replay['selected_controller']['stiffness']:.3f}, "
                f"Kd={replay['selected_controller']['damping']:.3f}, "
                f"validated_residual_delay_ms={[1000.0 * value for value in replay['residual_delay_s_range']]}, "
                "diagnostic_total_delay_ms="
                f"{[1000.0 * value for value in replay['recommended_total_residual_delay_s_range']]}"
            )
            print("RL integration remains disabled.")
        else:
            print("Cross-amplitude trace gate passed. Isaac simulator replay is still required.")
        return 0
    except (MultiAmplitudeModelError, DatasetError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
