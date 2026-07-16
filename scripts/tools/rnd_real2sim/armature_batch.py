"""Pure batch identification of residual joint armature from a cached PhysX trace."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import Real2SimDataset
from .torque_identification import (
    TorqueIdentificationError,
    fit_armature_residual,
    fit_harmonic_armature_cycles,
)


ARMATURE_TRACE_FIELDS = (
    "smoothed_velocity_rad_s",
    "smoothed_acceleration_rad_s2",
    "modeled_urdf_torque_nm",
)
TORQUE_CALIBRATION_MODEL_TYPE = "rnd_real2sim_all_joint_torque_calibration"
FRICTION_TRACE_MODEL_TYPE = "rnd_real2sim_dynamic_friction_analysis"


class ArmatureBatchError(ValueError):
    """Raised when dynamic data cannot support residual-inertia identification."""


@dataclass(frozen=True)
class ArmatureDynamicsTrace:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    sha256: str


@dataclass(frozen=True)
class TorqueCalibrationReport:
    path: Path
    data: dict[str, Any]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_armature_dynamics_trace(path: str | Path, dataset: Real2SimDataset) -> ArmatureDynamicsTrace:
    """Load and bind a zero-armature PhysX dynamics trace to its source dataset."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ArmatureBatchError(f"Dynamics trace does not exist: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            required = {"metadata_json", "time_s", "phase_id", *ARMATURE_TRACE_FIELDS}
            missing = required - set(archive.files)
            if missing:
                raise ArmatureBatchError(f"Dynamics trace is missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {name: np.asarray(archive[name]).copy() for name in required if name != "metadata_json"}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ArmatureBatchError):
            raise
        raise ArmatureBatchError(f"Could not load dynamics trace {resolved}: {error}") from error

    if metadata.get("model_type") != FRICTION_TRACE_MODEL_TYPE:
        raise ArmatureBatchError("Dynamics trace metadata has an unsupported model_type.")
    if metadata.get("source_dataset_sha256") != dataset.sha256:
        raise ArmatureBatchError("Dynamics trace source_dataset_sha256 does not match the requested dataset.")
    dynamics = metadata.get("dynamics")
    if not isinstance(dynamics, dict) or float(dynamics.get("armature_kg_m2", math.nan)) != 0.0:
        raise ArmatureBatchError("Armature identification requires a PhysX trace generated with armature_kg_m2=0.")

    matrix_shape = (dataset.sample_count, len(dataset.joint_names))
    for field in ARMATURE_TRACE_FIELDS:
        if arrays[field].shape != matrix_shape:
            raise ArmatureBatchError(f"{field} has shape {arrays[field].shape}; expected {matrix_shape}.")
        if not np.all(np.isfinite(arrays[field])):
            raise ArmatureBatchError(f"{field} contains non-finite values.")
    if arrays["time_s"].shape != (dataset.sample_count,) or not np.allclose(
        arrays["time_s"], dataset.arrays["time_s"], atol=1.0e-9, rtol=0.0
    ):
        raise ArmatureBatchError("Dynamics trace time_s does not match the source dataset.")
    if arrays["phase_id"].shape != (dataset.sample_count,) or not np.array_equal(
        arrays["phase_id"], dataset.arrays["phase_id"]
    ):
        raise ArmatureBatchError("Dynamics trace phase_id does not match the source dataset.")
    return ArmatureDynamicsTrace(resolved, metadata, arrays, _sha256(resolved))


def load_torque_calibration_report(path: str | Path) -> TorqueCalibrationReport:
    """Load the separately quality-gated current-to-joint-torque calibration."""

    resolved = Path(path).expanduser().resolve()
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArmatureBatchError(f"Torque calibration report does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise ArmatureBatchError(f"Torque calibration report is invalid JSON: {resolved}: {error}") from error
    if not isinstance(data, dict) or data.get("model_type") != TORQUE_CALIBRATION_MODEL_TYPE:
        raise ArmatureBatchError("Torque calibration report has an unsupported model_type.")
    if data.get("analysis_only") is not True or data.get("integration_enabled") is not False:
        raise ArmatureBatchError("Torque calibration report must remain analysis-only and non-integrating.")
    if not isinstance(data.get("joints"), dict):
        raise ArmatureBatchError("Torque calibration report contains no joints mapping.")
    return TorqueCalibrationReport(resolved, data, _sha256(resolved))


def _dynamic_phase_mask(
    dataset: Real2SimDataset,
    joint_name: str,
    margin: int,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    phase_metadata = dataset.metadata.get("phase_metadata")
    if not isinstance(phase_metadata, dict):
        raise ArmatureBatchError("Dataset metadata contains no phase_metadata mapping.")
    selected_phases: list[tuple[int, dict[str, Any]]] = []
    for phase_id, phase_info in phase_metadata.items():
        if not isinstance(phase_info, dict):
            continue
        profile_name = str(phase_info.get("profile_name", ""))
        if (
            phase_info.get("joint_name") == joint_name
            and phase_info.get("waveform") == "sine"
            and profile_name.startswith("armature_sine_")
        ):
            selected_phases.append((int(phase_id), phase_info))
    selected_phases.sort(key=lambda item: float(item[1].get("frequency_hz", math.inf)))

    mask = np.zeros(dataset.sample_count, dtype=np.bool_)
    joint_index = dataset.joint_names.index(joint_name)
    for phase_id, _ in selected_phases:
        indices = np.flatnonzero(
            (dataset.arrays["phase_id"] == phase_id)
            & (dataset.arrays["excitation_joint_id"] == joint_index)
        )
        if indices.size > 2 * margin:
            mask[indices[margin : indices.size - margin]] = True
    return mask, [phase_id for phase_id, _ in selected_phases], [info for _, info in selected_phases]


def _cycle_harmonic_projections(
    dataset: Real2SimDataset,
    trace: ArmatureDynamicsTrace,
    residual_torque_nm: np.ndarray,
    joint_index: int,
    phase_ids: list[int],
    phase_info: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Project each cycle's torque fundamental away from velocity and onto acceleration."""

    sample_hz = float(dataset.metadata.get("sample_hz", math.nan))
    if not math.isfinite(sample_hz) or sample_hz <= 0.0:
        raise ArmatureBatchError("Dataset metadata.sample_hz must be finite and positive.")

    frequency_values: list[float] = []
    residual_values: list[float] = []
    acceleration_values: list[float] = []
    position_values: list[float] = []
    collinearity_values: list[float] = []
    position = dataset.arrays["position_rad"][:, joint_index]
    velocity = trace.arrays["smoothed_velocity_rad_s"][:, joint_index]
    acceleration = trace.arrays["smoothed_acceleration_rad_s2"][:, joint_index]

    for phase_id, info in zip(phase_ids, phase_info, strict=True):
        frequency_hz = float(info.get("frequency_hz", math.nan))
        cycles_value = info.get("cycles")
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ArmatureBatchError(f"Phase {phase_id} has an invalid frequency_hz.")
        if isinstance(cycles_value, bool) or not isinstance(cycles_value, int) or cycles_value < 3:
            raise ArmatureBatchError(f"Phase {phase_id} must contain at least three integer cycles.")

        indices = np.flatnonzero(
            (dataset.arrays["phase_id"] == phase_id)
            & (dataset.arrays["excitation_joint_id"] == joint_index)
        )
        if indices.size < 16 * cycles_value or np.any(np.diff(indices) != 1):
            raise ArmatureBatchError(
                f"Phase {phase_id} is not a contiguous record with at least 16 samples per cycle."
            )
        boundaries = np.rint(np.linspace(0, indices.size, cycles_value + 1)).astype(np.int64)
        for cycle_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            cycle_indices = indices[start:stop]
            phase_time = np.arange(start, stop, dtype=np.float64) / sample_hz
            harmonic_design = np.column_stack(
                (
                    np.sin(2.0 * math.pi * frequency_hz * phase_time),
                    np.cos(2.0 * math.pi * frequency_hz * phase_time),
                    np.ones(phase_time.size),
                )
            )
            if np.linalg.matrix_rank(harmonic_design) < 3:
                raise ArmatureBatchError(f"Phase {phase_id} cycle {cycle_index} has a singular harmonic design.")

            harmonics = []
            for signal in (position, velocity, acceleration, residual_torque_nm):
                coefficients, *_ = np.linalg.lstsq(harmonic_design, signal[cycle_indices], rcond=None)
                harmonics.append(coefficients[:2])
            position_harmonic, velocity_harmonic, acceleration_harmonic, residual_harmonic = harmonics

            velocity_energy = float(np.dot(velocity_harmonic, velocity_harmonic))
            acceleration_norm = float(np.linalg.norm(acceleration_harmonic))
            velocity_norm = math.sqrt(velocity_energy)
            if velocity_energy <= 1.0e-12 or acceleration_norm <= 1.0e-9:
                raise ArmatureBatchError(f"Phase {phase_id} cycle {cycle_index} has no dynamic harmonic content.")
            acceleration_orthogonal = acceleration_harmonic - velocity_harmonic * (
                float(np.dot(acceleration_harmonic, velocity_harmonic)) / velocity_energy
            )
            orthogonal_norm = float(np.linalg.norm(acceleration_orthogonal))
            if orthogonal_norm <= 1.0e-9:
                raise ArmatureBatchError(
                    f"Phase {phase_id} cycle {cycle_index} cannot separate acceleration from velocity."
                )
            projection_axis = acceleration_orthogonal / orthogonal_norm
            frequency_values.append(frequency_hz)
            residual_values.append(float(np.dot(residual_harmonic, projection_axis)))
            acceleration_values.append(float(np.dot(acceleration_harmonic, projection_axis)))
            position_values.append(float(np.dot(position_harmonic, projection_axis)))
            collinearity_values.append(
                abs(float(np.dot(acceleration_harmonic, velocity_harmonic)))
                / max(acceleration_norm * velocity_norm, 1.0e-12)
            )

    return {
        "frequency_hz": np.asarray(frequency_values, dtype=np.float64),
        "residual_projection_nm": np.asarray(residual_values, dtype=np.float64),
        "acceleration_projection_rad_s2": np.asarray(acceleration_values, dtype=np.float64),
        "position_projection_rad": np.asarray(position_values, dtype=np.float64),
        "velocity_acceleration_collinearity": np.asarray(collinearity_values, dtype=np.float64),
    }


def _failed_result(reason: str, phase_ids: list[int] | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "phase_ids": phase_ids or [],
        "quality": {"pass": False, "reasons": [reason]},
    }


def analyze_armature_joint(
    dataset: Real2SimDataset,
    trace: ArmatureDynamicsTrace,
    calibration_report: TorqueCalibrationReport,
    joint_name: str,
    *,
    margin: int = 5,
    transition_velocity_rad_s: float = math.radians(4.0),
    maximum_armature_kg_m2: float = 0.1,
) -> dict[str, Any]:
    """Fit one joint using only phases that dynamically excite that joint."""

    if joint_name not in dataset.joint_names:
        raise ArmatureBatchError(f"Unknown dataset joint {joint_name!r}.")
    if margin < 1:
        raise ArmatureBatchError("Phase-interior margin must be positive.")
    mask, phase_ids, phase_info = _dynamic_phase_mask(dataset, joint_name, margin)
    if not phase_ids:
        return _failed_result("No armature_sine_* phase was recorded for this joint.")

    joint_calibration = calibration_report.data["joints"].get(joint_name)
    if not isinstance(joint_calibration, dict):
        return _failed_result("No current-to-torque calibration exists for this joint.", phase_ids)
    calibration = joint_calibration.get("low_current_torque_calibration")
    if not isinstance(calibration, dict) or calibration.get("quality", {}).get("pass") is not True:
        reasons = [] if not isinstance(calibration, dict) else calibration.get("quality", {}).get("reasons", [])
        detail = " ".join(str(reason) for reason in reasons) or "calibration quality gate failed"
        return _failed_result(
            f"Current-to-torque calibration is not valid, so physical armature cannot be identified: {detail}",
            phase_ids,
        )

    joint_index = dataset.joint_names.index(joint_name)
    torque_per_amp = float(calibration["torque_per_amp_nm"])
    calibration_bias = float(calibration["bias_nm"])
    estimated_joint_torque = torque_per_amp * dataset.arrays["current_a"][:, joint_index] + calibration_bias
    residual = estimated_joint_torque - trace.arrays["modeled_urdf_torque_nm"][:, joint_index]
    try:
        time_domain_fit = fit_armature_residual(
            residual,
            trace.arrays["smoothed_velocity_rad_s"][:, joint_index],
            trace.arrays["smoothed_acceleration_rad_s2"][:, joint_index],
            mask,
            transition_velocity_rad_s=transition_velocity_rad_s,
            maximum_armature_kg_m2=maximum_armature_kg_m2,
        )
        harmonic_projections = _cycle_harmonic_projections(
            dataset,
            trace,
            residual,
            joint_index,
            phase_ids,
            phase_info,
        )
        harmonic_fit = fit_harmonic_armature_cycles(
            harmonic_projections["residual_projection_nm"],
            harmonic_projections["acceleration_projection_rad_s2"],
            harmonic_projections["position_projection_rad"],
            harmonic_projections["frequency_hz"],
            maximum_armature_kg_m2=maximum_armature_kg_m2,
        )
    except TorqueIdentificationError as error:
        return _failed_result(str(error), phase_ids)
    except ArmatureBatchError as error:
        return _failed_result(str(error), phase_ids)

    frequencies = sorted({float(info["frequency_hz"]) for info in phase_info})
    amplitudes = np.asarray([float(info["amplitude_rad"]) for info in phase_info], dtype=np.float64)
    selected_tracking_error = (
        dataset.arrays["goal_position_rad"][mask, joint_index]
        - dataset.arrays["position_rad"][mask, joint_index]
    )
    selected_current = dataset.arrays["current_a"][mask, joint_index]
    tracking_rms = float(np.sqrt(np.mean(np.square(selected_tracking_error))))
    tracking_max = float(np.max(np.abs(selected_tracking_error)))
    amplitude_relative_span = float(np.ptp(amplitudes) / max(float(np.median(amplitudes)), 1.0e-12))

    reasons: list[str] = []
    if len(frequencies) < 3:
        reasons.append("Fewer than three distinct dynamic frequencies were recorded.")
    if amplitude_relative_span > 0.02:
        reasons.append("Dynamic profile amplitudes differ by more than 2%; frequency scaling is confounded.")
    if harmonic_fit["acceleration_p95_abs_rad_s2"] < 3.0:
        reasons.append("95th-percentile acceleration is below 3 rad/s^2.")
    if harmonic_fit["normalized_design_condition_number"] > 100.0:
        reasons.append("Normalized harmonic armature-regression condition number exceeds 100.")
    if not harmonic_fit["optimizer_success"]:
        reasons.append("Robust harmonic armature optimizer did not converge.")
    if harmonic_fit["r2"] is None or harmonic_fit["r2"] < 0.5:
        reasons.append("Cycle-harmonic armature fit R2 is below 0.5.")
    if harmonic_fit["rmse_improvement_over_position_only"] < 0.1:
        reasons.append("Adding armature improves harmonic RMSE by less than 10% over position-only fitting.")
    if harmonic_fit["armature_torque_rms_nm"] < 0.01:
        reasons.append("Fitted armature contributes less than 0.01 Nm RMS and is not resolved above telemetry noise.")
    if harmonic_fit["armature_kg_m2"] >= 0.95 * maximum_armature_kg_m2:
        reasons.append("Armature estimate is too close to the configured optimizer upper bound.")
    interval_low, _ = harmonic_fit["bootstrap_90pct_kg_m2"]
    if interval_low <= 0.0:
        reasons.append("Harmonic bootstrap 90% interval does not exclude zero armature.")
    if harmonic_fit["bootstrap_relative_interval_width"] > 0.5:
        reasons.append("Harmonic bootstrap 90% interval is wider than 50% of its median.")
    if harmonic_fit["reliable_frequency_count"] < 2:
        reasons.append("Fewer than two frequencies have enough acceleration for consistency validation.")
    frequency_span = harmonic_fit["reliable_frequency_relative_span"]
    if frequency_span is None or frequency_span > 0.5:
        reasons.append("Reliable-frequency armature estimates differ by more than 50% of their median.")
    max_collinearity = float(np.max(harmonic_projections["velocity_acceleration_collinearity"]))
    if max_collinearity > 0.5:
        reasons.append("Velocity and acceleration harmonics are too collinear for reliable separation.")
    if tracking_rms > math.radians(2.0):
        reasons.append("Dynamic tracking-error RMS exceeds 2 degrees.")
    if tracking_max > math.radians(8.0):
        reasons.append("Dynamic tracking-error peak exceeds 8 degrees.")

    return {
        "available": True,
        "phase_ids": phase_ids,
        "frequencies_hz": frequencies,
        "amplitudes_rad": amplitudes.tolist(),
        "amplitude_relative_span": amplitude_relative_span,
        "torque_conversion": {
            "method": "quality-gated matched-position gravity calibration",
            "torque_per_amp_nm": torque_per_amp,
            "bias_nm": calibration_bias,
            "source_quality_pass": True,
        },
        "dynamic_range": {
            "tracking_error_rms_rad": tracking_rms,
            "tracking_error_max_abs_rad": tracking_max,
            "current_rms_a": float(np.sqrt(np.mean(np.square(selected_current)))),
            "current_max_abs_a": float(np.max(np.abs(selected_current))),
            "maximum_velocity_acceleration_harmonic_collinearity": max_collinearity,
        },
        "selected_estimator": "cycle_harmonic_acceleration_projection",
        "fit": harmonic_fit,
        "time_domain_fit": time_domain_fit,
        "quality": {"pass": not reasons, "reasons": reasons},
    }


def analyze_armature_joints(
    dataset: Real2SimDataset,
    trace: ArmatureDynamicsTrace,
    calibration_report: TorqueCalibrationReport,
    joint_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Analyze selected joints and produce a deliberately non-integrating summary."""

    if not joint_names or len(joint_names) != len(set(joint_names)):
        raise ArmatureBatchError("Joint selection must be non-empty and contain no duplicates.")
    unknown = sorted(set(joint_names) - set(dataset.joint_names))
    if unknown:
        raise ArmatureBatchError(f"Unknown dataset joints: {unknown}.")

    results = {
        joint_name: analyze_armature_joint(dataset, trace, calibration_report, joint_name)
        for joint_name in joint_names
    }
    passed = [name for name, result in results.items() if result["quality"]["pass"] is True]
    failed = [name for name in joint_names if name not in passed]
    estimates = np.asarray([results[name]["fit"]["armature_kg_m2"] for name in passed], dtype=np.float64)
    aggregate = None
    if estimates.size:
        median = float(np.median(estimates))
        aggregate = {
            "joint_count": int(estimates.size),
            "median_kg_m2": median,
            "median_absolute_deviation_kg_m2": float(np.median(np.abs(estimates - median))),
            "minimum_kg_m2": float(np.min(estimates)),
            "maximum_kg_m2": float(np.max(estimates)),
        }
    return results, {
        "armature_passed_joints": passed,
        "armature_failed_joints": failed,
        "passed_armature_summary": aggregate,
        "automatic_integration_allowed": False,
    }
