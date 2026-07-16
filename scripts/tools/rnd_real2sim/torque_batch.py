"""Pure post-processing for all-joint torque calibration from a cached PhysX trace."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import Real2SimDataset
from .torque_identification import (
    TorqueIdentificationError,
    fit_friction_residual,
    fit_quasistatic_gravity_calibration,
)


TRACE_MATRIX_FIELDS = (
    "smoothed_velocity_rad_s",
    "gravity_torque_nm",
    "friction_residual_torque_nm",
    "low_current_extrapolation_mask",
    "high_current_clipping_mask",
)


class TorqueBatchError(ValueError):
    """Raised when a cached dynamics trace cannot support batch analysis."""


@dataclass(frozen=True)
class DynamicsTrace:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dynamics_trace(path: str | Path, dataset: Real2SimDataset) -> DynamicsTrace:
    """Load a PhysX dynamics trace and prove that it belongs to ``dataset``."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise TorqueBatchError(f"Dynamics trace does not exist: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            required = {"metadata_json", "time_s", "phase_id", *TRACE_MATRIX_FIELDS}
            missing = required - set(archive.files)
            if missing:
                raise TorqueBatchError(f"Dynamics trace is missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {name: np.asarray(archive[name]).copy() for name in required if name != "metadata_json"}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, TorqueBatchError):
            raise
        raise TorqueBatchError(f"Could not load dynamics trace {resolved}: {error}") from error

    if metadata.get("model_type") != "rnd_real2sim_dynamic_friction_analysis":
        raise TorqueBatchError("Dynamics trace metadata has an unsupported model_type.")
    if metadata.get("source_dataset_sha256") != dataset.sha256:
        raise TorqueBatchError("Dynamics trace source_dataset_sha256 does not match the requested dataset.")
    expected_matrix_shape = (dataset.sample_count, len(dataset.joint_names))
    for field in TRACE_MATRIX_FIELDS:
        if arrays[field].shape != expected_matrix_shape:
            raise TorqueBatchError(f"{field} has shape {arrays[field].shape}; expected {expected_matrix_shape}.")
        if np.issubdtype(arrays[field].dtype, np.floating) and not np.all(np.isfinite(arrays[field])):
            raise TorqueBatchError(f"{field} contains non-finite values.")
    if arrays["time_s"].shape != (dataset.sample_count,) or not np.allclose(
        arrays["time_s"], dataset.arrays["time_s"], atol=1.0e-9, rtol=0.0
    ):
        raise TorqueBatchError("Dynamics trace time_s does not match the source dataset.")
    if arrays["phase_id"].shape != (dataset.sample_count,) or not np.array_equal(
        arrays["phase_id"], dataset.arrays["phase_id"]
    ):
        raise TorqueBatchError("Dynamics trace phase_id does not match the source dataset.")
    return DynamicsTrace(resolved, metadata, arrays, _sha256(resolved))


def _selected_phase_mask(
    dataset: Real2SimDataset,
    joint_name: str,
    margin: int,
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[np.ndarray, list[int]]:
    phase_metadata = dataset.metadata.get("phase_metadata")
    if not isinstance(phase_metadata, dict):
        raise TorqueBatchError("Dataset metadata contains no phase_metadata mapping.")
    phase_ids = sorted(
        int(phase_id)
        for phase_id, phase_info in phase_metadata.items()
        if isinstance(phase_info, dict) and phase_info.get("joint_name") == joint_name and predicate(phase_info)
    )
    mask = np.zeros(dataset.sample_count, dtype=np.bool_)
    joint_index = dataset.joint_names.index(joint_name)
    for phase_id in phase_ids:
        indices = np.flatnonzero(
            (dataset.arrays["phase_id"] == phase_id) & (dataset.arrays["excitation_joint_id"] == joint_index)
        )
        if indices.size > 2 * margin:
            mask[indices[margin : indices.size - margin]] = True
    return mask, phase_ids


def _manufacturer_graph_fit(
    dataset: Real2SimDataset,
    trace: DynamicsTrace,
    joint_name: str,
    *,
    margin: int,
    transition_velocity_rad_s: float,
    minimum_fit_speed_rad_s: float,
) -> dict[str, Any]:
    joint_index = dataset.joint_names.index(joint_name)
    mask, phase_ids = _selected_phase_mask(
        dataset,
        joint_name,
        margin,
        lambda phase: phase.get("waveform") == "sine",
    )
    try:
        fit = fit_friction_residual(
            trace.arrays["friction_residual_torque_nm"][:, joint_index],
            trace.arrays["smoothed_velocity_rad_s"][:, joint_index],
            mask,
            transition_velocity_rad_s=transition_velocity_rad_s,
            minimum_speed_rad_s=minimum_fit_speed_rad_s,
        )
    except TorqueIdentificationError as error:
        return {"available": False, "phase_ids": phase_ids, "reason": str(error), "quality": {"pass": False}}

    below = trace.arrays["low_current_extrapolation_mask"][mask, joint_index]
    above = trace.arrays["high_current_clipping_mask"][mask, joint_index]
    low_fraction = float(np.mean(below))
    high_fraction = float(np.mean(above))
    observed_fraction = max(0.0, 1.0 - low_fraction - high_fraction)
    reasons: list[str] = []
    if observed_fraction < 0.8:
        reasons.append("Less than 80% of selected current samples are inside the manufacturer graph range.")
    if not fit["optimizer_success"]:
        reasons.append("Robust residual optimizer did not converge.")
    if fit["r2"] is None or fit["r2"] < 0.5:
        reasons.append("Friction residual fit R2 is below 0.5.")
    return {
        "available": True,
        "phase_ids": phase_ids,
        "observed_curve_fraction": observed_fraction,
        "low_current_extrapolation_fraction": low_fraction,
        "high_current_clipping_fraction": high_fraction,
        "fit": fit,
        "quality": {"pass": not reasons, "reasons": reasons},
    }


def analyze_joint(
    dataset: Real2SimDataset,
    trace: DynamicsTrace,
    joint_name: str,
    *,
    margin: int = 5,
    transition_velocity_rad_s: float = math.radians(4.0),
    minimum_fit_speed_rad_s: float = math.radians(1.0),
    calibration_minimum_speed_rad_s: float = math.radians(0.2),
) -> dict[str, Any]:
    """Analyze one joint while using only phases that explicitly excite it."""

    if joint_name not in dataset.joint_names:
        raise TorqueBatchError(f"Unknown dataset joint {joint_name!r}.")
    joint_index = dataset.joint_names.index(joint_name)
    calibration_mask, calibration_phase_ids = _selected_phase_mask(
        dataset,
        joint_name,
        margin,
        lambda phase: (
            phase.get("waveform") == "sine"
            and float(phase.get("frequency_hz", math.inf)) <= 0.03
            and float(phase.get("amplitude_rad", 0.0)) >= math.radians(15.0)
        ),
    )
    if calibration_phase_ids:
        try:
            calibration = {
                "available": True,
                "phase_ids": calibration_phase_ids,
                **fit_quasistatic_gravity_calibration(
                    dataset.arrays["position_rad"][:, joint_index],
                    dataset.arrays["current_a"][:, joint_index],
                    trace.arrays["smoothed_velocity_rad_s"][:, joint_index],
                    trace.arrays["gravity_torque_nm"][:, joint_index],
                    calibration_mask,
                    minimum_speed_rad_s=calibration_minimum_speed_rad_s,
                ),
            }
        except TorqueIdentificationError as error:
            calibration = {
                "available": False,
                "phase_ids": calibration_phase_ids,
                "reason": str(error),
                "quality": {"pass": False, "reasons": [str(error)]},
            }
    else:
        reason = "No qualifying quasi-static gravity-sine phase was recorded for this joint."
        calibration = {
            "available": False,
            "phase_ids": [],
            "reason": reason,
            "quality": {"pass": False, "reasons": [reason]},
        }
    return {
        "manufacturer_graph_friction": _manufacturer_graph_fit(
            dataset,
            trace,
            joint_name,
            margin=margin,
            transition_velocity_rad_s=transition_velocity_rad_s,
            minimum_fit_speed_rad_s=minimum_fit_speed_rad_s,
        ),
        "low_current_torque_calibration": calibration,
    }


def analyze_joints(
    dataset: Real2SimDataset,
    trace: DynamicsTrace,
    joint_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Analyze multiple joints and build a non-integrating quality summary."""

    if not joint_names or len(joint_names) != len(set(joint_names)):
        raise TorqueBatchError("Joint selection must be non-empty and contain no duplicates.")
    unknown = sorted(set(joint_names) - set(dataset.joint_names))
    if unknown:
        raise TorqueBatchError(f"Unknown dataset joints: {unknown}.")
    results = {joint_name: analyze_joint(dataset, trace, joint_name) for joint_name in joint_names}
    passed = [
        name
        for name, result in results.items()
        if result["low_current_torque_calibration"].get("quality", {}).get("pass") is True
    ]
    failed = [name for name in joint_names if name not in passed]
    torque_per_amp = np.asarray(
        [results[name]["low_current_torque_calibration"]["torque_per_amp_nm"] for name in passed],
        dtype=np.float64,
    )
    aggregate = None
    if torque_per_amp.size:
        median = float(np.median(torque_per_amp))
        aggregate = {
            "joint_count": int(torque_per_amp.size),
            "median_nm_per_a": median,
            "median_absolute_deviation_nm_per_a": float(np.median(np.abs(torque_per_amp - median))),
            "minimum_nm_per_a": float(np.min(torque_per_amp)),
            "maximum_nm_per_a": float(np.max(torque_per_amp)),
        }
    return results, {
        "calibration_passed_joints": passed,
        "calibration_failed_joints": failed,
        "passed_torque_per_amp_summary": aggregate,
        "automatic_integration_allowed": False,
    }
