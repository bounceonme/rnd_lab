"""Build and validate the promoted CMP10A policy-observation runtime contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .identification import load_imu_dataset


CMP10A_RUNTIME_SCHEMA_VERSION = 1
CMP10A_RUNTIME_MODEL_TYPE = "rnd_cmp10a_policy_observation"
CMP10A_RUNTIME_SIGNED_MAPPING = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)
CMP10A_POLICY_HZ = 50.0
CMP10A_POLICY_ANGULAR_VELOCITY_SCALE = 0.25
CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S = 0.030
CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S = 0.020
CMP10A_BASELINE_DISCARD_INITIAL_S = 1.0

CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S = (-0.01, 0.01)
CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S = (0.0003, 0.003)
CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD = (0.00005, 0.002)
CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S = (0.0, 0.005)
CMP10A_ORIENTATION_DELAY_RANGE_S = (0.0, 0.020)

CMP10A_STRESS_ONLY_GYRO_RANGE_RAD_S = (-0.2, 0.2)
CMP10A_STRESS_ONLY_GRAVITY_COMPONENT_RANGE = (-0.05, 0.05)

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[2]
_DYNAMIC_STAGES = ("dynamic_axis_x", "dynamic_axis_y", "dynamic_axis_z")
_AXIS_NAMES = ("x", "y", "z")
_GYRO_FRAME_TYPE = 0x52
_EULER_FRAME_TYPE = 0x53


class Cmp10aRuntimeConfigError(ValueError):
    """Raised when source evidence cannot support the runtime contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Cmp10aRuntimeConfigError(f"{label} must be a mapping.")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise Cmp10aRuntimeConfigError(f"{label} must be numeric, got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Cmp10aRuntimeConfigError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise Cmp10aRuntimeConfigError(f"{label} must be finite, got {result!r}.")
    if positive and result <= 0.0:
        raise Cmp10aRuntimeConfigError(f"{label} must be positive, got {result}.")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Cmp10aRuntimeConfigError(f"{label} must be an integer, got {value!r}.")
    result = int(value)
    if result < minimum:
        raise Cmp10aRuntimeConfigError(f"{label} must be at least {minimum}, got {result}.")
    return result


def _vector(value: Any, label: str, size: int = 3) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != size:
        raise Cmp10aRuntimeConfigError(f"{label} must contain exactly {size} numeric values.")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _matrix(value: Any, label: str) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise Cmp10aRuntimeConfigError(f"{label} must be a 3x3 matrix.")
    return [_vector(row, f"{label}[{row_index}]") for row_index, row in enumerate(value)]


def _expected_matrix(value: Any, label: str) -> list[list[float]]:
    matrix = _matrix(value, label)
    expected = [list(row) for row in CMP10A_RUNTIME_SIGNED_MAPPING]
    if matrix != expected:
        raise Cmp10aRuntimeConfigError(
            f"{label} must be the accepted aligned-mount mapping diag(-1, -1, +1), got {matrix}."
        )
    return matrix


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Cmp10aRuntimeConfigError(f"{label} must be a lowercase 64-character SHA-256 digest.")
    return value


def _source_path(path: str | Path, label: str) -> str:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (_REPO_ROOT / candidate).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError as error:
        raise Cmp10aRuntimeConfigError(f"{label} must be inside the repository: {resolved}") from error


def _validate_relative_source_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Cmp10aRuntimeConfigError(f"{label} must be a non-empty repository-relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Cmp10aRuntimeConfigError(f"{label} must be a repository-relative path without '..'.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Cmp10aRuntimeConfigError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise Cmp10aRuntimeConfigError(f"{label} is not valid JSON: {path}: {error}") from error
    except OSError as error:
        raise Cmp10aRuntimeConfigError(f"Unable to read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise Cmp10aRuntimeConfigError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_static_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise Cmp10aRuntimeConfigError("static_report must use schema_version=1.")
    policy_hz = _finite(report.get("policy_hz"), "static_report.policy_hz", positive=True)
    if policy_hz != CMP10A_POLICY_HZ:
        raise Cmp10aRuntimeConfigError(f"static_report.policy_hz must be {CMP10A_POLICY_HZ}.")

    runtime_gate = _mapping(report.get("runtime_gate"), "static_report.runtime_gate")
    if runtime_gate.get("quality_pass") is not True:
        raise Cmp10aRuntimeConfigError("static_report runtime quality gate did not pass.")
    mount = _mapping(report.get("mount_axis_identification"), "static_report.mount_axis_identification")
    if mount.get("quality_pass") is not True:
        raise Cmp10aRuntimeConfigError("static_report mount-axis quality gate did not pass.")

    signed_mapping = _expected_matrix(
        mount.get("signed_axis_approximation"),
        "static_report.mount_axis_identification.signed_axis_approximation",
    )
    fitted_matrix = _matrix(
        mount.get("sensor_to_base_matrix"),
        "static_report.mount_axis_identification.sensor_to_base_matrix",
    )
    fit_error_deg = _vector(
        mount.get("axis_fit_error_deg"),
        "static_report.mount_axis_identification.axis_fit_error_deg",
    )
    gyro_rate_hz = _finite(runtime_gate.get("gyro_rate_hz"), "static_report.runtime_gate.gyro_rate_hz", positive=True)
    orientation_rate_hz = _finite(
        runtime_gate.get("orientation_rate_hz"),
        "static_report.runtime_gate.orientation_rate_hz",
        positive=True,
    )
    if gyro_rate_hz < policy_hz or orientation_rate_hz < policy_hz:
        raise Cmp10aRuntimeConfigError("static_report packet rates must meet the policy rate.")
    if runtime_gate.get("orientation_source") != "euler_angle":
        raise Cmp10aRuntimeConfigError("static_report must select the measured Euler-angle orientation stream.")
    return {
        "policy_hz": policy_hz,
        "gyro_rate_hz": gyro_rate_hz,
        "orientation_rate_hz": orientation_rate_hz,
        "orientation_source": "euler_angle",
        "signed_mapping": signed_mapping,
        "fitted_matrix": fitted_matrix,
        "fit_error_deg": fit_error_deg,
        "fit_determinant": _finite(mount.get("determinant"), "static_report.mount_axis_identification.determinant"),
        "fit_condition_number": _finite(
            mount.get("source_direction_condition_number"),
            "static_report.mount_axis_identification.source_direction_condition_number",
            positive=True,
        ),
    }


def _validate_dynamic_report(report: Mapping[str, Any], static_sha256: str) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise Cmp10aRuntimeConfigError("dynamic_report must use schema_version=1.")
    if report.get("quality_pass") is not True:
        raise Cmp10aRuntimeConfigError("dynamic_report quality gate did not pass.")
    if report.get("absolute_usb_latency") is not False:
        raise Cmp10aRuntimeConfigError("dynamic_report must not claim an absolute USB latency measurement.")
    if _finite(report.get("policy_hz"), "dynamic_report.policy_hz", positive=True) != CMP10A_POLICY_HZ:
        raise Cmp10aRuntimeConfigError(f"dynamic_report.policy_hz must be {CMP10A_POLICY_HZ}.")

    communication = _mapping(report.get("communication"), "dynamic_report.communication")
    if communication.get("quality_pass") is not True:
        raise Cmp10aRuntimeConfigError("dynamic_report communication quality gate did not pass.")
    source = _mapping(report.get("source"), "dynamic_report.source")
    referenced_sha256 = _digest(
        source.get("identification_report_sha256"),
        "dynamic_report.source.identification_report_sha256",
    )
    if referenced_sha256 != static_sha256:
        raise Cmp10aRuntimeConfigError(
            "dynamic_report references a different static report SHA-256 than the promoted static report."
        )
    signed_mapping = _expected_matrix(
        source.get("sensor_to_base_signed_mapping"),
        "dynamic_report.source.sensor_to_base_signed_mapping",
    )
    sensor_baudrate = _integer(source.get("baudrate"), "dynamic_report.source.baudrate", minimum=1)

    stages = _mapping(report.get("stages"), "dynamic_report.stages")
    if set(stages) != set(_DYNAMIC_STAGES):
        raise Cmp10aRuntimeConfigError("dynamic_report.stages must contain exactly the X, Y, and Z dynamic trials.")
    if report.get("passing_stages") != list(_DYNAMIC_STAGES) or report.get("passing_stage_count") != 3:
        raise Cmp10aRuntimeConfigError("dynamic_report passing-stage summary does not pass all three axes.")

    axis_results: dict[str, Any] = {}
    delays_ms: list[float] = []
    for axis_name, stage_name in zip(_AXIS_NAMES, _DYNAMIC_STAGES):
        stage = _mapping(stages[stage_name], f"dynamic_report.stages.{stage_name}")
        if stage.get("quality_pass") is not True:
            raise Cmp10aRuntimeConfigError(f"dynamic_report stage {stage_name} did not pass its quality gate.")
        delay_ms = _finite(stage.get("delay_ms"), f"dynamic_report.stages.{stage_name}.delay_ms")
        gain_ratio = _finite(
            stage.get("gain_ratio"),
            f"dynamic_report.stages.{stage_name}.gain_ratio",
            positive=True,
        )
        delays_ms.append(delay_ms)
        axis_results[axis_name] = {
            "stage": stage_name,
            "correlation": _finite(
                stage.get("correlation"),
                f"dynamic_report.stages.{stage_name}.correlation",
            ),
            "relative_orientation_to_gyro_delay_ms": delay_ms,
            "euler_derived_angular_velocity_to_gyro_gain_ratio": gain_ratio,
        }

    median_delay_ms = _finite(
        report.get("median_relative_delay_ms"),
        "dynamic_report.median_relative_delay_ms",
    )
    if not math.isclose(median_delay_ms, float(np.median(delays_ms)), rel_tol=0.0, abs_tol=1.0e-12):
        raise Cmp10aRuntimeConfigError("dynamic_report median delay disagrees with its passing stage delays.")
    return {
        "source": source,
        "communication": communication,
        "signed_mapping": signed_mapping,
        "sensor_baudrate": sensor_baudrate,
        "axis_results": axis_results,
        "median_delay_ms": median_delay_ms,
        "delay_definition": str(report.get("delay_definition")),
    }


def _validate_dataset_link(
    dataset: Mapping[str, Any],
    dynamic: Mapping[str, Any],
    static_sha256: str,
) -> Mapping[str, Any]:
    metadata = _mapping(dataset.get("metadata"), "dynamic_dataset.metadata")
    if metadata.get("schema_version") != 1 or metadata.get("experiment") != "cmp10a_dynamic_consistency":
        raise Cmp10aRuntimeConfigError("dynamic_dataset metadata is not the accepted CMP10A dynamic experiment.")
    dataset_static_sha256 = _digest(
        metadata.get("identification_report_sha256"),
        "dynamic_dataset.metadata.identification_report_sha256",
    )
    if dataset_static_sha256 != static_sha256:
        raise Cmp10aRuntimeConfigError(
            "dynamic_dataset references a different static report SHA-256 than the promoted static report."
        )
    dataset_mapping = _expected_matrix(
        metadata.get("sensor_to_base_signed_mapping"),
        "dynamic_dataset.metadata.sensor_to_base_signed_mapping",
    )
    if dataset_mapping != dynamic["signed_mapping"]:
        raise Cmp10aRuntimeConfigError("dynamic_dataset and dynamic_report signed mappings disagree.")

    source = dynamic["source"]
    for key in ("experiment", "device", "baudrate", "mount_location"):
        if metadata.get(key) != source.get(key):
            raise Cmp10aRuntimeConfigError(f"dynamic_dataset metadata {key!r} disagrees with dynamic_report.source.")

    parser_stats = _mapping(metadata.get("parser_stats"), "dynamic_dataset.metadata.parser_stats")
    communication = dynamic["communication"]
    for key in ("valid_frames", "checksum_failures", "garbage_bytes"):
        if _integer(parser_stats.get(key), f"dynamic_dataset.metadata.parser_stats.{key}") != _integer(
            communication.get(key), f"dynamic_report.communication.{key}"
        ):
            raise Cmp10aRuntimeConfigError(f"dynamic_dataset parser count {key!r} disagrees with dynamic_report.")
    return metadata


def _baseline_stream(
    timestamps_ns: np.ndarray,
    frame_types: np.ndarray,
    values: np.ndarray,
    retained_mask: np.ndarray,
    frame_type: int,
    label: str,
    *,
    unwrap: bool = False,
) -> tuple[np.ndarray, float]:
    mask = retained_mask & (frame_types == frame_type)
    selected_timestamps = np.asarray(timestamps_ns[mask], dtype=np.int64)
    selected_values = np.asarray(values[mask], dtype=np.float64)
    if selected_values.ndim != 2 or selected_values.shape[1] != 3 or selected_values.shape[0] < 2:
        raise Cmp10aRuntimeConfigError(
            f"dynamic_dataset needs at least two retained {label} samples with shape [N, 3]."
        )
    if not np.all(np.isfinite(selected_values)):
        raise Cmp10aRuntimeConfigError(f"dynamic_dataset retained {label} samples contain non-finite values.")
    order = np.argsort(selected_timestamps, kind="stable")
    selected_timestamps = selected_timestamps[order]
    selected_values = selected_values[order]
    if np.any(np.diff(selected_timestamps) <= 0):
        raise Cmp10aRuntimeConfigError(f"dynamic_dataset retained {label} timestamps must be strictly increasing.")
    if unwrap:
        selected_values = np.unwrap(selected_values, axis=0)
    elapsed_s = float((selected_timestamps[-1] - selected_timestamps[0]) * 1.0e-9)
    if elapsed_s <= 0.0:
        raise Cmp10aRuntimeConfigError(f"dynamic_dataset retained {label} timestamps have no positive duration.")
    return selected_values, float((selected_values.shape[0] - 1) / elapsed_s)


def _held_baseline(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = ("timestamp_ns", "stage", "frame_type", "gyro_rad_s", "euler_rad")
    missing = [name for name in required if name not in dataset]
    if missing:
        raise Cmp10aRuntimeConfigError(f"dynamic_dataset is missing required arrays: {', '.join(missing)}.")

    timestamps_ns = np.asarray(dataset["timestamp_ns"])
    stages = np.asarray(dataset["stage"])
    frame_types = np.asarray(dataset["frame_type"])
    gyro = np.asarray(dataset["gyro_rad_s"])
    euler = np.asarray(dataset["euler_rad"])
    if timestamps_ns.ndim != 1 or stages.ndim != 1 or frame_types.ndim != 1:
        raise Cmp10aRuntimeConfigError(
            "dynamic_dataset timestamp, stage, and frame-type arrays must be one-dimensional."
        )
    row_count = timestamps_ns.shape[0]
    if stages.shape[0] != row_count or frame_types.shape[0] != row_count:
        raise Cmp10aRuntimeConfigError("dynamic_dataset arrays do not share a common row count.")
    if gyro.shape != (row_count, 3) or euler.shape != (row_count, 3):
        raise Cmp10aRuntimeConfigError("dynamic_dataset gyro and Euler arrays must have shape [rows, 3].")

    baseline_mask = stages == "dynamic_baseline"
    if not np.any(baseline_mask):
        raise Cmp10aRuntimeConfigError("dynamic_dataset has no dynamic_baseline stage.")
    baseline_start_ns = int(np.min(np.asarray(timestamps_ns[baseline_mask], dtype=np.int64)))
    discard_ns = int(round(CMP10A_BASELINE_DISCARD_INITIAL_S * 1.0e9))
    retained_mask = baseline_mask & (timestamps_ns >= baseline_start_ns + discard_ns)
    gyro_values, gyro_rate_hz = _baseline_stream(
        timestamps_ns,
        frame_types,
        gyro,
        retained_mask,
        _GYRO_FRAME_TYPE,
        "gyro",
    )
    euler_values, euler_rate_hz = _baseline_stream(
        timestamps_ns,
        frame_types,
        euler,
        retained_mask,
        _EULER_FRAME_TYPE,
        "Euler",
        unwrap=True,
    )
    return {
        "stage": "dynamic_baseline",
        "discard_initial_s": CMP10A_BASELINE_DISCARD_INITIAL_S,
        "statistics_frame": "sensor",
        "gyro": {
            "samples": int(gyro_values.shape[0]),
            "mean_rad_s": np.mean(gyro_values, axis=0).tolist(),
            "std_rad_s": np.std(gyro_values, axis=0, ddof=1).tolist(),
            "rate_hz": gyro_rate_hz,
        },
        "euler": {
            "samples": int(euler_values.shape[0]),
            "std_rad": np.std(euler_values, axis=0, ddof=1).tolist(),
            "rate_hz": euler_rate_hz,
            "unwrapped_before_std": True,
        },
        "standard_deviation_definition": "sample standard deviation (ddof=1)",
    }


def _created_utc(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value:
        raise Cmp10aRuntimeConfigError("created_utc must be a non-empty ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise Cmp10aRuntimeConfigError("created_utc must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise Cmp10aRuntimeConfigError("created_utc must include a timezone.")
    return value


def build_cmp10a_runtime_config(
    static_report: Mapping[str, Any],
    dynamic_report: Mapping[str, Any],
    dynamic_dataset: Mapping[str, Any],
    *,
    static_report_path: str | Path,
    static_report_sha256: str,
    dynamic_report_path: str | Path,
    dynamic_report_sha256: str,
    dynamic_dataset_path: str | Path,
    dynamic_dataset_sha256: str,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Promote passing CMP10A evidence into the policy-observation runtime model."""

    static_digest = _digest(static_report_sha256, "static_report_sha256")
    dynamic_digest = _digest(dynamic_report_sha256, "dynamic_report_sha256")
    dataset_digest = _digest(dynamic_dataset_sha256, "dynamic_dataset_sha256")
    static = _validate_static_report(_mapping(static_report, "static_report"))
    dynamic = _validate_dynamic_report(_mapping(dynamic_report, "dynamic_report"), static_digest)
    _validate_dataset_link(_mapping(dynamic_dataset, "dynamic_dataset"), dynamic, static_digest)
    if static["signed_mapping"] != dynamic["signed_mapping"]:
        raise Cmp10aRuntimeConfigError("static_report and dynamic_report signed mappings disagree.")
    baseline = _held_baseline(dynamic_dataset)

    result = {
        "schema_version": CMP10A_RUNTIME_SCHEMA_VERSION,
        "model_type": CMP10A_RUNTIME_MODEL_TYPE,
        "integration_enabled": True,
        "created_utc": _created_utc(created_utc),
        "sensor_to_base_matrix": dynamic["signed_mapping"],
        "gyro_bias_rad_s": baseline["gyro"]["mean_rad_s"],
        "policy_angular_velocity_scale": CMP10A_POLICY_ANGULAR_VELOCITY_SCALE,
        "sensor_baudrate": dynamic["sensor_baudrate"],
        "sensor_rates_hz": {
            "gyro": static["gyro_rate_hz"],
            "euler": static["orientation_rate_hz"],
        },
        "policy_hz": CMP10A_POLICY_HZ,
        "runtime": {
            "max_frame_age_ms": CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S * 1.0e3,
            "max_pair_skew_ms": CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S * 1.0e3,
        },
        "source_static_report": _source_path(static_report_path, "static_report_path"),
        "source_static_report_sha256": static_digest,
        "source_dynamic_report": _source_path(dynamic_report_path, "dynamic_report_path"),
        "source_dynamic_report_sha256": dynamic_digest,
        "source_dynamic_dataset": _source_path(dynamic_dataset_path, "dynamic_dataset_path"),
        "source_dynamic_dataset_sha256": dataset_digest,
        "provenance_chain": {
            "dynamic_report_identification_report_sha256": static_digest,
            "dynamic_dataset_identification_report_sha256": static_digest,
        },
        "quality_gates": {
            "static_runtime_gate_pass": True,
            "static_mount_axis_gate_pass": True,
            "dynamic_consistency_gate_pass": True,
            "dynamic_communication_gate_pass": True,
            "promotion_pass": True,
        },
        "runtime_transform": {
            "source_frame": "sensor",
            "target_frame": "base_link",
            "sensor_to_base_matrix": dynamic["signed_mapping"],
            "source": "dynamic_report.source.sensor_to_base_signed_mapping",
            "alignment_basis": "user_confirmed_sensor_aligned",
        },
        "runtime_limits": {
            "maximum_sample_age_s": CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S,
            "maximum_gyro_orientation_pair_skew_s": CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S,
        },
        "policy_observation": {
            "policy_hz": CMP10A_POLICY_HZ,
            "angular_velocity_scale": CMP10A_POLICY_ANGULAR_VELOCITY_SCALE,
            "angular_velocity_units": "rad/s before policy scaling",
            "projected_gravity_representation": "unit vector in base_link",
        },
        "measured": {
            "packet_rates": {
                "gyro_rate_hz": static["gyro_rate_hz"],
                "orientation_rate_hz": static["orientation_rate_hz"],
                "orientation_source": static["orientation_source"],
                "source": "static_report.runtime_gate",
            },
            "held_baseline": baseline,
            "dynamic_consistency": {
                "median_relative_orientation_to_gyro_delay_ms": dynamic["median_delay_ms"],
                "delay_definition": dynamic["delay_definition"],
                "axis_results": dynamic["axis_results"],
                "gain_scope": "Euler-derived body angular velocity divided by gyro angular velocity",
                "static_gravity_gain_measured": False,
            },
            "mount_fit_evidence": {
                "sensor_to_base_fitted_matrix": static["fitted_matrix"],
                "axis_fit_error_deg": static["fit_error_deg"],
                "determinant": static["fit_determinant"],
                "source_direction_condition_number": static["fit_condition_number"],
                "applied_at_runtime": False,
                "reason": (
                    "Manual rotation trials contained cross-axis error; the aligned-mount signed mapping is used "
                    "at runtime and the full fit is retained only as evidence."
                ),
            },
        },
        "assumed_simulation_envelopes": {
            "evidence_status": "assumed_not_measured",
            "residual_gyro_episode_bias": {
                "distribution": "uniform",
                "frame": "sensor",
                "range_rad_s_per_axis": list(CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S),
                "sampling_scope": "per_episode_per_sensor_axis",
            },
            "gyro_white_noise": {
                "distribution": "zero_mean_gaussian",
                "sigma_range_rad_s": list(CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S),
            },
            "projected_gravity_tangent_angle_noise": {
                "distribution": "zero_mean_tangent_plane_gaussian",
                "sigma_range_rad": list(CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD),
            },
            "gyro_sample_age_delay": {
                "range_s": list(CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S),
            },
            "orientation_delay": {
                "range_s": list(CMP10A_ORIENTATION_DELAY_RANGE_S),
            },
        },
        "unmeasured_quantities": {
            "absolute_transport_latency_s": None,
            "absolute_transport_latency_reason": "No synchronized external timing reference was used.",
            "level_offset_rad": None,
            "level_offset_reason": "The held baseline was not an externally referenced level calibration.",
        },
        "stress_only_provenance": {
            "evidence_status": "legacy_stress_only_not_measurement",
            "applied_to_runtime_model": False,
            "gyro_component_range_rad_s": list(CMP10A_STRESS_ONLY_GYRO_RANGE_RAD_S),
            "projected_gravity_component_range": list(CMP10A_STRESS_ONLY_GRAVITY_COMPONENT_RANGE),
        },
        "limitations": [
            "Absolute sensor-to-host transport latency remains unmeasured.",
            "No level offset is estimated or corrected by this model.",
            "Dynamic gain ratios describe Euler-derived angular velocity versus gyro only; they are not static gravity gains.",
            "Simulation envelopes are explicit assumptions and are not claimed as measured confidence intervals.",
        ],
    }
    validate_cmp10a_runtime_config(result)
    return result


def _validate_exact_range(value: Any, expected: tuple[float, float], label: str) -> None:
    actual = _vector(value, label, size=2)
    if actual != list(expected):
        raise Cmp10aRuntimeConfigError(f"{label} must equal {list(expected)}, got {actual}.")


def validate_cmp10a_runtime_config(config: Mapping[str, Any]) -> None:
    """Fail closed if a serialized CMP10A runtime contract drifts from schema v1."""

    model = _mapping(config, "runtime_config")
    if model.get("schema_version") != CMP10A_RUNTIME_SCHEMA_VERSION:
        raise Cmp10aRuntimeConfigError(f"runtime_config.schema_version must be {CMP10A_RUNTIME_SCHEMA_VERSION}.")
    if model.get("model_type") != CMP10A_RUNTIME_MODEL_TYPE:
        raise Cmp10aRuntimeConfigError(f"runtime_config.model_type must be {CMP10A_RUNTIME_MODEL_TYPE!r}.")
    if model.get("integration_enabled") is not True:
        raise Cmp10aRuntimeConfigError("runtime_config.integration_enabled must be true.")
    _created_utc(model.get("created_utc"))

    for path_key, digest_key in (
        ("source_static_report", "source_static_report_sha256"),
        ("source_dynamic_report", "source_dynamic_report_sha256"),
        ("source_dynamic_dataset", "source_dynamic_dataset_sha256"),
    ):
        _validate_relative_source_path(model.get(path_key), f"runtime_config.{path_key}")
        _digest(model.get(digest_key), f"runtime_config.{digest_key}")
    static_digest = model["source_static_report_sha256"]
    chain = _mapping(model.get("provenance_chain"), "runtime_config.provenance_chain")
    if (
        chain.get("dynamic_report_identification_report_sha256") != static_digest
        or chain.get("dynamic_dataset_identification_report_sha256") != static_digest
    ):
        raise Cmp10aRuntimeConfigError("runtime_config provenance chain must point to source_static_report_sha256.")

    quality = _mapping(model.get("quality_gates"), "runtime_config.quality_gates")
    required_gates = (
        "static_runtime_gate_pass",
        "static_mount_axis_gate_pass",
        "dynamic_consistency_gate_pass",
        "dynamic_communication_gate_pass",
        "promotion_pass",
    )
    if any(quality.get(name) is not True for name in required_gates):
        raise Cmp10aRuntimeConfigError("runtime_config must retain every passing promotion quality gate.")

    runtime_matrix = _expected_matrix(model.get("sensor_to_base_matrix"), "runtime_config.sensor_to_base_matrix")
    transform = _mapping(model.get("runtime_transform"), "runtime_config.runtime_transform")
    evidence_matrix = _expected_matrix(
        transform.get("sensor_to_base_matrix"), "runtime_config.runtime_transform.sensor_to_base_matrix"
    )
    if evidence_matrix != runtime_matrix:
        raise Cmp10aRuntimeConfigError("runtime transform evidence must match the canonical sensor_to_base_matrix.")
    if transform.get("source") != "dynamic_report.source.sensor_to_base_signed_mapping":
        raise Cmp10aRuntimeConfigError("runtime transform must come from the dynamic-report signed mapping.")

    runtime = _mapping(model.get("runtime"), "runtime_config.runtime")
    if runtime.get("max_frame_age_ms") != CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S * 1.0e3:
        raise Cmp10aRuntimeConfigError("runtime max_frame_age_ms has drifted from 30 ms.")
    if runtime.get("max_pair_skew_ms") != CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S * 1.0e3:
        raise Cmp10aRuntimeConfigError("runtime max_pair_skew_ms has drifted from 20 ms.")
    limits = _mapping(model.get("runtime_limits"), "runtime_config.runtime_limits")
    if limits.get("maximum_sample_age_s") != CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S:
        raise Cmp10aRuntimeConfigError("runtime maximum sample age has drifted from 0.030 s.")
    if limits.get("maximum_gyro_orientation_pair_skew_s") != CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S:
        raise Cmp10aRuntimeConfigError("runtime gyro/orientation pair skew has drifted from 0.020 s.")

    policy = _mapping(model.get("policy_observation"), "runtime_config.policy_observation")
    if model.get("policy_hz") != CMP10A_POLICY_HZ:
        raise Cmp10aRuntimeConfigError("canonical runtime policy_hz has drifted from 50 Hz.")
    if model.get("policy_angular_velocity_scale") != CMP10A_POLICY_ANGULAR_VELOCITY_SCALE:
        raise Cmp10aRuntimeConfigError("canonical runtime angular-velocity scale has drifted from 0.25.")
    if policy.get("policy_hz") != CMP10A_POLICY_HZ:
        raise Cmp10aRuntimeConfigError("runtime policy rate has drifted from 50 Hz.")
    if policy.get("angular_velocity_scale") != CMP10A_POLICY_ANGULAR_VELOCITY_SCALE:
        raise Cmp10aRuntimeConfigError("runtime angular-velocity observation scale has drifted from 0.25.")

    measured = _mapping(model.get("measured"), "runtime_config.measured")
    packet_rates = _mapping(measured.get("packet_rates"), "runtime_config.measured.packet_rates")
    measured_gyro_rate = _finite(
        packet_rates.get("gyro_rate_hz"), "runtime_config.measured.packet_rates.gyro_rate_hz", positive=True
    )
    measured_euler_rate = _finite(
        packet_rates.get("orientation_rate_hz"),
        "runtime_config.measured.packet_rates.orientation_rate_hz",
        positive=True,
    )
    sensor_rates = _mapping(model.get("sensor_rates_hz"), "runtime_config.sensor_rates_hz")
    if (
        _finite(sensor_rates.get("gyro"), "runtime_config.sensor_rates_hz.gyro", positive=True) != measured_gyro_rate
        or _finite(sensor_rates.get("euler"), "runtime_config.sensor_rates_hz.euler", positive=True)
        != measured_euler_rate
    ):
        raise Cmp10aRuntimeConfigError("canonical sensor rates must match the measured packet rates.")
    _integer(model.get("sensor_baudrate"), "runtime_config.sensor_baudrate", minimum=1)
    baseline = _mapping(measured.get("held_baseline"), "runtime_config.measured.held_baseline")
    if baseline.get("discard_initial_s") != CMP10A_BASELINE_DISCARD_INITIAL_S:
        raise Cmp10aRuntimeConfigError("held baseline must discard its first 1.0 s.")
    gyro = _mapping(baseline.get("gyro"), "runtime_config.measured.held_baseline.gyro")
    euler = _mapping(baseline.get("euler"), "runtime_config.measured.held_baseline.euler")
    _integer(gyro.get("samples"), "runtime_config.measured.held_baseline.gyro.samples", minimum=2)
    measured_bias = _vector(gyro.get("mean_rad_s"), "runtime_config.measured.held_baseline.gyro.mean_rad_s")
    if _vector(model.get("gyro_bias_rad_s"), "runtime_config.gyro_bias_rad_s") != measured_bias:
        raise Cmp10aRuntimeConfigError("canonical gyro bias must equal the measured held-baseline sensor-frame mean.")
    _vector(gyro.get("std_rad_s"), "runtime_config.measured.held_baseline.gyro.std_rad_s")
    _integer(euler.get("samples"), "runtime_config.measured.held_baseline.euler.samples", minimum=2)
    _vector(euler.get("std_rad"), "runtime_config.measured.held_baseline.euler.std_rad")

    dynamic = _mapping(measured.get("dynamic_consistency"), "runtime_config.measured.dynamic_consistency")
    _finite(
        dynamic.get("median_relative_orientation_to_gyro_delay_ms"),
        "runtime_config.measured.dynamic_consistency.median_relative_orientation_to_gyro_delay_ms",
    )
    if dynamic.get("static_gravity_gain_measured") is not False:
        raise Cmp10aRuntimeConfigError("dynamic consistency gains must not be promoted as a static gravity gain.")
    axis_results = _mapping(dynamic.get("axis_results"), "runtime_config.measured.dynamic_consistency.axis_results")
    if set(axis_results) != set(_AXIS_NAMES):
        raise Cmp10aRuntimeConfigError("runtime dynamic consistency must retain X, Y, and Z axis results.")
    for axis_name in _AXIS_NAMES:
        axis = _mapping(
            axis_results[axis_name], f"runtime_config.measured.dynamic_consistency.axis_results.{axis_name}"
        )
        _finite(axis.get("relative_orientation_to_gyro_delay_ms"), f"runtime_config dynamic {axis_name} delay")
        _finite(
            axis.get("euler_derived_angular_velocity_to_gyro_gain_ratio"),
            f"runtime_config dynamic {axis_name} gain",
            positive=True,
        )

    fit = _mapping(measured.get("mount_fit_evidence"), "runtime_config.measured.mount_fit_evidence")
    _matrix(fit.get("sensor_to_base_fitted_matrix"), "runtime_config.measured.mount_fit_evidence matrix")
    if fit.get("applied_at_runtime") is not False:
        raise Cmp10aRuntimeConfigError("the full manual-trial fit must remain evidence-only.")

    assumed = _mapping(model.get("assumed_simulation_envelopes"), "runtime_config.assumed_simulation_envelopes")
    if assumed.get("evidence_status") != "assumed_not_measured":
        raise Cmp10aRuntimeConfigError("simulation envelopes must remain explicitly assumed, not measured.")
    _validate_exact_range(
        _mapping(assumed.get("residual_gyro_episode_bias"), "residual gyro bias").get("range_rad_s_per_axis"),
        CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S,
        "residual gyro episode bias range",
    )
    _validate_exact_range(
        _mapping(assumed.get("gyro_white_noise"), "gyro white noise").get("sigma_range_rad_s"),
        CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S,
        "gyro white sigma range",
    )
    _validate_exact_range(
        _mapping(assumed.get("projected_gravity_tangent_angle_noise"), "gravity tangent noise").get("sigma_range_rad"),
        CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD,
        "gravity tangent-angle sigma range",
    )
    _validate_exact_range(
        _mapping(assumed.get("gyro_sample_age_delay"), "gyro sample-age delay").get("range_s"),
        CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S,
        "gyro sample-age delay range",
    )
    _validate_exact_range(
        _mapping(assumed.get("orientation_delay"), "orientation delay").get("range_s"),
        CMP10A_ORIENTATION_DELAY_RANGE_S,
        "orientation delay range",
    )

    unmeasured = _mapping(model.get("unmeasured_quantities"), "runtime_config.unmeasured_quantities")
    if unmeasured.get("absolute_transport_latency_s") is not None or unmeasured.get("level_offset_rad") is not None:
        raise Cmp10aRuntimeConfigError("absolute transport latency and level offset must remain unmeasured nulls.")

    stress = _mapping(model.get("stress_only_provenance"), "runtime_config.stress_only_provenance")
    if stress.get("applied_to_runtime_model") is not False:
        raise Cmp10aRuntimeConfigError("legacy stress ranges must not be applied to the runtime model.")
    _validate_exact_range(
        stress.get("gyro_component_range_rad_s"),
        CMP10A_STRESS_ONLY_GYRO_RANGE_RAD_S,
        "stress-only gyro range",
    )
    _validate_exact_range(
        stress.get("projected_gravity_component_range"),
        CMP10A_STRESS_ONLY_GRAVITY_COMPONENT_RANGE,
        "stress-only projected-gravity range",
    )


def build_cmp10a_runtime_config_from_files(
    static_report_path: str | Path,
    dynamic_report_path: str | Path,
    dynamic_dataset_path: str | Path,
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Load, hash, validate, and promote a concrete set of CMP10A evidence files."""

    static_path = Path(static_report_path).expanduser().resolve()
    dynamic_path = Path(dynamic_report_path).expanduser().resolve()
    dataset_path = Path(dynamic_dataset_path).expanduser().resolve()
    try:
        dataset = load_imu_dataset(dataset_path)
    except (OSError, ValueError) as error:
        raise Cmp10aRuntimeConfigError(f"Unable to load dynamic_dataset {dataset_path}: {error}") from error
    return build_cmp10a_runtime_config(
        _load_json(static_path, "static_report"),
        _load_json(dynamic_path, "dynamic_report"),
        dataset,
        static_report_path=static_path,
        static_report_sha256=_sha256(static_path),
        dynamic_report_path=dynamic_path,
        dynamic_report_sha256=_sha256(dynamic_path),
        dynamic_dataset_path=dataset_path,
        dynamic_dataset_sha256=_sha256(dataset_path),
        created_utc=created_utc,
    )


__all__ = [
    "CMP10A_BASELINE_DISCARD_INITIAL_S",
    "CMP10A_GRAVITY_TANGENT_ANGLE_SIGMA_RANGE_RAD",
    "CMP10A_GYRO_SAMPLE_AGE_DELAY_RANGE_S",
    "CMP10A_GYRO_WHITE_SIGMA_RANGE_RAD_S",
    "CMP10A_ORIENTATION_DELAY_RANGE_S",
    "CMP10A_POLICY_ANGULAR_VELOCITY_SCALE",
    "CMP10A_POLICY_HZ",
    "CMP10A_RESIDUAL_GYRO_EPISODE_BIAS_RANGE_RAD_S",
    "CMP10A_RUNTIME_MAXIMUM_PAIR_SKEW_S",
    "CMP10A_RUNTIME_MAXIMUM_SAMPLE_AGE_S",
    "CMP10A_RUNTIME_MODEL_TYPE",
    "CMP10A_RUNTIME_SCHEMA_VERSION",
    "CMP10A_RUNTIME_SIGNED_MAPPING",
    "Cmp10aRuntimeConfigError",
    "build_cmp10a_runtime_config",
    "build_cmp10a_runtime_config_from_files",
    "validate_cmp10a_runtime_config",
]
