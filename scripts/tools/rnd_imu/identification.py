"""Dataset persistence and analysis for RND STEP CMP10A identification."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import ImuIdentificationConfig


FRAME_NAMES = {
    0x50: "time",
    0x51: "acceleration",
    0x52: "angular_velocity",
    0x53: "euler_angle",
    0x54: "magnetic_field",
    0x55: "port_status",
    0x56: "barometer",
    0x57: "gps_position",
    0x58: "gps_velocity",
    0x59: "quaternion",
    0x5A: "gps_accuracy",
    0x5F: "register_read",
}

VECTOR_FIELDS = ("accel_mps2", "gyro_rad_s", "euler_rad", "mag_raw")
QUATERNION_FIELD = "quat_wxyz"
AXIS_STAGES = ("axis_pos_x", "axis_pos_y", "axis_pos_z")


def _vector(record: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    value = record.get(name)
    if value is None:
        return np.full(size, np.nan, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    return array


def save_imu_dataset(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Persist decoded frame events without pickle-backed arrays."""

    path = Path(path).expanduser().resolve()
    rows = list(records)
    if not rows:
        raise ValueError("Cannot save an empty IMU dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_frames = np.stack([
        np.frombuffer(row["raw"], dtype=np.uint8).copy()
        if isinstance(row["raw"], (bytes, bytearray))
        else np.asarray(row["raw"], dtype=np.uint8)
        for row in rows
    ])
    if raw_frames.shape[1:] != (11,):
        raise ValueError(f"Every raw CMP10A frame must contain 11 bytes, got {raw_frames.shape}.")

    arrays: dict[str, np.ndarray] = {
        "timestamp_ns": np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64),
        "stage": np.asarray([row["stage"] for row in rows], dtype="U32"),
        "frame_type": np.asarray([row["frame_type"] for row in rows], dtype=np.uint8),
        "raw_frame": raw_frames,
        "temperature_c": np.asarray([row.get("temperature_c", np.nan) for row in rows], dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    for name in VECTOR_FIELDS:
        arrays[name] = np.stack([_vector(row, name, 3) for row in rows])
    arrays[QUATERNION_FIELD] = np.stack([_vector(row, QUATERNION_FIELD, 4) for row in rows])
    np.savez_compressed(path, **arrays)
    return path


def load_imu_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset written by :func:`save_imu_dataset`."""

    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, Any] = {name: data[name].copy() for name in data.files if name != "metadata_json"}
        result["metadata"] = json.loads(str(data["metadata_json"].item()))
    lengths = {value.shape[0] for key, value in result.items() if key != "metadata" and value.ndim > 0}
    if len(lengths) != 1:
        raise ValueError(f"Dataset arrays do not have a common row count: {sorted(lengths)}")
    return result


def _stage_type_mask(data: Mapping[str, Any], stage: str, frame_type: int) -> np.ndarray:
    return (data["stage"] == stage) & (data["frame_type"] == frame_type)


def _rate_summary(timestamp_ns: np.ndarray) -> dict[str, float | int | None]:
    timestamps = np.asarray(timestamp_ns, dtype=np.int64)
    if timestamps.size < 2:
        return {"samples": int(timestamps.size), "rate_hz": None, "median_period_ms": None, "p95_period_ms": None}
    periods_s = np.diff(timestamps.astype(np.float64)) * 1.0e-9
    positive = periods_s[periods_s > 0.0]
    elapsed_s = (timestamps[-1] - timestamps[0]) * 1.0e-9
    rate_hz = (timestamps.size - 1) / elapsed_s if elapsed_s > 0.0 else None
    return {
        "samples": int(timestamps.size),
        "rate_hz": float(rate_hz) if rate_hz is not None else None,
        "median_period_ms": float(np.median(positive) * 1.0e3) if positive.size else None,
        "p95_period_ms": float(np.quantile(positive, 0.95) * 1.0e3) if positive.size else None,
    }


def _finite_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.all(np.isfinite(values), axis=1)]


def _axis_mapping(
    data: Mapping[str, Any],
    gyro_bias: np.ndarray,
    config: ImuIdentificationConfig,
) -> tuple[dict[str, Any], np.ndarray | None]:
    sensor_directions: list[np.ndarray] = []
    trials: dict[str, Any] = {}

    for base_axis, stage in enumerate(AXIS_STAGES):
        mask = _stage_type_mask(data, stage, 0x52)
        timestamps = np.asarray(data["timestamp_ns"][mask], dtype=np.int64)
        gyro = _finite_rows(data["gyro_rad_s"][mask])
        if gyro.shape[0] != timestamps.shape[0] or gyro.shape[0] < 3:
            trials[stage] = {"quality_pass": False, "reason": "fewer than three gyro samples"}
            continue
        order = np.argsort(timestamps)
        t_s = (timestamps[order] - timestamps[order][0]).astype(np.float64) * 1.0e-9
        corrected = gyro[order] - gyro_bias
        integrated = np.trapz(corrected, t_s, axis=0)
        magnitudes = np.abs(integrated)
        sensor_axis = int(np.argmax(magnitudes))
        sorted_magnitudes = np.sort(magnitudes)
        second = float(sorted_magnitudes[-2])
        dominance = float(magnitudes[sensor_axis] / max(second, 1.0e-12))
        rotation = float(np.linalg.norm(integrated))
        sign = 1.0 if integrated[sensor_axis] >= 0.0 else -1.0
        sensor_directions.append(integrated / max(rotation, 1.0e-12))
        quality_pass = rotation >= config.experiment.minimum_axis_rotation_rad
        trials[stage] = {
            "base_axis_index": base_axis,
            "integrated_sensor_rotation_rad": integrated.tolist(),
            "dominant_sensor_axis": sensor_axis,
            "dominant_sign": int(sign),
            "dominance_ratio": dominance,
            "total_rotation_rad": rotation,
            "near_axis_aligned": dominance >= config.experiment.minimum_axis_dominance_ratio,
            "quality_pass": quality_pass,
        }

    if len(sensor_directions) != 3:
        return {"quality_pass": False, "trials": trials, "reason": "axis rotation data is incomplete"}, None

    # Columns are sensor-frame directions measured during positive base X/Y/Z
    # rotations. Solve min ||R_BS * S - I|| with a proper-rotation Kabsch fit.
    source = np.stack(sensor_directions, axis=1)
    covariance = np.eye(3) @ source.T
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    matrix = left @ correction @ right_t
    fitted = matrix @ source
    fit_angles_deg = np.degrees(np.arccos(np.clip(np.sum(fitted * np.eye(3), axis=0), -1.0, 1.0)))
    condition_number = float(np.linalg.cond(source))
    determinant = float(np.linalg.det(matrix))
    signed_axis_approximation = np.zeros((3, 3), dtype=np.float64)
    for base_axis in range(3):
        sensor_axis = int(np.argmax(np.abs(matrix[base_axis])))
        signed_axis_approximation[base_axis, sensor_axis] = np.sign(matrix[base_axis, sensor_axis])
    trial_pass = all(bool(trials[stage]["quality_pass"]) for stage in AXIS_STAGES)
    quality_pass = bool(
        trial_pass
        and np.max(fit_angles_deg) <= config.experiment.maximum_axis_fit_error_deg
        and condition_number <= 3.0
    )
    reason = None if quality_pass else "rotation amplitude, axis independence, or 3D rotation-fit quality failed"
    return {
        "quality_pass": quality_pass,
        "trials": trials,
        "sensor_to_base_matrix": matrix.tolist(),
        "signed_axis_approximation": signed_axis_approximation.tolist(),
        "determinant": determinant,
        "source_direction_condition_number": condition_number,
        "axis_fit_error_deg": fit_angles_deg.tolist(),
        "maximum_axis_fit_error_deg": config.experiment.maximum_axis_fit_error_deg,
        "reason": reason,
    }, matrix


def identify_imu_dataset(data: Mapping[str, Any], config: ImuIdentificationConfig) -> dict[str, Any]:
    """Identify timing, static noise, and the signed sensor-to-base axis mapping."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "policy_hz": config.experiment.policy_hz,
        "base_frame_axes": {"x": "robot_left", "y": "robot_backward", "z": "up"},
    }
    metadata = dict(data.get("metadata", {}))
    report["mount_context"] = {
        "location": metadata.get("mount_location", "unknown"),
        "rigid_link": "Upper_Body" if metadata.get("upper_body_mount") else "unknown",
        "base_link_to_mount_link_joint": metadata.get("base_link_to_upper_body_joint", "unknown"),
        "translation_m": metadata.get("mount_translation_m"),
        "translation_used_for_this_identification": False,
        "translation_note": (
            "A rigid mount translation does not change angular velocity or projected gravity. "
            "Measure it before using linear acceleration because lever-arm acceleration then matters."
        ),
    }
    static_rates: dict[str, Any] = {}
    for frame_type, frame_name in FRAME_NAMES.items():
        mask = _stage_type_mask(data, "static_upright", frame_type)
        if np.any(mask):
            static_rates[frame_name] = _rate_summary(data["timestamp_ns"][mask])
    report["static_packet_rates"] = static_rates

    gyro_mask = _stage_type_mask(data, "static_upright", 0x52)
    static_gyro = _finite_rows(data["gyro_rad_s"][gyro_mask])
    if static_gyro.size:
        gyro_bias = np.mean(static_gyro, axis=0)
        gyro_std = np.std(static_gyro, axis=0, ddof=1) if static_gyro.shape[0] > 1 else np.zeros(3)
    else:
        gyro_bias = np.full(3, np.nan)
        gyro_std = np.full(3, np.nan)
    report["static_gyro"] = {
        "samples": int(static_gyro.shape[0]),
        "bias_rad_s": gyro_bias.tolist(),
        "std_rad_s": gyro_std.tolist(),
        "quality_pass": bool(
            np.all(np.isfinite(gyro_std)) and np.max(gyro_std) <= config.quality.maximum_static_gyro_std_rad_s
        ),
    }

    accel_mask = _stage_type_mask(data, "static_upright", 0x51)
    static_accel = _finite_rows(data["accel_mps2"][accel_mask])
    if static_accel.size:
        accel_mean = np.mean(static_accel, axis=0)
        accel_std = np.std(static_accel, axis=0, ddof=1) if static_accel.shape[0] > 1 else np.zeros(3)
        accel_norm = np.linalg.norm(static_accel, axis=1)
        norm_mean = float(np.mean(accel_norm))
        norm_std = float(np.std(accel_norm, ddof=1)) if accel_norm.size > 1 else 0.0
    else:
        accel_mean = np.full(3, np.nan)
        accel_std = np.full(3, np.nan)
        norm_mean = float("nan")
        norm_std = float("nan")
    report["static_accelerometer"] = {
        "samples": int(static_accel.shape[0]),
        "mean_mps2": accel_mean.tolist(),
        "std_mps2": accel_std.tolist(),
        "norm_mean_mps2": norm_mean,
        "norm_std_mps2": norm_std,
        "quality_pass": bool(
            np.isfinite(norm_mean)
            and abs(norm_mean - config.experiment.gravity_mps2) <= config.quality.maximum_static_accel_norm_error_mps2
        ),
    }

    mapping_report, sensor_to_base = _axis_mapping(data, gyro_bias, config)
    report["mount_axis_identification"] = mapping_report
    if sensor_to_base is not None and np.all(np.isfinite(accel_mean)) and np.linalg.norm(accel_mean) > 0.0:
        projected_gravity_b = sensor_to_base @ (-accel_mean / np.linalg.norm(accel_mean))
        report["static_projected_gravity_b_from_accel"] = projected_gravity_b.tolist()
        report["static_projected_gravity_error"] = float(
            np.linalg.norm(projected_gravity_b - np.array([0.0, 0.0, -1.0]))
        )

    gyro_rate = static_rates.get("angular_velocity", {}).get("rate_hz")
    quaternion_rate = static_rates.get("quaternion", {}).get("rate_hz")
    euler_rate = static_rates.get("euler_angle", {}).get("rate_hz")
    minimum_rate = config.quality.minimum_required_rate_hz
    if quaternion_rate is not None and quaternion_rate >= minimum_rate:
        orientation_source = "quaternion"
        orientation_rate = quaternion_rate
    elif euler_rate is not None and euler_rate >= minimum_rate:
        orientation_source = "euler_angle"
        orientation_rate = euler_rate
    else:
        orientation_source = None
        orientation_rate = (
            max(value for value in (quaternion_rate, euler_rate) if value is not None)
            if any(value is not None for value in (quaternion_rate, euler_rate))
            else None
        )

    parser_stats = dict(metadata.get("parser_stats", {}))
    valid_frames = int(parser_stats.get("valid_frames", data["frame_type"].shape[0]))
    checksum_errors = int(parser_stats.get("checksum_failures", parser_stats.get("checksum_errors", 0)))
    checksum_fraction = checksum_errors / max(valid_frames + checksum_errors, 1)
    runtime_pass = bool(
        gyro_rate is not None
        and gyro_rate >= minimum_rate
        and orientation_source is not None
        and mapping_report["quality_pass"]
        and checksum_fraction <= config.quality.maximum_checksum_error_fraction
    )
    report["runtime_gate"] = {
        "quality_pass": runtime_pass,
        "gyro_rate_hz": gyro_rate,
        "orientation_source": orientation_source,
        "orientation_rate_hz": orientation_rate,
        "required_rate_hz": minimum_rate,
        "checksum_error_fraction": checksum_fraction,
        "notes": [
            "The policy consumes 0.25 * base angular velocity in rad/s.",
            "Projected gravity is a unit vector in base_link, not raw acceleration in m/s^2.",
            "This gate does not identify absolute transport latency; a synchronized reference is required.",
        ],
    }
    return report


def write_identification_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write a JSON report next to the raw NPZ data."""

    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(report)), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
