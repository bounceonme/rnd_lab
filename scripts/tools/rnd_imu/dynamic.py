"""Pure NumPy dynamic consistency analysis for CMP10A Euler and gyro data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DYNAMIC_AXIS_STAGES = ("dynamic_axis_x", "dynamic_axis_y", "dynamic_axis_z")

_GYRO_FRAME_TYPE = 0x52
_EULER_FRAME_TYPE = 0x53
_MINIMUM_SOURCE_SAMPLES = 8
_MINIMUM_ANALYSIS_SAMPLES = 20
_RMS_FLOOR = 1.0e-12


def _validate_analysis_options(
    max_lag_ms: float,
    minimum_correlation: float,
    minimum_axis_rms_rad_s: float,
    minimum_dominance_ratio: float,
) -> tuple[float, float, float, float]:
    options = {
        "max_lag_ms": float(max_lag_ms),
        "minimum_correlation": float(minimum_correlation),
        "minimum_axis_rms_rad_s": float(minimum_axis_rms_rad_s),
        "minimum_dominance_ratio": float(minimum_dominance_ratio),
    }
    if not all(np.isfinite(value) for value in options.values()):
        raise ValueError("Dynamic-analysis options must be finite.")
    if options["max_lag_ms"] < 0.0:
        raise ValueError("max_lag_ms must be non-negative.")
    if not 0.0 <= options["minimum_correlation"] <= 1.0:
        raise ValueError("minimum_correlation must be in [0, 1].")
    if options["minimum_axis_rms_rad_s"] < 0.0:
        raise ValueError("minimum_axis_rms_rad_s must be non-negative.")
    if options["minimum_dominance_ratio"] < 0.0:
        raise ValueError("minimum_dominance_ratio must be non-negative.")
    return (
        options["max_lag_ms"],
        options["minimum_correlation"],
        options["minimum_axis_rms_rad_s"],
        options["minimum_dominance_ratio"],
    )


def _stage_result(expected_axis: int, *, samples: int = 0, reason: str) -> dict[str, Any]:
    return {
        "expected_dominant_axis": expected_axis,
        "motion_amplitude_rad": None,
        "dominant_axis_rms_ratio": None,
        "correlation": None,
        "delay_ms": None,
        "gain_ratio": None,
        "samples": int(samples),
        "sample_period_ms": None,
        "expected_axis_rms_rad_s": None,
        "quality_pass": False,
        "reason": reason,
    }


def _dataset_arrays(data: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    required = ("timestamp_ns", "stage", "frame_type", "gyro_rad_s", "euler_rad")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {', '.join(missing)}.")

    timestamp_ns = np.asarray(data["timestamp_ns"])
    stages = np.asarray(data["stage"])
    frame_types = np.asarray(data["frame_type"])
    gyro = np.asarray(data["gyro_rad_s"], dtype=np.float64)
    euler = np.asarray(data["euler_rad"], dtype=np.float64)
    if timestamp_ns.ndim != 1 or stages.ndim != 1 or frame_types.ndim != 1:
        raise ValueError("timestamp_ns, stage, and frame_type must be one-dimensional arrays.")
    row_count = timestamp_ns.shape[0]
    if stages.shape[0] != row_count or frame_types.shape[0] != row_count:
        raise ValueError("Dataset arrays do not have a common row count.")
    if gyro.shape != (row_count, 3) or euler.shape != (row_count, 3):
        raise ValueError("gyro_rad_s and euler_rad must have shape [rows, 3].")
    return timestamp_ns, stages, frame_types, gyro, euler


def _ordered_series(
    timestamp_ns: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected_timestamps = np.asarray(timestamp_ns[mask])
    selected_values = np.asarray(values[mask], dtype=np.float64)
    if selected_timestamps.size < _MINIMUM_SOURCE_SAMPLES:
        raise ValueError(f"fewer than {_MINIMUM_SOURCE_SAMPLES} {name} samples")
    try:
        finite_timestamps = np.isfinite(selected_timestamps)
    except TypeError as error:
        raise ValueError(f"{name} timestamps are not numeric") from error
    if not np.all(finite_timestamps) or not np.all(np.isfinite(selected_values)):
        raise ValueError(f"{name} contains non-finite timestamps or values")

    timestamps = np.asarray(selected_timestamps, dtype=np.int64)
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    selected_values = selected_values[order]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} timestamps are not unique and strictly increasing")
    return timestamps, selected_values


def _body_angular_velocity_zyx(euler_rad: np.ndarray, euler_rate_rad_s: np.ndarray) -> np.ndarray:
    """Convert ZYX roll-pitch-yaw rates to body angular velocity."""

    roll = euler_rad[:, 0]
    pitch = euler_rad[:, 1]
    roll_rate = euler_rate_rad_s[:, 0]
    pitch_rate = euler_rate_rad_s[:, 1]
    yaw_rate = euler_rate_rad_s[:, 2]
    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    return np.column_stack((
        roll_rate - sin_pitch * yaw_rate,
        cos_roll * pitch_rate + sin_roll * cos_pitch * yaw_rate,
        -sin_roll * pitch_rate + cos_roll * cos_pitch * yaw_rate,
    ))


def _rms(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def _lagged_signals(
    timestamp_s: np.ndarray,
    gyro_axis: np.ndarray,
    euler_axis: np.ndarray,
    lag_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_timestamp_s = timestamp_s + lag_s
    valid = (shifted_timestamp_s >= timestamp_s[0]) & (shifted_timestamp_s <= timestamp_s[-1])
    measured = gyro_axis[valid]
    derived = np.interp(shifted_timestamp_s[valid], timestamp_s, euler_axis)
    return measured, derived


def _relative_delay(
    timestamp_s: np.ndarray,
    gyro_axis: np.ndarray,
    euler_axis: np.ndarray,
    sample_period_s: float,
    max_lag_ms: float,
) -> tuple[float, float, float, int] | None:
    max_lag_steps = int(np.floor((max_lag_ms * 1.0e-3) / sample_period_s + 1.0e-9))
    minimum_overlap = max(10, timestamp_s.size // 2)
    best: tuple[float, float, float, int] | None = None
    best_lag_steps = 0

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        lag_s = lag_steps * sample_period_s
        measured, derived = _lagged_signals(timestamp_s, gyro_axis, euler_axis, lag_s)
        if measured.size < minimum_overlap:
            continue
        measured_centered = measured - np.mean(measured)
        derived_centered = derived - np.mean(derived)
        measured_energy = float(np.dot(measured_centered, measured_centered))
        derived_energy = float(np.dot(derived_centered, derived_centered))
        if measured_energy <= _RMS_FLOOR or derived_energy <= _RMS_FLOOR:
            continue
        correlation = float(np.dot(measured_centered, derived_centered) / np.sqrt(measured_energy * derived_energy))
        measured_rms = float(_rms(measured_centered))
        derived_rms = float(_rms(derived_centered))
        gain_ratio = derived_rms / measured_rms
        candidate = (correlation, lag_s * 1.0e3, gain_ratio, int(measured.size))
        if (
            best is None
            or correlation > best[0] + 1.0e-12
            or (abs(correlation - best[0]) <= 1.0e-12 and abs(lag_steps) < abs(best_lag_steps))
        ):
            best = candidate
            best_lag_steps = lag_steps
    return best


def _analyze_stage(
    timestamp_ns: np.ndarray,
    stages: np.ndarray,
    frame_types: np.ndarray,
    gyro_values: np.ndarray,
    euler_values: np.ndarray,
    stage: str,
    expected_axis: int,
    *,
    max_lag_ms: float,
    minimum_correlation: float,
    minimum_axis_rms_rad_s: float,
    minimum_dominance_ratio: float,
) -> dict[str, Any]:
    stage_mask = stages == stage
    gyro_mask = stage_mask & (frame_types == _GYRO_FRAME_TYPE)
    euler_mask = stage_mask & (frame_types == _EULER_FRAME_TYPE)
    selected_samples = int(min(np.count_nonzero(gyro_mask), np.count_nonzero(euler_mask)))
    try:
        gyro_timestamp_ns, gyro = _ordered_series(timestamp_ns, gyro_values, gyro_mask, "gyro")
        euler_timestamp_ns, euler = _ordered_series(timestamp_ns, euler_values, euler_mask, "Euler")
    except (TypeError, ValueError) as error:
        return _stage_result(expected_axis, samples=selected_samples, reason=str(error))

    origin_ns = min(int(gyro_timestamp_ns[0]), int(euler_timestamp_ns[0]))
    gyro_timestamp_s = (gyro_timestamp_ns - origin_ns).astype(np.float64) * 1.0e-9
    euler_timestamp_s = (euler_timestamp_ns - origin_ns).astype(np.float64) * 1.0e-9
    in_euler_range = (gyro_timestamp_s >= euler_timestamp_s[0]) & (gyro_timestamp_s <= euler_timestamp_s[-1])
    gyro_timestamp_s = gyro_timestamp_s[in_euler_range]
    gyro = gyro[in_euler_range]
    if gyro_timestamp_s.size < _MINIMUM_ANALYSIS_SAMPLES + 2:
        return _stage_result(
            expected_axis,
            samples=int(gyro_timestamp_s.size),
            reason=f"fewer than {_MINIMUM_ANALYSIS_SAMPLES} overlapping analysis samples",
        )

    # Unwrap before interpolation so a +/-pi transition is not interpolated through zero.
    unwrapped_euler = np.unwrap(euler, axis=0)
    euler_at_gyro = np.column_stack([
        np.interp(gyro_timestamp_s, euler_timestamp_s, unwrapped_euler[:, axis]) for axis in range(3)
    ])
    try:
        euler_rate = np.gradient(euler_at_gyro, gyro_timestamp_s, axis=0, edge_order=2)
    except (FloatingPointError, ValueError) as error:
        return _stage_result(expected_axis, samples=int(gyro_timestamp_s.size), reason=f"Euler rate failed: {error}")
    euler_body_omega = _body_angular_velocity_zyx(euler_at_gyro, euler_rate)
    if not np.all(np.isfinite(euler_body_omega)):
        return _stage_result(
            expected_axis,
            samples=int(gyro_timestamp_s.size),
            reason="Euler-derived body angular velocity is non-finite",
        )

    # Numerical derivatives are least reliable at the interpolation boundaries.
    analysis_timestamp_s = gyro_timestamp_s[1:-1]
    analysis_gyro = gyro[1:-1]
    analysis_euler = euler_at_gyro[1:-1]
    analysis_body_omega = euler_body_omega[1:-1]
    if analysis_timestamp_s.size < _MINIMUM_ANALYSIS_SAMPLES:
        return _stage_result(
            expected_axis,
            samples=int(analysis_timestamp_s.size),
            reason=f"fewer than {_MINIMUM_ANALYSIS_SAMPLES} analysis samples",
        )

    periods_s = np.diff(analysis_timestamp_s)
    if np.any(periods_s <= 0.0) or not np.all(np.isfinite(periods_s)):
        return _stage_result(
            expected_axis,
            samples=int(analysis_timestamp_s.size),
            reason="gyro timestamps do not define a finite positive sample period",
        )
    sample_period_s = float(np.median(periods_s))
    if sample_period_s <= 0.0:
        return _stage_result(
            expected_axis,
            samples=int(analysis_timestamp_s.size),
            reason="gyro median sample period is not positive",
        )

    centered_gyro = analysis_gyro - np.mean(analysis_gyro, axis=0)
    gyro_rms = np.asarray(_rms(centered_gyro, axis=0), dtype=np.float64)
    expected_axis_rms = float(gyro_rms[expected_axis])
    cross_axis_rms = float(np.max(np.delete(gyro_rms, expected_axis)))
    dominance_ratio = expected_axis_rms / max(cross_axis_rms, _RMS_FLOOR)
    motion_amplitude = 0.5 * float(np.ptp(analysis_euler[:, expected_axis]))

    delay_result = _relative_delay(
        analysis_timestamp_s,
        analysis_gyro[:, expected_axis],
        analysis_body_omega[:, expected_axis],
        sample_period_s,
        max_lag_ms,
    )
    correlation = delay_ms = gain_ratio = None
    if delay_result is not None:
        correlation, delay_ms, gain_ratio, _ = delay_result

    reasons = []
    if expected_axis_rms < minimum_axis_rms_rad_s:
        reasons.append("expected-axis RMS is below the motion threshold")
    if dominance_ratio < minimum_dominance_ratio:
        reasons.append("expected axis is not sufficiently dominant")
    if correlation is None:
        reasons.append("normalized cross-correlation is undefined")
    elif correlation < minimum_correlation:
        reasons.append("normalized cross-correlation is below the quality threshold")
    quality_pass = not reasons
    return {
        "expected_dominant_axis": expected_axis,
        "motion_amplitude_rad": float(motion_amplitude),
        "dominant_axis_rms_ratio": float(dominance_ratio),
        "correlation": float(correlation) if correlation is not None else None,
        "delay_ms": float(delay_ms) if delay_ms is not None else None,
        "gain_ratio": float(gain_ratio) if gain_ratio is not None else None,
        "samples": int(analysis_timestamp_s.size),
        "sample_period_ms": float(sample_period_s * 1.0e3),
        "expected_axis_rms_rad_s": expected_axis_rms,
        "quality_pass": quality_pass,
        "reason": "; ".join(reasons) if reasons else None,
    }


def analyze_dynamic_imu_dataset(
    data: Mapping[str, Any],
    *,
    max_lag_ms: float = 150.0,
    minimum_correlation: float = 0.8,
    minimum_axis_rms_rad_s: float = 0.05,
    minimum_dominance_ratio: float = 2.0,
) -> dict[str, Any]:
    """Compare CMP10A Euler dynamics with gyro dynamics for three sensor axes.

    Positive ``delay_ms`` means the Euler-derived body angular velocity lags
    the gyro stream. The result measures only relative timing between two
    CMP10A outputs and cannot identify absolute USB transport latency.
    """

    max_lag_ms, minimum_correlation, minimum_axis_rms_rad_s, minimum_dominance_ratio = _validate_analysis_options(
        max_lag_ms,
        minimum_correlation,
        minimum_axis_rms_rad_s,
        minimum_dominance_ratio,
    )
    try:
        timestamp_ns, stages, frame_types, gyro, euler = _dataset_arrays(data)
        schema_error = None
    except (KeyError, TypeError, ValueError) as error:
        timestamp_ns = stages = frame_types = gyro = euler = None
        schema_error = str(error)

    stage_reports: dict[str, dict[str, Any]] = {}
    for expected_axis, stage in enumerate(DYNAMIC_AXIS_STAGES):
        if schema_error is not None:
            stage_reports[stage] = _stage_result(expected_axis, reason=schema_error)
            continue
        stage_reports[stage] = _analyze_stage(
            timestamp_ns,
            stages,
            frame_types,
            gyro,
            euler,
            stage,
            expected_axis,
            max_lag_ms=max_lag_ms,
            minimum_correlation=minimum_correlation,
            minimum_axis_rms_rad_s=minimum_axis_rms_rad_s,
            minimum_dominance_ratio=minimum_dominance_ratio,
        )

    passing_stages = [stage for stage in DYNAMIC_AXIS_STAGES if stage_reports[stage]["quality_pass"]]
    passing_delays = [stage_reports[stage]["delay_ms"] for stage in passing_stages]
    median_delay_ms = float(np.median(passing_delays)) if passing_delays else None
    return {
        "schema_version": 1,
        "analysis": "CMP10A Euler-vs-gyro dynamic consistency",
        "delay_definition": (
            "Positive delay_ms means Euler-derived ZYX body angular velocity lags gyro angular velocity."
        ),
        "absolute_usb_latency": False,
        "latency_note": (
            "This is relative timing between CMP10A orientation and gyro outputs, not absolute USB latency."
        ),
        "quality_thresholds": {
            "max_lag_ms": max_lag_ms,
            "minimum_correlation": minimum_correlation,
            "minimum_axis_rms_rad_s": minimum_axis_rms_rad_s,
            "minimum_dominance_ratio": minimum_dominance_ratio,
        },
        "stages": stage_reports,
        "passing_stages": passing_stages,
        "passing_stage_count": len(passing_stages),
        "median_relative_delay_ms": median_delay_ms,
        "quality_pass": len(passing_stages) == len(DYNAMIC_AXIS_STAGES),
    }
