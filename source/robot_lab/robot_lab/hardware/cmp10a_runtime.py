"""Read-only CMP10A latest-frame adapter for 50 Hz policy observations."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if __package__:
    from .cmp10a import (
        CMP10A_FRAME_SIZE,
        CMP10AFrame,
        CMP10AParser,
        CMP10AParserCounters,
        CMP10AProtocolError,
        CMP10ASerialReader,
        decode_frame,
    )
else:
    from cmp10a import (  # type: ignore[no-redef]
        CMP10A_FRAME_SIZE,
        CMP10AFrame,
        CMP10AParser,
        CMP10AParserCounters,
        CMP10AProtocolError,
        CMP10ASerialReader,
        decode_frame,
    )


CMP10A_RUNTIME_SCHEMA_VERSION = 1
CMP10A_RUNTIME_MODEL_TYPE = "rnd_cmp10a_policy_observation"

_GYRO_FRAME_TYPE = 0x52
_EULER_FRAME_TYPE = 0x53
_ROTATION_TOLERANCE = 1.0e-5
_REQUIRED_QUALITY_GATES = (
    "promotion_pass",
    "static_runtime_gate_pass",
    "static_mount_axis_gate_pass",
    "dynamic_communication_gate_pass",
    "dynamic_consistency_gate_pass",
)

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
RuntimeModelSource = Mapping[str, Any] | str | Path


class CMP10ARuntimeModelError(ValueError):
    """Raised when a CMP10A policy-observation model is missing or unsafe."""


class CMP10ARuntimeFrameError(ValueError):
    """Raised when the runtime adapter receives a frame that is not protocol-valid."""


class CMP10ARuntimeSnapshotError(RuntimeError):
    """Raised when no coherent, fresh gyro/Euler pair is available."""


class CMP10ARuntimeSourceError(RuntimeError):
    """Raised when the background read-only source cannot provide data."""


@dataclass(frozen=True)
class CMP10ARuntimeConfig:
    """Validated values used by the runtime adapter."""

    sensor_to_base_matrix: Matrix3
    gyro_bias_sensor_rad_s: Vector3
    policy_angular_velocity_scale: float
    sensor_baudrate: int
    sensor_gyro_rate_hz: float
    sensor_euler_rate_hz: float
    policy_rate_hz: float
    max_frame_age_ns: int
    max_pair_skew_ns: int

    @property
    def gyro_bias_rad_s(self) -> Vector3:
        return self.gyro_bias_sensor_rad_s

    @property
    def policy_hz(self) -> float:
        return self.policy_rate_hz

    @property
    def max_frame_age_s(self) -> float:
        return self.max_frame_age_ns / 1.0e9

    @property
    def max_pair_skew_s(self) -> float:
        return self.max_pair_skew_ns / 1.0e9


@dataclass(frozen=True)
class CMP10ARuntimeCounters:
    """Cumulative checksum-valid frames seen by a latest-frame adapter."""

    ingested_frames: int
    gyro_frames: int
    euler_frames: int
    ignored_frames: int
    out_of_order_frames: int


@dataclass(frozen=True)
class CMP10ARuntimeSample:
    """One immutable policy-ready observation sampled from the latest frame pair."""

    base_angular_velocity_rad_s: Vector3
    policy_angular_velocity: Vector3
    projected_gravity_b: Vector3
    gyro_timestamp_ns: int
    euler_timestamp_ns: int
    gyro_age_ns: int
    euler_age_ns: int
    pair_skew_ns: int
    counters: CMP10ARuntimeCounters
    parser_counters: CMP10AParserCounters | None = None

    @property
    def raw_base_angular_velocity_rad_s(self) -> Vector3:
        """Return bias-corrected base angular velocity before policy scaling."""

        return self.base_angular_velocity_rad_s

    @property
    def angular_velocity_b_rad_s(self) -> Vector3:
        return self.base_angular_velocity_rad_s

    @property
    def policy_scaled_angular_velocity(self) -> Vector3:
        return self.policy_angular_velocity

    @property
    def gyro_age_s(self) -> float:
        return self.gyro_age_ns / 1.0e9

    @property
    def euler_age_s(self) -> float:
        return self.euler_age_ns / 1.0e9

    @property
    def pair_skew_s(self) -> float:
        return self.pair_skew_ns / 1.0e9


def _nested_value(model: Mapping[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    value: Any = model
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _required_value(model: Mapping[str, Any], label: str, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        found, value = _nested_value(model, path)
        if found:
            return value
    expected = ", ".join(".".join(path) for path in paths)
    raise CMP10ARuntimeModelError(f"CMP10A runtime model requires {label}; expected key {expected}.")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CMP10ARuntimeModelError(f"{label} must be a JSON number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise CMP10ARuntimeModelError(f"{label} must be finite, got {value!r}.")
    if positive and result <= 0.0:
        raise CMP10ARuntimeModelError(f"{label} must be positive, got {result}.")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CMP10ARuntimeModelError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _vector3(value: Any, label: str) -> Vector3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise CMP10ARuntimeModelError(f"{label} must be a three-element JSON array.")
    return (
        _number(value[0], f"{label}[0]"),
        _number(value[1], f"{label}[1]"),
        _number(value[2], f"{label}[2]"),
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _determinant(matrix: Matrix3) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _rotation_matrix(value: Any, label: str) -> Matrix3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise CMP10ARuntimeModelError(f"{label} must be a 3x3 JSON array.")
    matrix: Matrix3 = (
        _vector3(value[0], f"{label}[0]"),
        _vector3(value[1], f"{label}[1]"),
        _vector3(value[2], f"{label}[2]"),
    )
    for row_index, row in enumerate(matrix):
        for other_index, other in enumerate(matrix):
            target = 1.0 if row_index == other_index else 0.0
            if not math.isclose(_dot(row, other), target, rel_tol=0.0, abs_tol=_ROTATION_TOLERANCE):
                raise CMP10ARuntimeModelError(
                    f"{label} must be orthonormal; row {row_index} dot row {other_index} is {_dot(row, other):.9g}."
                )
    determinant = _determinant(matrix)
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=_ROTATION_TOLERANCE):
        raise CMP10ARuntimeModelError(f"{label} must be a proper rotation with determinant +1, got {determinant:.9g}.")
    return matrix


def _duration_ns(
    model: Mapping[str, Any],
    label: str,
    *,
    millisecond_paths: tuple[tuple[str, ...], ...],
    second_paths: tuple[tuple[str, ...], ...],
    nanosecond_paths: tuple[tuple[str, ...], ...],
) -> int:
    for paths, multiplier, unit in (
        (millisecond_paths, 1.0e6, "ms"),
        (second_paths, 1.0e9, "s"),
        (nanosecond_paths, 1.0, "ns"),
    ):
        for path in paths:
            found, value = _nested_value(model, path)
            if found:
                duration = _number(value, ".".join(path), positive=True)
                duration_ns = duration * multiplier
                if not math.isfinite(duration_ns) or duration_ns > 9.22e18:
                    raise CMP10ARuntimeModelError(f"{label} in {unit} is too large for nanosecond timestamps.")
                rounded_ns = int(round(duration_ns))
                if rounded_ns <= 0:
                    raise CMP10ARuntimeModelError(f"{label} must be at least one nanosecond.")
                return rounded_ns
    expected = ", ".join(".".join(path) for path in millisecond_paths)
    raise CMP10ARuntimeModelError(f"CMP10A runtime model requires {label}; expected key {expected}.")


def _validate_quality_gates(model: Mapping[str, Any]) -> None:
    gates = model.get("quality_gates")
    if not isinstance(gates, Mapping):
        raise CMP10ARuntimeModelError("CMP10A runtime model requires a quality_gates object.")
    failed = [gate for gate in _REQUIRED_QUALITY_GATES if gates.get(gate) is not True]
    if failed:
        raise CMP10ARuntimeModelError(
            "CMP10A runtime model cannot be consumed because these quality gates are not true: "
            + ", ".join(failed)
            + "."
        )


def _validate_duplicate_number(
    model: Mapping[str, Any], paths: tuple[tuple[str, ...], ...], expected: float, label: str
) -> None:
    for path in paths:
        found, value = _nested_value(model, path)
        if found and not math.isclose(_number(value, ".".join(path)), expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise CMP10ARuntimeModelError(f"Duplicated {label} fields disagree at {'.'.join(path)}.")


def _validate_duplicate_vector(
    model: Mapping[str, Any], paths: tuple[tuple[str, ...], ...], expected: Vector3, label: str
) -> None:
    for path in paths:
        found, value = _nested_value(model, path)
        if found:
            candidate = _vector3(value, ".".join(path))
            if any(
                not math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, target in zip(candidate, expected)
            ):
                raise CMP10ARuntimeModelError(f"Duplicated {label} fields disagree at {'.'.join(path)}.")


def _validate_duplicate_matrix(
    model: Mapping[str, Any], paths: tuple[tuple[str, ...], ...], expected: Matrix3, label: str
) -> None:
    for path in paths:
        found, value = _nested_value(model, path)
        if found:
            candidate = _rotation_matrix(value, ".".join(path))
            if any(
                not math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12)
                for candidate_row, expected_row in zip(candidate, expected)
                for actual, target in zip(candidate_row, expected_row)
            ):
                raise CMP10ARuntimeModelError(f"Duplicated {label} fields disagree at {'.'.join(path)}.")


def _validate_duplicate_duration(
    model: Mapping[str, Any], paths: tuple[tuple[tuple[str, ...], float], ...], expected_ns: int, label: str
) -> None:
    for path, multiplier in paths:
        found, value = _nested_value(model, path)
        if found and int(round(_number(value, ".".join(path), positive=True) * multiplier)) != expected_ns:
            raise CMP10ARuntimeModelError(f"Duplicated {label} fields disagree at {'.'.join(path)}.")


def _runtime_config(model: Mapping[str, Any]) -> CMP10ARuntimeConfig:
    if not isinstance(model, Mapping):
        raise CMP10ARuntimeModelError("CMP10A runtime model must be a mapping/JSON object.")
    schema_version = model.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        raise CMP10ARuntimeModelError(
            f"Unsupported schema_version {schema_version!r}; expected {CMP10A_RUNTIME_SCHEMA_VERSION}."
        )
    if model.get("model_type") != CMP10A_RUNTIME_MODEL_TYPE:
        raise CMP10ARuntimeModelError(
            f"Unsupported model_type {model.get('model_type')!r}; expected {CMP10A_RUNTIME_MODEL_TYPE!r}."
        )
    if model.get("integration_enabled") is not True:
        raise CMP10ARuntimeModelError("CMP10A runtime model must explicitly set integration_enabled=true.")
    _validate_quality_gates(model)

    matrix = _rotation_matrix(
        _required_value(
            model,
            "sensor_to_base_matrix",
            (
                ("sensor_to_base_matrix",),
                ("runtime_transform", "sensor_to_base_matrix"),
                ("sensor", "sensor_to_base_matrix"),
            ),
        ),
        "sensor_to_base_matrix",
    )
    bias = _vector3(
        _required_value(
            model,
            "gyro bias",
            (
                ("gyro_bias_rad_s",),
                ("gyro_bias_sensor_rad_s",),
                ("measured", "held_baseline", "gyro", "mean_rad_s"),
                ("sensor", "gyro_bias_rad_s"),
                ("sensor", "gyro_bias_sensor_rad_s"),
            ),
        ),
        "gyro_bias_rad_s",
    )
    scale = _number(
        _required_value(
            model,
            "policy angular-velocity scale",
            (
                ("policy_angular_velocity_scale",),
                ("angular_velocity_scale",),
                ("policy_observation", "angular_velocity_scale"),
                ("policy", "angular_velocity_scale"),
                ("observation", "angular_velocity_scale"),
            ),
        ),
        "policy_angular_velocity_scale",
        positive=True,
    )
    baudrate = _positive_integer(
        _required_value(
            model,
            "sensor baudrate",
            (
                ("sensor_baudrate",),
                ("baudrate",),
                ("sensor", "baudrate"),
                ("serial", "baudrate"),
                ("communication", "baudrate"),
                ("runtime_serial", "baudrate"),
            ),
        ),
        "sensor_baudrate",
    )
    policy_rate_hz = _number(
        _required_value(
            model,
            "policy rate",
            (
                ("policy_hz",),
                ("policy_observation", "policy_hz"),
                ("policy_rate_hz",),
                ("policy", "rate_hz"),
                ("rates_hz", "policy"),
            ),
        ),
        "policy_hz",
        positive=True,
    )
    gyro_rate_hz = _number(
        _required_value(
            model,
            "sensor gyro rate",
            (
                ("sensor_gyro_rate_hz",),
                ("gyro_rate_hz",),
                ("sensor_rates_hz", "gyro"),
                ("sensor_rates_hz", "angular_velocity"),
                ("measured", "packet_rates", "gyro_rate_hz"),
                ("sensor", "gyro_rate_hz"),
                ("sensor", "rates_hz", "gyro"),
                ("rates_hz", "gyro"),
            ),
        ),
        "sensor_gyro_rate_hz",
        positive=True,
    )
    euler_rate_hz = _number(
        _required_value(
            model,
            "sensor Euler rate",
            (
                ("sensor_euler_rate_hz",),
                ("euler_rate_hz",),
                ("orientation_rate_hz",),
                ("sensor_rates_hz", "euler"),
                ("sensor_rates_hz", "orientation"),
                ("measured", "packet_rates", "orientation_rate_hz"),
                ("sensor", "euler_rate_hz"),
                ("sensor", "orientation_rate_hz"),
                ("sensor", "rates_hz", "euler"),
                ("rates_hz", "euler"),
            ),
        ),
        "sensor_euler_rate_hz",
        positive=True,
    )
    if gyro_rate_hz < policy_rate_hz or euler_rate_hz < policy_rate_hz:
        raise CMP10ARuntimeModelError(
            "Sensor gyro and Euler rates must each be at least policy_hz; "
            f"got gyro={gyro_rate_hz}, euler={euler_rate_hz}, policy={policy_rate_hz}."
        )

    max_frame_age_ns = _duration_ns(
        model,
        "runtime maximum frame age",
        millisecond_paths=(
            ("runtime_max_age_ms",),
            ("max_frame_age_ms",),
            ("runtime", "max_age_ms"),
            ("runtime", "max_frame_age_ms"),
            ("runtime", "maximum_frame_age_ms"),
        ),
        second_paths=(
            ("runtime_limits", "maximum_sample_age_s"),
            ("runtime_max_age_s",),
            ("max_frame_age_s",),
            ("runtime", "max_frame_age_s"),
        ),
        nanosecond_paths=(("runtime_max_age_ns",), ("max_frame_age_ns",), ("runtime", "max_frame_age_ns")),
    )
    max_pair_skew_ns = _duration_ns(
        model,
        "runtime maximum gyro/Euler skew",
        millisecond_paths=(
            ("runtime_max_skew_ms",),
            ("max_pair_skew_ms",),
            ("runtime", "max_skew_ms"),
            ("runtime", "max_pair_skew_ms"),
            ("runtime", "maximum_pair_skew_ms"),
        ),
        second_paths=(
            ("runtime_limits", "maximum_gyro_orientation_pair_skew_s"),
            ("runtime_max_skew_s",),
            ("max_pair_skew_s",),
            ("runtime", "max_pair_skew_s"),
        ),
        nanosecond_paths=(("runtime_max_skew_ns",), ("max_pair_skew_ns",), ("runtime", "max_pair_skew_ns")),
    )
    _validate_duplicate_matrix(
        model,
        (("sensor_to_base_matrix",), ("runtime_transform", "sensor_to_base_matrix")),
        matrix,
        "sensor-to-base matrix",
    )
    _validate_duplicate_vector(
        model,
        (("gyro_bias_rad_s",), ("measured", "held_baseline", "gyro", "mean_rad_s")),
        bias,
        "gyro bias",
    )
    _validate_duplicate_number(
        model,
        (("policy_angular_velocity_scale",), ("policy_observation", "angular_velocity_scale")),
        scale,
        "policy angular-velocity scale",
    )
    _validate_duplicate_number(
        model,
        (("policy_hz",), ("policy_observation", "policy_hz")),
        policy_rate_hz,
        "policy rate",
    )
    _validate_duplicate_number(
        model,
        (("sensor_rates_hz", "gyro"), ("measured", "packet_rates", "gyro_rate_hz")),
        gyro_rate_hz,
        "gyro rate",
    )
    _validate_duplicate_number(
        model,
        (("sensor_rates_hz", "euler"), ("measured", "packet_rates", "orientation_rate_hz")),
        euler_rate_hz,
        "Euler rate",
    )
    _validate_duplicate_duration(
        model,
        (
            (("runtime", "max_frame_age_ms"), 1.0e6),
            (("runtime_limits", "maximum_sample_age_s"), 1.0e9),
        ),
        max_frame_age_ns,
        "maximum frame age",
    )
    _validate_duplicate_duration(
        model,
        (
            (("runtime", "max_pair_skew_ms"), 1.0e6),
            (("runtime_limits", "maximum_gyro_orientation_pair_skew_s"), 1.0e9),
        ),
        max_pair_skew_ns,
        "maximum pair skew",
    )
    return CMP10ARuntimeConfig(
        sensor_to_base_matrix=matrix,
        gyro_bias_sensor_rad_s=bias,
        policy_angular_velocity_scale=scale,
        sensor_baudrate=baudrate,
        sensor_gyro_rate_hz=gyro_rate_hz,
        sensor_euler_rate_hz=euler_rate_hz,
        policy_rate_hz=policy_rate_hz,
        max_frame_age_ns=max_frame_age_ns,
        max_pair_skew_ns=max_pair_skew_ns,
    )


def validate_cmp10a_runtime_model(model: Mapping[str, Any]) -> None:
    """Validate all fields needed for read-only policy observation conversion."""

    _runtime_config(model)


def load_cmp10a_runtime_model(source: RuntimeModelSource) -> dict[str, Any]:
    """Load a runtime model from a mapping or JSON path, then validate it."""

    if isinstance(source, Mapping):
        model = dict(source)
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise CMP10ARuntimeModelError(f"CMP10A runtime model does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise CMP10ARuntimeModelError(f"CMP10A runtime model is not valid JSON: {path}: {error}") from error
        except OSError as error:
            raise CMP10ARuntimeModelError(f"Unable to read CMP10A runtime model {path}: {error}") from error
        if not isinstance(model, dict):
            raise CMP10ARuntimeModelError(f"CMP10A runtime model must contain a JSON object: {path}")
    else:
        raise CMP10ARuntimeModelError("CMP10A runtime model source must be a mapping or filesystem path.")
    validate_cmp10a_runtime_model(model)
    return model


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (_dot(matrix[0], vector), _dot(matrix[1], vector), _dot(matrix[2], vector))


def _frame_vector(frame: CMP10AFrame, keys: tuple[str, str, str], label: str) -> Vector3:
    try:
        values = tuple(float(frame.values[key]) for key in keys)
    except (KeyError, TypeError, ValueError) as error:
        raise CMP10ARuntimeFrameError(f"CMP10A {label} frame is missing decoded values {keys}.") from error
    result = (values[0], values[1], values[2])
    if not all(math.isfinite(value) for value in result):
        raise CMP10ARuntimeFrameError(f"CMP10A {label} frame contains non-finite decoded values: {result!r}.")
    return result


class CMP10ARuntimeAdapter:
    """Retain the latest valid gyro/Euler frames and sample them atomically."""

    def __init__(self, model: RuntimeModelSource):
        loaded_model = load_cmp10a_runtime_model(model)
        self.config = _runtime_config(loaded_model)
        self._lock = threading.Lock()
        self._gyro_frame: CMP10AFrame | None = None
        self._euler_frame: CMP10AFrame | None = None
        self._ingested_frames = 0
        self._gyro_frames = 0
        self._euler_frames = 0
        self._ignored_frames = 0
        self._out_of_order_frames = 0

    def _counters_unlocked(self) -> CMP10ARuntimeCounters:
        return CMP10ARuntimeCounters(
            ingested_frames=self._ingested_frames,
            gyro_frames=self._gyro_frames,
            euler_frames=self._euler_frames,
            ignored_frames=self._ignored_frames,
            out_of_order_frames=self._out_of_order_frames,
        )

    @property
    def counters(self) -> CMP10ARuntimeCounters:
        with self._lock:
            return self._counters_unlocked()

    def ingest(self, frame: CMP10AFrame) -> bool:
        """Validate and retain a target frame if it is the latest of its type.

        Returns ``True`` when a gyro or Euler frame becomes the retained latest
        frame. Other protocol frame types and out-of-order target frames return
        ``False``.
        """

        if not isinstance(frame, CMP10AFrame):
            raise CMP10ARuntimeFrameError(f"frame must be CMP10AFrame, got {type(frame).__name__}.")
        if not isinstance(frame.timestamp_ns, int) or isinstance(frame.timestamp_ns, bool) or frame.timestamp_ns < 0:
            raise CMP10ARuntimeFrameError(
                f"frame.timestamp_ns must be a non-negative integer, got {frame.timestamp_ns!r}."
            )
        try:
            validated = decode_frame(frame.raw, frame.timestamp_ns)
        except (CMP10AProtocolError, TypeError) as error:
            raise CMP10ARuntimeFrameError(f"CMP10A frame failed checksum/protocol validation: {error}") from error
        if validated.frame_type != frame.frame_type:
            raise CMP10ARuntimeFrameError(
                f"CMP10AFrame frame_type 0x{frame.frame_type:02X} does not match raw type 0x{validated.frame_type:02X}."
            )

        with self._lock:
            self._ingested_frames += 1
            if validated.frame_type == _GYRO_FRAME_TYPE:
                self._gyro_frames += 1
                if self._gyro_frame is not None and validated.timestamp_ns < self._gyro_frame.timestamp_ns:
                    self._out_of_order_frames += 1
                    return False
                self._gyro_frame = validated
                return True
            if validated.frame_type == _EULER_FRAME_TYPE:
                self._euler_frames += 1
                if self._euler_frame is not None and validated.timestamp_ns < self._euler_frame.timestamp_ns:
                    self._out_of_order_frames += 1
                    return False
                self._euler_frame = validated
                return True
            self._ignored_frames += 1
            return False

    ingest_frame = ingest

    def snapshot(self, now_ns: int) -> CMP10ARuntimeSample:
        """Return the latest coherent pair or fail closed with a clear reason."""

        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
            raise CMP10ARuntimeSnapshotError(f"now_ns must be a non-negative integer, got {now_ns!r}.")
        with self._lock:
            gyro_frame = self._gyro_frame
            euler_frame = self._euler_frame
            counters = self._counters_unlocked()

        missing = []
        if gyro_frame is None:
            missing.append("gyro 0x52")
        if euler_frame is None:
            missing.append("Euler 0x53")
        if missing:
            raise CMP10ARuntimeSnapshotError(f"CMP10A snapshot is missing latest {' and '.join(missing)} frame(s).")

        gyro_age_ns = now_ns - gyro_frame.timestamp_ns
        euler_age_ns = now_ns - euler_frame.timestamp_ns
        if gyro_age_ns < 0 or euler_age_ns < 0:
            future = []
            if gyro_age_ns < 0:
                future.append(f"gyro timestamp {gyro_frame.timestamp_ns}")
            if euler_age_ns < 0:
                future.append(f"Euler timestamp {euler_frame.timestamp_ns}")
            raise CMP10ARuntimeSnapshotError(
                f"CMP10A snapshot contains future source timestamp(s) relative to now_ns={now_ns}: {', '.join(future)}."
            )
        stale = []
        if gyro_age_ns > self.config.max_frame_age_ns:
            stale.append(f"gyro age {gyro_age_ns} ns")
        if euler_age_ns > self.config.max_frame_age_ns:
            stale.append(f"Euler age {euler_age_ns} ns")
        if stale:
            raise CMP10ARuntimeSnapshotError(
                f"CMP10A snapshot is stale ({', '.join(stale)}); maximum is {self.config.max_frame_age_ns} ns."
            )
        pair_skew_ns = abs(gyro_frame.timestamp_ns - euler_frame.timestamp_ns)
        if pair_skew_ns > self.config.max_pair_skew_ns:
            raise CMP10ARuntimeSnapshotError(
                f"CMP10A gyro/Euler skew is {pair_skew_ns} ns; maximum is {self.config.max_pair_skew_ns} ns."
            )

        omega_sensor = _frame_vector(gyro_frame, ("x_rad_s", "y_rad_s", "z_rad_s"), "gyro")
        corrected_sensor = (
            omega_sensor[0] - self.config.gyro_bias_sensor_rad_s[0],
            omega_sensor[1] - self.config.gyro_bias_sensor_rad_s[1],
            omega_sensor[2] - self.config.gyro_bias_sensor_rad_s[2],
        )
        omega_base = _matrix_vector(self.config.sensor_to_base_matrix, corrected_sensor)
        policy_omega = tuple(value * self.config.policy_angular_velocity_scale for value in omega_base)

        roll, pitch, _yaw = _frame_vector(
            euler_frame,
            ("roll_rad", "pitch_rad", "yaw_rad"),
            "Euler",
        )
        cos_pitch = math.cos(pitch)
        gravity_sensor: Vector3 = (
            math.sin(pitch),
            -math.sin(roll) * cos_pitch,
            -math.cos(roll) * cos_pitch,
        )
        gravity_base_unscaled = _matrix_vector(self.config.sensor_to_base_matrix, gravity_sensor)
        gravity_norm = math.sqrt(_dot(gravity_base_unscaled, gravity_base_unscaled))
        if not math.isfinite(gravity_norm) or gravity_norm <= 1.0e-12:
            raise CMP10ARuntimeSnapshotError(
                f"CMP10A Euler-derived projected gravity has invalid norm {gravity_norm!r}."
            )
        gravity_base = tuple(value / gravity_norm for value in gravity_base_unscaled)

        return CMP10ARuntimeSample(
            base_angular_velocity_rad_s=omega_base,
            policy_angular_velocity=(policy_omega[0], policy_omega[1], policy_omega[2]),
            projected_gravity_b=(gravity_base[0], gravity_base[1], gravity_base[2]),
            gyro_timestamp_ns=gyro_frame.timestamp_ns,
            euler_timestamp_ns=euler_frame.timestamp_ns,
            gyro_age_ns=gyro_age_ns,
            euler_age_ns=euler_age_ns,
            pair_skew_ns=pair_skew_ns,
            counters=counters,
        )


class CMP10ARuntimeSource:
    """Daemon-thread serial source that only opens, reads, and closes CMP10A."""

    def __init__(
        self,
        device: str,
        model: RuntimeModelSource,
        *,
        read_size: int = CMP10A_FRAME_SIZE,
        read_timeout_s: float = 0.01,
        close_timeout_s: float = 1.0,
        startup_timeout_s: float = 1.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        reader_factory: Callable[..., Any] = CMP10ASerialReader,
    ):
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty serial-device path.")
        if not isinstance(read_size, int) or isinstance(read_size, bool) or read_size <= 0:
            raise ValueError("read_size must be a positive integer.")
        for value, label in (
            (read_timeout_s, "read_timeout_s"),
            (close_timeout_s, "close_timeout_s"),
            (startup_timeout_s, "startup_timeout_s"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{label} must be a finite number.")
            if float(value) <= 0.0:
                raise ValueError(f"{label} must be positive.")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable.")
        if not callable(reader_factory):
            raise TypeError("reader_factory must be callable.")

        self.device = device
        self.adapter = CMP10ARuntimeAdapter(model)
        self.config = self.adapter.config
        self.read_size = read_size
        self.read_timeout_s = float(read_timeout_s)
        self.close_timeout_s = float(close_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self._monotonic_ns = monotonic_ns
        self._reader_factory = reader_factory
        self._parser = CMP10AParser()
        self._data_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None
        self._started = False
        self._closed = False

    @property
    def parser_counters(self) -> CMP10AParserCounters:
        with self._data_lock:
            return self._parser.counters

    @property
    def counters(self) -> CMP10ARuntimeCounters:
        return self.adapter.counters

    @property
    def thread_error(self) -> BaseException | None:
        with self._state_lock:
            return self._thread_error

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def thread_is_daemon(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.daemon

    def _record_thread_error(self, error: BaseException) -> None:
        with self._state_lock:
            self._thread_error = error

    def _raise_thread_error(self) -> None:
        error = self.thread_error
        if error is not None:
            raise CMP10ARuntimeSourceError(
                f"CMP10A background reader failed with {type(error).__name__}: {error}"
            ) from error

    def _run(self) -> None:
        try:
            reader = self._reader_factory(
                self.device,
                baudrate=self.config.sensor_baudrate,
                timeout=self.read_timeout_s,
            )
            with reader:
                reset_input_buffer = getattr(reader, "reset_input_buffer", None)
                if callable(reset_input_buffer):
                    reset_input_buffer()
                self._ready_event.set()
                while not self._stop_event.is_set():
                    data = reader.read(self.read_size)
                    if not isinstance(data, bytes):
                        raise TypeError(f"CMP10A reader.read() must return bytes, got {type(data).__name__}.")
                    if not data:
                        continue
                    timestamp_ns = self._monotonic_ns()
                    with self._data_lock:
                        for frame in self._parser.feed(data, timestamp_ns):
                            self.adapter.ingest(frame)
        except BaseException as error:
            self._record_thread_error(error)
        finally:
            self._ready_event.set()

    def start(self) -> CMP10ARuntimeSource:
        """Start the one-shot daemon reader and wait until the port is open."""

        with self._state_lock:
            if self._closed:
                raise CMP10ARuntimeSourceError("CMP10A runtime source is closed and cannot be restarted.")
            if self._started:
                thread = self._thread
                if thread is not None and thread.is_alive():
                    return self
                error = self._thread_error
                if error is None:
                    raise CMP10ARuntimeSourceError("CMP10A runtime source stopped and cannot be restarted.")
            else:
                self._started = True
                self._thread = threading.Thread(
                    target=self._run,
                    name="cmp10a-runtime-reader",
                    daemon=True,
                )
                self._thread.start()
        if not self._ready_event.wait(self.startup_timeout_s):
            self._stop_event.set()
            raise CMP10ARuntimeSourceError(
                f"CMP10A background reader did not open within {self.startup_timeout_s:.3f} s."
            )
        self._raise_thread_error()
        return self

    def snapshot(self, now_ns: int | None = None) -> CMP10ARuntimeSample:
        """Sample the latest pair, using the injected monotonic clock by default."""

        self._raise_thread_error()
        with self._state_lock:
            if not self._started:
                raise CMP10ARuntimeSourceError("CMP10A runtime source has not been started.")
            if self._closed:
                raise CMP10ARuntimeSourceError("CMP10A runtime source is closed.")
        timestamp_ns = self._monotonic_ns() if now_ns is None else now_ns
        with self._data_lock:
            sample = self.adapter.snapshot(timestamp_ns)
            parser_counters = self._parser.counters
        self._raise_thread_error()
        return replace(sample, parser_counters=parser_counters)

    def close(self) -> None:
        """Stop and join the reader, then surface any background failure."""

        with self._state_lock:
            self._closed = True
            thread = self._thread
        self._stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.close_timeout_s)
            if thread.is_alive():
                raise CMP10ARuntimeSourceError(
                    f"CMP10A background reader did not stop within {self.close_timeout_s:.3f} s."
                )
        self._raise_thread_error()

    def __enter__(self) -> CMP10ARuntimeSource:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except CMP10ARuntimeSourceError:
            if exc_type is None:
                raise


CMP10ALatestFrameAdapter = CMP10ARuntimeAdapter
CMP10ABackgroundSource = CMP10ARuntimeSource


__all__ = [
    "CMP10ABackgroundSource",
    "CMP10ALatestFrameAdapter",
    "CMP10ARuntimeAdapter",
    "CMP10ARuntimeConfig",
    "CMP10ARuntimeCounters",
    "CMP10ARuntimeFrameError",
    "CMP10ARuntimeModelError",
    "CMP10ARuntimeSample",
    "CMP10ARuntimeSnapshotError",
    "CMP10ARuntimeSource",
    "CMP10ARuntimeSourceError",
    "CMP10A_RUNTIME_MODEL_TYPE",
    "CMP10A_RUNTIME_SCHEMA_VERSION",
    "load_cmp10a_runtime_model",
    "validate_cmp10a_runtime_model",
]
