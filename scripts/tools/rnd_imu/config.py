"""Configuration loading for the standalone CMP10A identification tool."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path


class ImuIdentificationConfigError(ValueError):
    """Raised when an IMU identification configuration is invalid."""


@dataclasses.dataclass(frozen=True)
class SerialConfig:
    port: str
    baud_candidates: tuple[int, ...]
    read_timeout_s: float
    probe_duration_s: float
    minimum_valid_probe_frames: int


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    policy_hz: float
    static_duration_s: float
    axis_duration_s: float
    countdown_s: int
    gravity_mps2: float
    minimum_axis_rotation_rad: float
    minimum_axis_dominance_ratio: float
    maximum_axis_fit_error_deg: float


@dataclasses.dataclass(frozen=True)
class QualityConfig:
    minimum_required_rate_hz: float
    maximum_checksum_error_fraction: float
    maximum_static_gyro_std_rad_s: float
    maximum_static_accel_norm_error_mps2: float


@dataclasses.dataclass(frozen=True)
class ImuIdentificationConfig:
    serial: SerialConfig
    experiment: ExperimentConfig
    quality: QualityConfig


def _positive_float(section: dict, name: str) -> float:
    value = float(section[name])
    if value <= 0.0:
        raise ImuIdentificationConfigError(f"{name} must be positive, got {value}.")
    return value


def load_imu_identification_config(path: str | Path) -> ImuIdentificationConfig:
    """Load and validate the CMP10A identification TOML."""

    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
        serial = raw["serial"]
        experiment = raw["experiment"]
        quality = raw["quality"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ImuIdentificationConfigError(f"Unable to load IMU config {path}: {error}") from error

    baud_candidates = tuple(int(value) for value in serial["baud_candidates"])
    if not baud_candidates or any(value <= 0 for value in baud_candidates):
        raise ImuIdentificationConfigError("baud_candidates must contain positive baud rates.")
    if len(baud_candidates) != len(set(baud_candidates)):
        raise ImuIdentificationConfigError("baud_candidates must not contain duplicates.")

    countdown_s = int(experiment["countdown_s"])
    minimum_valid_probe_frames = int(serial["minimum_valid_probe_frames"])
    if countdown_s < 0:
        raise ImuIdentificationConfigError("countdown_s must be non-negative.")
    if minimum_valid_probe_frames < 1:
        raise ImuIdentificationConfigError("minimum_valid_probe_frames must be at least one.")

    result = ImuIdentificationConfig(
        serial=SerialConfig(
            port=str(serial["port"]),
            baud_candidates=baud_candidates,
            read_timeout_s=_positive_float(serial, "read_timeout_s"),
            probe_duration_s=_positive_float(serial, "probe_duration_s"),
            minimum_valid_probe_frames=minimum_valid_probe_frames,
        ),
        experiment=ExperimentConfig(
            policy_hz=_positive_float(experiment, "policy_hz"),
            static_duration_s=_positive_float(experiment, "static_duration_s"),
            axis_duration_s=_positive_float(experiment, "axis_duration_s"),
            countdown_s=countdown_s,
            gravity_mps2=_positive_float(experiment, "gravity_mps2"),
            minimum_axis_rotation_rad=_positive_float(experiment, "minimum_axis_rotation_rad"),
            minimum_axis_dominance_ratio=_positive_float(experiment, "minimum_axis_dominance_ratio"),
            maximum_axis_fit_error_deg=_positive_float(experiment, "maximum_axis_fit_error_deg"),
        ),
        quality=QualityConfig(
            minimum_required_rate_hz=_positive_float(quality, "minimum_required_rate_hz"),
            maximum_checksum_error_fraction=_positive_float(quality, "maximum_checksum_error_fraction"),
            maximum_static_gyro_std_rad_s=_positive_float(quality, "maximum_static_gyro_std_rad_s"),
            maximum_static_accel_norm_error_mps2=_positive_float(quality, "maximum_static_accel_norm_error_mps2"),
        ),
    )
    if result.quality.minimum_required_rate_hz < result.experiment.policy_hz:
        raise ImuIdentificationConfigError("minimum_required_rate_hz must be at least policy_hz.")
    return result
