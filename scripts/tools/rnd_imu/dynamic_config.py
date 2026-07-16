"""Configuration loading for guided CMP10A dynamic identification."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path


class DynamicImuConfigError(ValueError):
    """Raised when a dynamic IMU configuration is invalid."""


@dataclasses.dataclass(frozen=True)
class DynamicSerialConfig:
    port: str
    baudrate: int
    read_timeout_s: float


@dataclasses.dataclass(frozen=True)
class DynamicExperimentConfig:
    policy_hz: float
    baseline_duration_s: float
    neutral_duration_s: float
    half_cycle_s: float
    cycles: int
    center_hold_s: float
    countdown_s: int

    @property
    def stage_duration_s(self) -> float:
        return self.neutral_duration_s + 2.0 * self.cycles * self.half_cycle_s + self.center_hold_s


@dataclasses.dataclass(frozen=True)
class DynamicQualityConfig:
    maximum_relative_lag_ms: float
    minimum_correlation: float
    minimum_axis_rms_rad_s: float
    minimum_dominance_ratio: float
    maximum_checksum_error_fraction: float


@dataclasses.dataclass(frozen=True)
class DynamicImuConfig:
    serial: DynamicSerialConfig
    experiment: DynamicExperimentConfig
    quality: DynamicQualityConfig


def _positive_float(section: dict, name: str) -> float:
    value = float(section[name])
    if value <= 0.0:
        raise DynamicImuConfigError(f"{name} must be positive, got {value}.")
    return value


def load_dynamic_imu_config(path: str | Path) -> DynamicImuConfig:
    """Load and validate the guided dynamic-identification TOML."""

    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
        serial = raw["serial"]
        experiment = raw["experiment"]
        quality = raw["quality"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise DynamicImuConfigError(f"Unable to load dynamic IMU config {path}: {error}") from error

    baudrate = int(serial["baudrate"])
    cycles = int(experiment["cycles"])
    countdown_s = int(experiment["countdown_s"])
    if baudrate <= 0 or cycles < 1 or countdown_s < 0:
        raise DynamicImuConfigError("baudrate and cycles must be positive; countdown_s must be non-negative.")

    result = DynamicImuConfig(
        serial=DynamicSerialConfig(
            port=str(serial["port"]),
            baudrate=baudrate,
            read_timeout_s=_positive_float(serial, "read_timeout_s"),
        ),
        experiment=DynamicExperimentConfig(
            policy_hz=_positive_float(experiment, "policy_hz"),
            baseline_duration_s=_positive_float(experiment, "baseline_duration_s"),
            neutral_duration_s=_positive_float(experiment, "neutral_duration_s"),
            half_cycle_s=_positive_float(experiment, "half_cycle_s"),
            cycles=cycles,
            center_hold_s=_positive_float(experiment, "center_hold_s"),
            countdown_s=countdown_s,
        ),
        quality=DynamicQualityConfig(
            maximum_relative_lag_ms=_positive_float(quality, "maximum_relative_lag_ms"),
            minimum_correlation=_positive_float(quality, "minimum_correlation"),
            minimum_axis_rms_rad_s=_positive_float(quality, "minimum_axis_rms_rad_s"),
            minimum_dominance_ratio=_positive_float(quality, "minimum_dominance_ratio"),
            maximum_checksum_error_fraction=_positive_float(quality, "maximum_checksum_error_fraction"),
        ),
    )
    if result.quality.minimum_correlation > 1.0:
        raise DynamicImuConfigError("minimum_correlation must not exceed one.")
    if result.quality.maximum_checksum_error_fraction > 1.0:
        raise DynamicImuConfigError("maximum_checksum_error_fraction must not exceed one.")
    return result
