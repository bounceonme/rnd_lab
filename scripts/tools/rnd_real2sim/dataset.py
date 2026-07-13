"""Versioned, atomic storage for RND actuator-identification telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bus import MotorTelemetry


SCHEMA_VERSION = 1
MATRIX_FIELDS = (
    "device_tick_ms",
    "goal_position_rad",
    "position_rad",
    "velocity_rad_s",
    "current_a",
    "pwm_fraction",
    "velocity_trajectory_rad_s",
    "position_trajectory_rad",
    "voltage_v",
    "temperature_c",
    "moving",
    "moving_status",
    "raw_position",
)


class DatasetError(ValueError):
    """Raised when a saved dataset is missing required or consistent data."""


@dataclass(frozen=True)
class Real2SimDataset:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    sha256: str

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["joint_names"])

    @property
    def sample_count(self) -> int:
        return int(self.arrays["time_s"].shape[0])


class DatasetRecorder:
    def __init__(self, joint_names: tuple[str, ...], metadata: dict[str, Any]):
        self.joint_names = tuple(joint_names)
        self.metadata = dict(metadata)
        self.metadata["schema_version"] = SCHEMA_VERSION
        self.metadata["joint_names"] = list(self.joint_names)
        self._rows: dict[str, list[Any]] = {
            "time_s": [],
            "phase_id": [],
            "excitation_joint_id": [],
            "deadline_overrun_s": [],
        }
        for field in MATRIX_FIELDS:
            self._rows[field] = []

    @property
    def sample_count(self) -> int:
        return len(self._rows["time_s"])

    def append(
        self,
        *,
        time_s: float,
        phase_id: int,
        excitation_joint_id: int,
        deadline_overrun_s: float,
        goals_rad: dict[str, float],
        telemetry: dict[str, MotorTelemetry],
    ) -> None:
        missing_goals = set(self.joint_names) - set(goals_rad)
        missing_telemetry = set(self.joint_names) - set(telemetry)
        if missing_goals or missing_telemetry:
            raise DatasetError(
                f"Incomplete sample. Missing goals={sorted(missing_goals)}, "
                f"missing telemetry={sorted(missing_telemetry)}."
            )
        self._rows["time_s"].append(float(time_s))
        self._rows["phase_id"].append(int(phase_id))
        self._rows["excitation_joint_id"].append(int(excitation_joint_id))
        self._rows["deadline_overrun_s"].append(float(deadline_overrun_s))
        self._rows["device_tick_ms"].append([telemetry[name].tick_ms for name in self.joint_names])
        self._rows["goal_position_rad"].append([goals_rad[name] for name in self.joint_names])
        self._rows["position_rad"].append([telemetry[name].position_rad for name in self.joint_names])
        self._rows["velocity_rad_s"].append([telemetry[name].velocity_rad_s for name in self.joint_names])
        self._rows["current_a"].append([telemetry[name].current_a for name in self.joint_names])
        self._rows["pwm_fraction"].append([telemetry[name].pwm_fraction for name in self.joint_names])
        self._rows["velocity_trajectory_rad_s"].append([
            telemetry[name].velocity_trajectory_rad_s for name in self.joint_names
        ])
        self._rows["position_trajectory_rad"].append([
            telemetry[name].position_trajectory_rad for name in self.joint_names
        ])
        self._rows["voltage_v"].append([telemetry[name].voltage_v for name in self.joint_names])
        self._rows["temperature_c"].append([telemetry[name].temperature_c for name in self.joint_names])
        self._rows["moving"].append([telemetry[name].moving for name in self.joint_names])
        self._rows["moving_status"].append([telemetry[name].moving_status for name in self.joint_names])
        self._rows["raw_position"].append([telemetry[name].raw_position for name in self.joint_names])

    def _arrays(self) -> dict[str, np.ndarray]:
        sample_count = self.sample_count
        joint_count = len(self.joint_names)
        arrays = {
            "time_s": np.asarray(self._rows["time_s"], dtype=np.float64),
            "phase_id": np.asarray(self._rows["phase_id"], dtype=np.int16),
            "excitation_joint_id": np.asarray(self._rows["excitation_joint_id"], dtype=np.int16),
            "deadline_overrun_s": np.asarray(self._rows["deadline_overrun_s"], dtype=np.float64),
        }
        integer_fields = {"device_tick_ms", "moving", "moving_status", "raw_position"}
        for field in MATRIX_FIELDS:
            dtype = np.int32 if field in integer_fields else np.float64
            value = np.asarray(self._rows[field], dtype=dtype)
            arrays[field] = value.reshape(sample_count, joint_count)
        return arrays

    def save(self, path: str | Path, *, status: str, status_detail: str = "") -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["status"] = status
        metadata["status_detail"] = status_detail
        metadata["sample_count"] = self.sample_count
        payload = self._arrays()
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: str | Path, *, allow_incomplete: bool = False) -> Real2SimDataset:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise DatasetError(f"Dataset does not exist: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "time_s",
                "phase_id",
                "excitation_joint_id",
                "deadline_overrun_s",
                *MATRIX_FIELDS,
            }
            missing = required - set(archive.files)
            if missing:
                raise DatasetError(f"Dataset is missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {name: np.asarray(archive[name]).copy() for name in required if name != "metadata_json"}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetError(f"Could not load {resolved}: {error}") from error

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"Unsupported dataset schema {metadata.get('schema_version')}; expected {SCHEMA_VERSION}.")
    if metadata.get("status") != "complete" and not allow_incomplete:
        raise DatasetError(
            f"Dataset status is {metadata.get('status')!r}. Pass allow_incomplete=True only for diagnostics."
        )
    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, list) or not joint_names:
        raise DatasetError("metadata.joint_names must be a non-empty list.")
    sample_count = arrays["time_s"].shape[0]
    if arrays["time_s"].ndim != 1 or sample_count < 2:
        raise DatasetError("Dataset requires at least two time samples.")
    if not np.all(np.isfinite(arrays["time_s"])) or np.any(np.diff(arrays["time_s"]) <= 0.0):
        raise DatasetError("time_s must be finite and strictly increasing.")
    for field in MATRIX_FIELDS:
        if arrays[field].shape != (sample_count, len(joint_names)):
            raise DatasetError(f"{field} has shape {arrays[field].shape}; expected {(sample_count, len(joint_names))}.")
        if np.issubdtype(arrays[field].dtype, np.floating) and not np.all(np.isfinite(arrays[field])):
            raise DatasetError(f"{field} contains non-finite values.")
    for field in ("phase_id", "excitation_joint_id", "deadline_overrun_s"):
        if arrays[field].shape != (sample_count,):
            raise DatasetError(f"{field} must have shape {(sample_count,)}.")
    return Real2SimDataset(resolved, metadata, arrays, _sha256(resolved))
