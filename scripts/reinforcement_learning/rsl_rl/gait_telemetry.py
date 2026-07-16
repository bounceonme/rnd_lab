"""Quantitative gait telemetry collection for RSL-RL policy playback."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class GaitTelemetryError(RuntimeError):
    """Raised when gait telemetry cannot be collected without ambiguous data."""


_KNOWN_BIPED_FOOT_PAIRS = (
    ("r_leg_foot", "l_leg_foot"),
    ("right_foot", "left_foot"),
    ("right_foot_link", "left_foot_link"),
    ("r_foot", "l_foot"),
    ("right_ankle_roll_link", "left_ankle_roll_link"),
)


def _side_from_name(name: str) -> str | None:
    normalized = name.lower()
    if re.search(r"(?:^|[^a-z0-9])(?:right|r)(?:[^a-z0-9]|$)", normalized):
        return "right"
    if re.search(r"(?:^|[^a-z0-9])(?:left|l)(?:[^a-z0-9]|$)", normalized):
        return "left"
    return None


def resolve_ordered_foot_names(body_names: Sequence[str]) -> tuple[str, str]:
    """Resolve exactly one right and one left foot, independent of body-list order."""
    names = tuple(str(name) for name in body_names)
    names_by_lower = {name.lower(): name for name in names}
    if len(names_by_lower) != len(names):
        raise GaitTelemetryError("Robot body names are not unique when compared case-insensitively.")

    for right_alias, left_alias in _KNOWN_BIPED_FOOT_PAIRS:
        if right_alias in names_by_lower and left_alias in names_by_lower:
            return names_by_lower[right_alias], names_by_lower[left_alias]

    for keyword in ("foot", "ankle_roll"):
        candidates = [name for name in names if keyword in name.lower()]
        right_candidates = [name for name in candidates if _side_from_name(name) == "right"]
        left_candidates = [name for name in candidates if _side_from_name(name) == "left"]
        if len(right_candidates) == 1 and len(left_candidates) == 1:
            return right_candidates[0], left_candidates[0]

    raise GaitTelemetryError(
        "Could not resolve one right and one left foot from robot bodies. "
        f"Bodies containing 'foot' or 'ankle_roll': "
        f"{[name for name in names if 'foot' in name.lower() or 'ankle_roll' in name.lower()]}"
    )


def _indices_for_names(all_names: Sequence[str], selected_names: Sequence[str], entity_label: str) -> tuple[int, ...]:
    names = tuple(str(name) for name in all_names)
    indices = []
    for selected_name in selected_names:
        matches = [index for index, name in enumerate(names) if name == selected_name]
        if len(matches) != 1:
            raise GaitTelemetryError(
                f"Expected exactly one {entity_label} body named '{selected_name}', found {len(matches)}."
            )
        indices.append(matches[0])
    if len(set(indices)) != len(indices):
        raise GaitTelemetryError(f"Resolved duplicate {entity_label} body indices: {indices}.")
    return tuple(indices)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _env_row(value: Any, env_index: int) -> np.ndarray:
    return np.array(_to_numpy(value[env_index]), copy=True)


def _env_entities(value: Any, env_index: int, entity_ids: Sequence[int]) -> np.ndarray:
    return np.array(_to_numpy(value[env_index, list(entity_ids)]), copy=True)


class GaitTelemetryLogger:
    """Collect synchronized post-step gait state from one vectorized environment."""

    _SERIES_KEYS = (
        "step",
        "time_s",
        "done",
        "command",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "foot_pos_w",
        "foot_quat_w",
        "foot_net_force_w",
        "foot_contact",
        "foot_current_air_time",
        "foot_last_air_time",
        "foot_current_contact_time",
        "foot_last_contact_time",
        "actions",
        "joint_pos",
        "joint_vel",
        "applied_torque",
        "computed_torque",
    )

    def __init__(
        self,
        sim_env: Any,
        *,
        task: str,
        checkpoint: str | Path,
        env_index: int,
        step_dt: float,
        command_name: str = "base_velocity",
    ) -> None:
        num_envs = int(sim_env.num_envs)
        if env_index < 0 or env_index >= num_envs:
            raise GaitTelemetryError(f"Gait log environment index {env_index} is outside [0, {num_envs}).")

        try:
            robot = sim_env.scene["robot"]
            contact_sensor = sim_env.scene["contact_forces"]
        except (KeyError, TypeError) as exc:
            raise GaitTelemetryError("Gait logging requires scene entities 'robot' and 'contact_forces'.") from exc

        foot_names = resolve_ordered_foot_names(robot.body_names)
        robot_foot_ids = _indices_for_names(robot.body_names, foot_names, "robot")
        sensor_foot_ids = _indices_for_names(contact_sensor.body_names, foot_names, "contact-sensor")

        contact_data = contact_sensor.data
        timer_names = (
            "current_air_time",
            "last_air_time",
            "current_contact_time",
            "last_contact_time",
        )
        missing_timers = [name for name in timer_names if getattr(contact_data, name, None) is None]
        if missing_timers:
            raise GaitTelemetryError(
                "Contact sensor must enable track_air_time; missing telemetry fields: " + ", ".join(missing_timers)
            )

        self._sim_env = sim_env
        self._robot = robot
        self._contact_sensor = contact_sensor
        self._env_index = env_index
        self._robot_foot_ids = robot_foot_ids
        self._sensor_foot_ids = sensor_foot_ids
        self._command_name = command_name
        self._step_dt = float(step_dt)
        self._task = str(task)
        self._checkpoint = str(Path(checkpoint).expanduser().resolve())
        self._foot_names = foot_names
        self._joint_names = tuple(str(name) for name in robot.joint_names)
        self._force_threshold = float(contact_sensor.cfg.force_threshold)
        self._records: dict[str, list[np.ndarray]] = {key: [] for key in self._SERIES_KEYS}

    @property
    def num_steps(self) -> int:
        return len(self._records["step"])

    def _append(self, key: str, value: Any) -> None:
        self._records[key].append(np.array(value, copy=True))

    def record(self, actions: Any, dones: Any | None = None) -> None:
        """Capture one post-step snapshot for the configured environment."""
        robot_data = self._robot.data
        contact_data = self._contact_sensor.data
        env_index = self._env_index
        robot_foot_ids = self._robot_foot_ids
        sensor_foot_ids = self._sensor_foot_ids

        foot_net_force_w = _env_entities(contact_data.net_forces_w, env_index, sensor_foot_ids)
        foot_contact = np.linalg.norm(foot_net_force_w, axis=-1) > self._force_threshold
        done = False if dones is None else bool(_to_numpy(dones)[env_index])
        step = self.num_steps

        self._append("step", np.int64(step))
        self._append("time_s", np.float64((step + 1) * self._step_dt))
        self._append("done", np.bool_(done))
        self._append("command", _env_row(self._sim_env.command_manager.get_command(self._command_name), env_index))
        self._append("root_pos_w", _env_row(robot_data.root_pos_w, env_index))
        self._append("root_quat_w", _env_row(robot_data.root_quat_w, env_index))
        self._append("root_lin_vel_w", _env_row(robot_data.root_lin_vel_w, env_index))
        self._append("root_ang_vel_w", _env_row(robot_data.root_ang_vel_w, env_index))
        self._append("foot_pos_w", _env_entities(robot_data.body_pos_w, env_index, robot_foot_ids))
        self._append("foot_quat_w", _env_entities(robot_data.body_quat_w, env_index, robot_foot_ids))
        self._append("foot_net_force_w", foot_net_force_w)
        self._append("foot_contact", foot_contact)
        self._append("foot_current_air_time", _env_entities(contact_data.current_air_time, env_index, sensor_foot_ids))
        self._append("foot_last_air_time", _env_entities(contact_data.last_air_time, env_index, sensor_foot_ids))
        self._append(
            "foot_current_contact_time",
            _env_entities(contact_data.current_contact_time, env_index, sensor_foot_ids),
        )
        self._append(
            "foot_last_contact_time",
            _env_entities(contact_data.last_contact_time, env_index, sensor_foot_ids),
        )
        self._append("actions", _env_row(actions, env_index))
        self._append("joint_pos", _env_row(robot_data.joint_pos, env_index))
        self._append("joint_vel", _env_row(robot_data.joint_vel, env_index))
        self._append("applied_torque", _env_row(robot_data.applied_torque, env_index))
        self._append("computed_torque", _env_row(robot_data.computed_torque, env_index))

    def _payload(self) -> dict[str, np.ndarray]:
        payload = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "task": np.asarray(self._task),
            "checkpoint": np.asarray(self._checkpoint),
            "step_dt": np.asarray(self._step_dt, dtype=np.float64),
            "env_index": np.asarray(self._env_index, dtype=np.int64),
            "command_name": np.asarray(self._command_name),
            "foot_order": np.asarray(("right", "left")),
            "foot_body_names": np.asarray(self._foot_names),
            "joint_names": np.asarray(self._joint_names),
            "contact_force_threshold": np.asarray(self._force_threshold, dtype=np.float64),
        }
        for key, values in self._records.items():
            payload[key] = np.stack(values, axis=0) if values else np.empty((0,), dtype=np.float64)
        return payload

    def save(self, output_path: str | Path) -> Path:
        """Save all collected samples as a compressed, pickle-free NPZ file."""
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output_file:
            np.savez_compressed(output_file, **self._payload())
        return path
