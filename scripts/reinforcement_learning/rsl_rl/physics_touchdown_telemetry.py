"""Chunked physics-rate telemetry for the RND STEP touchdown observer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2
SCHEMA_NAME = "robot_lab.physics_touchdown_telemetry"
DEFAULT_MONITOR_ATTRIBUTE = "physics_touchdown_monitor"


class PhysicsTouchdownTelemetryError(RuntimeError):
    """Raised when synchronized telemetry cannot be recorded unambiguously."""


def _scene_entity(scene: Any, name: str) -> Any:
    try:
        return scene[name]
    except (KeyError, TypeError):
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            try:
                return sensors[name]
            except (KeyError, TypeError):
                pass
    raise PhysicsTouchdownTelemetryError(f"Scene entity {name!r} is required for touchdown telemetry.")


class PhysicsTouchdownTelemetryLogger:
    """Buffer post-scene-update tensors on device and transfer complete chunks to CPU."""

    _SERIES_KEYS = (
        "episode_id",
        "physics_step",
        "episode_physics_step",
        "policy_step",
        "substep",
        "physics_time_s",
        "terminated",
        "truncated",
        "reset_after_sample",
        "command",
        "actions",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "foot_pos_w",
        "foot_lin_vel_w",
        "foot_ang_vel_w",
        "foot_force_w",
        "foot_contact",
        "foot_first",
        "foot_valid",
        "foot_preimpact_speed",
        "foot_preceding_air_time_s",
        "joint_pos",
        "joint_vel",
        "applied_torque",
        "computed_torque",
    )

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        env_ids: Sequence[int] | torch.Tensor | None,
        foot_body_names: Sequence[str],
        joint_names: Sequence[str],
        physics_dt: float,
        samples_per_policy: int,
        contact_force_threshold: float,
        min_air_time: float,
        chunk_size: int = 256,
        task: str = "",
        checkpoint: str | Path = "",
        command_name: str = "base_velocity",
        monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive; got {num_envs}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        if env_ids is None:
            selected_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            selected_ids = env_ids.to(device=self.device, dtype=torch.int64)
        else:
            selected_ids = torch.as_tensor(tuple(env_ids), dtype=torch.int64, device=self.device)
        if selected_ids.ndim != 1 or selected_ids.numel() == 0:
            raise PhysicsTouchdownTelemetryError("env_ids must select at least one environment.")
        if bool(torch.any((selected_ids < 0) | (selected_ids >= self.num_envs))):
            raise PhysicsTouchdownTelemetryError(f"env_ids contains an index outside [0, {self.num_envs}).")
        if torch.unique(selected_ids).numel() != selected_ids.numel():
            raise PhysicsTouchdownTelemetryError("env_ids must not contain duplicates.")
        foot_names = tuple(str(name) for name in foot_body_names)
        if len(foot_names) != 2 or len(set(foot_names)) != 2:
            raise PhysicsTouchdownTelemetryError(
                f"foot_body_names must contain two unique ordered feet; got {foot_names}."
            )

        self._env_ids = selected_ids
        self._foot_body_names = foot_names
        self._joint_names = tuple(str(name) for name in joint_names)
        self._physics_dt = float(physics_dt)
        self._samples_per_policy = int(samples_per_policy)
        self._contact_force_threshold = float(contact_force_threshold)
        self._min_air_time = float(min_air_time)
        self._task = str(task)
        self._checkpoint = str(checkpoint)
        self._command_name = str(command_name)
        self._monitor_attribute = str(monitor_attribute)

        self._buffers: dict[str, torch.Tensor] = {}
        self._specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._cpu_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in self._SERIES_KEYS}
        self._buffer_count = 0
        self._num_samples = 0
        self._num_cpu_flushes = 0

    @classmethod
    def from_attached_env(
        cls,
        env: Any,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        chunk_size: int = 256,
        task: str = "",
        checkpoint: str | Path = "",
        command_name: str = "base_velocity",
        monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
    ) -> PhysicsTouchdownTelemetryLogger:
        """Build logger metadata from the monitor attached to the environment."""
        monitor = getattr(env, monitor_attribute, None)
        if monitor is None:
            raise PhysicsTouchdownTelemetryError(
                f"env.{monitor_attribute} is missing; attach the touchdown monitor first."
            )
        robot = _scene_entity(env.scene, monitor.asset_name)
        return cls(
            num_envs=int(env.num_envs),
            device=env.device,
            env_ids=env_ids,
            foot_body_names=monitor.foot_body_names,
            joint_names=robot.joint_names,
            physics_dt=monitor.physics_dt,
            samples_per_policy=monitor.samples_per_policy,
            contact_force_threshold=monitor.force_threshold,
            min_air_time=monitor.min_air_time,
            chunk_size=chunk_size,
            task=task,
            checkpoint=checkpoint,
            command_name=command_name,
            monitor_attribute=monitor_attribute,
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_cpu_flushes(self) -> int:
        return self._num_cpu_flushes

    @property
    def env_ids(self) -> torch.Tensor:
        return self._env_ids.clone()

    def _select(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Telemetry field {key!r} must be a torch tensor.")
        if value.device != self.device:
            raise PhysicsTouchdownTelemetryError(
                f"Telemetry field {key!r} must remain on {self.device}; got {value.device}."
            )
        if value.ndim == 0 or value.shape[0] != self.num_envs:
            raise PhysicsTouchdownTelemetryError(
                f"Telemetry field {key!r} must lead with num_envs={self.num_envs}; got {tuple(value.shape)}."
            )
        return value.index_select(0, self._env_ids)

    def _validate_record_shapes(self, record: dict[str, torch.Tensor]) -> None:
        selected_envs = self._env_ids.numel()
        fixed_shapes = {
            "episode_id": (),
            "physics_step": (),
            "episode_physics_step": (),
            "policy_step": (),
            "substep": (),
            "physics_time_s": (),
            "terminated": (),
            "truncated": (),
            "reset_after_sample": (),
            "root_pos_w": (3,),
            "root_quat_w": (4,),
            "root_lin_vel_w": (3,),
            "root_ang_vel_w": (3,),
            "foot_pos_w": (2, 3),
            "foot_lin_vel_w": (2, 3),
            "foot_ang_vel_w": (2, 3),
            "foot_force_w": (2, 3),
            "foot_contact": (2,),
            "foot_first": (2,),
            "foot_valid": (2,),
            "foot_preimpact_speed": (2,),
            "foot_preceding_air_time_s": (2,),
        }
        for key, trailing_shape in fixed_shapes.items():
            expected = (selected_envs, *trailing_shape)
            if record[key].shape != expected:
                raise PhysicsTouchdownTelemetryError(
                    f"Telemetry field {key!r} must have selected shape {expected}; got {tuple(record[key].shape)}."
                )
        if record["command"].ndim != 2 or record["command"].shape[1] < 3:
            raise PhysicsTouchdownTelemetryError("command must provide at least x, y, and yaw for every environment.")
        if record["actions"].ndim != 2:
            raise PhysicsTouchdownTelemetryError("actions must be a two-dimensional tensor.")
        expected_joint_shape = (selected_envs, len(self._joint_names))
        for key in ("joint_pos", "joint_vel", "applied_torque", "computed_torque"):
            if record[key].shape != expected_joint_shape:
                raise PhysicsTouchdownTelemetryError(
                    f"Telemetry field {key!r} must have shape {expected_joint_shape}; got {tuple(record[key].shape)}."
                )

    def _flush_current_chunk(self) -> None:
        if self._buffer_count == 0:
            return
        for key in self._SERIES_KEYS:
            self._cpu_chunks[key].append(self._buffers[key][: self._buffer_count].detach().to(device="cpu", copy=True))
        self._buffer_count = 0
        self._num_cpu_flushes += 1

    def _append_record(self, record: dict[str, torch.Tensor]) -> None:
        if self._buffer_count == self.chunk_size:
            # A full chunk remains resident until the next sample so pre-reset can
            # still annotate the final row without a per-sample CPU transfer.
            self._flush_current_chunk()
        if not self._buffers:
            for key in self._SERIES_KEYS:
                value = record[key]
                self._specs[key] = (tuple(value.shape), value.dtype)
                self._buffers[key] = torch.empty((self.chunk_size, *value.shape), dtype=value.dtype, device=self.device)
        for key in self._SERIES_KEYS:
            value = record[key]
            expected_shape, expected_dtype = self._specs[key]
            if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
                raise PhysicsTouchdownTelemetryError(
                    f"Telemetry field {key!r} changed shape/dtype from {expected_shape}/{expected_dtype} "
                    f"to {tuple(value.shape)}/{value.dtype}."
                )
            self._buffers[key][self._buffer_count].copy_(value)
        self._buffer_count += 1
        self._num_samples += 1

    def record(
        self,
        *,
        touchdown_sample: Any,
        command: torch.Tensor,
        actions: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        foot_pos_w: torch.Tensor,
        foot_lin_vel_w: torch.Tensor,
        foot_ang_vel_w: torch.Tensor,
        foot_force_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        applied_torque: torch.Tensor,
        computed_torque: torch.Tensor,
    ) -> None:
        """Copy one synchronized all-env sample into the current device chunk."""
        raw_record = {
            "episode_id": touchdown_sample.episode_id,
            "physics_step": touchdown_sample.physics_step,
            "episode_physics_step": touchdown_sample.episode_physics_step,
            "policy_step": touchdown_sample.policy_step,
            "substep": touchdown_sample.substep,
            "physics_time_s": touchdown_sample.physics_time_s,
            "terminated": touchdown_sample.terminated,
            "truncated": touchdown_sample.truncated,
            "reset_after_sample": touchdown_sample.reset_after_sample,
            "command": command,
            "actions": actions,
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "root_lin_vel_w": root_lin_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "foot_pos_w": foot_pos_w,
            "foot_lin_vel_w": foot_lin_vel_w,
            "foot_ang_vel_w": foot_ang_vel_w,
            "foot_force_w": foot_force_w,
            "foot_contact": touchdown_sample.contact,
            "foot_first": touchdown_sample.first_contact,
            "foot_valid": touchdown_sample.valid_touchdown,
            "foot_preimpact_speed": touchdown_sample.preimpact_speed,
            "foot_preceding_air_time_s": touchdown_sample.preceding_air_time_s,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "applied_torque": applied_torque,
            "computed_torque": computed_torque,
        }
        selected_record = {key: self._select(value, key) for key, value in raw_record.items()}
        self._validate_record_shapes(selected_record)
        self._append_record(selected_record)

    def record_from_env(self, env: Any, actions: torch.Tensor, touchdown_sample: Any | None = None) -> None:
        """Read current scene tensors; call only from ``on_post_scene_update``."""
        monitor = getattr(env, self._monitor_attribute, None)
        if monitor is None:
            raise PhysicsTouchdownTelemetryError(f"env.{self._monitor_attribute} is not attached.")
        sample = monitor.latest if touchdown_sample is None else touchdown_sample
        robot = _scene_entity(env.scene, monitor.asset_name)
        sensor = _scene_entity(env.scene, monitor.sensor_name)
        robot_data = robot.data
        self.record(
            touchdown_sample=sample,
            command=env.command_manager.get_command(self._command_name),
            actions=actions,
            root_pos_w=robot_data.root_link_pos_w,
            root_quat_w=robot_data.root_link_quat_w,
            root_lin_vel_w=robot_data.root_link_lin_vel_w,
            root_ang_vel_w=robot_data.root_link_ang_vel_w,
            foot_pos_w=robot_data.body_link_pos_w[:, monitor.asset_foot_ids, :],
            foot_lin_vel_w=robot_data.body_link_lin_vel_w[:, monitor.asset_foot_ids, :],
            foot_ang_vel_w=robot_data.body_link_ang_vel_w[:, monitor.asset_foot_ids, :],
            foot_force_w=sensor.data.net_forces_w[:, monitor.sensor_foot_ids, :],
            joint_pos=robot_data.joint_pos,
            joint_vel=robot_data.joint_vel,
            applied_torque=robot_data.applied_torque,
            computed_torque=robot_data.computed_torque,
        )

    def _selected_boundary(self, value: bool | torch.Tensor, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                raise PhysicsTouchdownTelemetryError(f"{label} must be on {self.device}; got {value.device}.")
            if value.ndim == 0:
                return value.to(dtype=torch.bool).expand(self._env_ids.numel())
            if value.shape == (self.num_envs,):
                return value.to(dtype=torch.bool).index_select(0, self._env_ids)
            if value.shape == (self._env_ids.numel(),):
                return value.to(dtype=torch.bool)
            raise PhysicsTouchdownTelemetryError(
                f"{label} must be scalar, all-env, or selected-env shaped; got {tuple(value.shape)}."
            )
        return torch.full((self._env_ids.numel(),), value, dtype=torch.bool, device=self.device)

    def mark_last_boundary(
        self,
        *,
        terminated: bool | torch.Tensor,
        truncated: bool | torch.Tensor,
        reset_after_sample: bool | torch.Tensor,
    ) -> None:
        """Patch the latest resident row after termination is known and before reset."""
        if self._buffer_count == 0:
            raise PhysicsTouchdownTelemetryError(
                "No resident telemetry row is available for boundary annotation; annotate before the next record/save."
            )
        row = self._buffer_count - 1
        self._buffers["terminated"][row].copy_(self._selected_boundary(terminated, "terminated"))
        self._buffers["truncated"][row].copy_(self._selected_boundary(truncated, "truncated"))
        self._buffers["reset_after_sample"][row].copy_(
            self._selected_boundary(reset_after_sample, "reset_after_sample")
        )

    def mark_last_boundary_from_sample(self, touchdown_sample: Any) -> None:
        self.mark_last_boundary(
            terminated=touchdown_sample.terminated,
            truncated=touchdown_sample.truncated,
            reset_after_sample=touchdown_sample.reset_after_sample,
        )

    def _payload(self) -> dict[str, np.ndarray]:
        if self._num_samples == 0:
            raise PhysicsTouchdownTelemetryError("Cannot save touchdown telemetry before recording a sample.")
        self._flush_current_chunk()
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "schema_name": np.asarray(SCHEMA_NAME),
            "task": np.asarray(self._task),
            "checkpoint": np.asarray(self._checkpoint),
            "command_name": np.asarray(self._command_name),
            "physics_dt_s": np.asarray(self._physics_dt, dtype=np.float64),
            "samples_per_policy": np.asarray(self._samples_per_policy, dtype=np.int64),
            "contact_force_threshold": np.asarray(self._contact_force_threshold, dtype=np.float64),
            "min_air_time_s": np.asarray(self._min_air_time, dtype=np.float64),
            "chunk_size": np.asarray(self.chunk_size, dtype=np.int64),
            "num_samples": np.asarray(self._num_samples, dtype=np.int64),
            "env_ids": self._env_ids.detach().cpu().numpy(),
            "foot_body_names": np.asarray(self._foot_body_names),
            "joint_names": np.asarray(self._joint_names),
        }
        for key in self._SERIES_KEYS:
            payload[key] = torch.cat(self._cpu_chunks[key], dim=0).numpy()
        if any(value.dtype == object for value in payload.values()):
            raise PhysicsTouchdownTelemetryError("NPZ v2 payload must not contain object arrays.")
        return payload

    def save(self, output_path: str | Path) -> Path:
        """Save compressed NPZ v2 arrays that load with ``allow_pickle=False``."""
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output_file:
            np.savez_compressed(output_file, **self._payload())
        return path


class PhysicsTouchdownTelemetryAdapter:
    """Observer adapter registered after the shared touchdown monitor."""

    def __init__(
        self,
        logger: PhysicsTouchdownTelemetryLogger,
        *,
        output_path: str | Path | None = None,
        monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
    ) -> None:
        self.logger = logger
        self.output_path = None if output_path is None else Path(output_path)
        self.monitor_attribute = str(monitor_attribute)
        self._env: Any | None = None
        self._monitor: Any | None = None
        self._closed = False

    def attach(self, env: Any) -> PhysicsTouchdownTelemetryAdapter:
        """Register monitor first and this telemetry observer second."""
        if self._env is not None and self._env is not env:
            raise PhysicsTouchdownTelemetryError("Telemetry adapter is already attached to another environment.")
        monitor = getattr(env, self.monitor_attribute, None)
        if monitor is None:
            raise PhysicsTouchdownTelemetryError(
                f"env.{self.monitor_attribute} is missing; attach the touchdown monitor first."
            )
        add_observer = getattr(env, "add_rnd_physics_observer", None)
        if not callable(add_observer):
            raise PhysicsTouchdownTelemetryError("Telemetry adapter requires env.add_rnd_physics_observer(observer).")
        add_observer(monitor)
        add_observer(self)
        self._env = env
        self._monitor = monitor
        return self

    def _check_env(self, env: Any) -> None:
        if self._env is None or self._monitor is None:
            raise PhysicsTouchdownTelemetryError("Telemetry observer callback arrived before attach().")
        if env is not self._env:
            raise PhysicsTouchdownTelemetryError("Telemetry observer callback came from an unexpected environment.")

    def on_post_scene_update(
        self,
        env: Any,
        action: torch.Tensor,
        policy_step: int,
        substep_index: int,
    ) -> None:
        del policy_step, substep_index
        self._check_env(env)
        self.logger.record_from_env(env, action, self._monitor.latest)

    def on_pre_reset(
        self,
        env: Any,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        del env_ids, terminated, truncated
        self._check_env(env)
        self.logger.mark_last_boundary_from_sample(self._monitor.latest)

    def on_post_reset(self, env: Any, env_ids: torch.Tensor) -> None:
        del env_ids
        self._check_env(env)

    def save(self, output_path: str | Path | None = None) -> Path:
        path = self.output_path if output_path is None else Path(output_path)
        if path is None:
            raise PhysicsTouchdownTelemetryError("No telemetry output path was configured.")
        return self.logger.save(path)

    def close(self) -> None:
        if not self._closed and self.output_path is not None and self.logger.num_samples > 0:
            self.save()
        self._closed = True


def attach_physics_touchdown_telemetry(
    env: Any,
    *,
    output_path: str | Path,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    chunk_size: int = 256,
    task: str = "",
    checkpoint: str | Path = "",
    command_name: str = "base_velocity",
    monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
) -> PhysicsTouchdownTelemetryAdapter:
    """Create and register the telemetry adapter against the shared observer protocol."""
    if getattr(env, monitor_attribute, None) is None:
        from robot_lab.tasks.manager_based.locomotion.velocity.mdp.touchdown import (
            attach_physics_touchdown_monitor,
        )

        attach_physics_touchdown_monitor(env, monitor_attribute=monitor_attribute)
    logger = PhysicsTouchdownTelemetryLogger.from_attached_env(
        env,
        env_ids=env_ids,
        chunk_size=chunk_size,
        task=task,
        checkpoint=checkpoint,
        command_name=command_name,
        monitor_attribute=monitor_attribute,
    )
    return PhysicsTouchdownTelemetryAdapter(
        logger,
        output_path=output_path,
        monitor_attribute=monitor_attribute,
    ).attach(env)
