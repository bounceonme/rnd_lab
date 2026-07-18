"""Physics-rate touchdown detection and a one-shot vertical-impact cost.

The RND STEP environment owns the callback order.  A reward term attaches this
monitor with ``env.add_rnd_physics_observer(monitor)``.  The environment then
calls ``on_post_scene_update`` immediately after every ``scene.update()``,
``on_pre_reset`` after reward computation but before resetting bodies, and
``on_post_reset`` after the selected environments have been reset.

The monitor never uses Isaac Lab's recorder hooks.  Contact state, event state,
and pending reward events remain torch tensors on the simulation device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol, runtime_checkable

import torch

from isaaclab.managers import ManagerTermBase


DEFAULT_FOOT_BODY_NAMES = ("R_Leg_foot", "L_Leg_foot")
DEFAULT_MONITOR_ATTRIBUTE = "physics_touchdown_monitor"
DEFAULT_PHYSICS_DT = 0.005
DEFAULT_MIN_AIR_TIME = 0.06
DEFAULT_SHORT_AIR_TIME_FLOOR = 0.02


class PhysicsTouchdownError(RuntimeError):
    """Raised when touchdown samples or observer wiring are inconsistent."""


@dataclass(frozen=True)
class PhysicsTouchdownEvents:
    """Unconsumed touchdown impact and short-air-time costs."""

    valid: torch.Tensor
    preimpact_speed: torch.Tensor
    short_air_time_cost: torch.Tensor

    def clone(self) -> PhysicsTouchdownEvents:
        return PhysicsTouchdownEvents(
            self.valid.clone(),
            self.preimpact_speed.clone(),
            self.short_air_time_cost.clone(),
        )


@dataclass(frozen=True)
class PhysicsTouchdownSample:
    """Synchronized physics-sample timing, boundary, and touchdown state."""

    episode_id: torch.Tensor
    physics_step: torch.Tensor
    episode_physics_step: torch.Tensor
    policy_step: torch.Tensor
    substep: torch.Tensor
    physics_time_s: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reset_after_sample: torch.Tensor
    contact: torch.Tensor
    first_contact: torch.Tensor
    valid_touchdown: torch.Tensor
    preimpact_speed: torch.Tensor
    preceding_air_time_s: torch.Tensor

    @property
    def first(self) -> torch.Tensor:
        return self.first_contact

    @property
    def valid(self) -> torch.Tensor:
        return self.valid_touchdown

    @property
    def preimpact(self) -> torch.Tensor:
        return self.preimpact_speed

    def clone(self) -> PhysicsTouchdownSample:
        return PhysicsTouchdownSample(**{field.name: getattr(self, field.name).clone() for field in fields(self)})

    def index_select(self, env_ids: torch.Tensor) -> PhysicsTouchdownSample:
        return PhysicsTouchdownSample(
            **{field.name: getattr(self, field.name).index_select(0, env_ids) for field in fields(self)}
        )


@dataclass(frozen=True)
class PreservedPhysicsTouchdownSample:
    """Terminal samples retained across a subsequent per-environment reset."""

    env_ids: torch.Tensor
    sample: PhysicsTouchdownSample


@runtime_checkable
class RndPhysicsObserverHost(Protocol):
    """Structural subset required from the integration environment subclass."""

    num_envs: int
    device: str
    physics_dt: float
    step_dt: float
    scene: Any
    command_manager: Any

    def add_rnd_physics_observer(self, observer: Any) -> None: ...


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
    raise PhysicsTouchdownError(f"Scene entity {name!r} is required by the touchdown monitor.")


def _resolve_two_bodies(entity: Any, body_names: Sequence[str], label: str) -> tuple[tuple[int, int], tuple[str, str]]:
    requested = tuple(str(name) for name in body_names)
    if len(requested) != 2 or len(set(requested)) != 2:
        raise PhysicsTouchdownError(f"{label} foot_body_names must contain exactly two unique names; got {requested}.")
    try:
        body_ids, resolved_names = entity.find_bodies(requested, preserve_order=True)
    except AttributeError as error:
        raise PhysicsTouchdownError(f"Scene entity {label!r} does not expose find_bodies().") from error
    if len(body_ids) != 2 or tuple(resolved_names) != requested:
        raise PhysicsTouchdownError(
            f"{label} must resolve feet in requested order {requested}; got ids={body_ids}, names={resolved_names}."
        )
    return (int(body_ids[0]), int(body_ids[1])), requested


class PhysicsTouchdownMonitor:
    """Detect and retain two-foot touchdown events at the physics update rate.

    ``update`` is the pure device-side API used by focused tests.  Once bound by
    :func:`attach_physics_touchdown_monitor`, the three observer callbacks are
    the runtime API used by ``RndStepManagerBasedRLEnv``.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        physics_dt: float = DEFAULT_PHYSICS_DT,
        samples_per_policy: int = 4,
        force_threshold: float = 1.0,
        min_air_time: float = DEFAULT_MIN_AIR_TIME,
        short_air_time_floor: float = DEFAULT_SHORT_AIR_TIME_FLOOR,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive; got {num_envs}.")
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be finite and positive; got {physics_dt}.")
        if samples_per_policy <= 0:
            raise ValueError(f"samples_per_policy must be positive; got {samples_per_policy}.")
        if not math.isfinite(force_threshold) or force_threshold < 0.0:
            raise ValueError(f"force_threshold must be finite and non-negative; got {force_threshold}.")
        if not math.isfinite(min_air_time) or min_air_time < 0.0:
            raise ValueError(f"min_air_time must be finite and non-negative; got {min_air_time}.")
        if not math.isfinite(short_air_time_floor) or short_air_time_floor < 0.0:
            raise ValueError(
                "short_air_time_floor must be finite and non-negative; "
                f"got {short_air_time_floor}."
            )
        if short_air_time_floor > min_air_time:
            raise ValueError(
                "short_air_time_floor must not exceed min_air_time; "
                f"got {short_air_time_floor} > {min_air_time}."
            )
        if not dtype.is_floating_point:
            raise ValueError(f"Touchdown state dtype must be floating point; got {dtype}.")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.physics_dt = float(physics_dt)
        self.samples_per_policy = int(samples_per_policy)
        self.force_threshold = float(force_threshold)
        self.min_air_time = float(min_air_time)
        self.short_air_time_floor = float(short_air_time_floor)
        self.dtype = dtype

        foot_shape = (self.num_envs, 2)
        self._previous_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._previous_link_vz = torch.zeros(foot_shape, dtype=dtype, device=self.device)
        self._airborne_duration = torch.zeros(foot_shape, dtype=dtype, device=self.device)
        self._pending_valid = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._pending_preimpact = torch.zeros(foot_shape, dtype=dtype, device=self.device)
        self._pending_short_air_time_cost = torch.zeros(foot_shape, dtype=dtype, device=self.device)
        self._episode_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._episode_physics_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._next_physics_step = 0

        self._latest: PhysicsTouchdownSample | None = None
        self._preserved_terminal: PhysicsTouchdownSample | None = None
        self._preserved_terminal_available = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._bound_env: Any | None = None
        self._asset_name = "robot"
        self._sensor_name = "contact_forces"
        self._asset_foot_ids: tuple[int, int] | None = None
        self._sensor_foot_ids: tuple[int, int] | None = None
        self._foot_body_names: tuple[str, str] | None = None

    @property
    def latest(self) -> PhysicsTouchdownSample:
        if self._latest is None:
            raise PhysicsTouchdownError("No touchdown physics sample has been observed yet.")
        return self._latest

    @property
    def episode_id(self) -> torch.Tensor:
        return self._episode_id.clone()

    @property
    def airborne_duration(self) -> torch.Tensor:
        return self._airborne_duration.clone()

    @property
    def foot_body_names(self) -> tuple[str, str]:
        if self._foot_body_names is None:
            raise PhysicsTouchdownError("Touchdown monitor is not bound to scene foot bodies.")
        return self._foot_body_names

    @property
    def asset_foot_ids(self) -> tuple[int, int]:
        if self._asset_foot_ids is None:
            raise PhysicsTouchdownError("Touchdown monitor is not bound to an articulation.")
        return self._asset_foot_ids

    @property
    def sensor_foot_ids(self) -> tuple[int, int]:
        if self._sensor_foot_ids is None:
            raise PhysicsTouchdownError("Touchdown monitor is not bound to a contact sensor.")
        return self._sensor_foot_ids

    @property
    def asset_name(self) -> str:
        return self._asset_name

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.device != self.device:
                raise PhysicsTouchdownError(
                    f"env_ids must remain on monitor device {self.device}; got {env_ids.device}."
                )
            ids = env_ids.to(dtype=torch.int64)
        else:
            ids = torch.as_tensor(tuple(env_ids), dtype=torch.int64, device=self.device)
        if ids.ndim != 1:
            raise PhysicsTouchdownError(f"env_ids must be one-dimensional; got shape {tuple(ids.shape)}.")
        if ids.numel() == 0:
            return ids
        if bool(torch.any((ids < 0) | (ids >= self.num_envs))):
            raise PhysicsTouchdownError(f"env_ids contains an index outside [0, {self.num_envs}).")
        if torch.unique(ids).numel() != ids.numel():
            raise PhysicsTouchdownError("env_ids must not contain duplicates.")
        return ids

    def _per_env_tensor(self, value: int | bool | torch.Tensor, *, dtype: torch.dtype, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                raise PhysicsTouchdownError(f"{label} must be on {self.device}; got {value.device}.")
            if value.ndim == 0:
                return value.to(dtype=dtype).expand(self.num_envs).clone()
            if value.shape != (self.num_envs,):
                raise PhysicsTouchdownError(
                    f"{label} must be scalar or shape ({self.num_envs},); got {tuple(value.shape)}."
                )
            return value.to(dtype=dtype).clone()
        return torch.full((self.num_envs,), value, dtype=dtype, device=self.device)

    def _validate_sample_inputs(self, net_forces_w: torch.Tensor, foot_link_lin_vel_w: torch.Tensor) -> torch.Tensor:
        if not isinstance(net_forces_w, torch.Tensor) or not isinstance(foot_link_lin_vel_w, torch.Tensor):
            raise TypeError("net_forces_w and foot_link_lin_vel_w must be torch tensors.")
        if net_forces_w.device != self.device or foot_link_lin_vel_w.device != self.device:
            raise PhysicsTouchdownError(
                f"Physics inputs must remain on monitor device {self.device}; got "
                f"force={net_forces_w.device}, velocity={foot_link_lin_vel_w.device}."
            )
        if net_forces_w.shape != (self.num_envs, 2, 3):
            raise PhysicsTouchdownError(
                f"net_forces_w must have shape ({self.num_envs}, 2, 3); got {tuple(net_forces_w.shape)}."
            )
        if foot_link_lin_vel_w.shape == (self.num_envs, 2):
            link_vz = foot_link_lin_vel_w
        elif foot_link_lin_vel_w.shape == (self.num_envs, 2, 3):
            link_vz = foot_link_lin_vel_w[..., 2]
        else:
            raise PhysicsTouchdownError(
                "foot_link_lin_vel_w must contain either two vertical velocities or two xyz velocities per env; "
                f"got {tuple(foot_link_lin_vel_w.shape)}."
            )
        if not net_forces_w.dtype.is_floating_point or not link_vz.dtype.is_floating_point:
            raise PhysicsTouchdownError("Physics force and velocity inputs must use floating-point dtypes.")
        if link_vz.dtype != self.dtype:
            raise PhysicsTouchdownError(f"Velocity dtype must match monitor dtype {self.dtype}; got {link_vz.dtype}.")
        return link_vz

    def update(
        self,
        net_forces_w: torch.Tensor,
        foot_link_lin_vel_w: torch.Tensor,
        *,
        physics_step: int | torch.Tensor | None = None,
        policy_step: int | torch.Tensor | None = None,
        substep_index: int | torch.Tensor | None = None,
    ) -> PhysicsTouchdownSample:
        """Process one post-``scene.update`` sample without leaving the torch device."""
        link_vz = self._validate_sample_inputs(net_forces_w, foot_link_lin_vel_w)
        default_physics_step = self._next_physics_step
        physics = self._per_env_tensor(
            default_physics_step if physics_step is None else physics_step,
            dtype=torch.int64,
            label="physics_step",
        )
        policy = self._per_env_tensor(
            default_physics_step // self.samples_per_policy if policy_step is None else policy_step,
            dtype=torch.int64,
            label="policy_step",
        )
        raw_substep = default_physics_step % self.samples_per_policy if substep_index is None else substep_index
        if isinstance(raw_substep, int) and not 0 <= raw_substep < self.samples_per_policy:
            raise PhysicsTouchdownError(f"substep_index must be in [0, {self.samples_per_policy}); got {raw_substep}.")
        substep = self._per_env_tensor(
            raw_substep,
            dtype=torch.int64,
            label="substep_index",
        )

        contact = torch.linalg.vector_norm(net_forces_w, dim=-1) > self.force_threshold
        first_contact = contact & ~self._previous_contact
        preceding_air_time = self._airborne_duration.clone()
        tolerance = max(torch.finfo(self.dtype).eps * 8.0, 1.0e-9)
        valid_touchdown = first_contact & (preceding_air_time + tolerance >= self.min_air_time)
        preimpact_speed = torch.where(first_contact, torch.clamp(-self._previous_link_vz, min=0.0), 0.0)

        short_air_time_span = self.min_air_time - self.short_air_time_floor
        if short_air_time_span > tolerance:
            short_touchdown = (
                first_contact
                & (preceding_air_time + tolerance >= self.short_air_time_floor)
                & (preceding_air_time + tolerance < self.min_air_time)
            )
            short_air_time_cost = torch.where(
                short_touchdown,
                torch.clamp(
                    (self.min_air_time - preceding_air_time) / short_air_time_span,
                    min=0.0,
                    max=1.0,
                ),
                0.0,
            )
            # At most one cost per foot is retained within a policy step. This rejects
            # repeated contact chatter without hiding a genuine 20-60 ms tap.
            self._pending_short_air_time_cost = torch.maximum(
                self._pending_short_air_time_cost,
                short_air_time_cost,
            )

        self._pending_preimpact = torch.where(valid_touchdown, preimpact_speed, self._pending_preimpact)
        self._pending_valid |= valid_touchdown

        false_boundary = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sample = PhysicsTouchdownSample(
            episode_id=self._episode_id.clone(),
            physics_step=physics,
            episode_physics_step=self._episode_physics_step.clone(),
            policy_step=policy,
            substep=substep,
            physics_time_s=(physics.to(dtype=self.dtype) + 1.0) * self.physics_dt,
            terminated=false_boundary.clone(),
            truncated=false_boundary.clone(),
            reset_after_sample=false_boundary,
            contact=contact,
            first_contact=first_contact,
            valid_touchdown=valid_touchdown,
            preimpact_speed=preimpact_speed,
            preceding_air_time_s=preceding_air_time,
        )
        self._latest = sample

        self._previous_contact.copy_(contact)
        self._previous_link_vz.copy_(link_vz)
        self._airborne_duration = torch.where(
            contact,
            torch.zeros_like(self._airborne_duration),
            self._airborne_duration + self.physics_dt,
        )
        self._episode_physics_step += 1
        self._next_physics_step = default_physics_step + 1
        return sample

    def peek_pending(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> PhysicsTouchdownEvents:
        ids = self._normalize_env_ids(env_ids)
        return PhysicsTouchdownEvents(
            valid=self._pending_valid.index_select(0, ids).clone(),
            preimpact_speed=self._pending_preimpact.index_select(0, ids).clone(),
            short_air_time_cost=self._pending_short_air_time_cost.index_select(0, ids).clone(),
        )

    def clear_pending(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = self._normalize_env_ids(env_ids)
        self._pending_valid[ids] = False
        self._pending_preimpact[ids] = 0.0
        self._pending_short_air_time_cost[ids] = 0.0

    def consume_pending(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> PhysicsTouchdownEvents:
        """Return each pending valid touchdown once, then clear only those environments."""
        ids = self._normalize_env_ids(env_ids)
        events = PhysicsTouchdownEvents(
            valid=self._pending_valid.index_select(0, ids).clone(),
            preimpact_speed=self._pending_preimpact.index_select(0, ids).clone(),
            short_air_time_cost=self._pending_short_air_time_cost.index_select(0, ids).clone(),
        )
        self._pending_valid[ids] = False
        self._pending_preimpact[ids] = 0.0
        self._pending_short_air_time_cost[ids] = 0.0
        return events

    def _copy_sample_rows(
        self, destination: PhysicsTouchdownSample, source: PhysicsTouchdownSample, mask: torch.Tensor
    ) -> PhysicsTouchdownSample:
        copied: dict[str, torch.Tensor] = {}
        for field in fields(destination):
            value = getattr(destination, field.name).clone()
            value[mask] = getattr(source, field.name)[mask]
            copied[field.name] = value
        return PhysicsTouchdownSample(**copied)

    def mark_policy_boundary(
        self,
        *,
        terminated: bool | torch.Tensor,
        truncated: bool | torch.Tensor,
        reset_after_sample: bool | torch.Tensor | None = None,
    ) -> PhysicsTouchdownSample:
        """Annotate and preserve the latest sample before any selected reset occurs."""
        latest = self.latest
        terminated_tensor = self._per_env_tensor(terminated, dtype=torch.bool, label="terminated")
        truncated_tensor = self._per_env_tensor(truncated, dtype=torch.bool, label="truncated")
        if reset_after_sample is None:
            reset_tensor = terminated_tensor | truncated_tensor
        else:
            reset_tensor = self._per_env_tensor(reset_after_sample, dtype=torch.bool, label="reset_after_sample")
        annotated = replace(
            latest,
            terminated=terminated_tensor,
            truncated=truncated_tensor,
            reset_after_sample=reset_tensor,
        )
        self._latest = annotated

        preserve_mask = terminated_tensor | truncated_tensor | reset_tensor
        if bool(torch.any(preserve_mask)):
            if self._preserved_terminal is None:
                self._preserved_terminal = annotated.clone()
            else:
                self._preserved_terminal = self._copy_sample_rows(self._preserved_terminal, annotated, preserve_mask)
            self._preserved_terminal_available |= preserve_mask
        return annotated.clone()

    def preserve_terminal_before_reset(
        self,
        *,
        terminated: bool | torch.Tensor,
        truncated: bool | torch.Tensor,
        reset_after_sample: bool | torch.Tensor | None = None,
    ) -> PhysicsTouchdownSample:
        """Named alias documenting the required pre-reset ordering."""
        return self.mark_policy_boundary(
            terminated=terminated,
            truncated=truncated,
            reset_after_sample=reset_after_sample,
        )

    def take_preserved_terminal(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        clear: bool = False,
    ) -> PreservedPhysicsTouchdownSample:
        """Read terminal samples even after ``on_post_reset`` has reset live state."""
        if self._preserved_terminal is None:
            raise PhysicsTouchdownError("No terminal touchdown sample has been preserved.")
        if env_ids is None:
            ids = torch.nonzero(self._preserved_terminal_available, as_tuple=False).flatten()
        else:
            ids = self._normalize_env_ids(env_ids)
        if ids.numel() == 0:
            raise PhysicsTouchdownError("No preserved terminal samples are available for the requested environments.")
        if bool(torch.any(~self._preserved_terminal_available.index_select(0, ids))):
            raise PhysicsTouchdownError("At least one requested environment has no preserved terminal sample.")
        result = PreservedPhysicsTouchdownSample(ids.clone(), self._preserved_terminal.index_select(ids))
        if clear:
            self._preserved_terminal_available[ids] = False
        return result

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        increment_episode: bool = True,
    ) -> None:
        """Reset live state for selected environments without erasing terminal snapshots."""
        ids = self._normalize_env_ids(env_ids)
        self._previous_contact[ids] = False
        self._previous_link_vz[ids] = 0.0
        self._airborne_duration[ids] = 0.0
        self._pending_valid[ids] = False
        self._pending_preimpact[ids] = 0.0
        self._pending_short_air_time_cost[ids] = 0.0
        self._episode_physics_step[ids] = 0
        if increment_episode:
            self._episode_id[ids] += 1

    def bind_scene(
        self,
        env: RndPhysicsObserverHost,
        *,
        asset_name: str,
        sensor_name: str,
        foot_body_names: Sequence[str],
    ) -> None:
        """Resolve ordered articulation and contact-sensor body IDs once."""
        if self._bound_env is not None and self._bound_env is not env:
            raise PhysicsTouchdownError("A touchdown monitor cannot be rebound to another environment.")
        asset = _scene_entity(env.scene, asset_name)
        sensor = _scene_entity(env.scene, sensor_name)
        asset_ids, resolved_names = _resolve_two_bodies(asset, foot_body_names, asset_name)
        sensor_ids, _ = _resolve_two_bodies(sensor, resolved_names, sensor_name)
        self._bound_env = env
        self._asset_name = str(asset_name)
        self._sensor_name = str(sensor_name)
        self._asset_foot_ids = asset_ids
        self._sensor_foot_ids = sensor_ids
        self._foot_body_names = resolved_names

    def validate_attachment(
        self,
        env: RndPhysicsObserverHost,
        *,
        asset_name: str,
        sensor_name: str,
        foot_body_names: Sequence[str],
        force_threshold: float,
        min_air_time: float,
        short_air_time_floor: float,
        samples_per_policy: int,
    ) -> None:
        """Reject a second reward term that conflicts with the shared monitor."""
        expected_names = tuple(str(name) for name in foot_body_names)
        compatible = (
            self._bound_env is env
            and self._asset_name == asset_name
            and self._sensor_name == sensor_name
            and self._foot_body_names == expected_names
            and self.num_envs == int(env.num_envs)
            and self.device == torch.device(env.device)
            and self.samples_per_policy == samples_per_policy
            and math.isclose(self.physics_dt, float(env.physics_dt), rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(self.force_threshold, force_threshold, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(self.min_air_time, min_air_time, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(
                self.short_air_time_floor,
                short_air_time_floor,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        if not compatible:
            raise PhysicsTouchdownError("A conflicting PhysicsTouchdownMonitor is already attached to the environment.")

    def _require_bound_callback_env(self, env: Any) -> None:
        if self._bound_env is None:
            raise PhysicsTouchdownError("Touchdown observer callback arrived before bind_scene().")
        if env is not self._bound_env:
            raise PhysicsTouchdownError("Touchdown observer callback came from an unexpected environment.")

    def on_post_scene_update(
        self,
        env: Any,
        action: torch.Tensor,
        policy_step: int,
        substep_index: int,
    ) -> None:
        """Observer callback: sample current scene buffers at 200 Hz."""
        del action
        self._require_bound_callback_env(env)
        asset = _scene_entity(env.scene, self._asset_name)
        sensor = _scene_entity(env.scene, self._sensor_name)
        self.update(
            sensor.data.net_forces_w[:, self.sensor_foot_ids, :],
            asset.data.body_link_lin_vel_w[:, self.asset_foot_ids, :],
            physics_step=policy_step * self.samples_per_policy + substep_index,
            policy_step=policy_step,
            substep_index=substep_index,
        )

    def on_pre_reset(
        self,
        env: Any,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Observer callback: preserve the final sample before simulator reset."""
        self._require_bound_callback_env(env)
        ids = self._normalize_env_ids(env_ids)
        reset_after_sample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_after_sample[ids] = True
        terminated_tensor = self._per_env_tensor(terminated, dtype=torch.bool, label="terminated")
        truncated_tensor = self._per_env_tensor(truncated, dtype=torch.bool, label="truncated")
        self.mark_policy_boundary(
            terminated=terminated_tensor & reset_after_sample,
            truncated=truncated_tensor & reset_after_sample,
            reset_after_sample=reset_after_sample,
        )

    def on_post_reset(self, env: Any, env_ids: torch.Tensor) -> None:
        """Observer callback: isolate live-state reset to the selected environments."""
        self._require_bound_callback_env(env)
        self.reset(env_ids)

    def close(self) -> None:
        """Observer close hook; monitor tensors are owned by torch and need no teardown."""


def _samples_per_policy(env: Any, physics_dt: float) -> int:
    configured = getattr(getattr(env, "cfg", None), "decimation", None)
    if configured is not None:
        samples = int(configured)
    else:
        ratio = float(env.step_dt) / physics_dt
        samples = int(round(ratio))
        if not math.isclose(ratio, samples, rel_tol=0.0, abs_tol=1.0e-9):
            raise PhysicsTouchdownError(
                f"env.step_dt / env.physics_dt must be integral; got {env.step_dt} / {physics_dt}."
            )
    if samples <= 0:
        raise PhysicsTouchdownError(f"Environment decimation must be positive; got {samples}.")
    return samples


def attach_physics_touchdown_monitor(
    env: RndPhysicsObserverHost,
    *,
    foot_body_names: Sequence[str] = DEFAULT_FOOT_BODY_NAMES,
    asset_name: str = "robot",
    sensor_name: str = "contact_forces",
    force_threshold: float | None = None,
    min_air_time: float = DEFAULT_MIN_AIR_TIME,
    short_air_time_floor: float = DEFAULT_SHORT_AIR_TIME_FLOOR,
    monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
) -> PhysicsTouchdownMonitor:
    """Create, bind, expose, and register the environment's single monitor."""
    add_observer = getattr(env, "add_rnd_physics_observer", None)
    if not callable(add_observer):
        raise PhysicsTouchdownError(
            "PhysicsTouchdownMonitor requires an environment subclass exposing add_rnd_physics_observer(observer)."
        )
    physics_dt = float(env.physics_dt)
    if not math.isclose(physics_dt, DEFAULT_PHYSICS_DT, rel_tol=0.0, abs_tol=1.0e-12):
        raise PhysicsTouchdownError(
            f"Touchdown monitor requires 200 Hz physics_dt={DEFAULT_PHYSICS_DT}; got {physics_dt}."
        )
    samples_per_policy = _samples_per_policy(env, physics_dt)
    if samples_per_policy != 4 or not math.isclose(
        float(env.step_dt), physics_dt * samples_per_policy, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise PhysicsTouchdownError(
            "Touchdown monitor requires four 5 ms physics samples per 20 ms policy step; "
            f"got samples_per_policy={samples_per_policy}, step_dt={env.step_dt}."
        )
    sensor = _scene_entity(env.scene, sensor_name)
    threshold = float(sensor.cfg.force_threshold) if force_threshold is None else float(force_threshold)
    asset = _scene_entity(env.scene, asset_name)
    velocity = asset.data.body_link_lin_vel_w
    existing = getattr(env, monitor_attribute, None)
    if existing is not None:
        if not isinstance(existing, PhysicsTouchdownMonitor):
            raise PhysicsTouchdownError(f"env.{monitor_attribute} already exists but is not a PhysicsTouchdownMonitor.")
        existing.validate_attachment(
            env,
            asset_name=asset_name,
            sensor_name=sensor_name,
            foot_body_names=foot_body_names,
            force_threshold=threshold,
            min_air_time=min_air_time,
            short_air_time_floor=short_air_time_floor,
            samples_per_policy=samples_per_policy,
        )
        add_observer(existing)
        return existing
    monitor = PhysicsTouchdownMonitor(
        num_envs=int(env.num_envs),
        device=env.device,
        physics_dt=physics_dt,
        samples_per_policy=samples_per_policy,
        force_threshold=threshold,
        min_air_time=min_air_time,
        short_air_time_floor=short_air_time_floor,
        dtype=velocity.dtype,
    )
    monitor.bind_scene(
        env,
        asset_name=asset_name,
        sensor_name=sensor_name,
        foot_body_names=foot_body_names,
    )
    setattr(env, monitor_attribute, monitor)
    add_observer(monitor)
    return monitor


def get_physics_touchdown_monitor(
    env: Any, monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE
) -> PhysicsTouchdownMonitor:
    monitor = getattr(env, monitor_attribute, None)
    if not isinstance(monitor, PhysicsTouchdownMonitor):
        raise PhysicsTouchdownError(
            f"env.{monitor_attribute} is not attached; configure PhysicsTouchdownImpactCost first."
        )
    return monitor


def consume_touchdown_impact_cost(
    monitor: PhysicsTouchdownMonitor,
    command: torch.Tensor,
    *,
    command_xy_threshold: float = 0.10,
    command_yaw_threshold: float = 0.15,
    impact_speed_offset: float = 0.25,
    impact_speed_range: float = 0.50,
    short_air_time_penalty_scale: float = 0.0,
) -> torch.Tensor:
    """Consume pending events once and return impact plus short-air-time cost."""
    if command.device != monitor.device:
        raise PhysicsTouchdownError(f"command must be on monitor device {monitor.device}; got {command.device}.")
    if command.ndim != 2 or command.shape[0] != monitor.num_envs or command.shape[1] < 3:
        raise PhysicsTouchdownError(f"command must have shape ({monitor.num_envs}, >=3); got {tuple(command.shape)}.")
    if impact_speed_range <= 0.0:
        raise ValueError(f"impact_speed_range must be positive; got {impact_speed_range}.")
    if not math.isfinite(short_air_time_penalty_scale) or short_air_time_penalty_scale < 0.0:
        raise ValueError(
            "short_air_time_penalty_scale must be finite and non-negative; "
            f"got {short_air_time_penalty_scale}."
        )
    events = monitor.consume_pending()
    normalized = torch.clamp(
        (events.preimpact_speed - impact_speed_offset) / impact_speed_range,
        min=0.0,
        max=1.0,
    )
    active = (torch.linalg.vector_norm(command[:, :2], dim=1) > command_xy_threshold) | (
        torch.abs(command[:, 2]) > command_yaw_threshold
    )
    impact_cost = torch.sum(normalized * events.valid.to(dtype=normalized.dtype), dim=1)
    short_air_time_cost = torch.sum(events.short_air_time_cost, dim=1)
    total_cost = impact_cost + short_air_time_penalty_scale * short_air_time_cost
    return total_cost * active.to(dtype=normalized.dtype)


def physics_touchdown_impact_cost(
    env: Any,
    command_name: str,
    command_xy_threshold: float = 0.10,
    command_yaw_threshold: float = 0.15,
    impact_speed_offset: float = 0.25,
    impact_speed_range: float = 0.50,
    short_air_time_penalty_scale: float = 0.0,
    monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
) -> torch.Tensor:
    """Functional helper for callers that already attached the monitor."""
    monitor = get_physics_touchdown_monitor(env, monitor_attribute)
    return consume_touchdown_impact_cost(
        monitor,
        env.command_manager.get_command(command_name),
        command_xy_threshold=command_xy_threshold,
        command_yaw_threshold=command_yaw_threshold,
        impact_speed_offset=impact_speed_offset,
        impact_speed_range=impact_speed_range,
        short_air_time_penalty_scale=short_air_time_penalty_scale,
    )


class PhysicsTouchdownImpactCost(ManagerTermBase):
    """Manager reward term that owns observer registration and one-shot consumption.

    The returned value is a positive cost.  Reward configuration is expected to
    apply the requested negative weight (for example ``weight=-0.5``).
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self._monitor_attribute = str(params.get("monitor_attribute", DEFAULT_MONITOR_ATTRIBUTE))
        self.monitor = attach_physics_touchdown_monitor(
            env,
            foot_body_names=params.get("foot_body_names", DEFAULT_FOOT_BODY_NAMES),
            asset_name=str(params.get("asset_name", "robot")),
            sensor_name=str(params.get("sensor_name", "contact_forces")),
            force_threshold=params.get("force_threshold"),
            min_air_time=float(params.get("min_air_time", DEFAULT_MIN_AIR_TIME)),
            short_air_time_floor=float(
                params.get("short_air_time_floor", DEFAULT_SHORT_AIR_TIME_FLOOR)
            ),
            monitor_attribute=self._monitor_attribute,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        # The environment's on_post_reset callback is the sole owner of live monitor reset.
        del env_ids

    def __call__(
        self,
        env: Any,
        command_name: str,
        command_xy_threshold: float = 0.10,
        command_yaw_threshold: float = 0.15,
        impact_speed_offset: float = 0.25,
        impact_speed_range: float = 0.50,
        short_air_time_penalty_scale: float = 0.0,
        foot_body_names: Sequence[str] = DEFAULT_FOOT_BODY_NAMES,
        asset_name: str = "robot",
        sensor_name: str = "contact_forces",
        force_threshold: float | None = None,
        min_air_time: float = DEFAULT_MIN_AIR_TIME,
        short_air_time_floor: float = DEFAULT_SHORT_AIR_TIME_FLOOR,
        monitor_attribute: str = DEFAULT_MONITOR_ATTRIBUTE,
    ) -> torch.Tensor:
        del foot_body_names, asset_name, sensor_name, force_threshold, min_air_time, short_air_time_floor
        if monitor_attribute != self._monitor_attribute:
            raise PhysicsTouchdownError("monitor_attribute cannot change after reward-term initialization.")
        return physics_touchdown_impact_cost(
            env,
            command_name,
            command_xy_threshold=command_xy_threshold,
            command_yaw_threshold=command_yaw_threshold,
            impact_speed_offset=impact_speed_offset,
            impact_speed_range=impact_speed_range,
            short_air_time_penalty_scale=short_air_time_penalty_scale,
            monitor_attribute=monitor_attribute,
        )


TouchdownImpactCost = PhysicsTouchdownImpactCost
