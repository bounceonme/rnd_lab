"""Deterministic finite-suite runtime helpers for RSL-RL evaluation.

The module intentionally does not import Isaac Lab.  The runtime objects use the
public tensor/view interfaces exposed by an initialized Isaac articulation, while
the pure scheduling, hashing, and aggregation pieces remain CPU-testable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


COMMAND_NAME = "base_velocity"
CASE_DURATION_S = 20.0
PULSE_DURATION_S = 0.12
PULSE_PHYSICS_TICKS = 24
PULSE_POLICY_STEPS = 6
ENCODER_JOINT_COUNT = 12
ENCODER_ZERO_OFFSET_RANGE_RAD = (-0.005, 0.005)
ENCODER_SAMPLE_AGE_RANGE_S = (0.0, 0.005)


class EvaluationRuntimeError(RuntimeError):
    """Raised when deterministic evaluation cannot be guaranteed."""


class CommandScheduleError(EvaluationRuntimeError):
    """Raised for an invalid command schedule or command-term adapter."""


class PhysicalPulseError(EvaluationRuntimeError):
    """Raised for an invalid or conflicting physical pulse."""


class FixedDomainError(EvaluationRuntimeError):
    """Raised when a fixed domain cannot be applied and read back exactly."""


class EvaluationArtifactError(EvaluationRuntimeError):
    """Raised when evaluation artifacts are incomplete or unsafe to serialize."""


def joint_order_permutation(
    runtime_joint_names: Sequence[str], metric_joint_names: Sequence[str]
) -> np.ndarray:
    """Return indices that reorder runtime joint arrays into the metric contract order."""

    runtime = tuple(str(name) for name in runtime_joint_names)
    metric = tuple(str(name) for name in metric_joint_names)
    if not runtime or not metric:
        raise EvaluationRuntimeError("Runtime and metric joint orders must be non-empty.")
    if len(runtime) != len(set(runtime)):
        raise EvaluationRuntimeError("Runtime joint order contains duplicate names.")
    if len(metric) != len(set(metric)):
        raise EvaluationRuntimeError("Metric joint order contains duplicate names.")
    if set(runtime) != set(metric):
        missing = sorted(set(metric) - set(runtime))
        unexpected = sorted(set(runtime) - set(metric))
        raise EvaluationRuntimeError(
            f"Runtime and metric joint orders do not match; missing={missing}, unexpected={unexpected}."
        )
    runtime_index = {name: index for index, name in enumerate(runtime)}
    return np.asarray([runtime_index[name] for name in metric], dtype=np.int64)


def _finite_scalar(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FixedDomainError(f"{label} must be one finite scalar; ranges and containers are not allowed.")
    result = float(value)
    if not math.isfinite(result):
        raise FixedDomainError(f"{label} must be finite; got {result!r}.")
    if minimum is not None and result < minimum:
        raise FixedDomainError(f"{label} must be at least {minimum}; got {result}.")
    return result


def _strict_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FixedDomainError(f"{label} must be a string-keyed mapping.")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise FixedDomainError(f"{label} contains unsupported fields: {unknown}.")


def _optional_scalar(mapping: Mapping[str, Any], key: str, label: str, *, minimum: float | None = None):
    if key not in mapping or mapping[key] is None:
        return None
    return _finite_scalar(mapping[key], f"{label}.{key}", minimum=minimum)


def _fixed_vector(
    value: Any,
    label: str,
    *,
    length: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise FixedDomainError(f"{label} must contain exactly {length} scalar values.")
    result = tuple(_finite_scalar(item, f"{label}[{index}]") for index, item in enumerate(value))
    if minimum is not None and any(item < minimum for item in result):
        raise FixedDomainError(f"{label} values must be at least {minimum}.")
    if maximum is not None and any(item > maximum for item in result):
        raise FixedDomainError(f"{label} values must be at most {maximum}.")
    return result


@dataclass(frozen=True)
class CommandPhase:
    """One half-open command phase in policy time."""

    name: str
    start_s: float
    end_s: float
    target_b: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise CommandScheduleError("Command phase names must be non-empty.")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.start_s < 0.0:
            raise CommandScheduleError(f"Invalid phase interval [{self.start_s}, {self.end_s}).")
        if self.end_s <= self.start_s:
            raise CommandScheduleError(f"Phase {self.name!r} must have positive duration.")
        if len(self.target_b) != 3 or not all(math.isfinite(float(value)) for value in self.target_b):
            raise CommandScheduleError(f"Phase {self.name!r} target must contain three finite values.")
        if float(self.target_b[0]) != 0.0:
            raise CommandScheduleError(
                f"Phase {self.name!r} violates the STEP command contract: body x must be exactly zero."
            )


@dataclass(frozen=True)
class CommandSchedule:
    """Validated, gap-free command schedule with deterministic phase boundaries."""

    name: str
    phases: tuple[CommandPhase, ...]
    duration_s: float = CASE_DURATION_S

    def __post_init__(self) -> None:
        if not self.name or not self.phases:
            raise CommandScheduleError("A command schedule requires a name and at least one phase.")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise CommandScheduleError("Command schedule duration must be finite and positive.")
        cursor = 0.0
        for phase in self.phases:
            if not math.isclose(phase.start_s, cursor, rel_tol=0.0, abs_tol=1.0e-12):
                raise CommandScheduleError(
                    f"Schedule {self.name!r} has a gap or overlap before phase {phase.name!r}: "
                    f"expected {cursor}, got {phase.start_s}."
                )
            cursor = phase.end_s
        if not math.isclose(cursor, self.duration_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise CommandScheduleError(f"Schedule {self.name!r} ends at {cursor}s instead of {self.duration_s}s.")

    def phase_at(self, time_s: float) -> CommandPhase:
        """Return the phase active at ``time_s`` using half-open boundaries."""

        value = float(time_s)
        if not math.isfinite(value) or value < 0.0 or value > self.duration_s + 1.0e-12:
            raise CommandScheduleError(f"Schedule time must lie in [0, {self.duration_s}]; got {time_s!r}.")
        if math.isclose(value, self.duration_s, rel_tol=0.0, abs_tol=1.0e-12):
            return self.phases[-1]
        for phase in self.phases:
            if phase.start_s <= value < phase.end_s:
                return phase
        raise CommandScheduleError(f"No command phase resolved at t={value}.")

    def target_at(self, time_s: float) -> tuple[float, float, float]:
        return self.phase_at(time_s).target_b


def stand_start_stop_schedule(straight_velocity_y: float) -> CommandSchedule:
    """Build stand/start/stop: 0-2, 2-8, and 8-20 seconds."""

    speed = float(straight_velocity_y)
    if not math.isfinite(speed) or speed == 0.0:
        raise CommandScheduleError("straight_velocity_y must be finite and non-zero.")
    return CommandSchedule(
        name="stand-start-stop",
        phases=(
            CommandPhase("stand", 0.0, 2.0, (0.0, 0.0, 0.0)),
            CommandPhase("straight", 2.0, 8.0, (0.0, speed, 0.0)),
            CommandPhase("stop", 8.0, 20.0, (0.0, 0.0, 0.0)),
        ),
    )


def straight_turn_straight_stop_schedule(straight_velocity_y: float, turn_rate_z: float) -> CommandSchedule:
    """Build stand/straight/turn/straight/stop at the required boundaries."""

    speed = float(straight_velocity_y)
    yaw_rate = float(turn_rate_z)
    if not math.isfinite(speed) or speed == 0.0:
        raise CommandScheduleError("straight_velocity_y must be finite and non-zero.")
    if not math.isfinite(yaw_rate) or yaw_rate == 0.0:
        raise CommandScheduleError("turn_rate_z must be finite and non-zero.")
    return CommandSchedule(
        name="straight-turn-straight-stop",
        phases=(
            CommandPhase("stand", 0.0, 2.0, (0.0, 0.0, 0.0)),
            CommandPhase("straight-1", 2.0, 6.0, (0.0, speed, 0.0)),
            CommandPhase("turn", 6.0, 10.0, (0.0, speed, yaw_rate)),
            CommandPhase("straight-2", 10.0, 14.0, (0.0, speed, 0.0)),
            CommandPhase("stop", 14.0, 20.0, (0.0, 0.0, 0.0)),
        ),
    )


def build_command_schedule(profile: str, *, straight_velocity_y: float, turn_rate_z: float | None = None):
    if profile == "stand-start-stop":
        if turn_rate_z not in (None, 0.0):
            raise CommandScheduleError("stand-start-stop does not accept a non-zero turn rate.")
        return stand_start_stop_schedule(straight_velocity_y)
    if profile == "straight-turn-straight-stop":
        if turn_rate_z is None:
            raise CommandScheduleError("straight-turn-straight-stop requires turn_rate_z.")
        return straight_turn_straight_stop_schedule(straight_velocity_y, turn_rate_z)
    raise CommandScheduleError(f"Unsupported command profile: {profile!r}.")


def command_schedule_from_scenario(scenario: Mapping[str, Any], policy_hz: float) -> CommandSchedule:
    """Build a schedule from generic, contiguous suite segments."""

    if not isinstance(scenario, Mapping):
        raise CommandScheduleError("Scenario must be a mapping.")
    rate = float(policy_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise CommandScheduleError("policy_hz must be finite and positive.")
    horizon = scenario.get("horizon_steps")
    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or int(horizon) <= 0:
        raise CommandScheduleError("Scenario horizon_steps must be a positive integer.")
    horizon = int(horizon)
    duration_s = horizon / rate
    if not math.isclose(duration_s, CASE_DURATION_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise CommandScheduleError(f"Scenario must span exactly {CASE_DURATION_S}s; got {horizon} steps at {rate} Hz.")
    segments = scenario.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        raise CommandScheduleError("Scenario segments must be a non-empty sequence.")

    phases: list[CommandPhase] = []
    cursor = 0
    command_axes = ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s")
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, Mapping):
            raise CommandScheduleError(f"Scenario segment {index} must be a mapping.")
        start = raw_segment.get("start_step")
        end = raw_segment.get("end_step")
        if (
            isinstance(start, bool)
            or not isinstance(start, numbers.Integral)
            or isinstance(end, bool)
            or not isinstance(end, numbers.Integral)
        ):
            raise CommandScheduleError(f"Scenario segment {index} boundaries must be integers.")
        start, end = int(start), int(end)
        if start != cursor:
            issue = "overlap" if start < cursor else "gap"
            raise CommandScheduleError(
                f"Scenario segment {index} creates a {issue}: expected start_step {cursor}, got {start}."
            )
        if end <= start or end > horizon:
            raise CommandScheduleError(f"Scenario segment {index} has invalid range [{start}, {end}).")
        command = raw_segment.get("command")
        if not isinstance(command, Mapping) or set(command) != set(command_axes):
            raise CommandScheduleError(f"Scenario segment {index} command must define exactly {command_axes}.")
        try:
            target = tuple(float(command[axis]) for axis in command_axes)
        except (TypeError, ValueError) as exc:
            raise CommandScheduleError(f"Scenario segment {index} command values must be numeric.") from exc
        phases.append(
            CommandPhase(
                name=str(raw_segment.get("id", f"segment-{index}")),
                start_s=start / rate,
                end_s=end / rate,
                target_b=target,
            )
        )
        cursor = end
    if cursor != horizon:
        raise CommandScheduleError(f"Scenario segments end at step {cursor}, before horizon {horizon}.")
    return CommandSchedule(
        name=str(scenario.get("id", "generic-contiguous")),
        phases=tuple(phases),
        duration_s=duration_s,
    )


def physical_pulse_spec_from_scenario(scenario: Mapping[str, Any], policy_hz: float) -> dict[str, Any] | None:
    """Resolve one exact, translational 0.12-second pulse from suite data."""

    if not isinstance(scenario, Mapping):
        raise PhysicalPulseError("Scenario must be a mapping.")
    rate = float(policy_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise PhysicalPulseError("policy_hz must be finite and positive.")
    pulses = scenario.get("pulses")
    if not isinstance(pulses, Sequence) or isinstance(pulses, (str, bytes)):
        raise PhysicalPulseError("Scenario pulses must be a sequence.")
    if not pulses:
        return None
    if len(pulses) != 1:
        raise PhysicalPulseError("The deterministic runtime supports exactly one pulse per scenario.")
    pulse = pulses[0]
    if not isinstance(pulse, Mapping):
        raise PhysicalPulseError("Physical pulse must be a mapping.")
    start, end = pulse.get("start_step"), pulse.get("end_step")
    if (
        isinstance(start, bool)
        or not isinstance(start, numbers.Integral)
        or isinstance(end, bool)
        or not isinstance(end, numbers.Integral)
    ):
        raise PhysicalPulseError("Physical pulse boundaries must be integer policy steps.")
    start, end = int(start), int(end)
    duration_steps = end - start
    if duration_steps != PULSE_POLICY_STEPS or not math.isclose(
        duration_steps / rate, PULSE_DURATION_S, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise PhysicalPulseError(
            f"Physical pulse must span exactly {PULSE_POLICY_STEPS} policy steps/{PULSE_DURATION_S}s."
        )
    horizon = scenario.get("horizon_steps")
    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or start < 0 or end > int(horizon):
        raise PhysicalPulseError("Physical pulse lies outside the scenario horizon.")
    velocity = pulse.get("root_velocity")
    axes = ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s")
    if not isinstance(velocity, Mapping) or set(velocity) != set(axes):
        raise PhysicalPulseError(f"Physical pulse root_velocity must define exactly {axes}.")
    try:
        delta_x, delta_y, delta_yaw = (float(velocity[axis]) for axis in axes)
    except (TypeError, ValueError) as exc:
        raise PhysicalPulseError("Physical pulse delta velocities must be numeric.") from exc
    if not all(math.isfinite(value) for value in (delta_x, delta_y, delta_yaw)):
        raise PhysicalPulseError("Physical pulse delta velocities must be finite.")
    if delta_yaw != 0.0:
        raise PhysicalPulseError(
            "Angular delta velocity requires torque; the base-COM mass-scaled pulse is purely translational."
        )
    if delta_x == 0.0 and delta_y == 0.0:
        raise PhysicalPulseError("Physical pulse translation must be non-zero.")
    return {
        "id": str(pulse.get("id", "pulse")),
        "start_step": start,
        "end_step": end,
        "onset_s": start / rate,
        "delta_velocity_body_m_s": (delta_x, delta_y, 0.0),
    }


class SafeCommandTargetAdapter:
    """Inject only the target buffer while preserving the command term's ramp."""

    def __init__(self, command_term: Any):
        required = ("vel_command_target_b", "vel_command_b", "is_heading_env", "is_standing_env")
        missing = [name for name in required if not hasattr(command_term, name)]
        if missing:
            raise CommandScheduleError(f"Command term is missing deterministic target fields: {missing}.")
        target = command_term.vel_command_target_b
        current = command_term.vel_command_b
        if not isinstance(target, torch.Tensor) or target.ndim != 2 or target.shape[1] != 3:
            raise CommandScheduleError("vel_command_target_b must be a tensor shaped [num_envs, 3].")
        if not isinstance(current, torch.Tensor) or current.shape != target.shape:
            raise CommandScheduleError("vel_command_b must have the same shape as vel_command_target_b.")
        if command_term.is_heading_env.shape != (target.shape[0],):
            raise CommandScheduleError("is_heading_env must be shaped [num_envs].")
        if command_term.is_standing_env.shape != (target.shape[0],):
            raise CommandScheduleError("is_standing_env must be shaped [num_envs].")
        transition_probabilities = tuple(getattr(command_term.cfg, "transition_sequence_probabilities", (0.0, 0.0)))
        if transition_probabilities != (0.0, 0.0):
            raise CommandScheduleError(
                "Deterministic evaluation requires transition_sequence_probabilities=(0.0, 0.0)."
            )
        transition_mode = getattr(command_term, "transition_sequence_mode", None)
        if isinstance(transition_mode, torch.Tensor) and bool(torch.any(transition_mode != 0)):
            raise CommandScheduleError("Deterministic evaluation found a non-zero command transition mode.")
        self._term = command_term
        self._num_envs = target.shape[0]
        self._zero_threshold = float(getattr(command_term.cfg, "zero_velocity_threshold", 0.0))

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def inject(self, target_b: Sequence[float]) -> dict[str, Any]:
        if len(target_b) != 3:
            raise CommandScheduleError("Command targets must contain [body_x, body_y, yaw_rate].")
        target = torch.as_tensor(
            target_b,
            device=self._term.vel_command_target_b.device,
            dtype=self._term.vel_command_target_b.dtype,
        )
        if not bool(torch.isfinite(target).all()):
            raise CommandScheduleError("Command target contains non-finite values.")
        if float(target[0].item()) != 0.0:
            raise CommandScheduleError("RND STEP deterministic evaluation requires body x command to remain zero.")

        current_before = self._term.vel_command_b.clone()
        self._term.vel_command_target_b[:] = target.unsqueeze(0)
        self._term.is_heading_env[:] = False
        standing = float(torch.linalg.vector_norm(target).item()) < self._zero_threshold
        self._term.is_standing_env[:] = standing
        if hasattr(self._term, "is_pure_yaw_env"):
            self._term.is_pure_yaw_env[:] = bool(target[1] == 0.0 and target[2] != 0.0)
        if hasattr(self._term, "is_straight_env"):
            self._term.is_straight_env[:] = bool(target[1] != 0.0 and target[2] == 0.0)
        if not torch.equal(self._term.vel_command_b, current_before):
            raise CommandScheduleError("Target injection modified vel_command_b and bypassed the configured ramp.")
        return self.readback()

    def readback(self) -> dict[str, Any]:
        return {
            "target_b": _tensor_to_json(self._term.vel_command_target_b),
            "command_b": _tensor_to_json(self._term.vel_command_b),
            "heading_enabled": _tensor_to_json(self._term.is_heading_env),
            "standing": _tensor_to_json(self._term.is_standing_env),
        }


def prepare_deterministic_env_cfg(env_cfg: Any, *, num_envs: int, observation_corruption: bool) -> dict[str, Any]:
    """Disable stochastic command/disturbance sources before ``gym.make``.

    The configured command ramp is deliberately not changed.
    """

    if isinstance(num_envs, bool) or int(num_envs) <= 0:
        raise EvaluationRuntimeError("num_envs must be a positive integer.")
    env_cfg.scene.num_envs = int(num_envs)
    command_cfg = env_cfg.commands.base_velocity
    original_ramp = getattr(command_cfg, "command_ramp_rates", None)
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(command_cfg.ranges, "heading"):
        command_cfg.ranges.heading = (0.0, 0.0)
    command_cfg.resampling_time_range = (CASE_DURATION_S + 1.0, CASE_DURATION_S + 1.0)
    command_cfg.rel_standing_envs = 1.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False
    for name in ("rel_pure_yaw_envs", "rel_straight_envs"):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, 0.0)
    if hasattr(command_cfg, "transition_sequence_probabilities"):
        command_cfg.transition_sequence_probabilities = (0.0, 0.0)

    policy_cfg = env_cfg.observations.policy
    policy_cfg.enable_corruption = bool(observation_corruption)
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos"):
        term_cfg = getattr(policy_cfg, term_name, None)
        params = getattr(term_cfg, "params", None)
        if isinstance(params, dict) and "sample_randomization" in params:
            params["sample_randomization"] = False

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None:
        for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            if hasattr(curriculum_cfg, name):
                setattr(curriculum_cfg, name, None)

    events_cfg = env_cfg.events
    disabled_events = (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
        "randomize_joint_armature",
    )
    for name in disabled_events:
        if hasattr(events_cfg, name):
            setattr(events_cfg, name, None)
    reset_base = getattr(events_cfg, "randomize_reset_base", None)
    if reset_base is None or not isinstance(getattr(reset_base, "params", None), dict):
        raise EvaluationRuntimeError("Evaluation requires a verifiable reset-base event configuration.")
    reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}

    robot_cfg = env_cfg.scene.robot
    for actuator_cfg in getattr(robot_cfg, "actuators", {}).values():
        if hasattr(actuator_cfg, "sample_randomization"):
            actuator_cfg.sample_randomization = False

    return {
        "command_ramp_rates": None if original_ramp is None else list(original_ramp),
        "transition_sequence_probabilities": [0.0, 0.0],
        "observation_corruption": bool(observation_corruption),
        "disabled_events": list(disabled_events),
        "reset_pose": "fixed-default",
        "reset_velocity": "zero",
    }


def _callable_name(value: Any) -> str:
    if isinstance(value, type):
        return value.__name__
    return getattr(value, "__name__", value.__class__.__name__)


def _zero_singleton(value: Any, label: str) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise PhysicalPulseError(f"{label} must be an explicit two-value reset interval.")
    try:
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise PhysicalPulseError(f"{label} must contain numeric values.") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise PhysicalPulseError(f"{label} contains a non-finite value.")
    return lower == 0.0 and upper == 0.0


def reject_legacy_disturbances(events_cfg: Any) -> dict[str, Any]:
    """Reject direct velocity pushes, persistent wrench events, and reset velocity."""

    if events_cfg is None or not hasattr(events_cfg, "__dict__"):
        raise PhysicalPulseError("Cannot inspect the event configuration; refusing to combine pulse mechanisms.")
    active_terms: list[str] = []
    reset_velocity: dict[str, list[float]] = {}
    for term_name, term_cfg in vars(events_cfg).items():
        if term_cfg is None or term_name.startswith("_"):
            continue
        func = getattr(term_cfg, "func", None)
        params = getattr(term_cfg, "params", None)
        mode = getattr(term_cfg, "mode", None)
        if func is None or not isinstance(params, Mapping):
            continue
        func_name = _callable_name(func)
        active_terms.append(term_name)
        if term_name == "randomize_push_robot" or func_name == "push_by_setting_velocity":
            raise PhysicalPulseError(f"Physical pulse cannot run with direct root-velocity push event {term_name!r}.")
        if term_name == "randomize_apply_external_force_torque" or func_name == "apply_external_force_torque":
            raise PhysicalPulseError(f"Physical pulse cannot run with persistent wrench event {term_name!r}.")
        velocity_range = params.get("velocity_range")
        if mode == "reset" and velocity_range is not None:
            if isinstance(velocity_range, Mapping):
                for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                    interval = velocity_range.get(axis, (0.0, 0.0))
                    if not _zero_singleton(interval, f"{term_name}.velocity_range.{axis}"):
                        raise PhysicalPulseError(
                            f"Physical pulse cannot run with non-zero reset velocity in {term_name}.{axis}."
                        )
                    reset_velocity[axis] = [0.0, 0.0]
            elif not _zero_singleton(velocity_range, f"{term_name}.velocity_range"):
                raise PhysicalPulseError(
                    f"Physical pulse cannot run with non-zero reset velocity in {term_name!r}."
                )
    return {"active_event_terms": active_terms, "reset_velocity_range": reset_velocity}


def yaw_rotate_body_vector(vector_b: Sequence[float], yaw_rad: float) -> tuple[float, float, float]:
    """Rotate a body-frame vector into world using onset yaw only."""

    if len(vector_b) != 3:
        raise PhysicalPulseError("Body delta velocity must contain three values.")
    x, y, z = (float(value) for value in vector_b)
    yaw = float(yaw_rad)
    if not all(math.isfinite(value) for value in (x, y, z, yaw)):
        raise PhysicalPulseError("Body delta velocity and onset yaw must be finite.")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return cosine * x - sine * y, sine * x + cosine * y, z


def _yaw_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise PhysicalPulseError("Root quaternion readback must be shaped [num_envs, 4] in wxyz order.")
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MassScaledPhysicalPulse:
    """Apply ``F=m*delta_v/T`` at the base COM for exactly 24 physics ticks."""

    def __init__(
        self,
        robot: Any,
        *,
        base_body_name: str,
        onset_s: float,
        delta_velocity_body_m_s: Sequence[float],
        physics_dt: float,
        decimation: int,
        duration_s: float = PULSE_DURATION_S,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        self._robot = robot
        self._composer = getattr(robot, "permanent_wrench_composer", None)
        if self._composer is None:
            raise PhysicalPulseError("Articulation has no public permanent_wrench_composer.")
        body_ids, body_names = robot.find_bodies(base_body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise PhysicalPulseError(f"Base body {base_body_name!r} must resolve exactly once; matched {body_names!r}.")
        self._base_body_id = int(body_ids[0])
        if env_ids is None:
            selected_env_ids = torch.arange(robot.num_instances, device=robot.device, dtype=torch.long)
        else:
            selected_env_ids = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long).flatten()
        if selected_env_ids.numel() == 0:
            raise PhysicalPulseError("Physical pulse must select at least one environment.")
        if bool(torch.any(selected_env_ids < 0)) or bool(torch.any(selected_env_ids >= robot.num_instances)):
            raise PhysicalPulseError(f"Physical pulse env_ids must lie in [0, {robot.num_instances}).")
        if torch.unique(selected_env_ids).numel() != selected_env_ids.numel():
            raise PhysicalPulseError("Physical pulse env_ids must not contain duplicates.")
        self._env_ids = selected_env_ids
        self._live_env_mask = torch.zeros(robot.num_instances, device=robot.device, dtype=torch.bool)
        self._live_env_mask[self._env_ids] = True
        self._physics_dt = float(physics_dt)
        self._decimation = int(decimation)
        self._duration_s = float(duration_s)
        self._delta_velocity_body = tuple(float(value) for value in delta_velocity_body_m_s)
        if len(self._delta_velocity_body) != 3 or not all(math.isfinite(v) for v in self._delta_velocity_body):
            raise PhysicalPulseError("delta_velocity_body_m_s must contain three finite values.")
        if self._physics_dt <= 0.0 or not math.isfinite(self._physics_dt):
            raise PhysicalPulseError("physics_dt must be finite and positive.")
        if self._decimation <= 0:
            raise PhysicalPulseError("decimation must be positive.")
        duration_ticks = self._duration_s / self._physics_dt
        if not math.isclose(duration_ticks, round(duration_ticks), rel_tol=0.0, abs_tol=1.0e-9):
            raise PhysicalPulseError("Pulse duration must be an integer number of physics ticks.")
        self._duration_ticks = int(round(duration_ticks))
        if self._duration_ticks != PULSE_PHYSICS_TICKS or not math.isclose(
            self._duration_s, PULSE_DURATION_S, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise PhysicalPulseError(
                f"Evaluation pulse must be exactly {PULSE_DURATION_S}s/{PULSE_PHYSICS_TICKS} physics ticks."
            )
        if self._duration_ticks % self._decimation != 0:
            raise PhysicalPulseError("Pulse duration cannot be cleared exactly at a policy-step boundary.")
        onset_ticks = float(onset_s) / self._physics_dt
        if (
            not math.isfinite(onset_ticks)
            or onset_ticks < 0.0
            or not math.isclose(onset_ticks, round(onset_ticks), rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise PhysicalPulseError("Pulse onset must align with a physics tick.")
        self._onset_tick = int(round(onset_ticks))
        if self._onset_tick % self._decimation != 0:
            raise PhysicalPulseError("Pulse onset must align with a policy-step boundary.")
        self._onset_step = self._onset_tick // self._decimation
        self._next_step = 0
        self._active = False
        self._complete = False
        self._elapsed_ticks = 0
        self._ticks_before_policy_step = 0
        self._step_started_active = False
        self._force_world_by_env: torch.Tensor | None = None
        self._observed_ticks_by_env = torch.zeros(robot.num_instances, device=robot.device, dtype=torch.long)
        self._started_env_ids: torch.Tensor | None = None
        self._readback: dict[str, Any] | None = None
        self._assert_zero_wrench("before pulse initialization")

    @property
    def duration_ticks(self) -> int:
        return self._duration_ticks

    @property
    def active(self) -> bool:
        return self._active

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def observed_physics_ticks(self) -> int:
        return self._elapsed_ticks

    @property
    def readback(self) -> Mapping[str, Any] | None:
        return self._readback

    def _wrench_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        force = getattr(self._composer, "composed_force_as_torch", None)
        torque = getattr(self._composer, "composed_torque_as_torch", None)
        if not isinstance(force, torch.Tensor) or not isinstance(torque, torch.Tensor) or force.shape != torque.shape:
            raise PhysicalPulseError("Cannot read back permanent articulation wrench tensors.")
        if force.ndim != 3 or force.shape[1] <= self._base_body_id or force.shape[2] != 3:
            raise PhysicalPulseError("Permanent wrench readback has an unexpected shape.")
        return force, torque

    def _assert_zero_wrench(self, context: str) -> None:
        force, torque = self._wrench_tensors()
        if bool(torch.any(force != 0.0)) or bool(torch.any(torque != 0.0)):
            raise PhysicalPulseError(
                f"Non-zero persistent wrench detected {context}; refusing mixed disturbance paths."
            )

    def _active_env_ids(self) -> torch.Tensor:
        return torch.nonzero(self._live_env_mask, as_tuple=False).flatten()

    def _write_world_force(self, env_ids: torch.Tensor) -> None:
        if self._force_world_by_env is None:
            raise PhysicalPulseError("Pulse force was not resolved before the wrench write.")
        if env_ids.numel() == 0:
            return
        force_w = self._force_world_by_env.index_select(0, env_ids)
        torque_w = torch.zeros_like(force_w)
        self._composer.set_forces_and_torques(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            body_ids=[self._base_body_id],
            env_ids=env_ids,
            is_global=True,
        )
        _, composer_torque = self._wrench_tensors()
        selected_torque = composer_torque.index_select(0, env_ids)[:, self._base_body_id]
        if bool(torch.any(selected_torque != 0.0)):
            self._composer.reset()
            raise PhysicalPulseError("Base-COM pulse produced non-zero torque readback.")

    def _start(self) -> None:
        self._assert_zero_wrench("at pulse onset")
        self._observed_ticks_by_env[self._env_ids] = 0
        active_env_ids = self._active_env_ids()
        self._started_env_ids = active_env_ids.clone()
        if active_env_ids.numel() == 0:
            self._readback = {
                "env_ids": [],
                "duration_s": self._duration_s,
                "duration_physics_ticks": self._duration_ticks,
                "observed_physics_ticks": 0,
                "observed_physics_ticks_by_env": [],
                "status": "skipped-no-live-episode",
            }
            return
        masses = self._robot.root_physx_view.get_masses()
        if not isinstance(masses, torch.Tensor) or masses.ndim != 2 or masses.shape[1] != self._robot.num_bodies:
            raise PhysicalPulseError("Articulation mass readback must be shaped [num_envs, num_bodies].")
        if not bool(torch.isfinite(masses).all()) or bool(torch.any(masses <= 0.0)):
            raise PhysicalPulseError("Articulation mass readback contains invalid link masses.")
        total_mass = masses.sum(dim=1).to(device=self._robot.device).index_select(0, active_env_ids)
        quaternion = getattr(self._robot.data, "root_link_quat_w", None)
        if quaternion is None:
            quaternion = getattr(self._robot.data, "root_quat_w", None)
        if not isinstance(quaternion, torch.Tensor) or quaternion.shape[0] != self._robot.num_instances:
            raise PhysicalPulseError("Cannot read the onset root quaternion for every environment.")
        yaw = _yaw_from_quaternion_wxyz(quaternion.index_select(0, active_env_ids))
        delta_b = torch.tensor(self._delta_velocity_body, device=yaw.device, dtype=yaw.dtype).unsqueeze(0)
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        delta_w = torch.stack(
            (
                cosine * delta_b[:, 0] - sine * delta_b[:, 1],
                sine * delta_b[:, 0] + cosine * delta_b[:, 1],
                delta_b[:, 2].expand_as(yaw),
            ),
            dim=1,
        )
        force_w = total_mass.to(dtype=yaw.dtype).unsqueeze(1) * delta_w / self._duration_s
        force_w = force_w.to(device=self._robot.device)
        self._force_world_by_env = torch.zeros(
            (self._robot.num_instances, 3), device=self._robot.device, dtype=force_w.dtype
        )
        self._force_world_by_env[active_env_ids] = force_w
        self._write_world_force(active_env_ids)
        composer_force, composer_torque = self._wrench_tensors()
        self._readback = {
            "base_body_id": self._base_body_id,
            "env_ids": _tensor_to_json(active_env_ids),
            "mass_kg": _tensor_to_json(total_mass),
            "onset_yaw_rad": _tensor_to_json(yaw),
            "delta_velocity_body_m_s": list(self._delta_velocity_body),
            "delta_velocity_world_m_s": _tensor_to_json(delta_w),
            "force_world_n": _tensor_to_json(force_w),
            "composer_force_body_n": _tensor_to_json(
                composer_force.index_select(0, active_env_ids)[:, self._base_body_id]
            ),
            "composer_torque_body_nm": _tensor_to_json(
                composer_torque.index_select(0, active_env_ids)[:, self._base_body_id]
            ),
            "duration_s": self._duration_s,
            "duration_physics_ticks": self._duration_ticks,
            "observed_physics_ticks": 0,
            "observed_physics_ticks_by_env": [0] * active_env_ids.numel(),
            "status": "active",
        }
        self._active = True

    def before_policy_step(self, step_index: int) -> None:
        if int(step_index) != self._next_step:
            raise PhysicalPulseError(f"Pulse step ordering violation: expected {self._next_step}, got {step_index}.")
        if step_index == self._onset_step:
            if self._active or self._complete:
                raise PhysicalPulseError("Pulse onset was triggered more than once.")
            self._start()
        self._ticks_before_policy_step = self._elapsed_ticks
        self._step_started_active = self._active

    def after_policy_step(self, step_index: int) -> None:
        if int(step_index) != self._next_step:
            raise PhysicalPulseError(
                f"Pulse post-step ordering violation: expected {self._next_step}, got {step_index}."
            )
        if self._step_started_active:
            observed = self._elapsed_ticks - self._ticks_before_policy_step
            if observed != self._decimation:
                self.reset()
                raise PhysicalPulseError(
                    f"Pulse observed {observed} physics ticks in one policy step; expected {self._decimation}."
                )
        self._next_step += 1

    def on_post_scene_update(self, env: Any, action: Any, policy_step: int, substep_index: int) -> None:
        """Count an actual physics tick and refresh the fixed world-frame force."""

        del action, policy_step
        if not self._active:
            return
        try:
            observed_robot = env.scene["robot"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise PhysicalPulseError("Pulse observer cannot resolve scene entity 'robot'.") from exc
        if observed_robot is not self._robot:
            raise PhysicalPulseError("Pulse observer was attached to a different articulation.")
        active_env_ids = self._active_env_ids()
        if active_env_ids.numel() == 0:
            self._composer.reset()
            self._active = False
            if self._readback is not None:
                self._readback["status"] = "interrupted-by-reset"
            self._assert_zero_wrench("after all pulsed episodes reset")
            return
        expected_substep = self._elapsed_ticks % self._decimation
        if int(substep_index) != expected_substep:
            raise PhysicalPulseError(
                f"Pulse physics ordering violation: expected substep {expected_substep}, got {substep_index}."
            )
        self._observed_ticks_by_env[active_env_ids] += 1
        self._elapsed_ticks += 1
        if self._elapsed_ticks > self._duration_ticks:
            self.reset()
            raise PhysicalPulseError("Pulse exceeded its exact physics-tick budget.")
        if self._readback is not None:
            self._readback["observed_physics_ticks"] = self._elapsed_ticks
            if self._started_env_ids is not None:
                self._readback["observed_physics_ticks_by_env"] = _tensor_to_json(
                    self._observed_ticks_by_env.index_select(0, self._started_env_ids)
                )
        if self._elapsed_ticks == self._duration_ticks:
            self._composer.reset()
            self._active = False
            self._complete = True
            if self._readback is not None:
                all_complete = self._started_env_ids is not None and bool(
                    torch.all(
                        self._observed_ticks_by_env.index_select(0, self._started_env_ids) == self._duration_ticks
                    )
                )
                self._readback["status"] = (
                    "complete-cleared" if all_complete else "complete-cleared-with-partial-episodes"
                )
            self._assert_zero_wrench("after pulse clear")
            return

        self._composer.reset(env_ids=active_env_ids)
        self._write_world_force(active_env_ids)

    def on_pre_reset(
        self,
        env: Any,
        env_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Clear force for reset episodes so it cannot leak into replacement episodes."""

        del terminated, truncated
        try:
            observed_robot = env.scene["robot"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise PhysicalPulseError("Pulse reset observer cannot resolve scene entity 'robot'.") from exc
        if observed_robot is not self._robot:
            raise PhysicalPulseError("Pulse reset observer was attached to a different articulation.")
        reset_ids = torch.as_tensor(env_ids, device=self._robot.device, dtype=torch.long).flatten()
        if reset_ids.numel() == 0:
            return
        affected = reset_ids[self._live_env_mask.index_select(0, reset_ids)]
        if affected.numel() == 0:
            return
        self._composer.reset(env_ids=affected)
        self._live_env_mask[affected] = False
        if self._readback is not None:
            previous = list(self._readback.get("reset_env_ids", []))
            self._readback["reset_env_ids"] = sorted(set(previous + affected.detach().cpu().tolist()))
        if self._active and not bool(torch.any(self._live_env_mask)):
            self._composer.reset()
            self._active = False
            if self._readback is not None:
                self._readback["status"] = "interrupted-by-reset"
            self._assert_zero_wrench("after all pulsed episodes reset")

    def on_post_reset(self, env: Any, env_ids: torch.Tensor) -> None:
        del env, env_ids

    def reset(self) -> None:
        """Clear all pulse state and any buffered force, including on early episode reset."""

        self._composer.reset()
        self._assert_zero_wrench("after pulse reset")
        self._next_step = 0
        self._active = False
        self._complete = False
        self._elapsed_ticks = 0
        self._ticks_before_policy_step = 0
        self._step_started_active = False
        self._force_world_by_env = None
        self._observed_ticks_by_env.zero_()
        self._started_env_ids = None
        self._live_env_mask.zero_()
        self._live_env_mask[self._env_ids] = True
        self._readback = None


@dataclass(frozen=True)
class FixedMaterialSettings:
    static_friction: float
    dynamic_friction: float
    restitution: float

    def __post_init__(self) -> None:
        static = _finite_scalar(self.static_friction, "material.static_friction", minimum=0.0)
        dynamic = _finite_scalar(self.dynamic_friction, "material.dynamic_friction", minimum=0.0)
        restitution = _finite_scalar(self.restitution, "material.restitution", minimum=0.0)
        if dynamic > static:
            raise FixedDomainError("material.dynamic_friction cannot exceed static_friction.")
        if restitution > 1.0:
            raise FixedDomainError("material.restitution cannot exceed 1.0.")


@dataclass(frozen=True)
class FixedActuatorSettings:
    stiffness_scale: float = 1.0
    damping_scale: float = 1.0
    armature_scale: float = 1.0
    command_delay_s: float | None = None
    command_position_bias_rad: float | None = None
    play_threshold_scale: float = 1.0
    motor_strength_scale: float | None = None
    coulomb_torque_nm: float | None = None
    friction_transition_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        _finite_scalar(self.stiffness_scale, "actuator.stiffness_scale", minimum=0.0)
        _finite_scalar(self.damping_scale, "actuator.damping_scale", minimum=0.0)
        _finite_scalar(self.armature_scale, "actuator.armature_scale", minimum=0.0)
        _finite_scalar(self.play_threshold_scale, "actuator.play_threshold_scale", minimum=0.0)
        for name, value, minimum in (
            ("command_delay_s", self.command_delay_s, 0.0),
            ("command_position_bias_rad", self.command_position_bias_rad, None),
            ("motor_strength_scale", self.motor_strength_scale, 0.0),
            ("coulomb_torque_nm", self.coulomb_torque_nm, 0.0),
            ("friction_transition_velocity_rad_s", self.friction_transition_velocity_rad_s, 1.0e-12),
        ):
            if value is not None:
                _finite_scalar(value, f"actuator.{name}", minimum=minimum)


@dataclass(frozen=True)
class FixedImuChannelSettings:
    delay_s: float | None = None
    noise_sigma: float | None = None
    bias: float | None = None

    def __post_init__(self) -> None:
        if self.delay_s is not None:
            _finite_scalar(self.delay_s, "imu.delay_s", minimum=0.0)
        if self.noise_sigma is not None:
            _finite_scalar(self.noise_sigma, "imu.noise_sigma", minimum=0.0)
        if self.bias is not None:
            _finite_scalar(self.bias, "imu.bias")


@dataclass(frozen=True)
class FixedImuSettings:
    gyro: FixedImuChannelSettings = dataclasses.field(default_factory=FixedImuChannelSettings)
    gravity: FixedImuChannelSettings = dataclasses.field(default_factory=FixedImuChannelSettings)


@dataclass(frozen=True)
class FixedEncoderSettings:
    zero_offset_rad: tuple[float, ...]
    sample_age_s: tuple[float, ...]

    def __post_init__(self) -> None:
        _fixed_vector(
            self.zero_offset_rad,
            "encoder.zero_offset_rad",
            length=ENCODER_JOINT_COUNT,
            minimum=ENCODER_ZERO_OFFSET_RANGE_RAD[0],
            maximum=ENCODER_ZERO_OFFSET_RANGE_RAD[1],
        )
        _fixed_vector(
            self.sample_age_s,
            "encoder.sample_age_s",
            length=ENCODER_JOINT_COUNT,
            minimum=ENCODER_SAMPLE_AGE_RANGE_S[0],
            maximum=ENCODER_SAMPLE_AGE_RANGE_S[1],
        )


@dataclass(frozen=True)
class FixedDomainSettings:
    material: FixedMaterialSettings
    base_mass_add_kg: float
    other_mass_scale: float
    base_com_offset_m: tuple[float, float, float]
    encoder: FixedEncoderSettings
    actuator: FixedActuatorSettings = dataclasses.field(default_factory=FixedActuatorSettings)
    imu: FixedImuSettings = dataclasses.field(default_factory=FixedImuSettings)

    def __post_init__(self) -> None:
        _finite_scalar(self.base_mass_add_kg, "base_mass_add_kg")
        _finite_scalar(self.other_mass_scale, "other_mass_scale", minimum=1.0e-12)
        if len(self.base_com_offset_m) != 3:
            raise FixedDomainError("base_com_offset_m must contain exactly x, y, and z.")
        for axis, value in zip("xyz", self.base_com_offset_m):
            _finite_scalar(value, f"base_com_offset_m.{axis}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FixedDomainSettings:
        """Parse exact values and reject interval/range-shaped domain input."""

        mapping = _strict_mapping(value, "domain")
        allowed = {
            "material",
            "base_mass_add_kg",
            "other_mass_scale",
            "base_com_offset_m",
            "actuator",
            "encoder",
            "imu",
        }
        _reject_unknown(mapping, allowed, "domain")
        required = {"material", "base_mass_add_kg", "other_mass_scale", "base_com_offset_m", "encoder"}
        missing = sorted(required - set(mapping))
        if missing:
            raise FixedDomainError(f"domain is missing required fixed fields: {missing}.")

        material_map = _strict_mapping(mapping["material"], "domain.material")
        _reject_unknown(material_map, {"static_friction", "dynamic_friction", "restitution"}, "domain.material")
        if set(material_map) != {"static_friction", "dynamic_friction", "restitution"}:
            raise FixedDomainError("domain.material requires static_friction, dynamic_friction, and restitution.")
        material = FixedMaterialSettings(
            static_friction=_finite_scalar(material_map["static_friction"], "domain.material.static_friction"),
            dynamic_friction=_finite_scalar(material_map["dynamic_friction"], "domain.material.dynamic_friction"),
            restitution=_finite_scalar(material_map["restitution"], "domain.material.restitution"),
        )

        com_map = _strict_mapping(mapping["base_com_offset_m"], "domain.base_com_offset_m")
        _reject_unknown(com_map, {"x", "y", "z"}, "domain.base_com_offset_m")
        if set(com_map) != {"x", "y", "z"}:
            raise FixedDomainError("domain.base_com_offset_m requires x, y, and z scalars.")
        com = tuple(_finite_scalar(com_map[axis], f"domain.base_com_offset_m.{axis}") for axis in "xyz")

        actuator_map = _strict_mapping(mapping.get("actuator", {}), "domain.actuator")
        actuator_fields = {field.name for field in dataclasses.fields(FixedActuatorSettings)}
        _reject_unknown(actuator_map, actuator_fields, "domain.actuator")
        actuator_kwargs = {
            key: _finite_scalar(raw, f"domain.actuator.{key}") if raw is not None else None
            for key, raw in actuator_map.items()
        }
        actuator = FixedActuatorSettings(**actuator_kwargs)

        encoder_map = _strict_mapping(mapping["encoder"], "domain.encoder")
        _reject_unknown(encoder_map, {"zero_offset_rad", "sample_age_s"}, "domain.encoder")
        if set(encoder_map) != {"zero_offset_rad", "sample_age_s"}:
            raise FixedDomainError("domain.encoder requires zero_offset_rad and sample_age_s.")
        encoder = FixedEncoderSettings(
            zero_offset_rad=_fixed_vector(
                encoder_map["zero_offset_rad"],
                "domain.encoder.zero_offset_rad",
                length=ENCODER_JOINT_COUNT,
                minimum=ENCODER_ZERO_OFFSET_RANGE_RAD[0],
                maximum=ENCODER_ZERO_OFFSET_RANGE_RAD[1],
            ),
            sample_age_s=_fixed_vector(
                encoder_map["sample_age_s"],
                "domain.encoder.sample_age_s",
                length=ENCODER_JOINT_COUNT,
                minimum=ENCODER_SAMPLE_AGE_RANGE_S[0],
                maximum=ENCODER_SAMPLE_AGE_RANGE_S[1],
            ),
        )

        imu_map = _strict_mapping(mapping.get("imu", {}), "domain.imu")
        _reject_unknown(imu_map, {"gyro", "gravity"}, "domain.imu")
        imu_channels = {}
        for channel in ("gyro", "gravity"):
            channel_map = _strict_mapping(imu_map.get(channel, {}), f"domain.imu.{channel}")
            _reject_unknown(channel_map, {"delay_s", "noise_sigma", "bias"}, f"domain.imu.{channel}")
            imu_channels[channel] = FixedImuChannelSettings(**{
                key: _finite_scalar(raw, f"domain.imu.{channel}.{key}") if raw is not None else None
                for key, raw in channel_map.items()
            })

        return cls(
            material=material,
            base_mass_add_kg=_finite_scalar(mapping["base_mass_add_kg"], "domain.base_mass_add_kg"),
            other_mass_scale=_finite_scalar(mapping["other_mass_scale"], "domain.other_mass_scale"),
            base_com_offset_m=com,
            actuator=actuator,
            encoder=encoder,
            imu=FixedImuSettings(**imu_channels),
        )


@runtime_checkable
class FixedDomainBackendProtocol(Protocol):
    """Structural boundary used by the singleton domain orchestrator."""

    def apply_material(self, settings: FixedMaterialSettings) -> Mapping[str, Any]: ...

    def apply_mass(self, base_add_kg: float, other_scale: float) -> Mapping[str, Any]: ...

    def apply_base_com(self, offset_m: tuple[float, float, float]) -> Mapping[str, Any]: ...

    def apply_actuator(self, settings: FixedActuatorSettings) -> Mapping[str, Any]: ...

    def apply_encoder(self, settings: FixedEncoderSettings) -> Mapping[str, Any]: ...

    def apply_imu(self, settings: FixedImuSettings) -> Mapping[str, Any]: ...


class FixedSingletonDomainApplicator:
    """Apply components in a fixed order and retain complete readback."""

    def __init__(self, backend: FixedDomainBackendProtocol):
        if not isinstance(backend, FixedDomainBackendProtocol):
            raise FixedDomainError("Fixed domain backend does not satisfy the required structural protocol.")
        self._backend = backend

    def apply(self, settings: FixedDomainSettings) -> dict[str, Any]:
        if not isinstance(settings, FixedDomainSettings):
            raise FixedDomainError("FixedSingletonDomainApplicator requires validated FixedDomainSettings.")
        return {
            "material": dict(self._backend.apply_material(settings.material)),
            "mass": dict(self._backend.apply_mass(settings.base_mass_add_kg, settings.other_mass_scale)),
            "base_com": dict(self._backend.apply_base_com(settings.base_com_offset_m)),
            "actuator": dict(self._backend.apply_actuator(settings.actuator)),
            "encoder": dict(self._backend.apply_encoder(settings.encoder)),
            "imu": dict(self._backend.apply_imu(settings.imu)),
        }


class IsaacFixedDomainBackend:
    """Source-grounded fixed-domain writer/readback for one vectorized articulation."""

    def __init__(self, sim_env: Any, *, base_body_name: str):
        try:
            self._robot = sim_env.scene["robot"]
        except (KeyError, TypeError) as exc:
            raise FixedDomainError("Fixed domain requires scene entity 'robot'.") from exc
        robot = self._robot
        body_ids, body_names = robot.find_bodies(base_body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise FixedDomainError(f"Base body {base_body_name!r} must resolve exactly once; got {body_names!r}.")
        self._base_body_id = int(body_ids[0])
        if len(robot.body_names) != robot.num_bodies or len(set(robot.body_names)) != robot.num_bodies:
            raise FixedDomainError("Robot body-name readback is incomplete or ambiguous.")
        self._env_ids_cpu = torch.arange(robot.num_instances, dtype=torch.long, device="cpu")
        self._env_ids_device = self._env_ids_cpu.to(robot.device)
        self._default_mass = robot.data.default_mass.detach().cpu().clone()
        self._default_inertia = robot.data.default_inertia.detach().cpu().clone()
        self._default_com = robot.root_physx_view.get_coms().clone()
        if self._default_mass.shape != (robot.num_instances, robot.num_bodies):
            raise FixedDomainError("default_mass readback has an unexpected shape.")
        if self._default_inertia.shape[:2] != self._default_mass.shape:
            raise FixedDomainError("default_inertia readback does not align with body masses.")
        if self._default_com.shape[:2] != self._default_mass.shape or self._default_com.shape[2] != 7:
            raise FixedDomainError("COM readback must be shaped [num_envs, num_bodies, 7].")
        self._actuator_baselines = {
            name: (actuator.stiffness.detach().clone(), actuator.damping.detach().clone())
            for name, actuator in robot.actuators.items()
        }
        self._default_armature = robot.data.default_joint_armature.detach().clone()
        self._encoder_term = self._resolve_encoder_term(sim_env)
        self._imu_terms = self._resolve_imu_terms(sim_env)

    @staticmethod
    def _resolve_encoder_term(sim_env: Any) -> Any | None:
        manager = getattr(sim_env, "observation_manager", None)
        names_by_group = getattr(manager, "_group_obs_term_names", None)
        cfgs_by_group = getattr(manager, "_group_obs_term_cfgs", None)
        if not isinstance(names_by_group, dict) or not isinstance(cfgs_by_group, dict):
            return None
        names = names_by_group.get("policy", ())
        cfgs = cfgs_by_group.get("policy", ())
        if len(names) != len(cfgs):
            raise FixedDomainError("Policy observation term names/configs are inconsistent.")
        resolved = []
        for name, cfg in zip(names, cfgs):
            term = getattr(cfg, "func", None)
            state = getattr(term, "state", None)
            if state is None or not hasattr(term, "set_fixed_episode_parameters"):
                continue
            if not hasattr(state, "zero_offset_rad") or not hasattr(state, "sample_age_s"):
                continue
            resolved.append((name, term))
        if len(resolved) > 1:
            raise FixedDomainError(f"Multiple policy encoder terms were found: {[name for name, _ in resolved]!r}.")
        return None if not resolved else resolved[0][1]

    @staticmethod
    def _resolve_imu_terms(sim_env: Any) -> dict[str, Any]:
        manager = getattr(sim_env, "observation_manager", None)
        names_by_group = getattr(manager, "_group_obs_term_names", None)
        cfgs_by_group = getattr(manager, "_group_obs_term_cfgs", None)
        if not isinstance(names_by_group, dict) or not isinstance(cfgs_by_group, dict):
            return {}
        names = names_by_group.get("policy", ())
        cfgs = cfgs_by_group.get("policy", ())
        if len(names) != len(cfgs):
            raise FixedDomainError("Policy observation term names/configs are inconsistent.")
        resolved: dict[str, Any] = {}
        for name, cfg in zip(names, cfgs):
            term = getattr(cfg, "func", None)
            state = getattr(term, "state", None)
            channel = getattr(state, "channel", None)
            if channel not in ("gyro", "gravity"):
                continue
            if channel in resolved:
                raise FixedDomainError(f"Multiple CMP10A terms resolve channel {channel!r}.")
            if not hasattr(term, "reset"):
                raise FixedDomainError(f"CMP10A policy term {name!r} has no reset API.")
            resolved[channel] = term
        return resolved

    @staticmethod
    def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str, atol: float = 1.0e-6) -> None:
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        if actual_cpu.shape != expected_cpu.shape or not torch.allclose(actual_cpu, expected_cpu, rtol=0.0, atol=atol):
            raise FixedDomainError(f"{label} write/readback mismatch.")

    def apply_material(self, settings: FixedMaterialSettings) -> Mapping[str, Any]:
        materials = self._robot.root_physx_view.get_material_properties().clone()
        if materials.ndim != 3 or materials.shape[0] != self._robot.num_instances or materials.shape[2] != 3:
            raise FixedDomainError("Material readback must be shaped [num_envs, max_shapes, 3].")
        target = torch.tensor(
            [settings.static_friction, settings.dynamic_friction, settings.restitution],
            dtype=materials.dtype,
            device=materials.device,
        )
        materials[:] = target
        self._robot.root_physx_view.set_material_properties(materials, self._env_ids_cpu)
        actual = self._robot.root_physx_view.get_material_properties()
        self._assert_close(actual, materials, "material")
        return {"values": _tensor_to_json(actual[0]), "all_envs_equal": _all_rows_equal(actual)}

    def apply_mass(self, base_add_kg: float, other_scale: float) -> Mapping[str, Any]:
        target = self._default_mass.clone()
        other_ids = [index for index in range(self._robot.num_bodies) if index != self._base_body_id]
        target[:, self._base_body_id] += float(base_add_kg)
        target[:, other_ids] *= float(other_scale)
        if bool(torch.any(target <= 0.0)):
            raise FixedDomainError("Fixed mass settings produce a non-positive link mass.")
        self._robot.root_physx_view.set_masses(target, self._env_ids_cpu)
        ratio = target / self._default_mass
        inertia = self._default_inertia * ratio.unsqueeze(-1)
        self._robot.root_physx_view.set_inertias(inertia, self._env_ids_cpu)
        actual_mass = self._robot.root_physx_view.get_masses()
        actual_inertia = self._robot.root_physx_view.get_inertias()
        self._assert_close(actual_mass, target, "mass")
        self._assert_close(actual_inertia, inertia, "mass-scaled inertia", atol=1.0e-5)
        return {
            "body_names": list(self._robot.body_names),
            "mass_kg": _tensor_to_json(actual_mass),
            "total_mass_kg": _tensor_to_json(actual_mass.sum(dim=1)),
            "base_mass_add_kg": float(base_add_kg),
            "other_mass_scale": float(other_scale),
        }

    def apply_base_com(self, offset_m: tuple[float, float, float]) -> Mapping[str, Any]:
        target = self._default_com.clone()
        offset = torch.tensor(offset_m, dtype=target.dtype, device=target.device)
        target[:, self._base_body_id, :3] += offset
        self._robot.root_physx_view.set_coms(target, self._env_ids_cpu)
        actual = self._robot.root_physx_view.get_coms()
        self._assert_close(actual, target, "base COM")
        return {
            "base_body_id": self._base_body_id,
            "offset_m": list(offset_m),
            "pose_b": _tensor_to_json(actual[:, self._base_body_id]),
        }

    def apply_actuator(self, settings: FixedActuatorSettings) -> Mapping[str, Any]:
        robot = self._robot
        if settings.play_threshold_scale != 1.0:
            raise FixedDomainError(
                "The current command-path API has no public play-threshold override; refusing approximation."
            )
        target_armature = self._default_armature * float(settings.armature_scale)
        robot.write_joint_armature_to_sim(target_armature, env_ids=self._env_ids_device)
        actuator_readback: dict[str, Any] = {}
        for name, actuator in robot.actuators.items():
            base_stiffness, base_damping = self._actuator_baselines[name]
            target_stiffness = base_stiffness * float(settings.stiffness_scale)
            target_damping = base_damping * float(settings.damping_scale)
            actuator.stiffness.copy_(target_stiffness)
            actuator.damping.copy_(target_damping)
            if bool(getattr(actuator, "is_implicit_model", False)):
                robot.write_joint_stiffness_to_sim(
                    actuator.stiffness, joint_ids=actuator.joint_indices, env_ids=self._env_ids_device
                )
                robot.write_joint_damping_to_sim(
                    actuator.damping, joint_ids=actuator.joint_indices, env_ids=self._env_ids_device
                )
            self._assert_close(actuator.stiffness, target_stiffness, f"actuator {name} stiffness")
            self._assert_close(actuator.damping, target_damping, f"actuator {name} damping")

            command_path = getattr(actuator, "command_path", None)
            if command_path is not None:
                command_path.sample_randomization = False
                if settings.command_delay_s is not None:
                    command_path.set_delay_override(settings.command_delay_s)
                if settings.command_position_bias_rad is not None:
                    command_path.set_position_bias_override(settings.command_position_bias_rad)
            elif settings.command_delay_s is not None or settings.command_position_bias_rad is not None:
                raise FixedDomainError(
                    f"Actuator {name!r} has no command path for the requested delay/position-bias singleton."
                )

            torque_randomizer = getattr(actuator, "torque_randomizer", None)
            torque_values_requested = any(
                value is not None
                for value in (
                    settings.motor_strength_scale,
                    settings.coulomb_torque_nm,
                    settings.friction_transition_velocity_rad_s,
                )
            )
            if torque_randomizer is None and torque_values_requested:
                raise FixedDomainError(f"Actuator {name!r} has no torque randomizer for requested fixed values.")
            if torque_randomizer is not None:
                torque_randomizer.sample_randomization = False
                torque_randomizer.reset()
                if settings.motor_strength_scale is not None:
                    torque_randomizer.sampled_motor_strength_scale.fill_(settings.motor_strength_scale)
                if settings.coulomb_torque_nm is not None:
                    torque_randomizer.sampled_coulomb_torque_nm.fill_(settings.coulomb_torque_nm)
                if settings.friction_transition_velocity_rad_s is not None:
                    torque_randomizer.sampled_transition_velocity_rad_s.fill_(
                        settings.friction_transition_velocity_rad_s
                    )
                for field_name, expected in (
                    ("sampled_motor_strength_scale", settings.motor_strength_scale),
                    ("sampled_coulomb_torque_nm", settings.coulomb_torque_nm),
                    ("sampled_transition_velocity_rad_s", settings.friction_transition_velocity_rad_s),
                ):
                    if expected is None:
                        continue
                    actual = getattr(torque_randomizer, field_name)
                    target = torch.full_like(actual, float(expected))
                    self._assert_close(actual, target, f"actuator {name} {field_name}")

            entry: dict[str, Any] = {
                "stiffness": _tensor_to_json(actuator.stiffness),
                "damping": _tensor_to_json(actuator.damping),
            }
            if command_path is not None:
                if settings.command_delay_s is not None:
                    self._assert_close(
                        command_path.sampled_delay_s,
                        torch.full_like(command_path.sampled_delay_s, settings.command_delay_s),
                        f"actuator {name} command delay",
                    )
                if settings.command_position_bias_rad is not None:
                    self._assert_close(
                        command_path.sampled_position_bias_rad,
                        torch.full_like(
                            command_path.sampled_position_bias_rad,
                            settings.command_position_bias_rad,
                        ),
                        f"actuator {name} command position bias",
                    )
                entry["command_delay_s"] = _tensor_to_json(command_path.sampled_delay_s)
                entry["command_position_bias_rad"] = _tensor_to_json(command_path.sampled_position_bias_rad)
                entry["play_thresholds_rad"] = _tensor_to_json(command_path.sampled_play_thresholds_rad)
                entry["play_threshold_scale"] = float(settings.play_threshold_scale)
            if torque_randomizer is not None:
                entry["motor_strength_scale"] = _tensor_to_json(torque_randomizer.sampled_motor_strength_scale)
                entry["coulomb_torque_nm"] = _tensor_to_json(torque_randomizer.sampled_coulomb_torque_nm)
                entry["friction_transition_velocity_rad_s"] = _tensor_to_json(
                    torque_randomizer.sampled_transition_velocity_rad_s
                )
            actuator_readback[name] = entry

        actual_armature = robot.root_physx_view.get_dof_armatures().to(robot.device)
        self._assert_close(actual_armature, target_armature, "joint armature")
        return {
            "joint_names": list(robot.joint_names),
            "armature": _tensor_to_json(actual_armature),
            "requested": dataclasses.asdict(settings),
            "groups": actuator_readback,
        }

    def apply_encoder(self, settings: FixedEncoderSettings) -> Mapping[str, Any]:
        term = self._encoder_term
        if term is None:
            raise FixedDomainError("Fixed encoder settings require one RND Dynamixel policy observation term.")
        term.set_fixed_episode_parameters(
            zero_offset_rad=settings.zero_offset_rad,
            sample_age_s=settings.sample_age_s,
        )
        state = term.state
        target_offset = torch.tensor(
            settings.zero_offset_rad,
            dtype=state.zero_offset_rad.dtype,
            device=state.zero_offset_rad.device,
        ).unsqueeze(0).expand_as(state.zero_offset_rad)
        target_age = torch.tensor(
            settings.sample_age_s,
            dtype=state.sample_age_s.dtype,
            device=state.sample_age_s.device,
        ).unsqueeze(0).expand_as(state.sample_age_s)
        self._assert_close(state.zero_offset_rad, target_offset, "encoder zero offset")
        self._assert_close(state.sample_age_s, target_age, "encoder sample age")
        return {
            "joint_names": list(term.joint_names),
            "zero_offset_rad": _tensor_to_json(state.zero_offset_rad),
            "sample_age_s": _tensor_to_json(state.sample_age_s),
            "all_envs_equal": _all_rows_equal(state.zero_offset_rad) and _all_rows_equal(state.sample_age_s),
            "requested": dataclasses.asdict(settings),
        }

    def apply_imu(self, settings: FixedImuSettings) -> Mapping[str, Any]:
        requested = {
            "gyro": settings.gyro,
            "gravity": settings.gravity,
        }
        readback: dict[str, Any] = {}
        for channel, channel_settings in requested.items():
            values_requested = any(
                value is not None
                for value in (channel_settings.delay_s, channel_settings.noise_sigma, channel_settings.bias)
            )
            term = self._imu_terms.get(channel)
            if term is None:
                if values_requested:
                    raise FixedDomainError(f"Fixed IMU {channel} settings requested, but no CMP10A term exists.")
                continue
            state = term.state
            if channel_settings.delay_s is not None:
                state.delay_range_s = (channel_settings.delay_s, channel_settings.delay_s)
            if channel_settings.noise_sigma is not None:
                state.noise_sigma_range = (channel_settings.noise_sigma, channel_settings.noise_sigma)
            if channel_settings.bias is not None:
                if channel == "gravity" and channel_settings.bias != 0.0:
                    raise FixedDomainError("The CMP10A gravity channel does not support additive bias.")
                state.bias_range = (channel_settings.bias, channel_settings.bias)
            if values_requested:
                state.sample_randomization = True
                term.reset()
            for field_name, expected in (
                ("delay_s", channel_settings.delay_s),
                ("noise_sigma", channel_settings.noise_sigma),
                ("bias", channel_settings.bias),
            ):
                if expected is None:
                    continue
                actual = getattr(state, field_name, None)
                if not isinstance(actual, torch.Tensor):
                    raise FixedDomainError(f"CMP10A {channel} {field_name} has no tensor readback.")
                self._assert_close(
                    actual,
                    torch.full_like(actual, float(expected)),
                    f"CMP10A {channel} {field_name}",
                )
            readback[channel] = {
                "delay_s": _tensor_to_json(state.delay_s),
                "noise_sigma": _tensor_to_json(state.noise_sigma),
                "bias": _tensor_to_json(state.bias),
                "body_name": term.body_name,
                "body_id": term.body_id,
                "requested": dataclasses.asdict(channel_settings),
            }
        return readback


def checkpoint_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationRuntimeError(f"Checkpoint does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_actor_observation_dimension(path: str | Path) -> int:
    """Read the first actor linear layer and fail before constructing an environment."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationRuntimeError(f"Checkpoint does not exist: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise EvaluationRuntimeError(f"Could not inspect checkpoint actor dimensions: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationRuntimeError("Checkpoint payload must be a mapping.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise EvaluationRuntimeError("Checkpoint has no model_state_dict mapping.")
    candidates = [
        (str(key), value)
        for key, value in state_dict.items()
        if str(key).endswith("actor.0.weight") and isinstance(value, torch.Tensor) and value.ndim == 2
    ]
    if len(candidates) != 1:
        raise EvaluationRuntimeError(
            "Checkpoint must expose exactly one two-dimensional first actor weight ending in "
            f"'actor.0.weight'; found {[key for key, _ in candidates]!r}."
        )
    return int(candidates[0][1].shape[1])


def validate_checkpoint_actor_observation_dimension(path: str | Path, expected_dimension: int) -> int:
    if isinstance(expected_dimension, bool) or not isinstance(expected_dimension, numbers.Integral):
        raise EvaluationRuntimeError("expected actor observation dimension must be an integer.")
    expected = int(expected_dimension)
    if expected <= 0:
        raise EvaluationRuntimeError("expected actor observation dimension must be positive.")
    actual = checkpoint_actor_observation_dimension(path)
    if actual != expected:
        raise EvaluationRuntimeError(
            f"Checkpoint actor observation dimension is {actual}, but this suite requires {expected}. "
            "Older 45-D checkpoints cannot run with the 4-frame 171-D actor observation."
        )
    return actual


def validate_split_checkpoint(split: str, expected_sha256: str | None, actual_sha256: str) -> None:
    """Require and validate a frozen checkpoint hash for the test split."""

    normalized_split = str(split).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", actual_sha256):
        raise EvaluationRuntimeError("Actual checkpoint SHA-256 is invalid.")
    if normalized_split == "test" and expected_sha256 is None:
        raise EvaluationRuntimeError("The test split is locked unless the suite freezes checkpoint_sha256.")
    if expected_sha256 is None:
        return
    expected = str(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise EvaluationRuntimeError("Suite checkpoint_sha256 must contain exactly 64 hexadecimal characters.")
    if expected != actual_sha256.lower():
        raise EvaluationRuntimeError(
            f"Checkpoint SHA-256 mismatch: suite freezes {expected}, selected checkpoint is {actual_sha256.lower()}."
        )


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise EvaluationArtifactError("JSON mappings must use string keys.")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise EvaluationArtifactError("Object arrays are not allowed in evaluation artifacts.")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _tensor_to_json(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvaluationArtifactError("JSON artifacts cannot contain NaN or infinity.")
        return value
    raise EvaluationArtifactError(f"Unsupported JSON artifact value: {type(value).__name__}.")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def resolved_config_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tensor_to_json(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def _all_rows_equal(value: torch.Tensor) -> bool:
    if value.shape[0] <= 1:
        return True
    return bool(torch.all(value == value[0:1]).item())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    temporary.replace(path)


def _safe_case_id(case_id: str) -> str:
    value = str(case_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise EvaluationArtifactError(f"Unsafe case id for artifact path: {case_id!r}.")
    return value


class EvaluationArtifactWriter:
    """Write resolved config/hash, pickle-free raw NPZ, metrics, and summary."""

    def __init__(self, output_directory: str | Path):
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write_case(
        self,
        case_id: str,
        *,
        resolved_config: Mapping[str, Any],
        raw: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        safe_id = _safe_case_id(case_id)
        case_directory = self.output_directory / "cases" / safe_id
        case_directory.mkdir(parents=True, exist_ok=True)
        config_hash = resolved_config_sha256(resolved_config)
        resolved_path = case_directory / "resolved_config.json"
        raw_path = case_directory / "raw.npz"
        metrics_path = case_directory / "metrics.json"
        _write_json(resolved_path, {"sha256": config_hash, "config": resolved_config})

        arrays: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise EvaluationArtifactError("Raw NPZ keys must be non-empty strings.")
            array = np.asarray(value)
            if array.dtype == object:
                raise EvaluationArtifactError(f"Raw NPZ field {key!r} has object dtype.")
            arrays[key] = array
        temporary = raw_path.with_name(f".{raw_path.name}.tmp")
        with temporary.open("wb") as output_file:
            np.savez_compressed(output_file, **arrays)
        temporary.replace(raw_path)
        _write_json(metrics_path, {"resolved_config_sha256": config_hash, "metrics": metrics})
        return {
            "case_id": safe_id,
            "resolved_config_sha256": config_hash,
            "resolved_config": str(resolved_path),
            "raw_npz": str(raw_path),
            "metrics_json": str(metrics_path),
        }

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        path = self.output_directory / "summary.json"
        _write_json(path, summary)
        return path


def equally_weighted_summary(
    case_episode_metrics: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, Any]:
    """Average episodes within cases, then average case means with equal weight."""

    if not case_episode_metrics:
        raise EvaluationArtifactError("At least one case is required for a summary.")
    case_means: dict[str, dict[str, float]] = {}
    expected_keys: set[str] | None = None
    episode_count = 0
    for case_id, episodes in case_episode_metrics.items():
        if not episodes:
            raise EvaluationArtifactError(f"Case {case_id!r} has no episode metrics.")
        keys = set(episodes[0])
        if not keys:
            raise EvaluationArtifactError(f"Case {case_id!r} has an empty metric set.")
        for episode in episodes:
            if set(episode) != keys:
                raise EvaluationArtifactError(f"Case {case_id!r} episodes do not share identical metric keys.")
            for name, value in episode.items():
                if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                    raise EvaluationArtifactError(f"Metric {case_id}.{name} must be one finite scalar.")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise EvaluationArtifactError("Every case must expose the same metric keys for equal weighting.")
        case_means[str(case_id)] = {
            name: float(np.mean([float(episode[name]) for episode in episodes], dtype=np.float64))
            for name in sorted(keys)
        }
        episode_count += len(episodes)
    assert expected_keys is not None
    overall = {
        name: float(np.mean([case_means[case_id][name] for case_id in case_means], dtype=np.float64))
        for name in sorted(expected_keys)
    }
    return {
        "weighting": "episodes_equal_within_case_then_cases_equal",
        "case_count": len(case_means),
        "episode_count": episode_count,
        "case_means": case_means,
        "overall": overall,
    }


class EvaluationSuiteLoaderProtocol(Protocol):
    """Callable boundary implemented by ``evaluation_schema.load_evaluation_suite``."""

    def __call__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        verify_artifacts: bool = False,
        repository_root: str | Path | None = None,
    ) -> dict[str, Any]: ...


class PhysicsTelemetryAdapterProtocol(Protocol):
    """Runtime boundary implemented by the physics telemetry attachment."""

    logger: Any

    def save(self, output_path: str | Path | None = None) -> Path: ...

    def close(self) -> None: ...


class EpisodeMetricsEvaluatorProtocol(Protocol):
    """Callable boundary implemented by ``gait_metrics.evaluate_episode_metrics``."""

    def __call__(
        self,
        telemetry: Mapping[str, Any],
        *,
        horizon_steps: int,
        step_dt: float,
        effort_limits: Any,
        minimum_touchdown_progress_m: float,
        tap_max_air_time_s: float,
        command_speed_threshold_m_s: float,
        torque_saturation_threshold_fraction: float,
        joint_names: Sequence[str] | None = None,
        timeout: Any | None = None,
        push_end_step: int | None = None,
        linear_velocity_error_threshold_m_s: float | None = None,
        yaw_rate_error_threshold_rad_s: float | None = None,
        recovery_dwell_s: float | None = None,
    ) -> dict[str, Any]: ...
