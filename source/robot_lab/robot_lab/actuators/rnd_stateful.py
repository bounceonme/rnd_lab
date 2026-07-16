"""GPU-vectorized delay and generalized-play command path for RND STEP."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


RND_ACTUATOR_MODEL_SCHEMA_VERSION = 1
RND_ACTUATOR_MODEL_TYPE = "rnd_stateful_equivalent_actuator"


class RndActuatorModelError(ValueError):
    """Raised when an RND actuator model or command tensor is invalid."""


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RndActuatorModelError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise RndActuatorModelError(f"{label} must be finite, got {result!r}.")
    if minimum is not None and result < minimum:
        raise RndActuatorModelError(f"{label} must be >= {minimum}, got {result}.")
    return result


def _range_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RndActuatorModelError(f"{label} must be a two-element JSON list.")
    lower = _finite_float(value[0], f"{label}[0]", minimum=0.0)
    upper = _finite_float(value[1], f"{label}[1]", minimum=0.0)
    if lower > upper:
        raise RndActuatorModelError(f"{label} lower bound exceeds upper bound: {value!r}.")
    return lower, upper


def _signed_range_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RndActuatorModelError(f"{label} must be a two-element JSON list.")
    lower = _finite_float(value[0], f"{label}[0]")
    upper = _finite_float(value[1], f"{label}[1]")
    if lower > upper:
        raise RndActuatorModelError(f"{label} lower bound exceeds upper bound: {value!r}.")
    return lower, upper


def _unique_name_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise RndActuatorModelError(f"{label} must be a JSON list of non-empty strings.")
    names = tuple(value)
    if len(names) != len(set(names)):
        raise RndActuatorModelError(f"{label} must not contain duplicates.")
    return names


def validate_rnd_actuator_model(
    model: Mapping[str, Any],
    joint_names: Sequence[str] | None = None,
    *,
    require_sim_replay_validation: bool = False,
    require_command_path_seed: bool = False,
) -> None:
    """Validate the JSON contract used by the runtime command-path model.

    Args:
        model: Parsed runtime-model JSON.
        joint_names: Optional subset required by a particular actuator group.
        require_sim_replay_validation: Require both top-level and per-joint
            simulator replay gates to have passed.
        require_command_path_seed: Reject joints whose measured command path
            is unresolved instead of permitting their explicit identity seed.
    """

    if model.get("schema_version") != RND_ACTUATOR_MODEL_SCHEMA_VERSION:
        raise RndActuatorModelError(
            f"Unsupported schema_version {model.get('schema_version')!r}; expected {RND_ACTUATOR_MODEL_SCHEMA_VERSION}."
        )
    if model.get("model_type") != RND_ACTUATOR_MODEL_TYPE:
        raise RndActuatorModelError(f"Unsupported model_type {model.get('model_type')!r}.")

    physics_hz = _finite_float(model.get("physics_hz"), "physics_hz", minimum=1.0e-9)
    _finite_float(model.get("policy_hz"), "policy_hz", minimum=1.0e-9)
    integration_enabled = model.get("integration_enabled")
    if not isinstance(integration_enabled, bool):
        raise RndActuatorModelError("integration_enabled must be bool.")

    joints = model.get("joints")
    if not isinstance(joints, Mapping) or not joints:
        raise RndActuatorModelError("Actuator model must contain a non-empty joints mapping.")
    joint_order = list(_unique_name_list(model.get("joint_order"), "joint_order"))
    if set(joint_order) != set(joints):
        raise RndActuatorModelError("joint_order and joints keys do not match.")

    application_status = model.get("application_status")
    integration_scope: set[str] = set()
    if integration_enabled:
        if application_status == "sim_replay_validated":
            integration_names = _unique_name_list(
                model.get("integration_joint_names", joint_order), "integration_joint_names"
            )
            fallback_names = _unique_name_list(model.get("fallback_joint_names", []), "fallback_joint_names")
            if set(integration_names) != set(joint_order) or fallback_names:
                raise RndActuatorModelError(
                    "A fully validated actuator model must integrate every joint and have no fallback joints."
                )
        elif application_status == "sim_replay_validated_partial":
            integration_names = _unique_name_list(model.get("integration_joint_names"), "integration_joint_names")
            fallback_names = _unique_name_list(model.get("fallback_joint_names"), "fallback_joint_names")
            if not integration_names or not fallback_names:
                raise RndActuatorModelError(
                    "A partial actuator model requires non-empty integration_joint_names and fallback_joint_names."
                )
            if set(integration_names) & set(fallback_names):
                raise RndActuatorModelError("Integration and fallback joint sets must be disjoint.")
            if set(integration_names) | set(fallback_names) != set(joint_order):
                raise RndActuatorModelError("Integration and fallback joint sets must partition joint_order.")
        else:
            raise RndActuatorModelError(
                "An enabled actuator model must have application_status='sim_replay_validated' or "
                "'sim_replay_validated_partial'."
            )
        integration_scope = set(integration_names)
    elif require_sim_replay_validation:
        raise RndActuatorModelError(
            "Actuator model has not passed simulator replay validation for runtime use or integration is disabled; "
            f"application_status={application_status!r}, integration_enabled={integration_enabled!r}."
        )

    selected = tuple(joint_names) if joint_names is not None else tuple(joint_order)
    if len(selected) != len(set(selected)):
        raise RndActuatorModelError("Requested joint_names must be unique.")
    missing = sorted(set(selected) - set(joints))
    if missing:
        raise RndActuatorModelError(f"Actuator model is missing joints: {missing}.")
    if require_sim_replay_validation:
        outside_scope = sorted(set(selected) - integration_scope)
        if outside_scope:
            raise RndActuatorModelError(f"Actuator integration is not enabled for joints: {outside_scope}.")

    for joint_name in selected:
        joint = joints[joint_name]
        if not isinstance(joint, Mapping):
            raise RndActuatorModelError(f"joints.{joint_name} must be a mapping.")
        command_path = joint.get("command_path")
        quality = joint.get("quality")
        if not isinstance(command_path, Mapping) or not isinstance(quality, Mapping):
            raise RndActuatorModelError(f"joints.{joint_name} requires command_path and quality mappings.")

        residual_delay = _range_pair(
            command_path.get("residual_delay_s_range"),
            f"joints.{joint_name}.command_path.residual_delay_s_range",
        )
        if residual_delay[1] * physics_hz > 100_000:
            raise RndActuatorModelError(f"joints.{joint_name} residual delay would require an unsafe history size.")
        _signed_range_pair(
            command_path.get("residual_position_bias_rad_range", [0.0, 0.0]),
            f"joints.{joint_name}.command_path.residual_position_bias_rad_range",
        )

        thresholds = command_path.get("play_thresholds_rad")
        weights = command_path.get("play_weights")
        if not isinstance(thresholds, list) or not isinstance(weights, list) or len(thresholds) != len(weights):
            raise RndActuatorModelError(
                f"joints.{joint_name} play_thresholds_rad and play_weights must be equal-length lists."
            )
        parsed_thresholds = [
            _finite_float(value, f"joints.{joint_name}.play_thresholds_rad", minimum=0.0) for value in thresholds
        ]
        if parsed_thresholds != sorted(parsed_thresholds):
            raise RndActuatorModelError(f"joints.{joint_name} play thresholds must be sorted ascending.")
        parsed_weights = [_finite_float(value, f"joints.{joint_name}.play_weights", minimum=0.0) for value in weights]
        linear_weight = _finite_float(
            command_path.get("linear_weight"), f"joints.{joint_name}.command_path.linear_weight", minimum=0.0
        )
        if not math.isclose(linear_weight + sum(parsed_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise RndActuatorModelError(
                f"joints.{joint_name} linear and play weights must sum to 1.0, "
                f"got {linear_weight + sum(parsed_weights):.9f}."
            )
        _range_pair(
            command_path.get("play_threshold_scale_range"),
            f"joints.{joint_name}.command_path.play_threshold_scale_range",
        )

        seed_usable = quality.get("command_path_seed_usable")
        replay_validated = quality.get("sim_replay_validated")
        integration_allowed = quality.get("integration_allowed")
        if (
            not isinstance(seed_usable, bool)
            or not isinstance(replay_validated, bool)
            or not isinstance(integration_allowed, bool)
        ):
            raise RndActuatorModelError(
                f"joints.{joint_name} quality flags command_path_seed_usable, sim_replay_validated, and "
                "integration_allowed must be bool."
            )
        if require_command_path_seed and not seed_usable:
            raise RndActuatorModelError(f"Measured command-path seed is unresolved for joint {joint_name}.")
        if require_sim_replay_validation and not replay_validated:
            raise RndActuatorModelError(f"Simulator replay validation has not passed for joint {joint_name}.")
        if require_sim_replay_validation and not integration_allowed:
            raise RndActuatorModelError(f"Actuator integration is not allowed for joint {joint_name}.")

    torque_calibration = model.get("torque_calibration")
    if not isinstance(torque_calibration, Mapping) or torque_calibration.get("available") is not False:
        raise RndActuatorModelError(
            "This schema currently requires torque_calibration.available=false; measured current must not be used as torque."
        )


def load_rnd_actuator_model(
    path: str | Path,
    joint_names: Sequence[str] | None = None,
    *,
    require_sim_replay_validation: bool = False,
    require_command_path_seed: bool = False,
) -> dict[str, Any]:
    """Load and validate an RND actuator runtime-model JSON file."""

    resolved = Path(path).expanduser().resolve()
    try:
        model = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RndActuatorModelError(f"Actuator model does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise RndActuatorModelError(f"Actuator model is not valid JSON: {resolved}: {error}") from error
    validate_rnd_actuator_model(
        model,
        joint_names,
        require_sim_replay_validation=require_sim_replay_validation,
        require_command_path_seed=require_command_path_seed,
    )
    return model


class StatefulCommandPath:
    """Apply fractional delay and generalized play to batched joint targets.

    The state tensors are shaped by environment, joint, and play branch and
    remain on the requested Torch device. ``reset`` fills the complete delay
    history with the current target, preventing a zero-command transient after
    an environment reset.
    """

    def __init__(
        self,
        model: Mapping[str, Any],
        joint_names: Sequence[str],
        num_envs: int,
        device: str | torch.device,
        *,
        step_hz: float | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        sample_randomization: bool = True,
    ):
        validate_rnd_actuator_model(model, joint_names)
        if num_envs <= 0:
            raise RndActuatorModelError("num_envs must be positive.")
        self.model = model
        self.joint_names = tuple(joint_names)
        self.num_envs = int(num_envs)
        self.num_joints = len(self.joint_names)
        self.device = torch.device(device)
        self.dtype = dtype
        self.step_hz = _finite_float(model["physics_hz"] if step_hz is None else step_hz, "step_hz", minimum=1.0e-9)
        self.sample_randomization = bool(sample_randomization)

        joints = model["joints"]
        branch_count = max(1, max(len(joints[name]["command_path"]["play_weights"]) for name in self.joint_names))
        base_thresholds = torch.zeros((self.num_joints, branch_count), dtype=dtype, device=self.device)
        play_weights = torch.zeros_like(base_thresholds)
        linear_weights = torch.empty((self.num_joints,), dtype=dtype, device=self.device)
        delay_ranges = torch.empty((self.num_joints, 2), dtype=dtype, device=self.device)
        position_bias_ranges = torch.empty_like(delay_ranges)
        threshold_scale_ranges = torch.empty_like(delay_ranges)

        for joint_index, joint_name in enumerate(self.joint_names):
            command_path = joints[joint_name]["command_path"]
            thresholds = command_path["play_thresholds_rad"]
            weights = command_path["play_weights"]
            if thresholds:
                base_thresholds[joint_index, : len(thresholds)] = torch.tensor(
                    thresholds, dtype=dtype, device=self.device
                )
                play_weights[joint_index, : len(weights)] = torch.tensor(weights, dtype=dtype, device=self.device)
            linear_weights[joint_index] = float(command_path["linear_weight"])
            delay_ranges[joint_index] = torch.tensor(
                command_path["residual_delay_s_range"], dtype=dtype, device=self.device
            )
            position_bias_ranges[joint_index] = torch.tensor(
                command_path.get("residual_position_bias_rad_range", [0.0, 0.0]),
                dtype=dtype,
                device=self.device,
            )
            threshold_scale_ranges[joint_index] = torch.tensor(
                command_path["play_threshold_scale_range"], dtype=dtype, device=self.device
            )

        self._base_thresholds = base_thresholds
        self._play_weights = play_weights.unsqueeze(0)
        self._linear_weights = linear_weights.unsqueeze(0)
        self._delay_ranges_s = delay_ranges
        self._position_bias_ranges_rad = position_bias_ranges
        self._threshold_scale_ranges = threshold_scale_ranges
        max_delay_samples = float(delay_ranges[:, 1].max().item()) * self.step_hz
        self._history_capacity = max(2, math.ceil(max_delay_samples) + 2)
        self._history = torch.zeros(
            (self._history_capacity, self.num_envs, self.num_joints), dtype=dtype, device=self.device
        )
        self._play_state = torch.zeros((self.num_envs, self.num_joints, branch_count), dtype=dtype, device=self.device)
        self._sampled_delay_samples = torch.zeros((self.num_envs, self.num_joints), dtype=dtype, device=self.device)
        self._sampled_position_bias_rad = torch.zeros_like(self._sampled_delay_samples)
        self._sampled_thresholds = torch.zeros_like(self._play_state)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._env_grid = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.num_joints)
        self._joint_grid = torch.arange(self.num_joints, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        self._write_index = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed))

    @property
    def sampled_delay_s(self) -> torch.Tensor:
        """Sampled residual delay for each environment and joint."""

        return self._sampled_delay_samples / self.step_hz

    @property
    def sampled_play_thresholds_rad(self) -> torch.Tensor:
        """Sampled play half-width for each environment, joint, and branch."""

        return self._sampled_thresholds

    @property
    def sampled_position_bias_rad(self) -> torch.Tensor:
        """Sampled additive residual target bias for each environment and joint."""

        return self._sampled_position_bias_rad

    def set_position_bias_override(self, position_bias_rad: float) -> None:
        """Replace the configured bias range with one fixed diagnostic value."""

        value = _finite_float(position_bias_rad, "position_bias_rad")
        self._position_bias_ranges_rad[:, 0] = value
        self._position_bias_ranges_rad[:, 1] = value
        self._sampled_position_bias_rad.fill_(value)

    def set_delay_override(self, delay_s: float) -> None:
        """Replace the configured delay range with one fixed diagnostic value."""

        value = _finite_float(delay_s, "delay_s", minimum=0.0)
        delay_samples = value * self.step_hz
        required_capacity = max(2, math.ceil(delay_samples) + 2)
        if required_capacity > self._history_capacity:
            if bool(self._initialized.any().item()):
                raise RndActuatorModelError("Delay override requiring a larger history must be set before reset.")
            self._history_capacity = required_capacity
            self._history = torch.zeros(
                (self._history_capacity, self.num_envs, self.num_joints),
                dtype=self.dtype,
                device=self.device,
            )
            self._write_index = 0
        self._delay_ranges_s[:, 0] = value
        self._delay_ranges_s[:, 1] = value
        self._sampled_delay_samples.fill_(delay_samples)

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return ids
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_envs:
            raise RndActuatorModelError(f"env_ids are outside [0, {self.num_envs}).")
        if ids.unique().numel() != ids.numel():
            raise RndActuatorModelError("env_ids must not contain duplicates.")
        return ids

    def _targets(self, value: torch.Tensor, expected_rows: int, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise RndActuatorModelError(f"{label} must be a torch.Tensor.")
        if value.shape != (expected_rows, self.num_joints):
            raise RndActuatorModelError(
                f"{label} must have shape {(expected_rows, self.num_joints)}, got {tuple(value.shape)}."
            )
        value = value.to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(value).all():
            raise RndActuatorModelError(f"{label} contains non-finite values.")
        return value

    def reset(
        self,
        initial_targets_rad: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Reset selected environments and sample their model parameters."""

        ids = self._env_ids(env_ids)
        initial = self._targets(initial_targets_rad, ids.numel(), "initial_targets_rad")
        if ids.numel() == 0:
            return

        delay_low = self._delay_ranges_s[:, 0].unsqueeze(0)
        delay_high = self._delay_ranges_s[:, 1].unsqueeze(0)
        bias_low = self._position_bias_ranges_rad[:, 0].unsqueeze(0)
        bias_high = self._position_bias_ranges_rad[:, 1].unsqueeze(0)
        if self.sample_randomization:
            random_delay = torch.rand(
                (ids.numel(), self.num_joints), dtype=self.dtype, device=self.device, generator=self._generator
            )
            sampled_delay_s = delay_low + random_delay * (delay_high - delay_low)
            random_bias = torch.rand(
                (ids.numel(), self.num_joints), dtype=self.dtype, device=self.device, generator=self._generator
            )
            sampled_bias_rad = bias_low + random_bias * (bias_high - bias_low)
            random_scale = torch.rand(
                (ids.numel(), self.num_joints), dtype=self.dtype, device=self.device, generator=self._generator
            )
            scale_low = self._threshold_scale_ranges[:, 0].unsqueeze(0)
            scale_high = self._threshold_scale_ranges[:, 1].unsqueeze(0)
            sampled_scale = scale_low + random_scale * (scale_high - scale_low)
        else:
            sampled_delay_s = 0.5 * (delay_low + delay_high).expand(ids.numel(), -1)
            sampled_bias_rad = 0.5 * (bias_low + bias_high).expand(ids.numel(), -1)
            sampled_scale = torch.ones((ids.numel(), self.num_joints), dtype=self.dtype, device=self.device)
        self._sampled_delay_samples[ids] = sampled_delay_s * self.step_hz
        self._sampled_position_bias_rad[ids] = sampled_bias_rad
        self._sampled_thresholds[ids] = self._base_thresholds.unsqueeze(0) * sampled_scale.unsqueeze(-1)

        self._history[:, ids, :] = initial.unsqueeze(0)
        self._play_state[ids] = initial.unsqueeze(-1)
        self._initialized[ids] = True

    def transform(self, targets_rad: torch.Tensor) -> torch.Tensor:
        """Transform one physics-step batch of raw position targets."""

        target = self._targets(targets_rad, self.num_envs, "targets_rad")
        if not bool(self._initialized.all().item()):
            missing = torch.nonzero(~self._initialized, as_tuple=False).flatten().tolist()
            raise RndActuatorModelError(f"Call reset before transform for environments: {missing}.")

        self._history[self._write_index] = target
        delay_floor = torch.floor(self._sampled_delay_samples).to(dtype=torch.long)
        fraction = self._sampled_delay_samples - delay_floor
        recent_indices = torch.remainder(self._write_index - delay_floor, self._history_capacity)
        older_indices = torch.remainder(recent_indices - 1, self._history_capacity)
        recent = self._history[recent_indices, self._env_grid, self._joint_grid]
        older = self._history[older_indices, self._env_grid, self._joint_grid]
        delayed = recent + fraction * (older - recent)

        lower = delayed.unsqueeze(-1) - self._sampled_thresholds
        upper = delayed.unsqueeze(-1) + self._sampled_thresholds
        self._play_state = torch.minimum(torch.maximum(self._play_state, lower), upper)
        output = (
            self._linear_weights * delayed
            + torch.sum(self._play_weights * self._play_state, dim=-1)
            + self._sampled_position_bias_rad
        )
        self._write_index = (self._write_index + 1) % self._history_capacity
        return output


def compute_explicit_pd_effort(
    position_target: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    stiffness: torch.Tensor | float,
    damping: torch.Tensor | float,
    effort_limit_nm: torch.Tensor | float,
    *,
    velocity_target: torch.Tensor | None = None,
    feedforward_effort: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute and symmetrically clip explicit PD effort."""

    if position_target.shape != joint_position.shape or joint_velocity.shape != joint_position.shape:
        raise RndActuatorModelError("position_target, joint_position, and joint_velocity shapes must match.")
    desired_velocity = torch.zeros_like(joint_velocity) if velocity_target is None else velocity_target
    feedforward = torch.zeros_like(joint_position) if feedforward_effort is None else feedforward_effort
    if desired_velocity.shape != joint_velocity.shape or feedforward.shape != joint_position.shape:
        raise RndActuatorModelError("velocity_target and feedforward_effort must match the joint tensor shape.")
    effort = (
        stiffness * (position_target - joint_position) + damping * (desired_velocity - joint_velocity) + feedforward
    )
    limit = torch.as_tensor(effort_limit_nm, dtype=effort.dtype, device=effort.device)
    if not torch.isfinite(limit).all() or bool((limit < 0.0).any().item()):
        raise RndActuatorModelError("effort_limit_nm must be finite and non-negative.")
    return torch.minimum(torch.maximum(effort, -limit), limit)
