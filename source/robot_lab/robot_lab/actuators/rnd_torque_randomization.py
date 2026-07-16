"""Episode-level motor-strength and Coulomb-friction randomization for RND STEP."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


RND_TORQUE_RANDOMIZATION_SCHEMA_VERSION = 2
RND_TORQUE_RANDOMIZATION_MODEL_TYPE = "rnd_joint_torque_randomization"


class RndTorqueRandomizationError(ValueError):
    """Raised when torque-randomization evidence or tensors are invalid."""


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RndTorqueRandomizationError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise RndTorqueRandomizationError(f"{label} must be finite, got {result!r}.")
    if minimum is not None and result < minimum:
        raise RndTorqueRandomizationError(f"{label} must be >= {minimum}, got {result}.")
    return result


def _range(value: Any, label: str, *, positive_lower: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RndTorqueRandomizationError(f"{label} must be a two-element JSON list.")
    lower = _finite(value[0], f"{label}[0]", minimum=0.0)
    upper = _finite(value[1], f"{label}[1]", minimum=0.0)
    if lower > upper:
        raise RndTorqueRandomizationError(f"{label} lower bound exceeds upper bound: {value!r}.")
    if positive_lower and lower <= 0.0:
        raise RndTorqueRandomizationError(f"{label}[0] must be positive.")
    return lower, upper


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise RndTorqueRandomizationError(f"{label} must be a JSON list of non-empty strings.")
    names = tuple(value)
    if len(names) != len(set(names)):
        raise RndTorqueRandomizationError(f"{label} must not contain duplicates.")
    return names


def _bilateral_group_name(joint_name: str) -> str:
    """Return a stable group name shared by mirrored RND leg joints."""

    for prefix in ("R_Leg_", "L_Leg_"):
        if joint_name.startswith(prefix):
            return f"bilateral:{joint_name.removeprefix(prefix)}"
    return f"joint:{joint_name}"


def _stream_seed(seed: int, stream_name: str, group_name: str) -> int:
    """Derive a process-stable Torch seed for one randomization stream and joint group."""

    payload = f"{int(seed)}:{stream_name}:{group_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def validate_rnd_torque_randomization(model: Mapping[str, Any], joint_names: Sequence[str] | None = None) -> None:
    """Validate the compact evidence-backed torque-randomization contract."""

    if model.get("schema_version") != RND_TORQUE_RANDOMIZATION_SCHEMA_VERSION:
        raise RndTorqueRandomizationError(
            f"Unsupported schema_version {model.get('schema_version')!r}; "
            f"expected {RND_TORQUE_RANDOMIZATION_SCHEMA_VERSION}."
        )
    if model.get("model_type") != RND_TORQUE_RANDOMIZATION_MODEL_TYPE:
        raise RndTorqueRandomizationError(f"Unsupported model_type {model.get('model_type')!r}.")
    if model.get("integration_enabled") is not True:
        raise RndTorqueRandomizationError("Torque randomization must explicitly set integration_enabled=true.")
    if model.get("sample_per_episode") is not True:
        raise RndTorqueRandomizationError("Torque randomization must explicitly set sample_per_episode=true.")
    if model.get("sample_bilateral_pairs_with_shared_quantile") is not True:
        raise RndTorqueRandomizationError(
            "Torque randomization must explicitly set sample_bilateral_pairs_with_shared_quantile=true."
        )
    if model.get("viscous_friction_enabled") is not False:
        raise RndTorqueRandomizationError("Unidentified viscous friction must remain disabled.")
    if model.get("static_breakaway_enabled") is not False:
        raise RndTorqueRandomizationError(
            "Static breakaway must remain disabled because the command-path play model already contains hysteresis."
        )

    strength_range = _range(model.get("motor_strength_scale_range"), "motor_strength_scale_range", positive_lower=True)
    if strength_range[1] > 2.0:
        raise RndTorqueRandomizationError("motor_strength_scale_range upper bound must not exceed 2.0.")
    transition_range = _range(
        model.get("friction_transition_velocity_rad_s_range"),
        "friction_transition_velocity_rad_s_range",
        positive_lower=True,
    )
    if transition_range[1] > math.pi:
        raise RndTorqueRandomizationError("Friction transition velocity exceeds the supported range.")

    joints = model.get("joints")
    if not isinstance(joints, Mapping) or not joints:
        raise RndTorqueRandomizationError("Torque randomization requires a non-empty joints mapping.")
    joint_order = _names(model.get("joint_order"), "joint_order")
    if set(joint_order) != set(joints):
        raise RndTorqueRandomizationError("joint_order and joints keys do not match.")
    selected = tuple(joint_order) if joint_names is None else tuple(joint_names)
    if len(selected) != len(set(selected)):
        raise RndTorqueRandomizationError("Requested joint names must be unique.")
    missing = sorted(set(selected) - set(joints))
    if missing:
        raise RndTorqueRandomizationError(f"Torque randomization is missing joints: {missing}.")

    for joint_name in selected:
        joint = joints[joint_name]
        if not isinstance(joint, Mapping):
            raise RndTorqueRandomizationError(f"joints.{joint_name} must be a mapping.")
        evidence_status = joint.get("evidence_status")
        if evidence_status not in ("measured_quality_pass", "unidentified_prior"):
            raise RndTorqueRandomizationError(f"joints.{joint_name} has unsupported evidence_status.")
        torque_range = _range(joint.get("coulomb_torque_range_nm"), f"joints.{joint_name}.coulomb_torque_range_nm")
        if torque_range[1] > 1.0:
            raise RndTorqueRandomizationError(f"joints.{joint_name} Coulomb torque exceeds the 1.0 Nm safety bound.")
        nominal = joint.get("measured_coulomb_torque_nm")
        quality_pass = joint.get("source_quality_pass")
        if evidence_status == "measured_quality_pass":
            nominal_value = _finite(nominal, f"joints.{joint_name}.measured_coulomb_torque_nm", minimum=0.0)
            if quality_pass is not True:
                raise RndTorqueRandomizationError(f"joints.{joint_name} measured evidence must have passed quality.")
            if not torque_range[0] <= nominal_value <= torque_range[1]:
                raise RndTorqueRandomizationError(
                    f"joints.{joint_name} measured Coulomb torque is outside its randomization range."
                )
        else:
            if nominal is not None or quality_pass is not False:
                raise RndTorqueRandomizationError(
                    f"joints.{joint_name} unidentified prior must retain nominal=null and source_quality_pass=false."
                )


def load_rnd_torque_randomization(path: str | Path, joint_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Load and validate a torque-randomization JSON file."""

    resolved = Path(path).expanduser().resolve()
    try:
        model = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RndTorqueRandomizationError(f"Torque-randomization model does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise RndTorqueRandomizationError(
            f"Torque-randomization model is not valid JSON: {resolved}: {error}"
        ) from error
    if not isinstance(model, dict):
        raise RndTorqueRandomizationError(f"Torque-randomization model must contain a JSON object: {resolved}")
    validate_rnd_torque_randomization(model, joint_names)
    return model


class EpisodeTorqueRandomizer:
    """Sample episode torque uncertainty without injecting artificial left/right limping."""

    def __init__(
        self,
        model: Mapping[str, Any],
        joint_names: Sequence[str],
        num_envs: int,
        device: str | torch.device,
        *,
        seed: int = 0,
        sample_randomization: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        validate_rnd_torque_randomization(model, joint_names)
        if num_envs <= 0:
            raise RndTorqueRandomizationError("num_envs must be positive.")
        self.joint_names = tuple(joint_names)
        self.num_envs = int(num_envs)
        self.num_joints = len(self.joint_names)
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_randomization = bool(sample_randomization)

        joints = model["joints"]
        self._friction_ranges_nm = torch.tensor(
            [joints[name]["coulomb_torque_range_nm"] for name in self.joint_names],
            dtype=dtype,
            device=self.device,
        )
        self._strength_range = torch.tensor(model["motor_strength_scale_range"], dtype=dtype, device=self.device)
        self._transition_range_rad_s = torch.tensor(
            model["friction_transition_velocity_rad_s_range"], dtype=dtype, device=self.device
        )
        self._sampled_coulomb_torque_nm = torch.zeros((self.num_envs, self.num_joints), dtype=dtype, device=self.device)
        self._sampled_motor_strength_scale = torch.ones_like(self._sampled_coulomb_torque_nm)
        self._sampled_transition_velocity_rad_s = torch.ones_like(self._sampled_coulomb_torque_nm)
        self._last_friction_effort_nm = torch.zeros_like(self._sampled_coulomb_torque_nm)
        self._last_scaled_motor_effort_nm = torch.zeros_like(self._sampled_coulomb_torque_nm)
        self._joint_group_names = tuple(_bilateral_group_name(name) for name in self.joint_names)
        self._group_names = tuple(dict.fromkeys(self._joint_group_names))
        group_indices = {name: index for index, name in enumerate(self._group_names)}
        self._joint_group_indices = torch.tensor(
            [group_indices[name] for name in self._joint_group_names],
            dtype=torch.long,
            device=self.device,
        )
        self._generators: dict[str, dict[str, torch.Generator]] = {}
        for stream_name in ("friction", "strength", "transition"):
            self._generators[stream_name] = {}
            for group_name in self._group_names:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(_stream_seed(seed, stream_name, group_name))
                self._generators[stream_name][group_name] = generator
        self.reset()

    @property
    def sampled_coulomb_torque_nm(self) -> torch.Tensor:
        return self._sampled_coulomb_torque_nm

    @property
    def sampled_motor_strength_scale(self) -> torch.Tensor:
        return self._sampled_motor_strength_scale

    @property
    def sampled_transition_velocity_rad_s(self) -> torch.Tensor:
        return self._sampled_transition_velocity_rad_s

    @property
    def last_friction_effort_nm(self) -> torch.Tensor:
        return self._last_friction_effort_nm

    @property
    def last_scaled_motor_effort_nm(self) -> torch.Tensor:
        return self._last_scaled_motor_effort_nm

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return ids
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_envs:
            raise RndTorqueRandomizationError(f"env_ids are outside [0, {self.num_envs}).")
        if ids.unique().numel() != ids.numel():
            raise RndTorqueRandomizationError("env_ids must not contain duplicates.")
        return ids

    def _sample_grouped_unit(self, count: int, stream_name: str) -> torch.Tensor:
        """Sample one quantile per bilateral joint group and broadcast it to its members."""

        grouped = torch.empty((count, len(self._group_names)), dtype=self.dtype, device=self.device)
        for group_index, group_name in enumerate(self._group_names):
            grouped[:, group_index] = torch.rand(
                count,
                dtype=self.dtype,
                device=self.device,
                generator=self._generators[stream_name][group_name],
            )
        return grouped[:, self._joint_group_indices]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Resample selected environments; midpoint mode remains deterministic."""

        ids = self._env_ids(env_ids)
        if ids.numel() == 0:
            return
        count = ids.numel()
        if self.sample_randomization:
            friction_unit = self._sample_grouped_unit(count, "friction")
            strength_unit = self._sample_grouped_unit(count, "strength")
            transition_unit = self._sample_grouped_unit(count, "transition")
        else:
            friction_unit = torch.full((count, self.num_joints), 0.5, dtype=self.dtype, device=self.device)
            strength_unit = torch.full_like(friction_unit, 0.5)
            transition_unit = torch.full_like(friction_unit, 0.5)

        friction_low = self._friction_ranges_nm[:, 0].unsqueeze(0)
        friction_high = self._friction_ranges_nm[:, 1].unsqueeze(0)
        self._sampled_coulomb_torque_nm[ids] = friction_low + friction_unit * (friction_high - friction_low)
        self._sampled_motor_strength_scale[ids] = self._strength_range[0] + strength_unit * (
            self._strength_range[1] - self._strength_range[0]
        )
        self._sampled_transition_velocity_rad_s[ids] = self._transition_range_rad_s[0] + transition_unit * (
            self._transition_range_rad_s[1] - self._transition_range_rad_s[0]
        )
        self._last_friction_effort_nm[ids] = 0.0
        self._last_scaled_motor_effort_nm[ids] = 0.0

    def apply(
        self,
        motor_effort_nm: torch.Tensor,
        joint_velocity_rad_s: torch.Tensor,
        effort_limit_nm: torch.Tensor | float,
    ) -> torch.Tensor:
        """Apply strength uncertainty and a velocity-opposing smooth Coulomb torque."""

        expected = (self.num_envs, self.num_joints)
        if motor_effort_nm.shape != expected or joint_velocity_rad_s.shape != expected:
            raise RndTorqueRandomizationError(f"Motor effort and joint velocity must both have shape {expected}.")
        motor_effort = motor_effort_nm.to(device=self.device, dtype=self.dtype)
        joint_velocity = joint_velocity_rad_s.to(device=self.device, dtype=self.dtype)
        limit = torch.as_tensor(effort_limit_nm, dtype=self.dtype, device=self.device)
        if not torch.isfinite(motor_effort).all() or not torch.isfinite(joint_velocity).all():
            raise RndTorqueRandomizationError("Motor effort and joint velocity must be finite.")
        if not torch.isfinite(limit).all() or bool((limit < 0.0).any().item()):
            raise RndTorqueRandomizationError("effort_limit_nm must be finite and non-negative.")

        scaled_motor = torch.clamp(
            motor_effort * self._sampled_motor_strength_scale,
            min=-limit,
            max=limit,
        )
        friction_effort = self._sampled_coulomb_torque_nm * torch.tanh(
            joint_velocity / self._sampled_transition_velocity_rad_s
        )
        self._last_scaled_motor_effort_nm.copy_(scaled_motor)
        self._last_friction_effort_nm.copy_(friction_effort)
        return scaled_motor - friction_effort
