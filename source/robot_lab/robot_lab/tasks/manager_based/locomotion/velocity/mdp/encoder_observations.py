"""Stateful Dynamixel encoder observations for the opt-in RND STEP task."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg


_REPO_ROOT = Path(__file__).resolve().parents[8]
RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH = (
    _REPO_ROOT / "scripts" / "tools" / "config" / "rnd_encoder_observation_runtime.json"
)
RND_ENCODER_OBSERVATION_MODEL_PATH = RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH

RND_DYNAMIXEL_ENCODER_JOINT_ORDER = (
    "R_Leg_hip_yaw",
    "R_Leg_hip_roll",
    "R_Leg_hip_pitch",
    "R_Leg_knee",
    "R_Leg_ankle_pitch",
    "R_Leg_ankle_roll",
    "L_Leg_hip_yaw",
    "L_Leg_hip_roll",
    "L_Leg_hip_pitch",
    "L_Leg_knee",
    "L_Leg_ankle_pitch",
    "L_Leg_ankle_roll",
)
DYNAMIXEL_POSITION_QUANTUM_RAD = 2.0 * math.pi / 4096.0
DYNAMIXEL_VELOCITY_QUANTUM_RPM = 0.229
DYNAMIXEL_VELOCITY_QUANTUM_RAD_S = DYNAMIXEL_VELOCITY_QUANTUM_RPM * 2.0 * math.pi / 60.0

_MODEL_TYPE = "rnd_dynamixel_encoder_policy_observation"
_POLICY_HZ = 50.0
_VELOCITY_SCALE = 0.05
_SAMPLE_AGE_RANGE_S = (0.0, 0.005)
_ZERO_OFFSET_RANGE_RAD = (-0.005, 0.005)
_UNSEEN_STEP = torch.iinfo(torch.int64).min

_Q_REL_DEFINITION = "quantize(q_delayed + zero_offset) - default_joint_position"
_DQ_SCALED_DEFINITION = "quantize(dq_delayed) * velocity_scale"
_DELAY_DEFINITION = "delayed=(1-age/step_dt)*current+(age/step_dt)*previous"
_LIMITATIONS = (
    "Zero-offset and sample-age ranges are training priors, not measured confidence intervals.",
    "Delay uses only previous/current 50 Hz policy states, so intra-step motion is linearized.",
    "ZOH is keyed by env.common_step_counter and does not model asynchronous per-servo packet timing.",
    "Packet loss, bus skew, encoder wraparound, and calibration-direction errors are outside this model.",
)


class RndDynamixelEncoderObservationModelError(ValueError):
    """Raised when the encoder observation runtime artifact violates its schema."""


@dataclass(frozen=True)
class RndDynamixelEncoderObservationModel:
    """Strictly validated runtime parameters for the policy encoder observation."""

    path: Path
    joint_order: tuple[str, ...]
    policy_hz: float
    output_dimension: int
    velocity_scale: float
    position_quantum_rad: float
    velocity_quantum_rad_s: float
    sample_age_range_s: tuple[float, float]
    zero_offset_range_rad: tuple[float, float]


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RndDynamixelEncoderObservationModelError(f"Duplicate JSON key is not allowed: {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RndDynamixelEncoderObservationModelError(f"Non-finite JSON number is not allowed: {value}.")


def _object(value: Any, label: str, expected_keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be an object; got {value!r}.")
    expected = set(expected_keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise RndDynamixelEncoderObservationModelError(f"{label} has invalid keys ({', '.join(details)}).")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be finite; got {value!r}.")
    return result


def _exact_number(value: Any, expected: float, label: str, *, tolerance: float = 1.0e-12) -> float:
    result = _number(value, label)
    if not math.isclose(result, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be {expected!r}; got {result!r}.")
    return result


def _exact_integer(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RndDynamixelEncoderObservationModelError(f"{label} must be integer {expected}; got {value!r}.")
    return value


def _exact_string(value: Any, expected: str, label: str) -> str:
    if value != expected:
        raise RndDynamixelEncoderObservationModelError(f"{label} must be {expected!r}; got {value!r}.")
    return expected


def _exact_string_sequence(value: Any, expected: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be a string array; got {value!r}.")
    result = tuple(value)
    if result != expected or not all(isinstance(item, str) for item in result):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be {list(expected)!r}; got {value!r}.")
    return result


def _exact_range(value: Any, expected: tuple[float, float], label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise RndDynamixelEncoderObservationModelError(f"{label} must contain two numeric endpoints; got {value!r}.")
    result = (_number(value[0], f"{label}[0]"), _number(value[1], f"{label}[1]"))
    if result[0] > result[1]:
        raise RndDynamixelEncoderObservationModelError(f"{label} minimum exceeds maximum: {result!r}.")
    if not all(math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12) for actual, target in zip(result, expected)):
        raise RndDynamixelEncoderObservationModelError(f"{label} must be {expected!r}; got {result!r}.")
    return result


def load_rnd_dynamixel_encoder_observation_model(
    path: str | Path = RND_DYNAMIXEL_ENCODER_OBSERVATION_MODEL_PATH,
) -> RndDynamixelEncoderObservationModel:
    """Load the encoder observation artifact and reject any schema or semantic drift."""

    model_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            model_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError as error:
        raise RndDynamixelEncoderObservationModelError(
            f"Dynamixel encoder observation model does not exist: {model_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise RndDynamixelEncoderObservationModelError(
            f"Dynamixel encoder observation model is not valid JSON: {model_path}: {error}"
        ) from error

    root = _object(
        payload,
        "runtime model",
        (
            "schema_version",
            "model_type",
            "integration_enabled",
            "joint_order",
            "policy_hz",
            "policy_observation",
            "encoder_quantization",
            "sample_age",
            "zero_offset",
            "zoh",
            "quality",
            "limitations",
        ),
    )
    _exact_integer(root["schema_version"], 1, "schema_version")
    _exact_string(root["model_type"], _MODEL_TYPE, "model_type")
    if root["integration_enabled"] is not True:
        raise RndDynamixelEncoderObservationModelError("integration_enabled must be true.")
    joint_order = _exact_string_sequence(root["joint_order"], RND_DYNAMIXEL_ENCODER_JOINT_ORDER, "joint_order")
    policy_hz = _exact_number(root["policy_hz"], _POLICY_HZ, "policy_hz")

    policy = _object(
        root["policy_observation"],
        "policy_observation",
        (
            "output_order",
            "position_dimensions",
            "velocity_dimensions",
            "total_dimensions",
            "velocity_scale",
            "q_rel_definition",
            "dq_scaled_definition",
        ),
    )
    _exact_string_sequence(policy["output_order"], ("q_rel", "dq_scaled"), "policy_observation.output_order")
    _exact_integer(policy["position_dimensions"], 12, "policy_observation.position_dimensions")
    _exact_integer(policy["velocity_dimensions"], 12, "policy_observation.velocity_dimensions")
    output_dimension = _exact_integer(policy["total_dimensions"], 24, "policy_observation.total_dimensions")
    velocity_scale = _exact_number(policy["velocity_scale"], _VELOCITY_SCALE, "policy_observation.velocity_scale")
    _exact_string(policy["q_rel_definition"], _Q_REL_DEFINITION, "policy_observation.q_rel_definition")
    _exact_string(policy["dq_scaled_definition"], _DQ_SCALED_DEFINITION, "policy_observation.dq_scaled_definition")

    quantization = _object(root["encoder_quantization"], "encoder_quantization", ("position", "velocity"))
    position_quantization = _object(
        quantization["position"],
        "encoder_quantization.position",
        ("ticks_per_revolution", "quantum_rad", "rounding", "source", "quality"),
    )
    _exact_integer(
        position_quantization["ticks_per_revolution"], 4096, "encoder_quantization.position.ticks_per_revolution"
    )
    position_quantum_rad = _exact_number(
        position_quantization["quantum_rad"],
        DYNAMIXEL_POSITION_QUANTUM_RAD,
        "encoder_quantization.position.quantum_rad",
        tolerance=1.0e-15,
    )
    _exact_string(position_quantization["rounding"], "nearest_ties_to_even", "encoder_quantization.position.rounding")
    _exact_string(
        position_quantization["source"],
        "Dynamixel MX-106 4096-count position unit",
        "encoder_quantization.position.source",
    )
    _exact_string(
        position_quantization["quality"], "vendor_unit_and_repository_mapping", "encoder_quantization.position.quality"
    )

    velocity_quantization = _object(
        quantization["velocity"],
        "encoder_quantization.velocity",
        ("quantum_rpm", "quantum_rad_s", "rounding", "source", "quality"),
    )
    _exact_number(
        velocity_quantization["quantum_rpm"],
        DYNAMIXEL_VELOCITY_QUANTUM_RPM,
        "encoder_quantization.velocity.quantum_rpm",
    )
    velocity_quantum_rad_s = _exact_number(
        velocity_quantization["quantum_rad_s"],
        DYNAMIXEL_VELOCITY_QUANTUM_RAD_S,
        "encoder_quantization.velocity.quantum_rad_s",
        tolerance=1.0e-15,
    )
    _exact_string(velocity_quantization["rounding"], "nearest_ties_to_even", "encoder_quantization.velocity.rounding")
    _exact_string(
        velocity_quantization["source"],
        "Dynamixel MX-106 present-velocity unit 0.229 rpm",
        "encoder_quantization.velocity.source",
    )
    _exact_string(
        velocity_quantization["quality"], "vendor_unit_and_repository_mapping", "encoder_quantization.velocity.quality"
    )

    sample_age = _object(
        root["sample_age"],
        "sample_age",
        (
            "distribution",
            "range_s",
            "sampling_scope",
            "shared_between_q_and_dq",
            "definition",
            "interpolation",
            "source",
            "quality",
        ),
    )
    _exact_string(sample_age["distribution"], "uniform", "sample_age.distribution")
    sample_age_range_s = _exact_range(sample_age["range_s"], _SAMPLE_AGE_RANGE_S, "sample_age.range_s")
    _exact_string(sample_age["sampling_scope"], "per_episode_per_environment_per_joint", "sample_age.sampling_scope")
    if sample_age["shared_between_q_and_dq"] is not True:
        raise RndDynamixelEncoderObservationModelError("sample_age.shared_between_q_and_dq must be true.")
    _exact_string(sample_age["definition"], "age_of_encoder_sample_at_policy_observation_time", "sample_age.definition")
    _exact_string(sample_age["interpolation"], _DELAY_DEFINITION, "sample_age.interpolation")
    _exact_string(sample_age["source"], "training_prior_not_measured", "sample_age.source")
    _exact_string(sample_age["quality"], "assumed_for_training_only", "sample_age.quality")

    zero_offset = _object(
        root["zero_offset"],
        "zero_offset",
        (
            "distribution",
            "range_rad",
            "sampling_scope",
            "application",
            "source",
            "quality",
            "replaces_iid_position_noise_range_rad",
        ),
    )
    _exact_string(zero_offset["distribution"], "uniform", "zero_offset.distribution")
    zero_offset_range_rad = _exact_range(zero_offset["range_rad"], _ZERO_OFFSET_RANGE_RAD, "zero_offset.range_rad")
    _exact_string(zero_offset["sampling_scope"], "per_episode_per_environment_per_joint", "zero_offset.sampling_scope")
    _exact_string(
        zero_offset["application"],
        "added_to_delayed_absolute_position_before_quantization",
        "zero_offset.application",
    )
    _exact_string(zero_offset["source"], "training_prior_not_measured", "zero_offset.source")
    _exact_string(zero_offset["quality"], "assumed_for_training_only", "zero_offset.quality")
    previous_iid_range = _exact_range(
        zero_offset["replaces_iid_position_noise_range_rad"],
        (-0.01, 0.01),
        "zero_offset.replaces_iid_position_noise_range_rad",
    )
    if max(abs(value) for value in zero_offset_range_rad) >= max(abs(value) for value in previous_iid_range):
        raise RndDynamixelEncoderObservationModelError(
            "zero_offset.range_rad must be strictly narrower than the replaced iid position-noise range."
        )

    zoh = _object(root["zoh"], "zoh", ("clock", "hold_scope", "duplicate_compute"))
    _exact_string(zoh["clock"], "env.common_step_counter", "zoh.clock")
    _exact_string(zoh["hold_scope"], "one_output_per_policy_step", "zoh.hold_scope")
    _exact_string(
        zoh["duplicate_compute"],
        "return_cached_output_without_state_advance",
        "zoh.duplicate_compute",
    )

    quality = _object(
        root["quality"],
        "quality",
        ("runtime_status", "hardware_measurement_used_for_priors", "intended_use"),
    )
    _exact_string(quality["runtime_status"], "training_prior_not_measured", "quality.runtime_status")
    if quality["hardware_measurement_used_for_priors"] is not False:
        raise RndDynamixelEncoderObservationModelError("quality.hardware_measurement_used_for_priors must be false.")
    _exact_string(quality["intended_use"], "policy_training_domain_randomization", "quality.intended_use")
    _exact_string_sequence(root["limitations"], _LIMITATIONS, "limitations")

    return RndDynamixelEncoderObservationModel(
        path=model_path,
        joint_order=joint_order,
        policy_hz=policy_hz,
        output_dimension=output_dimension,
        velocity_scale=velocity_scale,
        position_quantum_rad=position_quantum_rad,
        velocity_quantum_rad_s=velocity_quantum_rad_s,
        sample_age_range_s=sample_age_range_s,
        zero_offset_range_rad=zero_offset_range_rad,
    )


def load_rnd_encoder_observation_model(
    path: str | Path = RND_ENCODER_OBSERVATION_MODEL_PATH,
) -> RndDynamixelEncoderObservationModel:
    """Compatibility entry point named after the runtime artifact."""

    return load_rnd_dynamixel_encoder_observation_model(path)


def quantize_to_increment(values: torch.Tensor, increment: float) -> torch.Tensor:
    """Round a floating-point tensor to the nearest increment using Torch tie handling."""

    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("values must be a floating-point torch.Tensor.")
    if not math.isfinite(increment) or increment <= 0.0:
        raise ValueError(f"increment must be finite and positive; got {increment!r}.")
    return torch.round(values / increment) * increment


class DynamixelEncoderObservationState:
    """Pure-Torch state for delayed, quantized, policy-rate encoder observations."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_joints: int,
        step_dt: float,
        position_quantum_rad: float,
        velocity_quantum_rad_s: float,
        velocity_scale: float,
        sample_age_range_s: tuple[float, float],
        zero_offset_range_rad: tuple[float, float],
        sample_randomization: bool,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if num_envs <= 0 or num_joints <= 0:
            raise ValueError(f"num_envs and num_joints must be positive; got {num_envs}, {num_joints}.")
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(f"step_dt must be finite and positive; got {step_dt!r}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; got {dtype}.")
        if not isinstance(sample_randomization, bool):
            raise TypeError("sample_randomization must be bool.")
        self._validate_range(sample_age_range_s, "sample_age_range_s", minimum=0.0)
        self._validate_range(zero_offset_range_rad, "zero_offset_range_rad")
        if sample_age_range_s[1] > step_dt + 1.0e-12:
            raise ValueError(
                "sample_age_range_s maximum cannot exceed step_dt when only previous/current policy states exist."
            )
        for value, label in (
            (position_quantum_rad, "position_quantum_rad"),
            (velocity_quantum_rad_s, "velocity_quantum_rad_s"),
            (velocity_scale, "velocity_scale"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive; got {value!r}.")

        self.num_envs = int(num_envs)
        self.num_joints = int(num_joints)
        self.step_dt = float(step_dt)
        self.position_quantum_rad = float(position_quantum_rad)
        self.velocity_quantum_rad_s = float(velocity_quantum_rad_s)
        self.velocity_scale = float(velocity_scale)
        self.sample_age_range_s = tuple(float(value) for value in sample_age_range_s)
        self.zero_offset_range_rad = tuple(float(value) for value in zero_offset_range_rad)
        self.sample_randomization = sample_randomization
        self.device = torch.device(device)
        self.dtype = dtype

        state_shape = (self.num_envs, self.num_joints)
        self.previous_position = torch.zeros(state_shape, device=self.device, dtype=self.dtype)
        self.current_position = torch.zeros_like(self.previous_position)
        self.previous_velocity = torch.zeros_like(self.previous_position)
        self.current_velocity = torch.zeros_like(self.previous_position)
        self.zero_offset_rad = torch.zeros_like(self.previous_position)
        self.sample_age_s = torch.zeros_like(self.previous_position)
        self.cached_output = torch.zeros((self.num_envs, 2 * self.num_joints), device=self.device, dtype=self.dtype)
        self.last_step_counter = torch.full((self.num_envs,), _UNSEEN_STEP, device=self.device, dtype=torch.int64)

    @staticmethod
    def _validate_range(values: tuple[float, float], label: str, minimum: float | None = None) -> None:
        if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{label} must contain two finite values; got {values!r}.")
        if values[0] > values[1]:
            raise ValueError(f"{label} minimum exceeds maximum: {values!r}.")
        if minimum is not None and values[0] < minimum:
            raise ValueError(f"{label} minimum must be at least {minimum}; got {values!r}.")

    def _state_tensor(self, value: torch.Tensor, label: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        expected_shape = (self.num_envs, self.num_joints)
        if result.shape != expected_shape:
            raise ValueError(f"{label} must have shape {expected_shape}; got {tuple(result.shape)}.")
        return result

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        result = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if result.numel() == 0:
            return result
        if bool(torch.any(result < 0)) or bool(torch.any(result >= self.num_envs)):
            raise IndexError(f"env_ids must lie in [0, {self.num_envs - 1}]; got {result}.")
        return torch.unique(result, sorted=False)

    def _uniform(self, shape: tuple[int, int], value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        if low == high:
            return torch.full(shape, low, device=self.device, dtype=self.dtype)
        return torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(low, high)

    def _compose_output(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        default_position: torch.Tensor,
        zero_offset: torch.Tensor,
    ) -> torch.Tensor:
        measured_position = quantize_to_increment(position + zero_offset, self.position_quantum_rad)
        measured_velocity = quantize_to_increment(velocity, self.velocity_quantum_rad_s)
        return torch.cat((measured_position - default_position, measured_velocity * self.velocity_scale), dim=-1)

    def reset(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        default_position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        fixed_zero_offset_rad: torch.Tensor | None = None,
        fixed_sample_age_s: torch.Tensor | None = None,
    ) -> None:
        """Resample selected episodes and prefill all state from current measurements."""

        position_tensor = self._state_tensor(position, "position")
        velocity_tensor = self._state_tensor(velocity, "velocity")
        default_tensor = self._state_tensor(default_position, "default_position")
        ids = self._env_ids(env_ids)
        if ids.numel() == 0:
            return
        if (fixed_zero_offset_rad is None) != (fixed_sample_age_s is None):
            raise ValueError("fixed_zero_offset_rad and fixed_sample_age_s must be provided together.")
        shape = (ids.numel(), self.num_joints)
        if fixed_zero_offset_rad is not None:
            fixed_offset = self._state_tensor(fixed_zero_offset_rad, "fixed_zero_offset_rad")
            fixed_age = self._state_tensor(fixed_sample_age_s, "fixed_sample_age_s")
            offset_low, offset_high = self.zero_offset_range_rad
            age_low, age_high = self.sample_age_range_s
            if bool(torch.any((fixed_offset < offset_low) | (fixed_offset > offset_high))):
                raise ValueError(
                    f"fixed_zero_offset_rad must stay inside {self.zero_offset_range_rad!r}."
                )
            if bool(torch.any((fixed_age < age_low) | (fixed_age > age_high))):
                raise ValueError(f"fixed_sample_age_s must stay inside {self.sample_age_range_s!r}.")
            self.zero_offset_rad[ids] = fixed_offset[ids]
            self.sample_age_s[ids] = fixed_age[ids]
        elif self.sample_randomization:
            self.zero_offset_rad[ids] = self._uniform(shape, self.zero_offset_range_rad)
            self.sample_age_s[ids] = self._uniform(shape, self.sample_age_range_s)
        else:
            self.zero_offset_rad[ids] = 0.0
            midpoint_age = 0.5 * (self.sample_age_range_s[0] + self.sample_age_range_s[1])
            self.sample_age_s[ids] = midpoint_age

        current_position = position_tensor[ids]
        current_velocity = velocity_tensor[ids]
        self.previous_position[ids] = current_position
        self.current_position[ids] = current_position
        self.previous_velocity[ids] = current_velocity
        self.current_velocity[ids] = current_velocity
        self.cached_output[ids] = self._compose_output(
            current_position,
            current_velocity,
            default_tensor[ids],
            self.zero_offset_rad[ids],
        )
        self.last_step_counter[ids] = _UNSEEN_STEP

    def observe(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        default_position: torch.Tensor,
        step_counter: int,
    ) -> torch.Tensor:
        """Advance each environment at most once and hold its output for the policy step."""

        position_tensor = self._state_tensor(position, "position")
        velocity_tensor = self._state_tensor(velocity, "velocity")
        default_tensor = self._state_tensor(default_position, "default_position")
        step = int(step_counter)
        update_ids = torch.nonzero(self.last_step_counter != step, as_tuple=False).flatten()
        if update_ids.numel() == 0:
            return self.cached_output.clone()

        self.previous_position[update_ids] = self.current_position[update_ids]
        self.current_position[update_ids] = position_tensor[update_ids]
        self.previous_velocity[update_ids] = self.current_velocity[update_ids]
        self.current_velocity[update_ids] = velocity_tensor[update_ids]

        age_fraction = self.sample_age_s[update_ids] / self.step_dt
        delayed_position = torch.lerp(
            self.current_position[update_ids], self.previous_position[update_ids], age_fraction
        )
        delayed_velocity = torch.lerp(
            self.current_velocity[update_ids], self.previous_velocity[update_ids], age_fraction
        )
        self.cached_output[update_ids] = self._compose_output(
            delayed_position,
            delayed_velocity,
            default_tensor[update_ids],
            self.zero_offset_rad[update_ids],
        )
        self.last_step_counter[update_ids] = step
        return self.cached_output.clone()


class RndDynamixelEncoderObservation(ManagerTermBase):
    """Produce ordered ``[q_rel[12], dq_scaled[12]]`` Dynamixel observations."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        expected_params = {"asset_cfg", "model_path", "sample_randomization"}
        if set(cfg.params) != expected_params:
            raise ValueError(
                f"Dynamixel encoder observation params must be {sorted(expected_params)}; got {sorted(cfg.params)}."
            )
        if getattr(cfg, "noise", None) is not None:
            raise ValueError("Dynamixel encoder observation must not be combined with an external noise config.")
        configured_scale = getattr(cfg, "scale", None)
        if configured_scale is not None:
            if isinstance(configured_scale, bool) or not isinstance(configured_scale, (int, float)):
                raise ValueError("Dynamixel encoder observation scale must be None or scalar 1.0.")
            if not math.isclose(float(configured_scale), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    "Dynamixel encoder observation scale must be 1.0 because dq is scaled inside the 24-D term."
                )

        params = cfg.params
        if not isinstance(params["sample_randomization"], bool):
            raise TypeError("sample_randomization must be bool.")
        self._sample_randomization = params["sample_randomization"]
        self._model_path_param = str(params["model_path"])
        self._model = load_rnd_dynamixel_encoder_observation_model(params["model_path"])
        self._asset_cfg = params["asset_cfg"]
        self._asset_name = str(self._asset_cfg.name)
        asset = env.scene[self._asset_name]
        self._joint_ids = self._resolve_joint_ids(self._asset_cfg, asset)
        self._asset_cfg_signature = self._scene_entity_signature(self._asset_cfg)

        step_dt = float(env.step_dt)
        environment_policy_hz = 1.0 / step_dt
        if not math.isclose(environment_policy_hz, self._model.policy_hz, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"Encoder model policy_hz={self._model.policy_hz} does not match 1/env.step_dt={environment_policy_hz}."
            )
        position, velocity, default_position = self._raw_joint_state(env)
        self._state = DynamixelEncoderObservationState(
            num_envs=env.num_envs,
            num_joints=len(self._joint_ids),
            step_dt=step_dt,
            position_quantum_rad=self._model.position_quantum_rad,
            velocity_quantum_rad_s=self._model.velocity_quantum_rad_s,
            velocity_scale=self._model.velocity_scale,
            sample_age_range_s=self._model.sample_age_range_s,
            zero_offset_range_rad=self._model.zero_offset_range_rad,
            sample_randomization=self._sample_randomization,
            device=position.device,
            dtype=position.dtype,
        )
        self._fixed_zero_offset_rad: torch.Tensor | None = None
        self._fixed_sample_age_s: torch.Tensor | None = None
        self._state.reset(position, velocity, default_position)

    @property
    def model(self) -> RndDynamixelEncoderObservationModel:
        return self._model

    @property
    def state(self) -> DynamixelEncoderObservationState:
        return self._state

    @property
    def joint_ids(self) -> tuple[int, ...]:
        return self._joint_ids

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._model.joint_order

    @staticmethod
    def _scene_entity_signature(asset_cfg: SceneEntityCfg) -> tuple[Any, ...]:
        joint_names = tuple(asset_cfg.joint_names) if asset_cfg.joint_names is not None else None
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            ids_signature: tuple[Any, ...] = (joint_ids.start, joint_ids.stop, joint_ids.step)
        else:
            ids_signature = tuple(int(index) for index in joint_ids)
        return (asset_cfg.name, joint_names, ids_signature, asset_cfg.preserve_order)

    def _resolve_joint_ids(self, asset_cfg: SceneEntityCfg, asset) -> tuple[int, ...]:
        if asset_cfg.preserve_order is not True:
            raise ValueError("Dynamixel encoder asset_cfg must set preserve_order=True.")
        configured_names = asset_cfg.joint_names
        if not isinstance(configured_names, Sequence) or isinstance(configured_names, (str, bytes)):
            raise ValueError("Dynamixel encoder asset_cfg must provide an explicit ordered joint_names list.")
        if tuple(configured_names) != self._model.joint_order:
            raise ValueError(
                "Dynamixel encoder asset_cfg.joint_names must exactly match the 12-joint runtime model order."
            )

        configured_ids = asset_cfg.joint_ids
        if isinstance(configured_ids, slice):
            joint_ids = tuple(range(len(asset.joint_names)))[configured_ids]
        else:
            joint_ids = tuple(int(index) for index in configured_ids)
        if len(joint_ids) != 12 or len(set(joint_ids)) != 12:
            raise ValueError(f"Dynamixel encoder asset_cfg must resolve to 12 unique joints; got {joint_ids!r}.")
        if any(index < 0 or index >= len(asset.joint_names) for index in joint_ids):
            raise ValueError(f"Dynamixel encoder asset_cfg contains an out-of-range joint id: {joint_ids!r}.")
        resolved_names = tuple(asset.joint_names[index] for index in joint_ids)
        if resolved_names != self._model.joint_order:
            raise ValueError(f"Resolved Dynamixel joint order does not match the runtime model: {resolved_names!r}.")
        return joint_ids

    def _raw_joint_state(self, env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset = env.scene[self._asset_name]
        joint_ids = list(self._joint_ids)
        return (
            asset.data.joint_pos[:, joint_ids],
            asset.data.joint_vel[:, joint_ids],
            asset.data.default_joint_pos[:, joint_ids],
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        position, velocity, default_position = self._raw_joint_state(self._env)
        fixed_zero_offset = None
        fixed_sample_age = None
        if self._fixed_zero_offset_rad is not None:
            fixed_zero_offset = self._fixed_zero_offset_rad.unsqueeze(0).expand(self._env.num_envs, -1)
            fixed_sample_age = self._fixed_sample_age_s.unsqueeze(0).expand(self._env.num_envs, -1)
        self._state.reset(
            position,
            velocity,
            default_position,
            env_ids,
            fixed_zero_offset_rad=fixed_zero_offset,
            fixed_sample_age_s=fixed_sample_age,
        )

    def set_fixed_episode_parameters(
        self,
        *,
        zero_offset_rad: Sequence[float] | torch.Tensor,
        sample_age_s: Sequence[float] | torch.Tensor,
    ) -> None:
        """Persist one deterministic per-joint encoder domain across episode resets."""

        zero_offset = torch.as_tensor(
            zero_offset_rad,
            device=self._state.device,
            dtype=self._state.dtype,
        ).flatten()
        sample_age = torch.as_tensor(
            sample_age_s,
            device=self._state.device,
            dtype=self._state.dtype,
        ).flatten()
        expected_shape = (self._state.num_joints,)
        if zero_offset.shape != expected_shape or sample_age.shape != expected_shape:
            raise ValueError(
                "Fixed encoder parameters must contain one value for each ordered joint; "
                f"expected {expected_shape}, got {tuple(zero_offset.shape)} and {tuple(sample_age.shape)}."
            )
        if not bool(torch.isfinite(zero_offset).all()) or not bool(torch.isfinite(sample_age).all()):
            raise ValueError("Fixed encoder parameters must be finite.")
        offset_low, offset_high = self._model.zero_offset_range_rad
        age_low, age_high = self._model.sample_age_range_s
        if bool(torch.any((zero_offset < offset_low) | (zero_offset > offset_high))):
            raise ValueError(f"zero_offset_rad must stay inside {self._model.zero_offset_range_rad!r}.")
        if bool(torch.any((sample_age < age_low) | (sample_age > age_high))):
            raise ValueError(f"sample_age_s must stay inside {self._model.sample_age_range_s!r}.")
        self._fixed_zero_offset_rad = zero_offset.clone()
        self._fixed_sample_age_s = sample_age.clone()
        self.reset()

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        model_path: str,
        sample_randomization: bool,
    ) -> torch.Tensor:
        if self._scene_entity_signature(asset_cfg) != self._asset_cfg_signature:
            raise RuntimeError("Dynamixel encoder asset_cfg changed after term initialization.")
        if str(model_path) != self._model_path_param:
            raise RuntimeError("Dynamixel encoder model_path changed after term initialization.")
        if sample_randomization is not self._sample_randomization:
            raise RuntimeError("Dynamixel encoder sample_randomization changed after term initialization.")
        position, velocity, default_position = self._raw_joint_state(env)
        return self._state.observe(position, velocity, default_position, env.common_step_counter)
