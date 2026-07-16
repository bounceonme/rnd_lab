"""Fail-closed startup armature randomization for the 12 RND leg joints."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch


RND_ARMATURE_RANDOMIZATION_SCHEMA_VERSION = 1
RND_ARMATURE_RANDOMIZATION_MODEL_TYPE = "rnd_joint_armature_randomization"
RND_ARMATURE_CORRELATION_MODE = "global_shared_quantile"
RND_ARMATURE_MEASURED_RELATIVE_SPAN = 0.25
RND_ARMATURE_UNIDENTIFIED_PRIOR_RANGE_KG_M2 = (0.005, 0.04)
RND_ARMATURE_MAX_KG_M2 = 0.1

RND_ARMATURE_JOINT_NAMES = (
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
RND_ARMATURE_MEASURED_JOINT_NAMES = (
    "R_Leg_hip_pitch",
    "R_Leg_knee",
    "L_Leg_knee",
)

_MEASURED_RANGE_METHOD = "estimate +/-25%, expanded to contain the bootstrap 90% interval"
_PRIOR_RANGE_METHOD = "user-approved unidentified training prior"


class RndArmatureRandomizationError(ValueError):
    """Raised when an armature-randomization model or sampling request is invalid."""


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RndArmatureRandomizationError(f"{label} must be numeric, got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RndArmatureRandomizationError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise RndArmatureRandomizationError(f"{label} must be finite, got {result!r}.")
    if positive and result <= 0.0:
        raise RndArmatureRandomizationError(f"{label} must be positive, got {result}.")
    return result


def _range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RndArmatureRandomizationError(f"{label} must be a two-element JSON list.")
    lower = _finite(value[0], f"{label}[0]", positive=True)
    upper = _finite(value[1], f"{label}[1]", positive=True)
    if lower > upper:
        raise RndArmatureRandomizationError(f"{label} lower bound exceeds upper bound: {value!r}.")
    if upper > RND_ARMATURE_MAX_KG_M2:
        raise RndArmatureRandomizationError(
            f"{label} upper bound exceeds {RND_ARMATURE_MAX_KG_M2} kg*m^2."
        )
    return lower, upper


def _json_names(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise RndArmatureRandomizationError(f"{label} must be a JSON list of non-empty strings.")
    names = tuple(value)
    if not allow_empty and not names:
        raise RndArmatureRandomizationError(f"{label} must not be empty.")
    if len(names) != len(set(names)):
        raise RndArmatureRandomizationError(f"{label} must not contain duplicates.")
    return names


def _requested_names(value: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise RndArmatureRandomizationError(f"{label} must be a sequence of joint names, not a string.")
    try:
        names = tuple(value)
    except TypeError as error:
        raise RndArmatureRandomizationError(f"{label} must be a sequence of joint names.") from error
    if not names or not all(isinstance(name, str) and name for name in names):
        raise RndArmatureRandomizationError(f"{label} must contain non-empty joint names.")
    if len(names) != len(set(names)):
        raise RndArmatureRandomizationError(f"{label} must not contain duplicates.")
    return names


def _quality_reasons(value: Any, label: str, *, require_nonempty: bool) -> tuple[str, ...]:
    reasons = _json_names(value, label, allow_empty=True)
    if require_nonempty and not reasons:
        raise RndArmatureRandomizationError(f"{label} must record why the source fit failed quality gates.")
    return reasons


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RndArmatureRandomizationError(f"{label} must be a non-empty relative path.")
    if "\\" in value or value.startswith("~"):
        raise RndArmatureRandomizationError(f"{label} must use a portable repository-relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or path.as_posix() != value:
        raise RndArmatureRandomizationError(f"{label} must use a normalized repository-relative path.")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RndArmatureRandomizationError(f"{label} must be a lowercase 64-character SHA-256 digest.")
    return value


def _expanded_measured_range(
    estimate_kg_m2: float,
    bootstrap_range_kg_m2: tuple[float, float],
) -> tuple[float, float]:
    relative_low = estimate_kg_m2 * (1.0 - RND_ARMATURE_MEASURED_RELATIVE_SPAN)
    relative_high = estimate_kg_m2 * (1.0 + RND_ARMATURE_MEASURED_RELATIVE_SPAN)
    return min(relative_low, bootstrap_range_kg_m2[0]), max(relative_high, bootstrap_range_kg_m2[1])


def validate_rnd_armature_randomization(
    model: Mapping[str, Any],
    joint_names: Sequence[str] | None = None,
) -> None:
    """Validate the complete evidence and sampling contract before integration."""

    if not isinstance(model, Mapping):
        raise RndArmatureRandomizationError("Armature-randomization model must be a mapping.")
    if model.get("schema_version") != RND_ARMATURE_RANDOMIZATION_SCHEMA_VERSION:
        raise RndArmatureRandomizationError(
            f"Unsupported schema_version {model.get('schema_version')!r}; "
            f"expected {RND_ARMATURE_RANDOMIZATION_SCHEMA_VERSION}."
        )
    if model.get("model_type") != RND_ARMATURE_RANDOMIZATION_MODEL_TYPE:
        raise RndArmatureRandomizationError(f"Unsupported model_type {model.get('model_type')!r}.")
    if model.get("integration_enabled") is not True:
        raise RndArmatureRandomizationError("Armature randomization must explicitly set integration_enabled=true.")
    if model.get("integration_mode") != "opt_in_rl_training_randomization":
        raise RndArmatureRandomizationError(
            "integration_mode must remain opt_in_rl_training_randomization."
        )
    if model.get("physical_parameter_promotion") is not False:
        raise RndArmatureRandomizationError("Measured armatures must not be promoted as fixed physical parameters.")
    if model.get("sample_on_startup") is not True:
        raise RndArmatureRandomizationError("Armatures must explicitly set sample_on_startup=true.")
    if model.get("sample_per_episode") is not False:
        raise RndArmatureRandomizationError("Armatures must explicitly set sample_per_episode=false.")
    if model.get("correlation_mode") != RND_ARMATURE_CORRELATION_MODE:
        raise RndArmatureRandomizationError(
            f"correlation_mode must be {RND_ARMATURE_CORRELATION_MODE!r}."
        )
    if model.get("sample_bilateral_pairs_with_shared_quantile") is not True:
        raise RndArmatureRandomizationError(
            "Global correlation must retain shared quantiles for bilateral pairs."
        )
    if model.get("measured_relative_span") != RND_ARMATURE_MEASURED_RELATIVE_SPAN:
        raise RndArmatureRandomizationError(
            f"measured_relative_span must remain {RND_ARMATURE_MEASURED_RELATIVE_SPAN}."
        )

    source_report = _relative_path(model.get("source_report"), "source_report")
    repeat_report = _relative_path(model.get("failed_repeat_report"), "failed_repeat_report")
    if source_report == repeat_report:
        raise RndArmatureRandomizationError("source_report and failed_repeat_report must be distinct.")
    _sha256(model.get("source_report_sha256"), "source_report_sha256")
    _sha256(model.get("failed_repeat_report_sha256"), "failed_repeat_report_sha256")

    repeat_evidence = model.get("failed_repeat_evidence")
    if not isinstance(repeat_evidence, Mapping):
        raise RndArmatureRandomizationError("failed_repeat_evidence must be a mapping.")
    if repeat_evidence.get("joint_name") != "L_Leg_hip_pitch":
        raise RndArmatureRandomizationError("Failed repeat evidence must identify L_Leg_hip_pitch.")
    if repeat_evidence.get("source_quality_pass") is not False:
        raise RndArmatureRandomizationError("Failed repeat evidence must retain source_quality_pass=false.")
    if repeat_evidence.get("used_for_range") is not False:
        raise RndArmatureRandomizationError("Failed repeat evidence must not be used for a randomization range.")
    _quality_reasons(
        repeat_evidence.get("source_quality_reasons"),
        "failed_repeat_evidence.source_quality_reasons",
        require_nonempty=True,
    )

    joints = model.get("joints")
    if not isinstance(joints, Mapping) or set(joints) != set(RND_ARMATURE_JOINT_NAMES):
        raise RndArmatureRandomizationError("joints must contain exactly the 12 RND leg joints.")
    joint_order = _json_names(model.get("joint_order"), "joint_order")
    if joint_order != RND_ARMATURE_JOINT_NAMES:
        raise RndArmatureRandomizationError("joint_order must match the canonical 12-joint RND leg order.")

    selected = joint_order if joint_names is None else _requested_names(joint_names, "joint_names")
    missing = sorted(set(selected) - set(joints))
    if missing:
        raise RndArmatureRandomizationError(f"Armature randomization is missing joints: {missing}.")

    measured_names: list[str] = []
    prior_names: list[str] = []
    for joint_name in joint_order:
        joint = joints[joint_name]
        if not isinstance(joint, Mapping):
            raise RndArmatureRandomizationError(f"joints.{joint_name} must be a mapping.")
        status = joint.get("evidence_status")
        if status not in ("measured_quality_pass", "unidentified_prior"):
            raise RndArmatureRandomizationError(f"joints.{joint_name} has unsupported evidence_status.")
        armature_range = _range(joint.get("armature_range_kg_m2"), f"joints.{joint_name}.armature_range_kg_m2")

        if status == "measured_quality_pass":
            if joint_name not in RND_ARMATURE_MEASURED_JOINT_NAMES:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} is not approved for a measured_quality_pass claim."
                )
            if joint.get("source_quality_pass") is not True:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} measured evidence must have passed source quality gates."
                )
            reasons = _quality_reasons(
                joint.get("source_quality_reasons"),
                f"joints.{joint_name}.source_quality_reasons",
                require_nonempty=False,
            )
            if reasons:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} quality-pass evidence must not retain failure reasons."
                )
            estimate = _finite(
                joint.get("measured_armature_kg_m2"),
                f"joints.{joint_name}.measured_armature_kg_m2",
                positive=True,
            )
            bootstrap_range = _range(
                joint.get("bootstrap_90pct_kg_m2"),
                f"joints.{joint_name}.bootstrap_90pct_kg_m2",
            )
            expected_range = _expanded_measured_range(estimate, bootstrap_range)
            if armature_range != expected_range:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} range must be estimate +/-25%, expanded to contain its bootstrap interval."
                )
            if not bootstrap_range[0] <= estimate <= bootstrap_range[1]:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} estimate is outside its bootstrap 90% interval."
                )
            if joint.get("range_method") != _MEASURED_RANGE_METHOD:
                raise RndArmatureRandomizationError(f"joints.{joint_name} has an unsupported measured range method.")
            measured_names.append(joint_name)
        else:
            if joint_name in RND_ARMATURE_MEASURED_JOINT_NAMES:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} must retain its approved measured_quality_pass evidence."
                )
            if joint.get("source_quality_pass") is not False:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} unidentified prior must retain source_quality_pass=false."
                )
            _quality_reasons(
                joint.get("source_quality_reasons"),
                f"joints.{joint_name}.source_quality_reasons",
                require_nonempty=True,
            )
            if joint.get("measured_armature_kg_m2") is not None or joint.get("bootstrap_90pct_kg_m2") is not None:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} unidentified prior must not retain failed fit values."
                )
            if armature_range != RND_ARMATURE_UNIDENTIFIED_PRIOR_RANGE_KG_M2:
                raise RndArmatureRandomizationError(
                    f"joints.{joint_name} unidentified prior must be "
                    f"{list(RND_ARMATURE_UNIDENTIFIED_PRIOR_RANGE_KG_M2)} kg*m^2."
                )
            if joint.get("range_method") != _PRIOR_RANGE_METHOD:
                raise RndArmatureRandomizationError(f"joints.{joint_name} has an unsupported prior range method.")
            prior_names.append(joint_name)

    if tuple(measured_names) != RND_ARMATURE_MEASURED_JOINT_NAMES:
        raise RndArmatureRandomizationError("Measured armature set does not match the three approved joints.")
    expected_prior_names = tuple(name for name in RND_ARMATURE_JOINT_NAMES if name not in measured_names)
    if tuple(prior_names) != expected_prior_names:
        raise RndArmatureRandomizationError("Unidentified-prior armature set is incomplete.")

    summary = model.get("quality_summary")
    if not isinstance(summary, Mapping):
        raise RndArmatureRandomizationError("quality_summary must be a mapping.")
    if summary.get("measured_joint_count") != len(measured_names):
        raise RndArmatureRandomizationError("quality_summary.measured_joint_count is inconsistent.")
    if summary.get("unidentified_prior_joint_count") != len(prior_names):
        raise RndArmatureRandomizationError("quality_summary.unidentified_prior_joint_count is inconsistent.")
    if _json_names(summary.get("measured_joint_names"), "quality_summary.measured_joint_names") != tuple(
        measured_names
    ):
        raise RndArmatureRandomizationError("quality_summary.measured_joint_names is inconsistent.")
    if _json_names(
        summary.get("unidentified_prior_joint_names"),
        "quality_summary.unidentified_prior_joint_names",
    ) != tuple(prior_names):
        raise RndArmatureRandomizationError("quality_summary.unidentified_prior_joint_names is inconsistent.")


def load_rnd_armature_randomization(
    path: str | Path,
    joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load and validate an RND startup armature-randomization JSON model."""

    resolved = Path(path).expanduser().resolve()
    try:
        model = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise RndArmatureRandomizationError(f"Unable to read armature-randomization model: {resolved}") from error
    except json.JSONDecodeError as error:
        raise RndArmatureRandomizationError(
            f"Armature-randomization model is not valid JSON: {resolved}: {error}"
        ) from error
    if not isinstance(model, dict):
        raise RndArmatureRandomizationError(
            f"Armature-randomization model must contain a JSON object: {resolved}"
        )
    validate_rnd_armature_randomization(model, joint_names)
    return model


def _stream_seed(seed: int) -> int:
    payload = f"{seed}:rnd_armature:{RND_ARMATURE_CORRELATION_MODE}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def sample_rnd_armatures(
    model: Mapping[str, Any],
    joint_names: Sequence[str],
    num_envs: int,
    device: str | torch.device,
    *,
    seed: int = 0,
    sample_randomization: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample one globally shared armature quantile per environment at startup."""

    validate_rnd_armature_randomization(model, joint_names)
    names = _requested_names(joint_names, "joint_names")
    if isinstance(num_envs, bool):
        raise RndArmatureRandomizationError("num_envs must be a positive integer.")
    try:
        env_count = operator.index(num_envs)
    except TypeError as error:
        raise RndArmatureRandomizationError("num_envs must be a positive integer.") from error
    if env_count <= 0:
        raise RndArmatureRandomizationError("num_envs must be a positive integer.")
    if isinstance(seed, bool):
        raise RndArmatureRandomizationError("seed must be an integer.")
    try:
        seed_value = operator.index(seed)
    except TypeError as error:
        raise RndArmatureRandomizationError("seed must be an integer.") from error
    if not isinstance(sample_randomization, bool):
        raise RndArmatureRandomizationError("sample_randomization must be a bool.")
    if not isinstance(dtype, torch.dtype):
        raise RndArmatureRandomizationError("dtype must be a torch.dtype.")
    try:
        dtype_probe = torch.empty((), dtype=dtype)
    except (RuntimeError, TypeError) as error:
        raise RndArmatureRandomizationError(f"Unsupported dtype: {dtype!r}.") from error
    if not dtype_probe.is_floating_point():
        raise RndArmatureRandomizationError("dtype must be a floating-point torch dtype.")
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise RndArmatureRandomizationError(f"Invalid torch device: {device!r}.") from error

    ranges = torch.tensor(
        [model["joints"][name]["armature_range_kg_m2"] for name in names],
        dtype=dtype,
        device=resolved_device,
    )
    if sample_randomization:
        try:
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(_stream_seed(seed_value))
            quantile = torch.rand(
                (env_count, 1),
                dtype=dtype,
                device=resolved_device,
                generator=generator,
            )
        except (RuntimeError, TypeError) as error:
            raise RndArmatureRandomizationError(
                f"Unable to sample armatures on torch device {resolved_device}."
            ) from error
    else:
        quantile = torch.full((env_count, 1), 0.5, dtype=dtype, device=resolved_device)

    lower = ranges[:, 0].unsqueeze(0)
    upper = ranges[:, 1].unsqueeze(0)
    return lower + quantile * (upper - lower)
