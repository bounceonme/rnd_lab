"""Validation and offline evaluation for RND current-domain friction compensation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CURRENT_COMPENSATION_SCHEMA_VERSION = 1
CURRENT_COMPENSATION_MODEL_TYPE = "rnd_current_domain_coulomb_compensation_candidate"


class CurrentCompensationError(ValueError):
    """Raised when a current-compensation candidate violates its safety contract."""


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_current_compensation_model(model: dict[str, Any]) -> None:
    """Validate an analysis-only current-domain compensation artifact."""

    if model.get("schema_version") != CURRENT_COMPENSATION_SCHEMA_VERSION:
        raise CurrentCompensationError("Unsupported current-compensation schema version.")
    if model.get("model_type") != CURRENT_COMPENSATION_MODEL_TYPE:
        raise CurrentCompensationError("Unsupported current-compensation model type.")
    if model.get("analysis_only") is not True:
        raise CurrentCompensationError("Current-compensation candidates must remain analysis_only=true.")
    if model.get("integration_enabled") is not False or model.get("hardware_write_enabled") is not False:
        raise CurrentCompensationError("Current-compensation candidates must fail closed before bench validation.")
    if model.get("torque_conversion", {}).get("available") is not False:
        raise CurrentCompensationError("Current-domain evidence must not claim a joint-torque conversion.")

    contract = model.get("control_contract")
    if not isinstance(contract, dict):
        raise CurrentCompensationError("control_contract is missing.")
    current_unit = contract.get("goal_current_unit_a_per_raw")
    maximum_gain = contract.get("maximum_candidate_gain")
    if not _finite_number(current_unit) or float(current_unit) <= 0.0:
        raise CurrentCompensationError("Goal Current unit must be finite and positive.")
    if not _finite_number(maximum_gain) or not 0.0 < float(maximum_gain) <= 1.0:
        raise CurrentCompensationError("maximum_candidate_gain must be in (0, 1].")
    if contract.get("position_mode_3_direct_application_supported") is not False:
        raise CurrentCompensationError("Position Control Mode must not claim direct current compensation support.")
    if contract.get("current_based_position_mode_5_goal_current_semantics") != "current_limit_not_additive_feedforward":
        raise CurrentCompensationError("Current-based Position Mode Goal Current semantics are malformed.")

    joint_order = model.get("joint_order")
    joints = model.get("joints")
    if not isinstance(joint_order, list) or not joint_order or len(joint_order) != len(set(joint_order)):
        raise CurrentCompensationError("joint_order must be a non-empty unique list.")
    if not isinstance(joints, dict) or set(joints) != set(joint_order):
        raise CurrentCompensationError("joints must exactly match joint_order.")

    usable_count = 0
    for joint_name in joint_order:
        joint = joints[joint_name]
        quality = joint.get("quality")
        if not isinstance(quality, dict):
            raise CurrentCompensationError(f"{joint_name} quality block is missing.")
        if quality.get("bench_validated") is not False or quality.get("hardware_integration_allowed") is not False:
            raise CurrentCompensationError(f"{joint_name} must remain bench-unvalidated and integration-disabled.")

        candidate_usable = quality.get("candidate_usable") is True
        current_model = joint.get("current_model")
        if not candidate_usable:
            if current_model is not None:
                raise CurrentCompensationError(f"{joint_name} is unusable but contains a current model.")
            continue

        usable_count += 1
        if not isinstance(current_model, dict):
            raise CurrentCompensationError(f"{joint_name} usable candidate is missing current_model.")
        nominal = current_model.get("nominal_coulomb_current_a")
        evidence_range = current_model.get("evidence_range_a")
        transition_velocity = current_model.get("transition_velocity_rad_s")
        nominal_raw = current_model.get("nominal_goal_current_raw")
        quantized = current_model.get("quantized_nominal_current_a")
        if not _finite_number(nominal) or float(nominal) <= 0.0:
            raise CurrentCompensationError(f"{joint_name} nominal current must be finite and positive.")
        if (
            not isinstance(evidence_range, list)
            or len(evidence_range) != 2
            or any(not _finite_number(value) or float(value) <= 0.0 for value in evidence_range)
            or float(evidence_range[0]) > float(nominal)
            or float(nominal) > float(evidence_range[1])
        ):
            raise CurrentCompensationError(f"{joint_name} evidence current range is invalid.")
        if not _finite_number(transition_velocity) or float(transition_velocity) <= 0.0:
            raise CurrentCompensationError(f"{joint_name} transition velocity must be finite and positive.")
        if not isinstance(nominal_raw, int) or isinstance(nominal_raw, bool) or nominal_raw <= 0:
            raise CurrentCompensationError(f"{joint_name} nominal Goal Current raw value is invalid.")
        expected_quantized = nominal_raw * float(current_unit)
        if not _finite_number(quantized) or not math.isclose(
            float(quantized), expected_quantized, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CurrentCompensationError(f"{joint_name} quantized current does not match the MX-106 unit.")
        if current_model.get("law") != "gain * Ic * tanh(4 * desired_velocity_rad_s / transition_velocity_rad_s)":
            raise CurrentCompensationError(f"{joint_name} compensation law is unsupported.")
        if current_model.get("viscous_current_a_per_rad_s") is not None:
            raise CurrentCompensationError(f"{joint_name} must not invent unmeasured viscous friction.")
        if current_model.get("static_breakaway_current_a") is not None:
            raise CurrentCompensationError(f"{joint_name} must not invent unmeasured static breakaway current.")

    summary = model.get("quality_summary")
    if not isinstance(summary, dict):
        raise CurrentCompensationError("quality_summary is missing.")
    if summary.get("joint_count") != len(joint_order) or summary.get("candidate_usable_joint_count") != usable_count:
        raise CurrentCompensationError("quality_summary counts do not match the joint models.")
    if summary.get("bench_validated_joint_count") != 0 or summary.get("hardware_integration_ready") is not False:
        raise CurrentCompensationError("Current-compensation artifact must remain integration-blocked.")


def load_current_compensation_model(path: str | Path) -> dict[str, Any]:
    """Load and validate an analysis-only current compensation artifact."""

    resolved = Path(path).expanduser().resolve()
    try:
        model = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CurrentCompensationError(f"Current-compensation model does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise CurrentCompensationError(f"Current-compensation model is invalid JSON: {resolved}: {error}") from error
    if not isinstance(model, dict):
        raise CurrentCompensationError("Current-compensation model must be a JSON object.")
    validate_current_compensation_model(model)
    return model


def evaluate_compensation_current_a(
    model: dict[str, Any],
    joint_name: str,
    desired_velocity_rad_s: float,
    *,
    gain: float,
) -> float:
    """Evaluate the offline feedforward candidate in motor-current units."""

    validate_current_compensation_model(model)
    if joint_name not in model["joints"]:
        raise CurrentCompensationError(f"Unknown joint: {joint_name}")
    if not _finite_number(desired_velocity_rad_s):
        raise CurrentCompensationError("desired_velocity_rad_s must be finite.")
    maximum_gain = float(model["control_contract"]["maximum_candidate_gain"])
    if not _finite_number(gain) or not 0.0 <= float(gain) <= maximum_gain:
        raise CurrentCompensationError(f"gain must be in [0, {maximum_gain}].")

    joint = model["joints"][joint_name]
    if joint["quality"]["candidate_usable"] is not True:
        raise CurrentCompensationError(f"{joint_name} has no quality-gated current compensation candidate.")
    current_model = joint["current_model"]
    argument = 4.0 * float(desired_velocity_rad_s) / float(current_model["transition_velocity_rad_s"])
    return float(gain) * float(current_model["nominal_coulomb_current_a"]) * math.tanh(argument)


def evaluate_compensation_current_raw(
    model: dict[str, Any],
    joint_name: str,
    desired_velocity_rad_s: float,
    *,
    gain: float,
) -> int:
    """Quantize an offline compensation candidate to MX-106 Goal Current counts."""

    current_a = evaluate_compensation_current_a(
        model,
        joint_name,
        desired_velocity_rad_s,
        gain=gain,
    )
    unit = float(model["control_contract"]["goal_current_unit_a_per_raw"])
    magnitude = math.floor(abs(current_a) / unit + 0.5 + 1.0e-12)
    return int(math.copysign(magnitude, current_a)) if magnitude else 0


def current_compensation_report(model: dict[str, Any]) -> str:
    """Render a compact review report without implying hardware readiness."""

    validate_current_compensation_model(model)
    lines = [
        "RND STEP current-domain friction compensation candidate",
        f"source: {model['source_baseline']}",
        "",
        "joint                     Ic(A)  raw  transition(deg/s)  span(%)  status",
    ]
    for joint_name in model["joint_order"]:
        joint = model["joints"][joint_name]
        if not joint["quality"]["candidate_usable"]:
            lines.append(f"{joint_name:25s} {'-':>6s}  {'-':>3s}  {'-':>17s}  {'-':>7s}  unavailable")
            continue
        current_model = joint["current_model"]
        span = 100.0 * float(joint["quality"]["relative_run_span"])
        lines.append(
            f"{joint_name:25s} {current_model['nominal_coulomb_current_a']:6.4f}  "
            f"{current_model['nominal_goal_current_raw']:3d}  "
            f"{math.degrees(current_model['transition_velocity_rad_s']):17.3f}  "
            f"{span:7.2f}  offline_candidate"
        )
    lines.extend(
        (
            "",
            "This artifact never writes hardware and cannot be converted to simulator torque.",
            "Bench validation and an explicit Current Control Mode controller are still required.",
        )
    )
    return "\n".join(lines)
