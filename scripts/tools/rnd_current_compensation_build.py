#!/usr/bin/env python3
"""Build an analysis-only current-domain Coulomb compensation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
sys.path.insert(0, str(_TOOL_DIR))

from rnd_real2sim.bus import Mx2TelemetryBus
from rnd_real2sim.config import RND_LEG_JOINT_NAMES
from rnd_real2sim.current_compensation import (
    CURRENT_COMPENSATION_MODEL_TYPE,
    CURRENT_COMPENSATION_SCHEMA_VERSION,
    current_compensation_report,
    validate_current_compensation_model,
)


DEFAULT_BASELINE = _TOOL_DIR / "config" / "rnd_real2sim_baseline.json"
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_current_compensation_candidate.json"
DEFAULT_TRANSITION_SCALE = 4.0
DEFAULT_MAX_RELATIVE_RUN_SPAN = 0.5


class CurrentCompensationBuildError(ValueError):
    """Raised when baseline evidence cannot support a current compensation candidate."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hardware-write-disabled Coulomb-current candidate from the accepted real2sim baseline. "
            "This command does not modify the RL environment or Dynamixel operating mode."
        )
    )
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Accepted analysis baseline JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output candidate JSON.")
    parser.add_argument(
        "--transition-scale",
        type=float,
        default=DEFAULT_TRANSITION_SCALE,
        help="Multiplier from identification velocity threshold to tanh transition velocity.",
    )
    parser.add_argument(
        "--max-relative-run-span",
        type=float,
        default=DEFAULT_MAX_RELATIVE_RUN_SPAN,
        help="Maximum accepted (max-min)/median across selected runs.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path.expanduser().resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CurrentCompensationBuildError(f"Input does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise CurrentCompensationBuildError(f"Input is invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CurrentCompensationBuildError(f"Expected a JSON object in {path}.")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-12)


def _source_current_evidence(
    joint_name: str,
    aggregate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    provenance = aggregate.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise CurrentCompensationBuildError(f"{joint_name} has no current provenance.")

    sources: list[dict[str, Any]] = []
    values: list[float] = []
    velocity_thresholds: list[float] = []
    for entry in provenance:
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            raise CurrentCompensationBuildError(f"{joint_name} has malformed current provenance.")
        path = _resolve(entry["model"])
        if _sha256(path) != entry.get("model_sha256"):
            raise CurrentCompensationBuildError(f"Selected current model changed after aggregation: {path}")
        model = _load_json(path)
        if model.get("schema_version") != 2 or model.get("source_dataset_dry_run") is not False:
            raise CurrentCompensationBuildError(f"Unsupported or synthetic current model: {path}")
        joints = model.get("joints")
        if not isinstance(joints, dict) or set(joints) != {joint_name}:
            raise CurrentCompensationBuildError(f"Current model joint mismatch for {joint_name}: {path}")
        joint_model = joints[joint_name]
        if joint_model.get("quality", {}).get("coulomb_randomization_usable") is not True:
            raise CurrentCompensationBuildError(f"Current quality gate is false for selected model: {path}")
        friction = joint_model.get("friction_current_model")
        if not isinstance(friction, dict) or friction.get("quality_pass") is not True:
            raise CurrentCompensationBuildError(f"Current fit quality is false for selected model: {path}")
        value = float(friction["coulomb_current_a"])
        velocity_threshold = float(model["identification_config"]["velocity_threshold_rad_s"])
        if not math.isfinite(value) or value <= 0.0 or not math.isfinite(velocity_threshold) or velocity_threshold <= 0.0:
            raise CurrentCompensationBuildError(f"Non-positive current or velocity threshold in {path}")
        values.append(value)
        velocity_thresholds.append(velocity_threshold)
        sources.append(
            {
                "model": _relative(path),
                "model_sha256": entry["model_sha256"],
                "source_dataset": _relative(Path(model["source_dataset"])),
                "source_dataset_sha256": model["source_dataset_sha256"],
                "coulomb_current_a": value,
                "velocity_threshold_rad_s": velocity_threshold,
                "coulomb_quality_pass": True,
            }
        )

    summary = aggregate.get("coulomb_current_a")
    expected = {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }
    if not isinstance(summary, dict) or summary.get("count") != expected["count"]:
        raise CurrentCompensationBuildError(f"{joint_name} aggregate current count does not match provenance.")
    for field in ("minimum", "median", "maximum"):
        if not _close(float(summary[field]), float(expected[field])):
            raise CurrentCompensationBuildError(f"{joint_name} aggregate {field} does not match provenance.")
    return sources, values, velocity_thresholds


def _pair_asymmetry(joints: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for joint_type in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
        right = joints[f"R_Leg_{joint_type}"]
        left = joints[f"L_Leg_{joint_type}"]
        right_model = right["current_model"]
        left_model = left["current_model"]
        if right_model is None or left_model is None:
            diagnostics[joint_type] = {
                "available": False,
                "reason": "Both sides require a quality-gated current candidate.",
            }
            continue
        right_current = float(right_model["nominal_coulomb_current_a"])
        left_current = float(left_model["nominal_coulomb_current_a"])
        diagnostics[joint_type] = {
            "available": True,
            "right_current_a": right_current,
            "left_current_a": left_current,
            "larger_to_smaller_ratio": max(right_current, left_current) / min(right_current, left_current),
            "interpretation": "diagnostic_only_preserve_measured_side_specific_values",
        }
    return diagnostics


def build_current_compensation_model(
    baseline_path: str | Path,
    *,
    transition_scale: float = DEFAULT_TRANSITION_SCALE,
    max_relative_run_span: float = DEFAULT_MAX_RELATIVE_RUN_SPAN,
) -> dict[str, Any]:
    """Build a fail-closed current-domain candidate from accepted Coulomb evidence."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    if not math.isfinite(transition_scale) or transition_scale <= 0.0:
        raise CurrentCompensationBuildError("transition_scale must be finite and positive.")
    if not math.isfinite(max_relative_run_span) or not 0.0 < max_relative_run_span <= 1.0:
        raise CurrentCompensationBuildError("max_relative_run_span must be in (0, 1].")

    baseline = _load_json(baseline_path)
    if baseline.get("schema_version") != 1 or baseline.get("analysis_only") is not True:
        raise CurrentCompensationBuildError("Baseline must be schema_version=1 and analysis_only=true.")
    baseline_joints = baseline.get("joints")
    if not isinstance(baseline_joints, dict) or set(baseline_joints) != set(RND_LEG_JOINT_NAMES):
        raise CurrentCompensationBuildError("Baseline joint set does not match the 12 RND leg joints.")

    source_thresholds: list[float] = []
    joints: dict[str, Any] = {}
    usable_joint_count = 0
    for joint_name in RND_LEG_JOINT_NAMES:
        baseline_joint = baseline_joints[joint_name]
        aggregate = baseline_joint.get("coulomb_current")
        if not isinstance(aggregate, dict):
            raise CurrentCompensationBuildError(f"{joint_name} current aggregate is missing.")
        notes = list(baseline_joint.get("notes", []))
        if aggregate.get("usable") is not True:
            joints[joint_name] = {
                "current_model": None,
                "source_evidence": {
                    "run_count": int(aggregate.get("run_count", 0)),
                    "domain": aggregate.get("domain"),
                    "sources": [],
                },
                "quality": {
                    "source_quality_pass": False,
                    "minimum_run_count_pass": False,
                    "repeatability_pass": False,
                    "relative_run_span": None,
                    "candidate_usable": False,
                    "bench_validated": False,
                    "hardware_integration_allowed": False,
                    "status": "unavailable_current_evidence",
                },
                "notes": notes or ["No selected Coulomb-current model passed the source quality gate."],
            }
            continue

        sources, values, velocity_thresholds = _source_current_evidence(joint_name, aggregate)
        source_thresholds.extend(velocity_thresholds)
        nominal = float(statistics.median(values))
        relative_span = (max(values) - min(values)) / nominal
        minimum_run_count_pass = len(values) >= 2
        repeatability_pass = relative_span <= max_relative_run_span
        candidate_usable = minimum_run_count_pass and repeatability_pass
        transition_velocity = transition_scale * float(statistics.median(velocity_thresholds))
        nominal_raw = int(math.floor(nominal / Mx2TelemetryBus.CURRENT_UNIT_A + 0.5 + 1.0e-12))
        quantized_nominal = nominal_raw * Mx2TelemetryBus.CURRENT_UNIT_A
        if candidate_usable:
            usable_joint_count += 1
            current_model = {
                "law": "gain * Ic * tanh(4 * desired_velocity_rad_s / transition_velocity_rad_s)",
                "velocity_source": "low_level_desired_joint_trajectory_velocity_not_encoder_velocity",
                "direction_convention": "positive desired URDF-joint velocity produces positive motor-current compensation",
                "nominal_coulomb_current_a": nominal,
                "evidence_range_a": [min(values), max(values)],
                "nominal_goal_current_raw": nominal_raw,
                "quantized_nominal_current_a": quantized_nominal,
                "quantization_error_a": quantized_nominal - nominal,
                "transition_velocity_rad_s": transition_velocity,
                "transition_velocity_derivation": (
                    "4 * source identification velocity threshold, so tanh argument equals 1 at the source threshold"
                ),
                "viscous_current_a_per_rad_s": None,
                "static_breakaway_current_a": None,
            }
        else:
            current_model = None

        joints[joint_name] = {
            "current_model": current_model,
            "source_evidence": {
                "run_count": len(values),
                "domain": aggregate["domain"],
                "coulomb_current_a_values": values,
                "sources": sources,
            },
            "quality": {
                "source_quality_pass": True,
                "minimum_run_count_pass": minimum_run_count_pass,
                "repeatability_pass": repeatability_pass,
                "maximum_relative_run_span_allowed": max_relative_run_span,
                "relative_run_span": relative_span,
                "candidate_usable": candidate_usable,
                "bench_validated": False,
                "hardware_integration_allowed": False,
                "status": "offline_candidate" if candidate_usable else "aggregate_repeatability_failed",
            },
            "notes": notes,
        }

    if not source_thresholds:
        raise CurrentCompensationBuildError("No quality-gated current evidence exists in the baseline.")
    source_velocity_threshold = float(statistics.median(source_thresholds))
    if any(not _close(value, source_velocity_threshold) for value in source_thresholds):
        raise CurrentCompensationBuildError("Selected source models use inconsistent velocity thresholds.")

    model = {
        "schema_version": CURRENT_COMPENSATION_SCHEMA_VERSION,
        "model_type": CURRENT_COMPENSATION_MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "application_status": "offline_candidate_requires_current_control_and_bench_validation",
        "integration_enabled": False,
        "hardware_write_enabled": False,
        "source_baseline": _relative(baseline_path),
        "source_baseline_sha256": _sha256(baseline_path),
        "measurement_domain": baseline.get("measurement_domain"),
        "joint_order": list(RND_LEG_JOINT_NAMES),
        "control_contract": {
            "identified_operating_mode": 3,
            "identified_command": "Goal Position",
            "identified_current_signal": "Present Current telemetry",
            "position_mode_3_direct_application_supported": False,
            "direct_current_command_operating_mode": 0,
            "direct_current_command_requires_external_position_controller": True,
            "current_based_position_mode_5_goal_current_semantics": "current_limit_not_additive_feedforward",
            "goal_current_unit_a_per_raw": Mx2TelemetryBus.CURRENT_UNIT_A,
            "compensation_velocity_input": "low_level_desired_joint_trajectory_velocity_rad_s",
            "maximum_candidate_gain": 1.0,
            "zero_velocity_output_a": 0.0,
        },
        "smoothing": {
            "source_velocity_threshold_rad_s": source_velocity_threshold,
            "transition_scale": transition_scale,
            "transition_velocity_rad_s": transition_scale * source_velocity_threshold,
            "reason": (
                "The source data does not identify presliding stiction. A smooth odd Coulomb term avoids a sign "
                "discontinuity and deliberately fades to zero near zero desired velocity."
            ),
        },
        "torque_conversion": {
            "available": False,
            "current_to_joint_torque_nm_per_a": None,
            "simulator_torque_application_allowed": False,
            "reason": "No measured current-to-joint-torque calibration or loaded ground-contact identification exists.",
        },
        "bench_validation": {
            "status": "not_run",
            "recommended_gain_sequence": [0.25, 0.5, 0.75, 1.0],
            "required_observations": [
                "tracking error around direction reversals",
                "zero-velocity creep or limit cycle",
                "peak and RMS Present Current",
                "temperature rise and hardware error status",
            ],
            "required_fixture": "rigidly fixed upper body with full leg clearance before any loaded test",
        },
        "pair_asymmetry_diagnostics": _pair_asymmetry(joints),
        "quality_summary": {
            "joint_count": len(joints),
            "candidate_usable_joint_count": usable_joint_count,
            "unavailable_joints": [
                name for name, joint in joints.items() if joint["quality"]["candidate_usable"] is not True
            ],
            "bench_validated_joint_count": 0,
            "hardware_integration_ready": False,
        },
        "references": [
            {
                "title": "ROBOTIS MX-106T/R(2.0) e-Manual",
                "url": "https://emanual.robotis.com/docs/en/dxl/mx/mx-106-2/",
                "use": "Operating modes, Goal Current semantics, and 3.36 mA current unit.",
            },
            {
                "title": "System identification and force estimation of robotic manipulator",
                "url": "https://doi.org/10.1007/s11044-024-10017-1",
                "use": "Smooth Coulomb friction regularization with a hyperbolic tangent.",
            },
            {
                "title": "A New Model for Control of Systems with Friction",
                "url": "https://doi.org/10.1109/9.376053",
                "use": "Dynamic friction phenomena reserved for future data that can identify presliding state.",
            },
        ],
        "limitations": list(baseline.get("limitations", []))
        + [
            "The model compensates only the repeatable direction-dependent current component measured in suspension.",
            "Desired trajectory velocity is required; encoder velocity must not drive this feedforward term directly.",
            "Static breakaway current, Stribeck behavior, and viscous current are not identified by the current dataset.",
            "Goal Current in Current-based Position Mode is a limit, not an additive friction feedforward input.",
            "No hardware command path or reinforcement-learning integration is enabled by this artifact.",
        ],
        "joints": joints,
    }
    validate_current_compensation_model(model)
    return model


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    args = _parser().parse_args()
    try:
        model = build_current_compensation_model(
            args.baseline,
            transition_scale=args.transition_scale,
            max_relative_run_span=args.max_relative_run_span,
        )
        output = Path(args.output).expanduser().resolve()
        _atomic_write_json(output, model)
        print(f"Saved analysis-only current compensation candidate: {output}")
        print(current_compensation_report(model))
        return 0
    except (CurrentCompensationBuildError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
