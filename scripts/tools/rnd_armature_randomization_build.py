#!/usr/bin/env python3
"""Build the deterministic RND startup armature-randomization contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_armature_randomization import (
    RND_ARMATURE_CORRELATION_MODE,
    RND_ARMATURE_JOINT_NAMES,
    RND_ARMATURE_MAX_KG_M2,
    RND_ARMATURE_MEASURED_JOINT_NAMES,
    RND_ARMATURE_MEASURED_RELATIVE_SPAN,
    RND_ARMATURE_RANDOMIZATION_MODEL_TYPE,
    RND_ARMATURE_RANDOMIZATION_SCHEMA_VERSION,
    RND_ARMATURE_UNIDENTIFIED_PRIOR_RANGE_KG_M2,
    validate_rnd_armature_randomization,
)


DEFAULT_REPORT = (
    _REPO_ROOT / "logs" / "rnd_real2sim" / "all_joints_armature_01_all_joint_armature.json"
)
DEFAULT_FAILED_REPEAT_REPORT = (
    _REPO_ROOT / "logs" / "rnd_real2sim" / "l_hip_pitch_armature_02_all_joint_armature.json"
)
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_armature_randomization.json"

_SOURCE_SCHEMA_VERSION = 2
_SOURCE_MODEL_TYPE = "rnd_real2sim_all_joint_armature_analysis"
_FAILED_REPEAT_JOINT = "L_Leg_hip_pitch"
_MEASURED_RANGE_METHOD = "estimate +/-25%, expanded to contain the bootstrap 90% interval"
_PRIOR_RANGE_METHOD = "user-approved unidentified training prior"


class ArmatureRandomizationBuildError(ValueError):
    """Raised when source reports cannot support the fixed training contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the opt-in RND startup armature randomization JSON.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Primary all-joint armature report JSON.")
    parser.add_argument(
        "--failed-repeat-report",
        default=str(DEFAULT_FAILED_REPEAT_REPORT),
        help="Failed L hip-pitch repeat report retained as non-promoted evidence.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output randomization JSON.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArmatureRandomizationBuildError(f"Armature report does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ArmatureRandomizationBuildError(f"Armature report is not valid JSON: {path}: {error}") from error
    except OSError as error:
        raise ArmatureRandomizationBuildError(f"Unable to read armature report: {path}") from error
    if not isinstance(value, dict):
        raise ArmatureRandomizationBuildError(f"Armature report must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(path: str | Path, label: str) -> str:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (_REPO_ROOT / candidate).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError as error:
        raise ArmatureRandomizationBuildError(f"{label} must be inside the repository: {resolved}") from error


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArmatureRandomizationBuildError(f"{label} must be a lowercase 64-character SHA-256 digest.")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ArmatureRandomizationBuildError(f"{label} must be numeric, got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ArmatureRandomizationBuildError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise ArmatureRandomizationBuildError(f"{label} must be finite, got {result!r}.")
    if positive and result <= 0.0:
        raise ArmatureRandomizationBuildError(f"{label} must be positive, got {result}.")
    return result


def _range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ArmatureRandomizationBuildError(f"{label} must be a two-element list.")
    lower = _finite(value[0], f"{label}[0]", positive=True)
    upper = _finite(value[1], f"{label}[1]", positive=True)
    if lower > upper or upper > RND_ARMATURE_MAX_KG_M2:
        raise ArmatureRandomizationBuildError(
            f"{label} must satisfy 0 < lower <= upper <= {RND_ARMATURE_MAX_KG_M2}."
        )
    return lower, upper


def _names(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise ArmatureRandomizationBuildError(f"{label} must be a list of non-empty strings.")
    names = tuple(value)
    if not allow_empty and not names:
        raise ArmatureRandomizationBuildError(f"{label} must not be empty.")
    if len(names) != len(set(names)):
        raise ArmatureRandomizationBuildError(f"{label} must not contain duplicates.")
    return names


def _quality(joint: Mapping[str, Any], label: str) -> tuple[bool, list[str]]:
    quality = joint.get("quality")
    if not isinstance(quality, Mapping) or quality.get("pass") not in (True, False):
        raise ArmatureRandomizationBuildError(f"{label}.quality must contain an explicit boolean pass flag.")
    reasons = list(_names(quality.get("reasons"), f"{label}.quality.reasons", allow_empty=True))
    if quality["pass"] is True and reasons:
        raise ArmatureRandomizationBuildError(f"{label} passes quality but retains failure reasons.")
    if quality["pass"] is False and not reasons:
        raise ArmatureRandomizationBuildError(f"{label} fails quality without recording a reason.")
    return quality["pass"] is True, reasons


def _validate_report_header(report: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if report.get("schema_version") != _SOURCE_SCHEMA_VERSION or report.get("model_type") != _SOURCE_MODEL_TYPE:
        raise ArmatureRandomizationBuildError(f"{label} must be an all-joint armature analysis schema v2 report.")
    if report.get("analysis_only") is not True or report.get("integration_enabled") is not False:
        raise ArmatureRandomizationBuildError(f"{label} must remain analysis-only and integration-disabled.")
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise ArmatureRandomizationBuildError(f"{label}.summary must be a mapping.")
    if summary.get("automatic_integration_allowed") is not False:
        raise ArmatureRandomizationBuildError(f"{label} must retain automatic_integration_allowed=false.")
    return summary


def _validate_primary_report(report: Mapping[str, Any]) -> dict[str, tuple[bool, list[str]]]:
    summary = _validate_report_header(report, "report")
    joints = report.get("joints")
    if not isinstance(joints, Mapping) or set(joints) != set(RND_ARMATURE_JOINT_NAMES):
        raise ArmatureRandomizationBuildError("report.joints must contain exactly the 12 RND leg joints.")

    qualities: dict[str, tuple[bool, list[str]]] = {}
    for joint_name in RND_ARMATURE_JOINT_NAMES:
        joint = joints[joint_name]
        if not isinstance(joint, Mapping):
            raise ArmatureRandomizationBuildError(f"report.joints.{joint_name} must be a mapping.")
        qualities[joint_name] = _quality(joint, f"report.joints.{joint_name}")

    passed = tuple(name for name in RND_ARMATURE_JOINT_NAMES if qualities[name][0])
    failed = tuple(name for name in RND_ARMATURE_JOINT_NAMES if not qualities[name][0])
    summary_passed = _names(summary.get("armature_passed_joints"), "report.summary.armature_passed_joints")
    summary_failed = _names(summary.get("armature_failed_joints"), "report.summary.armature_failed_joints")
    if set(summary_passed) != set(passed) or set(summary_failed) != set(failed):
        raise ArmatureRandomizationBuildError("report quality flags disagree with its passed/failed summary.")
    if passed != RND_ARMATURE_MEASURED_JOINT_NAMES:
        raise ArmatureRandomizationBuildError(
            "Primary report must quality-pass exactly R_Leg_hip_pitch, R_Leg_knee, and L_Leg_knee."
        )
    return qualities


def _validate_failed_repeat_report(report: Mapping[str, Any]) -> list[str]:
    summary = _validate_report_header(report, "failed_repeat_report")
    joints = report.get("joints")
    if not isinstance(joints, Mapping) or set(joints) != {_FAILED_REPEAT_JOINT}:
        raise ArmatureRandomizationBuildError(
            "failed_repeat_report.joints must contain only L_Leg_hip_pitch."
        )
    joint = joints[_FAILED_REPEAT_JOINT]
    if not isinstance(joint, Mapping):
        raise ArmatureRandomizationBuildError("failed_repeat_report L_Leg_hip_pitch entry must be a mapping.")
    quality_pass, reasons = _quality(joint, "failed_repeat_report.joints.L_Leg_hip_pitch")
    summary_passed = _names(
        summary.get("armature_passed_joints"),
        "failed_repeat_report.summary.armature_passed_joints",
        allow_empty=True,
    )
    summary_failed = _names(
        summary.get("armature_failed_joints"),
        "failed_repeat_report.summary.armature_failed_joints",
        allow_empty=True,
    )
    if quality_pass or summary_passed or summary_failed != (_FAILED_REPEAT_JOINT,):
        raise ArmatureRandomizationBuildError(
            "L_Leg_hip_pitch repeat must remain failed evidence and cannot be promoted."
        )
    return reasons


def _measured_joint(joint_name: str, source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("available") is not True:
        raise ArmatureRandomizationBuildError(f"Passing joint {joint_name} must be available.")
    if source.get("selected_estimator") != "cycle_harmonic_acceleration_projection":
        raise ArmatureRandomizationBuildError(f"Passing joint {joint_name} uses an unsupported estimator.")
    conversion = source.get("torque_conversion")
    if not isinstance(conversion, Mapping) or conversion.get("source_quality_pass") is not True:
        raise ArmatureRandomizationBuildError(
            f"Passing joint {joint_name} must retain a quality-passing torque conversion."
        )
    fit = source.get("fit")
    if not isinstance(fit, Mapping):
        raise ArmatureRandomizationBuildError(f"Passing joint {joint_name} has no fit mapping.")
    estimate = _finite(fit.get("armature_kg_m2"), f"{joint_name}.fit.armature_kg_m2", positive=True)
    bootstrap = _range(fit.get("bootstrap_90pct_kg_m2"), f"{joint_name}.fit.bootstrap_90pct_kg_m2")
    if not bootstrap[0] <= estimate <= bootstrap[1]:
        raise ArmatureRandomizationBuildError(f"Passing joint {joint_name} estimate is outside its bootstrap interval.")
    relative_range = (
        estimate * (1.0 - RND_ARMATURE_MEASURED_RELATIVE_SPAN),
        estimate * (1.0 + RND_ARMATURE_MEASURED_RELATIVE_SPAN),
    )
    randomization_range = [
        min(relative_range[0], bootstrap[0]),
        max(relative_range[1], bootstrap[1]),
    ]
    if randomization_range[1] > RND_ARMATURE_MAX_KG_M2:
        raise ArmatureRandomizationBuildError(f"Passing joint {joint_name} exceeds the armature safety bound.")
    return {
        "evidence_status": "measured_quality_pass",
        "source_quality_pass": True,
        "source_quality_reasons": [],
        "measured_armature_kg_m2": estimate,
        "bootstrap_90pct_kg_m2": list(bootstrap),
        "armature_range_kg_m2": randomization_range,
        "range_method": _MEASURED_RANGE_METHOD,
    }


def build_armature_randomization(
    report: dict[str, Any],
    failed_repeat_report: dict[str, Any],
    *,
    report_path: str | Path,
    report_sha256: str,
    failed_repeat_report_path: str | Path,
    failed_repeat_report_sha256: str,
) -> dict[str, Any]:
    """Build ranges from passing primary fits while retaining repeat failure only as provenance."""

    qualities = _validate_primary_report(report)
    repeat_reasons = _validate_failed_repeat_report(failed_repeat_report)
    source_path = _source_path(report_path, "report_path")
    repeat_path = _source_path(failed_repeat_report_path, "failed_repeat_report_path")
    if source_path == repeat_path:
        raise ArmatureRandomizationBuildError("Primary and failed-repeat report paths must be distinct.")
    source_digest = _digest(report_sha256, "report_sha256")
    repeat_digest = _digest(failed_repeat_report_sha256, "failed_repeat_report_sha256")

    source_joints = report["joints"]
    joints: dict[str, Any] = {}
    measured_names: list[str] = []
    prior_names: list[str] = []
    for joint_name in RND_ARMATURE_JOINT_NAMES:
        quality_pass, reasons = qualities[joint_name]
        if quality_pass:
            joints[joint_name] = _measured_joint(joint_name, source_joints[joint_name])
            measured_names.append(joint_name)
        else:
            joints[joint_name] = {
                "evidence_status": "unidentified_prior",
                "source_quality_pass": False,
                "source_quality_reasons": reasons,
                "measured_armature_kg_m2": None,
                "bootstrap_90pct_kg_m2": None,
                "armature_range_kg_m2": list(RND_ARMATURE_UNIDENTIFIED_PRIOR_RANGE_KG_M2),
                "range_method": _PRIOR_RANGE_METHOD,
            }
            prior_names.append(joint_name)

    result = {
        "schema_version": RND_ARMATURE_RANDOMIZATION_SCHEMA_VERSION,
        "model_type": RND_ARMATURE_RANDOMIZATION_MODEL_TYPE,
        "integration_enabled": True,
        "integration_mode": "opt_in_rl_training_randomization",
        "physical_parameter_promotion": False,
        "sample_on_startup": True,
        "sample_per_episode": False,
        "correlation_mode": RND_ARMATURE_CORRELATION_MODE,
        "sample_bilateral_pairs_with_shared_quantile": True,
        "measured_relative_span": RND_ARMATURE_MEASURED_RELATIVE_SPAN,
        "source_report": source_path,
        "source_report_sha256": source_digest,
        "failed_repeat_report": repeat_path,
        "failed_repeat_report_sha256": repeat_digest,
        "failed_repeat_evidence": {
            "joint_name": _FAILED_REPEAT_JOINT,
            "source_quality_pass": False,
            "source_quality_reasons": repeat_reasons,
            "used_for_range": False,
        },
        "joint_order": list(RND_ARMATURE_JOINT_NAMES),
        "quality_summary": {
            "measured_joint_count": len(measured_names),
            "measured_joint_names": measured_names,
            "unidentified_prior_joint_count": len(prior_names),
            "unidentified_prior_joint_names": prior_names,
        },
        "limitations": [
            "integration_enabled=true enables opt-in RL training randomization only; it does not promote any estimate to a fixed physical armature.",
            "Only primary-report fits that pass every source quality gate receive measured ranges.",
            "The failed L hip-pitch repeat remains unidentified evidence and contributes no fit value to its training prior.",
            "Unidentified joints use the user-approved [0.005, 0.04] kg*m^2 training prior.",
            "One normalized quantile is shared across all selected joints in each environment to represent common MX-106 uncertainty without artificial left/right asymmetry.",
            "Armatures are sampled once at environment startup and are not resampled on episode reset.",
        ],
        "joints": joints,
    }
    validate_rnd_armature_randomization(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
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
        report_path = Path(args.report).expanduser().resolve()
        repeat_path = Path(args.failed_repeat_report).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        result = build_armature_randomization(
            _load_json(report_path),
            _load_json(repeat_path),
            report_path=report_path,
            report_sha256=_sha256(report_path),
            failed_repeat_report_path=repeat_path,
            failed_repeat_report_sha256=_sha256(repeat_path),
        )
        _atomic_json(output_path, result)
        summary = result["quality_summary"]
        print(f"Saved armature-randomization config: {output_path}")
        print(
            f"measured_joints={summary['measured_joint_count']}, "
            f"unidentified_prior_joints={summary['unidentified_prior_joint_count']}, "
            "correlation_mode=global_shared_quantile, sample_per_episode=false"
        )
        return 0
    except (ArmatureRandomizationBuildError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
