#!/usr/bin/env python3
"""Build a compact RND torque-randomization config from the gated all-joint report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_torque_randomization import (
    RND_TORQUE_RANDOMIZATION_MODEL_TYPE,
    RND_TORQUE_RANDOMIZATION_SCHEMA_VERSION,
    validate_rnd_torque_randomization,
)
from rnd_real2sim.config import RND_LEG_JOINT_NAMES


DEFAULT_REPORT = (
    _REPO_ROOT / "logs" / "rnd_real2sim" / "all_joints_torque_calibration_01_all_joint_torque_calibration.json"
)
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_torque_randomization.json"


class TorqueRandomizationBuildError(ValueError):
    """Raised when the source calibration report cannot support a safe prior."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the opt-in RND joint-torque domain-randomization config.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="All-joint torque calibration report JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output randomization JSON.")
    parser.add_argument(
        "--unidentified-friction-max-nm",
        type=float,
        default=0.30,
        help="Upper Coulomb-torque prior for quality-rejected joints.",
    )
    parser.add_argument(
        "--measured-relative-span",
        type=float,
        default=0.25,
        help="Relative half-width around each passing measured Coulomb torque.",
    )
    parser.add_argument("--strength-scale-min", type=float, default=0.80)
    parser.add_argument("--strength-scale-max", type=float, default=1.25)
    parser.add_argument("--transition-velocity-min-deg-s", type=float, default=2.0)
    parser.add_argument("--transition-velocity-max-deg-s", type=float, default=8.0)
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


def _portable_provenance_path(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    return _relative(path) if path.is_absolute() else str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TorqueRandomizationBuildError(f"Calibration report does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise TorqueRandomizationBuildError(f"Calibration report is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise TorqueRandomizationBuildError("Calibration report must contain a JSON object.")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
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


def build_torque_randomization(
    report: dict[str, Any],
    *,
    report_path: str | Path,
    report_sha256: str,
    unidentified_friction_max_nm: float = 0.30,
    measured_relative_span: float = 0.25,
    strength_scale_range: tuple[float, float] = (0.80, 1.25),
    transition_velocity_deg_s_range: tuple[float, float] = (2.0, 8.0),
) -> dict[str, Any]:
    """Convert quality-passing evidence into measured ranges and failures into a broad prior."""

    if report.get("schema_version") != 1 or report.get("model_type") != "rnd_real2sim_all_joint_torque_calibration":
        raise TorqueRandomizationBuildError("Expected an all-joint torque calibration schema v1 report.")
    if report.get("analysis_only") is not True or report.get("integration_enabled") is not False:
        raise TorqueRandomizationBuildError("Source report must remain analysis-only and integration-disabled.")
    source_joints = report.get("joints")
    if not isinstance(source_joints, dict) or set(source_joints) != set(RND_LEG_JOINT_NAMES):
        raise TorqueRandomizationBuildError("Source report joint set does not match the 12 RND leg joints.")
    if not math.isfinite(unidentified_friction_max_nm) or not 0.0 < unidentified_friction_max_nm <= 1.0:
        raise TorqueRandomizationBuildError("unidentified_friction_max_nm must be in (0, 1].")
    if not math.isfinite(measured_relative_span) or not 0.0 < measured_relative_span <= 1.0:
        raise TorqueRandomizationBuildError("measured_relative_span must be in (0, 1].")
    if not 0.0 < strength_scale_range[0] <= strength_scale_range[1] <= 2.0:
        raise TorqueRandomizationBuildError("strength_scale_range must satisfy 0 < lower <= upper <= 2.")
    if not 0.0 < transition_velocity_deg_s_range[0] <= transition_velocity_deg_s_range[1] <= 180.0:
        raise TorqueRandomizationBuildError("transition velocity range is invalid.")

    passed_from_summary = set(report.get("summary", {}).get("calibration_passed_joints", []))
    joints: dict[str, Any] = {}
    measured_names: list[str] = []
    prior_names: list[str] = []
    for joint_name in RND_LEG_JOINT_NAMES:
        calibration = source_joints[joint_name].get("low_current_torque_calibration")
        if not isinstance(calibration, dict):
            raise TorqueRandomizationBuildError(f"Joint {joint_name} has no low-current calibration mapping.")
        quality = calibration.get("quality")
        quality_pass = isinstance(quality, dict) and quality.get("pass") is True
        if quality_pass != (joint_name in passed_from_summary):
            raise TorqueRandomizationBuildError(f"Joint {joint_name} quality flag disagrees with report summary.")

        if quality_pass:
            nominal = float(calibration["coulomb_torque_nm"])
            coulomb_current = float(calibration["coulomb_current_a"])
            torque_per_amp = float(calibration["torque_per_amp_nm"])
            interval = calibration["bootstrap_90pct_nm_per_a"]
            bootstrap_torque_range = sorted((
                float(interval[0]) * coulomb_current,
                float(interval[1]) * coulomb_current,
            ))
            relative_range = (nominal * (1.0 - measured_relative_span), nominal * (1.0 + measured_relative_span))
            randomization_range = [
                max(0.0, min(bootstrap_torque_range[0], relative_range[0])),
                max(bootstrap_torque_range[1], relative_range[1]),
            ]
            joints[joint_name] = {
                "evidence_status": "measured_quality_pass",
                "source_quality_pass": True,
                "source_quality_reasons": [],
                "measured_coulomb_torque_nm": nominal,
                "measured_coulomb_current_a": coulomb_current,
                "measured_torque_per_amp_nm": torque_per_amp,
                "bootstrap_90pct_torque_per_amp_nm": [float(interval[0]), float(interval[1])],
                "coulomb_torque_range_nm": randomization_range,
                "range_method": (
                    f"measured torque +/-{100.0 * measured_relative_span:g}%, "
                    "expanded to contain the Kt bootstrap interval"
                ),
            }
            measured_names.append(joint_name)
        else:
            reasons = quality.get("reasons", []) if isinstance(quality, dict) else []
            joints[joint_name] = {
                "evidence_status": "unidentified_prior",
                "source_quality_pass": False,
                "source_quality_reasons": [str(reason) for reason in reasons],
                "measured_coulomb_torque_nm": None,
                "measured_coulomb_current_a": None,
                "measured_torque_per_amp_nm": None,
                "bootstrap_90pct_torque_per_amp_nm": None,
                "coulomb_torque_range_nm": [0.0, float(unidentified_friction_max_nm)],
                "range_method": "operator-approved broad prior from zero through the passing-joint envelope",
            }
            prior_names.append(joint_name)

    result = {
        "schema_version": RND_TORQUE_RANDOMIZATION_SCHEMA_VERSION,
        "model_type": RND_TORQUE_RANDOMIZATION_MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "integration_enabled": True,
        "integration_scope": "RND STEP opt-in actuator task only",
        "sample_per_episode": True,
        "sample_bilateral_pairs_with_shared_quantile": True,
        "source_report": _relative(Path(report_path)),
        "source_report_sha256": report_sha256,
        "source_dataset": _portable_provenance_path(report.get("source_dataset")),
        "source_dataset_sha256": report.get("source_dataset_sha256"),
        "source_dynamics_trace": _portable_provenance_path(report.get("source_dynamics_trace")),
        "source_dynamics_trace_sha256": report.get("source_dynamics_trace_sha256"),
        "joint_order": list(RND_LEG_JOINT_NAMES),
        "motor_strength_scale_range": [float(strength_scale_range[0]), float(strength_scale_range[1])],
        "friction_transition_velocity_rad_s_range": [
            math.radians(transition_velocity_deg_s_range[0]),
            math.radians(transition_velocity_deg_s_range[1]),
        ],
        "viscous_friction_enabled": False,
        "static_breakaway_enabled": False,
        "quality_summary": {
            "measured_joint_count": len(measured_names),
            "measured_joint_names": measured_names,
            "unidentified_prior_joint_count": len(prior_names),
            "unidentified_prior_joint_names": prior_names,
        },
        "limitations": [
            "Passing values are effective low-current joint-domain fits, not pure motor constants.",
            "Quality-rejected joints use an uncertainty prior; zero remains possible and no failed fit value is used.",
            "The smooth Coulomb term excludes static breakaway and viscous friction to avoid unsupported parameters.",
            "Mirrored joints share random quantiles while retaining their own measured or prior parameter ranges.",
            "Motor strength is scaled before reapplying the configured actuator effort limit.",
            "This training randomization layer has not itself passed hardware trajectory replay.",
        ],
        "joints": joints,
    }
    validate_rnd_torque_randomization(result)
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        report_path = Path(args.report).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        result = build_torque_randomization(
            _load_json(report_path),
            report_path=report_path,
            report_sha256=_sha256(report_path),
            unidentified_friction_max_nm=args.unidentified_friction_max_nm,
            measured_relative_span=args.measured_relative_span,
            strength_scale_range=(args.strength_scale_min, args.strength_scale_max),
            transition_velocity_deg_s_range=(
                args.transition_velocity_min_deg_s,
                args.transition_velocity_max_deg_s,
            ),
        )
        _atomic_json(output_path, result)
        summary = result["quality_summary"]
        print(f"Saved torque-randomization config: {output_path}")
        print(
            f"measured_joints={summary['measured_joint_count']}, "
            f"unidentified_prior_joints={summary['unidentified_prior_joint_count']}, integration_enabled=True"
        )
        return 0
    except (OSError, TorqueRandomizationBuildError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
