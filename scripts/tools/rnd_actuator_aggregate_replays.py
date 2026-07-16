#!/usr/bin/env python3
"""Aggregate accepted RND simulator replays without enabling training integration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import validate_rnd_actuator_model
from rnd_real2sim.config import RND_LEG_JOINT_NAMES


DEFAULT_MANIFEST = _TOOL_DIR / "config" / "rnd_real2sim_baseline_manifest.toml"
DEFAULT_MODEL = _TOOL_DIR / "config" / "rnd_actuator_model.json"
DEFAULT_REPORT_DIRECTORY = _REPO_ROOT / "logs" / "rnd_real2sim"
DEFAULT_SUMMARY_OUTPUT = _TOOL_DIR / "config" / "rnd_actuator_sim_replay_summary.json"
DEFAULT_CANDIDATE_OUTPUT = _TOOL_DIR / "config" / "rnd_actuator_model_candidate.json"


class ReplayAggregationError(ValueError):
    """Raised when replay evidence is missing, inconsistent, or fails its gate."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate fixed-base Isaac replay reports into an analysis summary and a disabled actuator-model "
            "candidate. The runtime model and reinforcement-learning configuration are never edited."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Accepted target-model manifest TOML.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Unvalidated actuator seed used by the replays.")
    parser.add_argument(
        "--report-directory",
        default=str(DEFAULT_REPORT_DIRECTORY),
        help="Directory containing *_sim_replay*.json reports.",
    )
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT), help="Analysis-only summary JSON.")
    parser.add_argument(
        "--candidate-output",
        default=str(DEFAULT_CANDIDATE_OUTPUT),
        help="Validated but integration-disabled actuator candidate JSON.",
    )
    return parser


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReplayAggregationError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ReplayAggregationError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayAggregationError(f"{label} must contain a JSON object: {path}")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReplayAggregationError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise ReplayAggregationError(f"{label} must be finite, got {result!r}.")
    if minimum is not None and result < minimum:
        raise ReplayAggregationError(f"{label} must be >= {minimum}, got {result}.")
    return result


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_expected_datasets(manifest_path: Path) -> tuple[dict[Path, dict[str, Any]], dict[str, list[Path]]]:
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReplayAggregationError(f"Unable to load manifest {manifest_path}: {error}") from error
    if manifest.get("schema_version") != 1 or manifest.get("analysis_only") is not True:
        raise ReplayAggregationError("Replay aggregation requires a schema_version=1, analysis_only=true manifest.")
    entries = manifest.get("joints")
    if not isinstance(entries, list):
        raise ReplayAggregationError("Manifest requires [[joints]] entries.")
    names = [entry.get("name") for entry in entries]
    if len(names) != len(set(names)) or set(names) != set(RND_LEG_JOINT_NAMES):
        raise ReplayAggregationError("Manifest joint set must match the 12 RND leg joints exactly.")

    expected: dict[Path, dict[str, Any]] = {}
    by_joint: dict[str, list[Path]] = {name: [] for name in RND_LEG_JOINT_NAMES}
    for entry in entries:
        joint_name = str(entry["name"])
        target_models = entry.get("target_models", [])
        command_path_value = entry.get("command_path_model")
        if target_models and command_path_value is not None:
            raise ReplayAggregationError(f"Joint {joint_name} cannot select both target_models and command_path_model.")
        for model_value in target_models:
            model_path = _resolve_repo_path(str(model_value))
            target_model = _load_json(model_path, "Selected target model")
            if target_model.get("schema_version") != 2 or target_model.get("source_dataset_dry_run"):
                raise ReplayAggregationError(f"Selected target model is not accepted hardware schema v2: {model_path}")
            joints = target_model.get("joints")
            if not isinstance(joints, dict) or set(joints) != {joint_name}:
                raise ReplayAggregationError(f"Selected target model {model_path} does not contain only {joint_name}.")
            if joints[joint_name].get("quality", {}).get("target_randomization_usable") is not True:
                raise ReplayAggregationError(f"Selected target model failed its target gate: {model_path}")
            dataset_path = Path(str(target_model.get("source_dataset", ""))).expanduser().resolve()
            if dataset_path in expected:
                raise ReplayAggregationError(f"Dataset appears more than once in target selections: {dataset_path}")
            expected_sha256 = str(target_model.get("source_dataset_sha256", ""))
            if not expected_sha256 or _sha256(dataset_path) != expected_sha256:
                raise ReplayAggregationError(f"Dataset hash does not match target-model provenance: {dataset_path}")
            expected[dataset_path] = {
                "joint": joint_name,
                "target_model": _relative(model_path),
                "target_model_sha256": _sha256(model_path),
                "dataset": _relative(dataset_path),
                "dataset_sha256": expected_sha256,
            }
            by_joint[joint_name].append(dataset_path)
        if command_path_value is None:
            continue

        command_path = _resolve_repo_path(str(command_path_value))
        command_model = _load_json(command_path, "Selected command-path model")
        quality = command_model.get("quality")
        replay = command_model.get("sim_replay")
        sources = command_model.get("source_datasets")
        reports = replay.get("reports") if isinstance(replay, dict) else None
        if (
            command_model.get("schema_version") != 1
            or command_model.get("model_type") != "rnd_multi_amplitude_generalized_play"
            or command_model.get("joint") != joint_name
            or command_model.get("analysis_only") is not True
            or not isinstance(quality, dict)
            or quality.get("cross_amplitude_usable") is not True
            or quality.get("sim_replay_validated") is not True
            or quality.get("integration_allowed") is not False
            or not isinstance(sources, list)
            or not isinstance(reports, list)
            or len(sources) != len(reports)
        ):
            raise ReplayAggregationError(f"Selected command-path model is not replay-validated: {command_path}")
        reports_by_dataset = {
            _resolve_repo_path(str(report.get("dataset", ""))): report for report in reports if isinstance(report, dict)
        }
        if len(reports_by_dataset) != len(reports):
            raise ReplayAggregationError(f"Command-path replay provenance is malformed: {command_path}")
        for source in sources:
            if not isinstance(source, dict):
                raise ReplayAggregationError(f"Command-path source provenance is malformed: {command_path}")
            dataset_path = _resolve_repo_path(str(source.get("dataset", "")))
            if dataset_path in expected:
                raise ReplayAggregationError(f"Dataset appears more than once in target selections: {dataset_path}")
            expected_sha256 = str(source.get("dataset_sha256", ""))
            if not expected_sha256 or _sha256(dataset_path) != expected_sha256:
                raise ReplayAggregationError(f"Command-path source dataset hash mismatch: {dataset_path}")
            report = reports_by_dataset.get(dataset_path)
            if report is None:
                raise ReplayAggregationError(f"Command-path source has no selected replay report: {dataset_path}")
            report_path = _resolve_repo_path(str(report.get("report", "")))
            if _sha256(report_path) != report.get("report_sha256"):
                raise ReplayAggregationError(f"Command-path replay report hash mismatch: {report_path}")
            expected[dataset_path] = {
                "joint": joint_name,
                "target_model": _relative(command_path),
                "target_model_sha256": _sha256(command_path),
                "dataset": _relative(dataset_path),
                "dataset_sha256": expected_sha256,
                "required_report": report_path,
                "residual_delay_source": "applied",
            }
            by_joint[joint_name].append(dataset_path)
    return expected, by_joint


def _phase_metrics(report: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    thresholds = report.get("gate_thresholds")
    phases = report.get("phases")
    if not isinstance(thresholds, dict) or not isinstance(phases, list) or not phases:
        raise ReplayAggregationError(f"Replay report is missing gate thresholds or phases: {report_path}")
    minimum_r2 = _finite(thresholds.get("minimum_phase_r2"), "minimum_phase_r2")
    maximum_nrmse = _finite(thresholds.get("maximum_phase_normalized_rmse"), "maximum_phase_normalized_rmse")
    maximum_delay_error = _finite(
        thresholds.get("maximum_reference_delay_error_s"), "maximum_reference_delay_error_s", minimum=0.0
    )
    maximum_gain_error = _finite(
        thresholds.get("maximum_reference_gain_relative_error"),
        "maximum_reference_gain_relative_error",
        minimum=0.0,
    )
    reference_profile = str(report.get("reference_profile"))
    reduced: list[dict[str, Any]] = []
    reference_found = False
    for phase in phases:
        profile = str(phase.get("profile_name"))
        comparison = phase.get("hardware_vs_simulation")
        if not isinstance(comparison, dict):
            raise ReplayAggregationError(f"Phase {profile} is missing hardware_vs_simulation in {report_path}")
        r2 = _finite(comparison.get("r2"), f"{profile}.r2")
        nrmse = _finite(comparison.get("normalized_rmse"), f"{profile}.normalized_rmse", minimum=0.0)
        if r2 < minimum_r2 or nrmse > maximum_nrmse:
            raise ReplayAggregationError(f"Phase shape gate does not pass in {report_path}: {profile}")
        phase_summary: dict[str, Any] = {
            "profile_name": profile,
            "frequency_hz": _finite(phase.get("frequency_hz"), f"{profile}.frequency_hz", minimum=0.0),
            "sample_count": int(phase.get("sample_count")),
            "r2": r2,
            "normalized_rmse": nrmse,
            "rmse_rad": _finite(comparison.get("rmse_rad"), f"{profile}.rmse_rad", minimum=0.0),
        }
        if profile == reference_profile:
            reference_found = True
            delay_error = _finite(phase.get("delay_error_s"), f"{profile}.delay_error_s")
            gain_error = _finite(phase.get("gain_relative_error"), f"{profile}.gain_relative_error", minimum=0.0)
            if abs(delay_error) > maximum_delay_error or gain_error > maximum_gain_error:
                raise ReplayAggregationError(f"Reference response gate does not pass in {report_path}: {profile}")
            phase_summary["delay_error_s"] = delay_error
            phase_summary["gain_relative_error"] = gain_error
        reduced.append(phase_summary)
    if not reference_found:
        raise ReplayAggregationError(f"Reference profile {reference_profile!r} is absent from {report_path}")
    return reduced


def _load_replay_report(
    report_path: Path,
    expected: dict[str, Any],
    base_model_path: Path,
    base_model: dict[str, Any],
) -> dict[str, Any]:
    report = _load_json(report_path, "Simulator replay report")
    if report.get("schema_version") != 1:
        raise ReplayAggregationError(f"Unsupported replay schema in {report_path}")
    if report.get("validation_type") != "fixed_base_isaac_explicit_pd_replay":
        raise ReplayAggregationError(f"Unexpected replay validation type in {report_path}")
    if report.get("sim_replay_gate_satisfied") is not True:
        raise ReplayAggregationError(f"Selected replay gate did not pass: {report_path}")
    if report.get("automatic_model_update_performed") is not False:
        raise ReplayAggregationError(f"Replay report must not have modified the model: {report_path}")
    if str(report.get("joint")) != expected["joint"]:
        raise ReplayAggregationError(
            f"Replay joint mismatch for {report_path}: expected {expected['joint']}, got {report.get('joint')}."
        )
    if Path(str(report.get("model", ""))).expanduser().resolve() != base_model_path:
        raise ReplayAggregationError(f"Replay used a different actuator-model path: {report_path}")
    physics_hz = _finite(report.get("physics_hz"), "physics_hz", minimum=1.0e-9)
    sample_hz = _finite(report.get("sample_hz"), "sample_hz", minimum=1.0e-9)
    if not math.isclose(physics_hz, float(base_model["physics_hz"]), rel_tol=0.0, abs_tol=1.0e-9):
        raise ReplayAggregationError(f"Replay physics_hz does not match the base model: {report_path}")
    if not math.isclose(sample_hz, float(base_model["policy_hz"]), rel_tol=0.0, abs_tol=1.0e-9):
        raise ReplayAggregationError(f"Replay sample_hz does not match the base model policy_hz: {report_path}")

    controller = report.get("controller_settings")
    if not isinstance(controller, dict):
        raise ReplayAggregationError(f"Replay report is missing controller_settings: {report_path}")
    stiffness = _finite(controller.get("stiffness"), "controller_settings.stiffness", minimum=1.0e-9)
    damping = _finite(controller.get("damping"), "controller_settings.damping", minimum=0.0)
    position_bias = _finite(controller.get("residual_position_bias_rad", 0.0), "residual_position_bias_rad")
    if expected.get("residual_delay_source") == "applied":
        residual_delay = _finite(
            report.get("applied_residual_delay_s"),
            "applied_residual_delay_s",
            minimum=0.0,
        )
    else:
        residual_delay = _finite(
            report.get("recommended_total_residual_delay_s", report.get("recommended_residual_delay_s")),
            "recommended_total_residual_delay_s",
            minimum=0.0,
        )
    hardware_delay = _finite(report.get("reference_hardware_delay_s"), "reference_hardware_delay_s", minimum=0.0)
    simulation_delay = _finite(report.get("reference_simulation_delay_s"), "reference_simulation_delay_s", minimum=0.0)
    phases = _phase_metrics(report, report_path)
    return {
        "report": _relative(report_path),
        "report_sha256": _sha256(report_path),
        "dataset": expected["dataset"],
        "dataset_sha256": expected["dataset_sha256"],
        "target_model": expected["target_model"],
        "target_model_sha256": expected["target_model_sha256"],
        "controller": {
            "stiffness": stiffness,
            "damping": damping,
            "residual_position_bias_rad": position_bias,
        },
        "reference_profile": str(report.get("reference_profile")),
        "reference_hardware_delay_s": hardware_delay,
        "reference_simulation_delay_s": simulation_delay,
        "recommended_residual_delay_s": residual_delay,
        "phases": phases,
    }


def _consistent_value(values: list[float], label: str) -> float:
    if not values:
        raise ReplayAggregationError(f"Cannot select {label} from no values.")
    reference = values[0]
    if any(not math.isclose(value, reference, rel_tol=0.0, abs_tol=1.0e-9) for value in values[1:]):
        raise ReplayAggregationError(f"Repeated replay reports disagree on {label}: {values}")
    return float(statistics.median(values))


def aggregate_replay_reports(
    manifest_path: str | Path,
    model_path: str | Path,
    report_directory: str | Path,
) -> dict[str, Any]:
    """Aggregate every accepted target dataset into a fail-closed replay summary."""

    manifest_path = _resolve_repo_path(manifest_path)
    model_path = _resolve_repo_path(model_path)
    report_directory = _resolve_repo_path(report_directory)
    base_model = _load_json(model_path, "Base actuator model")
    validate_rnd_actuator_model(base_model)
    if base_model.get("integration_enabled") is not False:
        raise ReplayAggregationError("Base actuator model must remain integration_enabled=false during aggregation.")
    expected, expected_by_joint = _load_expected_datasets(manifest_path)

    selected_reports: dict[Path, tuple[Path, dict[str, Any]]] = {}
    unselected_reports: list[dict[str, Any]] = []
    report_paths = sorted(report_directory.glob("*_sim_replay*.json"))
    for report_path in report_paths:
        report = _load_json(report_path, "Simulator replay report")
        dataset_value = report.get("dataset")
        if not isinstance(dataset_value, str) or not dataset_value:
            raise ReplayAggregationError(f"Replay report has no dataset path: {report_path}")
        dataset_path = Path(dataset_value).expanduser().resolve()
        if dataset_path not in expected:
            unselected_reports.append({
                "report": _relative(report_path),
                "report_sha256": _sha256(report_path),
                "dataset": _relative(dataset_path),
                "joint": report.get("joint"),
                "sim_replay_gate_satisfied": report.get("sim_replay_gate_satisfied"),
            })
            continue
        required_report = expected[dataset_path].get("required_report")
        if required_report is not None and report_path.resolve() != required_report.resolve():
            unselected_reports.append({
                "report": _relative(report_path),
                "report_sha256": _sha256(report_path),
                "dataset": _relative(dataset_path),
                "joint": report.get("joint"),
                "sim_replay_gate_satisfied": report.get("sim_replay_gate_satisfied"),
                "reason": "superseded_by_command_path_selected_report",
            })
            continue
        if dataset_path in selected_reports:
            first = selected_reports[dataset_path][0]
            raise ReplayAggregationError(
                f"More than one replay report targets {dataset_path}: {first} and {report_path}"
            )
        selected_reports[dataset_path] = (report_path, report)

    missing = sorted(str(path) for path in set(expected) - set(selected_reports))
    if missing:
        raise ReplayAggregationError(f"Accepted target datasets are missing replay reports: {missing}")

    gate_thresholds: dict[str, Any] | None = None
    joints: dict[str, Any] = {}
    for joint_name in RND_LEG_JOINT_NAMES:
        datasets = expected_by_joint[joint_name]
        if not datasets:
            joints[joint_name] = {
                "command_path_seed_usable": False,
                "sim_replay_validated": False,
                "status": "unresolved_command_path",
                "report_count": 0,
                "reports": [],
            }
            continue
        reports: list[dict[str, Any]] = []
        for dataset_path in datasets:
            report_path, raw_report = selected_reports[dataset_path]
            current_thresholds = raw_report.get("gate_thresholds")
            if gate_thresholds is None:
                gate_thresholds = copy.deepcopy(current_thresholds)
            elif current_thresholds != gate_thresholds:
                raise ReplayAggregationError(f"Replay gate thresholds differ in {report_path}")
            reports.append(_load_replay_report(report_path, expected[dataset_path], model_path, base_model))
        stiffness = _consistent_value(
            [report["controller"]["stiffness"] for report in reports], f"{joint_name} stiffness"
        )
        damping = _consistent_value([report["controller"]["damping"] for report in reports], f"{joint_name} damping")
        position_bias = _consistent_value(
            [report["controller"]["residual_position_bias_rad"] for report in reports],
            f"{joint_name} residual position bias",
        )
        residual_delays = [report["recommended_residual_delay_s"] for report in reports]
        joints[joint_name] = {
            "command_path_seed_usable": True,
            "sim_replay_validated": True,
            "status": "sim_replay_validated_not_enabled",
            "report_count": len(reports),
            "selected_controller": {"stiffness": stiffness, "damping": damping},
            "residual_delay_s_range": [min(residual_delays), max(residual_delays)],
            "residual_position_bias_rad_range": [position_bias, position_bias],
            "reports": reports,
        }

    validated = [name for name, value in joints.items() if value["sim_replay_validated"]]
    unresolved = [name for name, value in joints.items() if not value["command_path_seed_usable"]]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "application_status": "sim_replay_aggregated_not_integrated",
        "integration_enabled": False,
        "manifest": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "base_actuator_model": _relative(model_path),
        "base_actuator_model_sha256": _sha256(model_path),
        "report_directory": _relative(report_directory),
        "gate_thresholds": gate_thresholds,
        "quality_summary": {
            "joint_count": len(joints),
            "accepted_target_dataset_count": len(expected),
            "discovered_report_count": len(report_paths),
            "selected_report_count": len(selected_reports),
            "unselected_report_count": len(unselected_reports),
            "passed_selected_report_count": len(selected_reports),
            "sim_replay_validated_joint_count": len(validated),
            "sim_replay_validated_joints": validated,
            "unresolved_joints": unresolved,
            "integration_ready": False,
        },
        "limitations": [
            "All replays are fixed-base, suspended-reference-pose validations; ground-contact dependence is unknown.",
            "Residual delay is phase-equivalent closed-loop delay, not pure communication latency.",
            "A repeated fixed position bias validates that candidate but does not estimate a bias uncertainty range.",
            "Replay reports record the actuator-model path but not its content hash; the current base-model hash is retained.",
            "Current-domain friction evidence is not converted to joint torque.",
            "This summary does not enable the actuator model or modify reinforcement-learning configuration.",
        ],
        "joints": joints,
        "unselected_reports": unselected_reports,
    }


def build_candidate_model(
    base_model: dict[str, Any],
    summary: dict[str, Any],
    summary_path: str | Path,
    summary_sha256: str,
) -> dict[str, Any]:
    """Apply aggregate values to a separate candidate while keeping it disabled."""

    candidate = copy.deepcopy(base_model)
    candidate["created_utc"] = summary["created_utc"]
    candidate["application_status"] = "sim_replay_aggregated_not_enabled"
    candidate["integration_enabled"] = False
    candidate["source_sim_replay_summary"] = _relative(_resolve_repo_path(summary_path))
    candidate["source_sim_replay_summary_sha256"] = summary_sha256
    for joint_name in RND_LEG_JOINT_NAMES:
        aggregate = summary["joints"][joint_name]
        joint = candidate["joints"][joint_name]
        joint["sim_replay"] = copy.deepcopy(aggregate)
        if not aggregate["sim_replay_validated"]:
            joint["quality"]["sim_replay_validated"] = False
            joint["quality"]["integration_allowed"] = False
            joint["quality"]["status"] = "unresolved_command_path"
            continue
        seed = joint["controller_seed"]
        seed["pre_replay_stiffness"] = float(seed["stiffness"])
        seed["pre_replay_damping"] = float(seed["damping"])
        seed["stiffness"] = float(aggregate["selected_controller"]["stiffness"])
        seed["damping"] = float(aggregate["selected_controller"]["damping"])
        joint["command_path"]["residual_delay_s_range"] = list(aggregate["residual_delay_s_range"])
        joint["command_path"]["residual_position_bias_rad_range"] = list(aggregate["residual_position_bias_rad_range"])
        joint["quality"]["sim_replay_validated"] = True
        joint["quality"]["integration_allowed"] = False
        joint["quality"]["status"] = "sim_replay_validated_not_enabled"

    aggregate_quality = summary["quality_summary"]
    candidate["quality_summary"]["sim_replay_validated_joint_count"] = aggregate_quality[
        "sim_replay_validated_joint_count"
    ]
    candidate["quality_summary"]["sim_replay_validated_joints"] = list(aggregate_quality["sim_replay_validated_joints"])
    candidate["quality_summary"]["integration_ready"] = False
    candidate["quality_summary"]["unresolved_joints"] = list(aggregate_quality["unresolved_joints"])
    candidate["limitations"] = list(candidate.get("limitations", [])) + [
        "This is a replay-validated candidate, not the default runtime model; integration_enabled remains false."
    ]
    if "L_Leg_ankle_roll" in aggregate_quality["unresolved_joints"]:
        candidate["limitations"].append(
            "The left ankle-roll command path remains unresolved and is not replaced with mirrored right-side values."
        )
    validate_rnd_actuator_model(candidate)
    return candidate


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = aggregate_replay_reports(args.manifest, args.model, args.report_directory)
        summary_output = _resolve_repo_path(args.summary_output)
        candidate_output = _resolve_repo_path(args.candidate_output)
        summary_text = _json_text(summary)
        summary_sha256 = _sha256_bytes(summary_text.encode("utf-8"))
        base_model = _load_json(_resolve_repo_path(args.model), "Base actuator model")
        candidate = build_candidate_model(base_model, summary, summary_output, summary_sha256)
        _atomic_write_text(summary_output, summary_text)
        _atomic_write_text(candidate_output, _json_text(candidate))
        quality = summary["quality_summary"]
        print(f"Saved replay aggregate: {summary_output}")
        print(f"Saved disabled actuator candidate: {candidate_output}")
        print(
            f"reports={quality['selected_report_count']}/{quality['accepted_target_dataset_count']}, "
            f"validated_joints={quality['sim_replay_validated_joint_count']}/{quality['joint_count']}, "
            f"unresolved={quality['unresolved_joints']}"
        )
        print("Default runtime model and reinforcement-learning configuration were not modified.")
        return 0
    except (OSError, ReplayAggregationError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
