#!/usr/bin/env python3
"""Promote an aggregated RND actuator candidate into an explicit runtime scope."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[1]
_ROBOT_PACKAGE_DIR = _REPO_ROOT / "source" / "robot_lab" / "robot_lab"
sys.path.insert(0, str(_ROBOT_PACKAGE_DIR))

from actuators.rnd_stateful import validate_rnd_actuator_model


DEFAULT_CANDIDATE = _TOOL_DIR / "config" / "rnd_actuator_model_candidate.json"
DEFAULT_OUTPUT = _TOOL_DIR / "config" / "rnd_actuator_model_runtime.json"
DEFAULT_FALLBACK_JOINTS = ("L_Leg_ankle_roll",)

_CANDIDATE_ONLY_LIMITATION_PREFIXES = (
    "Every joint must pass fixed-base simulator replay before this file may be enabled in training.",
    "This is a replay-validated candidate, not the default runtime model;",
)


class ActuatorPromotionError(ValueError):
    """Raised when a candidate cannot be promoted without bypassing a quality gate."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a replay aggregate into a runtime model. The default retains the legacy left ankle-roll "
            "fallback; --full requires and integrates every replay-validated joint."
        )
    )
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE), help="Aggregated candidate JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Runtime model JSON.")
    parser.add_argument(
        "--fallback-joint",
        action="append",
        default=None,
        help="Joint excluded from the stateful model. Repeat for multiple joints.",
    )
    parser.add_argument("--full", action="store_true", help="Require and integrate all joints without fallbacks.")
    return parser


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ActuatorPromotionError(f"Candidate does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ActuatorPromotionError(f"Candidate is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ActuatorPromotionError(f"Candidate must contain a JSON object: {path}")
    return value


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


def promote_actuator_model(
    candidate: dict[str, Any],
    fallback_joint_names: tuple[str, ...] | list[str],
    *,
    source_candidate: str | None = None,
    source_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a fail-closed full or partial runtime model from an aggregate."""

    validate_rnd_actuator_model(candidate)
    if candidate.get("integration_enabled") is not False:
        raise ActuatorPromotionError("The input candidate must still have integration_enabled=false.")
    if candidate.get("application_status") != "sim_replay_aggregated_not_enabled":
        raise ActuatorPromotionError(
            "The input must be an aggregated, disabled candidate; "
            f"got application_status={candidate.get('application_status')!r}."
        )

    joint_order = tuple(candidate["joint_order"])
    fallback = tuple(fallback_joint_names)
    if len(fallback) != len(set(fallback)):
        raise ActuatorPromotionError("Fallback joint names must be unique.")
    unknown = sorted(set(fallback) - set(joint_order))
    if unknown:
        raise ActuatorPromotionError(f"Fallback joints are absent from the candidate: {unknown}.")
    integration = tuple(name for name in joint_order if name not in set(fallback))
    if not integration:
        raise ActuatorPromotionError("At least one joint must remain in the stateful integration scope.")

    runtime = copy.deepcopy(candidate)
    for joint_name in integration:
        quality = runtime["joints"][joint_name]["quality"]
        if quality.get("command_path_seed_usable") is not True or quality.get("sim_replay_validated") is not True:
            raise ActuatorPromotionError(f"Joint {joint_name} has not passed both command-path and replay gates.")
        quality["integration_allowed"] = True
        quality["status"] = "sim_replay_validated_runtime"
    for joint_name in fallback:
        quality = runtime["joints"][joint_name]["quality"]
        if quality.get("sim_replay_validated") is True:
            raise ActuatorPromotionError(
                f"Fallback joint {joint_name} is already replay validated; use full promotion instead."
            )
        quality["integration_allowed"] = False
        quality["status"] = "explicit_pd_fallback_pending_replacement"

    full_integration = not fallback
    runtime["created_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["application_status"] = "sim_replay_validated" if full_integration else "sim_replay_validated_partial"
    runtime["integration_enabled"] = True
    runtime["integration_joint_names"] = list(integration)
    runtime["fallback_joint_names"] = list(fallback)
    if source_candidate is not None:
        runtime["source_candidate"] = source_candidate
    if source_candidate_sha256 is not None:
        runtime["source_candidate_sha256"] = source_candidate_sha256

    quality_summary = runtime["quality_summary"]
    quality_summary["integration_ready"] = full_integration
    quality_summary["partial_integration_ready"] = not full_integration
    quality_summary["integration_joint_count"] = len(integration)
    quality_summary["integration_joint_names"] = list(integration)
    quality_summary["fallback_joint_count"] = len(fallback)
    quality_summary["fallback_joint_names"] = list(fallback)
    limitations = [
        limitation
        for limitation in runtime.get("limitations", [])
        if not any(limitation.startswith(prefix) for prefix in _CANDIDATE_ONLY_LIMITATION_PREFIXES)
    ]
    limitations.extend(
        [
            "Only joints listed in integration_joint_names are applied by the stateful runtime actuator.",
            "Changing or replacing any integrated motor requires new collection, fitting, replay, and promotion.",
        ]
    )
    if fallback:
        limitations.append(
            "Joints listed in fallback_joint_names use plain explicit PD and exclude the measured command path."
        )
    runtime["limitations"] = limitations

    validate_rnd_actuator_model(runtime)
    validate_rnd_actuator_model(
        runtime,
        integration,
        require_sim_replay_validation=True,
        require_command_path_seed=True,
    )
    return runtime


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.full and args.fallback_joint:
            raise ActuatorPromotionError("--full cannot be combined with --fallback-joint.")
        fallback = () if args.full else tuple(args.fallback_joint or DEFAULT_FALLBACK_JOINTS)
        candidate_path = _resolve_repo_path(args.candidate)
        output_path = _resolve_repo_path(args.output)
        runtime = promote_actuator_model(
            _load_json(candidate_path),
            fallback,
            source_candidate=_relative(candidate_path),
            source_candidate_sha256=_sha256(candidate_path),
        )
        _atomic_write_json(output_path, runtime)
        print(f"Saved actuator runtime model: {output_path}")
        print(
            f"integration_joints={len(runtime['integration_joint_names'])}, "
            f"fallback_joints={runtime['fallback_joint_names']}"
        )
        return 0
    except (OSError, ActuatorPromotionError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
