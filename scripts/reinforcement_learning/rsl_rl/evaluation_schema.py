"""Strict schema and hashing helpers for fixed-domain policy evaluation suites.

The schema is intentionally independent of Isaac Lab.  It accepts JSON-native
values only, rejects unknown fields, and treats every half-open scenario range
as part of the reproducibility contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EVALUATION_SPLITS = ("validation", "test")

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationSchemaError(ValueError):
    """Raised when a fixed-evaluation document is ambiguous or invalid."""


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationSchemaError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _validate_json_native(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationSchemaError(f"{path} must not contain NaN or infinity.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_native(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvaluationSchemaError(f"{path} contains a non-string object key: {key!r}.")
            _validate_json_native(item, f"{path}.{key}")
        return
    raise EvaluationSchemaError(f"{path} contains a non-JSON value of type {type(value).__name__}.")


def canonical_json_bytes(document: Any) -> bytes:
    """Return the project's canonical UTF-8 JSON representation.

    Canonical documents contain JSON-native finite values, sorted object keys,
    no insignificant whitespace, and no trailing newline.
    """

    _validate_json_native(document)
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise EvaluationSchemaError(f"Document cannot be encoded as canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def canonical_json_sha256(document: Any) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file as raw bytes without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, path: str, required_keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationSchemaError(f"{path} must be a JSON object.")
    expected = set(required_keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise EvaluationSchemaError(f"{path} has an invalid shape: {', '.join(details)}.")
    return value


def _list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluationSchemaError(f"{path} must be a JSON list.")
    if nonempty and not value:
        raise EvaluationSchemaError(f"{path} must not be empty.")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationSchemaError(f"{path} must be a non-empty string.")
    return value


def _identifier(value: Any, path: str) -> str:
    identifier = _string(value, path)
    if _ID_PATTERN.fullmatch(identifier) is None:
        raise EvaluationSchemaError(f"{path} must match {_ID_PATTERN.pattern!r}; got {identifier!r}.")
    return identifier


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationSchemaError(f"{path} must be a bool.")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationSchemaError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise EvaluationSchemaError(f"{path} must be >= {minimum}; got {value}.")
    if maximum is not None and value > maximum:
        raise EvaluationSchemaError(f"{path} must be <= {maximum}; got {value}.")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationSchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationSchemaError(f"{path} must be finite.")
    if minimum is not None:
        if strict_minimum and result <= minimum:
            raise EvaluationSchemaError(f"{path} must be > {minimum}; got {result}.")
        if not strict_minimum and result < minimum:
            raise EvaluationSchemaError(f"{path} must be >= {minimum}; got {result}.")
    if maximum is not None and result > maximum:
        raise EvaluationSchemaError(f"{path} must be <= {maximum}; got {result}.")
    return result


def _unique_string_list(value: Any, path: str, *, nonempty: bool = True) -> tuple[str, ...]:
    values = _list(value, path, nonempty=nonempty)
    names = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(values))
    if len(names) != len(set(names)):
        raise EvaluationSchemaError(f"{path} contains duplicate values.")
    return names


def _split(value: Any, path: str) -> str:
    split = _string(value, path)
    if split not in EVALUATION_SPLITS:
        raise EvaluationSchemaError(f"{path} must be one of {EVALUATION_SPLITS}; got {split!r}.")
    return split


def _validate_bounds(value: Any, path: str, axes: Sequence[str]) -> dict[str, tuple[float, float]]:
    bounds = _object(value, path, axes)
    parsed: dict[str, tuple[float, float]] = {}
    for axis in axes:
        entry = _object(bounds[axis], f"{path}.{axis}", ("minimum", "maximum"))
        lower = _number(entry["minimum"], f"{path}.{axis}.minimum")
        upper = _number(entry["maximum"], f"{path}.{axis}.maximum")
        if lower >= upper:
            raise EvaluationSchemaError(
                f"{path}.{axis} is an invalid range: minimum {lower} must be less than maximum {upper}."
            )
        parsed[axis] = (lower, upper)
    return parsed


def _validate_in_bounds(value: Any, path: str, bounds: tuple[float, float]) -> float:
    parsed = _number(value, path)
    if not bounds[0] <= parsed <= bounds[1]:
        raise EvaluationSchemaError(f"{path}={parsed} is outside the closed training range [{bounds[0]}, {bounds[1]}].")
    return parsed


def _validate_singleton_tree(value: Any, path: str) -> None:
    """Reject unresolved collections and range-shaped keys below ``resolved``.

    Encoder vectors are the only lists allowed because each of their 12 entries
    resolves one named joint rather than representing an interval.
    """

    if isinstance(value, list):
        if path.endswith(".encoder.zero_offset_rad") or path.endswith(".encoder.sample_age_s"):
            if len(value) != 12:
                raise EvaluationSchemaError(f"{path} must contain exactly 12 resolved joint values.")
            for index, item in enumerate(value):
                _number(item, f"{path}[{index}]")
            return
        raise EvaluationSchemaError(
            f"{path} must be resolved to singleton values; JSON lists, including two-value ranges, are forbidden."
        )
    if isinstance(value, dict):
        if not value:
            raise EvaluationSchemaError(f"{path} must not be an empty object.")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EvaluationSchemaError(f"{path} contains an invalid object key.")
            if "range" in key.lower():
                raise EvaluationSchemaError(f"{path}.{key} is unresolved: range fields are forbidden.")
            _validate_singleton_tree(item, f"{path}.{key}")
        return
    if value is None:
        raise EvaluationSchemaError(f"{path} must be a resolved singleton, not null.")
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        if not value:
            raise EvaluationSchemaError(f"{path} must not be an empty string.")
        return
    if isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise EvaluationSchemaError(f"{path} must be a finite JSON scalar or a nested object of scalars.")


def _bounded_domain_scalar(value: Any, path: str, lower: float, upper: float) -> float:
    result = _number(value, path)
    if not lower <= result <= upper:
        raise EvaluationSchemaError(f"{path}={result} is outside the current training envelope [{lower}, {upper}].")
    return result


def _fixed_one(value: Any, path: str) -> None:
    if not math.isclose(_number(value, path), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise EvaluationSchemaError(f"{path} must remain 1.0 because that quantity is not randomized in training.")


def _validate_resolved_domain(value: Any, path: str) -> None:
    """Validate the runtime-applicable RND STEP fixed-domain contract."""

    domain = _object(
        value,
        path,
        ("material", "base_mass_add_kg", "other_mass_scale", "base_com_offset_m", "encoder", "actuator", "imu"),
    )
    material = _object(
        domain["material"],
        f"{path}.material",
        ("static_friction", "dynamic_friction", "restitution"),
    )
    static_friction = _bounded_domain_scalar(
        material["static_friction"], f"{path}.material.static_friction", 0.55, 1.15
    )
    dynamic_friction = _bounded_domain_scalar(
        material["dynamic_friction"], f"{path}.material.dynamic_friction", 0.40, 0.90
    )
    if dynamic_friction > static_friction:
        raise EvaluationSchemaError(f"{path}.material.dynamic_friction cannot exceed static_friction.")
    _bounded_domain_scalar(material["restitution"], f"{path}.material.restitution", 0.0, 0.15)
    _bounded_domain_scalar(domain["base_mass_add_kg"], f"{path}.base_mass_add_kg", 0.1952887, 0.4152887)
    _bounded_domain_scalar(domain["other_mass_scale"], f"{path}.other_mass_scale", 0.95, 1.05)

    com = _object(domain["base_com_offset_m"], f"{path}.base_com_offset_m", ("x", "y", "z"))
    _bounded_domain_scalar(com["x"], f"{path}.base_com_offset_m.x", -0.015, 0.015)
    _bounded_domain_scalar(com["y"], f"{path}.base_com_offset_m.y", -0.015, 0.015)
    _bounded_domain_scalar(com["z"], f"{path}.base_com_offset_m.z", -0.010, 0.010)

    encoder = _object(domain["encoder"], f"{path}.encoder", ("zero_offset_rad", "sample_age_s"))
    zero_offsets = _list(encoder["zero_offset_rad"], f"{path}.encoder.zero_offset_rad")
    sample_ages = _list(encoder["sample_age_s"], f"{path}.encoder.sample_age_s")
    if len(zero_offsets) != 12 or len(sample_ages) != 12:
        raise EvaluationSchemaError(f"{path}.encoder vectors must each contain exactly 12 joint values.")
    for index, raw in enumerate(zero_offsets):
        _bounded_domain_scalar(raw, f"{path}.encoder.zero_offset_rad[{index}]", -0.005, 0.005)
    for index, raw in enumerate(sample_ages):
        _bounded_domain_scalar(raw, f"{path}.encoder.sample_age_s[{index}]", 0.0, 0.005)

    actuator = _object(
        domain["actuator"],
        f"{path}.actuator",
        (
            "stiffness_scale",
            "damping_scale",
            "motor_strength_scale",
            "coulomb_torque_nm",
            "friction_transition_velocity_rad_s",
        ),
    )
    for key in ("stiffness_scale", "damping_scale"):
        _fixed_one(actuator[key], f"{path}.actuator.{key}")
    _bounded_domain_scalar(actuator["motor_strength_scale"], f"{path}.actuator.motor_strength_scale", 0.8, 1.25)
    # This interval is the intersection of every joint's current Coulomb range.
    _bounded_domain_scalar(
        actuator["coulomb_torque_nm"],
        f"{path}.actuator.coulomb_torque_nm",
        0.17186861675963908,
        0.17405707995827602,
    )
    _bounded_domain_scalar(
        actuator["friction_transition_velocity_rad_s"],
        f"{path}.actuator.friction_transition_velocity_rad_s",
        0.03490658503988659,
        0.13962634015954636,
    )

    imu = _object(domain["imu"], f"{path}.imu", ("gyro", "gravity"))
    gyro = _object(imu["gyro"], f"{path}.imu.gyro", ("delay_s", "noise_sigma", "bias"))
    gravity = _object(imu["gravity"], f"{path}.imu.gravity", ("delay_s", "noise_sigma", "bias"))
    _bounded_domain_scalar(gyro["delay_s"], f"{path}.imu.gyro.delay_s", 0.0, 0.005)
    _bounded_domain_scalar(gyro["noise_sigma"], f"{path}.imu.gyro.noise_sigma", 0.0003, 0.003)
    _bounded_domain_scalar(gyro["bias"], f"{path}.imu.gyro.bias", -0.01, 0.01)
    _bounded_domain_scalar(gravity["delay_s"], f"{path}.imu.gravity.delay_s", 0.0, 0.02)
    _bounded_domain_scalar(gravity["noise_sigma"], f"{path}.imu.gravity.noise_sigma", 0.00005, 0.002)
    if _number(gravity["bias"], f"{path}.imu.gravity.bias") != 0.0:
        raise EvaluationSchemaError(f"{path}.imu.gravity.bias must be 0.0; the runtime channel has no bias model.")


def _validate_metric_contract(value: Any, task_foot_order: tuple[str, ...]) -> None:
    contract = _object(
        value,
        "$.metric_contract",
        (
            "version",
            "fall",
            "tracking",
            "posture",
            "gait",
            "torque_saturation",
            "push_recovery",
            "aggregation",
        ),
    )
    if _integer(contract["version"], "$.metric_contract.version") != 1:
        raise EvaluationSchemaError("$.metric_contract.version must be 1.")

    fall = _object(
        contract["fall"],
        "$.metric_contract.fall",
        ("termination_semantics", "horizon_completion_is_survival"),
    )
    if _string(fall["termination_semantics"], "$.metric_contract.fall.termination_semantics") != (
        "early_termination_is_fall"
    ):
        raise EvaluationSchemaError("Unsupported fall termination semantics.")
    _boolean(fall["horizon_completion_is_survival"], "$.metric_contract.fall.horizon_completion_is_survival")

    tracking = _object(
        contract["tracking"],
        "$.metric_contract.tracking",
        ("linear_velocity_frame", "yaw_rate_frame", "root_quaternion_order", "rmse_definition"),
    )
    expected_tracking = {
        "linear_velocity_frame": "root_yaw",
        "yaw_rate_frame": "world_z",
        "root_quaternion_order": "wxyz",
        "rmse_definition": "root_mean_squared_euclidean_error",
    }
    for key, expected in expected_tracking.items():
        actual = _string(tracking[key], f"$.metric_contract.tracking.{key}")
        if actual != expected:
            raise EvaluationSchemaError(f"$.metric_contract.tracking.{key} must be {expected!r}; got {actual!r}.")

    posture = _object(
        contract["posture"],
        "$.metric_contract.posture",
        ("root_quaternion_order", "lateral_tilt_component", "sagittal_tilt_component", "sample_gate"),
    )
    expected_posture = {
        "root_quaternion_order": "wxyz",
        "lateral_tilt_component": "projected_gravity_body_x",
        "sagittal_tilt_component": "projected_gravity_body_y",
        "sample_gate": "command_xy_speed_above_gait_threshold",
    }
    for key, expected in expected_posture.items():
        actual = _string(posture[key], f"$.metric_contract.posture.{key}")
        if actual != expected:
            raise EvaluationSchemaError(f"$.metric_contract.posture.{key} must be {expected!r}; got {actual!r}.")

    gait = _object(
        contract["gait"],
        "$.metric_contract.gait",
        ("foot_order", "minimum_touchdown_progress_m", "tap_max_air_time_s", "command_speed_threshold_m_s"),
    )
    gait_foot_order = _unique_string_list(gait["foot_order"], "$.metric_contract.gait.foot_order")
    if gait_foot_order != task_foot_order:
        raise EvaluationSchemaError("Metric-contract foot_order must match task.foot_order.")
    _number(
        gait["minimum_touchdown_progress_m"],
        "$.metric_contract.gait.minimum_touchdown_progress_m",
        minimum=0.0,
        strict_minimum=True,
    )
    _number(gait["tap_max_air_time_s"], "$.metric_contract.gait.tap_max_air_time_s", minimum=0.0)
    _number(
        gait["command_speed_threshold_m_s"],
        "$.metric_contract.gait.command_speed_threshold_m_s",
        minimum=0.0,
    )

    torque = _object(
        contract["torque_saturation"],
        "$.metric_contract.torque_saturation",
        ("threshold_fraction",),
    )
    _number(
        torque["threshold_fraction"],
        "$.metric_contract.torque_saturation.threshold_fraction",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )

    recovery = _object(
        contract["push_recovery"],
        "$.metric_contract.push_recovery",
        ("linear_velocity_error_threshold_m_s", "yaw_rate_error_threshold_rad_s", "dwell_s"),
    )
    for key in ("linear_velocity_error_threshold_m_s", "yaw_rate_error_threshold_rad_s", "dwell_s"):
        _number(
            recovery[key],
            f"$.metric_contract.push_recovery.{key}",
            minimum=0.0,
            strict_minimum=True,
        )

    aggregation = _object(
        contract["aggregation"],
        "$.metric_contract.aggregation",
        ("order", "episode_weighting", "case_weighting"),
    )
    expected_aggregation = {
        "order": "episode_then_case",
        "episode_weighting": "equal",
        "case_weighting": "equal",
    }
    for key, expected in expected_aggregation.items():
        actual = _string(aggregation[key], f"$.metric_contract.aggregation.{key}")
        if actual != expected:
            raise EvaluationSchemaError(f"$.metric_contract.aggregation.{key} must be {expected!r}; got {actual!r}.")


def _scenario_fingerprint(scenario: Mapping[str, Any]) -> str:
    semantic = {
        "horizon_steps": scenario["horizon_steps"],
        "segments": [
            {
                "start_step": segment["start_step"],
                "end_step": segment["end_step"],
                "command": segment["command"],
            }
            for segment in scenario["segments"]
        ],
        "pulses": [
            {
                "start_step": pulse["start_step"],
                "end_step": pulse["end_step"],
                "root_velocity": pulse["root_velocity"],
            }
            for pulse in scenario["pulses"]
        ],
    }
    return canonical_json_sha256(semantic)


def _register_unique_id(identifier: str, all_entity_ids: set[str]) -> None:
    if identifier in all_entity_ids:
        raise EvaluationSchemaError(f"Duplicate ID: {identifier!r}.")
    all_entity_ids.add(identifier)


def _other_split(split: str) -> str:
    return "test" if split == "validation" else "validation"


def _validate_suite_metadata(value: Any) -> None:
    suite = _object(value, "$.suite", ("id", "description", "splits"))
    _identifier(suite["id"], "$.suite.id")
    _string(suite["description"], "$.suite.description")
    splits = _unique_string_list(suite["splits"], "$.suite.splits")
    if splits != EVALUATION_SPLITS:
        raise EvaluationSchemaError(f"$.suite.splits must be ordered exactly as {EVALUATION_SPLITS}.")


def _validate_task(
    value: Any,
) -> tuple[int, tuple[str, ...], tuple[str, ...], dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    task = _object(
        value,
        "$.task",
        (
            "id",
            "command_name",
            "episode_horizon_steps",
            "expected_actor_observation_dimension",
            "root_quaternion_order",
            "foot_order",
            "joint_order",
            "artifact_ids",
            "command_bounds",
            "push_velocity_bounds",
        ),
    )
    _string(task["id"], "$.task.id")
    _string(task["command_name"], "$.task.command_name")
    episode_horizon = _integer(task["episode_horizon_steps"], "$.task.episode_horizon_steps", minimum=1)
    if _integer(
        task["expected_actor_observation_dimension"],
        "$.task.expected_actor_observation_dimension",
        minimum=1,
    ) != 171:
        raise EvaluationSchemaError("$.task.expected_actor_observation_dimension must be 171 for this suite.")
    if _string(task["root_quaternion_order"], "$.task.root_quaternion_order") != "wxyz":
        raise EvaluationSchemaError("$.task.root_quaternion_order must be 'wxyz'.")
    foot_order = _unique_string_list(task["foot_order"], "$.task.foot_order")
    if foot_order != ("right", "left"):
        raise EvaluationSchemaError("$.task.foot_order must be ['right', 'left'].")
    _unique_string_list(task["joint_order"], "$.task.joint_order")
    artifact_ids = _unique_string_list(task["artifact_ids"], "$.task.artifact_ids")
    axes = ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s")
    command_bounds = _validate_bounds(task["command_bounds"], "$.task.command_bounds", axes)
    push_bounds = _validate_bounds(task["push_velocity_bounds"], "$.task.push_velocity_bounds", axes)
    return episode_horizon, foot_order, artifact_ids, command_bounds, push_bounds


def _validate_rates(value: Any) -> None:
    rates = _object(value, "$.rates", ("physics_hz", "policy_hz", "contact_hz", "metric_hz"))
    physics_hz = _number(rates["physics_hz"], "$.rates.physics_hz", minimum=0.0, strict_minimum=True)
    policy_hz = _number(rates["policy_hz"], "$.rates.policy_hz", minimum=0.0, strict_minimum=True)
    contact_hz = _number(rates["contact_hz"], "$.rates.contact_hz", minimum=0.0, strict_minimum=True)
    metric_hz = _number(rates["metric_hz"], "$.rates.metric_hz", minimum=0.0, strict_minimum=True)
    decimation = physics_hz / policy_hz
    if not math.isclose(decimation, round(decimation), rel_tol=0.0, abs_tol=1.0e-12):
        raise EvaluationSchemaError("$.rates.physics_hz / policy_hz must be an integer decimation.")
    if not math.isclose(contact_hz, physics_hz, rel_tol=0.0, abs_tol=1.0e-12):
        raise EvaluationSchemaError("$.rates.contact_hz must equal physics_hz.")
    if not math.isclose(metric_hz, policy_hz, rel_tol=0.0, abs_tol=1.0e-12):
        raise EvaluationSchemaError("$.rates.metric_hz must equal policy_hz.")


def _validate_artifacts(
    value: Any,
    task_artifact_ids: tuple[str, ...],
    all_entity_ids: set[str],
) -> None:
    artifacts = _list(value, "$.artifacts", nonempty=True)
    artifact_ids: set[str] = set()
    for index, raw_artifact in enumerate(artifacts):
        path = f"$.artifacts[{index}]"
        artifact = _object(raw_artifact, path, ("id", "path", "sha256"))
        artifact_id = _identifier(artifact["id"], f"{path}.id")
        _register_unique_id(artifact_id, all_entity_ids)
        artifact_ids.add(artifact_id)
        artifact_path = _string(artifact["path"], f"{path}.path")
        path_parts = Path(artifact_path).parts
        if Path(artifact_path).is_absolute() or ".." in path_parts or "\\" in artifact_path:
            raise EvaluationSchemaError(f"{path}.path must be a repository-relative POSIX path.")
        sha256 = _string(artifact["sha256"], f"{path}.sha256")
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise EvaluationSchemaError(f"{path}.sha256 must be a lowercase 64-character SHA-256 digest.")
    if set(task_artifact_ids) != artifact_ids:
        missing = sorted(set(task_artifact_ids) - artifact_ids)
        unreferenced = sorted(artifact_ids - set(task_artifact_ids))
        raise EvaluationSchemaError(
            f"$.task.artifact_ids has undefined references {missing} or unreferenced artifacts {unreferenced}."
        )


def _validate_domains(value: Any, all_entity_ids: set[str]) -> dict[str, dict[str, Any]]:
    domains = _list(value, "$.domains", nonempty=True)
    domains_by_id: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {split: set() for split in EVALUATION_SPLITS}
    for index, raw_domain in enumerate(domains):
        path = f"$.domains[{index}]"
        domain = _object(raw_domain, path, ("id", "split", "resolved"))
        domain_id = _identifier(domain["id"], f"{path}.id")
        _register_unique_id(domain_id, all_entity_ids)
        split = _split(domain["split"], f"{path}.split")
        _validate_singleton_tree(domain["resolved"], f"{path}.resolved")
        _validate_resolved_domain(domain["resolved"], f"{path}.resolved")
        fingerprint = canonical_json_sha256(domain["resolved"])
        other_split = _other_split(split)
        if fingerprint in fingerprints[other_split]:
            raise EvaluationSchemaError(
                f"{path}.resolved duplicates a fixed domain in the {other_split} split (split leakage)."
            )
        fingerprints[split].add(fingerprint)
        domains_by_id[domain_id] = domain
    return domains_by_id


def _validate_segments(
    scenario: Mapping[str, Any],
    path: str,
    scenario_id: str,
    horizon: int,
    command_bounds: Mapping[str, tuple[float, float]],
) -> None:
    axes = tuple(command_bounds)
    segments = _list(scenario["segments"], f"{path}.segments", nonempty=True)
    segment_ids: set[str] = set()
    cursor = 0
    for index, raw_segment in enumerate(segments):
        segment_path = f"{path}.segments[{index}]"
        segment = _object(raw_segment, segment_path, ("id", "start_step", "end_step", "command"))
        segment_id = _identifier(segment["id"], f"{segment_path}.id")
        if segment_id in segment_ids:
            raise EvaluationSchemaError(f"Duplicate segment ID {segment_id!r} in scenario {scenario_id!r}.")
        segment_ids.add(segment_id)
        start = _integer(segment["start_step"], f"{segment_path}.start_step", minimum=0)
        end = _integer(segment["end_step"], f"{segment_path}.end_step", minimum=1)
        if start != cursor:
            issue = "overlap" if start < cursor else "gap"
            raise EvaluationSchemaError(
                f"{segment_path} creates a segment {issue}: expected start_step {cursor}, got {start}."
            )
        if end <= start or end > horizon:
            raise EvaluationSchemaError(
                f"{segment_path} has invalid half-open range [{start}, {end}) for horizon {horizon}."
            )
        command = _object(segment["command"], f"{segment_path}.command", axes)
        for axis in axes:
            _validate_in_bounds(command[axis], f"{segment_path}.command.{axis}", command_bounds[axis])
        if _number(command["lin_vel_x_m_s"], f"{segment_path}.command.lin_vel_x_m_s") != 0.0:
            raise EvaluationSchemaError(f"{segment_path}.command.lin_vel_x_m_s must be 0.0 for the fixed STEP runtime.")
        cursor = end
    if cursor != horizon:
        raise EvaluationSchemaError(f"{path}.segments leave a gap [{cursor}, {horizon}) at the end of the scenario.")


def _validate_pulses(
    scenario: Mapping[str, Any],
    path: str,
    scenario_id: str,
    horizon: int,
    push_bounds: Mapping[str, tuple[float, float]],
) -> set[str]:
    axes = tuple(push_bounds)
    pulses = _list(scenario["pulses"], f"{path}.pulses")
    if len(pulses) > 1:
        raise EvaluationSchemaError(f"{path}.pulses supports at most one isolated physical force pulse.")
    pulse_ids: set[str] = set()
    previous_end = 0
    for index, raw_pulse in enumerate(pulses):
        pulse_path = f"{path}.pulses[{index}]"
        pulse = _object(raw_pulse, pulse_path, ("id", "start_step", "end_step", "root_velocity"))
        pulse_id = _identifier(pulse["id"], f"{pulse_path}.id")
        if pulse_id in pulse_ids:
            raise EvaluationSchemaError(f"Duplicate pulse ID {pulse_id!r} in scenario {scenario_id!r}.")
        pulse_ids.add(pulse_id)
        start = _integer(pulse["start_step"], f"{pulse_path}.start_step", minimum=0)
        end = _integer(pulse["end_step"], f"{pulse_path}.end_step", minimum=1)
        if start < previous_end:
            raise EvaluationSchemaError(f"{pulse_path} overlaps the previous pulse.")
        if end <= start or end > horizon:
            raise EvaluationSchemaError(
                f"{pulse_path} has invalid half-open range [{start}, {end}) for horizon {horizon}."
            )
        if end - start != 6:
            raise EvaluationSchemaError(f"{pulse_path} must last exactly 6 policy steps (0.12 s at 50 Hz).")
        root_velocity = _object(pulse["root_velocity"], f"{pulse_path}.root_velocity", axes)
        parsed_velocity: dict[str, float] = {}
        for axis in axes:
            parsed_velocity[axis] = _validate_in_bounds(
                root_velocity[axis], f"{pulse_path}.root_velocity.{axis}", push_bounds[axis]
            )
        if parsed_velocity["ang_vel_z_rad_s"] != 0.0:
            raise EvaluationSchemaError(
                f"{pulse_path}.root_velocity.ang_vel_z_rad_s must be 0.0 for a zero-torque force pulse."
            )
        if parsed_velocity["lin_vel_x_m_s"] == 0.0 and parsed_velocity["lin_vel_y_m_s"] == 0.0:
            raise EvaluationSchemaError(f"{pulse_path} must request a non-zero translational impulse.")
        previous_end = end
    return pulse_ids


def _validate_scenarios(
    value: Any,
    all_entity_ids: set[str],
    episode_horizon: int,
    command_bounds: Mapping[str, tuple[float, float]],
    push_bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    scenarios = _list(value, "$.scenarios", nonempty=True)
    scenarios_by_id: dict[str, dict[str, Any]] = {}
    pulse_ids_by_scenario: dict[str, set[str]] = {}
    fingerprints: dict[str, set[str]] = {split: set() for split in EVALUATION_SPLITS}
    for index, raw_scenario in enumerate(scenarios):
        path = f"$.scenarios[{index}]"
        scenario = _object(raw_scenario, path, ("id", "split", "horizon_steps", "segments", "pulses"))
        scenario_id = _identifier(scenario["id"], f"{path}.id")
        _register_unique_id(scenario_id, all_entity_ids)
        split = _split(scenario["split"], f"{path}.split")
        horizon = _integer(scenario["horizon_steps"], f"{path}.horizon_steps", minimum=1)
        if horizon > episode_horizon:
            raise EvaluationSchemaError(
                f"{path}.horizon_steps={horizon} exceeds task episode horizon {episode_horizon}."
            )
        _validate_segments(scenario, path, scenario_id, horizon, command_bounds)
        pulse_ids_by_scenario[scenario_id] = _validate_pulses(scenario, path, scenario_id, horizon, push_bounds)
        fingerprint = _scenario_fingerprint(scenario)
        other_split = _other_split(split)
        if fingerprint in fingerprints[other_split]:
            raise EvaluationSchemaError(
                f"{path} duplicates scenario content in the {other_split} split (split leakage)."
            )
        fingerprints[split].add(fingerprint)
        scenarios_by_id[scenario_id] = scenario
    return scenarios_by_id, pulse_ids_by_scenario


def _validate_case_pulse_reference(
    case: Mapping[str, Any], path: str, scenario_id: str, pulse_ids_by_scenario: Mapping[str, set[str]]
) -> None:
    recovery_pulse_id = case["recovery_pulse_id"]
    pulse_ids = pulse_ids_by_scenario[scenario_id]
    if recovery_pulse_id is None:
        if pulse_ids:
            raise EvaluationSchemaError(f"{path}.recovery_pulse_id must select the scenario pulse.")
        return
    recovery_pulse_id = _identifier(recovery_pulse_id, f"{path}.recovery_pulse_id")
    if recovery_pulse_id not in pulse_ids:
        raise EvaluationSchemaError(f"{path}.recovery_pulse_id references undefined pulse {recovery_pulse_id!r}.")


def _validate_cases(
    value: Any,
    all_entity_ids: set[str],
    domains_by_id: Mapping[str, Mapping[str, Any]],
    scenarios_by_id: Mapping[str, Mapping[str, Any]],
    pulse_ids_by_scenario: Mapping[str, set[str]],
) -> None:
    cases = _list(value, "$.cases", nonempty=True)
    referenced_domains: set[str] = set()
    referenced_scenarios: set[str] = set()
    count_by_split = {split: 0 for split in EVALUATION_SPLITS}
    fingerprints: dict[str, set[str]] = {split: set() for split in EVALUATION_SPLITS}
    for index, raw_case in enumerate(cases):
        path = f"$.cases[{index}]"
        case = _object(
            raw_case,
            path,
            ("id", "split", "domain_id", "scenario_id", "seed", "episodes", "recovery_pulse_id"),
        )
        case_id = _identifier(case["id"], f"{path}.id")
        _register_unique_id(case_id, all_entity_ids)
        split = _split(case["split"], f"{path}.split")
        domain_id = _identifier(case["domain_id"], f"{path}.domain_id")
        scenario_id = _identifier(case["scenario_id"], f"{path}.scenario_id")
        if domain_id not in domains_by_id:
            raise EvaluationSchemaError(f"{path}.domain_id references undefined domain {domain_id!r}.")
        if scenario_id not in scenarios_by_id:
            raise EvaluationSchemaError(f"{path}.scenario_id references undefined scenario {scenario_id!r}.")
        if domains_by_id[domain_id]["split"] != split:
            raise EvaluationSchemaError(f"{path}.domain_id crosses the validation/test split (split leakage).")
        if scenarios_by_id[scenario_id]["split"] != split:
            raise EvaluationSchemaError(f"{path}.scenario_id crosses the validation/test split (split leakage).")
        _integer(case["seed"], f"{path}.seed", minimum=0, maximum=(1 << 63) - 1)
        _integer(case["episodes"], f"{path}.episodes", minimum=1)
        _validate_case_pulse_reference(case, path, scenario_id, pulse_ids_by_scenario)

        fingerprint = canonical_json_sha256({
            "domain": domains_by_id[domain_id]["resolved"],
            "scenario": _scenario_fingerprint(scenarios_by_id[scenario_id]),
        })
        if fingerprint in fingerprints[_other_split(split)]:
            raise EvaluationSchemaError(f"{path} duplicates a domain/scenario combination across splits.")
        fingerprints[split].add(fingerprint)
        referenced_domains.add(domain_id)
        referenced_scenarios.add(scenario_id)
        count_by_split[split] += 1

    unused_domains = sorted(set(domains_by_id) - referenced_domains)
    if unused_domains:
        raise EvaluationSchemaError(f"Every domain must be referenced by a case; unused IDs: {unused_domains}.")
    unused_scenarios = sorted(set(scenarios_by_id) - referenced_scenarios)
    if unused_scenarios:
        raise EvaluationSchemaError(f"Every scenario must be referenced by a case; unused IDs: {unused_scenarios}.")
    if any(count_by_split[split] == 0 for split in EVALUATION_SPLITS):
        raise EvaluationSchemaError("Both validation and test splits must contain at least one case.")


def validate_evaluation_suite(document: Mapping[str, Any]) -> None:
    """Validate a parsed fixed-domain evaluation suite.

    Validation is fail-closed: unknown fields, unresolved domains, cross-split
    references or duplicate split content, and incomplete scenario timelines are
    all rejected.
    """

    _validate_json_native(document)
    root = _object(
        document,
        "$",
        ("schema_version", "suite", "task", "rates", "artifacts", "metric_contract", "domains", "scenarios", "cases"),
    )
    if _integer(root["schema_version"], "$.schema_version") != SCHEMA_VERSION:
        raise EvaluationSchemaError(f"$.schema_version must be {SCHEMA_VERSION}.")

    _validate_suite_metadata(root["suite"])
    episode_horizon, foot_order, artifact_ids, command_bounds, push_bounds = _validate_task(root["task"])
    _validate_rates(root["rates"])
    all_entity_ids: set[str] = set()
    _validate_artifacts(root["artifacts"], artifact_ids, all_entity_ids)
    _validate_metric_contract(root["metric_contract"], foot_order)
    domains_by_id = _validate_domains(root["domains"], all_entity_ids)
    scenarios_by_id, pulse_ids_by_scenario = _validate_scenarios(
        root["scenarios"], all_entity_ids, episode_horizon, command_bounds, push_bounds
    )
    _validate_cases(root["cases"], all_entity_ids, domains_by_id, scenarios_by_id, pulse_ids_by_scenario)


def verify_artifact_hashes(document: Mapping[str, Any], repository_root: str | Path) -> None:
    """Verify every declared artifact against its repository-relative SHA-256."""

    validate_evaluation_suite(document)
    root = Path(repository_root).expanduser().resolve()
    for artifact in document["artifacts"]:
        artifact_path = (root / artifact["path"]).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as error:
            raise EvaluationSchemaError(
                f"Artifact {artifact['id']!r} resolves outside repository root: {artifact_path}."
            ) from error
        if not artifact_path.is_file():
            raise EvaluationSchemaError(f"Artifact {artifact['id']!r} does not exist: {artifact_path}.")
        actual = sha256_file(artifact_path)
        if actual != artifact["sha256"]:
            raise EvaluationSchemaError(
                f"Artifact {artifact['id']!r} SHA-256 mismatch: expected {artifact['sha256']}, got {actual}."
            )


def load_evaluation_suite(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_artifacts: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load, strictly validate, and optionally hash-check an evaluation suite.

    ``expected_sha256`` refers to canonical JSON content, not the raw file bytes.
    ``repository_root`` is required when ``verify_artifacts`` is true.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        document = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except OSError as error:
        raise EvaluationSchemaError(f"Unable to read evaluation suite {resolved}: {error}") from error
    except json.JSONDecodeError as error:
        raise EvaluationSchemaError(f"Evaluation suite is not valid JSON: {resolved}: {error}") from error
    if not isinstance(document, dict):
        raise EvaluationSchemaError("Evaluation suite root must be a JSON object.")
    validate_evaluation_suite(document)

    actual_sha256 = canonical_json_sha256(document)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise EvaluationSchemaError("expected_sha256 must be a lowercase 64-character SHA-256 digest.")
        if actual_sha256 != expected_sha256:
            raise EvaluationSchemaError(
                f"Evaluation suite canonical SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}."
            )

    if verify_artifacts:
        if repository_root is None:
            raise EvaluationSchemaError("repository_root is required when verify_artifacts=True.")
        verify_artifact_hashes(document, repository_root)
    return document


__all__ = [
    "EVALUATION_SPLITS",
    "SCHEMA_VERSION",
    "EvaluationSchemaError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_evaluation_suite",
    "sha256_file",
    "validate_evaluation_suite",
    "verify_artifact_hashes",
]
