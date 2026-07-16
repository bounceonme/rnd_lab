"""Pure helpers for selecting explicit-PD candidates from simulator replay results."""

from __future__ import annotations

import math
from typing import Any


class PDSweepError(ValueError):
    """Raised when a PD sweep cannot be configured or selected safely."""


def parse_positive_scales(value: str, option_name: str) -> list[float]:
    """Parse a comma-separated list of finite positive gain multipliers."""

    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise PDSweepError(f"{option_name} must be a comma-separated list of positive numbers.")
    scales: list[float] = []
    for field in fields:
        try:
            scale = float(field)
        except ValueError as error:
            raise PDSweepError(f"{option_name} contains a non-numeric value: {field!r}.") from error
        if not math.isfinite(scale) or scale <= 0.0:
            raise PDSweepError(f"{option_name} values must be finite and positive, got {scale!r}.")
        if scale not in scales:
            scales.append(scale)
    return scales


def build_pd_candidates(
    seed_stiffness: float,
    seed_damping: float,
    stiffness_scales: list[float],
    damping_scales: list[float],
) -> list[tuple[float, float]]:
    """Build a deterministic, duplicate-free Cartesian gain grid."""

    if not math.isfinite(seed_stiffness) or seed_stiffness <= 0.0:
        raise PDSweepError(f"Seed stiffness must be finite and positive, got {seed_stiffness!r}.")
    if not math.isfinite(seed_damping) or seed_damping < 0.0:
        raise PDSweepError(f"Seed damping must be finite and non-negative, got {seed_damping!r}.")
    if not stiffness_scales or not damping_scales:
        raise PDSweepError("PD sweep scale lists cannot be empty.")

    candidates: list[tuple[float, float]] = []
    for stiffness_scale in stiffness_scales:
        for damping_scale in damping_scales:
            candidate = (seed_stiffness * stiffness_scale, seed_damping * damping_scale)
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _require_finite(candidate: dict[str, Any], field: str) -> float:
    value = candidate.get(field)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise PDSweepError(f"Candidate field {field!r} must be finite, got {value!r}.")
    return float(value)


def select_pd_candidate(
    candidates: list[dict[str, Any]],
    *,
    seed_stiffness: float,
    seed_damping: float,
    maximum_delay_error_s: float,
) -> dict[str, Any]:
    """Select a conservative PD candidate without forcing residual compensation.

    Candidates already inside the replay delay gate are preferred and ranked by
    distance from the existing seed, then effort and gain magnitude. Residual
    delay is considered only when no valid candidate satisfies the delay gate.
    """

    if not math.isfinite(seed_stiffness) or seed_stiffness <= 0.0:
        raise PDSweepError(f"Seed stiffness must be finite and positive, got {seed_stiffness!r}.")
    if not math.isfinite(seed_damping) or seed_damping < 0.0:
        raise PDSweepError(f"Seed damping must be finite and non-negative, got {seed_damping!r}.")
    if not math.isfinite(maximum_delay_error_s) or maximum_delay_error_s < 0.0:
        raise PDSweepError(f"Maximum delay error must be finite and non-negative, got {maximum_delay_error_s!r}.")

    valid_candidates = [candidate for candidate in candidates if candidate.get("valid") is True]
    if not valid_candidates:
        raise PDSweepError("PD sweep produced no valid candidate.")

    def candidate_values(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        delay_error = _require_finite(candidate, "delay_error_s")
        gain_error = _require_finite(candidate, "gain_relative_error")
        normalized_rmse = _require_finite(candidate, "normalized_rmse")
        stiffness = _require_finite(candidate, "stiffness")
        damping = _require_finite(candidate, "damping")
        max_abs_effort = _require_finite(candidate, "max_abs_effort_nm")
        return delay_error, gain_error, normalized_rmse, stiffness, damping, max_abs_effort

    def seed_distance(stiffness: float, damping: float) -> float:
        distance = abs(stiffness / seed_stiffness - 1.0)
        if seed_damping > 0.0:
            distance += abs(damping / seed_damping - 1.0)
        else:
            distance += abs(damping)
        return distance

    def conservative_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        delay_error, gain_error, normalized_rmse, stiffness, damping, max_abs_effort = candidate_values(candidate)
        return (
            seed_distance(stiffness, damping),
            max_abs_effort,
            stiffness,
            damping,
            abs(delay_error),
            gain_error,
            normalized_rmse,
        )

    def correction_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        delay_error, gain_error, normalized_rmse, stiffness, damping, max_abs_effort = candidate_values(candidate)
        return (
            abs(delay_error),
            seed_distance(stiffness, damping),
            max_abs_effort,
            stiffness,
            damping,
            gain_error,
            normalized_rmse,
        )

    within_gate = [
        candidate
        for candidate in valid_candidates
        if abs(_require_finite(candidate, "delay_error_s")) <= maximum_delay_error_s
    ]
    compensable = [candidate for candidate in valid_candidates if _require_finite(candidate, "delay_error_s") <= 0.0]
    if within_gate:
        selected = min(within_gate, key=conservative_key)
        selection_mode = "within_gate_minimum_change"
    elif compensable:
        selected = min(compensable, key=correction_key)
        selection_mode = "non_negative_residual_compensable"
    else:
        selected = min(valid_candidates, key=correction_key)
        selection_mode = "least_slow_uncompensable_fallback"

    selected_delay_error = _require_finite(selected, "delay_error_s")
    return {
        "selected": selected,
        "selection_mode": selection_mode,
        "positive_residual_compensation_available": selected_delay_error <= 0.0,
        "selected_within_delay_gate": abs(selected_delay_error) <= maximum_delay_error_s,
        "within_delay_gate_candidate_count": len(within_gate),
        "residual_compensable_candidate_count": len(compensable),
        "valid_candidate_count": len(valid_candidates),
    }
