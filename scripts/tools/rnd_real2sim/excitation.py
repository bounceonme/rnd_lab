"""Safe command scheduling for a rigidly suspended RND STEP robot."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig, ExcitationProfile, MappingConfig, Real2SimConfigError


@dataclass(frozen=True)
class ScheduledCommand:
    goals_rad: dict[str, float]
    phase_id: int
    phase_name: str
    excitation_joint_id: int


def load_urdf_joint_limits(path: str | Path, joint_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise Real2SimConfigError(f"URDF does not exist: {resolved}")
    try:
        root = ET.parse(resolved).getroot()
    except ET.ParseError as error:
        raise Real2SimConfigError(f"Could not parse URDF {resolved}: {error}") from error
    limits: dict[str, tuple[float, float]] = {}
    expected = set(joint_names)
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in expected:
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise Real2SimConfigError(f"URDF joint {name} has no finite lower/upper limit.")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise Real2SimConfigError(f"URDF joint {name} has invalid limits [{lower}, {upper}].")
        limits[name] = (lower, upper)
    if set(limits) != expected:
        raise Real2SimConfigError(f"URDF is missing mapped joints: {sorted(expected - set(limits))}")
    return limits


def effective_joint_limits(
    mapping: MappingConfig,
    urdf_limits: dict[str, tuple[float, float]],
    margin_rad: float,
) -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in mapping.joints:
        raw_endpoints = (
            joint.raw_to_radians(joint.min_raw),
            joint.raw_to_radians(joint.max_raw),
        )
        raw_lower, raw_upper = sorted(raw_endpoints)
        urdf_lower, urdf_upper = urdf_limits[joint.name]
        lower = max(raw_lower, urdf_lower) + margin_rad
        upper = min(raw_upper, urdf_upper) - margin_rad
        if lower >= upper:
            raise Real2SimConfigError(f"No usable range remains for {joint.name} after applying safety margin.")
        limits[joint.name] = (lower, upper)
    return limits


def _waveform(profile: ExcitationProfile, time_s: float) -> float:
    phase = 2.0 * math.pi * profile.frequency_hz * time_s
    if profile.waveform == "sine":
        return math.sin(phase)
    return 2.0 / math.pi * math.asin(math.sin(phase))


def _constant_segment(
    baseline: dict[str, float],
    duration_s: float,
    sample_hz: float,
    phase_id: int,
    phase_name: str,
    excitation_joint_id: int = -1,
) -> list[ScheduledCommand]:
    count = max(0, round(duration_s * sample_hz))
    return [ScheduledCommand(dict(baseline), phase_id, phase_name, excitation_joint_id) for _ in range(count)]


def build_schedule(
    mapping: MappingConfig,
    experiment: ExperimentConfig,
    baseline: dict[str, float],
    excitation_joint_names: tuple[str, ...],
    profiles: tuple[ExcitationProfile, ...],
    allowed_limits: dict[str, tuple[float, float]],
) -> tuple[list[ScheduledCommand], dict[int, dict[str, object]]]:
    if set(baseline) != set(mapping.joints_by_name):
        raise Real2SimConfigError("Baseline positions do not cover exactly the mapped joints.")
    if not excitation_joint_names:
        raise Real2SimConfigError("At least one excitation joint is required.")
    if not profiles:
        raise Real2SimConfigError("At least one excitation profile is required.")

    max_amplitude = max(profile.amplitude_rad for profile in profiles)
    for name, center in baseline.items():
        lower, upper = allowed_limits[name]
        if not lower <= center <= upper:
            raise Real2SimConfigError(
                f"{name} measured position {center:+.4f} rad is outside the allowed range [{lower:+.4f}, {upper:+.4f}]."
            )
        if name in excitation_joint_names and (center - max_amplitude < lower or center + max_amplitude > upper):
            raise Real2SimConfigError(
                f"{name} cannot move +/-{math.degrees(max_amplitude):.2f} deg around its current position without "
                "crossing a calibrated/URDF safety limit. Reposition it with torque OFF or reduce the profile."
            )

    schedule = _constant_segment(baseline, experiment.initial_settle_s, experiment.sample_hz, -1, "initial_settle")
    phase_metadata: dict[int, dict[str, object]] = {}
    phase_id = 0
    joint_index = {joint.name: index for index, joint in enumerate(mapping.joints)}
    dt = experiment.dt
    previous_goals = dict(baseline)
    for joint_name in excitation_joint_names:
        for profile in profiles:
            segments: list[tuple[int, str, int, bool]] = []
            if profile.precondition_cycles:
                segments.append((-1, f"precondition_{profile.name}", profile.precondition_cycles, False))
            segments.append((phase_id, profile.name, profile.cycles, True))

            for segment_phase_id, segment_name, segment_cycles, identify_segment in segments:
                duration_s = segment_cycles / profile.frequency_hz
                # Ceil guarantees an integer-cycle endpoint before the following baseline settle.
                sample_count = max(1, math.ceil(duration_s * experiment.sample_hz))
                if identify_segment:
                    phase_metadata[phase_id] = {
                        "joint_name": joint_name,
                        "profile_name": profile.name,
                        "waveform": profile.waveform,
                        "amplitude_rad": profile.amplitude_rad,
                        "frequency_hz": profile.frequency_hz,
                        "cycles": profile.cycles,
                        "precondition_cycles": profile.precondition_cycles,
                    }
                for sample_index in range(1, sample_count + 1):
                    local_time = min(sample_index * dt, duration_s)
                    goals = dict(baseline)
                    goals[joint_name] += profile.amplitude_rad * _waveform(profile, local_time)
                    for name, goal in goals.items():
                        step = abs(goal - previous_goals[name])
                        if step > experiment.safety.max_goal_step_rad + 1.0e-12:
                            raise Real2SimConfigError(
                                f"Profile {segment_name} produces a {math.degrees(step):.3f} deg sample step on "
                                f"{name}, above safety.max_goal_step_deg. Lower frequency/amplitude or raise sample_hz."
                            )
                        if abs(goal - baseline[name]) > experiment.safety.max_excursion_rad + 1.0e-12:
                            raise Real2SimConfigError(f"Profile {segment_name} exceeds max excursion for {name}.")
                    schedule.append(
                        ScheduledCommand(goals, segment_phase_id, segment_name, joint_index[joint_name])
                    )
                    previous_goals = goals
                schedule.extend(
                    _constant_segment(
                        baseline,
                        experiment.inter_profile_settle_s,
                        experiment.sample_hz,
                        -1,
                        f"settle_after_{segment_name}",
                        joint_index[joint_name],
                    )
                )
                previous_goals = dict(baseline)
            phase_id += 1

    schedule.extend(_constant_segment(baseline, experiment.return_duration_s, experiment.sample_hz, -1, "final_return"))
    return schedule, phase_metadata
