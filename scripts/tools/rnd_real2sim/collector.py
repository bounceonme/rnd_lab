"""Collection orchestration shared by the real hardware and dry-run CLI."""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .bus import MotorTelemetry
from .config import ExperimentConfig, ExcitationProfile, MappingConfig
from .dataset import DatasetRecorder
from .excitation import ScheduledCommand, build_schedule, effective_joint_limits, load_urdf_joint_limits


class SafetyTrip(RuntimeError):
    """Raised when live telemetry crosses a configured hard limit."""


class TickWatchdog:
    def __init__(self, joint_names: tuple[str, ...], max_stale_samples: int = 3):
        self._last = {name: None for name in joint_names}
        self._stale = {name: 0 for name in joint_names}
        self._max_stale_samples = max_stale_samples

    def update(self, telemetry: dict[str, MotorTelemetry]) -> None:
        stale_names = []
        for name, sample in telemetry.items():
            if self._last[name] == sample.tick_ms:
                self._stale[name] += 1
            else:
                self._stale[name] = 0
            self._last[name] = sample.tick_ms
            if self._stale[name] >= self._max_stale_samples:
                stale_names.append(name)
        if stale_names:
            raise SafetyTrip(f"Realtime Tick stopped changing for: {', '.join(stale_names)}")


def validate_telemetry(
    telemetry: dict[str, MotorTelemetry],
    goals_rad: dict[str, float] | None,
    experiment: ExperimentConfig,
    *,
    enforce_load_limits: bool,
    max_tracking_error_rad: float | None = None,
    max_current_a: float | None = None,
    max_pwm_fraction: float | None = None,
) -> None:
    safety = experiment.safety
    tracking_limit = safety.max_tracking_error_rad if max_tracking_error_rad is None else max_tracking_error_rad
    current_limit = safety.max_current_a if max_current_a is None else max_current_a
    pwm_limit = safety.max_pwm_fraction if max_pwm_fraction is None else max_pwm_fraction
    violations: list[str] = []
    for name, sample in telemetry.items():
        if not safety.min_voltage_v <= sample.voltage_v <= safety.max_voltage_v:
            violations.append(f"{name} voltage={sample.voltage_v:.2f}V")
        if sample.temperature_c > safety.max_temperature_c:
            violations.append(f"{name} temperature={sample.temperature_c:.1f}C")
        if enforce_load_limits:
            if abs(sample.current_a) > current_limit:
                violations.append(f"{name} current={sample.current_a:+.3f}A")
            if abs(sample.pwm_fraction) > pwm_limit:
                violations.append(f"{name} pwm={sample.pwm_fraction:+.3f}")
            if goals_rad is not None:
                error = abs(goals_rad[name] - sample.position_rad)
                if error > tracking_limit:
                    violations.append(f"{name} tracking_error={math.degrees(error):.2f}deg")
    if violations:
        raise SafetyTrip("Safety limit exceeded: " + "; ".join(violations))


def _runtime_metadata(bus) -> dict[str, dict[str, int]]:
    return {name: dataclasses.asdict(info) for name, info in bus.runtime_info.items()}


def _mapping_metadata(mapping: MappingConfig) -> list[dict[str, object]]:
    return [
        {
            "name": joint.name,
            "motor_id": joint.motor_id,
            "zero_raw": joint.zero_raw,
            "direction": joint.direction,
            "min_raw": joint.min_raw,
            "max_raw": joint.max_raw,
            "ticks_per_revolution": joint.ticks_per_revolution,
        }
        for joint in mapping.joints
    ]


def _default_metadata(
    mapping: MappingConfig,
    experiment: ExperimentConfig,
    urdf_path: Path,
    excitation_joint_names: tuple[str, ...],
    profiles: tuple[ExcitationProfile, ...],
    phase_metadata: dict[int, dict[str, object]],
    reference_pose_transition: dict[str, object],
    bus,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collector": "rnd_real2sim_collect",
        "dry_run": dry_run,
        "mapping_path": str(mapping.source_path),
        "experiment_config_path": str(experiment.source_path),
        "urdf_path": str(urdf_path),
        "device": "synthetic" if dry_run else mapping.device,
        "baudrate": mapping.baudrate,
        "protocol": mapping.protocol,
        "control_table": mapping.control_table,
        "sample_hz": experiment.sample_hz,
        "upper_body_constraint": "rigid_external_fixture",
        "imu_used": False,
        "excitation_joint_names": list(excitation_joint_names),
        "profile_names": [profile.name for profile in profiles],
        "phase_metadata": {str(key): value for key, value in phase_metadata.items()},
        "reference_pose": {
            "config": dataclasses.asdict(experiment.reference_pose),
            "transition": reference_pose_transition,
        },
        "joint_calibration": _mapping_metadata(mapping),
        "motor_runtime": _runtime_metadata(bus),
        "safety": dataclasses.asdict(experiment.safety),
    }


def describe_prearm_state(telemetry: dict[str, MotorTelemetry]) -> str:
    lines = ["All mapped motors are torque OFF. Measured pre-arm state:"]
    for name, sample in telemetry.items():
        lines.append(
            f"  {name:24s} q={math.degrees(sample.position_rad):+8.3f} deg  "
            f"V={sample.voltage_v:4.1f}  T={sample.temperature_c:4.1f} C"
        )
    return "\n".join(lines)


def describe_reference_pose_transition(
    telemetry: dict[str, MotorTelemetry], target_positions_rad: dict[str, float]
) -> str:
    lines = ["Configured automatic reference-pose transition:"]
    for name, sample in telemetry.items():
        target = target_positions_rad[name]
        delta = target - sample.position_rad
        lines.append(
            f"  {name:24s} current={math.degrees(sample.position_rad):+8.3f} deg  "
            f"target={math.degrees(target):+8.3f} deg  delta={math.degrees(delta):+8.3f} deg"
        )
    return "\n".join(lines)


def _validate_pose_within_limits(
    positions_rad: dict[str, float], allowed_limits: dict[str, tuple[float, float]], label: str
) -> None:
    violations = []
    for name, position in positions_rad.items():
        lower, upper = allowed_limits[name]
        if not lower <= position <= upper:
            violations.append(
                f"{name}={math.degrees(position):+.2f}deg outside "
                f"[{math.degrees(lower):+.2f}, {math.degrees(upper):+.2f}]deg"
            )
    if violations:
        raise SafetyTrip(f"{label} violates calibrated/URDF limits: " + "; ".join(violations))


def _validate_reference_start(
    positions_rad: dict[str, float], target_positions_rad: dict[str, float], max_deviation_rad: float
) -> None:
    violations = []
    for name, position in positions_rad.items():
        deviation = abs(target_positions_rad[name] - position)
        if deviation > max_deviation_rad:
            violations.append(f"{name} deviation={math.degrees(deviation):.2f}deg")
    if violations:
        raise SafetyTrip(
            f"Reference-pose start deviation exceeds {math.degrees(max_deviation_rad):.1f}deg: " + "; ".join(violations)
        )


def _move_to_reference_pose(
    *,
    bus,
    experiment: ExperimentConfig,
    start_positions_rad: dict[str, float],
    dry_run: bool,
) -> dict[str, object]:
    reference = experiment.reference_pose
    target_positions = reference.positions_rad
    joint_names = tuple(start_positions_rad)
    dt = experiment.dt
    max_delta = max(abs(target_positions[name] - start_positions_rad[name]) for name in joint_names)
    max_step_rad = min(experiment.safety.max_goal_step_rad, reference.move_speed_rad_s * dt)
    move_steps = max(1, math.ceil(max_delta / max_step_rad))
    settle_steps = max(1, round(reference.settle_s * experiment.sample_hz))
    command_count = move_steps + settle_steps
    sample_count = command_count + 1

    max_abs_current = {name: 0.0 for name in joint_names}
    max_abs_pwm = {name: 0.0 for name in joint_names}
    max_tracking_error = {name: 0.0 for name in joint_names}
    max_deadline_overrun = 0.0
    recent_cycle_intervals: deque[float] = deque(maxlen=50)
    consecutive_deadline_misses = 0
    previous_read_complete: float | None = None
    watchdog = TickWatchdog(joint_names)
    applied_goals = dict(start_positions_rad)
    final_telemetry: dict[str, MotorTelemetry] | None = None
    start_time = time.perf_counter()

    print(
        f"[INFO] Moving all 12 leg joints to the configured reference pose: "
        f"steps={move_steps}, settle={reference.settle_s:.1f}s, "
        f"estimated_duration={sample_count * dt:.1f}s"
    )
    for cycle_index in range(command_count + 1):
        target_time = start_time + (cycle_index + 1) * dt
        if dry_run:
            bus.advance(dt)
            wake_lateness = 0.0
        else:
            remaining = target_time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            before_read = time.perf_counter()
            wake_lateness = max(0.0, before_read - target_time)

        telemetry = bus.read_telemetry()
        read_complete = time.perf_counter()
        final_telemetry = telemetry
        if not dry_run:
            if previous_read_complete is not None:
                recent_cycle_intervals.append(read_complete - previous_read_complete)
            previous_read_complete = read_complete
            missed = wake_lateness > 0.5 * dt
            consecutive_deadline_misses = consecutive_deadline_misses + 1 if missed else 0
            if consecutive_deadline_misses >= experiment.safety.max_consecutive_deadline_misses:
                observed_hz = 1.0 / statistics.median(recent_cycle_intervals) if recent_cycle_intervals else 0.0
                raise SafetyTrip(
                    f"Reference-pose loop missed {consecutive_deadline_misses} consecutive deadlines at "
                    f"{experiment.sample_hz:.1f} Hz; observed throughput was about {observed_hz:.1f} Hz."
                )
        max_deadline_overrun = max(max_deadline_overrun, wake_lateness)

        watchdog.update(telemetry)
        validate_telemetry(
            telemetry,
            applied_goals,
            experiment,
            enforce_load_limits=True,
            max_tracking_error_rad=reference.max_tracking_error_rad,
            max_current_a=reference.max_current_a,
            max_pwm_fraction=reference.max_pwm_fraction,
        )
        for name, sample in telemetry.items():
            max_abs_current[name] = max(max_abs_current[name], abs(sample.current_a))
            max_abs_pwm[name] = max(max_abs_pwm[name], abs(sample.pwm_fraction))
            max_tracking_error[name] = max(max_tracking_error[name], abs(applied_goals[name] - sample.position_rad))

        if cycle_index == command_count:
            break
        if cycle_index < move_steps:
            fraction = (cycle_index + 1) / move_steps
            next_goals = {
                name: start_positions_rad[name] + fraction * (target_positions[name] - start_positions_rad[name])
                for name in joint_names
            }
        else:
            next_goals = dict(target_positions)
        bus.write_goal_positions(next_goals, max_step_rad)
        applied_goals = next_goals
        if cycle_index % max(1, round(experiment.sample_hz)) == 0:
            bus.check_hardware_errors()

    if final_telemetry is None:
        raise SafetyTrip("Reference-pose transition produced no telemetry.")
    final_positions = {name: final_telemetry[name].position_rad for name in joint_names}
    final_errors = {name: target_positions[name] - final_positions[name] for name in joint_names}
    out_of_tolerance = [
        f"{name} error={math.degrees(error):+.3f}deg"
        for name, error in final_errors.items()
        if abs(error) > reference.tolerance_rad
    ]
    if out_of_tolerance:
        raise SafetyTrip(
            f"Reference pose did not settle within {math.degrees(reference.tolerance_rad):.2f}deg: "
            + "; ".join(out_of_tolerance)
        )
    bus.check_hardware_errors()

    max_final_error_joint = max(final_errors, key=lambda name: abs(final_errors[name]))
    max_final_error = abs(final_errors[max_final_error_joint])
    max_current = max(max_abs_current.values())
    max_pwm = max(max_abs_pwm.values())
    print(
        f"[INFO] Reference pose reached: max_error={math.degrees(max_final_error):.3f}deg "
        f"({max_final_error_joint}), "
        f"max_current={max_current:.3f}A, max_pwm={max_pwm:.3f}"
    )
    return {
        "start_positions_rad": dict(start_positions_rad),
        "target_positions_rad": dict(target_positions),
        "final_positions_rad": final_positions,
        "final_errors_rad": final_errors,
        "move_steps": move_steps,
        "settle_steps": settle_steps,
        "nominal_duration_s": sample_count * dt,
        "actual_duration_s": sample_count * dt if dry_run else time.perf_counter() - start_time,
        "max_abs_current_a": max_abs_current,
        "max_abs_pwm_fraction": max_abs_pwm,
        "max_tracking_error_rad": max_tracking_error,
        "max_deadline_overrun_s": max_deadline_overrun,
    }


def collect_dataset(
    *,
    bus,
    mapping: MappingConfig,
    experiment: ExperimentConfig,
    urdf_path: str | Path,
    excitation_joint_names: tuple[str, ...],
    profiles: tuple[ExcitationProfile, ...],
    output_path: str | Path,
    dry_run: bool,
    confirm_arm: Callable[[str], bool],
) -> Path:
    joint_names = tuple(joint.name for joint in mapping.joints)
    resolved_urdf = Path(urdf_path).expanduser().resolve()
    urdf_limits = load_urdf_joint_limits(resolved_urdf, joint_names)
    allowed_limits = effective_joint_limits(mapping, urdf_limits, experiment.safety.position_limit_margin_rad)
    recorder: DatasetRecorder | None = None
    output = Path(output_path).expanduser().resolve()
    status = "aborted"
    status_detail = "Collection did not start."
    armed = False

    try:
        bus.open()
        bus.check_hardware_errors()
        prearm_telemetry = bus.read_telemetry()
        validate_telemetry(prearm_telemetry, None, experiment, enforce_load_limits=False)
        reference_positions = experiment.reference_pose.positions_rad
        _validate_pose_within_limits(reference_positions, allowed_limits, "Configured reference pose")
        build_schedule(
            mapping,
            experiment,
            reference_positions,
            excitation_joint_names,
            profiles,
            allowed_limits,
        )
        prearm_positions = {name: prearm_telemetry[name].position_rad for name in joint_names}
        print(describe_prearm_state(prearm_telemetry))
        print(describe_reference_pose_transition(prearm_telemetry, reference_positions))
        _validate_pose_within_limits(prearm_positions, allowed_limits, "Pre-arm pose")
        _validate_reference_start(
            prearm_positions,
            reference_positions,
            experiment.reference_pose.max_start_deviation_rad,
        )
        if not confirm_arm(
            "Authorize torque enable, automatic reference-pose motion, and data collection: "
        ):
            raise SafetyTrip("Hardware arming authorization was rejected.")

        # Re-read immediately before torque enable because joints can move freely while torque is OFF.
        pre_enable = bus.read_telemetry()
        pre_enable_positions = {name: pre_enable[name].position_rad for name in joint_names}
        _validate_pose_within_limits(pre_enable_positions, allowed_limits, "Pose immediately before torque enable")
        _validate_reference_start(
            pre_enable_positions,
            reference_positions,
            experiment.reference_pose.max_start_deviation_rad,
        )

        armed_positions = bus.enable_torque_safely(joint_names)
        armed = True
        _validate_reference_start(
            armed_positions,
            reference_positions,
            experiment.reference_pose.max_start_deviation_rad,
        )
        reference_pose_transition = _move_to_reference_pose(
            bus=bus,
            experiment=experiment,
            start_positions_rad=armed_positions,
            dry_run=dry_run,
        )
        schedule, phase_metadata = build_schedule(
            mapping,
            experiment,
            reference_positions,
            excitation_joint_names,
            profiles,
            allowed_limits,
        )
        estimated_duration = len(schedule) * experiment.dt
        print(
            f"[INFO] Armed at configured reference pose. samples={len(schedule) + 1}, "
            f"estimated_duration={estimated_duration:.1f}s, output={output}"
        )

        recorder = DatasetRecorder(
            joint_names,
            _default_metadata(
                mapping,
                experiment,
                resolved_urdf,
                excitation_joint_names,
                profiles,
                phase_metadata,
                reference_pose_transition,
                bus,
                dry_run,
            ),
        )
        watchdog = TickWatchdog(joint_names)
        applied = ScheduledCommand(dict(reference_positions), -1, "reference_pose", -1)
        previous_phase_key: tuple[int, int] | None = (-1, -1)
        consecutive_deadline_misses = 0
        recent_cycle_intervals: deque[float] = deque(maxlen=50)
        previous_read_complete: float | None = None
        start = time.perf_counter()

        # The extra final iteration records the response to the final scheduled command.
        for cycle_index in range(len(schedule) + 1):
            target_time = start + (cycle_index + 1) * experiment.dt
            if dry_run:
                bus.advance(experiment.dt)
                sample_time_s = (cycle_index + 1) * experiment.dt
                wake_lateness = 0.0
            else:
                remaining = target_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                before_read = time.perf_counter()
                wake_lateness = max(0.0, before_read - target_time)
                sample_time_s = before_read - start

            telemetry = bus.read_telemetry()
            read_complete = time.perf_counter()
            if not dry_run:
                if previous_read_complete is not None:
                    recent_cycle_intervals.append(read_complete - previous_read_complete)
                previous_read_complete = read_complete
                sample_time_s = read_complete - start
                missed = wake_lateness > 0.5 * experiment.dt
                consecutive_deadline_misses = consecutive_deadline_misses + 1 if missed else 0
                if consecutive_deadline_misses >= experiment.safety.max_consecutive_deadline_misses:
                    observed_hz = 1.0 / statistics.median(recent_cycle_intervals) if recent_cycle_intervals else 0.0
                    recommended_hz = max(20.0, 5.0 * math.floor(0.8 * observed_hz / 5.0))
                    raise SafetyTrip(
                        f"Collection loop missed {consecutive_deadline_misses} consecutive deadlines at "
                        f"{experiment.sample_hz:.1f} Hz. Observed throughput was about {observed_hz:.1f} Hz; "
                        f"set experiment.sample_hz to {recommended_hz:.0f} Hz or lower."
                    )

            watchdog.update(telemetry)
            validate_telemetry(telemetry, applied.goals_rad, experiment, enforce_load_limits=True)
            recorder.append(
                time_s=sample_time_s,
                phase_id=applied.phase_id,
                excitation_joint_id=applied.excitation_joint_id,
                deadline_overrun_s=wake_lateness,
                goals_rad=applied.goals_rad,
                telemetry=telemetry,
            )

            if cycle_index == len(schedule):
                break
            next_command = schedule[cycle_index]
            phase_key = (next_command.phase_id, next_command.excitation_joint_id)
            if phase_key != previous_phase_key:
                if next_command.phase_id >= 0:
                    bus.check_hardware_errors()
                    joint_name = joint_names[next_command.excitation_joint_id]
                    print(
                        f"[INFO] phase={next_command.phase_id:03d} joint={joint_name} profile={next_command.phase_name}"
                    )
                previous_phase_key = phase_key
            bus.write_goal_positions(next_command.goals_rad, experiment.safety.max_goal_step_rad)
            applied = next_command

        bus.check_hardware_errors()
        bus.disable_torque(joint_names)
        armed = False
        status = "complete"
        status_detail = "Collection completed and all mapped motors were torque-disabled."
    except BaseException as error:
        status_detail = f"{type(error).__name__}: {error}"
        bus.emergency_torque_off()
        armed = False
        raise
    finally:
        if armed:
            bus.emergency_torque_off()
        bus.close()
        if recorder is not None:
            saved = recorder.save(output, status=status, status_detail=status_detail)
            print(f"[INFO] Saved {status} dataset with {recorder.sample_count} samples: {saved}")

    return output
