"""Configuration and calibration loading for standalone real-to-sim tests."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RND_LEG_JOINT_NAMES = (
    "R_Leg_hip_yaw",
    "R_Leg_hip_roll",
    "R_Leg_hip_pitch",
    "R_Leg_knee",
    "R_Leg_ankle_pitch",
    "R_Leg_ankle_roll",
    "L_Leg_hip_yaw",
    "L_Leg_hip_roll",
    "L_Leg_hip_pitch",
    "L_Leg_knee",
    "L_Leg_ankle_pitch",
    "L_Leg_ankle_roll",
)


class Real2SimConfigError(ValueError):
    """Raised when an experiment or motor mapping is incomplete or unsafe."""


@dataclass(frozen=True)
class JointCalibration:
    name: str
    motor_id: int
    zero_raw: int
    direction: int
    min_raw: int
    max_raw: int
    ticks_per_revolution: int

    @property
    def radians_per_tick(self) -> float:
        return 2.0 * math.pi / self.ticks_per_revolution

    def raw_to_radians(self, raw_position: int, *, strict: bool = True) -> float:
        if strict and not self.min_raw <= raw_position <= self.max_raw:
            raise Real2SimConfigError(
                f"{self.name} (ID {self.motor_id}) returned raw position {raw_position}, outside "
                f"[{self.min_raw}, {self.max_raw}]."
            )
        return self.direction * (raw_position - self.zero_raw) * self.radians_per_tick

    def radians_to_raw(self, position_rad: float) -> int:
        raw = round(self.zero_raw + self.direction * position_rad / self.radians_per_tick)
        if not self.min_raw <= raw <= self.max_raw:
            raise Real2SimConfigError(
                f"{self.name} target {position_rad:+.5f} rad converts to raw {raw}, outside "
                f"[{self.min_raw}, {self.max_raw}]."
            )
        return raw


@dataclass(frozen=True)
class MappingConfig:
    source_path: Path
    device: str
    baudrate: int
    protocol: float
    control_table: str
    joints: tuple[JointCalibration, ...]

    @property
    def joints_by_name(self) -> dict[str, JointCalibration]:
        return {joint.name: joint for joint in self.joints}


@dataclass(frozen=True)
class ExcitationProfile:
    name: str
    waveform: str
    amplitude_rad: float
    frequency_hz: float
    cycles: int
    precondition_cycles: int

    @property
    def duration_s(self) -> float:
        return self.cycles / self.frequency_hz


@dataclass(frozen=True)
class SafetyConfig:
    max_goal_step_rad: float
    max_excursion_rad: float
    position_limit_margin_rad: float
    max_tracking_error_rad: float
    max_current_a: float
    max_pwm_fraction: float
    max_temperature_c: float
    min_voltage_v: float
    max_voltage_v: float
    max_consecutive_deadline_misses: int


@dataclass(frozen=True)
class ReferencePoseConfig:
    positions_rad: dict[str, float]
    move_speed_rad_s: float
    settle_s: float
    tolerance_rad: float
    max_start_deviation_rad: float
    max_tracking_error_rad: float
    max_current_a: float
    max_pwm_fraction: float


@dataclass(frozen=True)
class IdentificationConfig:
    max_delay_s: float
    motion_threshold_rad: float
    velocity_threshold_rad_s: float
    backlash_velocity_threshold_rad_s: float
    min_reversal_events: int
    validation_fraction: float
    nominal_torque_per_amp_nm: float
    randomization_margin: float


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    sample_hz: float
    initial_settle_s: float
    inter_profile_settle_s: float
    return_duration_s: float
    reference_pose: ReferencePoseConfig
    profiles: tuple[ExcitationProfile, ...]
    safety: SafetyConfig
    identification: IdentificationConfig

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_hz


def _load_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise Real2SimConfigError(f"Configuration file does not exist: {resolved}")
    try:
        with resolved.open("rb") as stream:
            return resolved, tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise Real2SimConfigError(f"Invalid TOML in {resolved}: {error}") from error


def _require_number(table: dict[str, Any], key: str, context: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Real2SimConfigError(f"{context}.{key} must be numeric, got {value!r}.")
    return float(value)


def _require_int(table: dict[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Real2SimConfigError(f"{context}.{key} must be an integer, got {value!r}.")
    return value


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    table = data.get(key)
    if not isinstance(table, dict):
        raise Real2SimConfigError(f"Configuration requires a [{key}] table.")
    return table


def _load_reference_pose(data: dict[str, Any], safety: SafetyConfig) -> ReferencePoseConfig:
    positions_data = _require_table(data, "positions_deg")
    missing_joints = sorted(set(RND_LEG_JOINT_NAMES) - set(positions_data))
    extra_joints = sorted(set(positions_data) - set(RND_LEG_JOINT_NAMES))
    if missing_joints or extra_joints:
        raise Real2SimConfigError(
            "reference_pose.positions_deg must contain exactly the 12 RND leg joints. "
            f"Missing={missing_joints}, extra={extra_joints}."
        )

    positions_rad = {}
    for name in RND_LEG_JOINT_NAMES:
        position_deg = _require_number(positions_data, name, "reference_pose.positions_deg")
        if not -180.0 <= position_deg <= 180.0:
            raise Real2SimConfigError(f"reference_pose.positions_deg.{name} must be in [-180, 180].")
        positions_rad[name] = math.radians(position_deg)

    reference_pose = ReferencePoseConfig(
        positions_rad=positions_rad,
        move_speed_rad_s=math.radians(_require_number(data, "move_speed_deg_s", "reference_pose")),
        settle_s=_require_number(data, "settle_s", "reference_pose"),
        tolerance_rad=math.radians(_require_number(data, "tolerance_deg", "reference_pose")),
        max_start_deviation_rad=math.radians(_require_number(data, "max_start_deviation_deg", "reference_pose")),
        max_tracking_error_rad=math.radians(_require_number(data, "max_tracking_error_deg", "reference_pose")),
        max_current_a=_require_number(data, "max_current_a", "reference_pose"),
        max_pwm_fraction=_require_number(data, "max_pwm_fraction", "reference_pose"),
    )
    if not math.radians(1.0) <= reference_pose.move_speed_rad_s <= math.radians(30.0):
        raise Real2SimConfigError("reference_pose.move_speed_deg_s must be in [1, 30].")
    if not 0.2 <= reference_pose.settle_s <= 10.0:
        raise Real2SimConfigError("reference_pose.settle_s must be in [0.2, 10].")
    if not math.radians(0.1) <= reference_pose.tolerance_rad <= math.radians(3.0):
        raise Real2SimConfigError("reference_pose.tolerance_deg must be in [0.1, 3].")
    if not math.radians(5.0) <= reference_pose.max_start_deviation_rad <= math.pi:
        raise Real2SimConfigError("reference_pose.max_start_deviation_deg must be in [5, 180].")
    if not reference_pose.tolerance_rad < reference_pose.max_tracking_error_rad <= safety.max_tracking_error_rad:
        raise Real2SimConfigError(
            "reference_pose.max_tracking_error_deg must exceed tolerance_deg and not exceed "
            "safety.max_tracking_error_deg."
        )
    if not 0.1 <= reference_pose.max_current_a <= safety.max_current_a:
        raise Real2SimConfigError("reference_pose.max_current_a must be in [0.1, safety.max_current_a].")
    if not 0.05 <= reference_pose.max_pwm_fraction <= safety.max_pwm_fraction:
        raise Real2SimConfigError("reference_pose.max_pwm_fraction must be in [0.05, safety.max_pwm_fraction].")
    return reference_pose


def load_mapping_config(path: str | Path) -> MappingConfig:
    """Read the verified joint-test TOML as immutable calibration data.

    This parser is deliberately independent of ``robot_lab.hardware``. Only
    Protocol 2.0 MX-106(2.0) mappings are accepted by the real-to-sim tool.
    """

    resolved, data = _load_toml(path)
    bus = _require_table(data, "bus")
    joints_data = data.get("joints")
    if bus.get("configured") is not True:
        raise Real2SimConfigError(f"{resolved}: bus.configured must be true.")
    if not isinstance(joints_data, list):
        raise Real2SimConfigError("Configuration requires repeated [[joints]] tables.")

    device = bus.get("device")
    baudrate = bus.get("baudrate")
    protocol = bus.get("protocol")
    control_table = bus.get("control_table")
    ticks_per_revolution = bus.get("ticks_per_revolution", 4096)
    if not isinstance(device, str) or not device:
        raise Real2SimConfigError("bus.device must be a non-empty string.")
    if isinstance(baudrate, bool) or not isinstance(baudrate, int) or baudrate <= 0:
        raise Real2SimConfigError("bus.baudrate must be a positive integer.")
    if protocol not in (2, 2.0):
        raise Real2SimConfigError("Real-to-sim collection requires Dynamixel Protocol 2.0.")
    if control_table != "mx_2":
        raise Real2SimConfigError("Real-to-sim collection requires control_table='mx_2'.")
    if not isinstance(ticks_per_revolution, int) or ticks_per_revolution != 4096:
        raise Real2SimConfigError("MX-106(2.0) mapping must use ticks_per_revolution=4096.")

    calibrations: list[JointCalibration] = []
    for index, item in enumerate(joints_data):
        context = f"joints[{index}]"
        if not isinstance(item, dict):
            raise Real2SimConfigError(f"{context} must be a table.")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise Real2SimConfigError(f"{context}.name must be a non-empty string.")
        calibration = JointCalibration(
            name=name,
            motor_id=_require_int(item, "id", context),
            zero_raw=_require_int(item, "zero_raw", context),
            direction=_require_int(item, "direction", context),
            min_raw=_require_int(item, "min_raw", context),
            max_raw=_require_int(item, "max_raw", context),
            ticks_per_revolution=ticks_per_revolution,
        )
        if not 0 <= calibration.motor_id <= 252:
            raise Real2SimConfigError(f"{context}.id must be in [0, 252].")
        if calibration.direction not in (-1, 1):
            raise Real2SimConfigError(f"{context}.direction must be -1 or 1.")
        if not calibration.min_raw < calibration.max_raw:
            raise Real2SimConfigError(f"{context} requires min_raw < max_raw.")
        if not calibration.min_raw <= calibration.zero_raw <= calibration.max_raw:
            raise Real2SimConfigError(f"{context}.zero_raw must be inside the raw limits.")
        calibrations.append(calibration)

    names = [joint.name for joint in calibrations]
    ids = [joint.motor_id for joint in calibrations]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise Real2SimConfigError("Joint names and Dynamixel IDs must each be unique.")
    if set(names) != set(RND_LEG_JOINT_NAMES):
        missing = sorted(set(RND_LEG_JOINT_NAMES) - set(names))
        extra = sorted(set(names) - set(RND_LEG_JOINT_NAMES))
        raise Real2SimConfigError(f"RND leg mapping mismatch. Missing={missing}, extra={extra}.")

    by_name = {joint.name: joint for joint in calibrations}
    ordered = tuple(by_name[name] for name in RND_LEG_JOINT_NAMES)
    return MappingConfig(resolved, device, baudrate, 2.0, "mx_2", ordered)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    resolved, data = _load_toml(path)
    experiment = _require_table(data, "experiment")
    safety_data = _require_table(data, "safety")
    reference_pose_data = _require_table(data, "reference_pose")
    identification_data = _require_table(data, "identification")
    profiles_data = data.get("profiles")
    if not isinstance(profiles_data, list) or not profiles_data:
        raise Real2SimConfigError("Configuration requires at least one [[profiles]] table.")

    sample_hz = _require_number(experiment, "sample_hz", "experiment")
    initial_settle_s = _require_number(experiment, "initial_settle_s", "experiment")
    inter_profile_settle_s = _require_number(experiment, "inter_profile_settle_s", "experiment")
    return_duration_s = _require_number(experiment, "return_duration_s", "experiment")
    if not 20.0 <= sample_hz <= 200.0:
        raise Real2SimConfigError("experiment.sample_hz must be in [20, 200] Hz.")
    if min(initial_settle_s, inter_profile_settle_s, return_duration_s) < 0.0:
        raise Real2SimConfigError("Experiment durations must be non-negative.")

    profiles: list[ExcitationProfile] = []
    seen_profile_names: set[str] = set()
    for index, item in enumerate(profiles_data):
        context = f"profiles[{index}]"
        if not isinstance(item, dict):
            raise Real2SimConfigError(f"{context} must be a table.")
        name = item.get("name")
        waveform = item.get("waveform")
        if not isinstance(name, str) or not name or name in seen_profile_names:
            raise Real2SimConfigError(f"{context}.name must be non-empty and unique.")
        if waveform not in ("sine", "triangle"):
            raise Real2SimConfigError(f"{context}.waveform must be 'sine' or 'triangle'.")
        amplitude_deg = _require_number(item, "amplitude_deg", context)
        frequency_hz = _require_number(item, "frequency_hz", context)
        cycles = _require_int(item, "cycles", context)
        precondition_cycles = item.get("precondition_cycles", 0)
        if isinstance(precondition_cycles, bool) or not isinstance(precondition_cycles, int):
            raise Real2SimConfigError(f"{context}.precondition_cycles must be an integer.")
        if not 0.25 <= amplitude_deg <= 20.0:
            raise Real2SimConfigError(f"{context}.amplitude_deg must be in [0.25, 20.0].")
        if not 0.02 <= frequency_hz <= 2.0:
            raise Real2SimConfigError(f"{context}.frequency_hz must be in [0.02, 2.0].")
        if not 1 <= cycles <= 20:
            raise Real2SimConfigError(f"{context}.cycles must be in [1, 20].")
        if not 0 <= precondition_cycles <= 5:
            raise Real2SimConfigError(f"{context}.precondition_cycles must be in [0, 5].")
        profiles.append(
            ExcitationProfile(
                name=name,
                waveform=waveform,
                amplitude_rad=math.radians(amplitude_deg),
                frequency_hz=frequency_hz,
                cycles=cycles,
                precondition_cycles=precondition_cycles,
            )
        )
        seen_profile_names.add(name)

    safety = SafetyConfig(
        max_goal_step_rad=math.radians(_require_number(safety_data, "max_goal_step_deg", "safety")),
        max_excursion_rad=math.radians(_require_number(safety_data, "max_excursion_deg", "safety")),
        position_limit_margin_rad=math.radians(_require_number(safety_data, "position_limit_margin_deg", "safety")),
        max_tracking_error_rad=math.radians(_require_number(safety_data, "max_tracking_error_deg", "safety")),
        max_current_a=_require_number(safety_data, "max_current_a", "safety"),
        max_pwm_fraction=_require_number(safety_data, "max_pwm_fraction", "safety"),
        max_temperature_c=_require_number(safety_data, "max_temperature_c", "safety"),
        min_voltage_v=_require_number(safety_data, "min_voltage_v", "safety"),
        max_voltage_v=_require_number(safety_data, "max_voltage_v", "safety"),
        max_consecutive_deadline_misses=_require_int(safety_data, "max_consecutive_deadline_misses", "safety"),
    )
    if not math.radians(0.05) <= safety.max_goal_step_rad <= math.radians(5.0):
        raise Real2SimConfigError("safety.max_goal_step_deg must be in [0.05, 5.0].")
    if not math.radians(1.0) <= safety.max_excursion_rad <= math.radians(30.0):
        raise Real2SimConfigError("safety.max_excursion_deg must be in [1.0, 30.0].")
    if not 0.0 <= safety.position_limit_margin_rad <= math.radians(15.0):
        raise Real2SimConfigError("safety.position_limit_margin_deg must be in [0, 15.0].")
    if not safety.max_goal_step_rad < safety.max_excursion_rad:
        raise Real2SimConfigError("max_goal_step_deg must be smaller than max_excursion_deg.")
    if safety.max_tracking_error_rad <= safety.max_goal_step_rad:
        raise Real2SimConfigError("max_tracking_error_deg must exceed max_goal_step_deg.")
    if not 0.1 <= safety.max_current_a <= 5.0:
        raise Real2SimConfigError("safety.max_current_a must be in [0.1, 5.0] A.")
    if not 0.05 <= safety.max_pwm_fraction <= 1.0:
        raise Real2SimConfigError("safety.max_pwm_fraction must be in [0.05, 1.0].")
    if not 20.0 <= safety.max_temperature_c <= 75.0:
        raise Real2SimConfigError("safety.max_temperature_c must be in [20, 75] C.")
    if not 8.0 <= safety.min_voltage_v < safety.max_voltage_v <= 16.8:
        raise Real2SimConfigError("Safety voltage range must lie inside [8.0, 16.8] V.")
    if not 1 <= safety.max_consecutive_deadline_misses <= 100:
        raise Real2SimConfigError("max_consecutive_deadline_misses must be in [1, 100].")
    if any(profile.amplitude_rad > safety.max_excursion_rad for profile in profiles):
        raise Real2SimConfigError("Every profile amplitude must be <= safety.max_excursion_deg.")

    reference_pose = _load_reference_pose(reference_pose_data, safety)

    identification = IdentificationConfig(
        max_delay_s=_require_number(identification_data, "max_delay_ms", "identification") / 1000.0,
        motion_threshold_rad=math.radians(
            _require_number(identification_data, "motion_threshold_deg", "identification")
        ),
        velocity_threshold_rad_s=math.radians(
            _require_number(identification_data, "velocity_threshold_deg_s", "identification")
        ),
        backlash_velocity_threshold_rad_s=math.radians(
            _require_number(identification_data, "backlash_velocity_threshold_deg_s", "identification")
        ),
        min_reversal_events=_require_int(identification_data, "min_reversal_events", "identification"),
        validation_fraction=_require_number(identification_data, "validation_fraction", "identification"),
        nominal_torque_per_amp_nm=_require_number(identification_data, "nominal_torque_per_amp_nm", "identification"),
        randomization_margin=_require_number(identification_data, "randomization_margin", "identification"),
    )
    if not 0.0 <= identification.max_delay_s <= 0.5:
        raise Real2SimConfigError("identification.max_delay_ms must be in [0, 500].")
    if not 0.0 < identification.backlash_velocity_threshold_rad_s <= identification.velocity_threshold_rad_s:
        raise Real2SimConfigError("backlash_velocity_threshold_deg_s must be positive and <= velocity_threshold_deg_s.")
    if not 0.01 <= identification.validation_fraction <= 0.5:
        raise Real2SimConfigError("identification.validation_fraction must be in [0.01, 0.5].")
    if identification.min_reversal_events < 2:
        raise Real2SimConfigError("identification.min_reversal_events must be >= 2.")
    if not 0.0 < identification.nominal_torque_per_amp_nm <= 5.0:
        raise Real2SimConfigError("nominal_torque_per_amp_nm must be in (0, 5].")
    if not 0.0 <= identification.randomization_margin <= 1.0:
        raise Real2SimConfigError("randomization_margin must be in [0, 1].")

    return ExperimentConfig(
        source_path=resolved,
        sample_hz=sample_hz,
        initial_settle_s=initial_settle_s,
        inter_profile_settle_s=inter_profile_settle_s,
        return_duration_s=return_duration_s,
        reference_pose=reference_pose,
        profiles=tuple(profiles),
        safety=safety,
        identification=identification,
    )
