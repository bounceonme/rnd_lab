"""MX-106(2.0) telemetry and command bus used only by real-to-sim tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .config import JointCalibration, MappingConfig, Real2SimConfigError

try:
    from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler
except ImportError:  # Keep config, dataset, and fitting usable without hardware dependencies.
    COMM_SUCCESS = 0
    GroupSyncRead = None
    GroupSyncWrite = None
    PacketHandler = None
    PortHandler = None


class DynamixelReal2SimError(RuntimeError):
    """Raised on a rejected command or a Dynamixel communication failure."""


@dataclass(frozen=True)
class MotorTelemetry:
    tick_ms: int
    moving: int
    moving_status: int
    pwm_fraction: float
    current_a: float
    velocity_rad_s: float
    position_rad: float
    velocity_trajectory_rad_s: float
    position_trajectory_rad: float
    voltage_v: float
    temperature_c: float
    raw_position: int


@dataclass(frozen=True)
class MotorRuntimeInfo:
    model_number: int
    firmware_version: int
    operating_mode: int
    drive_mode: int
    homing_offset_raw: int
    pwm_limit_raw: int
    current_limit_raw: int
    velocity_limit_raw: int
    position_d_gain: int
    position_i_gain: int
    position_p_gain: int
    feedforward_2nd_gain: int
    feedforward_1st_gain: int
    profile_acceleration: int
    profile_velocity: int


class Mx2ControlTable:
    MODEL_NUMBER = 0
    FIRMWARE_VERSION = 6
    DRIVE_MODE = 10
    OPERATING_MODE = 11
    HOMING_OFFSET = 20
    PWM_LIMIT = 36
    CURRENT_LIMIT = 38
    VELOCITY_LIMIT = 44
    TORQUE_ENABLE = 64
    HARDWARE_ERROR_STATUS = 70
    POSITION_D_GAIN = 80
    POSITION_I_GAIN = 82
    POSITION_P_GAIN = 84
    FEEDFORWARD_2ND_GAIN = 88
    FEEDFORWARD_1ST_GAIN = 90
    PROFILE_ACCELERATION = 108
    PROFILE_VELOCITY = 112
    GOAL_POSITION = 116
    TELEMETRY_START = 120
    TELEMETRY_LENGTH = 27
    REALTIME_TICK = 120
    MOVING = 122
    MOVING_STATUS = 123
    PRESENT_PWM = 124
    PRESENT_CURRENT = 126
    PRESENT_VELOCITY = 128
    PRESENT_POSITION = 132
    VELOCITY_TRAJECTORY = 136
    POSITION_TRAJECTORY = 140
    PRESENT_INPUT_VOLTAGE = 144
    PRESENT_TEMPERATURE = 146


def _signed(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def _little_endian_4(value: int) -> list[int]:
    return list(int(value).to_bytes(4, byteorder="little", signed=False))


class Mx2TelemetryBus:
    """Protocol 2.0 bus with fail-closed torque behavior and grouped I/O."""

    EXPECTED_MODEL_NUMBER = 321
    POSITION_CONTROL_MODE = 3
    CURRENT_UNIT_A = 0.00336
    VELOCITY_UNIT_RAD_S = 0.229 * 2.0 * math.pi / 60.0
    PWM_UNIT_FRACTION = 0.00113
    VOLTAGE_UNIT_V = 0.1

    def __init__(self, config: MappingConfig):
        if PortHandler is None or PacketHandler is None or GroupSyncRead is None or GroupSyncWrite is None:
            raise DynamixelReal2SimError(
                "dynamixel_sdk is not installed. Install the project dependencies before using hardware collection."
            )
        if config.protocol != 2.0 or config.control_table != "mx_2":
            raise Real2SimConfigError("Mx2TelemetryBus only supports Protocol 2.0 with control_table='mx_2'.")
        self.config = config
        self.port = PortHandler(config.device)
        self.packet = PacketHandler(config.protocol)
        self.connected = False
        self._port_open = False
        self.torque_enabled = {joint.name: False for joint in config.joints}
        self.last_goals_rad: dict[str, float] = {}
        self.runtime_info: dict[str, MotorRuntimeInfo] = {}
        self._telemetry_reader = GroupSyncRead(
            self.port,
            self.packet,
            Mx2ControlTable.TELEMETRY_START,
            Mx2ControlTable.TELEMETRY_LENGTH,
        )
        self._health_reader = GroupSyncRead(
            self.port,
            self.packet,
            Mx2ControlTable.HARDWARE_ERROR_STATUS,
            1,
        )
        self._goal_writer = GroupSyncWrite(self.port, self.packet, Mx2ControlTable.GOAL_POSITION, 4)
        for joint in config.joints:
            if not self._telemetry_reader.addParam(joint.motor_id):
                raise Real2SimConfigError(f"Could not register {joint.name} in telemetry GroupSyncRead.")
            if not self._health_reader.addParam(joint.motor_id):
                raise Real2SimConfigError(f"Could not register {joint.name} in health GroupSyncRead.")

    def __enter__(self) -> Mx2TelemetryBus:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _check(self, comm_result: int, device_error: int, context: str) -> None:
        if comm_result != COMM_SUCCESS:
            raise DynamixelReal2SimError(f"{context}: {self.packet.getTxRxResult(comm_result)}")
        if device_error:
            raise DynamixelReal2SimError(f"{context}: {self.packet.getRxPacketError(device_error)}")

    def _read(self, motor_id: int, address: int, size: int, context: str) -> int:
        if size == 1:
            value, comm_result, device_error = self.packet.read1ByteTxRx(self.port, motor_id, address)
        elif size == 2:
            value, comm_result, device_error = self.packet.read2ByteTxRx(self.port, motor_id, address)
        elif size == 4:
            value, comm_result, device_error = self.packet.read4ByteTxRx(self.port, motor_id, address)
        else:
            raise ValueError(f"Unsupported register size: {size}")
        self._check(comm_result, device_error, context)
        return value

    def _write(self, motor_id: int, address: int, size: int, value: int, context: str) -> None:
        if size == 1:
            comm_result, device_error = self.packet.write1ByteTxRx(self.port, motor_id, address, value)
        elif size == 2:
            comm_result, device_error = self.packet.write2ByteTxRx(self.port, motor_id, address, value)
        elif size == 4:
            comm_result, device_error = self.packet.write4ByteTxRx(self.port, motor_id, address, value)
        else:
            raise ValueError(f"Unsupported register size: {size}")
        self._check(comm_result, device_error, context)

    def _set_torque(self, joint: JointCalibration, enabled: bool) -> None:
        state = "ON" if enabled else "OFF"
        self._write(
            joint.motor_id,
            Mx2ControlTable.TORQUE_ENABLE,
            1,
            int(enabled),
            f"Torque {state} failed for {joint.name} (ID {joint.motor_id})",
        )
        readback = self._read(
            joint.motor_id,
            Mx2ControlTable.TORQUE_ENABLE,
            1,
            f"Torque {state} verification failed for {joint.name} (ID {joint.motor_id})",
        )
        if bool(readback) is not enabled:
            raise DynamixelReal2SimError(f"{joint.name} torque {state} readback was {readback}.")
        self.torque_enabled[joint.name] = enabled

    def _best_effort_torque_off(self) -> None:
        for joint in self.config.joints:
            try:
                self._write(
                    joint.motor_id,
                    Mx2ControlTable.TORQUE_ENABLE,
                    1,
                    0,
                    f"Emergency torque OFF failed for {joint.name}",
                )
                self.torque_enabled[joint.name] = False
            except Exception as error:
                print(f"[ERROR] Emergency torque OFF failed for {joint.name} (ID {joint.motor_id}): {error}")

    def open(self) -> None:
        if self.connected:
            return
        if not self.port.openPort():
            raise DynamixelReal2SimError(f"Could not open Dynamixel port {self.config.device}.")
        self._port_open = True
        try:
            if not self.port.setBaudRate(self.config.baudrate):
                raise DynamixelReal2SimError(f"Could not set baudrate {self.config.baudrate}.")
            for joint in self.config.joints:
                model, comm_result, device_error = self.packet.ping(self.port, joint.motor_id)
                self._check(comm_result, device_error, f"Ping failed for {joint.name} (ID {joint.motor_id})")
                self._set_torque(joint, False)
                if model != self.EXPECTED_MODEL_NUMBER:
                    raise DynamixelReal2SimError(
                        f"{joint.name} (ID {joint.motor_id}) is model {model}; expected MX-106(2.0) model 321."
                    )
                operating_mode = self._read(
                    joint.motor_id, Mx2ControlTable.OPERATING_MODE, 1, f"Operating Mode read failed for {joint.name}"
                )
                if operating_mode != self.POSITION_CONTROL_MODE:
                    raise DynamixelReal2SimError(
                        f"{joint.name} (ID {joint.motor_id}) Operating Mode is {operating_mode}; "
                        "set it to Position Control Mode (3) with torque OFF before collecting."
                    )
                self.runtime_info[joint.name] = MotorRuntimeInfo(
                    model_number=model,
                    firmware_version=self._read(
                        joint.motor_id,
                        Mx2ControlTable.FIRMWARE_VERSION,
                        1,
                        f"Firmware Version read failed for {joint.name}",
                    ),
                    operating_mode=operating_mode,
                    drive_mode=self._read(
                        joint.motor_id, Mx2ControlTable.DRIVE_MODE, 1, f"Drive Mode read failed for {joint.name}"
                    ),
                    homing_offset_raw=_signed(
                        self._read(
                            joint.motor_id,
                            Mx2ControlTable.HOMING_OFFSET,
                            4,
                            f"Homing Offset read failed for {joint.name}",
                        ),
                        32,
                    ),
                    pwm_limit_raw=self._read(
                        joint.motor_id,
                        Mx2ControlTable.PWM_LIMIT,
                        2,
                        f"PWM Limit read failed for {joint.name}",
                    ),
                    current_limit_raw=self._read(
                        joint.motor_id,
                        Mx2ControlTable.CURRENT_LIMIT,
                        2,
                        f"Current Limit read failed for {joint.name}",
                    ),
                    velocity_limit_raw=self._read(
                        joint.motor_id,
                        Mx2ControlTable.VELOCITY_LIMIT,
                        4,
                        f"Velocity Limit read failed for {joint.name}",
                    ),
                    position_d_gain=self._read(
                        joint.motor_id,
                        Mx2ControlTable.POSITION_D_GAIN,
                        2,
                        f"Position D Gain read failed for {joint.name}",
                    ),
                    position_i_gain=self._read(
                        joint.motor_id,
                        Mx2ControlTable.POSITION_I_GAIN,
                        2,
                        f"Position I Gain read failed for {joint.name}",
                    ),
                    position_p_gain=self._read(
                        joint.motor_id,
                        Mx2ControlTable.POSITION_P_GAIN,
                        2,
                        f"Position P Gain read failed for {joint.name}",
                    ),
                    feedforward_2nd_gain=self._read(
                        joint.motor_id,
                        Mx2ControlTable.FEEDFORWARD_2ND_GAIN,
                        2,
                        f"Feedforward 2nd Gain read failed for {joint.name}",
                    ),
                    feedforward_1st_gain=self._read(
                        joint.motor_id,
                        Mx2ControlTable.FEEDFORWARD_1ST_GAIN,
                        2,
                        f"Feedforward 1st Gain read failed for {joint.name}",
                    ),
                    profile_acceleration=self._read(
                        joint.motor_id,
                        Mx2ControlTable.PROFILE_ACCELERATION,
                        4,
                        f"Profile Acceleration read failed for {joint.name}",
                    ),
                    profile_velocity=self._read(
                        joint.motor_id,
                        Mx2ControlTable.PROFILE_VELOCITY,
                        4,
                        f"Profile Velocity read failed for {joint.name}",
                    ),
                )
                print(
                    f"[INFO] {joint.name}: ID={joint.motor_id}, model={model}, "
                    f"firmware={self.runtime_info[joint.name].firmware_version}, torque=OFF"
                )
            self.check_hardware_errors()
            self.connected = True
        except Exception:
            self._best_effort_torque_off()
            self.port.closePort()
            self._port_open = False
            raise

    def close(self) -> None:
        if self._port_open:
            self._best_effort_torque_off()
            self.port.closePort()
        self._port_open = False
        self.connected = False

    def emergency_torque_off(self) -> None:
        if self._port_open:
            self._best_effort_torque_off()

    def check_hardware_errors(self) -> None:
        result = self._health_reader.txRxPacket()
        if result != COMM_SUCCESS:
            raise DynamixelReal2SimError(f"Hardware Error GroupSyncRead failed: {self.packet.getTxRxResult(result)}")
        faults: list[str] = []
        for joint in self.config.joints:
            if not self._health_reader.isAvailable(joint.motor_id, Mx2ControlTable.HARDWARE_ERROR_STATUS, 1):
                faults.append(f"{joint.name}=unavailable")
                continue
            status = self._health_reader.getData(joint.motor_id, Mx2ControlTable.HARDWARE_ERROR_STATUS, 1)
            if status:
                faults.append(f"{joint.name}=0x{status:02x}")
        if faults:
            raise DynamixelReal2SimError("Dynamixel hardware fault: " + ", ".join(faults))

    def read_telemetry(self) -> dict[str, MotorTelemetry]:
        result = self._telemetry_reader.txRxPacket()
        if result != COMM_SUCCESS:
            raise DynamixelReal2SimError(f"Telemetry GroupSyncRead failed: {self.packet.getTxRxResult(result)}")

        telemetry: dict[str, MotorTelemetry] = {}
        for joint in self.config.joints:
            motor_id = joint.motor_id
            if not self._telemetry_reader.isAvailable(
                motor_id, Mx2ControlTable.TELEMETRY_START, Mx2ControlTable.TELEMETRY_LENGTH
            ):
                raise DynamixelReal2SimError(f"Telemetry unavailable for {joint.name} (ID {motor_id}).")
            get = self._telemetry_reader.getData
            raw_position = _signed(get(motor_id, Mx2ControlTable.PRESENT_POSITION, 4), 32)
            raw_position_trajectory = _signed(get(motor_id, Mx2ControlTable.POSITION_TRAJECTORY, 4), 32)
            raw_velocity = _signed(get(motor_id, Mx2ControlTable.PRESENT_VELOCITY, 4), 32)
            raw_velocity_trajectory = _signed(get(motor_id, Mx2ControlTable.VELOCITY_TRAJECTORY, 4), 32)
            raw_current = _signed(get(motor_id, Mx2ControlTable.PRESENT_CURRENT, 2), 16)
            raw_pwm = _signed(get(motor_id, Mx2ControlTable.PRESENT_PWM, 2), 16)
            telemetry[joint.name] = MotorTelemetry(
                tick_ms=get(motor_id, Mx2ControlTable.REALTIME_TICK, 2),
                moving=get(motor_id, Mx2ControlTable.MOVING, 1),
                moving_status=get(motor_id, Mx2ControlTable.MOVING_STATUS, 1),
                pwm_fraction=joint.direction * raw_pwm * self.PWM_UNIT_FRACTION,
                current_a=joint.direction * raw_current * self.CURRENT_UNIT_A,
                velocity_rad_s=joint.direction * raw_velocity * self.VELOCITY_UNIT_RAD_S,
                position_rad=joint.raw_to_radians(raw_position),
                velocity_trajectory_rad_s=(joint.direction * raw_velocity_trajectory * self.VELOCITY_UNIT_RAD_S),
                position_trajectory_rad=joint.raw_to_radians(raw_position_trajectory, strict=False),
                voltage_v=get(motor_id, Mx2ControlTable.PRESENT_INPUT_VOLTAGE, 2) * self.VOLTAGE_UNIT_V,
                temperature_c=float(get(motor_id, Mx2ControlTable.PRESENT_TEMPERATURE, 1)),
                raw_position=raw_position,
            )
        return telemetry

    def _write_raw_goals(self, raw_goals: dict[str, int]) -> None:
        self._goal_writer.clearParam()
        try:
            for name, raw_goal in raw_goals.items():
                joint = self.config.joints_by_name[name]
                if not self._goal_writer.addParam(joint.motor_id, _little_endian_4(raw_goal)):
                    raise DynamixelReal2SimError(f"Could not add {name} to Goal Position GroupSyncWrite.")
            result = self._goal_writer.txPacket()
            if result != COMM_SUCCESS:
                raise DynamixelReal2SimError(
                    f"Goal Position GroupSyncWrite failed: {self.packet.getTxRxResult(result)}"
                )
        finally:
            self._goal_writer.clearParam()

    def enable_torque_safely(self, joint_names: Iterable[str]) -> dict[str, float]:
        names = tuple(joint_names)
        if not names:
            raise DynamixelReal2SimError("At least one joint is required for torque enable.")
        telemetry = self.read_telemetry()
        raw_goals = {name: telemetry[name].raw_position for name in names}
        self._write_raw_goals(raw_goals)
        enabled: list[str] = []
        try:
            for name in names:
                joint = self.config.joints_by_name[name]
                self._set_torque(joint, True)
                enabled.append(name)
        except Exception:
            for name in enabled:
                try:
                    self._set_torque(self.config.joints_by_name[name], False)
                except Exception as error:
                    print(f"[ERROR] Torque-enable rollback failed for {name}: {error}")
            raise
        positions = {name: telemetry[name].position_rad for name in names}
        self.last_goals_rad.update(positions)
        return positions

    def disable_torque(self, joint_names: Iterable[str]) -> None:
        errors: list[str] = []
        for name in joint_names:
            try:
                self._set_torque(self.config.joints_by_name[name], False)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise DynamixelReal2SimError("; ".join(errors))

    def write_goal_positions(self, goals_rad: dict[str, float], max_step_rad: float) -> None:
        if not goals_rad:
            raise DynamixelReal2SimError("Goal Position command is empty.")
        raw_goals: dict[str, int] = {}
        for name, goal in goals_rad.items():
            if name not in self.torque_enabled:
                raise DynamixelReal2SimError(f"Unknown joint: {name}")
            if not self.torque_enabled[name]:
                raise DynamixelReal2SimError(f"Refusing Goal Position for {name} while torque is OFF.")
            if not math.isfinite(goal):
                raise DynamixelReal2SimError(f"Refusing non-finite Goal Position for {name}: {goal}")
            previous = self.last_goals_rad.get(name)
            if previous is None:
                raise DynamixelReal2SimError(f"No seeded Goal Position exists for {name}.")
            step = abs(goal - previous)
            if step > max_step_rad + 1.0e-12:
                raise DynamixelReal2SimError(
                    f"{name} Goal Position step {math.degrees(step):.3f} deg exceeds "
                    f"{math.degrees(max_step_rad):.3f} deg."
                )
            raw_goals[name] = self.config.joints_by_name[name].radians_to_raw(goal)
        self._write_raw_goals(raw_goals)
        self.last_goals_rad.update(goals_rad)
