"""Safety-oriented Dynamixel SDK interface for the RND STEP joint tester."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, PacketHandler, PortHandler


class DynamixelError(RuntimeError):
    """Base error for Dynamixel configuration and communication failures."""


class DynamixelConfigError(DynamixelError):
    """Raised when the hardware mapping is incomplete or unsafe."""


class DynamixelCommunicationError(DynamixelError):
    """Raised when the Dynamixel bus or a servo reports an error."""


@dataclass(frozen=True)
class ControlTable:
    torque_enable: int
    goal_position: int
    goal_position_size: int
    present_position: int
    present_position_size: int
    expected_model_numbers: tuple[int, ...]

    @classmethod
    def for_name(cls, name: str) -> ControlTable:
        if name == "mx_legacy":
            return cls(
                torque_enable=24,
                goal_position=30,
                goal_position_size=2,
                present_position=36,
                present_position_size=2,
                expected_model_numbers=(320,),
            )
        if name == "mx_2":
            return cls(
                torque_enable=64,
                goal_position=116,
                goal_position_size=4,
                present_position=132,
                present_position_size=4,
                expected_model_numbers=(321,),
            )
        raise DynamixelConfigError(f"Unsupported control_table '{name}'. Use 'mx_legacy' or 'mx_2'.")


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

    def raw_to_radians(self, raw_position: int) -> float:
        if not self.min_raw <= raw_position <= self.max_raw:
            raise DynamixelCommunicationError(
                f"{self.name} (ID {self.motor_id}) returned {raw_position}, outside calibrated raw range "
                f"[{self.min_raw}, {self.max_raw}]."
            )
        return self.direction * (raw_position - self.zero_raw) * self.radians_per_tick

    def radians_to_raw(self, position: float) -> int:
        raw_position = round(self.zero_raw + self.direction * position / self.radians_per_tick)
        return max(self.min_raw, min(self.max_raw, raw_position))


@dataclass(frozen=True)
class DynamixelConfig:
    device: str
    baudrate: int
    protocol: float
    control_table_name: str
    poll_hz: float
    max_goal_step_rad: float
    joints: tuple[JointCalibration, ...]

    @property
    def control_table(self) -> ControlTable:
        return ControlTable.for_name(self.control_table_name)

    @property
    def joints_by_name(self) -> dict[str, JointCalibration]:
        return {joint.name: joint for joint in self.joints}


def _require_int(table: dict[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if not isinstance(value, int):
        raise DynamixelConfigError(f"{context}.{key} must be an integer, got {value!r}.")
    return value


def load_dynamixel_config(path: str | Path, expected_joint_names: list[str]) -> DynamixelConfig:
    """Load and validate a complete joint-to-servo calibration file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise DynamixelConfigError(f"Dynamixel configuration file does not exist: {config_path}")

    data = toml.load(config_path)
    bus = data.get("bus")
    joints = data.get("joints")
    if not isinstance(bus, dict) or not isinstance(joints, list):
        raise DynamixelConfigError("Configuration requires one [bus] table and repeated [[joints]] tables.")
    if bus.get("configured") is not True:
        raise DynamixelConfigError(
            f"Hardware mapping is locked in {config_path}. Set bus.configured=true only after filling every ID, "
            "zero_raw, direction, and raw limit."
        )

    device = bus.get("device")
    baudrate = bus.get("baudrate")
    protocol = bus.get("protocol")
    control_table_name = bus.get("control_table")
    poll_hz = bus.get("poll_hz", 10.0)
    max_goal_step_deg = bus.get("max_goal_step_deg", 5.0)
    ticks_per_revolution = bus.get("ticks_per_revolution", 4096)
    if not isinstance(device, str) or not device:
        raise DynamixelConfigError("bus.device must be a non-empty serial-device path.")
    if not isinstance(baudrate, int) or baudrate <= 0:
        raise DynamixelConfigError("bus.baudrate must be a positive integer.")
    if protocol not in (1, 1.0, 2, 2.0):
        raise DynamixelConfigError("bus.protocol must be 1.0 or 2.0.")
    if not isinstance(control_table_name, str):
        raise DynamixelConfigError("bus.control_table must be 'mx_legacy' or 'mx_2'.")
    ControlTable.for_name(control_table_name)
    if control_table_name == "mx_legacy" and float(protocol) != 1.0:
        raise DynamixelConfigError("The legacy MX control table only supports Dynamixel Protocol 1.0.")
    if not isinstance(poll_hz, (int, float)) or not 0.5 <= float(poll_hz) <= 30.0:
        raise DynamixelConfigError("bus.poll_hz must be between 0.5 and 30.0 Hz.")
    if not isinstance(max_goal_step_deg, (int, float)) or not 0.1 <= float(max_goal_step_deg) <= 20.0:
        raise DynamixelConfigError("bus.max_goal_step_deg must be between 0.1 and 20.0 degrees.")
    if not isinstance(ticks_per_revolution, int) or ticks_per_revolution <= 0:
        raise DynamixelConfigError("bus.ticks_per_revolution must be a positive integer.")

    calibrations = []
    for index, joint_data in enumerate(joints):
        context = f"joints[{index}]"
        if not isinstance(joint_data, dict):
            raise DynamixelConfigError(f"{context} must be a table.")
        name = joint_data.get("name")
        if not isinstance(name, str) or not name:
            raise DynamixelConfigError(f"{context}.name must be a non-empty string.")
        motor_id = _require_int(joint_data, "id", context)
        zero_raw = _require_int(joint_data, "zero_raw", context)
        direction = _require_int(joint_data, "direction", context)
        min_raw = _require_int(joint_data, "min_raw", context)
        max_raw = _require_int(joint_data, "max_raw", context)

        if not 0 <= motor_id <= 252:
            raise DynamixelConfigError(f"{context}.id must be in [0, 252].")
        if direction not in (-1, 1):
            raise DynamixelConfigError(f"{context}.direction must be -1 or 1.")
        if not min_raw < max_raw or not min_raw <= zero_raw <= max_raw:
            raise DynamixelConfigError(f"{context} requires min_raw < max_raw and zero_raw inside that range.")

        calibrations.append(
            JointCalibration(
                name=name,
                motor_id=motor_id,
                zero_raw=zero_raw,
                direction=direction,
                min_raw=min_raw,
                max_raw=max_raw,
                ticks_per_revolution=ticks_per_revolution,
            )
        )

    configured_names = [joint.name for joint in calibrations]
    configured_ids = [joint.motor_id for joint in calibrations]
    if len(configured_names) != len(set(configured_names)):
        raise DynamixelConfigError("Joint names must be unique.")
    if len(configured_ids) != len(set(configured_ids)):
        raise DynamixelConfigError("Dynamixel IDs must be unique.")
    if set(configured_names) != set(expected_joint_names):
        missing = sorted(set(expected_joint_names) - set(configured_names))
        extra = sorted(set(configured_names) - set(expected_joint_names))
        raise DynamixelConfigError(f"Joint mapping mismatch. Missing={missing}, extra={extra}.")

    by_name = {joint.name: joint for joint in calibrations}
    ordered_joints = tuple(by_name[name] for name in expected_joint_names)
    return DynamixelConfig(
        device=device,
        baudrate=baudrate,
        protocol=float(protocol),
        control_table_name=control_table_name,
        poll_hz=float(poll_hz),
        max_goal_step_rad=math.radians(float(max_goal_step_deg)),
        joints=ordered_joints,
    )


class DynamixelBus:
    """Synchronous Dynamixel bus with torque-safe enable and shutdown behavior."""

    def __init__(self, config: DynamixelConfig):
        self.config = config
        self.control_table = config.control_table
        self.port = PortHandler(config.device)
        self.packet = PacketHandler(config.protocol)
        self.connected = False
        self.torque_enabled = {joint.name: False for joint in config.joints}
        self.last_raw_positions: dict[str, int] = {}
        self._position_sync_reader = self._create_position_sync_reader(config.joints) if config.protocol == 2.0 else None

    def _create_position_sync_reader(self, joints: tuple[JointCalibration, ...]) -> GroupSyncRead:
        reader = GroupSyncRead(
            self.port,
            self.packet,
            self.control_table.present_position,
            self.control_table.present_position_size,
        )
        for joint in joints:
            if not reader.addParam(joint.motor_id):
                raise DynamixelConfigError(f"Failed to add {joint.name} (ID {joint.motor_id}) to GroupSyncRead.")
        return reader

    def _check_result(self, comm_result: int, device_error: int, context: str):
        if comm_result != COMM_SUCCESS:
            raise DynamixelCommunicationError(f"{context}: {self.packet.getTxRxResult(comm_result)}")
        if device_error != 0:
            raise DynamixelCommunicationError(f"{context}: {self.packet.getRxPacketError(device_error)}")

    def _write_value(self, motor_id: int, address: int, size: int, value: int, context: str):
        if size == 1:
            comm_result, device_error = self.packet.write1ByteTxRx(self.port, motor_id, address, value)
        elif size == 2:
            comm_result, device_error = self.packet.write2ByteTxRx(self.port, motor_id, address, value)
        elif size == 4:
            comm_result, device_error = self.packet.write4ByteTxRx(self.port, motor_id, address, value)
        else:
            raise ValueError(f"Unsupported Dynamixel register size: {size}")
        self._check_result(comm_result, device_error, context)

    def _read_value(self, motor_id: int, address: int, size: int, context: str) -> int:
        if size == 1:
            value, comm_result, device_error = self.packet.read1ByteTxRx(self.port, motor_id, address)
        elif size == 2:
            value, comm_result, device_error = self.packet.read2ByteTxRx(self.port, motor_id, address)
        elif size == 4:
            value, comm_result, device_error = self.packet.read4ByteTxRx(self.port, motor_id, address)
        else:
            raise ValueError(f"Unsupported Dynamixel register size: {size}")
        self._check_result(comm_result, device_error, context)
        if size == 4 and value >= 2**31:
            value -= 2**32
        return value

    def _set_torque_raw(self, joint: JointCalibration, enabled: bool):
        state = "ON" if enabled else "OFF"
        self._write_value(
            joint.motor_id,
            self.control_table.torque_enable,
            1,
            int(enabled),
            f"Torque {state} failed for {joint.name} (ID {joint.motor_id})",
        )
        self.torque_enabled[joint.name] = enabled

    def _best_effort_disable_all(self):
        for joint in self.config.joints:
            try:
                self._set_torque_raw(joint, False)
            except Exception as error:
                print(f"[ERROR]: Emergency torque OFF failed for {joint.name} (ID {joint.motor_id}): {error}")

    def open(self):
        if self.connected:
            return
        if not self.port.openPort():
            raise DynamixelCommunicationError(f"Failed to open Dynamixel port: {self.config.device}")
        try:
            if not self.port.setBaudRate(self.config.baudrate):
                raise DynamixelCommunicationError(f"Failed to set Dynamixel baudrate: {self.config.baudrate}")

            for joint in self.config.joints:
                model_number, comm_result, device_error = self.packet.ping(self.port, joint.motor_id)
                self._check_result(comm_result, device_error, f"Ping failed for {joint.name} (ID {joint.motor_id})")
                self._set_torque_raw(joint, False)
                if model_number not in self.control_table.expected_model_numbers:
                    raise DynamixelCommunicationError(
                        f"{joint.name} (ID {joint.motor_id}) reported model {model_number}; expected one of "
                        f"{self.control_table.expected_model_numbers} for {self.config.control_table_name}."
                    )
                print(f"[INFO]: Connected {joint.name}: ID={joint.motor_id}, model={model_number}, torque=OFF")
            self.connected = True
        except Exception:
            self._best_effort_disable_all()
            self.port.closePort()
            raise

    def close(self):
        if self.port.is_open:
            self._best_effort_disable_all()
            self.port.closePort()
        self.connected = False

    def read_raw_position(self, joint_name: str) -> int:
        joint = self.config.joints_by_name[joint_name]
        raw_position = self._read_value(
            joint.motor_id,
            self.control_table.present_position,
            self.control_table.present_position_size,
            f"Present Position read failed for {joint.name} (ID {joint.motor_id})",
        )
        self.last_raw_positions[joint_name] = raw_position
        return raw_position

    def read_raw_positions(self, joint_names: list[str] | None = None) -> dict[str, int]:
        names = joint_names or [joint.name for joint in self.config.joints]
        if self.config.protocol != 2.0 or len(names) == 1:
            return {name: self.read_raw_position(name) for name in names}

        configured_names = [joint.name for joint in self.config.joints]
        if names == configured_names:
            reader = self._position_sync_reader
        else:
            joints = tuple(self.config.joints_by_name[name] for name in names)
            reader = self._create_position_sync_reader(joints)
        if reader is None:
            raise DynamixelCommunicationError("Protocol 2.0 GroupSyncRead is not initialized.")

        comm_result = reader.txRxPacket()
        if comm_result != COMM_SUCCESS:
            raise DynamixelCommunicationError(
                f"Present Position GroupSyncRead failed: {self.packet.getTxRxResult(comm_result)}"
            )

        raw_positions = {}
        address = self.control_table.present_position
        size = self.control_table.present_position_size
        for name in names:
            joint = self.config.joints_by_name[name]
            if not reader.isAvailable(joint.motor_id, address, size):
                raise DynamixelCommunicationError(
                    f"Present Position unavailable for {joint.name} (ID {joint.motor_id}) after GroupSyncRead."
                )
            raw_position = reader.getData(joint.motor_id, address, size)
            if size == 4 and raw_position >= 2**31:
                raw_position -= 2**32
            raw_positions[name] = raw_position

        self.last_raw_positions.update(raw_positions)
        return raw_positions

    def read_positions(self, joint_names: list[str] | None = None) -> dict[str, float]:
        names = joint_names or [joint.name for joint in self.config.joints]
        raw_positions = self.read_raw_positions(names)
        return {
            name: self.config.joints_by_name[name].raw_to_radians(raw_position)
            for name, raw_position in raw_positions.items()
        }

    def write_goal_position(self, joint_name: str, position: float, require_torque: bool = True) -> int:
        if require_torque and not self.torque_enabled[joint_name]:
            raise DynamixelCommunicationError(f"Refusing Goal Position for {joint_name} while torque is OFF.")
        joint = self.config.joints_by_name[joint_name]
        raw_position = joint.radians_to_raw(position)
        self._write_value(
            joint.motor_id,
            self.control_table.goal_position,
            self.control_table.goal_position_size,
            raw_position,
            f"Goal Position write failed for {joint.name} (ID {joint.motor_id})",
        )
        return raw_position

    def enable_torque(self, joint_names: list[str]) -> dict[str, float]:
        """Capture positions, seed goals, then enable torque for all requested joints."""
        raw_positions = self.read_raw_positions(joint_names)
        positions = {
            name: self.config.joints_by_name[name].raw_to_radians(raw_position)
            for name, raw_position in raw_positions.items()
        }
        for name, raw_position in raw_positions.items():
            joint = self.config.joints_by_name[name]
            self._write_value(
                joint.motor_id,
                self.control_table.goal_position,
                self.control_table.goal_position_size,
                raw_position,
                f"Safe Goal Position seed failed for {joint.name} (ID {joint.motor_id})",
            )

        enabled_names = []
        try:
            for name in joint_names:
                self._set_torque_raw(self.config.joints_by_name[name], True)
                enabled_names.append(name)
        except Exception:
            for name in enabled_names:
                try:
                    self._set_torque_raw(self.config.joints_by_name[name], False)
                except Exception as error:
                    print(f"[ERROR]: Rollback torque OFF failed for {name}: {error}")
            raise

        return positions

    def disable_torque(self, joint_names: list[str]):
        errors = []
        for name in joint_names:
            try:
                self._set_torque_raw(self.config.joints_by_name[name], False)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise DynamixelCommunicationError("; ".join(errors))

    def emergency_disable_all(self):
        self._best_effort_disable_all()
