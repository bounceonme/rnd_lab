"""Deterministic motor-bus substitute for pipeline and safety dry-runs."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .bus import MotorRuntimeInfo, MotorTelemetry
from .config import MappingConfig


class SyntheticMx2Bus:
    """Small encoder-domain actuator model exposing the hardware-bus API."""

    def __init__(self, config: MappingConfig, sample_hz: float):
        self.config = config
        self.sample_hz = sample_hz
        self.connected = False
        self.torque_enabled = {joint.name: False for joint in config.joints}
        self.last_goals_rad = {joint.name: 0.0 for joint in config.joints}
        self.runtime_info = {
            joint.name: MotorRuntimeInfo(
                model_number=321,
                firmware_version=99,
                operating_mode=3,
                drive_mode=0,
                homing_offset_raw=0,
                pwm_limit_raw=885,
                current_limit_raw=2047,
                velocity_limit_raw=210,
                position_d_gain=0,
                position_i_gain=0,
                position_p_gain=850,
                feedforward_2nd_gain=0,
                feedforward_1st_gain=0,
                profile_acceleration=0,
                profile_velocity=0,
            )
            for joint in config.joints
        }
        count = len(config.joints)
        self._position = np.zeros(count, dtype=np.float64)
        self._velocity = np.zeros(count, dtype=np.float64)
        self._play_position = np.zeros(count, dtype=np.float64)
        self._current = np.zeros(count, dtype=np.float64)
        self._tick_ms = 0
        self._delays = [2 + index % 2 for index in range(count)]
        self._queues = [deque([0.0] * (delay + 1), maxlen=delay + 1) for delay in self._delays]
        self._backlash = np.asarray([0.0048 + 0.00018 * (index % 4) for index in range(count)])
        self._coulomb = np.asarray([0.075 + 0.004 * (index % 3) for index in range(count)])
        self._viscous = np.asarray([0.045 + 0.003 * (index % 5) for index in range(count)])

    def __enter__(self) -> SyntheticMx2Bus:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> None:
        self.connected = True
        self.emergency_torque_off()

    def close(self) -> None:
        self.emergency_torque_off()
        self.connected = False

    def emergency_torque_off(self) -> None:
        for name in self.torque_enabled:
            self.torque_enabled[name] = False

    def check_hardware_errors(self) -> None:
        return

    def enable_torque_safely(self, joint_names) -> dict[str, float]:
        names = tuple(joint_names)
        positions = {}
        by_name = {joint.name: index for index, joint in enumerate(self.config.joints)}
        for name in names:
            index = by_name[name]
            position = float(self._position[index])
            self.last_goals_rad[name] = position
            self._queues[index] = deque([position] * (self._delays[index] + 1), maxlen=self._delays[index] + 1)
            self._play_position[index] = position
            self.torque_enabled[name] = True
            positions[name] = position
        return positions

    def disable_torque(self, joint_names) -> None:
        for name in joint_names:
            self.torque_enabled[name] = False

    def write_goal_positions(self, goals_rad: dict[str, float], max_step_rad: float) -> None:
        by_name = self.config.joints_by_name
        for name, value in goals_rad.items():
            if not self.torque_enabled[name]:
                raise RuntimeError(f"Synthetic torque is OFF for {name}.")
            if abs(value - self.last_goals_rad[name]) > max_step_rad + 1.0e-12:
                raise RuntimeError(f"Synthetic Goal Position step exceeded for {name}.")
            by_name[name].radians_to_raw(value)
        self.last_goals_rad.update(goals_rad)

    def advance(self, dt: float) -> None:
        for index, joint in enumerate(self.config.joints):
            if not self.torque_enabled[joint.name]:
                self._velocity[index] *= math.exp(-8.0 * dt)
                continue
            self._queues[index].append(self.last_goals_rad[joint.name])
            delayed_goal = self._queues[index][0]
            half_gap = 0.5 * self._backlash[index]
            if delayed_goal > self._play_position[index] + half_gap:
                self._play_position[index] = delayed_goal - half_gap
            elif delayed_goal < self._play_position[index] - half_gap:
                self._play_position[index] = delayed_goal + half_gap

            error = self._play_position[index] - self._position[index]
            direction = math.tanh(self._velocity[index] / 0.025)
            acceleration = 400.0 * error - 30.0 * self._velocity[index] - 0.08 * direction
            self._velocity[index] += acceleration * dt
            self._position[index] += self._velocity[index] * dt
            self._current[index] = (
                2.2 * error
                + self._coulomb[index] * math.tanh(self._velocity[index] / 0.02)
                + self._viscous[index] * self._velocity[index]
                + 0.0015 * acceleration
                + 0.025 * math.sin(self._position[index])
            )
        self._tick_ms = (self._tick_ms + max(1, round(dt * 1000.0))) % 32768

    def read_telemetry(self) -> dict[str, MotorTelemetry]:
        telemetry = {}
        for index, joint in enumerate(self.config.joints):
            raw_position = joint.radians_to_raw(float(self._position[index]))
            telemetry[joint.name] = MotorTelemetry(
                tick_ms=self._tick_ms,
                moving=int(abs(self._velocity[index]) > 0.01),
                moving_status=0,
                pwm_fraction=float(np.clip(self._current[index] / 2.5, -1.0, 1.0)),
                current_a=float(self._current[index]),
                velocity_rad_s=float(self._velocity[index]),
                position_rad=float(self._position[index]),
                velocity_trajectory_rad_s=0.0,
                position_trajectory_rad=self.last_goals_rad[joint.name],
                voltage_v=12.0,
                temperature_c=30.0,
                raw_position=raw_position,
            )
        return telemetry
