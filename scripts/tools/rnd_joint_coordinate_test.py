# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Mirror and command a physical RND STEP robot from Omniverse.

The physical backend uses the official Dynamixel SDK. Every hardware torque-on
operation captures Present Position, seeds Goal Position with the same value,
and only then enables torque.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


_DEFAULT_HARDWARE_CONFIG = Path(__file__).resolve().parent / "config" / "rnd_dynamixel.toml"

parser = argparse.ArgumentParser(description="Compare physical and URDF joint coordinates for RND STEP.")
parser.add_argument(
    "--hardware_config",
    type=str,
    default=str(_DEFAULT_HARDWARE_CONFIG),
    help="Dynamixel bus and joint calibration TOML file.",
)
parser.add_argument(
    "--sim_only",
    action="store_true",
    default=False,
    help="Run without opening a Dynamixel port. Intended for UI and CI checks.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after this many simulation steps. Zero runs until the app closes.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import asyncio
import math
import time
from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_lab.assets.rnd import STEP_CFG
from robot_lab.hardware import DynamixelBus, DynamixelError, load_dynamixel_config

import omni.kit.app

if not args_cli.headless:
    import omni.ui as ui
else:
    ui = None


_WINDOW_TITLE = "RND STEP Hardware Joint Coordinates"
_ROOT_HEIGHT = 0.3905
_ROBOT_PRIM_PATH = "/World/Robot"
_RENDER_HZ = 30.0
_UI_REFRESH_HZ = 20.0
_LEFT_LEG_BODY_NAMES = (
    "L_Leg_hip",
    "L_Leg_hip_double",
    "L_Leg_thighs",
    "L_Leg_calf",
    "L_Leg_ankle",
    "L_Leg_foot",
)
_RIGHT_LEG_BODY_NAMES = (
    "R_Leg_hip",
    "R_Leg_hip_double",
    "R_Leg_thighs",
    "R_Leg_calf",
    "R_Leg_ankle",
    "R_Leg_foot",
)


class CommandRejectedError(DynamixelError):
    """Raised when an unsafe UI command is rejected before hardware I/O."""


def _apply_coordinate_debug_materials():
    material_groups = (
        ("Body", (0.48, 0.52, 0.58), ("base_link",)),
        ("LeftLeg", (0.08, 0.42, 0.78), _LEFT_LEG_BODY_NAMES),
        ("RightLeg", (0.82, 0.28, 0.08), _RIGHT_LEG_BODY_NAMES),
    )
    for material_name, color, body_names in material_groups:
        material_path = f"/World/Looks/RNDCoordinate{material_name}"
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.72)
        material_cfg.func(material_path, material_cfg)
        for body_name in body_names:
            sim_utils.bind_visual_material(f"{_ROBOT_PRIM_PATH}/{body_name}", material_path)


def _design_scene() -> Articulation:
    ambient_light_cfg = sim_utils.DomeLightCfg(intensity=450.0, color=(0.18, 0.22, 0.28))
    ambient_light_cfg.func("/World/AmbientLight", ambient_light_cfg)
    key_light_cfg = sim_utils.DistantLightCfg(intensity=1600.0, color=(0.92, 0.92, 0.90), angle=4.0)
    key_light_cfg.func("/World/KeyLight", key_light_cfg)

    floor_cfg = sim_utils.CuboidCfg(
        size=(4.0, 4.0, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.055, 0.065, 0.08),
            roughness=0.85,
        ),
    )
    floor_cfg.func("/World/Floor", floor_cfg, translation=(0.0, 0.0, -0.005))

    robot_cfg = STEP_CFG.copy()
    robot_cfg.prim_path = _ROBOT_PRIM_PATH
    robot_cfg.init_state.pos = (0.0, 0.0, _ROOT_HEIGHT)
    robot_cfg.soft_joint_pos_limit_factor = 1.0
    robot_cfg.spawn.fix_base = True
    robot_cfg.spawn.activate_contact_sensors = False
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False

    # The simulated robot is a kinematic mirror. Physical torque is controlled by Dynamixel.
    for actuator_cfg in robot_cfg.actuators.values():
        actuator_cfg.stiffness = 0.0
        actuator_cfg.damping = 0.0

    robot = Articulation(robot_cfg)
    _apply_coordinate_debug_materials()
    return robot


class RobotMirror:
    """Write measured hardware positions directly into the fixed-base URDF."""

    def __init__(self, robot: Articulation):
        self.robot = robot
        self.joint_names = list(robot.joint_names)
        self.joint_name_to_id = {name: index for index, name in enumerate(self.joint_names)}
        self.num_joints = robot.num_joints
        self.joint_limits = robot.data.joint_pos_limits[0].detach().cpu().clone()
        self.current_pos = robot.data.joint_pos[0].detach().cpu().clone()
        self.target_pos = self.current_pos.clone()

        zero_gains = torch.zeros((1, self.num_joints), device=robot.device)
        robot.write_joint_stiffness_to_sim(zero_gains)
        robot.write_joint_damping_to_sim(zero_gains)
        self.apply_positions({
            name: float(robot.data.default_joint_pos[0, joint_id].item())
            for joint_id, name in enumerate(self.joint_names)
        })

    def clamp_position(self, joint_name: str, position: float) -> float:
        joint_id = self.joint_name_to_id[joint_name]
        lower = float(self.joint_limits[joint_id, 0].item())
        upper = float(self.joint_limits[joint_id, 1].item())
        return max(lower, min(upper, float(position)))

    def validate_position(self, joint_name: str, position: float):
        joint_id = self.joint_name_to_id[joint_name]
        lower = float(self.joint_limits[joint_id, 0].item())
        upper = float(self.joint_limits[joint_id, 1].item())
        if not lower <= position <= upper:
            raise DynamixelError(
                f"{joint_name} converted position {position:+.5f} rad exceeds URDF limit [{lower:+.5f}, {upper:+.5f}]. "
                "Check zero_raw and direction before enabling torque."
            )

    def apply_positions(self, positions: dict[str, float]):
        for joint_name, position in positions.items():
            self.validate_position(joint_name, position)
            joint_id = self.joint_name_to_id[joint_name]
            self.current_pos[joint_id] = position

        joint_pos = self.current_pos.to(device=self.robot.device).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)
        self.robot.set_joint_velocity_target(joint_vel)
        self.robot.set_joint_effort_target(joint_vel)
        self.robot.write_data_to_sim()

    def set_target(self, joint_name: str, position: float):
        joint_id = self.joint_name_to_id[joint_name]
        self.target_pos[joint_id] = position

    def set_targets(self, positions: dict[str, float]):
        for joint_name, position in positions.items():
            self.set_target(joint_name, position)

    def preview_position(self, joint_name: str, position: float):
        position = self.clamp_position(joint_name, position)
        self.set_target(joint_name, position)
        self.apply_positions({joint_name: position})


class CoordinateTestController:
    """Coordinate hardware safety operations and the Omniverse mirror."""

    def __init__(self, mirror: RobotMirror, bus: DynamixelBus | None):
        self.mirror = mirror
        self.bus = bus
        self.fault = None
        self.torque_enabled = {name: False for name in mirror.joint_names}

        if bus is not None:
            positions = bus.read_positions()
            mirror.apply_positions(positions)
            mirror.set_targets(positions)

    @property
    def backend_status(self) -> str:
        if self.bus is None:
            return "Simulation-only backend"
        return f"Dynamixel {self.bus.config.device} @ {self.bus.config.baudrate} bps"

    @property
    def poll_hz(self) -> float:
        return self.bus.config.poll_hz if self.bus is not None else 0.0

    def _require_healthy(self):
        if self.fault is not None:
            raise DynamixelError(f"Hardware commands are locked after fault: {self.fault}")

    def enable_torque(self, joint_names: list[str]):
        self._require_healthy()
        if self.bus is None:
            positions = {
                name: float(self.mirror.current_pos[self.mirror.joint_name_to_id[name]].item())
                for name in joint_names
            }
        else:
            positions = self.bus.enable_torque(joint_names)

        self.mirror.apply_positions(positions)
        self.mirror.set_targets(positions)
        for name in joint_names:
            self.torque_enabled[name] = True

    def disable_torque(self, joint_names: list[str]):
        if self.bus is not None:
            self.bus.disable_torque(joint_names)
            positions = self.bus.read_positions(joint_names)
            self.mirror.apply_positions(positions)
            self.mirror.set_targets(positions)
        for name in joint_names:
            self.torque_enabled[name] = False

    def set_goal_position(self, joint_name: str, position: float) -> float:
        self._require_healthy()
        if not self.torque_enabled[joint_name]:
            raise CommandRejectedError(f"Refusing to move {joint_name}: physical torque is OFF.")

        position = self.mirror.clamp_position(joint_name, position)
        if self.bus is not None:
            joint_id = self.mirror.joint_name_to_id[joint_name]
            measured_position = float(self.mirror.current_pos[joint_id].item())
            position_delta = abs(position - measured_position)
            if position_delta > self.bus.config.max_goal_step_rad:
                raise CommandRejectedError(
                    f"Refusing {joint_name} goal step of {math.degrees(position_delta):.2f} deg; configured maximum is "
                    f"{math.degrees(self.bus.config.max_goal_step_rad):.2f} deg. Move the slider in smaller steps."
                )
        if self.bus is None:
            quantized_position = position
            self.mirror.preview_position(joint_name, position)
        else:
            raw_position = self.bus.write_goal_position(joint_name, position)
            calibration = self.bus.config.joints_by_name[joint_name]
            quantized_position = calibration.raw_to_radians(raw_position)
        self.mirror.set_target(joint_name, quantized_position)
        return quantized_position

    def poll_hardware(self):
        if self.bus is None or self.fault is not None:
            return
        self.mirror.apply_positions(self.bus.read_positions())

    def enter_fault(self, error: Exception):
        if self.fault is not None:
            return
        self.fault = str(error)
        if self.bus is not None:
            self.bus.emergency_disable_all()
        self.torque_enabled = {name: False for name in self.mirror.joint_names}
        print(f"[ERROR]: Hardware fault; all torque commands locked: {self.fault}")

    def close(self):
        if self.bus is not None:
            self.bus.close()


class JointCoordinateWindow:
    """Omniverse UI for physical torque, goal position, and measured position."""

    def __init__(self, controller: CoordinateTestController):
        if ui is None:
            raise RuntimeError("Omniverse UI is unavailable in headless mode.")
        self.controller = controller
        self._commands: deque[tuple[str, str | None, float | bool | None]] = deque()
        self._joint_widgets: list[dict] = []
        self._suppress_model_callbacks = False
        self._window = None
        self._global_status_label = None
        self._error_label = None
        self._build_window()
        self.sync_target_models()
        self.refresh()
        asyncio.ensure_future(self._dock_window())

    def _build_window(self):
        self._window = ui.Window(
            _WINDOW_TITLE,
            width=630,
            height=780,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=6, height=ui.Fraction(1), width=ui.Fraction(1)):
                with ui.HStack(height=28, spacing=6):
                    ui.Label("RND STEP Hardware Joint Coordinates", width=ui.Fraction(1), style={"font_size": 16})
                    self._global_status_label = ui.Label("ALL OFF", width=90, alignment=ui.Alignment.RIGHT_CENTER)

                with ui.HStack(height=30, spacing=6):
                    ui.Button("ALL ON", clicked_fn=lambda: self._queue("all_torque", None, True))
                    ui.Button("ALL OFF", clicked_fn=lambda: self._queue("all_torque", None, False))

                ui.Label(
                    self.controller.backend_status,
                    height=18,
                    style={"font_size": 11, "color": 0xFFB8B8B8},
                )
                self._error_label = ui.Label(
                    "", height=34, word_wrap=True, style={"font_size": 11, "color": 0xFF6E6EFF}
                )
                ui.Separator(height=4)

                with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                ):
                    with ui.VStack(spacing=4, height=0):
                        for joint_id, joint_name in enumerate(self.controller.mirror.joint_names):
                            self._build_joint_row(joint_id, joint_name)

    def _build_joint_row(self, joint_id: int, joint_name: str):
        mirror = self.controller.mirror
        lower_deg = math.degrees(float(mirror.joint_limits[joint_id, 0].item()))
        upper_deg = math.degrees(float(mirror.joint_limits[joint_id, 1].item()))
        target_deg = math.degrees(float(mirror.target_pos[joint_id].item()))

        with ui.VStack(height=62, spacing=2):
            with ui.HStack(height=25, spacing=5):
                ui.Label(joint_name, width=175, style={"font_size": 12})
                status_label = ui.Label("OFF", width=42, alignment=ui.Alignment.CENTER)
                actual_label = ui.Label("", width=220, alignment=ui.Alignment.RIGHT_CENTER)
                ui.Button("ON", width=54, clicked_fn=lambda name=joint_name: self._queue("torque", name, True))
                ui.Button("OFF", width=54, clicked_fn=lambda name=joint_name: self._queue("torque", name, False))

            with ui.HStack(height=27, spacing=5):
                target_field = ui.FloatField(width=74)
                target_field.model.set_value(target_deg)
                target_field.model.add_value_changed_fn(
                    lambda model, name=joint_name: self._target_changed(name, model.as_float)
                )
                ui.Label("deg", width=28, alignment=ui.Alignment.LEFT_CENTER)
                slider = ui.FloatSlider(
                    model=target_field.model,
                    min=lower_deg,
                    max=upper_deg,
                    step=0.1,
                    width=ui.Fraction(1),
                )
            ui.Separator(height=2)

        self._joint_widgets.append({
            "name": joint_name,
            "model": target_field.model,
            "field": target_field,
            "slider": slider,
            "status_label": status_label,
            "actual_label": actual_label,
        })

    async def _dock_window(self):
        for _ in range(10):
            if ui.Workspace.get_window(_WINDOW_TITLE):
                break
            await omni.kit.app.get_app().next_update_async()

        window = ui.Workspace.get_window(_WINDOW_TITLE)
        viewport = ui.Workspace.get_window("Viewport")
        if window and viewport:
            window.dock_in(viewport, ui.DockPosition.RIGHT, 0.43)
            window.focus()

    def _queue(self, command: str, joint_name: str | None, value: float | bool | None):
        if (
            command == "target"
            and self._commands
            and self._commands[-1][0] == "target"
            and self._commands[-1][1] == joint_name
        ):
            self._commands[-1] = (command, joint_name, value)
            return
        self._commands.append((command, joint_name, value))

    def _target_changed(self, joint_name: str, value_deg: float):
        if not self._suppress_model_callbacks:
            self._queue("target", joint_name, math.radians(float(value_deg)))

    def _handle_error(self, error: Exception):
        self.controller.enter_fault(error)
        self.show_error(self.controller.fault or str(error))

    def show_error(self, message: str):
        self._error_label.text = message

    def process_commands(self):
        if not self._commands:
            return

        sync_targets = False
        command_error = False
        while self._commands:
            command, joint_name, value = self._commands.popleft()
            try:
                if command == "target":
                    assert joint_name is not None and isinstance(value, float)
                    applied = self.controller.set_goal_position(joint_name, value)
                    self._set_target_model(joint_name, applied)
                elif command == "torque":
                    assert joint_name is not None and isinstance(value, bool)
                    if value:
                        self.controller.enable_torque([joint_name])
                    else:
                        self.controller.disable_torque([joint_name])
                    sync_targets = True
                elif command == "all_torque":
                    assert isinstance(value, bool)
                    names = self.controller.mirror.joint_names
                    if value:
                        self.controller.enable_torque(names)
                    else:
                        self.controller.disable_torque(names)
                    sync_targets = True
                else:
                    raise ValueError(f"Unknown UI command: {command}")
            except CommandRejectedError as error:
                self.show_error(str(error))
                sync_targets = True
                command_error = True
                self._commands.clear()
                break
            except DynamixelError as error:
                if self.controller.fault is None:
                    self._handle_error(error)
                else:
                    self.show_error(self.controller.fault)
                command_error = True
                self._commands.clear()
                break

        if sync_targets:
            self.sync_target_models()
        if not command_error:
            self.show_error("")
        self.refresh()

    def _set_target_model(self, joint_name: str, value_rad: float):
        joint_id = self.controller.mirror.joint_name_to_id[joint_name]
        self._suppress_model_callbacks = True
        try:
            self._joint_widgets[joint_id]["model"].set_value(math.degrees(value_rad))
        finally:
            self._suppress_model_callbacks = False

    def sync_target_models(self):
        mirror = self.controller.mirror
        for joint_id, joint_name in enumerate(mirror.joint_names):
            self._set_target_model(joint_name, float(mirror.target_pos[joint_id].item()))

    def refresh(self):
        mirror = self.controller.mirror
        joint_pos = mirror.current_pos.tolist()
        enabled_count = sum(self.controller.torque_enabled.values())
        if self.controller.fault is not None:
            self._global_status_label.text = "FAULT"
        elif enabled_count == 0:
            self._global_status_label.text = "ALL OFF"
        elif enabled_count == mirror.num_joints:
            self._global_status_label.text = "ALL ON"
        else:
            self._global_status_label.text = f"{enabled_count}/{mirror.num_joints} ON"

        for joint_id, value_rad in enumerate(joint_pos):
            widgets = self._joint_widgets[joint_id]
            joint_name = widgets["name"]
            enabled = self.controller.torque_enabled[joint_name] and self.controller.fault is None
            widgets["status_label"].text = "ON" if enabled else "OFF"
            if self.controller.bus is not None and joint_name in self.controller.bus.last_raw_positions:
                raw_position = self.controller.bus.last_raw_positions[joint_name]
                widgets[
                    "actual_label"
                ].text = f"q {math.degrees(value_rad):+6.1f} deg | {value_rad:+.3f} rad | raw {raw_position}"
            else:
                widgets["actual_label"].text = f"q {math.degrees(value_rad):+7.2f} deg | {value_rad:+.5f} rad"
            widgets["field"].enabled = enabled
            widgets["slider"].enabled = enabled

    def close(self):
        if self._window is not None:
            self._window.visible = False
            self._window = None


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.15, 1.25, 0.78), target=(0.0, 0.0, 0.34))

    robot = _design_scene()
    sim.reset()
    robot.update(sim.get_physics_dt())
    mirror = RobotMirror(robot)

    bus = None
    controller = None
    window = None
    try:
        if not args_cli.sim_only:
            hardware_config = load_dynamixel_config(args_cli.hardware_config, mirror.joint_names)
            bus = DynamixelBus(hardware_config)
            bus.open()

        controller = CoordinateTestController(mirror, bus)
        window = None if args_cli.headless else JointCoordinateWindow(controller)
        print(f"[INFO]: RND STEP coordinate tester ready: {controller.backend_status}")
        print("[INFO]: Initial physical torque state: ALL OFF")

        poll_period = 1.0 / controller.poll_hz if controller.poll_hz > 0.0 else None
        next_poll_time = time.monotonic()
        ui_refresh_period = 1.0 / _UI_REFRESH_HZ
        next_ui_refresh_time = time.monotonic()
        frame_period = 1.0 / _RENDER_HZ
        next_frame_time = time.monotonic()
        step_count = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                if window is not None:
                    window.process_commands()

                now = time.monotonic()
                if poll_period is not None and now >= next_poll_time and controller.fault is None:
                    try:
                        controller.poll_hardware()
                    except DynamixelError as error:
                        controller.enter_fault(error)
                        if window is not None:
                            window.show_error(controller.fault or str(error))
                    next_poll_time = now + poll_period

                sim.render()
                now = time.monotonic()
                if window is not None and now >= next_ui_refresh_time:
                    window.refresh()
                    next_ui_refresh_time = now + ui_refresh_period

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

            next_frame_time += frame_period
            sleep_duration = next_frame_time - time.monotonic()
            if sleep_duration > 0.0:
                time.sleep(sleep_duration)
            else:
                next_frame_time = time.monotonic()
    finally:
        if window is not None:
            window.close()
        if controller is not None:
            controller.close()
        elif bus is not None:
            bus.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
