# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive PD gain test for the RND step robot."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="PD gain test for the RND step robot.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-RND-Step-v0",
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: F401
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_tasks.utils import parse_env_cfg

from robot_lab.assets.rnd import STEP_CFG

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
)
from torque_plot import TorquePlotWindow  # isort: skip

try:
    import omni.kit.app
    import omni.ui as ui
except ImportError:
    ui = None


_STEP_PD_TEST_ROOT_Z = 0.3905
_STEP_4BAR_PD_TEST_ROOT_Z = 0.4294


class PDGainWindow:
    """Simple runtime tuner for implicit actuator stiffness and damping."""

    _WINDOW_TITLE = "PD Gain Tuner"
    _PRIMARY_DOCK_TARGET_WINDOW = TorquePlotWindow._WINDOW_TITLE
    _FALLBACK_DOCK_TARGET_WINDOW = "Viewport"
    _DOCK_POSITION_RATIO = 0.55

    def __init__(self, robot, defaults: dict[str, dict[str, float]]):
        self._robot = robot
        self._defaults = defaults
        self._pending_reset = False
        self._window = None
        self._group_widgets: dict[str, dict] = {}

        if ui is None:
            return

        self._build_window()
        asyncio.ensure_future(self._dock_window())

    def _build_gain_row(self, group_name: str, gain_name: str, current: float, max_value: float, step: float):
        with ui.HStack(height=26):
            ui.Label(gain_name, width=70, style={"font_size": 12, "color": 0xFFDDDDDD})
            field = ui.FloatField(width=80, alignment=ui.Alignment.LEFT_CENTER).model
            field.set_value(current)
            field.add_value_changed_fn(
                lambda model, group_name=group_name, gain_name=gain_name: self._apply_gain(
                    group_name, gain_name, model.as_float
                )
            )
            ui.Spacer(width=8)
            ui.FloatSlider(
                width=ui.Fraction(1),
                alignment=ui.Alignment.LEFT_CENTER,
                min=0.0,
                max=max_value,
                step=step,
                model=field,
            )
        return field

    def _build_group_card(self, group_name: str):
        default_stiffness = self._defaults[group_name]["stiffness"]
        default_damping = self._defaults[group_name]["damping"]

        with ui.ZStack(height=150, width=ui.Fraction(1)):
            ui.Rectangle(
                width=ui.Fraction(1),
                height=ui.Fraction(1),
                style={
                    "background_color": 0xFF303030,
                    "border_color": 0xFF5A5A5A,
                    "border_width": 1.0,
                },
            )
            with ui.VStack(spacing=6, height=ui.Fraction(1), width=ui.Fraction(1)):
                with ui.HStack(height=20):
                    ui.Label(
                        group_name,
                        width=ui.Fraction(1),
                        style={"font_size": 15, "color": 0xFFFFFFFF},
                    )
                    current_label = ui.Label(
                        "",
                        width=150,
                        style={"font_size": 11, "color": 0xFFBDBDBD},
                    )

                stiffness_model = self._build_gain_row(
                    group_name,
                    "stiffness",
                    default_stiffness,
                    max(default_stiffness * 4.0, 10.0),
                    0.1,
                )
                damping_model = self._build_gain_row(
                    group_name,
                    "damping",
                    default_damping,
                    max(default_damping * 4.0, 2.0),
                    0.05,
                )

                with ui.HStack(height=26, spacing=8):
                    ui.Button(
                        "Restore Default",
                        width=140,
                        clicked_fn=lambda group_name=group_name: self._restore_group_defaults(group_name),
                    )
                    ui.Spacer(width=ui.Fraction(1))

        self._group_widgets[group_name] = {
            "current_label": current_label,
            "stiffness_model": stiffness_model,
            "damping_model": damping_model,
        }
        self._refresh_group_label(group_name)

    def _build_window(self):
        self._window = ui.Window(
            self._WINDOW_TITLE,
            width=380,
            height=520,
            visible=True,
            dock_preference=ui.DockPreference.LEFT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=8, height=ui.Fraction(1), width=ui.Fraction(1)):
                ui.Label(
                    "PD Gain Tuner",
                    height=22,
                    style={"font_size": 16, "color": 0xFFFFFFFF},
                )
                ui.Label(
                    "Zero-action hold of STEP_CFG default pose",
                    height=16,
                    style={"font_size": 12, "color": 0xFFB0B0B0},
                )
                for group_name in self._defaults:
                    self._build_group_card(group_name)
                with ui.HStack(height=30, spacing=8):
                    ui.Button("Reset Pose", clicked_fn=self._request_reset)
                    ui.Button("Restore All Defaults", clicked_fn=self._restore_all_defaults)

    async def _dock_window(self):
        for _ in range(10):
            if ui.Workspace.get_window(self._WINDOW_TITLE):
                break
            await omni.kit.app.get_app().next_update_async()

        custom_window = ui.Workspace.get_window(self._WINDOW_TITLE)
        target_window = ui.Workspace.get_window(self._PRIMARY_DOCK_TARGET_WINDOW)
        if target_window is None:
            target_window = ui.Workspace.get_window(self._FALLBACK_DOCK_TARGET_WINDOW)
        if custom_window and target_window:
            custom_window.dock_in(target_window, ui.DockPosition.RIGHT, self._DOCK_POSITION_RATIO)
            custom_window.focus()

    def _request_reset(self):
        self._pending_reset = True

    def consume_reset_requested(self) -> bool:
        if not self._pending_reset:
            return False
        self._pending_reset = False
        return True

    def _refresh_group_label(self, group_name: str):
        actuator = self._robot.actuators[group_name]
        stiffness = float(actuator.stiffness[0, 0].item())
        damping = float(actuator.damping[0, 0].item())
        self._group_widgets[group_name]["current_label"].text = f"Kp {stiffness:.2f} | Kd {damping:.2f}"

    def _apply_gain(self, group_name: str, gain_name: str, value: float):
        value = max(0.0, float(value))
        actuator = self._robot.actuators[group_name]
        joint_ids = actuator.joint_indices

        if gain_name == "stiffness":
            actuator.stiffness[:] = value
            self._robot.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=joint_ids)
        elif gain_name == "damping":
            actuator.damping[:] = value
            self._robot.write_joint_damping_to_sim(actuator.damping, joint_ids=joint_ids)
        else:
            raise ValueError(f"Unsupported gain name: {gain_name}")

        self._refresh_group_label(group_name)

    def _restore_group_defaults(self, group_name: str):
        widgets = self._group_widgets[group_name]
        widgets["stiffness_model"].set_value(self._defaults[group_name]["stiffness"])
        widgets["damping_model"].set_value(self._defaults[group_name]["damping"])

    def _restore_all_defaults(self):
        for group_name in self._defaults:
            self._restore_group_defaults(group_name)


def _configure_pd_test_env(env_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    init_x, init_y, _ = env_cfg.scene.robot.init_state.pos
    if "RND-Step-4Bar" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (init_x, init_y, _STEP_4BAR_PD_TEST_ROOT_Z)
    else:
        # FK over the default pose puts the foot mesh minimum at about -0.3855 m from base_link.
        # Keep a small clearance so Reset Pose does not drop the robot from STEP_CFG's training spawn height.
        env_cfg.scene.robot.init_state.pos = (init_x, init_y, _STEP_PD_TEST_ROOT_Z)

    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False

    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    env_cfg.events.randomize_rigid_body_material = None
    env_cfg.events.randomize_rigid_body_mass_base = None
    env_cfg.events.randomize_rigid_body_mass_others = None
    env_cfg.events.randomize_com_positions = None
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.zero_velocity_threshold = 0.0

    env_cfg.terminations.time_out = None
    env_cfg.terminations.illegal_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None


def _validate_pd_test_env(env):
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if not isinstance(action_term, JointPositionAction):
        raise TypeError(
            "rnd_pd_test.py requires the 'joint_pos' action term to be JointPositionAction."
        )
    if not getattr(action_term.cfg, "use_default_offset", False):
        raise ValueError(
            "rnd_pd_test.py requires JointPositionActionCfg(use_default_offset=True) so zero actions hold the default pose."
        )

    required_groups = ("legs", "feet")
    if "RND-Step-4Bar" in args_cli.task:
        required_groups += ("waist",)
    missing_groups = [name for name in required_groups if name not in env.unwrapped.scene["robot"].actuators]
    if missing_groups:
        raise ValueError(f"Missing actuator groups for PD test: {missing_groups}")


def main():
    if "RND-Step" not in args_cli.task:
        raise ValueError("rnd_pd_test.py currently supports only RND-Step flat tasks.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_pd_test_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    _validate_pd_test_env(env)

    robot = env.unwrapped.scene["robot"]
    default_gains = {
        "legs": {
            "stiffness": float(STEP_CFG.actuators["legs"].stiffness),
            "damping": float(STEP_CFG.actuators["legs"].damping),
        },
        "feet": {
            "stiffness": float(STEP_CFG.actuators["feet"].stiffness),
            "damping": float(STEP_CFG.actuators["feet"].damping),
        },
    }
    if "waist" in STEP_CFG.actuators:
        default_gains["waist"] = {
            "stiffness": float(STEP_CFG.actuators["waist"].stiffness),
            "damping": float(STEP_CFG.actuators["waist"].damping),
        }

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    torque_plot = None
    pd_gain_window = None
    if not getattr(args_cli, "headless", False):
        torque_plot = TorquePlotWindow(robot, env_idx=0, update_interval=1)
        torque_plot.update(force=True)
        pd_gain_window = PDGainWindow(robot, default_gains)

    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            if pd_gain_window is not None and pd_gain_window.consume_reset_requested():
                env.reset()
                if torque_plot is not None:
                    torque_plot.update(force=True)

            env.step(zero_actions)
            if torque_plot is not None:
                torque_plot.update()

    if torque_plot is not None:
        torque_plot.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
