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
import math
import os
import sys
from pathlib import Path

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
    default="RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--fixed_actuator_sample",
    action="store_true",
    default=False,
    help="Use midpoint actuator-model parameters instead of sampling the validated ranges.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many environment steps; zero keeps the interactive test running.",
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

from robot_lab.actuators.rnd_stateful import load_rnd_actuator_model
from robot_lab.assets.rnd_actuator import (
    RND_ACTUATOR_RUNTIME_MODEL_PATH,
    RND_ARMATURE_RANDOMIZATION_PATH,
    RND_TORQUE_RANDOMIZATION_PATH,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl")))
from torque_plot import TorquePlotWindow  # isort: skip

try:
    import omni.kit.app
    import omni.ui as ui
except ImportError:
    ui = None


_STEP_PD_TEST_ROOT_Z = 0.3905
_STEP_4BAR_PD_TEST_ROOT_Z = 0.4294
_STATEFUL_ACTUATOR_GROUP = "replay_validated"


class PDGainWindow:
    """Per-joint runtime monitor and PD tuner for implicit or explicit actuators."""

    _WINDOW_TITLE = "PD Gain Tuner"
    _PRIMARY_DOCK_TARGET_WINDOW = TorquePlotWindow._WINDOW_TITLE
    _FALLBACK_DOCK_TARGET_WINDOW = "Viewport"
    _DOCK_POSITION_RATIO = 0.55
    _MODEL_UPDATE_INTERVAL = 15

    def __init__(self, robot, defaults: dict[str, dict]):
        self._robot = robot
        self._defaults = defaults
        self._pending_reset = False
        self._window = None
        self._joint_widgets: dict[str, dict] = {}
        self._model_status_label = None
        self._model_update_counter = 0

        if ui is None:
            return

        self._build_window()
        asyncio.ensure_future(self._dock_window())

    def _build_gain_field(self, joint_name: str, gain_name: str, current: float):
        field = ui.FloatField(width=72, alignment=ui.Alignment.LEFT_CENTER)
        field.model.set_value(current)
        field.model.add_value_changed_fn(
            lambda model, joint_name=joint_name, gain_name=gain_name: self._apply_gain(
                joint_name, gain_name, model.as_float
            )
        )
        return field.model

    def _build_joint_row(self, joint_name: str):
        defaults = self._defaults[joint_name]
        with ui.VStack(height=56, spacing=2):
            with ui.HStack(height=25, spacing=6):
                ui.Label(joint_name, width=165, style={"font_size": 12, "color": 0xFFFFFFFF})
                stiffness_model = self._build_gain_field(joint_name, "stiffness", defaults["stiffness"])
                damping_model = self._build_gain_field(joint_name, "damping", defaults["damping"])
                ui.Label(
                    f"{defaults['effort_limit']:.2f} Nm",
                    width=75,
                    alignment=ui.Alignment.RIGHT_CENTER,
                    style={"font_size": 11, "color": 0xFFBDBDBD},
                )
                ui.Label(
                    f"{defaults['velocity_limit']:.2f} rad/s",
                    width=92,
                    alignment=ui.Alignment.RIGHT_CENTER,
                    style={"font_size": 11, "color": 0xFFBDBDBD},
                )
                ui.Label(
                    f"J {defaults['armature']:.3f}",
                    width=68,
                    alignment=ui.Alignment.RIGHT_CENTER,
                    style={"font_size": 11, "color": 0xFFBDBDBD},
                )
            with ui.HStack(height=23, spacing=6):
                ui.Label(defaults["actuator_group"], width=165, style={"font_size": 10, "color": 0xFF909090})
                model_label = ui.Label(
                    "",
                    width=ui.Fraction(1),
                    elided_text=True,
                    style={"font_size": 11, "color": 0xFFB0B0B0},
                )
                ui.Button(
                    "Restore",
                    width=72,
                    clicked_fn=lambda joint_name=joint_name: self._restore_joint_defaults(joint_name),
                )
            ui.Separator(height=2)

        self._joint_widgets[joint_name] = {
            "stiffness_model": stiffness_model,
            "damping_model": damping_model,
            "model_label": model_label,
        }

    def _build_window(self):
        self._window = ui.Window(
            self._WINDOW_TITLE,
            width=700,
            height=760,
            visible=True,
            dock_preference=ui.DockPreference.LEFT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=8, height=ui.Fraction(1), width=ui.Fraction(1)):
                ui.Label(
                    "RND STEP PD / Actuator Model",
                    height=22,
                    style={"font_size": 16, "color": 0xFFFFFFFF},
                )
                with ui.HStack(height=30, spacing=8):
                    ui.Button("Reset Pose", clicked_fn=self._request_reset)
                    ui.Button("Restore All Defaults", clicked_fn=self._restore_all_defaults)
                self._model_status_label = ui.Label(
                    "",
                    height=18,
                    style={"font_size": 11, "color": 0xFFB0B0B0},
                )
                with ui.HStack(height=20, spacing=6):
                    ui.Label("Joint", width=165, style={"font_size": 11, "color": 0xFFBDBDBD})
                    ui.Label("Kp", width=72, style={"font_size": 11, "color": 0xFFBDBDBD})
                    ui.Label("Kd", width=72, style={"font_size": 11, "color": 0xFFBDBDBD})
                    ui.Label(
                        "Torque",
                        width=75,
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": 11, "color": 0xFFBDBDBD},
                    )
                    ui.Label(
                        "Velocity",
                        width=92,
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": 11, "color": 0xFFBDBDBD},
                    )
                    ui.Label(
                        "Armature",
                        width=68,
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": 11, "color": 0xFFBDBDBD},
                    )
                ui.Separator(height=2)
                with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                ):
                    with ui.VStack(spacing=3, height=0):
                        for joint_name in self._defaults:
                            self._build_joint_row(joint_name)

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

    def _apply_gain(self, joint_name: str, gain_name: str, value: float):
        value = max(0.0, float(value))
        defaults = self._defaults[joint_name]
        actuator = defaults["actuator"]
        actuator_joint_id = defaults["actuator_joint_id"]

        if gain_name == "stiffness":
            actuator.stiffness[:, actuator_joint_id] = value
            if actuator.is_implicit_model:
                self._robot.write_joint_stiffness_to_sim(value, joint_ids=[defaults["robot_joint_id"]])
        elif gain_name == "damping":
            actuator.damping[:, actuator_joint_id] = value
            if actuator.is_implicit_model:
                self._robot.write_joint_damping_to_sim(value, joint_ids=[defaults["robot_joint_id"]])
        else:
            raise ValueError(f"Unsupported gain name: {gain_name}")

    def _restore_joint_defaults(self, joint_name: str):
        widgets = self._joint_widgets[joint_name]
        widgets["stiffness_model"].set_value(self._defaults[joint_name]["stiffness"])
        widgets["damping_model"].set_value(self._defaults[joint_name]["damping"])

    def _restore_all_defaults(self):
        for joint_name in self._defaults:
            self._restore_joint_defaults(joint_name)

    def update(self, force: bool = False):
        self._model_update_counter += 1
        if not force and self._model_update_counter % self._MODEL_UPDATE_INTERVAL != 0:
            return

        stateful_count = 0
        randomized_count = 0
        for joint_name, defaults in self._defaults.items():
            actuator = defaults["actuator"]
            command_path = getattr(actuator, "command_path", None)
            torque_randomizer = getattr(actuator, "torque_randomizer", None)
            label = self._joint_widgets[joint_name]["model_label"]
            if command_path is None and torque_randomizer is None:
                label.text = "implicit PD | no actuator randomization"
                continue

            if command_path is not None:
                stateful_count += 1
            if (command_path is not None and command_path.sample_randomization) or (
                torque_randomizer is not None and torque_randomizer.sample_randomization
            ):
                randomized_count += 1
            actuator_joint_id = defaults["actuator_joint_id"]
            if command_path is not None:
                delay_ms = float(command_path.sampled_delay_s[0, actuator_joint_id].item()) * 1000.0
                bias_deg = math.degrees(float(command_path.sampled_position_bias_rad[0, actuator_joint_id].item()))
                thresholds = command_path.sampled_play_thresholds_rad[0, actuator_joint_id]
                active_thresholds = thresholds[thresholds > 0.0]
                if active_thresholds.numel() > 0:
                    play_deg = math.degrees(float(active_thresholds.max().item()))
                    play_text = f"play +/-{play_deg:.3f} deg"
                else:
                    play_text = "play identity"
                command_text = f"delay {delay_ms:.2f} ms | {play_text} | bias {bias_deg:+.3f} deg"
            else:
                command_text = "plain explicit PD"
            if torque_randomizer is not None:
                friction_nm = float(torque_randomizer.sampled_coulomb_torque_nm[0, actuator_joint_id].item())
                strength = float(torque_randomizer.sampled_motor_strength_scale[0, actuator_joint_id].item())
                torque_text = f"friction {friction_nm:.3f} Nm | strength x{strength:.3f}"
            else:
                torque_text = "torque randomization OFF"
            label.text = f"{command_text} | {torque_text}"

        if self._model_status_label is not None:
            if stateful_count:
                sample_mode = "RANDOMIZED" if randomized_count else "MIDPOINT"
                self._model_status_label.text = (
                    f"stateful joints {stateful_count} | {sample_mode} | runtime seed restore available"
                )
            else:
                self._model_status_label.text = "implicit actuator configuration"


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
    if hasattr(env_cfg.events, "randomize_joint_armature"):
        env_cfg.events.randomize_joint_armature.params["sample_randomization"] = not args_cli.fixed_actuator_sample
    if args_cli.fixed_actuator_sample and _STATEFUL_ACTUATOR_GROUP in env_cfg.scene.robot.actuators:
        env_cfg.scene.robot.actuators[_STATEFUL_ACTUATOR_GROUP].sample_randomization = False
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


def _resolve_robot_joint_ids(robot, actuator) -> list[int]:
    joint_indices = actuator.joint_indices
    if isinstance(joint_indices, slice):
        resolved = list(range(len(robot.joint_names)))[joint_indices]
    elif isinstance(joint_indices, torch.Tensor):
        resolved = [int(value) for value in joint_indices.detach().cpu().tolist()]
    else:
        resolved = [int(value) for value in joint_indices]
    if len(resolved) != len(actuator.joint_names):
        raise ValueError(
            f"Actuator joint index count does not match joint names: {len(resolved)} != {len(actuator.joint_names)}."
        )
    return resolved


def _collect_joint_defaults(robot, model: dict | None) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    model_joints = {} if model is None else model["joints"]
    for actuator_group, actuator in robot.actuators.items():
        robot_joint_ids = _resolve_robot_joint_ids(robot, actuator)
        for actuator_joint_id, (joint_name, robot_joint_id) in enumerate(
            zip(actuator.joint_names, robot_joint_ids, strict=True)
        ):
            if joint_name in by_name:
                raise ValueError(f"Joint {joint_name!r} belongs to more than one actuator group.")
            by_name[joint_name] = {
                "actuator": actuator,
                "actuator_group": actuator_group,
                "actuator_joint_id": actuator_joint_id,
                "robot_joint_id": robot_joint_id,
                "stiffness": float(actuator.stiffness[0, actuator_joint_id].item()),
                "damping": float(actuator.damping[0, actuator_joint_id].item()),
                "effort_limit": float(actuator.effort_limit[0, actuator_joint_id].item()),
                "velocity_limit": float(actuator.velocity_limit[0, actuator_joint_id].item()),
                "armature": float(robot.data.joint_armature[0, robot_joint_id].item()),
                "armature_seed": float(actuator.armature[0, actuator_joint_id].item()),
                "model_joint": model_joints.get(joint_name, {}),
            }

    ordered = {joint_name: by_name[joint_name] for joint_name in robot.joint_names if joint_name in by_name}
    if len(ordered) != len(by_name):
        missing = sorted(set(by_name) - set(ordered))
        raise ValueError(f"Actuator joints are missing from the robot joint list: {missing}.")
    return ordered


def _print_actuator_summary(defaults: dict[str, dict], model: dict | None) -> None:
    if model is not None:
        print(f"[INFO]: Runtime actuator model: {Path(RND_ACTUATOR_RUNTIME_MODEL_PATH).resolve()}")
        print(f"[INFO]: Startup armature randomization: {Path(RND_ARMATURE_RANDOMIZATION_PATH).resolve()}")
        print(
            f"[INFO]: status={model['application_status']}, physics={model['physics_hz']:.1f} Hz, "
            f"policy={model['policy_hz']:.1f} Hz, joints={len(model['integration_joint_names'])}"
        )

    print("Active joint controller and sampled actuator values for env[0]:")
    print(
        "joint                       group                Kp     Kd  limit  velocity  armature  "
        "delay   play   bias strength friction"
    )
    for joint_name, values in defaults.items():
        actuator = values["actuator"]
        command_path = getattr(actuator, "command_path", None)
        torque_randomizer = getattr(actuator, "torque_randomizer", None)
        joint_id = values["actuator_joint_id"]
        if command_path is None:
            delay_text = "   n/a"
            play_text = "   n/a"
            bias_text = "   n/a"
        else:
            delay_ms = float(command_path.sampled_delay_s[0, joint_id].item()) * 1000.0
            bias_deg = math.degrees(float(command_path.sampled_position_bias_rad[0, joint_id].item()))
            thresholds = command_path.sampled_play_thresholds_rad[0, joint_id]
            active_thresholds = thresholds[thresholds > 0.0]
            play_deg = 0.0 if active_thresholds.numel() == 0 else math.degrees(float(active_thresholds.max().item()))
            delay_text = f"{delay_ms:5.2f}ms"
            play_text = f"{play_deg:5.3f}d"
            bias_text = f"{bias_deg:+5.3f}d"
        if torque_randomizer is None:
            strength_text = "  n/a"
            friction_text = "   n/a"
        else:
            strength = float(torque_randomizer.sampled_motor_strength_scale[0, joint_id].item())
            friction_nm = float(torque_randomizer.sampled_coulomb_torque_nm[0, joint_id].item())
            strength_text = f"{strength:5.3f}"
            friction_text = f"{friction_nm:5.3f}N"
        print(
            f"{joint_name:27s} {values['actuator_group']:19s} "
            f"{values['stiffness']:5.2f} {values['damping']:6.2f} "
            f"{values['effort_limit']:5.2f} {values['velocity_limit']:8.2f} "
            f"{values['armature']:9.3f} {delay_text:>7s} {play_text:>7s} {bias_text:>7s} "
            f"{strength_text:>8s} {friction_text:>8s}"
        )

    if any(getattr(values["actuator"], "torque_randomizer", None) is not None for values in defaults.values()):
        print(f"[INFO]: Torque/friction domain randomization is ON: {Path(RND_TORQUE_RANDOMIZATION_PATH).resolve()}")


def _validate_pd_test_env(env):
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if not isinstance(action_term, JointPositionAction):
        raise TypeError("rnd_pd_test.py requires the 'joint_pos' action term to be JointPositionAction.")
    if not getattr(action_term.cfg, "use_default_offset", False):
        raise ValueError(
            "rnd_pd_test.py requires JointPositionActionCfg(use_default_offset=True) so zero actions hold the default pose."
        )

    robot = env.unwrapped.scene["robot"]
    if "Actuator" in args_cli.task:
        if _STATEFUL_ACTUATOR_GROUP not in robot.actuators:
            raise ValueError(
                f"Actuator task requires the {_STATEFUL_ACTUATOR_GROUP!r} actuator group; "
                f"found {tuple(robot.actuators)}."
            )
        actuator = robot.actuators[_STATEFUL_ACTUATOR_GROUP]
        if not hasattr(actuator, "command_path"):
            raise TypeError("Actuator task did not instantiate the RND stateful command path.")
        model_path = Path(actuator.cfg.model_path).expanduser().resolve()
        expected_path = Path(RND_ACTUATOR_RUNTIME_MODEL_PATH).resolve()
        if model_path != expected_path:
            raise ValueError(f"PD test expected runtime model {expected_path}, got {model_path}.")
        model = load_rnd_actuator_model(
            model_path,
            actuator.joint_names,
            require_sim_replay_validation=True,
            require_command_path_seed=True,
        )
        if set(actuator.joint_names) != set(model["integration_joint_names"]):
            raise ValueError("Instantiated stateful actuator joints do not match runtime integration_joint_names.")
        return model

    required_groups = ("legs", "feet")
    if "RND-Step-4Bar" in args_cli.task:
        required_groups += ("waist",)
    missing_groups = [name for name in required_groups if name not in robot.actuators]
    if missing_groups:
        raise ValueError(f"Missing actuator groups for PD test: {missing_groups}")
    return None


def main():
    if "RND-Step" not in args_cli.task:
        raise ValueError("rnd_pd_test.py currently supports only RND-Step flat tasks.")
    if args_cli.max_steps < 0:
        raise ValueError("--max_steps must be zero or positive.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_pd_test_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    runtime_model = _validate_pd_test_env(env)

    robot = env.unwrapped.scene["robot"]
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env.reset()
    env.step(zero_actions)

    default_gains = _collect_joint_defaults(robot, runtime_model)
    _print_actuator_summary(default_gains, runtime_model)

    torque_plot = None
    pd_gain_window = None
    if not getattr(args_cli, "headless", False):
        torque_plot = TorquePlotWindow(robot, env_idx=0, update_interval=1)
        torque_plot.update(force=True)
        pd_gain_window = PDGainWindow(robot, default_gains)
        pd_gain_window.update(force=True)

    step_count = 0
    while simulation_app.is_running() and (args_cli.max_steps == 0 or step_count < args_cli.max_steps):
        with torch.inference_mode():
            if pd_gain_window is not None and pd_gain_window.consume_reset_requested():
                env.reset()
                if torque_plot is not None:
                    torque_plot.update(force=True)

            env.step(zero_actions)
            step_count += 1
            if torque_plot is not None:
                torque_plot.update()
            if pd_gain_window is not None:
                pd_gain_window.update()

    if torque_plot is not None:
        torque_plot.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
