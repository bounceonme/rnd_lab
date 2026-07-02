# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--disable_torque_plot",
    action="store_true",
    default=False,
    help="Disable the automatic omni.ui torque monitor window during play.",
)
parser.add_argument(
    "--torque_plot_interval",
    type=int,
    default=1,
    help="Update interval in simulation steps for the torque monitor window.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip
from torque_plot import TorquePlotWindow  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _keyboard_sensitivity(axis_range: tuple[float, float], fallback: float) -> float:
    sensitivity = max(abs(axis_range[0]), abs(axis_range[1]))
    return sensitivity if sensitivity > 0.0 else fallback


def _add_wasd_keyboard_bindings(controller: Se2Keyboard):
    """Mirror IsaacLab's arrow-key SE(2) controls onto WASD."""
    controller._INPUT_KEY_MAPPING["W"] = controller._INPUT_KEY_MAPPING["UP"].copy()
    controller._INPUT_KEY_MAPPING["S"] = controller._INPUT_KEY_MAPPING["DOWN"].copy()
    controller._INPUT_KEY_MAPPING["A"] = controller._INPUT_KEY_MAPPING["LEFT"].copy()
    controller._INPUT_KEY_MAPPING["D"] = controller._INPUT_KEY_MAPPING["RIGHT"].copy()


def _set_rnd_step_keyboard_bindings(controller: Se2Keyboard, env_cfg) -> None:
    """Bind keys directly in the RND STEP policy command frame."""
    lateral_speed = _keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.lin_vel_x, fallback=0.5)
    forward_speed = _keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.lin_vel_y, fallback=0.5)
    yaw_speed = _keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.ang_vel_z, fallback=0.5)

    controller._INPUT_KEY_MAPPING.update(
        {
            # STEP forward is body -y; body +x is the robot's left side.
            "NUMPAD_8": np.asarray([0.0, -forward_speed, 0.0]),
            "UP": np.asarray([0.0, -forward_speed, 0.0]),
            "W": np.asarray([0.0, -forward_speed, 0.0]),
            "NUMPAD_2": np.asarray([0.0, forward_speed, 0.0]),
            "DOWN": np.asarray([0.0, forward_speed, 0.0]),
            "S": np.asarray([0.0, forward_speed, 0.0]),
            "NUMPAD_4": np.asarray([lateral_speed, 0.0, 0.0]),
            "LEFT": np.asarray([lateral_speed, 0.0, 0.0]),
            "A": np.asarray([lateral_speed, 0.0, 0.0]),
            "NUMPAD_6": np.asarray([-lateral_speed, 0.0, 0.0]),
            "RIGHT": np.asarray([-lateral_speed, 0.0, 0.0]),
            "D": np.asarray([-lateral_speed, 0.0, 0.0]),
            "NUMPAD_7": np.asarray([0.0, 0.0, yaw_speed]),
            "Z": np.asarray([0.0, 0.0, yaw_speed]),
            "NUMPAD_9": np.asarray([0.0, 0.0, -yaw_speed]),
            "X": np.asarray([0.0, 0.0, -yaw_speed]),
        }
    )


def _make_keyboard_command_manager_fn(
    controller: Se2Keyboard,
    env_cfg,
    command_name: str = "base_velocity",
    ramp_rates: tuple[float, float, float] | None = None,
):
    ramp_rates_tensor = torch.tensor(ramp_rates, dtype=torch.float32) if ramp_rates is not None else None
    command_state = {"value": None}
    heading_offset = getattr(env_cfg.commands.base_velocity, "heading_offset", 0.0)
    zero_velocity_threshold = env_cfg.commands.base_velocity.zero_velocity_threshold

    def command_fn(env):
        target_command = torch.as_tensor(controller.advance(), dtype=torch.float32, device=env.device).unsqueeze(0)
        if command_state["value"] is None:
            command_state["value"] = torch.zeros_like(target_command)

        if ramp_rates_tensor is None:
            command_state["value"] = target_command.clone()
        else:
            max_delta = ramp_rates_tensor.to(env.device).unsqueeze(0) * env.step_dt
            command_state["value"] += torch.clamp(
                target_command - command_state["value"], min=-max_delta, max=max_delta
            )

        command_term = env.command_manager.get_term(command_name)
        command = command_state["value"]
        if hasattr(command_term, "vel_command_target_b"):
            command_term.vel_command_target_b[: command.shape[0]] = command
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[: command.shape[0]] = command
        if hasattr(command_term, "is_heading_env"):
            command_term.is_heading_env[: command.shape[0]] = False
        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[: command.shape[0]] = torch.linalg.norm(command, dim=1) < zero_velocity_threshold
        if hasattr(command_term, "heading_target"):
            command_term.heading_target[: command.shape[0]] = env.scene["robot"].data.heading_w[: command.shape[0]] + heading_offset

        return command_term.command[: command.shape[0]]

    return command_fn


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Keep environment randomization events enabled so play can inspect sim-to-real disturbances.
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=_keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.lin_vel_x, fallback=0.5),
            v_y_sensitivity=_keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.lin_vel_y, fallback=0.5),
            omega_z_sensitivity=_keyboard_sensitivity(env_cfg.commands.base_velocity.ranges.ang_vel_z, fallback=0.5),
        )
        controller = Se2Keyboard(config)
        if "RND-Step" in task_name:
            _set_rnd_step_keyboard_bindings(controller, env_cfg)
            keyboard_ramp_rates = getattr(env_cfg.commands.base_velocity, "command_ramp_rates", (0.4, 0.8, 1.8))
        else:
            _add_wasd_keyboard_bindings(controller)
            keyboard_ramp_rates = None
        keyboard_command_fn = _make_keyboard_command_manager_fn(controller, env_cfg, ramp_rates=keyboard_ramp_rates)
        env_cfg.observations.policy.velocity_commands = ObsTerm(func=keyboard_command_fn)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    sim_env = env.unwrapped

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    torque_plot = None
    if not getattr(args_cli, "headless", False) and not args_cli.disable_torque_plot:
        try:
            torque_plot = TorquePlotWindow(
                sim_env.scene["robot"],
                env_idx=0,
                update_interval=args_cli.torque_plot_interval,
            )
            torque_plot.update(force=True)
        except Exception as exc:
            print(f"[WARN] Failed to create torque monitor window: {exc}")
    if args_cli.keyboard:
        camera_follow(env, task_name=task_name)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if torque_plot is not None:
                torque_plot.update()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env, task_name=task_name)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if torque_plot is not None:
        torque_plot.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
