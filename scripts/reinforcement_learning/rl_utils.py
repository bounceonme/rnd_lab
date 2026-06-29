# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def _camera_offset_for_task(task_name: str | None, device: str | torch.device) -> torch.Tensor:
    if task_name and "RND-Step" in task_name:
        # STEP robot forward axis is body -y, so place the camera behind it along +y.
        return torch.tensor([0.0, 3.0, 0.6], dtype=torch.float32, device=device)
    return torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=device)


def camera_follow(env, task_name: str | None = None):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = _camera_offset_for_task(task_name, env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    # camera_pos[2] = torch.clamp(camera_pos[2], min=0.1)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
