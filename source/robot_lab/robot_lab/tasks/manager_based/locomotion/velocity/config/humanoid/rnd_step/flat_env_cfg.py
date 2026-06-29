# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .flat_behavior_cfg import (
    apply_step_flat_commands,
    apply_step_flat_stable_walk_rewards,
    apply_step_flat_terminations,
    dampen_arm_actions as _dampen_arm_actions,
)
from .flat_domain_randomization import apply_step_flat_domain_randomization
from .offline_ground_plane import maybe_enable_offline_ground_plane
from .rough_env_cfg import RndStepRoughEnvCfg


@configclass
class RndStepFlatEnvCfg(RndStepRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        maybe_enable_offline_ground_plane()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        apply_step_flat_domain_randomization(self)
        apply_step_flat_stable_walk_rewards(self)
        apply_step_flat_terminations(self)
        apply_step_flat_commands(self)
        self.actions.joint_pos.scale = _dampen_arm_actions(self.actions.joint_pos.scale)

        if self.__class__.__name__ == "RndStepFlatEnvCfg":
            self.disable_zero_weight_rewards()
