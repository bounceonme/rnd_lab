# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents


gym.register(
    id="RNDLab-Isaac-Velocity-Rough-RND-Step-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RndStepRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RndStepRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:RndStepRoughTrainerCfg",
    },
)


gym.register(
    id="RNDLab-Isaac-Velocity-Flat-RND-Step-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RndStepFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RndStepFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:RndStepFlatTrainerCfg",
    },
)


gym.register(
    id="RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_actuator_env_cfg:RndStepFlatActuatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_actuator_ppo_cfg:RndStepFlatActuatorPPORunnerCfg",
    },
)


gym.register(
    id="RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_actuator_imu_env_cfg:RndStepFlatActuatorImuEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_actuator_imu_ppo_cfg:RndStepFlatActuatorImuPPORunnerCfg"),
    },
)
