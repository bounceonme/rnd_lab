"""RSL-RL runner configuration for the opt-in RND actuator and IMU task."""

from isaaclab.utils import configclass

from .rsl_rl_actuator_ppo_cfg import RndStepFlatActuatorPPORunnerCfg


@configclass
class RndStepFlatActuatorImuPPORunnerCfg(RndStepFlatActuatorPPORunnerCfg):
    experiment_name = "rnd_step/flat_actuator_imu"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rnd_step/flat_actuator_imu"
        # History stacking mixes commands, IMU values, encoder state, and previous actions.
        # Keep the initial search broad, but limit late entropy-driven noise growth that was
        # observed after the tracking plateau in the 2026-07-17 cycle.
        self.policy.actor_obs_normalization = True
        self.policy.init_noise_std = 0.5
        self.algorithm.entropy_coef = 0.004
