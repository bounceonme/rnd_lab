"""RSL-RL runner configuration for the opt-in RND actuator and IMU task."""

from isaaclab.utils import configclass

from .rsl_rl_actuator_ppo_cfg import RndStepFlatActuatorPPORunnerCfg


@configclass
class RndStepFlatActuatorImuPPORunnerCfg(RndStepFlatActuatorPPORunnerCfg):
    experiment_name = "rnd_step/flat_actuator_imu"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rnd_step/flat_actuator_imu"
