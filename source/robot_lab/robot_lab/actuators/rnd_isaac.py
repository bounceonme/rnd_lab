"""Isaac Lab explicit-PD adapter for the RND stateful command path.

Import this module only after :class:`isaaclab.app.AppLauncher` has started.
The adapter is intentionally not exported by ``robot_lab.actuators`` so that
the Torch kernel remains testable without Omniverse modules such as ``pxr``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .rnd_stateful import StatefulCommandPath, load_rnd_actuator_model
from .rnd_torque_randomization import EpisodeTorqueRandomizer, load_rnd_torque_randomization


class RndTorqueRandomizedPDActuator(IdealPDActuator):
    """Explicit PD actuator with episode-sampled strength and Coulomb resistance."""

    cfg: RndTorqueRandomizedPDActuatorCfg

    def __init__(self, cfg: RndTorqueRandomizedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        torque_model = load_rnd_torque_randomization(cfg.torque_randomization_model_path, self.joint_names)
        self.torque_randomizer = EpisodeTorqueRandomizer(
            model=torque_model,
            joint_names=self.joint_names,
            num_envs=self._num_envs,
            device=self._device,
            seed=cfg.random_seed + 1_000_003,
            sample_randomization=cfg.sample_randomization,
        )

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice) and env_ids == slice(None):
            return torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long).flatten()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice):
        ids = self._normalize_env_ids(env_ids)
        self.torque_randomizer.reset(ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        if control_action.joint_efforts is None:
            raise ValueError("RndTorqueRandomizedPDActuator requires explicit joint efforts from IdealPDActuator.")
        net_effort = self.torque_randomizer.apply(control_action.joint_efforts, joint_vel, self.effort_limit)
        self.applied_effort = net_effort
        control_action.joint_efforts = net_effort
        return control_action


class RndEquivalentActuator(RndTorqueRandomizedPDActuator):
    """Explicit PD actuator preceded by the measured equivalent command path."""

    cfg: RndEquivalentActuatorCfg

    def __init__(self, cfg: RndEquivalentActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        model = load_rnd_actuator_model(
            cfg.model_path,
            self.joint_names,
            require_sim_replay_validation=not cfg.allow_unvalidated_model,
            require_command_path_seed=not cfg.allow_unresolved_joints,
        )
        self._validate_controller_seed(model)
        model_hz = float(model["physics_hz"])
        if abs(model_hz - float(cfg.physics_hz)) > 1.0e-6:
            raise ValueError(
                f"Actuator model physics_hz={model_hz} does not match adapter physics_hz={cfg.physics_hz}."
            )
        self.command_path = StatefulCommandPath(
            model=model,
            joint_names=self.joint_names,
            num_envs=self._num_envs,
            device=self._device,
            step_hz=cfg.physics_hz,
            seed=cfg.random_seed,
            sample_randomization=cfg.sample_randomization,
        )
        self._command_path_initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

    def _validate_controller_seed(self, model: dict) -> None:
        checks = (
            ("stiffness", "stiffness"),
            ("damping", "damping"),
            ("effort_limit", "effort_limit_nm"),
            ("velocity_limit", "velocity_limit_rad_s"),
            ("armature", "armature"),
        )
        for actuator_field, model_field in checks:
            expected = torch.tensor(
                [model["joints"][name]["controller_seed"][model_field] for name in self.joint_names],
                dtype=torch.float32,
                device=self._device,
            )
            actual = getattr(self, actuator_field)
            if not torch.allclose(actual, expected.unsqueeze(0).expand_as(actual), rtol=0.0, atol=1.0e-6):
                raise ValueError(
                    f"RndEquivalentActuator {actuator_field} does not match controller_seed {model_field}: "
                    f"expected={expected.tolist()}, actual={actual[0].tolist()}."
                )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Mark reset environments for target-seeded initialization on next compute."""

        ids = self._normalize_env_ids(env_ids)
        super().reset(ids)
        self._command_path_initialized[ids] = False

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("RndEquivalentActuator requires joint position targets.")
        pending = torch.nonzero(~self._command_path_initialized, as_tuple=False).flatten()
        if pending.numel() > 0:
            self.command_path.reset(control_action.joint_positions[pending], env_ids=pending)
            self._command_path_initialized[pending] = True
        control_action.joint_positions = self.command_path.transform(control_action.joint_positions)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class RndTorqueRandomizedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for explicit PD plus training-time torque uncertainty."""

    class_type: type = RndTorqueRandomizedPDActuator
    torque_randomization_model_path: str = MISSING
    random_seed: int = 0
    sample_randomization: bool = True


@configclass
class RndEquivalentActuatorCfg(RndTorqueRandomizedPDActuatorCfg):
    """Configuration for :class:`RndEquivalentActuator`."""

    class_type: type = RndEquivalentActuator
    model_path: str = MISSING
    physics_hz: float = 200.0
    allow_unvalidated_model: bool = False
    allow_unresolved_joints: bool = False
