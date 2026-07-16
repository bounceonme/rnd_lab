"""Opt-in RND STEP articulation with command-path and joint-torque randomization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_lab.actuators.rnd_isaac import RndEquivalentActuatorCfg, RndTorqueRandomizedPDActuatorCfg
from robot_lab.actuators.rnd_stateful import load_rnd_actuator_model
from robot_lab.assets.rnd import STEP_CFG


_REPO_ROOT = Path(__file__).resolve().parents[4]
RND_ACTUATOR_RUNTIME_MODEL_PATH = _REPO_ROOT / "scripts" / "tools" / "config" / "rnd_actuator_model_runtime.json"
RND_TORQUE_RANDOMIZATION_PATH = _REPO_ROOT / "scripts" / "tools" / "config" / "rnd_torque_randomization.json"
RND_ARMATURE_RANDOMIZATION_PATH = _REPO_ROOT / "scripts" / "tools" / "config" / "rnd_armature_randomization.json"


def _controller_values(model: dict[str, Any], joint_names: tuple[str, ...], field: str) -> dict[str, float]:
    return {name: float(model["joints"][name]["controller_seed"][field]) for name in joint_names}


def build_step_actuator_cfg(model_path: str | Path = RND_ACTUATOR_RUNTIME_MODEL_PATH):
    """Build a separate STEP config without mutating the baseline ``STEP_CFG``."""

    model_path = Path(model_path).expanduser().resolve()
    model = load_rnd_actuator_model(model_path)
    integration_joint_names = tuple(model["integration_joint_names"])
    fallback_joint_names = tuple(model["fallback_joint_names"])
    model = load_rnd_actuator_model(
        model_path,
        integration_joint_names,
        require_sim_replay_validation=True,
        require_command_path_seed=True,
    )
    if fallback_joint_names not in ((), ("L_Leg_ankle_roll",)):
        raise ValueError(
            "STEP actuator integration permits either full validation or only L_Leg_ankle_roll as fallback; "
            f"got {fallback_joint_names}."
        )

    stateful_actuator = RndEquivalentActuatorCfg(
        joint_names_expr=list(integration_joint_names),
        model_path=str(model_path),
        torque_randomization_model_path=str(RND_TORQUE_RANDOMIZATION_PATH),
        physics_hz=float(model["physics_hz"]),
        random_seed=0,
        sample_randomization=True,
        allow_unvalidated_model=False,
        allow_unresolved_joints=False,
        stiffness=_controller_values(model, integration_joint_names, "stiffness"),
        damping=_controller_values(model, integration_joint_names, "damping"),
        effort_limit=_controller_values(model, integration_joint_names, "effort_limit_nm"),
        velocity_limit=_controller_values(model, integration_joint_names, "velocity_limit_rad_s"),
        armature=_controller_values(model, integration_joint_names, "armature"),
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    actuators = {"replay_validated": stateful_actuator}
    if fallback_joint_names:
        actuators["left_ankle_roll_fallback"] = RndTorqueRandomizedPDActuatorCfg(
            joint_names_expr=list(fallback_joint_names),
            torque_randomization_model_path=str(RND_TORQUE_RANDOMIZATION_PATH),
            random_seed=0,
            sample_randomization=True,
            stiffness=_controller_values(model, fallback_joint_names, "stiffness"),
            damping=_controller_values(model, fallback_joint_names, "damping"),
            effort_limit=_controller_values(model, fallback_joint_names, "effort_limit_nm"),
            velocity_limit=_controller_values(model, fallback_joint_names, "velocity_limit_rad_s"),
            armature=_controller_values(model, fallback_joint_names, "armature"),
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        )
    return STEP_CFG.replace(actuators=actuators)


STEP_ACTUATOR_CFG = build_step_actuator_cfg()
