# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0


def apply_step_flat_domain_randomization(env_cfg) -> None:
    """Apply STEP flat-environment sim-to-real randomization.

    Actuator gain randomization is intentionally left disabled here because actuator
    behavior is tuned separately from the rest of the domain randomization.
    """

    env_cfg.events.randomize_rigid_body_material.params["static_friction_range"] = (0.55, 1.15)
    env_cfg.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.40, 0.90)
    env_cfg.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.15)
    env_cfg.events.randomize_rigid_body_material.params["make_consistent"] = True

    # The IMU-equipped URDF totals 5.3407113 kg while the assembled robot measured
    # 5.646 kg. Treat the remaining difference as uncertain upper-body payload merged into base_link.
    measured_mass_offset = 5.646 - 5.3407113
    env_cfg.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (
        measured_mass_offset - 0.11,
        measured_mass_offset + 0.11,
    )
    env_cfg.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)

    env_cfg.events.randomize_com_positions.params["asset_cfg"].body_names = [env_cfg.base_link_name]
    env_cfg.events.randomize_com_positions.params["com_range"] = {
        "x": (-0.015, 0.015),
        "y": (-0.015, 0.015),
        "z": (-0.010, 0.010),
    }

    # This reset event applies a persistent wrench for the whole episode. Keep it
    # below the level that makes a continuously leaned posture the optimal gait;
    # interval velocity pushes still provide larger transient disturbances.
    env_cfg.events.randomize_apply_external_force_torque.params["force_range"] = (-1.0, 1.0)
    env_cfg.events.randomize_apply_external_force_torque.params["torque_range"] = (-0.25, 0.25)
    env_cfg.events.randomize_actuator_gains = None

    env_cfg.events.randomize_push_robot.interval_range_s = (10.0, 15.0)
    env_cfg.events.randomize_push_robot.params["velocity_range"] = {
        "x": (-0.20, 0.20),
        "y": (-0.20, 0.20),
        "yaw": (-0.18, 0.18),
    }

    env_cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.010, 0.010),
        "roll": (-0.025, 0.025),
        "pitch": (-0.025, 0.025),
        "yaw": (-3.14, 3.14),
    }
    env_cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (-0.10, 0.10),
        "y": (-0.10, 0.10),
        "z": (0.0, 0.0),
        "roll": (-0.08, 0.08),
        "pitch": (-0.08, 0.08),
        "yaw": (-0.15, 0.15),
    }
