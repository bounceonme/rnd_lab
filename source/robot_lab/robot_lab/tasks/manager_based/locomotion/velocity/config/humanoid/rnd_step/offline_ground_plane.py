# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os


def _spawn_offline_ground_plane(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, PhysicsSchemaTools

    from isaaclab.sim.utils import (
        add_labels,
        bind_physics_material,
        get_current_stage,
        get_first_matching_child_prim,
        set_prim_visibility,
    )

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    z_position = 0.0 if translation is None else translation[2]
    color = cfg.color if getattr(cfg, "color", None) is not None else (0.0, 0.0, 0.0)
    size = float(max(getattr(cfg, "size", (100.0, 100.0))))
    PhysicsSchemaTools.addGroundPlane(stage, prim_path, "Z", size, Gf.Vec3f(0.0, 0.0, z_position), Gf.Vec3f(*color))

    if getattr(cfg, "physics_material", None) is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)
        collision_prim = get_first_matching_child_prim(
            prim_path,
            predicate=lambda child_prim: child_prim.GetTypeName() == "Plane",
            stage=stage,
        )
        if collision_prim is not None:
            bind_physics_material(str(collision_prim.GetPath()), f"{prim_path}/physicsMaterial", stage=stage)

    prim = stage.GetPrimAtPath(prim_path)
    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
        for semantic_type, semantic_value in cfg.semantic_tags:
            add_labels(
                prim,
                labels=[semantic_value.replace(" ", "_")],
                instance_name=semantic_type.replace(" ", "_"),
            )
    set_prim_visibility(prim, cfg.visible)
    return prim


def maybe_enable_offline_ground_plane() -> None:
    if os.environ.get("ROBOT_LAB_FORCE_OFFLINE_GROUND_PLANE") != "1":
        return

    import isaaclab.sim as sim_utils

    if sim_utils.GroundPlaneCfg.func is not _spawn_offline_ground_plane:
        sim_utils.GroundPlaneCfg.func = _spawn_offline_ground_plane
