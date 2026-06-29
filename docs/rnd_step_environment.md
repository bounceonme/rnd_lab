# RND STEP Environment

This document describes the maintained RND STEP scope of this fork.

## Supported Tasks

| Environment ID | Purpose |
| --- | --- |
| `RNDLab-Isaac-Velocity-Flat-RND-Step-v0` | Primary flat-ground sim-to-real walking task |
| `RNDLab-Isaac-Velocity-Rough-RND-Step-v0` | Rough-terrain base config retained for later work |

The non-4bar STEP model is the maintained robot target. The 4bar model and
other upstream robots are outside the public fork scope unless a future change
states otherwise.

## Robot Asset Scope

Maintained STEP asset paths:

```text
source/robot_lab/data/Robots/rnd/step/urdf/step.urdf
source/robot_lab/data/Robots/rnd/step/urdf/step.csv
source/robot_lab/data/Robots/rnd/step/meshes/
source/robot_lab/robot_lab/assets/rnd.py
```

The current hardware assumption fixes the waist joint. The STEP asset config
therefore removes waist control from action/default/actuator configuration.

## Flat Environment Structure

The flat STEP environment is assembled in:

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/rnd_step/flat_env_cfg.py
```

The implementation is intentionally split by responsibility:

| File | Responsibility |
| --- | --- |
| `flat_env_cfg.py` | Flat terrain assembly and STEP-specific wiring |
| `flat_domain_randomization.py` | Sim-to-real randomization and disturbances |
| `flat_behavior_cfg.py` | Stable walking rewards, command ranges, and terminations |
| `offline_ground_plane.py` | Optional offline ground-plane patch |

This keeps `flat_env_cfg.py` from becoming a large mixed-responsibility config.

## Randomization And Disturbances

The flat task includes randomization for:

- ground friction and restitution;
- base and non-base rigid-body mass;
- base center of mass;
- reset pose and reset velocity;
- external force/torque on the base link;
- push events through randomized base velocity.

Actuator gain randomization is intentionally disabled in the current profile
because actuator behavior is expected to be tuned separately.

## Stable Walking Rewards

The flat task rewards and penalties are biased toward:

- stable base orientation;
- reduced upper-body tilt;
- reduced upper-body linear acceleration;
- velocity tracking in the STEP yaw frame;
- controlled turning behavior;
- sufficient lateral foot spacing;
- reduced foot collision risk;
- reduced foot sliding;
- left/right gait timing balance;
- standing still when commanded velocity is near zero.

The reward profile is meant to prefer a practical, low-sway walking gait over a
fast but unstable gait.

## Play-Time Checks

Play scripts keep environment randomization and push events enabled. This is
intentional: playback should expose whether the policy remains stable under the
same disturbance classes used in training.

Observation corruption remains disabled during play so debugging is focused on
environment events and policy behavior rather than sensor-noise effects.

## Known Validation Limitations

Static Python checks can validate imports and syntax, but full environment
validation requires an Isaac Sim/Isaac Lab runtime with `pxr` available.

Recommended runtime checks:

```bash
python scripts/tools/list_envs.py | grep RND-Step
python scripts/tools/zero_agent.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 --num_envs 4
python scripts/reinforcement_learning/rsl_rl/play.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 --num_envs 16
```
