# rnd_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

`rnd_lab` is a focused public fork of
[fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) for the non-4bar
RND STEP humanoid reinforcement-learning environment.

The maintained scope of this fork is the RND STEP humanoid asset, Isaac Lab
velocity-tracking tasks, sim-to-real domain randomization, and training/play
workflows. Upstream non-RND robot assets and task configs are excluded from the
public fork scope.

## Maintained Environments

| Robot | Terrain | Environment ID | Status |
| --- | --- | --- | --- |
| RND STEP humanoid, non-4bar | Flat plane | `RNDLab-Isaac-Velocity-Flat-RND-Step-v0` | Primary development target |
| RND STEP humanoid, non-4bar | Rough terrain | `RNDLab-Isaac-Velocity-Rough-RND-Step-v0` | Base config, lower priority |

The flat environment is the main sim-to-real development path. It keeps the
play-time randomization events enabled so that trained policies can be checked
under the same disturbance/randomization classes used during training.

## Fork Changes

Compared with upstream `robot_lab`, this fork focuses on:

- RND STEP non-4bar robot integration with STEP URDF/CSV metadata.
- Fixed waist joint modeling for the current STEP hardware assumption.
- STEP-specific actuator/action configuration and joint defaults.
- Flat-environment randomization for friction, mass, COM, reset state, external
  force/torque, and push events.
- Stable walking rewards that penalize upper-body tilt, upper-body vibration,
  foot collision risk, over-narrow stance, excessive hip yaw, and left/right
  gait timing imbalance.
- Play scripts that preserve environment randomization and push events for
  inspection while keeping observation corruption disabled.
- RND STEP agent configuration that is independent from other robot agent
  config modules.

More detail is recorded in [CHANGELOG.md](CHANGELOG.md) and
[docs/rnd_step_environment.md](docs/rnd_step_environment.md).

## Installation

Install Isaac Lab first by following the official Isaac Lab installation guide.
This fork expects the same Isaac Lab/Isaac Sim environment used by upstream
`robot_lab`.

Clone the fork outside the Isaac Lab repository:

```bash
git clone https://github.com/bounceonme/rnd_lab.git
cd rnd_lab
```

Install the extension with the Python interpreter that has Isaac Lab available:

```bash
python -m pip install -e source/robot_lab
```

The distribution name is `rnd_lab`. The internal Python module path remains
`robot_lab` to keep the upstream Isaac Lab extension imports stable.

Verify that the STEP tasks are registered:

```bash
python scripts/tools/list_envs.py | grep RND-Step
```

## Training And Play

Train the primary flat RND STEP policy with RSL-RL:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 \
  --headless
```

Play a trained policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 \
  --num_envs 32
```

Play with keyboard command control:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 \
  --num_envs 1 \
  --keyboard
```

CusRL configs are also kept for the STEP tasks:

```bash
python scripts/reinforcement_learning/cusrl/train.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-v0 \
  --headless
```

## RND STEP Layout

The maintained files for this fork are concentrated in these paths:

```text
source/robot_lab/data/Robots/rnd/step/
source/robot_lab/robot_lab/assets/rnd.py
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/rnd_step/
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py
scripts/reinforcement_learning/rsl_rl/play.py
scripts/reinforcement_learning/cusrl/play.py
scripts/reinforcement_learning/rsl_rl/play_cs.py
```

The flat STEP environment is intentionally split into small files:

- `flat_env_cfg.py`: Isaac Lab flat-terrain environment assembly.
- `flat_domain_randomization.py`: sim-to-real randomization and disturbances.
- `flat_behavior_cfg.py`: stable walking rewards, command ranges, and
  terminations.
- `offline_ground_plane.py`: optional offline ground-plane patch.

## Attribution

This project is derived from
[fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab), originally authored
by Ziqi Fan and released under the Apache License 2.0.

Please cite the upstream project if you use this code or parts of it:

```bibtex
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

## License

This fork keeps the upstream Apache License 2.0 license. Modified files should
retain upstream copyright and license headers where present.
