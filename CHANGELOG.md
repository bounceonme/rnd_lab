# Changelog

## Unreleased

### Added

- Added a focused public-fork README for the non-4bar RND STEP humanoid environment.
- Added public-fork scope, attribution, and publication checklist documentation.
- Added STEP flat-environment helper modules for behavior rewards, domain randomization, and optional offline ground-plane setup.
- Added STEP-specific reward terms for upper-body stability, lateral stance width, foot collision risk, yaw-gated hip posture, and left/right gait timing balance.

### Changed

- Converted the STEP waist joint assumption from actuated revolute behavior to fixed behavior for the current hardware target.
- Updated STEP URDF/CSV metadata and `robot_lab.assets.rnd` so the action/default/actuator configuration matches the fixed-waist model.
- Split the RND STEP flat config so terrain setup, randomization, rewards, terminations, and commands have clear responsibilities.
- Kept play-time randomization events enabled in RSL-RL/CusRL play scripts so force/push/randomization behavior can be inspected during playback.
- Tuned the flat STEP command and reward profile toward stable walking with reduced upper-body tilt, reduced vibration, and safer foot spacing.
- Made RND STEP agent configs independent from other humanoid robot config modules.
- Renamed the public project/distribution identity to `rnd_lab` while keeping the internal Python module path as `robot_lab` for compatibility.
- Removed upstream non-RND robot assets, task configs, and screenshots from the public fork index.

### Scope

- The maintained public-fork target is `RNDLab-Isaac-Velocity-Flat-RND-Step-v0`.
- `RNDLab-Isaac-Velocity-Rough-RND-Step-v0` is retained as a base rough-terrain config.
- Upstream non-RND robot environments are not included as maintained fork targets.
