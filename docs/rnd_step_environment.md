# RND STEP Environment

This document describes the maintained RND STEP scope of this fork.

## Supported Tasks

| Environment ID | Purpose |
| --- | --- |
| `RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0` | Actuator-aware task with the CMP10A observation model |
| `RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0` | Current actuator-aware sim-to-real walking task |
| `RNDLab-Isaac-Velocity-Flat-RND-Step-v0` | Implicit-actuator flat-ground baseline |
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

The actuator-aware task instead loads the replay-validated per-joint command
paths from `scripts/tools/config/rnd_actuator_model_runtime.json`. Its stateful
actuator samples the measured residual-delay, position-bias, and play-threshold
ranges independently per environment where a measured range is available.

It also loads `scripts/tools/config/rnd_armature_randomization.json` and writes
one absolute armature sample to PhysX for each environment at startup. The
sample remains fixed across episode resets. `R_Leg_hip_pitch`, `R_Leg_knee`,
and `L_Leg_knee` use quality-passing dynamic-identification ranges; the other
nine joints use the explicitly marked `0.005-0.04 kg*m^2` training prior. One
normalized quantile is shared across all 12 joints within an environment, so
the randomization represents a common hardware/identification condition rather
than twelve unrelated disturbances. These values are training uncertainty, not
fixed physical-parameter promotion.

## Actuator Model Status

The current promoted command-path runtime model has all 12 leg joints in
`integration_joint_names` and no fallback joints. Every integrated joint passed
the command-path quality gate and fixed-base Isaac replay. The task enforces
200 Hz simulation physics, 50 Hz policy execution, and the promoted per-joint
stiffness, damping, effort limit, velocity limit, and nominal armature seeds.
The separate startup event then replaces the PhysX armature with the sampled
training value without changing the validated controller seed.

The underlying measurements were collected with the upper body rigidly fixed
and the feet suspended. They identify an encoder-domain equivalent response,
not separately measured gear backlash, joint friction torque, or loaded ground
contact behavior. Four gravity-observable joints passed the low-current torque
calibration. The opt-in actuator task samples their measured Coulomb-torque
ranges and uses a broad zero-inclusive uncertainty prior for the eight rejected
joints. This is a training domain-randomization layer, not a claim that the
rejected joints were measured. Mirrored joints share the same sampled quantile,
which is mapped through each joint's own measured or prior range. This preserves
the supported left/right parameter differences without injecting an unrelated
independent strength mismatch into every episode. The baseline Flat task remains
unchanged.

## CMP10A Observation Model

The opt-in `Actuator-IMU` task replaces only the actor's angular-velocity and
projected-gravity terms. The critic still receives the clean simulator values,
and the observation order and existing scales remain unchanged. The model is
promoted from the passing static and dynamic captures into
`scripts/tools/config/rnd_cmp10a_runtime.json`.

At runtime the physical sensor path uses the user-confirmed aligned-mount
mapping `diag(-1, -1, +1)`, subtracts the measured gyro bias in the sensor
frame, and converts CMP10A Euler roll/pitch into a unit projected-gravity vector
in `base_link`. `robot_lab.hardware.CMP10ARuntimeSource` reads the roughly
200 Hz stream in a read-only background thread; a 50 Hz controller calls
`snapshot()` to obtain the latest coherent gyro/Euler pair. Stale frames,
future timestamps, and excessive pair skew fail closed.

The simulated actor observation samples one fixed sensor condition per episode:
gyro sample age `0-5 ms`, gyro residual bias `+/-0.01 rad/s` per axis, gyro
white-noise sigma `0.0003-0.003 rad/s`, orientation delay `0-20 ms`, and
projected-gravity tangent-angle noise sigma `0.00005-0.002 rad`. These are
explicit training envelopes around the measured behavior, not measured
confidence intervals. Axis signs and static gravity gain are not randomized.
The checked-in model is authoritative for the actor angular-velocity scale;
the task fails during construction if its observation scale disagrees.

## Temporal Actor Observation

The IMU-aware actor now uses four policy-rate samples for the hidden sensor and
actuator state that cannot be inferred from one frame. At 50 Hz this covers the
current sample plus 60 ms of history. Isaac Lab flattens each term from oldest
to newest and concatenates the terms in this order:

1. CMP10A angular velocity: `4 x 3 = 12` values;
2. CMP10A projected gravity: `4 x 3 = 12` values;
3. current velocity command: `3` values;
4. Dynamixel position/velocity sample: `4 x 24 = 96` values;
5. previous action: `4 x 12 = 48` values.

The actor input is therefore 171 values. On reset, all four slots are filled
with the current value; no artificial zero-history transient is introduced.
The critic remains a current-state privileged observation and is not expanded.
Consequently, older 45-value actor checkpoints are not compatible with this
task and a fresh training run is required.

The 24-value encoder term is ordered as 12 relative positions followed by 12
scaled velocities. It applies MX-106 position and velocity quantization, one
episode-persistent zero offset per joint, a shared position/velocity sample age
per joint, previous/current interpolation, and policy-step zero-order hold.
The offset `+/-0.005 rad` and age `0-5 ms` envelopes are training priors rather
than measurements.

## Touchdown Objective And Telemetry

The IMU-aware task observes contact and foot vertical velocity at every 200 Hz
physics update. A touchdown is valid only after at least 60 ms airborne. During
walking commands, downward speed above `0.25 m/s` receives a small linear hinge
penalty that reaches its cap at `0.75 m/s`; standing commands are excluded. The
monitor handles independent and simultaneous foot contacts and snapshots a
terminal event before an environment reset.

The evaluation telemetry is written as chunked, pickle-free NPZ data at 200 Hz.
This preserves the pre-impact sample that a 50 Hz policy log can miss and keeps
the reward calculation separate from the recorded behavior metrics. Evaluation
uses the monitor's exact 200 Hz first-contact event, preceding air time, and
pre-impact speed directly; it does not reconstruct touchdown from a downsampled
50 Hz contact edge. The telemetry path also attaches the shared monitor itself
when the reward term is absent, so metric collection does not depend on reward
configuration.

## Command Transitions

Training retains the original random command process in 70% of environments.
The remaining environments sample one deterministic time structure per episode:

- 15% use stand `0-2 s`, translate `2-8 s`, then stop;
- 15% use stand `0-2 s`, translate `2-6 s`, turn while translating `6-10 s`,
  translate again `10-14 s`, then stop.

The turn phase enforces at least `0.35 rad/s` absolute yaw command while keeping
the existing command limits and ramp rates. This changes temporal coverage, not
the command envelope.

## Fixed Evaluation Suite

`scripts/reinforcement_learning/rsl_rl/config/rnd_step_actuator_imu_eval_v1.json`
defines separate validation and held-out test domains. Every domain fixes and
reads back material, mass, COM, actuator state, per-joint encoder offset/age,
and CMP10A parameters. Legacy velocity kicks and external-wrench events are
disabled so the only disturbance in a pulse case is the declared force pulse.
The pulse is applied at the base COM for exactly 24 physics ticks (`0.12 s`) as
`F = total_mass * requested_delta_velocity / 0.12`, so changing robot mass does
not silently change the intended velocity impulse. Delivery ticks are recorded
per environment. If an episode resets before all 24 ticks are delivered, that
episode is marked as partial delivery and push recovery is right-censored rather
than reported as a recovery success or failure.

Evaluation reports reward-independent survival/fall, yaw-frame linear speed
RMSE, yaw-rate RMSE, gait timing and touchdown symmetry, foot progress/taps,
torque-saturation dwell, and push-recovery time. Episodes are weighted equally
inside a case and cases are weighted equally in the split summary.

Run validation on a newly trained 171-value checkpoint with:

```bash
python scripts/reinforcement_learning/rsl_rl/evaluate.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 \
  --suite=scripts/reinforcement_learning/rsl_rl/config/rnd_step_actuator_imu_eval_v1.json \
  --split=validation --checkpoint=/path/to/model.pt \
  --num_envs=64 --output=logs/rsl_rl/rnd_step/evaluation/validation --headless
```

The checked-in test split is intentionally locked until a new 171-value
checkpoint is selected and its path and SHA-256 are frozen in the suite. This
prevents using the held-out cases while choosing a checkpoint. Evaluation also
checks the checkpoint actor input dimension before constructing the runner or
environment, so an older 45-value checkpoint fails immediately.

## Stable Walking Rewards

The flat task rewards and penalties are biased toward:

- stable base orientation;
- reduced upper-body tilt;
- reduced upper-body linear acceleration;
- velocity tracking in the STEP yaw frame;
- controlled turning behavior;
- sufficient lateral foot spacing;
- a fore-aft foot-link center aligned with the default-pose FK target;
- a low gait-cycle average of the signed right/left fore-aft foot separation;
- a minimum command-direction foot pass at each ordered right/left touchdown;
- reduced foot collision risk;
- reduced foot sliding;
- left/right gait timing balance;
- standing still when commanded velocity is near zero.

The reward profile is meant to prefer a practical, low-sway walking gait over a
fast but unstable gait. The actuator task additionally strengthens straight-foot
heading and hip-yaw neutrality. Its runner defaults to 6,500 iterations so a
long CLI override does not remain the normal configuration after late-policy
regression was observed in a 10,000-iteration run.

The command sampler reserves explicit zero-yaw translation environments so the
spatial foot-pass and fore-aft balance terms receive enough straight-walking
training samples without narrowing the full yaw-command range.

## Play-Time Checks

Play scripts keep environment randomization and push events enabled. This is
intentional: playback should expose whether the policy remains stable under the
same disturbance classes used in training.

Observation corruption remains disabled by default during play so debugging is
focused on environment events and policy behavior rather than sensor-noise
effects. Pass `--enable_observation_corruption` to sample the CMP10A envelope
and other policy observation corruption during a robustness check.

Playback validates the checkpoint run's saved `experiment_name` against the
selected task before constructing the environment. This prevents an
actuator-trained checkpoint from being evaluated accidentally in the implicit
actuator baseline. Intentional cross-task tests require
`--allow_task_mismatch`.

## Hardware Deployment Status

The policy observation excludes base linear velocity and the flat task removes
the terrain height scan. The remaining policy input can be assembled from a
calibrated CMP10A IMU, Dynamixel joint position/velocity, the velocity command,
and the previous policy action. The hardware implementation must reproduce the
training observation order, signs, scales, default offsets, normalization, and
50 Hz timing exactly.

`scripts/reinforcement_learning/rsl_rl/play.py` exports JIT and ONNX policies,
but a complete hardware inference and Dynamixel command runner is not yet
implemented. The CMP10A runtime adapter supplies the two IMU-derived policy
inputs but does not execute the policy or command motors. The measured actuator
command-path transform is simulation-only and must not be applied to physical
servo targets a second time. Hardware validation must begin in a safety fixture
with pose hold and standing before progressing to loaded stepping and walking.

The current IMU evidence does not measure absolute sensor-to-host latency,
temperature drift, walking-vibration behavior, or an externally referenced
level offset. The dynamic trial measured only the relative timing and
consistency of the Euler and gyro streams.

## Known Validation Limitations

Static Python checks can validate imports and syntax, but full environment
validation requires an Isaac Sim/Isaac Lab runtime with `pxr` available.

Recommended runtime checks:

```bash
python scripts/tools/list_envs.py | grep RND-Step
python scripts/tools/zero_agent.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0 --num_envs 4
python scripts/reinforcement_learning/rsl_rl/play.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0 --num_envs 16
python scripts/tools/zero_agent.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 --num_envs 4
python scripts/reinforcement_learning/rsl_rl/play.py --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 --num_envs 16 --enable_observation_corruption
```

Start a fresh IMU-aware training run with:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 \
  --headless --num_envs=4096 --max_iterations=6500
```
