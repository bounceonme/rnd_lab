# RND STEP Real-to-Sim Identification

This is a standalone actuator-identification system. It does not import Isaac Lab, start Omniverse, or modify
`rnd_joint_coordinate_test.py`. The only shared input is the verified motor ID, zero, direction, and raw-limit mapping
in `scripts/tools/config/rnd_dynamixel.toml`.

## Measurement boundary

The test uses MX-106(2.0) Goal Position and onboard feedback only:

- Present Position and Velocity
- Present Current and PWM
- Position and Velocity Trajectory
- Realtime Tick, voltage, temperature, and moving state

It identifies a closed-loop, encoder-domain equivalent model. Without an external output encoder or torque sensor,
gear backlash, static friction, structural compliance, encoder quantization, and servo-loop behavior are not uniquely
separable. Older per-run fit files retain a nominal torque/current proxy for diagnostics, but that proxy is not a
measured joint torque and is not exported into the runtime actuator model.

No CMP10A IMU is used. That is valid only while the upper body is rigidly fixed by an external fixture. A flexible rope
or a suspension that allows torso rotation invalidates the assumption and requires base-motion measurement.

## Safety behavior

- Opening the bus verifies MX-106(2.0) model 321 and Position Control Mode 3.
- Every mapped motor is forced torque OFF during startup, errors, interrupts, and shutdown.
- Torque ON first reads Present Position and writes the same value to Goal Position.
- All 12 motors then move together from their measured positions to the configured RL reference pose at 8 deg/s.
- The reference transition uses stricter current, PWM, and tracking-error limits than the excitation phase.
- Excitation starts only after every joint settles within 1.5 degrees of its configured reference angle.
- Commands use GroupSyncWrite and telemetry uses one contiguous GroupSyncRead block for all 12 motors.
- Raw limits, URDF limits, current position, maximum excursion, command step, tracking error, current, PWM, voltage,
  temperature, hardware error status, stale device ticks, and repeated control-loop deadline misses are checked.
- A safety violation immediately requests torque OFF. It does not attempt a powered return trajectory after a fault.
- Once identification recording begins, an aborted run is saved with `status="aborted"` and cannot be fitted unless
  explicitly allowed for diagnostics. A failure during automatic positioning produces no identification dataset.

The reference transition takes up to roughly 13 seconds from the allowed starting range. The default profiles then
take roughly 94 seconds per excited joint, or about 19 minutes for all 12 joints.

## Dry run

Use the project Isaac Python environment because it already contains NumPy and Dynamixel SDK:

```bash
python scripts/tools/rnd_real2sim_collect.py \
  --dry-run \
  --joint R_Leg_knee \
  --output /tmp/rnd_knee_dry.npz

python scripts/tools/rnd_real2sim_fit.py \
  /tmp/rnd_knee_dry.npz \
  --allow-dry-run
```

Synthetic output verifies the pipeline only. The fit command refuses it without `--allow-dry-run` so it cannot be
mistaken for measured parameters.

## Hardware collection

Rigidly fix the upper body, suspend both feet with full clearance, power the robot, and begin with one joint:

```bash
python scripts/tools/rnd_real2sim_collect.py \
  --joint R_Leg_knee \
  --enable-hardware \
  --confirm-rigidly-fixed \
  --confirm-clearance
```

The program opens `/dev/ttyUSB0`, disables torque, and prints every measured angle together with its configured target
and required move. The three hardware safety flags are the explicit operator authorization, so there is no additional
interactive arming prompt. After validation, it re-reads the joint positions, seeds all goals at the measured positions,
enables all 12 motors, moves them slowly to the reference pose, verifies the settled pose, and only then starts
excitation. Review the fixture and full motion clearance before launching the command, and keep a physical power cut-off
within reach.

Profiles may define `precondition_cycles`. These cycles use the same bounded waveform and safety checks but are recorded
with phase `-1`, so they settle load-dependent slack without entering actuator identification. The default
`micro_triangle` runs one preconditioning cycle, settles at the reference pose, and then records six identification
cycles.

Automatic positioning is safe only after every `zero_raw` and `direction` in `rnd_dynamixel.toml` has been verified
against the URDF coordinates. Joint-space interpolation cannot detect physical fixture or self-collision, so the
clearance confirmation must include the entire path from the displayed starting pose to the reference pose.

Run all joints after the one-joint current, temperature, clearance, and timing results are acceptable:

```bash
python scripts/tools/rnd_real2sim_collect.py \
  --enable-hardware \
  --confirm-rigidly-fixed \
  --confirm-clearance
```

The default is 50 Hz because the measured 12-servo read/write path sustained about 62.5 Hz. If 50 Hz still produces
repeated deadline misses, use the observed-throughput recommendation printed by the collector; do not loosen the
deadline shutdown to hide an overloaded bus.

Inspect controller settings without enabling torque or moving the robot:

```bash
python scripts/tools/rnd_real2sim_collect.py \
  --inspect-runtime-only \
  --enable-hardware \
  --confirm-rigidly-fixed \
  --confirm-clearance
```

This mode prints the raw Position PID, feedforward, PWM/current limit, velocity limit, and homing-offset registers;
normal collection also stores them in dataset metadata. Use the snapshot to rule out controller-configuration differences
before interpreting a side-specific response as mechanical.

## Identification and training artifact

```bash
python scripts/tools/rnd_real2sim_fit.py logs/rnd_real2sim/rnd_real2sim_YYYYMMDD_HHMMSS.npz
```

The JSON output contains, per joint:

- the configured reference pose, settled position, and automatic-transition diagnostics copied from the dataset
- phase-equivalent command delay and gain from cycle-by-cycle sine frequency response
- delay-compensated rising/falling hysteresis from the low-speed triangle profile
- signed command-minus-position center bias from the same branch-paired triangle cycles
- Coulomb-like current from positive/negative motion at matched positions in each slow-sine cycle
- explicit quality gates that suppress randomization ranges when repeatability or fit quality fails
- conservative delay, backlash, and Coulomb-current randomization ranges

The command-domain fit uses the configured control index, not the median USB read-completion interval. Viscous friction
is left unidentified instead of being inferred from acceleration/current terms that cannot be separated reliably in
this suspended encoder-only experiment.

## Current-domain friction compensation candidate

Build the independent current-domain candidate after regenerating the accepted baseline:

```bash
python scripts/tools/rnd_real2sim_aggregate.py
python scripts/tools/rnd_current_compensation_build.py
```

The generated `scripts/tools/config/rnd_current_compensation_candidate.json` is deliberately analysis-only. It
revalidates every selected model hash and per-run Coulomb-current quality flag, requires at least two repeated runs,
and rejects an aggregate whose `(maximum - minimum) / median` exceeds `0.5`. The current output is

```text
I_ff = gain * I_c * tanh(4 * desired_joint_velocity / transition_velocity)
```

where `I_c` remains in motor-current amperes. The transition velocity is four times the velocity threshold used by
the source identification, making the compensation smooth and zero at rest instead of applying a discontinuous sign
term. The candidate uses desired low-level trajectory velocity, not noisy encoder velocity. It does not identify or
invent viscous friction, static breakaway current, Stribeck behavior, or a current-to-joint-torque conversion.

This artifact cannot be written directly through the existing collector. Collection enforces Position Control Mode 3,
where current is telemetry. MX-106 Current-based Position Control Mode 5 uses Goal Current as a current limit rather
than an additive feedforward input. Direct current compensation therefore requires a separately implemented and
validated external position controller using Current Control Mode 0. Mode changes reset controller/profile values, so
that work must remain separate from identification and start with suspended low-gain bench tests.

Model choices and control-table semantics are documented against:

- [ROBOTIS MX-106T/R(2.0) e-Manual](https://emanual.robotis.com/docs/en/dxl/mx/mx-106-2/)
- [System identification and force estimation of robotic manipulator](https://doi.org/10.1007/s11044-024-10017-1)
- [A New Model for Control of Systems with Friction](https://doi.org/10.1109/9.376053)

## All-joint low-current torque calibration

The PhysX analysis trace stores dynamics for every joint, even though the original command reports one selected
joint. Once that cache exists, analyze every recorded joint without starting Isaac Sim again:

```bash
python scripts/tools/rnd_real2sim_friction_batch.py \
  --dataset logs/rnd_real2sim/all_joints_torque_calibration_01.npz
```

The batch analyzer verifies both source hashes, restricts each fit to the phase that actually excited that joint, and
writes `all_joints_torque_calibration_01_all_joint_torque_calibration.json`. Each joint keeps its own torque constant,
Coulomb-current estimate, confidence interval, and quality reasons. Results that lack enough gravity/current span,
hit an optimizer bound, or fail the fit-quality gates remain rejected. The report is analysis-only and cannot update
the RL actuator model automatically.

Generate the separately gated training randomization after explicitly accepting the broad prior for unidentified
joints:

```bash
python scripts/tools/rnd_torque_randomization_build.py
```

## All-joint dynamic armature experiment

Armature is a residual reflected joint inertia. It must not be inferred from the quasi-static torque-calibration run,
whose acceleration is too small. The dedicated experiment excites all 12 joints sequentially at the same 5 degree
amplitude and at 0.5, 1.0, and 1.5 Hz. Each frequency records ten seconds, so acceleration changes with frequency
squared without changing the gravity range. The complete run records 20,051 samples and takes about 401 seconds after
the automatic reference-pose transition.

Before hardware collection, confirm that the USB serial latency timer is still `1`; reconnects and reboots can reset it:

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

Collect all 12 joints in one command. Do not pass `--joint`:

```bash
python scripts/tools/rnd_real2sim_collect.py \
  --config scripts/tools/config/rnd_real2sim_armature.toml \
  --output logs/rnd_real2sim/all_joints_armature_01.npz \
  --enable-hardware \
  --confirm-rigidly-fixed \
  --confirm-clearance
```

The upper body must be rigidly fixed and every link must have clearance at the 5 degree excursion. The configuration
keeps the measured 50 Hz bus rate; raising it is not required for these frequencies and can reintroduce deadline or
status-packet failures.

After collection, generate the zero-armature PhysX dynamics cache once. The selected joint is required by the existing
friction-report interface, but the resulting NPZ stores URDF dynamics for all 12 joints:

```bash
python scripts/tools/rnd_real2sim_friction.py \
  --dataset logs/rnd_real2sim/all_joints_armature_01.npz \
  --joint R_Leg_hip_pitch \
  --filter-window 7 \
  --headless
```

Then fit every joint without restarting Isaac Sim:

```bash
python scripts/tools/rnd_real2sim_armature_batch.py \
  --dataset logs/rnd_real2sim/all_joints_armature_01.npz
```

The batch model first computes `current-calibrated torque - zero-armature URDF dynamics`. It fits the fundamental
harmonic of every complete cycle, projects out the velocity-aligned component, and then separates residual armature
from a shared position-dependent torque error using the frequency-squared acceleration scaling. The original
sample-domain `J*qdd + friction + bias` fit remains in the report as a diagnostic but does not select the armature.
The selected fit requires three equal-amplitude frequencies, a positive and narrow stratified-cycle bootstrap interval,
consistent estimates at the two frequencies with sufficient acceleration, acceptable tracking, and a previously
passing current-to-torque calibration. The current report passes only `R_Leg_hip_pitch`, `R_Leg_knee`, and
`L_Leg_knee`. The separate `L_Leg_hip_pitch` repeat still fails its harmonic fit and is retained only as failed
evidence. The remaining nine joints are not assigned fabricated measured values.

The source reports remain analysis-only and never edit the URDF, controller seeds, PD gains, an RL checkpoint, or any
runtime file automatically. The following explicit build step converts the three passing fits into measured training
ranges and labels the other nine joints with the reviewed broad prior:

```bash
python scripts/tools/rnd_armature_randomization_build.py
```

This writes `scripts/tools/config/rnd_armature_randomization.json`. The opt-in actuator task samples the config once at
environment startup and writes the absolute values to PhysX. It is domain randomization for policy robustness, not a
claim that the prior-only joints have measured armatures and not a fixed physical-parameter promotion.

The resulting `scripts/tools/config/rnd_torque_randomization.json` is enabled only by the opt-in
`RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0` task. Passing hip-pitch and knee measurements receive per-joint
ranges centered on their measured Coulomb torque. Quality-rejected joints do not use their failed fit values; they
sample a broad `0.0-0.30 Nm` uncertainty prior instead. Every joint also samples a `0.8-1.25` motor-strength scale,
which is reapplied through the original effort limit, and a `2-8 deg/s` smooth Coulomb transition. Viscous friction
and static breakaway remain disabled. The baseline Flat task is unchanged.

When one constant-play width cannot explain repeated amplitudes, fit a common generalized-play path instead of
loosening the single-run `R2 >= 0.95` gate. For the post-alignment left ankle-roll data:

```bash
python scripts/tools/rnd_actuator_multiamplitude.py \
  logs/rnd_real2sim/l_ankle_roll_post_alignment_standard_01.npz \
  logs/rnd_real2sim/l_ankle_roll_post_alignment_standard_02.npz \
  logs/rnd_real2sim/l_ankle_roll_post_alignment_diagnostic_01.npz \
  --joint L_Leg_ankle_roll \
  --minimum-threshold-deg 0.08 \
  --maximum-threshold-deg 2.8 \
  --output logs/rnd_real2sim/l_ankle_roll_post_alignment_multiamplitude_model.json
```

The fitter uses early complete triangle cycles for one common convex linear-plus-play model and reserves the final
cycles of every dataset for validation. It accepts the smallest branch count for which every amplitude has
`R2 >= 0.95` and normalized RMSE at most `0.10`. Dataset and companion-fit hashes are checked again by baseline
aggregation. A passing result is still trace-only evidence: residual delay, residual bias, threshold randomization,
friction torque, and RL integration remain disabled until simulator replay.

For residual-delay calibration, replay every source dataset with the same controller and diagnostic delay override.
The replay path accepts fractional 200 Hz physics steps:

```bash
python scripts/tools/rnd_actuator_sim_replay.py \
  --dataset DATASET.npz \
  --model scripts/tools/config/rnd_actuator_model.json \
  --joint L_Leg_ankle_roll \
  --stiffness 26.25 \
  --damping 1.08 \
  --residual-delay-s 0.004716525306678723 \
  --headless
```

After every source dataset passes, rerun `rnd_actuator_multiamplitude.py` with one `--sim-replay-report REPORT.json`
argument per source dataset. The finalizer rejects missing, duplicated, failed, or controller-inconsistent reports and
stores the common delay value that was directly replayed. Any post-replay total-delay recommendation is retained as a
diagnostic instead of being exported as an unvalidated randomization range. It still leaves RL integration disabled.

`rnd_real2sim.model.EncoderDomainActuatorRandomizer` is the original per-fit analysis primitive. Do not use its nominal
current-to-torque proxy for training. Only gravity-observable joints that pass the all-joint calibration gate have a
supported low-current conversion. The promoted command-path runtime JSON remains independent of that calibration;
the opt-in task consumes it only through the separate `rnd_torque_randomization.json` layer described above. The
current-domain compensation candidate itself does not modify either task.

## Gated runtime actuator model

The accepted per-joint runs are selected explicitly by the baseline manifest and then converted into a separate
runtime seed:

```bash
python scripts/tools/rnd_real2sim_aggregate.py
python scripts/tools/rnd_actuator_build.py
```

The generated `scripts/tools/config/rnd_actuator_model.json` has four deliberate safeguards:

- measured delay is retained as closed-loop evidence, while runtime residual delay starts at zero; simulator response
  must be measured before adding only the residual
- measured center bias is retained as suspended-pose evidence, while runtime residual position bias starts at zero so
  the simulator's existing equilibrium error is not counted twice
- measured current is retained as current-domain evidence and is never converted to torque
- every joint starts with `sim_replay_validated=false`; unresolved command paths use an explicit identity placeholder,
  while a cross-amplitude path remains integration-blocked until its own Isaac replay passes

The pure Torch model applies fractional residual delay, a stateful generalized-play operator, and an additive residual
position bias at the physics rate. It is batched across environments and joints, and partial reset fills delay history
with the current target to avoid an initial zero-target transient.

First check the command-path seed against an accepted hardware trace without Isaac:

```bash
python scripts/tools/rnd_actuator_replay.py \
  logs/rnd_real2sim/rnd_real2sim_YYYYMMDD_HHMMSS.npz
```

This diagnostic cannot satisfy the simulator gate. Run the fixed-base explicit-PD replay separately:

```bash
python scripts/tools/rnd_actuator_sim_replay.py \
  --dataset logs/rnd_real2sim/rnd_real2sim_YYYYMMDD_HHMMSS.npz \
  --headless
```

The Isaac replay writes a simulator trace and JSON report, compares response gain and phase-equivalent delay against
hardware, and reports a non-negative residual-delay candidate. It never edits the model or training configuration.
When the simulator is slower than hardware, sweep the explicit-PD seed automatically:

```bash
python scripts/tools/rnd_actuator_sim_replay.py \
  --dataset logs/rnd_real2sim/rnd_real2sim_YYYYMMDD_HHMMSS.npz \
  --sweep-pd \
  --headless
```

The sweep evaluates the configured `Kp`/`Kd` scale grid on the fastest sine phase. It first considers candidates that
already pass the response, gain, saturation, and delay gates, then selects the smallest change from the existing seed
with lower effort and gain as tie-breakers. Non-negative residual-delay compensation is considered only when no
candidate already passes the delay gate. The selected pair is replayed over the complete trace. If that replay fails
only because of a repeatable constant trajectory offset, the tool calibrates one residual position bias from the
triangle phase and replays the complete trace again. Calibration is refused when a constant shift cannot make every
phase pass its shape gate or when the required shift exceeds 2 degrees. All candidate and bias-calibration metrics are
saved in the `_sim_replay_pd_sweep.json` report. Validate the selected values on repeated data with
`--stiffness VALUE --damping VALUE --position-bias-rad VALUE`. None of these modes edits the actuator model or any
reinforcement-learning configuration.

Review repeated runs before changing any quality gate. `L_Leg_ankle_roll` now has a finalized cross-amplitude model
from two post-alignment 1.5-degree runs and one 3-degree run. All three fixed-base Isaac replays passed with
`Kp=26.25`, `Kd=1.08`, and one directly validated residual delay of approximately `4.72 ms`. The post-replay
total-delay estimates of approximately `5.53-6.05 ms` remain diagnostic only. This validated cross-amplitude command
path is included in the current promoted runtime.

After every target dataset selected by the baseline manifest has a passing replay report, aggregate the complete set
without changing the default runtime model:

```bash
python scripts/tools/rnd_actuator_aggregate_replays.py
```

This writes `scripts/tools/config/rnd_actuator_sim_replay_summary.json` and a separate
`scripts/tools/config/rnd_actuator_model_candidate.json`. The aggregate requires exactly one passing report for every
accepted target dataset, rejects inconsistent repeated `Kp`/`Kd` or position-bias values, and preserves the observed
residual-delay minimum/maximum per joint. The candidate remains `integration_enabled=false`; it does not edit
`rnd_actuator_model.json` or any reinforcement-learning configuration. Unresolved joints remain unresolved instead of
borrowing mirrored values.

Promote the disabled candidate into the explicit runtime scope only after reviewing the aggregate:

```bash
python scripts/tools/rnd_actuator_promote.py --full
```

The current `rnd_actuator_model_runtime.json` is a full promotion: all 12 leg joints passed their command-path and
fixed-base simulator-replay gates, `integration_joint_names` contains all 12 joints, and `fallback_joint_names` is
empty. Full promotion remains an explicit decision; aggregation never edits or enables the runtime model
automatically. A future candidate with an unresolved joint must remain disabled or use an explicitly reviewed partial
promotion rather than borrowing a mirrored value.

The existing Flat task retains its implicit actuators. Use the separate opt-in task for smoke testing and training:

```bash
python scripts/tools/zero_agent.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0 \
  --num_envs=16

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-v0 \
  --headless \
  --num_envs=512 \
  --max_iterations=300
```

The actuator task verifies the model's 200 Hz physics and 50 Hz policy rates at construction. Its explicit actuator
also rejects any configured per-joint stiffness, damping, effort limit, velocity limit, or nominal armature seed that
differs from the promoted command-path model. After that validation, the startup-only armature event writes the
training sample to the articulation and PhysX; it does not mutate the validated actuator seed.

## Hardware deployment boundary

The promoted actuator model belongs only in simulation. It makes simulated position targets pass through the measured
delay and generalized-play response before explicit PD is evaluated. Do not apply the same command-path transform to
real Dynamixel targets, because the physical actuator already contains that behavior.

`scripts/reinforcement_learning/rsl_rl/play.py` exports JIT and ONNX policies, but this repository does not yet contain
a hardware inference runner that synchronizes CMP10A IMU data, Dynamixel encoder telemetry, command input, policy
normalization, and goal-position writes at 50 Hz. Fixed-base replay validation therefore does not by itself establish
untethered walking readiness.

The first hardware policy tests must be staged with a rigid safety fixture: torque-on pose hold, zero-command standing,
weight shift, stepping in place, and low-speed commanded walking. Log timestamped policy observations and actions,
post-scale position targets, encoder position/velocity, current, PWM, IMU orientation/angular velocity, voltage,
temperature, hardware errors, and loop timing. Walking residuals must be separated into actuator response, mass/COM,
ground contact, sensor-frame, and transport-timing causes before any model is updated. The present actuator data was
collected in suspension, so ground-contact load dependence and measured friction torque remain unidentified.
