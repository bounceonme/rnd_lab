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
separable. The exported torque values are clearly labeled proxies based on a nominal torque/current ratio.

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
- Coulomb-like current from positive/negative motion at matched positions in each slow-sine cycle
- explicit quality gates that suppress randomization ranges when repeatability or fit quality fails
- conservative delay, backlash, and Coulomb-current randomization ranges

The command-domain fit uses the configured control index, not the median USB read-completion interval. Viscous friction
is left unidentified instead of being inferred from acceleration/current terms that cannot be separated reliably in
this suspended encoder-only experiment.

`rnd_real2sim.model.EncoderDomainActuatorRandomizer` consumes those ranges at a chosen policy rate. Call `reset` once
per environment and `transform` before writing position targets. Its friction torque is only a proxy and must be added
through an actuator implementation that explicitly supports additive joint torque. The current RND training environment
is intentionally not modified until real data passes the reported quality checks.
