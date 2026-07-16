# RND STEP CMP10A identification

This tool identifies the read-only CMP10A data path needed by the 50 Hz RND
STEP policy. It records packet rates, checksum quality, static gyro bias/noise,
accelerometer norm, and the complete sensor-to-`base_link` rotation.

## Mount assumption

The CMP10A is rigidly mounted on top of `Upper_Body`. `Upper_Body` is fixed to
`base_link`, so the mount translation does not affect angular velocity or
projected gravity. The translation is intentionally left unknown and unused.
It must be measured later only if linear acceleration is consumed, because an
offset IMU then observes tangential and centripetal acceleration.

## Safety

- Support the robot securely and stop every walking or motor-control process.
- The identification tool never enables motor torque and never writes a CMP10A
  register or calibration command.
- Move the complete rigid robot during the three axis trials. Do not move the
  sensor relative to `Upper_Body`.

## Procedure

List serial devices without opening them:

```bash
python scripts/tools/rnd_imu_identify.py --list-ports
```

Probe the selected CMP10A port without transmitting bytes:

```bash
python scripts/tools/rnd_imu_identify.py \
  --probe \
  --port /dev/serial/by-id/<CMP10A_DEVICE> \
  --enable-hardware
```

Run the guided static and three-axis identification using the baud rate printed
by the probe:

```bash
python scripts/tools/rnd_imu_identify.py \
  --identify \
  --port /dev/serial/by-id/<CMP10A_DEVICE> \
  --baud <DETECTED_BAUD> \
  --enable-hardware
```

The default outputs are written under `logs/rnd_imu/` as an NPZ dataset and a
JSON report. A passing report confirms that the packet stream, static noise,
and mount rotation are usable at 50 Hz. It does not identify absolute USB
transport latency or dynamic attitude-filter error; those require a later
synchronized reference experiment.

## Guided dynamic consistency test

After a static/axis report passes, run the guided dynamic test with that report
as provenance:

```bash
python scripts/tools/rnd_imu_dynamic_test.py \
  --collect \
  --port /dev/serial/by-id/<CMP10A_DEVICE> \
  --baud <DETECTED_BAUD> \
  --identification-report logs/rnd_imu/<PASSING_REPORT>.json \
  --enable-hardware
```

Keep motor torque off and move the complete, rigidly supported robot by hand.
For each prompted axis, hold the center pose for two seconds, then smoothly
follow the one-second alternating cues for six cycles. Use the complete second
to travel smoothly from one side to the other instead of moving immediately
and waiting at the target. Use about 10-15 degrees for forward/backward and
left/right rocking, and 15-20 degrees for yaw. Do not jerk the robot or move
the sensor relative to `Upper_Body`.

The analysis differentiates the CMP10A Euler stream, converts ZYX Euler rates
to body angular velocity, and compares that result with the gyro stream. It
checks motion amplitude, commanded-axis dominance, correlation, gain, and the
relative delay between those two sensor outputs. This is useful for deciding
whether the internally filtered orientation is suitable for a 50 Hz policy.
It is not an absolute sensor-to-host latency measurement because both signals
come from the same unsynchronized serial stream.

An existing dataset can be analyzed again without hardware:

```bash
python scripts/tools/rnd_imu_dynamic_test.py \
  --analyze logs/rnd_imu/<DYNAMIC_DATASET>.npz
```

## Runtime-model promotion

Promote the accepted static report, dynamic report, and dynamic NPZ after both
quality gates pass:

```bash
python scripts/tools/rnd_imu_promote.py
```

The command defaults to the accepted files under `logs/rnd_imu/` and writes
`scripts/tools/config/rnd_cmp10a_runtime.json`. The logs are intentionally not
tracked. The checked-in model stores repository-relative source labels and the
SHA256 of all three inputs, so validating the checked-in contract does not
require the local provenance files. Regeneration does require the exact local
files and fails closed if either quality gate fails or the dynamic report and
dataset do not reference the promoted static-report hash.

The runtime transform is the signed `diag(-1, -1, +1)` mapping recorded by the
dynamic report. The full manual-trial rotation fit remains measured evidence
only because those trials contain cross-axis handling error and the physical
mount is aligned. Held-baseline gyro mean/std and Euler std discard the first
1.0 s of `dynamic_baseline`.

The measured packet rates, relative Euler-to-gyro delays, and dynamic angular
velocity gain ratios are separate from the assumed simulation envelopes. The
dynamic gains are not a static projected-gravity gain. Absolute transport
latency and level offset remain unmeasured; the older gyro `+/-0.2 rad/s` and
projected-gravity `+/-0.05` perturbations are retained only as stress-test
provenance and are not applied by the runtime model.

## Runtime consumption

The promoted model is consumed by the read-only latest-frame adapter:

```python
from robot_lab.hardware import CMP10ARuntimeSource

with CMP10ARuntimeSource(
    "/dev/serial/by-id/<CMP10A_DEVICE>",
    "scripts/tools/config/rnd_cmp10a_runtime.json",
) as imu:
    sample = imu.snapshot()
    omega_raw = sample.base_angular_velocity_rad_s
    omega_policy = sample.policy_angular_velocity
    gravity_policy = sample.projected_gravity
```

`base_angular_velocity_rad_s` is the bias-corrected base-frame value in rad/s.
`policy_angular_velocity` already includes the policy scale `0.25`; do not
multiply it again. `projected_gravity` is a unit base-frame vector and has scale
`1.0`. A 50 Hz controller should request one latest snapshot per policy tick;
the adapter deliberately does not average the roughly 200 Hz sensor frames.

The source flushes the host serial input queue once when it opens, never writes
to the CMP10A, and fails closed when the latest gyro/Euler pair exceeds the
tracked 30 ms host-age or 20 ms pair-skew limits. Frame age starts when bytes
are read by the host. It is therefore not a measurement of the sensor's
absolute sampling or USB transport latency.

## Training integration

The existing actuator task remains unchanged. Train the opt-in actor-observation
model as a new run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=6500
```

Nominal play uses deterministic midpoint delay and no sampled bias/noise.
Enable the promoted simulation envelope explicitly for a robustness check:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RNDLab-Isaac-Velocity-Flat-RND-Step-Actuator-IMU-v0 \
  --num_envs=16 \
  --enable_observation_corruption
```
