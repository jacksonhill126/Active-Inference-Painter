# Native Plant And Material Reference

Reference version: `native-abstract-v0`

Status: accepted abstract reference for `T-101` and material-invariant contract
for `T-102` on 2026-07-24.

This document freezes the identity of the current Python simulator as a
comparison reference. It does not claim that the values are measured hardware
parameters. A change to a value listed as contractual requires a new reference
version and an updated baseline artifact.

## 1. Intended Use

`native-abstract-v0` is used to:

- make current simulator behavior reproducible;
- define the logical command/canvas reference that the MuJoCo physical draft
  must preserve or explicitly retarget;
- distinguish exact implementation constants from representative mechanics;
- protect material semantics while the inference architecture changes;
- identify which process values a physical robot would need to sense or infer.

It must not be described as:

- a calibrated digital twin;
- a final mechanical design;
- a validated actuator model;
- a physical sensor model;
- evidence of sim-to-real transfer.

## 2. Coordinate And Unit Contract

The native arm uses:

- joint angles in degrees at the `ArmPose` and controller boundary;
- radians internally in kinematics and dynamics;
- world geometry in abstract length units currently corresponding to inches;
- motor dynamics in SI-like electrical/mechanical units where documented;
- normalized painting endpoints in `[0,1] x [0,1]`;
- pixel material arrays indexed `[row, column]`;
- a vertical canvas lying in the world `x-z` plane at constant positive `y`.

World origin is the shoulder/base joint. The kinematic point order is:

```text
base -> elbow -> brush tip
```

The arm-to-canvas mapping in `stroke_execution.py` is:

```text
x_world = (x_normalized - 0.5) * canvas_width * 0.98
z_world = (0.5 - y_normalized) * canvas_height * 0.98
```

The `0.98` reach factor is an execution margin, not part of normalized
painting intent.

## 3. Joint Contract

Joint order is contractual:

```text
yaw, pitch, roll, elbow
```

| Joint | Native axis construction | Range |
| --- | --- | --- |
| Yaw | shoulder rotation about world `z` before pitch | `[-90, 90] deg` |
| Pitch | shoulder rotation about local `x` after yaw | `[-90, 90] deg` |
| Roll | upper-arm-axis rotation about local `y` | `[-180, 180] deg` |
| Elbow | forearm rotation about local `x` after roll | `[0, 150] deg` |

Safe home pose:

```text
yaw = 0 deg
pitch = -50 deg
roll = 0 deg
elbow = 100 deg
```

Forward kinematics:

```math
R_s = R_z(yaw) R_x(pitch)
p_elbow = R_s [0,L_upper,0]^T
R_f = R_s R_y(roll) R_x(elbow)
p_tip = p_elbow + R_f [0,L_lower,0]^T
```

Contractual abstract link lengths:

```text
upper arm = 13.0 world units
lower arm = 13.0 world units
```

These are abstract reference dimensions. The current dynamics separately use
`0.3302 m` as the representative link length, which is the metric conversion
of 13 inches. Physical offsets, brush-handle length, bearing spacing, motor
housing geometry, and hard-stop placement are not yet measured.

## 4. Canvas And Contact Contract

Default canvas geometry:

| Field | Value | Status |
| --- | ---: | --- |
| Width | 20.0 world units | Contractual abstract geometry |
| Height | 20.0 world units | Contractual abstract geometry |
| Plane distance | `y = 17.0` | Contractual abstract geometry |
| Bushing travel | 0.5 world units | Representative |
| Contact stiffness | 55.0 force units/world unit | Representative |
| Default ground tone | 0.34 | Contractual material/render parameter |

Canvas inclusion is:

```text
abs(x) <= width/2 and abs(z) <= height/2
```

Contact deflection is positive penetration beyond the canvas plane and is
clamped to bushing travel. Contact force is the greater of geometric spring
force and intended-pressure force:

```text
geometric_force = contact_stiffness * deflection
intended_force = pressure * contact_stiffness * bushing_travel
```

The intended-pressure path is an execution convenience. It is not a physical
force-sensor likelihood and must be replaced or calibrated before hardware
claims.

The round-bundle envelope diameter is 0.50 world units (12.7 mm / 0.5 in),
aligned with the canonical MJCF. The contacting fraction grows continuously
from the tuft tip toward that envelope, and pressure then supplies bounded 20%
radial splay:

```text
p = clip(pressure, 0, 1)
contact_fraction = 0.12 + 0.88 * sqrt(p)
radius = 0.5 * 0.50 * contact_fraction * (1 + 0.20 * p)
```

The footprint aspect is then conditioned on `beta`, the acute angle between
the end-effector/brush axis and the canvas plane:

```text
aspect = clamp(1 + 0.35 * cot(beta), 1, 1.65)
```

At 90 degrees the end effector is perpendicular to the canvas and the patch is
circular. Smaller angles elongate it along the end-effector projection.

During the final stroke taper, desired Cartesian penetration and intended
pressure unload with width before lateral motion stops. Lift begins from the
canvas plane with taper flow zero. Positive residual physical contact can still
deposit; there is no paint-enable gate.

The contact state exposed by the simulator contains:

- on-canvas flag;
- deflection;
- force;
- normalized pressure;
- brush width in pixels;
- brush world position.

These are process-truth or derived process values.

## 5. Representative Joint Plant

`JointPlant` is a stochastic coupled actuator/link process. Its values are
representative small-arm values, not vendor measurements.

Global defaults:

| Parameter | Value |
| --- | ---: |
| Supply voltage | 24 V |
| Nominal current limit | 7 A |
| Servo stiffness coefficient | 1.0 |
| Damping coefficient | 0.80 |
| Torque constant | 0.42 N m/A |
| Terminal resistance | 2.1 ohm |
| Maximum motor velocity | 7 rad/s |
| Maximum link velocity | 5 rad/s |
| Gravity compensation fraction | 0.985 |
| Upper-arm mass | 1.35 kg |
| Lower-arm mass | 0.85 kg |
| Brush payload | 0.18 kg |
| Thermal heating time constant | 18 s |
| Cooling time constant | 65 s |

Per-joint representative defaults:

| Parameter | Yaw | Pitch | Roll | Elbow |
| --- | ---: | ---: | ---: | ---: |
| Motor inertia | 0.012 | 0.014 | 0.006 | 0.010 |
| Link inertia | 0.060 | 0.074 | 0.036 | 0.060 |
| Transmission stiffness | 28 | 32 | 18 | 24 |
| Transmission damping | 0.72 | 0.82 | 0.46 | 0.62 |
| Motor viscous friction | 0.018 | 0.022 | 0.012 | 0.016 |
| Link viscous friction | 0.010 | 0.014 | 0.007 | 0.010 |
| Coulomb friction | 0.018 | 0.025 | 0.012 | 0.018 |
| Static friction | 0.030 | 0.040 | 0.020 | 0.030 |
| Backlash deadband | 0.035 deg | 0.045 deg | 0.060 deg | 0.040 deg |
| Process torque noise std | 0.006 | 0.010 | 0.005 | 0.008 |

The mass matrix includes pitch/elbow and yaw/roll coupling. The process also
models:

- motor/link elastic deflection;
- backlash;
- static, Coulomb, and viscous friction;
- current, voltage, torque, and velocity saturation;
- residual gravity after compensation;
- contact-dependent load;
- thermal current derating;
- encoder bias and heteroscedastic noise;
- seeded process torque noise.

These mechanisms are useful hypotheses for forecasting and control tests. They
remain uncalibrated until joint-rig measurements identify their values and
residual structure.

## 6. Material-State Contract

The persistent native canvas arrays are:

| Field | Meaning | Constraint |
| --- | --- | --- |
| `thickness` | deposited paint volume per pixel proxy | nonnegative |
| `wetness` | persistent wet material available for interaction | nonnegative; no temporal decay |
| `black_mass` | black pigment mass in bulk paint | `0 <= black_mass <= thickness` within numerical tolerance |
| `surface_tone` | optically dominant surface black fraction | clamped to `[0,1]` |

Derived fields:

- material coverage from the thickness presence threshold;
- optical opacity from thickness;
- observed tone from ground plus surface layer;
- ground contrast from observed tone.

### 6.1 Coverage invariant

Material coverage is occupied substrate area:

```text
coverage[p] = 1 if thickness[p] >= paint_presence_threshold else 0
```

Consequences:

- white paint on a white or gray ground increases coverage;
- black and white have identical coverage semantics for equal thickness;
- additional paint in an already occupied pixel does not count that pixel
  again;
- coverage is bounded in `[0,1]`;
- coverage is not visible darkness, contrast, opacity, or accumulated layer
  count.

### 6.2 Wetness invariant

There is no drying term in this project phase. Wetness:

- increases with deposition;
- may move or scale when wet paint is picked up;
- does not decay merely because time passes or the brush is lifted.

Introducing wetness decay requires a new process version and an explicit
project decision.

### 6.3 Pigment accounting

Fresh black paint adds black mass; fresh white paint adds volume with zero
black mass. Brush pickup transfers volume and pigment from canvas to the
brush's held reservoir. Release transfers both back to the canvas.

For a transfer interval without black paint entering or leaving the modeled
canvas-plus-brush subsystem:

```text
canvas_black_before
= canvas_black_after + held_brush_black_after
```

up to declared floating-point tolerance.

The per-stroke brush reset represents cleaning/reloading and therefore opens
the subsystem boundary. Cross-stroke held paint is intentionally not
persistent in `native-abstract-v0`.

### 6.4 Surface optics

New surface tone is an opacity-weighted mixture of incoming pigment and picked
up wet surface tone. White over black is expected to remain visibly on top
while retaining some wet pickup. Optical opacity and material coverage remain
separate.

### 6.5 Configurability

The following remain configuration, not hard-coded policy decisions:

- presence and opacity thickness scales;
- deposition rate;
- pressure response;
- brush loading;
- brush width and taper;
- bristle furrows and edge variation;
- canvas grain;
- pickup, capacity, release, and push-forward behavior;
- tip lag;
- random seeds.

Changing these values does not change the meaning of material coverage,
wetness persistence, or pigment accounting.

## 7. Reset And Clear Semantics

`ArmPainterSim.reset_pose()`:

- returns actual and target pose to safe home;
- resets motor/link dynamic state and temperatures;
- disables painting;
- preserves the canvas.

`VerticalCanvas.clear()`:

- zeros thickness, wetness, black mass, and surface tone;
- preserves configured geometry and fixed canvas grain.

The web runtime's reset and clear commands may combine higher-level agent,
telemetry, or painting-count behavior. Those semantics are versioned
separately and are not part of this plant contract.

## 8. Sensor-Access Classification

The following values are physically plausible observations after adding
sensor models:

- encoder angle and estimated velocity;
- motor-driver current and voltage;
- temperature;
- contact/force measurement;
- fixed-camera images;
- timestamps and command acknowledgements.

The following are hidden process state or simulator-only evaluation labels:

- exact joint/link state before sensor corruption;
- exact motor torque and process-noise draw;
- exact contact penetration and intended pressure;
- exact pixel thickness, wetness, black mass, and surface tone;
- exact material coverage;
- exact brush reservoir state;
- exact pigment transfer.

The current planner receives several values from the second list. `AI-102`
must enumerate every access path before M2 replaces it.

## 9. MuJoCo Clone Requirement

S1 must first match:

- joint order, axis convention, and range;
- home pose;
- link lengths;
- shoulder origin;
- canvas plane and dimensions;
- brush-tip site;
- normalized stroke-to-canvas mapping.

S1 must not introduce measured offsets, housings, collision constraints,
physical inertias, or vendor actuator models into the abstract clone unless
they are versioned as a separate model. There is currently no accepted MJCF
artifact in the repository, so S1 remains blocked until its implementation and
tests are present.

## 10. Contract Tests

The reference is protected by:

- `tests/test_native_contract.py`;
- `tests/test_canvas.py`;
- relevant contact, dynamics, brush, and reset tests in
  `tests/test_arm_sim.py`;
- spatial material tests in `tests/test_spatial_state.py`.

Passing these tests establishes parity with `native-abstract-v0`. It does not
establish physical fidelity.
