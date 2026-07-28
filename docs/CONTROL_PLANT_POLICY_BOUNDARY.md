# Control, Plant, And Policy Boundary

Status: accepted interface contract
Task: T-103
Date: 2026-07-24
Interface: `plant-interface-v1`

## Purpose

This contract separates active-inference painting decisions from conventional
robot execution and from the generative process. It is shared by the native
simulator, a future MuJoCo backend, and eventual hardware.

Defining this boundary does not make the native runtime sensor-equivalent.
The live default now fails closed before policy inference or learning can read
process truth. The old forecast implementation still deep-copies hidden body,
material, contact, parameter, brush, and RNG state, but that path is reachable
only through the explicit `oracle_material_state` diagnostic mode. It remains
a documented `baseline-oracle-v0` exception and must not satisfy M2 embodiment
claims.

## Semantic Layers

### Painting policy

`StrokeAction` is normalized Cartesian/contact intent:

- start and end location on the canvas;
- mark width and material amount;
- black/white tone;
- terminal `stop`.

It is not a joint trajectory, actuator command, or low-level control law.
Active-inference policy selection compares predicted consequences of
`StrokeAction` sequences.

### Policy realization

IK, motor-primitive selection, trajectory interpolation, collision checking,
and low-level control realize a selected painting policy. They may:

- reject physically inadmissible candidates;
- predict energy, uncertainty, contact loss, and execution error;
- expose those predictions to declared body/viability likelihoods and EFE;
- satisfy hard safety constraints.

They may not add an undeclared reward for visually preferred marks or choose a
painting because a controller happens to favor its geometry.

### Plant

The plant receives timestamped SI-unit actuator/tool commands and emits
physically realizable sensor packets. It does not expose exact pose, exact
contact, hidden material state, process parameters, or RNG state through the
agent-facing interface.

### Inference

State estimation transforms sensor history and transition priors into a
`BodyBeliefSnapshot`. Planning and counterfactual motor prediction consume
that posterior, not the live generative-process object.

### Evaluation

Simulation may expose exact process labels through the separate
`EvaluationTruthProvider`. Evaluation truth is permitted for tests,
calibration targets, plots, and failure analysis. It is never an agent
observation.

## Interface Records

The executable definitions are in
`src/active_painter/plant_interface.py`.

| Record or protocol | Direction | Contents |
| --- | --- | --- |
| `PlantCapabilities` | backend to runtime at setup | joint order, limits, command modes, physical sensor fields, control period, stepping and truth capabilities |
| `PlantCommand` | controller to backend | sequence, monotonic timestamp, joint position/velocity targets, optional feedforward torque, tool execution command |
| `PhysicalSensorPacket` | backend to estimator | encoder position/velocity, motor current, bus voltage, optional temperature/contact/deflection samples, faults |
| `BodyBeliefSnapshot` | estimator to control/planning | posterior means and variances for joint position/velocity and contact |
| `CounterfactualRolloutRequest` | planner to execution model | body belief, `StrokeAction`, motor primitive, immutable model snapshot ID, sample count, independent future-noise seed |
| `CounterfactualRolloutResult` | execution model to planner | predicted body consequences, energy, contact loss, and execution uncertainty |
| `SimulatorEvaluationTruth` | simulator to evaluation only | exact joint state, applied torque, and exact contact labels |
| `PlantBackend` | runtime/backend contract | capabilities, command write, sensor read, close |
| `SteppablePlantBackend` | simulation extension | explicit deterministic time advance |
| `ExecutionForecaster` | agent model contract | belief-conditioned counterfactual prediction |
| `EvaluationTruthProvider` | optional diagnostic contract | privileged truth read |

## Units, Order, And Time

- All interface positions are radians.
- All angular velocities are radians per second.
- Currents are amperes, voltages are volts, torque is newton-metres, force is
  newtons, deflection is metres, energy is joules, and variance carries the
  square of the corresponding unit.
- Joint-vector order is always the declared `joint_names` order.
- Command and sample clocks are monotonic seconds. Wall-clock time is metadata,
  not a control coordinate.
- Sequence numbers detect dropped, repeated, or reordered command and sensor
  packets.
- Missing optional sensors are `None`; exact simulator truth is not substituted
  for them.

## Safety And Replaceability

Hard joint, current, force, workspace, collision, watchdog, and non-finite
limits remain external safety constraints. They are not negative preferences
inside painting EFE.

Low-level gains and laws remain backend-specific and replaceable. A native
servo, MuJoCo actuator, ROS 2 controller, or physical motor drive can implement
the same command/sensor contract without changing painting-policy semantics.
Backend identity and capabilities must be recorded in each experiment
manifest.

## Counterfactual Noise Rule

A conforming `ExecutionForecaster` starts from:

1. a declared posterior snapshot;
2. a versioned model/parameter snapshot;
3. an independent rollout-noise seed.

It must not accept or copy a live `ArmPainterSim`, `JointPlant`, canvas, brush,
contact object, generative-process RNG, or future sensor-noise continuation.
This prevents the agent from predicting with information a physical robot
could not possess.

## Current Conformance

| Component | Status |
| --- | --- |
| Interface types and field validation | conforming |
| Separation of simulator-only truth type | conforming |
| Counterfactual request schema | conforming |
| Current native command loop | legacy, not yet adapted |
| Current motor forecast initialization | nonconforming oracle path |
| Live proprioceptive posterior feeding forecasts | not implemented |
| MuJoCo backend | not implemented |
| Hardware backend | not implemented |

Migration of the runtime forecast path depends on the M2 sensor package,
observation likelihood, and compact state estimator. Until then, diagnostics
and research reports must retain the `baseline-oracle-v0` label.

## Verification

`tests/test_plant_interface.py` verifies joint-order and SI-unit validation,
sensor packet shape checks, the absence of simulator-truth/process fields from
agent-facing records, and belief-conditioned counterfactual requests with an
independent noise seed.
