# Control, Plant, And Policy Boundary

Status: accepted interface contract
Task: T-103
Date: 2026-07-24
Implementation status reviewed: 2026-08-04
Interface: `plant-interface-v1`

## Purpose

This contract separates active-inference painting decisions from conventional
robot execution and from the generative process. It is shared by the native
simulator, the implemented MuJoCo simulation backend, and eventual hardware.

Defining this boundary does not make the native runtime sensor-equivalent.
The live default now fails closed before policy inference or learning can read
process truth. The old forecast implementation still deep-copies hidden body,
material, contact, parameter, brush, and RNG state, but that path is reachable
only through the explicit `oracle_material_state` diagnostic mode. It remains
a documented `baseline-oracle-v0` exception and must not satisfy M2 embodiment
claims.

## Plant And Motor Vocabulary

| Concept | Contract meaning |
| --- | --- |
| Hardware actuator assignment | Fixed in the current hardware-oriented draft: RobStride 03 at `yaw`/`pitch`, RobStride 02 at `roll`/`elbow`. It is configuration, not an inferred policy variable. |
| Native plant | `native-abstract-v0` / `JointPlant`; representative mechanics not identified from the selected motors. |
| MuJoCo plant | `mujoco-robstride-electromechanical-v4`; a selectable vendor-grounded realized-execution model, not yet a calibrated hardware twin. |
| Motor realization | A conditional controller/trajectory latent. It does not select an actuator product. |

Realized execution and counterfactual prediction preserve the selected plant.
Native execution forecasts with `native-abstract-v0`; MuJoCo execution
forecasts with independent data under
`mujoco-robstride-electromechanical-v4`. The MuJoCo runtime now updates a
`BodyStateEstimator` from physical sensor packets and freezes that posterior
for each planning pass. Motor particles initialize joint position/velocity
from its mean and diagonal variance under an independent future-noise seed.
The containing material/brush/model snapshot remains oracle-conditioned, so
the overall path is still an explicit `baseline-oracle-v0` diagnostic rather
than a fully conforming `ExecutionForecaster`. The canonical plant fields and
actuator assignment live in `models/README.md` and
`models/active_inference_painter.xml`.

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

For each painting-policy candidate, active inference may infer a conditional
motor-realization latent from declared likelihood, precision, and EFE terms.
The latent selects a controller/trajectory family, not hardware. IK,
trajectory interpolation, collision checking, and low-level control implement
that realization. They may:

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
that posterior for MuJoCo joint-state initialization. The current rollout
container still copies material, brush, contact/model context from the live
generative-process object; this is a declared nonconformance, not body
evidence.

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
| Native realized-execution path | implemented; legacy direct `JointPlant` loop, not yet a `PlantBackend` adapter |
| MuJoCo realized-execution path | implemented and selectable; `PlantBackend` in SI units |
| Selected-plant motor forecast | implemented; native-to-native and MuJoCo-to-MuJoCo with independent rollout state and explicit provenance |
| MuJoCo forecast joint-state initialization | implemented from frozen `BodyBeliefSnapshot`; posterior mean plus diagonal particles and independent future-noise seed |
| MuJoCo body likelihood | explicit `mujoco-ideal-sensor-body-likelihood-v0`; provisional simulation-only, not hardware-calibrated |
| Remaining forecast initialization | nonconforming copied material, brush, contact/model context; oracle diagnostic only |
| MuJoCo forecast parameter uncertainty | not implemented; deterministic plant particles currently share the immutable MJCF model |
| Contact-posterior initialization of brush compliance | not implemented; forecast provenance names this approximation |
| Live proprioceptive posterior feeding forecasts | implemented for the MuJoCo runtime; native `PlantBackend` adapter remains open |
| Hardware backend | not implemented |

The joint-state part of the runtime forecast path now crosses the M2 sensor
boundary. The named likelihood/profile is a numerical simulation assumption,
not measured RobStride or assembled-arm calibration. Diagnostics and research
reports using forecast-driven painting policy inference must retain the
`baseline-oracle-v0` label until material/brush/contact initialization is
belief-derived and the action-conditioned observation loop is live. Selecting
the MuJoCo execution backend removes the plant-family substitution and exact
joint-state initialization, but not those remaining oracle dependencies.

## Verification

`tests/test_plant_interface.py` verifies joint-order and SI-unit validation,
sensor packet shape checks, the absence of simulator-truth/process fields from
agent-facing records, and belief-conditioned counterfactual requests with an
independent noise seed. `tests/test_body_inference.py`,
`tests/test_stroke_execution.py`, and `tests/test_mujoco_backend.py` verify
estimator provenance, posterior-particle initialization, independence from the
live process RNG/state, MuJoCo plant preservation, and separate body-VFE
diagnostics.
