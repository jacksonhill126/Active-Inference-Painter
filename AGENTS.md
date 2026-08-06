# AGENTS.md — Active-Inference Painter

## Communication and report audience

Keep durable technical reports useful for later agents, and explain completed
work separately to the project owner. Assume the owner has undergraduate
aerospace-engineering foundations, some self-directed ML/data-science and
active-inference experience, but not specialist depth in every contributing
field. Results matter, and so does a clear account of how and why they were
obtained.

## Governing constraint

Painting cognition must be active inference from top to bottom. Do not add vague aesthetic heuristics, scalar rewards, or weighted score soups.

At painting level, every decision-relevant quantity must be explicitly grounded as one of:

- a likelihood in the generative model;
- a transition prior;
- a prior preference over outcomes or latent trajectories;
- a precision belief;
- a variational free-energy term;
- an expected-free-energy term;
- a policy prior or posterior.

## Allowed conventional engineering boundary

Conventional forward kinematics, inverse kinematics, trajectory interpolation, robot dynamics, motor control, collision checking, and hard safety constraints are allowed below the selected painting policy.

IK may realize or predict a Cartesian painting policy. It must not choose the painting policy.

## Plant and motor vocabulary

Keep these four concepts distinct in code, documentation, and status reports:

- **Hardware actuator assignment:** RobStride 03 at `yaw` and `pitch`, and
  RobStride 02 at `roll` and `elbow`, are selected and fixed in the current
  hardware-oriented draft. This is a design configuration, not a runtime
  latent. The associated plant is not yet calibrated against assembled
  hardware.
- **Native plant:** `native-abstract-v0` / `JointPlant` is a representative
  abstract actuator and arm model. Its parameters are not identified from the
  selected RobStride motors.
- **MuJoCo plant:** `mujoco-robstride-electromechanical-v4` is the selectable,
  hardware-oriented realized-execution backend. It is vendor-grounded but
  still contains documented derived and approximate parameters.
- **Motor realization:** active inference may infer a conditional
  controller/trajectory realization such as Cartesian IK, joint spline, elbow
  pivot, or upper-arm roll. This does not select an actuator product or SKU.

Realized execution and counterfactual motor prediction must preserve the
selected plant: native execution forecasts through `native-abstract-v0`, while
MuJoCo execution forecasts through independent MuJoCo data under
`mujoco-robstride-electromechanical-v4`. In the MuJoCo runtime,
`BodyStateEstimator` now conditions forecast joint position and velocity on
the latest `PhysicalSensorPacket`; particle zero uses the posterior mean and
later particles sample its declared diagonal variance with future plant noise
isolated from the live process RNG. The named likelihood is explicitly
`provisional_simulation_only_not_hardware_calibrated`.

Forecasts that receive a `SpatialCanvasState` now also initialize thickness,
wetness, black pigment mass, and surface tone from that frozen material
posterior. Particle zero uses its mean; later particles sample its declared
diagonal variance at posterior-cell resolution before piecewise-constant
upsampling and physical projection. Coverage and ground contrast remain
derived, not independently sampled. Brush forecasts freeze the selected
`BrushLoadBelief`; particle zero uses load/pigment means and a deterministic
representative microstructure, while later particles sample its diagonal
load/pigment variance and `brush-microstructure-prior-v0` under independent
forecast noise. Never continue the live brush RNG. This prior is provisional
and uncalibrated, and held paint/bristle history are collapsed into the compact
load/average-pigment belief. The legacy/oracle forecast container still
supplies process substrate grain and model parameters; the provisional sensor
simulation instead uses an independent fixed prior template. In both paths the
inferred contact probability/force
is not yet mapped into MuJoCo brush-compliance state; native runtime execution
has no `PlantBackend` sensor adapter; and MuJoCo body-parameter uncertainty is
not sampled.

The sensor path has an explicit `action-conditioned-camera-update-v0` clock.
A completed action first creates `spatial-action-transition-prior-v0` and the
brush depletion prior without reading process material. The MuJoCo runtime then
registers a post-physics capture boundary and polls until a registered camera
exposure captured at or after that boundary is delivered. Older frames are
rejected. The camera likelihood supplies the posterior and its separately
logged VFE; prediction alone never creates VFE. A pending update gates the next
planning pass. The brush camera likelihood is still open because no local
mark-deposition statistic has been declared from camera evidence.

The default remains fail-closed, but the explicit opt-in
`provisional-sensor-simulation-v0` MuJoCo smoke profile can run repeated
painting-policy cycles without copying the live `ArmPainterSim`. It requires a
registered initial camera likelihood and body posterior. Counterfactuals start
from a separately constructed MuJoCo model with independent substrate-grain
and brush seeds, then overwrite joint, material, and compact brush slices from
frozen beliefs. Its bounded profile uses eight candidate policies, one-step
temporal depth, one Cartesian-IK motor realization, and one forecast particle.
It also reduces the native camera acquisition renders for throughput; this is
a declared simulation approximation, not an operational sensor-equivalent
camera realization.
The oracle bootstrap is disabled; camera-derived replay is the only live
transition evidence. Exact live pose/contact may still be used by conventional
execution and hard safety below the selected painting policy.

Describe this mode as an uncalibrated simulation-only integration baseline,
never as sensor-equivalent hardware cognition, hardware-calibrated control, a
painting-quality result, or sufficient for embodiment claims.
`baseline-oracle-v0` remains the only mode allowed to expose exact canvas state
to policy inference.
Never state that no specific motors have been selected without explicitly
limiting that statement to the native abstract plant.

The canonical hardware-plant record is `models/README.md` plus
`models/active_inference_painter.xml`; the semantic boundary is
`docs/CONTROL_PLANT_POLICY_BOUNDARY.md`.

## Terminal coverage rule

Coverage is a material state derived from paint thickness, not visible tone. White paint on white ground still increases coverage.

The strong 80–90% coverage preference is terminal and conditional on `stop`. Do not apply it at every intermediate time step. Every candidate policy must terminate in `stop`, and the immediate `stop` policy must always be available.

## Contact rule

Do not introduce a globally preferred contact-pressure scalar. Pressure/contact predictions must be conditional on intended mark consequences, stroke phase, speed, curvature, brush state, local canvas state, and model uncertainty.

## Higher-level priors

Separate posterior beliefs, transition priors, and preferences. Higher-level priors must have slower dynamics and higher temporal depth; do not let them copy lower-level observations through fast moving averages.

## Safety

Hard joint, current, force, workspace, watchdog, and non-finite-state limits remain external to the active-inference painting model.

## Required development practice

- Add tests for each probabilistic claim.
- Log VFE and EFE decompositions separately.
- Mark approximations as approximations.
- Never rename an ordinary reward or controller as active inference.
