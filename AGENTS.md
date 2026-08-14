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

## Visual generative-model boundary

The canonical perceptual direction is
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`. Read it before changing
perception, canvas state, local transition learning, composition,
counterfactual mark prediction, or corpus schemas.

Keep the generative process and generative model asymmetrical:

- The process may retain detailed thickness, wetness, pigment, brush, grain,
  contact, pickup, and blending variables so it can generate credible sensory
  consequences.
- The target painting model is observation-first. Its persistent hierarchy
  must prioritize registered visual tone, oriented edges, continuous masses,
  uncertain boundaries, and their spatial/temporal relations.
- Do not promote a simulator variable into a persistent agent latent merely
  because it exists in the process.
- Persistent canvas-wide wetness is not part of the target agent belief.
  Wetness is process state, not a composition or painting-level latent.
- If material ambiguity becomes necessary for local mark prediction, infer an
  ephemeral action-conditioned interaction/affordance latent from a fresh
  image crop of the proposed target region. It may implicitly compress
  wetness-, thickness-, pickup-, viscosity-, and texture-like causes without
  naming or maintaining them canvas-wide.
- Material behavior that is not visually identifiable should remain calibrated
  uncertainty over visual mark outcomes. Exact material predictability must
  not become the primary painting objective.

The implemented six-channel, usually 16x16 `SpatialCanvasState` is a
provisional compatibility and simulation-integration baseline. It is not the
accepted visual or composition representation. A low-bandwidth hierarchy may
be compact, but it must preserve meaningful boundary orientation and shape;
do not mistake square raster cells for tone masses.

The owner's original VAE proposal is an action-conditioned **visual
mark-consequence model** trained from registered pre/post image patches,
actions, and brush context. Its first shadow implementation is
`action-conditioned-visual-mark-cvae-v0` in
`src/active_painter/visual_mark_vae.py`; its corpus and trainer are
`visual_trajectory_corpus.py`, `visual_collect.py`, and `visual_vae_train.py`.
It uses a learned conditional prior for planning-time latent samples, a
post-image recognition density only during training, and a normalized Beta
image likelihood. It has no policy influence and has not passed its admission
gates. Never use image error, edge error, or latent distance from this model as
an aesthetic reward.

The separately implemented
`conditional-local-material-transition-cvae-v0` predicts coarse material
posterior patches. Its negative AI-107/AI-109 result applies only to that
material target and does not test or reject the visual VAE. Never conflate the
two.

The central modeling target is the hierarchical visual active-inference model:
multiscale tone and oriented edges, continuous masses with uncertain
boundaries, spatial relations, recurring motifs, and slower structure across
marks and passages. Active attention remains important, but do not let camera
foveation work displace this hierarchy. An attention trajectory is a temporal
sequence of spatial sensory-access and precision allocations; it need not be a
smoothly steered rectangular crop. Treat `FoveaRequest` as one sensor-level
realization, preserve the high-resolution non-leakage boundary, and defer an
active attention policy until the visual likelihood and hierarchy can state
what hidden hypotheses an observation would resolve.

`registered-visual-trajectory-corpus-v1` retains registered rectified pre/post camera images,
calibration/timing/mask provenance, action-aligned crops, action, compact brush
belief, motor realization, whole-trajectory split identity, and genuine-stop
versus truncation provenance. Exact process material arrays are denied as
agent inputs. Treat this as simulation-only corpus evidence, not sensor-equivalent
hardware data. The existing v2 posterior corpus and 2026-08-12 stop pilot are
useful for their declared material-transition and termination evidence, but
are not end-to-end visual-model corpora.

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
temporal depth, one forecast particle, and three conditional motor
realizations: neutral Cartesian IK plus symmetric fixed upper-arm roll at
approximately +24 and -24 degrees. Those independent counterfactuals are
scheduled concurrently and use a declared 30 Hz simulation-throughput
approximation. Dynamic ±32-degree roll sweeps remain available in the broader
research configuration but are not enabled in this repeated smoke profile.
The corpus collector's explicit opt-in `research_full_roll` profile expands
conditional motor support to neutral IK, fixed ±24-degree roll, and dynamic
±32-degree roll sweeps. This is a data-collection profile, not the default live
profile, a hardware actuator change, or a painting preference.
The profile's finite mark proposal support contains straight and symmetric
signed quadratic curves with continuously sampled nonzero magnitude; curvature
is a painting-policy variable, while fixed or swept roll is a conditional
motor-realization variable. Do not conflate the two or describe curve support
as an aesthetic preference. The profile uses an
explicit equal preserve/reload policy prior because its initial compact brush
belief is empty; reload is still inferred from conditional material/pigment
outcomes. Its camera process uses `identity_canvas_appearance` for its
registered canvas product: geometry, occlusion, and acquisition noise remain,
but the surface intensity matches the provisional superficial-grayscale
likelihood. This is a declared photometric simulation approximation, not a
hardware calibration claim.
It also reduces the native camera acquisition renders for throughput; this is
a declared simulation approximation, not an operational sensor-equivalent
camera realization.
Because that profile deliberately begins with no bootstrap corpus, its local
material transition likelihood remains in an explicit conservative action-local
warm-up mode for its first 64 camera-derived transitions: the prior leaves the
material mean unchanged and widens variance only under the selected brush
footprint. A registered camera likelihood, not the random neural
initialization, supplies the first evidence for deposited paint. This too is a
provisional simulation-only integration approximation.
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

The generated `mujoco-robstride-angled-wrist-roll-exploration-v0` branch in
`src/active_painter/wrist_roll_design.py` is explicitly non-canonical and not a
runtime backend or selected hardware change. It relocates the roll RS02 to a
distal wrist and angles the brush for comparative simulation. Do not describe
it as replacing the accepted upper-arm roll unless a later recorded hardware
decision does so. Its evidence and approximation ledger are in
`docs/ANGLED_WRIST_ROLL_EXPLORATION_2026-08-07.md`.

## Camera baseline

Keep the current compact rig aligned across MJCF, runtime metadata, tests, and
support documentation. `provisional-compact-dual-imx296-v1` consists of
exactly two selected-but-not-yet-purchased Raspberry Pi Global Shutter Cameras
(Sony IMX296), one left and one right on a rigid crossbar. Their centers are
650 mm forward of the canvas plane, 300 mm to either side of canvas center,
and 220 mm above it. The provisional lens assumption is 4 mm CS mount. There
is no inspection, overhead, or profile camera in the initial baseline.

The declared intrinsics, noise, timing, 60 Hz rate, synchronization, housing,
and mount geometry are simulation/design assumptions pending purchase and
physical calibration. Do not describe the camera path as sensor-equivalent or
hardware calibrated. The canonical record is `models/README.md` plus
`models/active_inference_painter.xml`; the reproducible geometric evidence is
`docs/CAMERA_OBSERVABILITY_BRIEF.md`.

## Terminal coverage rule

Coverage is a material state derived from paint thickness, not visible tone. White paint on white ground still increases coverage.

The strong 80–90% coverage preference is terminal and conditional on `stop`. Do not apply it at every intermediate time step. Every candidate policy must terminate in `stop`, and the immediate `stop` policy must always be available.

## Contact rule

Do not introduce a globally preferred contact-pressure scalar. Pressure/contact predictions must be conditional on intended mark consequences, stroke phase, speed, curvature, brush state, local canvas state, and model uncertainty.

The canonical process now includes the provisional Baxter-inspired anisotropic
round-brush contact model recorded in
`docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md`.  Handle-leading motion has lower
directional friction; ferrule-leading push motion can stick and release against
frozen canvas tooth. The footprint determinant is the acute angle between the
end-effector/brush axis and the canvas plane: 90 degrees gives the circular
normal-incidence patch, and smaller angles elongate it along the handle's
canvas-plane projection. The axisymmetric tuft
has no independent chisel/roll orientation. This is a process consequence,
never a direct reward or policy preference. Keep normal and tangential force
distinct. The current
counterfactuals still reuse process-model equations under independent state;
do not mistake that simulation-only baseline for the eventual learned,
uncertain brush generative model.

## Parallel collection and shared pretraining

`src/active_painter/parallel_collect.py` is the canonical affordable data-path
baseline. Every spawned worker must retain independent process state, camera
clock/RNG, material/body/brush posteriors, replay, precision beliefs, passage
history, and persistent hierarchy latents. Workers may emit full
camera-derived posterior trajectories but must never write a shared checkpoint.
`trajectory-posterior-corpus-v2` is split by entire trajectory before local
patch extraction. Validation/test trajectories must never enter a training
replay. Fixed-horizon archives are truncations, not policy-selected terminal
paintings.

The v2 record also carries the compact inferred pre-stroke `BrushLoadBelief`.
It never carries exact held paint or bristle microstructure. Legacy v1 shards
remain readable with an explicit unavailable brush-context bit.

AI-108's accepted simulation-only baseline is the multi-root manifest
`runs/corpus-ai108-combined-20260811/split_manifest.json`: 16 v2 trajectories
and 256 transitions split 10/3/3. It separately records 256 process canvas, 64
reduced inference/acquisition approximation, and 16×16 posterior grid; every
split contains neutral/fixed/swept roll and the required mark/reach/material
bins. It contains no policy-selected stop trajectories and no camera-identified
bulk wetness. Do not describe it as a terminal composition corpus, a wet-paint
ground-truth corpus, a hardware-calibrated corpus, or a painting-quality result.

`src/active_painter/offline_train.py` is conventional centralized parameter
learning and must be described as `shared_pretraining`, not active-inference
policy selection or one agent's individual development. It may pool shared
generative-model parameters. It must not pool online canvas posteriors,
precision beliefs, brush state, passage histories, or persistent slow latents.
The current shards store camera-derived posteriors but not raw camera frames;
do not claim that this corpus trains the perception likelihood end to end.

`conditional-local-material-transition-cvae-v0` is an implemented
offline/shadow likelihood experiment. It conditions local transition samples
on current posterior mean/log variance, the selected mark, conditional motor
realization, and optional compact brush posterior. It has no policy influence,
does not replace the live local CNN or process-equation counterfactuals, and is
not a composition/order model. Its beta-one reconstruction NLL and latent KL
are logged separately; decoder likelihood variance, within-member latent
variance, and between-member epistemic disagreement remain distinct. Never
use its reconstruction error or latent distance as an aesthetic reward. The
canonical record and admission gates are
`docs/CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md`.
This material-field experiment is not the owner's proposed visual
mark-consequence VAE; see `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

## Higher-level priors

Separate posterior beliefs, transition priors, and preferences. Higher-level priors must have slower dynamics and higher temporal depth; do not let them copy lower-level observations through fast moving averages.

## Safety

Hard joint, current, force, workspace, watchdog, and non-finite-state limits remain external to the active-inference painting model.

## Required development practice

- Add tests for each probabilistic claim.
- Log VFE and EFE decompositions separately.
- Mark approximations as approximations.
- Never rename an ordinary reward or controller as active inference.
