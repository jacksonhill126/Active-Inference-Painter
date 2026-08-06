# Variable And Sensor-Access Ledger

Ledger version: `sensor-access-ledger-v0`

Baseline: `baseline-oracle-v0`

Status: accepted implementation audit for `AI-102` on 2026-07-24. The
machine-readable inventory is
[`planning/variable-sensor-access-ledger.json`](../planning/variable-sensor-access-ledger.json).

## 1. Purpose

This ledger answers a deliberately strict question:

> What information crosses from the generative process into inference,
> planning, learning, control, and diagnostics, and could a physical robot
> actually obtain it?

The answer matters because adding uncertainty after reading exact simulator
state does not create a physical observation model. A Gaussian centered on the
true paint-thickness array is still an oracle-state likelihood. Likewise, a
counterfactual rollout initialized by deep-copying the complete simulator is
not equivalent to prediction from an inferred bodily state.

This document classifies the current implementation. It does not yet replace
the privileged paths. That replacement is the core of M2.

## 2. Classification Vocabulary

Every inventoried value has one primary classification:

| Class | Meaning |
| --- | --- |
| Physical sensor observation | A simulated value with a direct planned hardware sensor analogue. It may still be unused by the current agent. |
| Derived observation | A deterministic or statistical transform. It is sensor-equivalent only when all of its inputs are permitted observations or posterior beliefs. |
| Hidden state | Process state or a process parameter that a physical agent cannot read directly. |
| Simulator-only evaluation label | Truth retained for rendering, debugging, calibration assessment, or failure analysis, not agent inference. |
| Execution-only state | Commands, calibrated geometry, low-level controller state, or hard-safety state below painting-policy selection. |

Classification describes what a value *is*. Access status describes how it is
currently used. For example, Cartesian tip position is a derived quantity, but
the current controller derives it from true process pose, making that access
privileged.

## 3. Current Boundary

The intended physical boundary is:

```text
hidden world and body state
        |
        v
sensor generative process
        |
        v
camera + encoder + current + contact observations
        |
        v
posterior beliefs -> EFE -> painting policy
        |
        v
commands -> conventional controller -> hard safety -> plant
```

The retained oracle dependencies and corrected MuJoCo body boundary are:

```text
VerticalCanvas material arrays --------------------+
                                                     |
true ArmPainterSim pose/contact ---------------------+--> agent and controller
                                                     |
copied grain/model context ---------------------------+--> motor-policy rollout
native dynamic state/RNG (no body belief) -----------+

MuJoCo PhysicalSensorPacket -> BodyStateEstimator -> q/qvel rollout particles
                                              independent future plant seed

camera products -> SpatialCanvasState -> material-field rollout particles
                       mean particle + diagonal posterior-cell samples

BrushLoadBelief -> load/pigment particles
declared microstructure prior -> independent bristle-scale rollout noise

perfect canvas render -----------------------------------> browser only
```

That diagram is useful for diagnostic development, but it is not a
sensor-equivalent active-inference agent.

As of 2026-07-28 the live default is `sensor-boundary-v0` with observation
mode `sensor_equivalent`. It fails closed: oracle bootstrap is skipped,
policy inference and learning remain disabled, and planner reset/step return
without dereferencing the process object. Any attempted construction of a
planner state from `ArmPainterSim` raises `PrivilegedStateAccessError`.

This is boundary enforcement, not completed sensor inference. The default
therefore reports `sensor_equivalent=false` and `model_access_blocked=true`
until belief-derived substrate-grain/contact/model forecast construction and a
sufficient brush-history state replace the remaining process approximations,
and the action-conditioned transition/camera update is scheduled in the live
loop. Camera-conditioned painting, material-field, compact brush, and MuJoCo
body-conditioned forecast initialization are now implemented, but do not
weaken those remaining blocks. The old behavior is
available only through the explicit
`oracle_material_state` diagnostic mode.

## 4. Temporary Oracle Baseline

`baseline-oracle-v0` is the explicit name for the retained diagnostic
observation condition. It is no longer the fail-closed live default.

Its runtime observation mode is `oracle_material_state`. It may be used for:

- native process and controller development;
- transition-learning smoke tests;
- VFE/EFE equation tests under a declared oracle likelihood;
- a matched upper-bound comparator for later sensor-equivalent experiments;
- visualization and simulator-only evaluation.

It may not support claims of:

- sensor-only state inference;
- physical deployability;
- calibrated camera, proprioceptive, or contact perception;
- learning exclusively from physically available sensory history;
- embodied prediction without simulator-state leakage.

Every experiment manifest opting into this diagnostic mode must record:

```text
observation_baseline = baseline-oracle-v0
observation_mode = oracle_material_state
sensor_equivalent = false
```

An interesting painting does not relax this declaration.

## 5. Canvas And Material Variables

### 5.1 Hidden process fields

`VerticalCanvas` stores:

- `thickness[N,N]`;
- `wetness[N,N]`;
- `black_mass[N,N]`;
- `surface_tone[N,N]`;
- fixed substrate `grain[N,N]`.

These are hidden material states. A physical camera can observe reflected
light from the surface, not bulk pigment mass, true thickness, wetness, or the
simulator's internal surface-tone variable.

The current spatial path reads these arrays in
`spatial_state.material_grid_from_canvas()`. The summary path reads them in
`arm_agent_driver.canvas_summary_state()`. They then become:

- posterior means;
- replay-buffer inputs and targets;
- pixel and coarse hierarchy inputs;
- EFE rollout initial states;
- passage-observation evidence.

This is the largest direct oracle path.

### 5.2 Deterministic derived fields

The process derives:

- material occupancy from thresholded thickness;
- opacity from thickness;
- visible tone from surface tone;
- observed tone from ground tone, opacity, and surface tone;
- ground contrast from observed tone;
- scalar material coverage from occupancy.

These transforms are useful definitions, but most are not independent sensor
channels. Coverage and contrast inherit the same hidden material information
as their parents. Treating the parent fields and their deterministic
transforms as independent Gaussian evidence can double-count information.
`AI-103` owns that decision.

### 5.3 Obsolete summary compatibility observation

The obsolete summary compatibility fixture receives six exact aggregates:

1. material coverage mean;
2. mean thickness;
3. maximum thickness;
4. mean wetness;
5. overlap fraction;
6. painted-ground contrast.

All six are derived from hidden process arrays. None is currently inferred
from an image. They are retained for regression and tractable reference tests,
not as a proposed high-level painting representation.

### 5.4 Provisional spatial material observation

Spatial mode receives native-pixel material means, configured log variances,
and deterministic coarse-grained copies. Its six channels are thickness,
wetness, black mass, surface tone, ground contrast, and material coverage.

The word "observation" here means an observation of simulator state, not a
camera observation. The configured variance changes posterior precision but
does not make the mean sensor-equivalent.

### 5.5 Browser canvas

The browser PNG is produced from perfect orthographic `observed_tone`. It is
the closest current variable to a camera observation, but it has no camera
optics, occlusion, glare, latency, quantization, or sensor noise, and it is not
fed into inference.

This browser visualization remains privileged even though a separate
model-facing camera process now exists.

### 5.6 Model-facing camera observations

`CameraObservationProcess` renders MuJoCo geometry and occlusion, composites
only the superficial grayscale canvas appearance into visible canvas pixels,
then applies XML-declared provisional read/signal noise, quantization, delay,
and a weak rendered-lighting specular residual once at the native-frame
boundary. Homography views are rectified to shared canvas UV; the overhead
standoff camera retains a native frame and emits an edge-profile product.

Agent-facing `CameraFrame` contains only grayscale pixels, a static
calibration-validity mask, timestamps, sequence, role, registration, product,
sampling, source-resolution, and version metadata. Native, global, and foveal
products from the same exposure share capture identity. The internal dynamic
segmentation used by the generative process, exact visibility, pose, contact,
and material fields are not returned. White-on-white material coverage is
intentionally unobservable in this provisional model.

`CameraSpatialLikelihood` consumes only registered global/foveal products.
It predicts superficial grayscale from the spatial posterior's thickness and
surface-tone factors and gives wetness and black pigment mass zero direct
image Jacobian. A latent inlier/outlier mixture handles rendered occluders from
pixel residuals; it receives no visibility mask. Global and foveal products
from one native exposure are mosaicked before one likelihood update to avoid
double-counting their correlated acquisition noise. The separately logged VFE
contains state complexity, occlusion complexity, and expected negative log
likelihood. Its precisions are XML-declared simulation priors pending physical
calibration.

The `provisional-multiview-v4` acquisition contract assigns the owned OM-1
with its 25 mm lens and A7R II with its 35 mm lens to separate oblique views at
3840 x 2160. The Sony is provisionally configured for Super 35. Nominal
intrinsics live in the XML but remain hidden calibration parameters, not
agent observations. The contract declares independent 512 x 512 global and
256 x 256 foveal products. The simulator now retains the native frame, emits
the global product, and samples each explicitly requested fovea directly from
native pixels. A `FoveaRequest` is addressed in canvas UV and may be selected
from a sensor posterior, a policy prediction, or an operator diagnostic. No
request means no fovea. Exact simulator pose, segmentation, contact, material,
and visibility state may not select or populate it.

## 6. Bodily Variables

### 6.1 True pose and Cartesian geometry

`ArmPainterSim.actual_pose` is the true link-side joint pose. The driver and
stroke controllers use it directly for:

- forward kinematics and tip position;
- approach timing;
- contact release and retraction;
- stroke tracking error;
- motor rollout initialization;
- visualization and telemetry.

The simulated encoder packet is not used for these operations. On hardware,
the controller should consume encoder samples or an estimator posterior, and
painting inference should receive a declared proprioceptive likelihood.

Cartesian tip, joint points, and arm axes are derived values. They are
permitted below policy selection when computed from calibrated geometry and
sensor estimates. They are privileged in the present implementation because
they are computed from true pose.

### 6.2 Plant state and parameters

The native `JointPlant` contains exact:

- link and motor velocities;
- motor-side angle;
- thermal state;
- inertia, friction, stiffness, backlash, coupling, and gravity parameters;
- process-noise parameters and RNG state.

The rollout container still receives these through a deep copy of
`ArmPainterSim`. When the MuJoCo runtime supplies a `BodyBeliefSnapshot`, the
forecast resets joint position/velocity from that posterior and isolates future
plant randomness rather than retaining the copied q/qvel/RNG continuation.
Native oracle forecasts without a body belief still retain the copied dynamic
state. Nominal plant parameters may legitimately become part of the agent's
generative model, but reading the process instance's exact parameter and
dynamic state is not parameter inference. Future work must distinguish:

- versioned nominal parameters;
- calibrated parameter uncertainty;
- online parameter beliefs;
- hidden true process values retained only for evaluation.

### 6.3 Simulated physical sensors

The process already emits plausible sensor analogues for:

- joint encoder position;
- joint encoder velocity;
- motor current;
- applied or reported voltage.

These are currently exported to the web UI and telemetry CSV. A reusable,
sensor-only `BodyStateEstimator` now assimilates encoder position and velocity
under an explicitly supplied Gaussian likelihood and transition prior; it can
also assimilate optional contact-switch and force factors. It produces the
agent-safe `BodyBeliefSnapshot` and a per-factor VFE decomposition without
reading a process object.

The MuJoCo runtime now updates that estimator at initialization and after each
physics step, exposes body VFE separately, and freezes one posterior revision
per planning pass. Forecast particle zero uses its q/qvel mean and later
particles sample its diagonal variance with an independent future-noise seed.
The named `mujoco-ideal-sensor-body-likelihood-v0` profile is explicitly
provisional simulation-only and not hardware-calibrated. This connection does
not enable the live painting policy loop because its brush/contact/model and
continuous-update boundaries remain oracle-conditioned. Motor current, bus
voltage,
temperature, tool deflection, and faults remain explicitly unassimilated:
faults belong to hard safety, while current and deflection require conditional
load/contact likelihoods rather than an informal confidence score.

Computed torque and position error are derived signals. Exact friction,
backlash, elastic deflection, load decomposition, process torque, and encoder
noise standard deviation are simulator-only labels unless a future estimator
produces beliefs over them.

### 6.4 Contact

`ContactState` contains exact on-canvas status, bushing deflection, force,
pressure, brush width, and contact point. It is computed from geometry and the
intended pressure command. There is no contact sensor likelihood.

The exact contact state currently affects:

- contact-aware control;
- retraction and contact release;
- forecast feasibility and motor EFE;
- realized path and pressure residuals;
- motor-reliability learning.

This means the reliability learner is also oracle-conditioned. A future
physical path must derive its residuals from camera/encoder estimates and
force, deflection, current, or contact-switch observations.

## 7. Brush State

The brush process holds load, pigment contamination, carried tone, path
distance, bristle gains, streak phases, edge-wobble phases, and RNG state.
These are hidden causes of visible marks.

The process has separate persistent white and black brushes. Fresh load is
finite, contact depletes it, canvas pickup contaminates it, and instantaneous
reload restores the selected brush to full uniformly selected-color paint.
Tracking error still cannot toggle deposition.

The model maintains compact load and average-pigment beliefs per dedicated
brush and infers preserve versus reload from an explicit preparation-policy
prior and conditional mark-outcome EFE. One revision is frozen per planning
pass. Forecast particle zero uses its load/pigment means and a deterministic
representative microstructure; later particles sample its diagonal moments and
`brush-microstructure-prior-v0` under request-derived noise. The live brush RNG
and exact bristle realization no longer initialize counterfactual forecasts.
The microstructure prior is provisional and uncalibrated, and held paint plus
persistent bristle history are collapsed into the compact belief. The
camera/material path is implemented, but the local per-mark deposition
statistic has not yet been derived from it for the brush-load likelihood.

Nominal brush identity and issued reload commands are known. Actual loading,
contamination, bristle state, and tip lag remain latent and must ultimately be
inferred from sensory consequences.

## 8. Counterfactual Forecast Leakage

Global and local planning still call `copy.deepcopy(sim)` to construct the
rollout container. Before a MuJoCo motor forecast, q/qvel is replaced from the
frozen body posterior and future plant noise uses the request-derived seed.
When a spatial posterior is supplied, thickness, wetness, black pigment mass,
and surface tone are also replaced before the before-state summary is read.
Particle zero uses the posterior mean. Later particles sample diagonal variance
at posterior-cell resolution, then upsample those cells piecewise constantly
and physically project them. This avoids inventing native-pixel independence;
coverage and ground contrast are recomputed rather than sampled. Brush loading
is likewise sampled from a frozen `BrushLoadBelief`, while bristle-scale
variation comes from independent versioned prior noise. The remaining snapshot
includes:

- exact substrate grain (the four independent material fields use the
  posterior when supplied, but the oracle/native fallback can still copy them);
- no resolved held-paint or persistent bristle-history state; these are
  collapsed into load/average pigment and a provisional microstructure prior;
- exact process parameters;
- native plant dynamic/RNG state when no body belief is supplied.

The contact state is regenerated after sampled q/qvel initialization, but the
posterior contact probability/force is not yet mapped into MuJoCo brush
compression/flexure. Each motor alternative deep-copies the container again.
Plant parameter jitter adds limited uncertainty; MuJoCo parameters are not yet
sampled. Native oracle forecasts that receive no `BodyBeliefSnapshot` retain
exact plant dynamic state and copied plant RNG. Thus the body-state leak is corrected for
the MuJoCo runtime and the independent material-field leak is corrected when a
spatial posterior is supplied. Brush load/pigment and microstructure noise now
have their own boundary, while substrate-grain/model, collapsed brush history,
and native-fallback leakage remain.

This does **not** mean simulation-based prediction is illegitimate. The
principled replacement is:

1. infer a posterior over bodily, brush, contact, and material state from
   permitted observations;
2. initialize forecast particles from that posterior;
3. sample independent future process noise;
4. propagate parameter uncertainty separately;
5. compare predicted sensory outcomes with later sensor observations.

Until the remaining path exists, embodied EFE results must be labeled
oracle-conditioned.

## 9. Passage And Hierarchy Leakage

The passage posterior combines executed action geometry with an exact
before/after thickness-delta centroid and post-stroke surface tone. This is a
mixed action/outcome pseudo-likelihood. Its geometry is partly commanded and
its visual evidence is materially privileged.

The canvas and relational hierarchies inherit exact spatial material fields.
Therefore, evidence that those latents retain or organize structure is valid
only under the oracle-observation baseline. It is not yet evidence that a
sensor-driven agent can infer the same structure.

## 10. Diagnostics Boundary

The browser and telemetry log intentionally expose process truth:

- true joint pose, points, and tip;
- exact contact state;
- exact coverage;
- true link velocity;
- transmission and load decomposition;
- controller phase and target state.

These are useful evaluation labels. They are acceptable so long as:

- the diagnostics path never feeds painting inference;
- exported columns retain truth-versus-sensor labels;
- experiment analysis does not call process truth an agent observation;
- hardware comparisons use matched sensor estimates where required.

Commands, target pose, controller phase, and hard safety checks are
execution-only. Hard joint, force, current, workspace, and watchdog limits
remain external to active-inference preferences.

## 11. Blocking Matrix

| Privileged path | Claims blocked | Required tasks |
| --- | --- | --- |
| Exact material arrays and aggregates | Sensor-only perception, physical deployment, sensor-driven hierarchy | `AI-103`, `AI-201` through `AI-204`, `AI-216` |
| True pose and Cartesian tip | Sensor-driven embodiment and controller portability | `T-109`, `AI-201`, `AI-203`, `AI-204`, `AI-216` |
| Exact contact state in control and learning | Calibrated contact inference and physical reliability learning | `T-109`, `AI-201`, `AI-203`, `AI-204`, `AI-216` |
| Copied substrate grain/model snapshot, collapsed brush history, and native oracle dynamic/RNG fallback | Fully non-oracle embodied prediction; MuJoCo q/qvel, independent material fields, and compact brush state are posterior-conditioned when their snapshots are supplied | `T-109`, `AI-202`, `AI-203`, `AI-204`, `AI-216` |
| Exact plant parameters | Identified dynamics or calibrated energetic prediction | `AI-107`, `T-109`, `T-106`, later calibration work |
| Exact material delta in passage update | Sensor-driven passage belief | `AI-103`, `AI-203`, `AI-204`, `AI-207`, `AI-210`, `AI-216` |
| Mixed truth and sensor telemetry | Reproducible hardware-comparable diagnostics | `T-105`, `T-106` |

`AI-216` must not accept M2 while any research-critical live path depends on
the oracle material state, true pose/contact, or copied process RNG.

## 12. Immediate Decisions Handed To AI-103

This ledger deliberately does not decide the observation factorization. It
makes the next questions concrete:

1. Which visual quantity is actually emitted by the fixed-camera process?
2. Which material variables remain hidden causes?
3. Are coverage and contrast posterior summaries or likelihood channels?
4. Which deterministic transforms must be excluded from independent evidence?
5. What units and normalizations define each retained likelihood?
6. Which proprioceptive channels are independently observed versus derived?
7. How are contact and force represented when no direct force sensor exists?

The initial bodily answer is deliberately narrow: encoder angle and velocity
are independent Gaussian likelihood factors; a present contact switch is a
Bernoulli likelihood; and a present force sample is a Gaussian latent-force
likelihood. Force is not yet treated as independent evidence for the contact
factor, avoiding accidental double-counting until their joint likelihood is
specified.

## 13. Maintenance Contract

The JSON ledger is the machine-readable source for field coverage. Tests
require every dataclass field in `ArmPainterSim`, `VerticalCanvas`,
`JointPlant`, `MotorTelemetry`, `ContactState`, `Brush`, `ArmKinematics`, and
`ExecutionForecast` to appear in a classified entry.

When a field or access path is added:

1. classify it before using it in inference, planning, or learning;
2. state whether a physical platform can obtain it;
3. name its likelihood or explicitly mark that none exists;
4. assign blockers for privileged research-critical access;
5. update the runtime observation-boundary diagnostic when the baseline mode
   changes.

Static field coverage cannot detect every indirect leak. Gate reviews must
still trace copied objects, closures, callbacks, replay targets, and
diagnostic feedback paths.
