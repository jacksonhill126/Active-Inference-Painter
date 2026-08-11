# Project Progress

## 2026-08-11 - AI-107 calibration measured and failed the M2 gate

AI-107 trained substantial CNN and conditional patch-cVAE ensembles on the
accepted AI-108 train split, fitted one variance scale on validation only, and
evaluated the frozen test trajectories. Both models improved held-out density
but produced badly shaped Gaussian intervals: nominal 90% intervals contained
about 99.4% of test residuals. The same failure remained under an action-
footprint-only diagnostic. A CNN trained only on neutral/fixed-roll cases
increased ensemble disagreement just 1.087x on held-out dynamic-roll cases,
below the predeclared 1.50x gate. All precision-ledger entries remained
unobserved declared priors, and the cVAE's within-member latent variance was
effectively collapsed. AI-107 closes as an evidence task with a negative M2
calibration result; AI-109 learning curves are next. See
`docs/AI107_UNCERTAINTY_CALIBRATION_TECHNICAL_2026-08-11.md` and
`docs/AI107_UNCERTAINTY_CALIBRATION_OWNER_BRIEF_2026-08-11.md`.

## 2026-08-11 - AI-108 leakage-resistant corpus accepted

AI-108 now has a provenance-complete `trajectory-posterior-corpus-v2` dataset:
16 independent trajectories and 256 camera-posterior transitions, split
10/3/3 before patch extraction. Every split contains the required tone,
material/overlap, edge, width, length, signed-curvature, vertical-direction,
reach, neutral/fixed-roll, and dynamic-roll-sweep conditions. All transitions
carry compact inferred brush context and exclude exact process material from
training input. Shards separately attest 256 process canvas, reduced 64
inference/acquisition approximation, and 16x16 posterior grid. A two-step cVAE
consumer smoke proves the multi-root manifest is usable but is not a learning
result. Bulk wetness remains structurally unobserved, and the corpus has no
genuine stop trajectories, so terminal composition training remains blocked.
See `docs/AI108_CORPUS_TECHNICAL_2026-08-11.md` and
`docs/AI108_CORPUS_OWNER_BRIEF_2026-08-11.md`.

## 2026-08-11 — conditional patch-transition VAE shadow baseline

The owner's original VAE direction has been reinstated as a bounded
offline/shadow likelihood experiment. `ConditionalPatchVAEEnsemble` models
camera-posterior local material consequences conditioned on the current
posterior mean/log variance, selected mark, conditional motor realization, and
optional compact inferred pre-stroke brush posterior. The v2 trajectory corpus
records that brush belief without exact held paint or bristle state; v1 shards
remain readable with context explicitly unavailable. Training uses a beta-one
negative ELBO, reports reconstruction NLL and latent KL separately, and splits
decoder, latent-outcome, and bootstrap-member uncertainty. Held-out reports
include paired condition ablations and moment calibration. The model has no
policy influence and is not a composition/order model. Ten focused cVAE/corpus
tests passed; no substantial model has yet been trained. See
`docs/CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md`.

## 2026-08-10 — variable-curvature mark proposal

The bounded live profile no longer proposes every curved mark with the same
quadratic bend. Its hand-written proposal now has one straight atom plus
continuous, symmetric positive/negative curvature magnitudes. Proposed mark
length is decoded as brush travel rather than endpoint chord length, removing
the hidden extra-arclength/coverage advantage that curved candidates previously
received. This changes candidate support and decoder geometry only; no curve
reward or aesthetic preference was introduced. The remaining single-mark
family is still a symmetric quadratic, so apex timing and inflection/S-curves
remain future action-space work. See
`docs/CURVED_MARK_AND_ROLL_REALIZATION_2026-08-07.md`.

## 2026-08-10 — anisotropic round-brush process baseline

The canonical upper-arm-roll simulator now projects an angle-dependent round-
brush footprint and advances a Baxter-inspired aggregate bristle stick/slip
state. Its determinant is the angle between the end-effector axis and canvas
plane: 90 degrees is circular, and smaller angles elongate along the projected
axis. The contact fraction grows toward the canonical 12.7 mm tuft envelope
with pressure, followed by bounded splay; it no longer behaves like a fixed
chisel edge or a full-envelope zero-load stamp. Stroke exit now unloads depth,
pressure, and width together before tangential motion stops, eliminating the
systematic circular endpoint node caused by the old stationary lift stamp.
Directional friction is conditioned on handle-leading alignment, normal force,
and frozen canvas tooth. It changes the deposited painting-point trajectory,
native abstract plant load, forecast path residual, and camera-visible material
consequence; no direction reward or direct preference was added. Parameters
are provisional and uncalibrated. Exact bristle/contact fields are diagnostic
labels, and counterfactual reuse of the process equations is explicitly not the
desired final stochastic generative model. See
`docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md`.

## 2026-08-07 — compact matched global-shutter camera baseline

The camera baseline is now `provisional-compact-dual-imx296-v1`: exactly two
selected-but-not-purchased Raspberry Pi Global Shutter Cameras, 650 mm in
front of the canvas plane, ±300 mm laterally, and 220 mm above canvas center.
The inspection/overhead views and full-size OM-1/A7R II assignments were
removed from current metadata. A 243-pose MuJoCo sweep gives 100% sampled
bristle-tip visibility from each camera. Lens, rate, timing, noise, housing,
and mount remain provisional and uncalibrated. See
`docs/COMPACT_DUAL_IMX296_CAMERA_RIG_2026-08-07.md`.

This document is the concise public record of what has been demonstrated, what
has failed, and what comes next. Detailed task state remains in
`planning/PROJECT_TRACKER.md`.

## Current Snapshot

Snapshot date: 2026-08-10.

Phase: M1 formal baseline audit, with bounded M2 sensor-path and simulation
support work proceeding in parallel. M1 has not passed its lock decision.

The current prototype can:

- run the native arm, contact, and oil-paint generative process;
- infer individual-mark and hierarchical passage candidates;
- compare stochastic motor realizations;
- learn local transition and hierarchy models online;
- construct registered global and requested-foveal camera observations and
  assimilate them through a provisional spatial likelihood;
- infer a compact encoder/contact body posterior through an isolated estimator;
- preserve the selected native or MuJoCo plant in counterfactual motor
  forecasts, with independent MuJoCo rollout state and explicit provenance;
- feed the MuJoCo sensor packet into that body estimator and initialize
  forecast joint position/velocity from a frozen posterior revision with
  independent future-noise seeds;
- initialize forecast brush load and average pigment from a frozen compact
  posterior while drawing bristle-scale mark variation from an independent,
  versioned microstructure prior;
- propagate a completed action through an explicit learned material transition
  prior, reject camera frames captured before completion, and automatically
  wait for a causally later camera posterior update;
- run an opt-in, bounded MuJoCo policy loop from registered camera/body beliefs
  and an independent forecast model, completing repeated paint/camera cycles
  without exact live process state in painting-policy inference;
- verify the AI-104/AI-105 VFE/EFE acceptance matrix against independent
  analytic, enumerated, and fine-grid references;
- checkpoint learned state and export telemetry;
- render the live arm and canvas in a browser.

It has not yet demonstrated:

- a trained, calibrated, full-depth live painting loop driven only by
  sensor-equivalent observations;
- calibrated predictive uncertainty at live scale;
- a predictively necessary global or relational latent;
- a proposal-invariant policy posterior or a correction for finite-candidate
  bias; the completed convergence audit was negative;
- a frozen or cross-fitted safeguard for the self-trained composition
  preference;
- MuJoCo parity;
- physical hardware control or sim-to-real transfer;
- emergent composition.

The default `sensor_equivalent` runtime remains deliberately fail-closed:
visualization and scripted execution are available, but painting-policy
inference is disabled. The explicit `--enable-provisional-sensor-policy`
MuJoCo profile runs a bounded integration loop with independent fixed priors;
it is not hardware calibrated or a painting-quality result. The
`oracle_material_state` mode remains a diagnostic upper-bound comparator, not
a sensor-based agent.

## Verification Snapshot

Local environment: Windows 11, Python 3.14.3, PyTorch 2.11.0+cu126. CUDA was
available, but this test result is not a GPU performance benchmark.

| Check | Result | Interpretation |
| --- | --- | --- |
| Current test collection | 463 tests collected | Includes the learned-proposal, AI-111 convergence, completed AI-104/AI-105 reference, body/material/brush-posterior forecast, action/camera-clock, provisional sensor simulation, documentation-contract, and MuJoCo alignment coverage |
| Complete suite, uncontended audit run | 415 passed; 527 seconds observed | Recorded in `docs/DEVELOPMENT_AUDIT.md`; deadlines were not relaxed |
| Independent 2026-08-04 review | 414 passed; one Windows temp-directory setup error | The affected synthetic calibration test body passed separately with 11 usable views and 0.120 px RMS error |
| Proposal and AI-111 focused suites | 20 passed; 8.67 seconds observed | Covers normalized support, parity, training, checkpointing, deterministic convergence metrics, and retained run provenance |
| Current deterministic CI gate | 167 passed; 49.31 seconds observed | Exact file list from `.github/workflows/ci.yml`; includes the action-transition-prior test and both proposal suites, with two expected obsolete-summary warnings |
| Pre-AI111 complete-suite attempt | No terminal result before the fixed 15-minute limit | Stopped under load with no failure traceback; this is not reported as either a pass or a code failure |
| Source checks | Python compilation and `git diff --check` passed | No truncated source or malformed patch was found |
| Body/MuJoCo forecast alignment | 90 affected motor, plant, MuJoCo, web-runtime, documentation, and boundary tests passed in 278.69 seconds | Posterior sampling, frozen planning revisions, selected-plant copying, independent rollout noise, live-state non-mutation, provenance/VFE, and the fail-closed sensor boundary are covered |
| Material forecast alignment | 93 affected boundary, documentation, spatial-state, camera, stroke-execution, and driver tests passed; the broad 91-test run took 107.48 seconds and two final provenance/support checks took 1.95 seconds | Frozen material revisions, posterior mean/variance particles, process-material independence, physical projection, provenance, and planning integration are covered |
| Brush forecast alignment | 203 affected brush, stroke, driver, sensor-boundary, body/material inference, plant, MuJoCo, motor, web-runtime, camera, and independent-reference tests reported passed | Frozen brush revisions, mean/variance particles, live-brush/RNG independence, reload/preserve transitions, provenance, selected-plant behavior, and fail-closed boundary contracts are covered. Two broad Windows invocations printed complete passing pytest summaries before the outer command wrapper timed out while exiting; the other 72 checks exited normally. |
| Action/camera clock | 114 affected spatial, camera, camera-process, driver, web-runtime, brush, reference, documentation, and boundary checks passed | Covers transition uncertainty without synthetic VFE, post-action capture causality, stale-frame rejection, exactly-once replay, automatic MuJoCo delivery, process-material independence, and unchanged fail-closed contracts. The 41-test driver run printed its passing pytest summary before the Windows command wrapper timed out while exiting; the other 73 checks exited normally. |
| Provisional sensor simulation | 48 affected boundary, documentation, camera/spatial, MuJoCo, driver, and repeated web-runtime checks passed | Covers default fail-closed behavior, independent forecast context, denied exact planner state, initial sensor gating, two repeated paint/camera cycles, nonzero test-only process coverage, replay cardinality, forecast provenance, and monotonic body/MuJoCo time |

These timings are local observations, not stable performance claims. Hardware,
operating system, dependency versions, and concurrent load were not yet
captured in a run manifest.

## Current Priorities

1. Run AI-109 learning curves against the frozen AI-107 evaluator. AI-108's
   live-scale 160/48/48 transition split is accepted and AI-107 is complete,
   but both the CNN and cVAE failed interval calibration: nominal 90% intervals
   covered about 99.4% of test residuals. The fixed-to-dynamic-roll ensemble
   disagreement ratio was only 1.087x against a declared 1.50x gate. Curves
   must separate data, capacity, seed, and likelihood-family effects and test
   the explicit near-no-change/continuous-consequence mixture hypothesis.
2. Resolve the composition-preference closed loop in AI-110 and carry AI-111's
   negative convergence result into an explicit M3 proposal correction. Until
   then, keep the learned mix at zero and label posterior mass as conditional on
   the sampled candidate set.
3. Define checkpoint inheritance and online-learning semantics, profile the
   major runtime phases, and capture three manifested baseline replicas.
4. Make the M1 lock decision before treating ongoing M2 work as an accepted
   sensor model.
5. Collect episode-split camera/body posterior transitions and train a
   sensor-compatible checkpoint for the new provisional loop. Then derive the
   local camera statistic for the brush-load likelihood without counting the
   transition prediction as evidence; fit a compact stochastic brush-contact
   transition/observation model from permitted sensor history so
   counterfactuals no longer reuse the process contact equations; and expand
   candidates, particles, motor realizations, and temporal depth one dimension
   at a time.

## Progress Log

### 2026-08-07: curved marks and decision-relevant roll realization

Technical record: `docs/CURVED_MARK_AND_ROLL_REALIZATION_2026-08-07.md`.

- Added signed quadratic curvature as painting-policy geometry through action
  encoding, proposal density, spatial rasterization, deposition, timing, and
  physical reference tracking.
- Added symmetric fixed +24/-24 degree upper-arm-roll motor realizations and
  verified that they predict distinct proprioceptive outcomes.
- Expanded the bounded sensor profile from neutral Cartesian IK to neutral plus
  fixed ±roll, while retaining dynamic roll sweeps in the broader research
  configuration.
- Kept curve support and bodily realization separate: no curve reward, roll
  reward, or global contact-pressure preference was added.
- Restored paint-bearing repeated smoke execution with an explicit equal
  preserve/reload policy prior for the initially empty compact brush belief.

### 2026-08-06: provisional sensor-only MuJoCo painting loop

Technical record: `docs/PROVISIONAL_SENSOR_SIMULATION_2026-08-06.md`.

- Added the opt-in `provisional-sensor-simulation-v0` profile. It requires
  MuJoCo, a registered initial camera likelihood, and a body posterior.
- Removed the live `ArmPainterSim` snapshot from this policy path. Forecasts
  use a fresh MuJoCo/material template with independent grain and brush seeds,
  then initialize body/material/compact-brush state from frozen beliefs.
- Kept `_planner_state` and oracle bootstrap forbidden. Accepted camera
  transitions are the only live transition targets.
- Added a bounded smoke configuration: 8 policies, depth 1, one Cartesian-IK
  realization, one forecast particle, no passage proposals, and reduced native
  camera renders. The resolution override is a declared simulation-throughput
  approximation, not an operational sensor-equivalent camera claim.
- As of 2026-08-07 that profile has three motor realizations (neutral IK and
  fixed ±24-degree roll), symmetric straight/curved proposal support, and a
  declared 30 Hz counterfactual integration approximation.
- Demonstrated two repeated selected strokes, two causal camera corrections,
  two replay transitions, and nonzero process coverage. Exact coverage was
  test-only evaluation, not an inference input.
- Fixed an in-episode MuJoCo contact-release reset that rewound sensor time;
  full episode resets still begin at zero.

### 2026-08-06: action-conditioned transition/camera clock

Technical record: `docs/ACTION_CONDITIONED_CAMERA_LOOP_2026-08-06.md`.

- Added `spatial-action-transition-prior-v0`, which propagates the frozen
  material posterior through the learned action/motor transition and increases
  uncertainty without creating an observation or VFE record.
- Added `action-conditioned-camera-update-v0`. A completed action advances the
  material and compact brush transition priors, then blocks another planning
  pass until a registered post-action camera exposure updates the posterior.
- The MuJoCo runtime records the capture boundary after the completing physics
  step and automatically polls through declared camera latency. Frames captured
  before the boundary are rejected even if delivered later.
- The eligible camera update retains its separate VFE decomposition and adds
  one sensor-derived transition to replay. The brush likelihood remains
  prior-only because its local camera-derived deposition statistic is not yet
  connected.
- Kept the default policy loop fail-closed for substrate/model forecast state,
  contact/compliance inference, persistent brush history, and the native plant
  sensor adapter.

### 2026-08-05: brush-posterior motor-forecast initialization

Technical record: `docs/BRUSH_FORECAST_INITIALIZATION_2026-08-05.md`.

- Froze the selected `BrushLoadBelief` revision for each planning pass and
  included both posterior moments and provenance in the forecast cache key.
- Forecast particle zero now uses the compact load/average-pigment means and a
  deterministic representative microstructure. Later particles sample the
  declared diagonal moments and independent `brush-microstructure-prior-v0`
  noise under request-derived seeds.
- Preserve forecasts now require a brush belief. Reload forecasts apply the
  explicit reload transition. Neither path continues the live brush RNG or
  initializes from its exact bristle realization.
- Kept the limitation explicit: held paint, persistent bristle history, and
  tip-lag history are collapsed; the microstructure prior and brush likelihood
  are provisional simulation-only. At that checkpoint the live sensor loop
  remained fail-closed; the bounded integration part is superseded by the
  2026-08-06 provisional-sensor entry above.

### 2026-08-05: material-posterior motor-forecast initialization

Technical record: `docs/MATERIAL_FORECAST_INITIALIZATION_2026-08-05.md`.

- Added version/calibration/revision provenance to `SpatialCanvasState` and
  froze one material posterior for each planning pass and forecast cache key.
- Forecast particle zero now starts from the spatial posterior mean for
  thickness, wetness, black pigment mass, and surface tone. Later particles
  sample its declared diagonal variance independently of hidden live material.
- Samples are drawn at posterior-cell resolution, upsampled piecewise
  constantly, and physically projected. Coverage and ground contrast are
  recomputed from primary material factors rather than sampled as extra
  evidence; black pigment mass is constrained not to exceed thickness.
- Kept the live policy loop fail-closed. At that checkpoint substrate grain,
  brush realization/RNG, contact-to-compliance initialization, model
  parameters, and continuous camera scheduling remained open; the brush
  initialization and camera-scheduling parts are superseded by the entries
  above.

### 2026-08-05: body-posterior motor-forecast initialization

- Connected the MuJoCo physical sensor packet to `BodyStateEstimator` at
  runtime initialization and after every physics step. Body VFE complexity and
  negative log likelihood remain separately logged.
- Each planning pass freezes one `BodyBeliefSnapshot`; motor particle zero uses
  the posterior q/qvel mean and later particles sample its declared diagonal
  uncertainty under an independent seed. Forecasts cannot continue the live
  native process RNG.
- Added version/calibration provenance to body snapshots and execution
  forecasts. The current MuJoCo likelihood is explicitly provisional
  simulation-only and not hardware-calibrated.
- Kept the default policy loop fail-closed. At that checkpoint exact
  material/brush/model context, contact-to-compliance initialization,
  continuous camera scheduling, and MuJoCo parameter uncertainty remained
  open; the material-field, compact-brush, and camera-scheduling parts are
  superseded by the entries above.

### 2026-08-04: selected-plant motor forecast alignment

- Removed the compatibility hook that silently replaced MuJoCo with
  `native-abstract-v0` during deep-copied policy forecasts.
- MuJoCo forecast particles now own independent `MjData` under the same
  immutable `mujoco-robstride-electromechanical-v4` model used for execution.
- Proprioceptive forecasts use physical MuJoCo positions, targets, joint limits,
  current, torque, velocity, contact, and per-joint RobStride/MJCF scales.
- Added explicit backend, initialization, and approximation provenance. The
  historical path at this checkpoint remained `baseline-oracle-v0` because it
  started from exact process state; MuJoCo body-parameter uncertainty was also
  open. The 2026-08-05 entry supersedes the joint-state-initialization part of
  that limitation.

### 2026-08-04: formal-reference progress, proposal recovery, and AI-111 result

- Accepted AI-104 and AI-105 through
  `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`. The summary VFE diagnostic
  now uses a declared 4096-sample reporting-only budget (measured maximum error
  0.02317 nats over seeds 0-4) whose RNG draw cannot perturb later stochastic
  work. The harness covers scalar/multivariate VFE, precision sweeps, local
  identity, and an enumerated deterministic/ambiguous/epistemic/preference EFE
  matrix. AI-106 is now ready. Low-coverage terminal behavior and the AI-110
  composition decision remain explicitly open.
- Added an independent reference-oracle harness covering terminal coverage
  risk, transition information gain, policy posterior normalization, summary
  and spatial VFE, and motor EFE decomposition. It found and forced the removal
  of a motor-ambiguity double count.
- Added per-modality Gamma precision beliefs, explicit per-channel units, a
  gap-progress stop-prior factor, a stronger compression-gap baseline family,
  and a motion-manifold bootstrap with named approximations and ablations.
- Recorded that the precision update did not produce the hoped modality
  ordering, motor normalization materially flattened the realization
  posterior, and the composition bootstrap needs a much larger explicit
  gradient budget than the default supplies.
- Began an amortized learned policy-proposal distribution conditioned on canvas
  and relational beliefs. Sampling, training, checkpoint, and diagnostic paths
  exist; dedicated tests and documentation reconciliation were not completed
  before the work stopped. The handoff recovery added a 14-test suite and fixed
  learned-density support rejection plus private-RNG checkpoint continuation.
  A subsequent 360-cell AI-111 audit found that stop posterior mass and
  deep-horizon winning geometry do not converge under the tested candidate
  budgets. The current result is therefore explicitly
  `Q(pi | sampled candidate set S)`, not a proposal-invariant posterior. Its
  mixture weight remains zero and it is not an accepted solution to
  finite-proposal bias.

### 2026-07-28: MuJoCo physical draft, backend, and contact-driven paint

- Reframed S1 from a future exact co-located abstract clone to the implemented
  `mujoco-robstride-electromechanical-v4` physical draft with a named logical
  canvas retarget.
- Implemented the S2 live MuJoCo backend path, electromechanical RobStride
  approximation, direct runtime selection, telemetry, XML-driven frontend
  geometry, and Python-owned paint boundary.
- Added separated shoulder axes, reachable canvas/keyframe tests, a half-inch
  brush, axial compression, isotropic tangential friction, and lumped
  tangential brush compliance.
- Removed the controller paint-permission gate. Brush loading is material
  state, while actual contact and pressure determine deposition continuously.
- Left T-208 viewer acceptance, T-309 matched-backend parity, and T-501
  backend-neutral geometry extraction open rather than treating a running
  simulation as a calibrated digital twin.

### 2026-07-24: observation factorization and plant interface

- Classified thickness, wetness, black pigment mass, and surface tone as the
  four primary spatial material factors.
- Removed deterministic ground contrast and material coverage from spatial
  observation VFE, transition NLL, and EFE uncertainty/information terms.
- Defined normalization and coordinate-unit behavior for baseline likelihoods
  and VFE reports in `docs/OBSERVATION_FACTOR_AUDIT.md`.
- Defined `plant-interface-v1` with separate command, physical-sensor,
  posterior-belief, counterfactual, capability, and evaluation-truth records.
- Kept the copied-simulator motor forecast path labeled nonconforming and
  moved its migration to T-109 rather than claiming the oracle leak was fixed.
- The focused AI-103/T-103 contract suite passed 12 tests.
- The complete test suite then passed 252 tests in 349.09 seconds with exit
  status 0; environment and source identity are recorded in
  `docs/BASELINE_TEST_RESULT_2026-07-24.md`.

### 2026-07-24: variable and sensor-access ledger

- Classified every field crossing the simulator, canvas, plant, motor
  telemetry, contact, brush, kinematics, and execution-forecast boundaries.
- Named the live observation condition `baseline-oracle-v0` and exposed it in
  runtime diagnostics as non-sensor-equivalent.
- Found that motor planning copies true body, material, contact, brush,
  parameter, and RNG state; this blocks non-oracle embodiment claims.
- Added contract tests requiring privileged entries to carry explicit
  blockers and requiring future boundary fields to be added to the ledger.

### 2026-07-24: M0 gate contract and formal baseline specification

- Completed the project-wide validation-gate contract with explicit evidence,
  pass conditions, and stop conditions.
- Added the `baseline-oracle-v0` generative-model specification, including
  factorization, variational families, VFE/EFE mapping, policy/proposal
  separation, and an approximation register.
- Versioned the Python arm and material process as `native-abstract-v0`.
- Added native geometry and material invariant tests; the focused
  native-contract, canvas, and arm suite passed 55 tests.
- The complete suite reported 235 passing tests; the outer command deadline
  fired immediately after pytest printed the summary, so clean process exit
  remains to be repeated with a longer harness deadline.
- Corrected MuJoCo tracker statuses: the clone is ready to implement, but no
  MJCF artifact or clone tests are currently present.

### 2026-07-23: public project foundation

- Replaced the theory-first repository front page with a concise project,
  architecture, status, quick-start, and roadmap overview.
- Moved detailed technical, research, audit, and historical material under
  `docs/`.
- Added a push/PR smoke-test workflow and retained the complete suite as a
  manually invoked integration check.
- Recorded the current test failure and runtime rather than presenting the
  repository as fully green.

### 2026-07-23: M0 operating contracts

- Defined tracker, dependency, status, and acceptance conventions.
- Defined artifact version identities and a machine-readable version manifest.
- Defined experiment-manifest and append-only failure-log contracts.
- Added a milestone dependency and status index.

## Public Update Template

Use one update per demonstrated milestone or meaningful negative result. A
useful GitHub release note or LinkedIn post has five parts:

1. **Question:** the engineering or research question being addressed.
2. **Change:** what was built or changed, in plain language.
3. **Evidence:** a video, plot, benchmark, test, or reproducible artifact.
4. **Limitation:** what the result does not establish and what failed.
5. **Next test:** the specific uncertainty the next step will reduce.

Example structure:

> I am building a robotic painting system to study whether spatial organization
> can emerge from sensorimotor prediction rather than image targets or aesthetic
> rewards.
>
> This week I separated pixel-local paint prediction from slower passage
> planning and added stochastic motor forecasts. In the attached comparison,
> [measured result] changed from [before] to [after] under the same seeds.
>
> This remains a simulation result. The model still has privileged canvas
> access and has not been calibrated against hardware. Next I am testing
> [specific capability or failure].

Prefer short videos, before/after plots, and failure analysis over screenshots
of planning documents. Avoid claiming intelligence, creativity, composition,
or biological plausibility when the evidence only supports a narrower
mechanism.

## Release Checkpoints

Suggested public releases:

- `baseline-v0`: M1-accepted formal and predictive baseline.
- `sensor-model-v0`: M2 sensor-equivalent fixed-view inference.
- `foveated-agent-v0`: M3 active-observation and hierarchy experiments.
- `mujoco-backend-v0`: S2 backend parity artifact.
- `hardware-rig-v0`: first calibrated physical joint or contact rig.

Each release should include a manifest, exact command, representative artifacts,
known failures, and a short result summary.
