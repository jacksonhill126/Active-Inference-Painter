# Project Tracker

This tracker uses milestone-scale tasks, roughly 1-5 day chunks. It is a
planning artifact, not a replacement for the research charter.

> **Perceptual dependency rule (2026-08-12):** tasks involving perception,
> local transition learning, canvas/composition latents, counterfactual mark
> prediction, or corpus schemas must follow
> `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`. The detailed material process is
> retained; the target persistent agent hierarchy is visual. The implemented
> 16x16 six-channel material posterior and its material cVAE are baselines, not
> the selected representation or the owner's original visual VAE.

## Status Vocabulary

- `Backlog`: not started.
- `Ready`: dependencies are satisfied.
- `Active`: currently being worked.
- `Blocked`: waiting on a decision, measurement, dependency, or tool.
- `Validate`: implemented and awaiting tests, review, or acceptance.
- `Done`: accepted and documented.

## Task Template

```md
### T-000 or AI-000 Task title

Status:
Track:
Depends on:
Owner:
Estimate:
Acceptance:
Notes:
```

Every task must include all fields in the template. Use `none` for no
dependencies and `TBD` for an owner or estimate that has not been assigned;
do not leave a field blank. Acceptance must describe observable evidence, not
only an activity such as "investigate" or "implement."

## Identifier Vocabulary

- `T-###`: operations, simulation support, visualization, geometry, hardware,
  and downstream experiment tasks.
- `AI-###`: active-inference research-spine tasks in M1-M3.

Task detail lives in the matching milestone document. This tracker keeps the
dependency, status, estimate, and short acceptance statement synchronized.

## Dependency Rules

- Dependencies name task IDs, support milestones, or research milestones.
- A task dependency is satisfied only when that task is `Done`.
- A milestone dependency is satisfied by its accepted lock/gate task. M0 is
  satisfied when all M0 tasks are `Done`.
- Dependencies must not be circular. A newly discovered prerequisite is added
  to both the milestone plan and this tracker before dependent work proceeds.
- `Ready` means all dependencies are satisfied. `Active` normally requires the
  same; an exception must be recorded in Notes and must not consume or alter an
  unavailable upstream artifact.
- `Backlog` may have unsatisfied dependencies. Use `Blocked` when work was
  selected or started but cannot proceed because of a named dependency,
  decision, measurement, fault, or tool.
- When the same task appears in two planning files, this tracker is
  authoritative for status. Dependencies and estimates must be synchronized
  before the task moves to `Active`.

## Status Transitions And Acceptance

Normal progress is `Backlog` to `Ready` to `Active` to `Validate` to `Done`.
A task may move to `Blocked` from any non-Done state and returns to `Ready` or
`Active` when the blocker is removed. A failed validation returns to `Active`
with the failed criterion recorded. Reopening a `Done` task requires a dated
note explaining what evidence or assumption changed.

Before a task enters `Validate`, its Notes must identify the implementation or
document artifact and the evidence used for every acceptance criterion. Code
tasks include tests and relevant logs; research tasks include manifests and
analysis outputs; planning tasks include link and consistency checks. `Done`
means the acceptance evidence has been reviewed by the named owner. Milestone,
safety, hardware, and research-claim gates require Jackson's explicit
acceptance even when implementation and automated checks are complete.

## M0: Project Operating System

### T-001 Define tracker conventions

Status: `Done`
Track: Operations  
Depends on: none  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Task fields, status meanings, dependency rules, and validation rules are documented.
Notes: Accepted in this tracker on 2026-07-23; field, dependency, transition, and evidence rules are defined above.

### T-002 Define versioning scheme

Status: `Done`
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5-1 day  
Acceptance: Version labels exist for code, MuJoCo XML, CAD, calibration, hardware build, and experiment config.
Notes: Accepted on 2026-07-23 in `planning/VERSIONING.md` with a machine-readable example manifest.

### T-003 Define experiment manifest requirements

Status: `Done`  
Track: Research Ops  
Depends on: T-001, T-002  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Run metadata and required traces are documented.
Notes: Accepted on 2026-07-23 in `planning/EXPERIMENT_MANIFEST.md` with a machine-readable example manifest.

### T-004 Define failure-mode log

Status: `Done`  
Track: Validation  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Failure categories and log fields are documented.
Notes: Accepted on 2026-07-23 in `planning/FAILURE_LOG.md` with an append-only JSONL example.

### T-005 Define validation gates

Status: `Done`
Track: Validation  
Depends on: T-003, T-004  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Each project gate lists required tests, logs, and stop conditions.
Notes: Accepted 2026-07-24 in `planning/VALIDATION_GATES.md`; it defines evidence, pass conditions, and stop conditions for native, inference, MuJoCo, observatory, geometry, dry/wet hardware, and research gates.

### T-006 Create milestone index

Status: `Done`  
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: M0-M8 and S0-S2 are indexed with status and dependency summaries.
Notes: Accepted on 2026-07-23 in `planning/MILESTONE_INDEX.md`; the index distinguishes entry dependencies from later study and parity gates.

## M1: Formal Baseline And Inference Audit

### AI-101 Write the executable generative-model specification

Status: `Done`
Track: Active Inference/Formalization  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Every implemented belief, likelihood, transition, preference, precision, policy prior, proposal, VFE term, and EFE term maps to a declared factor or named approximation.
Notes: Accepted 2026-07-24 in `docs/GENERATIVE_MODEL_SPEC.md` as `baseline-oracle-v0`; current privileged observations and unresolved proposal/composition semantics are explicitly recorded.

### AI-102 Create the variable and sensor-access ledger

Status: `Done`
Track: Active Inference/Sensors  
Depends on: AI-101  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Every simulator-to-agent value is classified as physical observation, derived observation, hidden state, evaluation label, or execution-only state.
Notes: Accepted 2026-07-24 in `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md` and `planning/variable-sensor-access-ledger.json`; contract tests enforce dataclass field coverage and the runtime oracle-baseline declaration.

### AI-103 Audit observation independence and units

Status: `Done`
Track: Active Inference/Observation Model  
Depends on: AI-101, AI-102  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Derived material channels, likelihood independence, units, normalization, and provisional observation terms are explicitly decided.
Notes: Accepted 2026-07-24 in `docs/OBSERVATION_FACTOR_AUDIT.md`; contrast and coverage are deterministic views rather than separate spatial likelihood evidence, spatial VFE reports nats per independent cell-channel, and eight focused tests pass.

### AI-104 Verify VFE against tractable reference models

Status: `Done`
Track: Active Inference/Validation  
Depends on: AI-101, AI-103  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Analytic linear-Gaussian fixtures independently verify posterior moments and VFE decomposition across precision regimes.
Notes: Accepted 2026-08-04 in `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`. Scalar and multivariate conjugate fixtures verify posterior moments, all VFE components, analytic minimization, four-order precision behavior, and the local outside-patch identity prior. The declared 4096-sample summary VFE report has measured maximum absolute error 0.02317 nats over seeds 0-4 inside the accepted `+-0.05` nat band; spatial VFE remains within `1e-6` of independent integration.

### AI-105 Verify EFE against enumerated or Monte Carlo references

Status: `Done`
Track: Active Inference/Validation  
Depends on: AI-101, AI-104  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Independent fixtures verify risk, ambiguity, information gain, policy priors, signs, and posterior normalization.
Notes: Accepted 2026-08-04 in `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`. The independent matrix covers deterministic, ambiguity-only, purely epistemic, and preference-dominated controls plus policy posteriors with and without priors. Terminal, transition, motor, and posterior identities pass. Low-coverage terminal behavior moves to AI-106; the composition-gap preference is explicitly deferred to AI-110 rather than approved by this task.

### AI-106 Stress-test terminal coverage and stopping

Status: `Done`
Track: Preferences/Validation  
Depends on: AI-105  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Beta moment matching, clamps, alternatives, terminal-only application, and immediate-stop support have a documented decision.
Notes: Closed 2026-08-11 in
`docs/TERMINAL_COVERAGE_STOPPING_ACCEPTANCE_2026-08-11.md`. Direct Monte Carlo
agrees with the analytic Beta--Beta KL when the forecast is actually Beta, but
a broad bounded logit-normal with the same measured mean/variance differs by
23.6 nats. The near-blank concentration floor preserves mean while reducing
represented variance by more than 500x and changing risk from 53,248 to 892
nats. Terminal-only application, finite always-admissible immediate stop, and
declared-not-learned preference parameters pass. Decision: retain the terminal
coverage preference and stop prior, but replace the single moment-matched Beta
forecast before M2 acceptance with a converged sample-based estimate or a
calibrated richer bounded family. Focused result: 48 tests passed.

### AI-108 Build a leakage-resistant baseline corpus

Status: `Done`
Track: Data/Validation  
Depends on: AI-102, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Live-scale train/validation/test transitions are split by trajectory before patch extraction and stratified by material, action, reach, and motor conditions.
Notes: The 2026-08-07 action-space correction adds continuously sampled signed
quadratic curvature and fixed/swept roll realization conditions. Corpus strata
must explicitly cover straight and several positive/negative curvature bands,
vertical direction, neutral/fixed
±roll, and both roll-sweep directions; splitting must occur before patch
extraction so neighboring samples from one physical curve cannot leak across
train/validation/test.
Implementation baseline 2026-08-10: `trajectory-posterior-corpus-v1` records
full camera-derived posterior trajectories before patch extraction;
`split_manifest.json` assigns whole trajectories with best-effort tone,
curvature, region, motor-realization, and coverage stratification. Synthetic
round-trip and no-overlap tests pass. On 2026-08-11 the schema advanced to
`trajectory-posterior-corpus-v2`: trajectory records may also carry frozen
compact brush-load belief moments, and train-only patch materialization can
feed the shadow `conditional-local-material-transition-cvae-v0`. Neither the
schema nor the cVAE admits exact simulator material state to live inference.
Accepted 2026-08-11. The canonical manifest
`runs/corpus-ai108-combined-20260811/split_manifest.json` contains 16
provenance-complete v2 trajectories and 256 transitions split 10/3/3 before
patch extraction. Every split contains both tones, blank/existing material,
fresh/overlap, interior/edge, narrow/broad, short/long, five signed curvature
bands, vertical direction, center/outer reach, neutral IK, fixed +/-roll, and
dynamic +/-roll sweeps. All transitions carry compact brush context, deny
process-truth training input, and attest a 256 process canvas separately from
the reduced 64 inference/acquisition approximation and 16x16 posterior grid.
The multi-root manifest passes a cVAE consumer smoke. Bulk wetness remains
structurally unobserved by the camera likelihood, and all trajectories are
fixed-horizon truncations rather than genuine stops; those are explicit
downstream gaps, not manufactured corpus labels. See
`docs/AI108_CORPUS_TECHNICAL_2026-08-11.md` and
`docs/AI108_CORPUS_OWNER_BRIEF_2026-08-11.md`.

### AI-107 Calibrate transition and precision approximations

Status: `Done`
Track: Uncertainty  
Depends on: AI-103, AI-108  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Held-out NLL, interval coverage, z-scores, and OOD disagreement distinguish aleatoric, ensemble, likelihood, and fixed-precision terms.
Notes: Completed 2026-08-11 on AI-108's trajectory-isolated 160/48/48
train/validation/test transition corpus. Substantial CNN (2,500 steps),
fixed-condition CNN (2,500 steps), and shadow cVAE (3,000 steps) checkpoints
were evaluated with validation-only variance scaling and a frozen test set.
Both models failed the provisional M2 interval gate: nominal 90% intervals
covered about 99.4% of test residuals, including an action-footprint-only
diagnostic. The meaningful fixed-to-dynamic-roll OOD disagreement ratio was
1.087 versus a declared 1.50 requirement. cVAE latent variance was effectively
collapsed and did not outperform the CNN. All checkpoint precision entries
had zero observations and are reported as declared unobserved priors, not
inferred posteriors. Wet-over-wet remains structurally unavailable to the
camera likelihood, and all patches occupy one size bin. AI-107 is complete as
an evidence task; its negative calibration result blocks any M2 calibrated-
model claim and makes AI-109 the next modeling task. See
`docs/AI107_UNCERTAINTY_CALIBRATION_TECHNICAL_2026-08-11.md` and the owner
brief.

### AI-109 Establish predictive learning curves

Status: `Done`
Track: Modeling/Validation  
Depends on: AI-107, AI-108  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Data/capacity curves determine whether current local and hierarchy models are underfitting, data-limited, or overfitting.
Notes: The local-model branch was measured on 2026-08-12 with 27 unique runs,
three seeds, nested 3/6/10-trajectory training subsets, fixed 3-trajectory
validation/test splits, CNN capacity and ensemble axes, the shadow cVAE, and a
normalized identity-plus-consequence mixture. More data modestly improved CNN
test NLL, generic width/depth did not, five ensemble members improved density,
and the material-posterior cVAE did not materially outperform the CNN. That
comparison did not train on registered image patches and does not test the
owner's proposed visual cVAE. The mixture improved mean
test NLL by 1.392 nats per independent material cell-channel and retained low
eight-mark rollout error, but every seed still failed exact predictive-mixture
calibration; it remains shadow-only. The hierarchy branch is not yet
admissible because AI-108 contains zero policy-selected stops: every endpoint
is a fixed-horizon truncation. A 2026-08-12 collection pilot produced six
policy-selected stops and two truncations at a 192-step cap, demonstrating
that genuine-stop collection is feasible. It retained coarse final posteriors,
not the registered pre/post image stream required by the selected hierarchy.
Decision 2026-08-26: this closes the M1 question with a measured negative/
inconclusive result for the then-current coarse-material models. Requiring a
new terminal registered-image corpus and tone/edge/mass hierarchy here created
a circular M1 -> M2 -> M1 dependency. That unfinished visual work is preserved
under AI-205, AI-206, AI-208, and AI-214. Do not relabel truncations as
completed paintings. See
`docs/AI109_PREDICTIVE_LEARNING_CURVES_TECHNICAL_2026-08-12.md` and the owner
brief, plus `docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md`.

### AI-110 Resolve the composition-preference closed loop

Status: `Done`
Track: Active Inference/Preferences  
Depends on: AI-101, AI-105, AI-109  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: A painting-level structural preference is formally approved under
frozen/cross-fitted safeguards or disabled. The candidate formulation selects
patches whose observations are locally difficult to explain or incompatible
with slower painting structure, then prefers predicted outcomes that reduce
that posterior predictive mismatch. The material-transition predictor may
forecast mark consequences but is not itself the painting objective.
Notes: Closed 2026-08-26 with the allowed disabled decision. The canonical
`m1-formal-policy-baseline-v0` sets the legacy coarse-material hierarchy,
compression-gap preference, canvas/relational transition precisions,
passage-trajectory terms, and gap-progress stop prior off by default. Explicit
opt-in is identified as `legacy-material-composition-diagnostic-v0`; it is not
an accepted visual composition model. A future visual structural preference
requires a new M2 admission decision with frozen/cross-fitted safeguards. See
`docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md` and the separate owner brief.

### AI-111 Separate proposal distributions from policy priors

Status: `Done`
Track: Policy Inference  
Depends on: AI-101, AI-105  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Candidate frequency, intended policy prior, proposal density, and finite-budget bias have explicit semantics and convergence tests.
Notes: Closed 2026-08-04 with a negative convergence result; a negative result
is the documented decision required by this audit task, not an unfinished
status. A bounded implementation experiment proceeded ahead of AI-105
acceptance. `proposal.py` adds an amortized belief-conditioned mark/passage
proposal trained toward the existing base-EFE posterior and mixed with the
hand-written sampler; its emission mix defaults to zero and it never enters
EFE, VFE, preferences, or the normalized policy posterior. It does not correct
finite-candidate bias. `tests/test_proposal.py` covers normalized/support-
bounded densities, empirical hand-sampler agreement, exact zero-mixture/RNG
parity, mixed-source attribution, posterior-only training, checkpoint
continuation, and unchanged EFE under the default gate.
`tests/test_proposal_convergence.py` plus
`docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md` add an equal-EFE control and a
360-cell candidate-count/horizon/seed/mixture audit. Stop mass and deep-horizon
winning geometry do not converge under the tested budgets. The current
posterior is accepted only as `Q(pi | sampled candidate set S)`; mixtures are
computational budget splits, learned emission remains zero, and AI-306 requires
a complete base-measure/prior/proposal correction before proposal-invariant
claims.

### AI-112 Define online learning and inheritance semantics

Status: `Done`
Track: Continual Learning/Research Ops  
Depends on: AI-101, AI-108  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Shared parameters, optimizer, replay, calibration, episodic beliefs, and canvas history have separate persistence and reporting rules.
Notes: Checkpoint schema 7 embeds `online-learning-inheritance-v0` and exposes
separate exact-individual-resume, shared-parameter-initialization, and
shared-pretraining-continuation modes. Episode beliefs reset; individual replay
and calibration persist only for an individual resume; model-imagined
rollouts are rejected from all runtime replay APIs. The retention and fixed
anchor held-out forgetting protocol is defined in
`docs/ONLINE_LEARNING_INHERITANCE_2026-08-27.md`; AI-114 must execute it rather
than reinterpret its ownership semantics.

### AI-113 Profile inference, rollout, and learning separately

Status: `Ready`
Track: Performance/Feasibility  
Depends on: AI-109  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: State inference, EFE, motor forecasting, hierarchy, training, serialization, and rendering have separate timings and a ranked optimization plan.
Notes: Phase timers and a 1/4/6/8-worker collection benchmark harness now exist,
but a representative measured profile and ranked optimization decision do not.
AI-109 closed for M1 on 2026-08-26, so this task is ready; existing harness code
alone is not acceptance.

### AI-114 Capture reproducible baseline replicas

Status: `Blocked`  
Track: Research Ops  
Depends on: AI-104, AI-105, AI-106, AI-107, AI-109, AI-110, AI-111, AI-112, AI-113
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: At least three fixed-seed runs archive manifests, beliefs, VFE/EFE, calibration, proposal statistics, telemetry, failures, and canvases.
Notes: The 2026-08-26 gate repair defines
`m1-formal-policy-baseline-v0`, but no three-replica artifact set exists for
that profile. AI-112 is complete; replica capture remains blocked on AI-113's
representative phase profile and the resulting bounded run budget.

### AI-115 M1 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-114  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: Formal, sensor-access, VFE/EFE, calibration, terminal, composition, proposal, and inheritance decisions are accepted before M2.

## M2: Calibrated Multiscale Generative Model

### AI-201 Define the sensor-equivalent M2 observation package

Status: `Active`
Track: Sensors/Active Inference  
Depends on: M1  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Fixed camera, encoders, current, and contact observations have
declared channels, units, rates, noise, and physical analogues; the corpus
contract retains registered rectified pre/post images, timing/calibration/mask
provenance, action crops, brush/action/motor context, and termination provenance.
Notes: Boundary enforcement began 2026-07-28: the live default is now
`sensor-boundary-v0`, which skips oracle bootstrap and blocks policy inference,
learning, and planner-state construction from `ArmPainterSim`. The legacy
`oracle_material_state` path requires explicit diagnostic opt-in. AI-201
remains not accepted because the complete bodily observation package and
physical calibration are not yet defined. A
`provisional-multiview-v2` MJCF rig and role-dependent registration interface
were added 2026-07-29, establishing versioned geometry, grayscale channels,
model-input resolutions, rates, roles, availability, an ideal-pinhole
normalization contract for canvas views, a separate overhead brush-standoff
edge profile, and optical-frame-aligned generic camera housing envelopes
without claiming sensor equivalence or final mechanical mounts. The current
provisional contract is 512 x 512 at 30 Hz for each continuous oblique view,
512 x 512 at 5 Hz/on-demand for parked inspection, and 640 x 480 at 60 Hz for
the standoff profile. On 2026-07-30 the XML contract gained explicit
per-camera provisional read/signal noise, latency, dropout, and quantization
parameters plus observation/specular model versions. These assumptions are
declared but not calibrated.
On 2026-07-31 the contract advanced to `provisional-multiview-v3`: the owned
OM System OM-1 and Sony A7R II are the provisional opposing oblique pair,
captured as separate 3840 x 2160 rolling-shutter HDMI streams at 30 Hz. The
contract distinguishes native acquisition, 512 x 512 global inputs, and
native-derived 256 x 256 foveae. The extra fixed head-on and overhead
global-shutter cameras remain planned rather than being assigned fictitiously
to either owned body. Later that day, `provisional-multiview-v4` fixed the
owned lens baseline at OM-1/25 mm and A7R II/35 mm in Super 35 mode. Nominal
16:9 intrinsics and 7%-retracted mount poses now leave physical framing margin
while preserving 100% combined contact-tip visibility. Intrinsics remain
provisional until checkerboard calibration of the actual HDMI streams. The
metric A3 target generator and native-frame Brown-Conrady solver now exist in
`active_painter.camera_calibration`, with residual, coverage, tilt-diversity,
and minimum-view acceptance gates. Physical image capture is still pending.
The current selected design baseline supersedes those historical camera drafts:
`provisional-compact-dual-imx296-v1` is exactly two selected-but-not-yet-
purchased Raspberry Pi Global Shutter Cameras (Sony IMX296), left/right on one
rigid crossbar, with provisional 4 mm CS lenses. Camera centers are 650 mm
forward of the canvas, 300 mm to either side, and 220 mm above canvas center.
There is no initial inspection, overhead, or profile camera. Intrinsics, noise,
timing, 60 Hz rate, synchronization, housings, and mounts remain uncalibrated
simulation/design assumptions, so AI-201 remains active rather than accepted.

### AI-202 Implement the fixed-view observation generative process

Status: `Active`
Track: Generative Process/Sensors  
Depends on: AI-201, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: The process emits declared provisional camera observations while
hidden material arrays remain evaluation-only; sensor-equivalence claims wait
for purchased-camera calibration and validation.
Notes: Geometry/preprocessing work began 2026-07-29 with forward/inverse
canvas homographies, frustum masks, image rectification, a camera-clear park,
arbitrary world-point projection for an edge-profile camera, and XML-derived
web metadata. A reproducible 9 x 9 x 3 MuJoCo contact-pose sweep now reports
100% combined tip visibility from the opposing oblique cameras across 243
sampled poses; the overhead standoff profile also retains 100%. The brief,
figures, per-pose CSVs, approximations, and hardware-class rationale are in
`docs/CAMERA_OBSERVABILITY_BRIEF.md`. On 2026-07-30 a versioned
`CameraObservationProcess` began emitting multi-rate grayscale
`CameraObservationBundle` records: MuJoCo provides geometry and occlusion,
the superficial Python canvas image is composited only through visible canvas
pixels, canvas views are homography-rectified, and XML-declared provisional
noise/latency/quantization plus a weak rendered-lighting specular
approximation are applied. Exact segmentation remains process-internal and is
not present in agent-facing records. On 2026-07-31,
`camera-observation-interface-v1` added XML-declared native-resolution
rendering, independent 512 x 512 global derivation, and explicit 256 x 256
foveal requests sampled directly from native pixels. Fovea requests are
addressed in canvas UV, have no default, and may cite only sensor-posterior,
policy-prediction, or operator-diagnostic selection bases; exact simulator
pose, contact, visibility, segmentation, and material state are prohibited.
The web viewer now renders the current delivered foveal extent and a fading
canvas-registered delivery trace. Its 10-second default is labeled as a
visualization fallback; it will follow a future declared foveation-memory
horizon. Preview clicks are explicitly `operator_diagnostic` requests and do
not satisfy the still-blocked active-gaze-policy work in M3.
AI-202 remains not accepted: physical capture and calibration are absent and
the photometric parameters are uncalibrated. A provisional analytic likelihood
now consumes these products, but no learned encoder or physical-camera
validation exists yet.
The 2026-08-07 compact-rig correction removes the old inspection/overhead/
profile products from the initial baseline and makes the two IMX296 oblique
views the only camera products. MJCF, runtime metadata, pose-sweep evidence,
tests, and support documentation now use `provisional-compact-dual-imx296-v1`.
This is still a provisional simulation process, not sensor-equivalent evidence.

### AI-203 Specify and implement the observation likelihood

Status: `Active`
Track: Generative Model  
Depends on: AI-201, AI-202  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Registered-image, action-conditioned visual-transition,
proprioceptive, current, and contact likelihoods are explicit, calibrated, and
avoid double-counting deterministic transforms such as image intensity and its
edge map.
Notes: A bounded body-likelihood slice was added 2026-07-29 as
`body-inference-v0`: encoder position/velocity and optional contact-switch/force
factors are explicit and require a versioned, nondefault precision profile.
Current, voltage, deflection, temperature, and faults remain explicitly
unassimilated. On 2026-07-31, `camera-spatial-likelihood-v0` connected
registered global/foveal grayscale products to the spatial posterior through
an explicit nonlinear superficial-appearance likelihood. A mean-field
inlier/outlier latent handles occlusion without segmentation; correlated
global/foveal products from one exposure are mosaicked before one update; VFE
logs state complexity, occlusion complexity, and expected negative log
likelihood separately. Thickness and surface tone receive image evidence,
while wetness and bulk pigment remain transition-prior factors. The XML owns
the provisional per-camera precision and inlier assumptions. A scalar camera-
derived mark-deposition likelihood and VFE fixture still exist separately for
brush loading, but no image-derived mark statistic supplies it yet. AI-203 is
not accepted because physical calibration and current/deflection likelihoods
are absent.

### AI-204 Build the compact state-inference path

Status: `Active`
Track: Variational Inference  
Depends on: AI-203  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: An explicit compact visual/body posterior fuses transition priors
with permitted observations and reports calibrated uncertainty. Persistent
canvas-wide wetness is excluded; any material interaction latent is local,
action-conditioned, and ephemeral.
Notes: `BodyStateEstimator` now maps `PhysicalSensorPacket` to
`BodyBeliefSnapshot` with a constant-velocity transition prior, conjugate
Gaussian/Bernoulli updates, and global/per-factor VFE decomposition. It never
receives the simulator process object. `BrushLoadingModel` now maintains
persistent compact load/average-pigment beliefs for dedicated white and black
brushes, with explicit depletion/reload transitions and preserve/reload policy
inference. The process has matching finite reservoirs and pure-color reloads.
The camera-conditioned material posterior now exists and is reachable through
`SpatialActiveInferencePainter.assimilate_camera_observation` and the driver
sensor boundary. Camera evidence is not yet reduced to the local statistic
used by the brush posterior. On 2026-08-05 the MuJoCo runtime began updating
the body posterior from each `PhysicalSensorPacket`; planning now freezes that
revision and initializes motor-forecast q/qvel particles from its mean and
diagonal variance under an independent future-noise seed. The named MuJoCo
likelihood is provisional simulation-only and body VFE is logged separately.
Also on 2026-08-05, planning began freezing one `SpatialCanvasState` revision
and initializing forecast thickness, wetness, black mass, and surface tone
from its mean and diagonal particles. Material samples cannot read the hidden
live fields, and derived coverage/contrast are recomputed. Planning now also
freezes the selected `BrushLoadBelief`: particle zero uses load/average-pigment
means and a representative microstructure, while later particles sample its
diagonal moments and independent `brush-microstructure-prior-v0` noise. The
live brush RNG and exact bristle realization no longer initialize forecasts.
Held paint and persistent bristle history remain collapsed, and the camera
path does not yet supply the local mark statistic used by the brush
likelihood. The forecast also does not yet initialize substrate grain, contact
compliance, or model parameters from beliefs. On 2026-08-06,
`action-conditioned-camera-update-v0` began pairing each recorded executed
action with `spatial-action-transition-prior-v0`, a post-physics camera capture
boundary, stale-frame rejection, automatic MuJoCo delivery polling, and one
registered camera posterior update before another plan can start. Prediction
does not create VFE; the camera update logs it and supplies one sensor-derived
replay transition. The brush path advances its depletion prior but still lacks
the camera-derived local mark statistic needed by its likelihood. Later on
2026-08-06, `provisional-sensor-simulation-v0` supplied a bounded opt-in
integration path: initial camera/body gating, a separately constructed MuJoCo
and material forecast template with independent grain/brush seeds, frozen
posterior initialization, 8 candidates, depth 1, one forecast particle, and
three conditional motor realizations: neutral Cartesian IK plus fixed
upper-arm roll at approximately +24 and -24 degrees. `_planner_state` and
oracle bootstrap remain forbidden.
The repeated-stroke integration test closes two action/camera cycles and uses
exact process coverage only as an evaluation assertion. This makes AI-204
runnable for data collection but does not accept it: the transition model has
no approved sensor corpus/checkpoint, the brush likelihood remains prior-only,
and the fixed context/compliance priors are uncalibrated. The 2026-08-10
anisotropic round-brush process adds exact aggregate deflection,
stick/slip, tangential-load, and angle-dependent-footprint labels. They are useful
training/evaluation targets but are not agent observations. Counterfactuals
still reuse those process equations under independently initialized state;
the missing stochastic brush-contact posterior and transition/observation
likelihood remain part of AI-204/AI-205 acceptance. On 2026-07-29 the
six-aggregate summary planner was formally marked
`obsolete_compatibility_fixture`; it is not an acceptable compact posterior
for AI-204.
Direction correction 2026-08-12: the frozen-state and sensor-boundary work is
retained, but completion no longer means expanding the persistent canvas-wide
material posterior. The selected target is visual/body inference with an
optional ephemeral local interaction latent from fresh image evidence; see
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

### AI-205 Align local visual dynamics training with live execution

Status: `Blocked`  
Track: Transition Model/Data  
Depends on: AI-108, AI-202  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Corpus and live execution share camera registration, physical
pixel scale, action-support geometry, and held-out visual calibration. The
corpus retains registered pre/post image patches and required provenance.
Notes: The 2026-08-11 `conditional-local-material-transition-cvae-v0` can be
trained offline on train-only v2 trajectory patches and reports held-out
metrics beside the CNN baseline. It remains shadow-only and has not been
admitted to live/counterfactual dynamics; physical pixel-scale equivalence and
held-out calibration are still absent. It is a material-posterior experiment,
not completion or rejection of the visual-dynamics task.

### AI-206 Validate multi-step visual prediction

Status: `Blocked`  
Track: Transition Model/Validation  
Depends on: AI-204, AI-205  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: One-mark through passage-length visual rollouts have measured
normalized likelihood, tone/boundary fidelity, error growth, and uncertainty
calibration under an approved approximation.

### AI-207 Define the multiscale latent clocks and messages

Status: `Blocked`  
Track: Hierarchical Active Inference  
Depends on: AI-101, AI-204  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Pixel, mark, tile, passage, and painting levels have explicit priors, likelihood messages, posterior updates, precision, and persistence.
Notes: The hierarchy must learn feature contents through camera-conditioned
predictive likelihoods rather than replace the obsolete six summaries with
another hand-selected global feature list. Hand-defined material variables are
restricted to local physical prediction, diagnostics, or declared terminal
readouts/preferences. Latents require held-out predictive and temporal
intervention evidence. Persistent canvas-wide wetness is prohibited in the
target hierarchy; an ephemeral local interaction latent may be inferred from a
fresh target image when it earns predictive necessity.

### AI-208 Make the global canvas latent predictively necessary

Status: `Blocked`  
Track: Hierarchical Modeling  
Depends on: AI-206, AI-207  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: A modest hierarchy improves held-out future prediction and responds correctly to spatial interventions without collapse.
Notes: `spatial_material` is an interim low-level baseline, not the accepted
global representation. AI-208 must establish predictive necessity for flexible
learned visual tone/edge/mass latents before they enter painting-level inference
claims, including tests that oblique and curved boundaries survive compression
without square-cell artifacts.

### AI-209 Replace or demote deterministic relational beliefs

Status: `Blocked`  
Track: Relational Inference  
Depends on: AI-207, AI-208  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Region existence, assignment, and relations are uncertain, or deterministic slots are relabeled and their claims restricted.

### AI-210 Validate slow-state temporal persistence

Status: `Blocked`  
Track: Hierarchical Inference/Validation  
Depends on: AI-207, AI-208, AI-209  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Slow states predict later events beyond the current frame and pass freeze, shuffle, reset, and update-clock tests.

### AI-211 Implement the M1 structural-preference decision

Status: `Blocked`  
Track: Preferences  
Depends on: AI-110, AI-208, AI-210  
Owner: Jackson/Codex  
Estimate: 2-4 days  
Acceptance: Only the approved structural preference is enabled, separately logged, tested for self-reinforcement, and paired with a zero-preference ablation.

### AI-212 Separate shared pretraining from individual development

Status: `Blocked`  
Track: Continual Learning  
Depends on: AI-112, AI-205, AI-210  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Shared low-level parameters may pool data while every agent retains separate posteriors, histories, slow states, and online adaptation records.

### AI-213 Build the affordable parallel data and training path

Status: `Blocked`
Track: Performance/Data  
Depends on: AI-113, AI-205, AI-212  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Parallel lower-level experience and batched training improve throughput without contaminating test data or canonical development.
Notes: Implementation proceeded ahead of its dependencies as a bounded support
experiment. The 2026-08-10 baseline adds independent spawned MuJoCo/camera workers,
atomic trajectory shards, train-only post-split patch materialization, central
batched training, held-out NLL reporting, shared-pretraining provenance, and a
1/4/6/8-worker benchmark harness. Unit smoke tests pass. The dependency-level
scientific acceptance remains blocked on its declared dependencies and the
AI-212 developmental interpretation tests. AI-108 now contributes a
provenance-complete 256-transition corpus and measured three-way versus
five-way motor-profile throughput, but that bounded evidence does not close
AI-213 by itself.

### AI-214 Evaluate model size and pretrained readiness

Status: `Blocked`  
Track: Feasibility/Modeling  
Depends on: AI-204, AI-205, AI-206, AI-208, AI-213
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Visual local and hierarchy data/capacity/seed learning curves,
latency, memory, cost, and a narrow frozen-feature adapter define whether
larger or pretrained perception is justified. This owns the visual learning-
curve continuation moved out of M1 AI-109 on 2026-08-26.

### AI-215 Run the M2 capability-gate suite

Status: `Blocked`  
Track: Validation  
Depends on: AI-203, AI-204, AI-206, AI-208, AI-209, AI-210, AI-211, AI-212  
Owner: Jackson/Codex  
Estimate: 3 days  
Acceptance: Observation, local prediction, embodiment, relational representation, and temporal-persistence gates pass under sensor-equivalent input.
Notes: The gate must include the AI-106 correction: terminal coverage risk may
not rely on the current single moment-matched Beta forecast. It requires a
sample-count-converged particle estimate or a calibrated richer bounded family
while retaining the fixed terminal preference and separate stop policy prior.

### AI-216 M2 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-215  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: Sensor-equivalent inference, calibrated dynamics, multiscale uncertainty, temporal persistence, and preference status are accepted before M3.

## M3: Foveated Hierarchical Policy Inference

### AI-301 Specify the foveated generative process

Status: `Blocked`  
Track: Active Vision/Generative Process  
Depends on: M2  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Attention is defined as a budgeted trajectory of spatial access,
scale, camera support, timing, and precision; crop/tile realization, sensor
geometry, periphery, latency, uncertainty, non-leakage, and accessible samples
are explicitly defined without assuming a smooth eye-like crop path.

### AI-302 Implement foveal and peripheral observations

Status: `Blocked`  
Track: Active Vision/Sensors  
Depends on: AI-301, AI-202  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Active, random, and uniform observation modes share one image process without leaking unsampled detail.

### AI-303 Maintain beliefs under partial observation

Status: `Blocked`  
Track: Variational Inference  
Depends on: AI-204, AI-302  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Foveated likelihood updates and unobserved transition uncertainty produce calibrated partial-observation posteriors.

### AI-304 Define attention policies and their EFE

Status: `Blocked`  
Track: Active Inference/Policy  
Depends on: AI-105, AI-303  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Attention risk, ambiguity, and information gain are derived,
separately logged, and resolve controlled reducible ambiguity without saliency
rewards; allocation is distinguished from its crop/tile sensor realization.

### AI-305 Factor attention, mark, passage, and motor policies

Status: `Blocked`  
Track: Hierarchical Active Inference  
Depends on: AI-207, AI-304  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Temporal levels exchange declared priors and evidence while low-level control and hard safety remain outside painting-policy inference.

### AI-306 Correct finite-proposal policy inference

Status: `Blocked`  
Track: Policy Inference/Approximation  
Depends on: AI-111, AI-305  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Proposal density and policy prior are distinct, immediate stop retains support, and finite-budget posterior bias is measured.

### AI-307 Replace hand-shaped continuity scores with transition priors

Status: `Blocked`  
Track: Priors/Embodiment  
Depends on: AI-101, AI-305  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Retained continuity effects are conditional trajectory priors or transitions with a flat-prior ablation, not aesthetic scores.

### AI-308 Formalize bodily viability and energy preferences

Status: `Blocked`  
Track: Embodied Active Inference  
Depends on: AI-105, AI-205, AI-305  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Observable current, temperature, limits, contact, and tracking outcomes have declared priors separate from hard safety and avoid trivial inactivity.

### AI-309 Test whether epistemic action is informative or pathological

Status: `Blocked`  
Track: Active Inference/Validation  
Depends on: AI-304, AI-308  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Ambiguity, noise, misspecification, unreachable state, repeat-fixation, and model-hacking fixtures separate reducible information from novelty.

### AI-310 Make passage and painting timescales causally testable

Status: `Blocked`  
Track: Hierarchical Inference/Validation  
Depends on: AI-210, AI-305  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Freeze, shuffle, reset, and substitution interventions establish whether slow states causally affect later prediction and policy.

### AI-311 Parallelize counterfactual rollout without changing the agent

Status: `Blocked`  
Track: Performance/Inference  
Depends on: AI-113, AI-206, AI-305  
Owner: Jackson/Codex  
Estimate: 3-6 days  
Acceptance: Concurrent policy and motor particles preserve equations, seeds, marginalization, and replay boundaries while reducing measured wall time.

### AI-312 Define the single-agent developmental protocol

Status: `Blocked`  
Track: Research Design  
Depends on: AI-112, AI-212, AI-305  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Inherited parameters, individual beliefs, replay, histories, independent replicas, and online persistence have explicit experimental rules.

### AI-313 Run the core foveation and hierarchy ablations

Status: `Blocked`  
Track: Research Experiments  
Depends on: AI-306, AI-307, AI-308, AI-309, AI-310, AI-312  
Owner: Jackson/Codex  
Estimate: 5-8 days  
Acceptance: Matched foveation, hierarchy, timescale, embodiment, uncertainty, and passage ablations report mechanism measures and complete artifacts.

### AI-314 Evaluate frozen pretrained perception only after baseline

Status: `Blocked`  
Track: Perception/Feasibility  
Depends on: AI-214, AI-313  
Owner: Jackson/Codex  
Estimate: 4-7 days  
Acceptance: At most two frozen encoders are compared through the same probabilistic adapter, data split, and metrics without pretrained policy or aesthetic scores.

### AI-315 Run the M3 capability-gate suite

Status: `Blocked`  
Track: Validation  
Depends on: AI-303, AI-304, AI-306, AI-308, AI-309, AI-310, AI-311, AI-313  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Partial beliefs, informative gaze, proposal convergence, slow-state causality, embodiment, rollout parity, and VFE/EFE logging pass.

### AI-316 M3 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-315  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: The core foveated hierarchical active-inference model and ablations are accepted before compositional claims or visual-transfer expansion.

## S0: Plant Reference Contract

### T-101 Confirm Python arm sim as canonical abstract reference

Status: `Done`
Track: Simulation  
Depends on: M0  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: Native simulator constants, limits, home pose, canvas frame, and known shortcuts are documented.
Notes: Accepted 2026-07-24 as `native-abstract-v0` in `docs/NATIVE_PLANT_REFERENCE.md`; protected by `tests/test_native_contract.py` and existing arm tests.

### T-102 Version and protect canvas material invariants

Status: `Done`
Track: Painting Model  
Depends on: M0  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Tests confirm thickness, wetness, pigment mass, visible tone, white-on-white coverage, and material coverage behavior.
Notes: Accepted 2026-07-24 with added repeated-layer, tone-independence, clear, configurability, and native-contract tests; focused native contract/canvas/arm suite passed 55 tests.

### T-103 Define controller, plant, and policy interfaces

Status: `Done`
Track: Control  
Depends on: T-101  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: `StrokeAction` remains Cartesian/contact intent, IK remains below policy selection, and the backend boundary separates commands, physical observations, inferred state, and simulator-only truth.
Notes: Accepted 2026-07-24 as `plant-interface-v1` in `docs/CONTROL_PLANT_POLICY_BOUNDARY.md` and `src/active_painter/plant_interface.py`; four focused tests pass. The native runtime remains an explicitly nonconforming oracle path until T-109.

### T-104 Record full baseline test result

Status: `Done`
Track: Validation  
Depends on: T-101, T-102, T-103  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: Current planner, arm, canvas, and web tests have a recorded baseline result.
Notes: Accepted 2026-07-24 in `docs/BASELINE_TEST_RESULT_2026-07-24.md`; the complete suite passed 252 tests in 349.09 seconds with exit status 0.

### T-105 Capture baseline telemetry and web-runtime behavior

Status: `Done`
Track: Web/Telemetry  
Depends on: T-101, T-103  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: Default web runtime endpoints, frontend state, canvas image, and telemetry CSV have a recorded baseline.
Notes: Accepted 2026-08-11 in
`runs/baseline/s0-native-abstract-v0-2026-08-11/`. The default native runtime
served finite state, a 256 x 256 canvas PNG, 56 telemetry rows across 96 named
columns, version/robot-model data, and the frontend asset snapshot. The default
sensor boundary correctly failed closed instead of exposing oracle material
state. The current focused runtime/telemetry selection passed 11 tests with
exit status 0; exact config, source fingerprint, hashes, warnings, and one
superseded wrapper-timeout attempt are recorded in the bundle.

### T-106 Document known simulator shortcuts and limitations

Status: `Done`
Track: Documentation  
Depends on: T-101, T-102, T-103  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: Simulator shortcuts are categorized as acceptable baseline, MuJoCo calibration need, or hardware validation need.
Notes: Accepted 2026-08-11 in
`docs/SIMULATOR_SHORTCUT_CLASSIFICATION_2026-08-11.md`. The consolidated table
lists simulator-only observations, representative dynamics, brush/contact and
material simplifications, camera/sensor assumptions, planning approximations,
and counterfactual limits. Every row distinguishes S0 acceptable reference
behavior from MuJoCo calibration work and hardware validation work; acceptable
baseline never means physically calibrated.

### T-107 Define baseline artifact bundle

Status: `Done`
Track: Research Ops  
Depends on: T-003, T-104, T-105  
Owner: TBD  
Estimate: 0.5-1 day  
Acceptance: Baseline bundle contents and location are defined, including test summary, config, telemetry, canvas image, and notes.
Notes: Accepted 2026-08-11. The immutable location/content/manifest convention
is `docs/BASELINE_ARTIFACT_BUNDLE_CONVENTION_2026-08-11.md`; the first complete
instance is `runs/baseline/s0-native-abstract-v0-2026-08-11/` with config,
version and experiment manifests, test summary, empty failure log, state,
canvas, telemetry, frontend assets, server logs, and SHA-256 identities.

### T-108 S0 reference-contract decision

Status: `Ready`
Track: Validation  
Depends on: T-104, T-105, T-106, T-107  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S0 is marked locked only if baseline tests pass or failures are documented and judged non-blocking.
Notes: All T-101 through T-107 dependencies are complete. The evidence and
recommended decision text are consolidated in
`docs/S0_REFERENCE_CONTRACT_ACCEPTANCE_CANDIDATE_2026-08-11.md`. Jackson's
explicit acceptance remains required; T-109's missing native sensor adapter is
recorded as a nonconformance, not hidden or silently treated as gate closure.

### T-109 Migrate native execution to `plant-interface-v1`

Status: `Blocked`
Track: Control/Inference Boundary
Depends on: T-103, AI-201, AI-203, AI-204
Owner: Jackson/Codex
Estimate: 3-5 days
Acceptance: Native execution and motor forecasts consume physical sensor packets and posterior snapshots without copied live simulator or process-RNG state.
Notes: The live default now fails closed before copied simulator state can
reach the model. On 2026-08-05 the MuJoCo runtime began initializing forecast
q/qvel from `BodyBeliefSnapshot` with independent future plant noise. T-109
remains blocked because native execution lacks a `PlantBackend` sensor adapter.
The default/oracle forecast container still copies substrate grain and model
state rather than building all initial latents from beliefs. The opt-in
`provisional-sensor-simulation-v0` MuJoCo path now avoids that live copy with
independent fixed context priors, but it does not supply native adaptation or
calibrated posteriors for those latents. The four independent material
fields are overwritten from `SpatialCanvasState` when supplied; compact brush
load/pigment is overwritten from `BrushLoadBelief`, and bristle-scale variation
uses independent prior noise. Held paint and persistent bristle history remain
collapsed rather than inferred.

## S1: MuJoCo Physical Draft And Logical Retarget

### T-201 Define the versioned physical MuJoCo draft and logical command subset

Status: `Done`
Track: MuJoCo  
Depends on: T-101  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: XML defines the four-joint logical command order, explicit axes/signs/ranges, separated physical anchors, link geometry, canvas frame, brush contact reference, and safe keyframes, while intentional differences from the native logical body are named.
Notes: Accepted 2026-07-28 in `models/active_inference_painter.xml` and `models/README.md` as `mujoco-robstride-electromechanical-v4`; the native command subset is preserved through a named physical retarget instead of hidden controller offsets.

### T-202 Add XML geometry, range, actuator, and contact tests

Status: `Done`
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Tests protect joint order/axes/ranges, separated anchors, keyframes, canvas dimensions, direct-drive actuator limits, brush geometry/compliance, friction, and stable adapter names.
Notes: Accepted 2026-07-28 in `tests/test_mujoco_model.py`; the focused XML/model suite passes with optional MuJoCo compile coverage.

### T-203 Validate physical kinematics and logical retarget transforms

Status: `Done`
Track: MuJoCo  
Depends on: T-201, T-202  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Representative poses verify joint signs, offset kinematics, brush-tip transforms, canvas reach, and the named logical-canvas-to-physical-target retarget within declared tolerances.
Notes: Accepted 2026-07-28 through `test_mujoco_joint_signs_match_the_declared_offset_kinematics`, reach/keyframe tests, and `test_legacy_canvas_points_retarget_to_the_physical_robot`; the physical shoulder is intentionally not co-located like the native abstraction.

### T-204 Separate visual, collision, and contact geometry

Status: `Done`
Track: MuJoCo  
Depends on: T-201  
Owner: Jackson/Codex
Estimate: 0.5 day  
Acceptance: Decorative housings remain non-colliding, collision-relevant links are explicit, and canvas/brush contact remains isolated and testable.
Notes: Accepted 2026-07-28 in the MJCF collision masks and `models/README.md`; compile/contact tests exercise the intended collision pair.

### T-205 Document exact versus approximate model fields

Status: `Done`
Track: Documentation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Model documentation distinguishes simulator-truth fields from visual placeholders.
Notes: Accepted 2026-07-28 in `models/README.md`, including provisional datasheet-derived RobStride values, lumped brush compliance, Python-owned paint, the then-native counterfactual approximation, and first calibration measurements. Updated 2026-08-04 for matched MuJoCo counterfactuals and their remaining exact-state-oracle limitation.

### T-206 Add MuJoCo load/compile smoke test

Status: `Done`
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex
Estimate: 0.5-1 day  
Acceptance: Optional MuJoCo package loads the XML when installed and skips cleanly when unavailable.
Notes: Accepted 2026-07-28 in `tests/test_mujoco_model.py`; the model compiles with stable joint/site/sensor/actuator names and the test module skips explicitly when MuJoCo is unavailable.

### T-207 Define model version label

Status: `Done`
Track: Operations  
Depends on: T-002, T-201  
Owner: Jackson/Codex
Estimate: 0.5 day  
Acceptance: The model has a runtime-visible version label that distinguishes the physical draft from native abstract and future calibrated hardware revisions.
Notes: Accepted 2026-07-28 as `mujoco-robstride-electromechanical-v4` in MJCF text metadata, backend identity, runtime state, and model documentation.

### T-208 Compare model behavior in MuJoCo viewer

Status: `Active`
Track: Manual Validation  
Depends on: T-201, T-206  
Owner: Jackson  
Estimate: 0.5-1 day  
Acceptance: Manual viewer load confirms expected joint sliders, tip/canvas alignment, and triaged discrepancies.
Notes: The XML-driven Three.js frontend has been inspected through home, canvas-facing, top, contact, lower-arm-down, and canvas-edge views. A dated standalone MuJoCo viewer inspection record and any discrepancy entries remain before acceptance.

### T-209 S1 lock decision

Status: `Blocked`
Track: Validation  
Depends on: T-203, T-204, T-205, T-206, T-208  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S1 is locked only after XML tests, load/compile status, and viewer issues are accepted or triaged.
Notes: Blocked only on T-208 manual acceptance and Jackson's explicit physical-draft/retarget contract decision.

## S2: MuJoCo Backend Adapter

### T-301 Define common backend surface

Status: `Done`
Track: Architecture  
Depends on: T-103, T-201, T-203
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Python sim and MuJoCo sim expose a shared controller-facing interface.
Notes: Accepted 2026-07-28 through `plant-interface-v1`, `MujocoPlantBackend`, and `MujocoJointPlant`; backend-specific calls remain below `ArmPainterSim`, stroke execution, telemetry, and web runtime.

### T-302 Map `ArmPose` targets to MuJoCo controls

Status: `Done`
Track: Control  
Depends on: T-301  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: Degree-based controller targets are converted correctly for MuJoCo runtime controls.
Notes: Accepted 2026-07-28 in `MujocoJointPlant.step`; degree/radian conversion, joint order, target clipping, encoder state, and telemetry targets have focused tests.

### T-303 Read MuJoCo state into existing pose/contact structures

Status: `Done`
Track: MuJoCo  
Depends on: T-301, T-302  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Joint, tip, contact, and telemetry values can be consumed by current runtime code.
Notes: Accepted 2026-07-28 with direct qpos/encoder state, physical tip/site state, exact brush-canvas contact, force/pressure/compression/bend state, and runtime payload coverage.

### T-304 Reuse `VerticalCanvas` for MuJoCo-driven paint

Status: `Done`
Track: Painting Model  
Depends on: T-303  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: MuJoCo brush contact deposits paint through the existing material model.
Notes: Accepted 2026-07-28: brush material is loaded before motion and deposition is driven by physical contact/pressure through `VerticalCanvas`; unloading changes material availability without changing arm motion. Python remains the declared paint-material boundary.

### T-305 Add scripted-stroke smoke tests

Status: `Done`
Track: Validation  
Depends on: T-302, T-303, T-304  
Owner: Jackson/Codex
Estimate: 1 day  
Acceptance: A scripted MuJoCo stroke reaches the canvas and updates material coverage.
Notes: Accepted 2026-07-28 in `tests/test_mujoco_backend.py`; contact, force, pressure, coverage, deterministic snapshot/restore, and a 370-sample continuous-contact deposition stroke are covered.

### T-306 Add backend selection to web runtime

Status: `Done`
Track: Web/Runtime  
Depends on: T-301, T-304  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Web runtime can choose native or MuJoCo backend while keeping native as the default.
Notes: Accepted 2026-07-28 via `python -m active_painter.web_server --plant-backend {native,mujoco}`; state, canvas PNG, telemetry CSV, and XML-driven frontend geometry share the runtime.

### T-307 Adapt telemetry for MuJoCo backend

Status: `Done`
Track: Telemetry  
Depends on: T-303, T-306  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Telemetry remains schema-compatible where possible and explicitly marks unavailable MuJoCo fields.
Notes: Accepted 2026-07-28 with backend/model identity, joint/actuator/encoder state, current, torque, voltage, elastic/backlash/friction/load terms, contact, brush-loaded, and actual-deposition fields. Declared approximations remain documented.

### T-308 Define MuJoCo forecast strategy

Status: `Done`
Track: Planning/Forecasting  
Depends on: T-301, T-305  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Initial MuJoCo live-execution versus forecast-rollout scope is decided and documented.
Notes: Initially accepted 2026-07-28 with a deferred native forecast approximation. Revised 2026-08-04: MuJoCo execution now produces independent same-model MuJoCo/contact counterfactuals with explicit backend and approximation provenance. Revised 2026-08-05: MuJoCo q/qvel initialization is sensor-conditioned through a frozen `BodyBeliefSnapshot` and independent seed; material fields initialize from `SpatialCanvasState`; compact brush load/pigment initializes from `BrushLoadBelief` and bristle-scale variation uses an independent versioned prior. Substrate grain/model context, held-paint and persistent-bristle history, contact-to-compliance initialization, and MuJoCo body-parameter particles remain deferred.

### T-309 Add backend parity checks

Status: `Active`
Track: Validation  
Depends on: T-305, T-306  
Owner: Jackson/Codex
Estimate: 1-2 days  
Acceptance: Same scripted stroke can run on native and MuJoCo backends with path/contact/coverage differences recorded.
Notes: Canonical transform, logical retarget, state-shape, contact/deposition, runtime selection, same-model counterfactual copying, rollout independence, and live-state non-mutation tests pass. A versioned matched-stroke artifact comparing path, timing, pressure, current, and material coverage remains.

### T-310 S2 lock decision

Status: `Blocked`
Track: Validation  
Depends on: T-305, T-306, T-307, T-308, T-309  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S2 is locked only after MuJoCo execution, paint update, backend selection, and known gaps are documented.
Notes: Blocked on the T-309 matched parity artifact and Jackson's explicit S2 lock decision. The native counterfactual substitution was removed 2026-08-04; exact-state oracle initialization remains documented M2 work.

## M4: Experimental Observatory And Digital Twin

### T-401 Define the experimental-state contract

Status: `Blocked`  
Track: Architecture/Research Ops  
Depends on: T-003, T-103, T-105  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Truth, observation, belief, prediction, policy, control, telemetry, canvas, capability, and provenance fields have versioned units, frames, shapes, and access roles.

### T-402 Define clocks, revisions, and provenance

Status: `Blocked`  
Track: Research Ops/Telemetry  
Depends on: T-002, T-401  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Runtime outputs distinguish simulation, sensor, inference, policy, control, canvas, telemetry, and wall clocks and refer to exact revisions.

### T-403 Implement the native runtime adapter

Status: `Blocked`  
Track: Runtime/Web  
Depends on: T-401, T-402  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Native state populates the versioned contract without browser-side physical or inferential corrections.

### T-404 Reuse the Three.js arm and canvas scene

Status: `Blocked`  
Track: Web  
Depends on: T-403  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: One stable scene renders backend-provided geometry and contact while preserving existing navigation and native appearance.

### T-405 Separate material truth, observation, and prediction views

Status: `Blocked`  
Track: Web/Observation Boundary  
Depends on: T-102, T-403  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Process truth, agent-accessible observation, and predicted canvas statistics are separately labeled and revision-synchronized.

### T-406 Handle capabilities, stale state, and runtime errors

Status: `Blocked`  
Track: Web/Runtime  
Depends on: T-402, T-403, T-404  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Unsupported, stale, invalid, faulted, paused, planning, retracting, and disconnected states cannot masquerade as valid execution.

### T-407 Add belief, prediction, and policy overlays

Status: `Blocked`  
Track: Web/Active-Inference Diagnostics  
Depends on: AI-101, T-403, T-405  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Prior/posterior, predicted/realized, precision, VFE, EFE, and approximation diagnostics are distinct and optional.

### T-408 Implement synchronized experiment artifact capture

Status: `Blocked`  
Track: Research Ops  
Depends on: T-003, T-402, T-405, T-407  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Named runtime events capture synchronized manifests, state, telemetry, truth, observation, prediction, and screenshot artifacts without entering policy inference.

### T-409 Add contract and synchronization tests

Status: `Blocked`  
Track: Validation  
Depends on: T-403, T-405, T-406, T-407, T-408  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Tests cover schema, unit, frame, revision, null, stale, non-finite, hidden-truth, and VFE/EFE separation invariants.

### T-410 Capture native/MuJoCo parity artifacts

Status: `Blocked`  
Track: Validation/Simulation  
Depends on: T-309, T-404, T-405, T-408  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Matched pose, motion, contact, and stroke cases produce synchronized artifacts and triaged numeric backend differences.

### T-411 Validate responsiveness and visual legibility

Status: `Blocked`  
Track: Web/Manual Validation  
Depends on: T-404, T-405, T-406, T-407  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Desktop/mobile views remain legible and observatory overhead is measured and does not materially change painting behavior.

### T-412 M4 observatory gate

Status: `Blocked`  
Track: Validation  
Depends on: T-409, T-410, T-411  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: Shared contracts, provenance, clocks, access boundaries, synchronized artifacts, parity evidence, and bounded overhead are accepted.

## M5: Versioned Geometry, Actuation, And Calibration

### T-501 Define `RobotGeometrySpec`

Status: `Active`
Track: Geometry/Architecture  
Depends on: T-101, T-103  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: A versioned SI-unit schema covers frames, joints, limits, transforms, mass properties, collision geometry, brush, canvas, and camera references without controller corrections.
Notes: Work began in the versioned MJCF and XML-derived web model with separated shoulder anchors, direct-drive joints, canvas frame, brush dimensions, and SI transforms. Acceptance remains open until these fields move into an authoritative backend-neutral `RobotGeometrySpec` with mass/collision/camera coverage.

### T-502 Define the frame graph and naming convention

Status: `Blocked`  
Track: Geometry  
Depends on: T-501  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Design and calibration frames share declared names, handedness, transform direction, timestamps, and canonical pose fixtures.

### T-503 Define actuator, transmission, and sensor specifications

Status: `Blocked`  
Track: Actuation/Control Interface  
Depends on: T-103, T-501  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Motor, driver, transmission, encoder, current, temperature, contact, force, and exposed command-mode fields are versioned per joint.
Notes: The hardware-oriented draft fixes RobStride 03 at yaw/pitch and
RobStride 02 at upper-arm roll/elbow. That actuator assignment is a selected
design configuration, not an inferred motor-realization latent. The
`mujoco-robstride-electromechanical-v4` backend is vendor-grounded but still
contains derived/approximate parameters and is not calibrated against assembled
hardware; transmissions, full sensing fields, and command contracts therefore
remain insufficient for acceptance. The separate `native-abstract-v0` plant is
not identified from these selected motors.

### T-504 Add parameter provenance and uncertainty

Status: `Blocked`  
Track: Calibration/Research Ops  
Depends on: T-501, T-502, T-503  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Every physical parameter distinguishes placeholder, datasheet, CAD, measured, fitted, and validated provenance plus meaningful uncertainty.

### T-505 Define the preliminary design and operating envelope

Status: `Blocked`  
Track: Requirements/Mechanical  
Depends on: T-501, T-503  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Canvas, workspace, tip speed, contact, payload, duty, joint, retraction, service, paint, and cleaning requirements are separated from provisional desires.

### T-506 Run workspace, singularity, and collision studies

Status: `Blocked`  
Track: Kinematics/Mechanical  
Depends on: T-502, T-505  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Painting, retract, loading, cleaning, limit, collision, and poor-conditioning regions are mapped and exported as validation cases.

### T-507 Compare direct drive and belt-reduction architectures

Status: `Blocked`  
Track: Actuation/Mechanical Decision  
Depends on: T-503, T-505, T-506  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Joint-local direct, belt, and mixed options are compared across dynamics, control, packaging, sensing, cost, service, and reversible risk.

### T-508 Build mass, inertia, torque, stiffness, and thermal budgets

Status: `Blocked`  
Track: Dynamics/Mechanical  
Depends on: T-503, T-505, T-507  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Gravity, acceleration, contact, fault, stiffness, bandwidth, duty, thermal, and distal-mass sensitivities have traceable estimates and margins.

### T-509 Define brush, canvas, camera, paint, and cleaning interfaces

Status: `Blocked`  
Track: End Effector/Workcell  
Depends on: T-502, T-505  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Repeatable brush, registered canvas, fixed camera, secured paint, and supervised cleaning interfaces have geometry and operating envelopes.
Notes: The fixed-camera portion now has a selected-but-not-purchased compact
baseline: exactly two Raspberry Pi Global Shutter Cameras (Sony IMX296) on a
rigid crossbar under `provisional-compact-dual-imx296-v1`. Brush anisotropy and
an exploratory angled-wrist design are simulation studies, not accepted
mechanical interfaces. Brush replacement/datum, registered physical canvas,
paint security, cleaning, and operating envelopes remain open, so this task is
still blocked.

### T-510 Define calibration and system-identification procedures

Status: `Blocked`  
Track: Calibration  
Depends on: T-502, T-504, T-509  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Geometry, encoder, camera, current/torque, friction, compliance, brush, canvas, and contact procedures declare fixtures, fit/validation splits, residuals, and thresholds.

### T-511 Implement parameter export and validation adapters

Status: `Blocked`  
Track: Tooling/Architecture  
Depends on: T-501, T-502, T-503, T-504  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Native, MJCF, CAD, controller, and manifest inputs derive from one validated specification with explicit backend approximations.

### T-512 Run sensitivity and identifiability analysis

Status: `Blocked`  
Track: Validation/Modeling  
Depends on: T-506, T-508, T-510, T-511  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Physical parameters are ranked by consequence, unidentifiable combinations are named, and measurement effort follows research sensitivity.

### T-513 Select a preliminary geometry and actuation baseline

Status: `Blocked`  
Track: Mechanical Decision  
Depends on: T-507, T-508, T-509, T-512  
Owner: Jackson  
Estimate: 1 day  
Acceptance: A reversible versioned choice covers links, joints, actuator class, sensing, brush, and workcell with alternatives, cost, margins, and risks.

### T-514 M5 calibration-readiness gate

Status: `Blocked`  
Track: Validation  
Depends on: T-511, T-513  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: Schemas, provenance, uncertainty, sizing, calibration, exports, and bounded M6 fabrication questions are accepted.

## M6: CAD And Prototype Iteration

### T-601 Define CAD traceability and revision policy

Status: `Blocked`  
Track: CAD/Research Ops  
Depends on: T-501, T-504, T-505  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: CAD parameters, exports, assemblies, prototypes, and as-built revisions map traceably to versioned M5 specifications.

### T-602 Build the parametric CAD skeleton and frame map

Status: `Blocked`  
Track: CAD/Geometry  
Depends on: T-502, T-505, T-601  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: A parameter-driven skeleton represents joint, brush, canvas, camera, paint, and cleaning frames before detailed housings.

### T-603 Analyze structural load paths and stiffness

Status: `Blocked`  
Track: Mechanical Analysis  
Depends on: T-508, T-602  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Gravity, acceleration, contact, belt, bearing, fault, base, easel, and brush-tip deflection risks are estimated and prioritized.

### T-604 Design modular joint and transmission concepts

Status: `Blocked`  
Track: Mechanical/Actuation  
Depends on: T-507, T-603  
Owner: Jackson/Codex  
Estimate: 4-7 days  
Acceptance: Testable direct-drive, belt, or mixed joint modules define bearings, shafts, pulleys, tensioning, encoders, stops, compliance, and service access.

### T-605 Integrate actuators, encoders, and drivers mechanically

Status: `Blocked`  
Track: Mechatronics  
Depends on: T-503, T-604  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Mounting, thermal, connector, strain-relief, encoder, sensing, and maintenance interfaces are mechanically credible and fed back to M5.

### T-606 Design cable routing, stops, covers, and service access

Status: `Blocked`  
Track: Mechanical/Safety  
Depends on: T-604, T-605  
Owner: Jackson/Codex  
Estimate: 2-4 days  
Acceptance: Cable life, joint clearance, hard stops, pinch/snag exposure, contamination protection, and service access are addressed.

### T-607 Design the brush mount and contact-compliance module

Status: `Blocked`  
Track: End Effector  
Depends on: T-509, T-602, T-603  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Brush datum, replacement, loading, cleaning, compliance/sensing, bending error, and bench-contact testing support wet oil painting.

### T-608 Design the base, easel, camera, paint, and cleaning fixtures

Status: `Blocked`  
Track: Workcell  
Depends on: T-509, T-602, T-603  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Rigid adjustable fixtures support calibration, retraction, manual access, secured vessels, spill containment, and the intended work envelope.

### T-609 Define tolerances, manufacturing processes, and inspection

Status: `Blocked`  
Track: Manufacturing  
Depends on: T-604, T-606, T-607, T-608  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Sensitivity-driven tolerances, processes, datums, inspection, adjustment, and expensive-redesign risks are explicit.

### T-610 Maintain BOM, fabrication, and iteration risk estimates

Status: `Blocked`  
Track: Cost/Risk  
Depends on: T-604, T-605, T-608, T-609  
Owner: Jackson  
Estimate: 2 days  
Acceptance: The approximately USD 4,000 prototype target includes controls, structure, fabrication, wiring, safety, fixtures, shipping, spares, and iteration contingency.

### T-611 Build and characterize a single-joint prototype

Status: `Blocked`  
Track: Prototype/Actuation  
Depends on: T-604, T-605, T-606, T-609  
Owner: Jackson  
Estimate: 5-8 days plus fabrication lead time  
Acceptance: The riskiest joint topology is measured for current, tracking, backlash, compliance, temperature, hold, stop, and model mismatch before replication.

### T-612 Build and characterize a linkage/brush-contact rig

Status: `Blocked`  
Track: Prototype/Contact  
Depends on: T-607, T-608, T-609  
Owner: Jackson  
Estimate: 4-7 days plus fabrication lead time  
Acceptance: Approach, contact, release, retract, loading, cleaning, compliance, tip repeatability, and wet-paint consequences are measured under supervision.

### T-613 Feed prototype measurements back into all models

Status: `Blocked`  
Track: Calibration/Digital Thread  
Depends on: T-511, T-611, T-612  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Measured parameters create new geometry, MuJoCo, controller, and CAD revisions while preserving pre-update models and residual mismatch.

### T-614 Conduct the integrated prototype design review

Status: `Blocked`  
Track: Mechanical Review  
Depends on: T-610, T-613  
Owner: Jackson  
Estimate: 2 days  
Acceptance: Requirements, mechanics, wiring, calibration, safety interfaces, cost, evidence, and residual risks support a bounded release or named rework.

### T-615 M6 prototype-release gate

Status: `Blocked`  
Track: Validation  
Depends on: T-614  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: CAD and M5 identify one body, risky assumptions have prototype evidence, manufacture is bounded, and M7 receives a testable release.

## M7: Safety And Staged Hardware Bring-Up

### T-701 Perform hazard analysis and define operating modes

Status: `Blocked`  
Track: Safety  
Depends on: T-503, T-505  
Owner: Jackson  
Estimate: 2-3 days  
Acceptance: Hazards, modes, command authority, supervision, mitigation, and residual risk cover startup through cleaning and faults.

### T-702 Design the independent safety architecture

Status: `Blocked`  
Track: Safety/Electrical  
Depends on: T-701  
Owner: Jackson/Codex  
Estimate: 2-4 days  
Acceptance: Power isolation, enable, E-stop, driver, crash, communication, sensor, and power-failure behavior remain independent of painting policy inference.

### T-703 Specify emergency stop, recovery, and restart

Status: `Blocked`  
Track: Safety/Operations  
Depends on: T-702  
Owner: Jackson  
Estimate: 1-2 days  
Acceptance: Stop classes, power state, canvas-contact recovery, reset inspection, and policy invalidation are explicit.

### T-704 Define hard motion, electrical, thermal, and contact limits

Status: `Blocked`  
Track: Safety/Control  
Depends on: T-508, T-701, T-702  
Owner: Jackson/Codex  
Estimate: 3-4 days  
Acceptance: Conservative joint, speed, acceleration, current, voltage, temperature, force, contact, and workspace limits are externally enforced.

### T-705 Implement watchdog and state-validity rules

Status: `Blocked`  
Track: Safety/Software  
Depends on: T-702, T-704  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Stale command/sensor, non-finite, revision, timing, connection, and mode errors stop motion and produce complete fault records.

### T-706 Define hardware calibration and encoder-zero procedure

Status: `Blocked`  
Track: Calibration/Operations  
Depends on: T-510, T-703, T-704  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Restricted-power direction, limit, zero, current, unit, assembly, and brush checks produce a compatible calibration manifest.

### T-707 Commission and validate one powered joint

Status: `Blocked`  
Track: Hardware Bring-Up  
Depends on: T-611, T-702, T-704, T-705, T-706  
Owner: Jackson  
Estimate: 3-5 days  
Acceptance: Enable, stop, sensing, limits, watchdog, tracking, hold, damping, friction, backlash, thermal behavior, and model residuals are validated.

### T-708 Commission and validate a two-link chain

Status: `Blocked`  
Track: Hardware Bring-Up/Dynamics  
Depends on: T-707, T-613  
Owner: Jackson  
Estimate: 4-7 days  
Acceptance: Coupled dynamics, gravity, tracking, damping, retraction, stops, current, structural modes, and cable effects pass conservative tests.

### T-709 Commission the integrated arm in dry free space

Status: `Blocked`  
Track: Hardware Bring-Up  
Depends on: T-614, T-708  
Owner: Jackson  
Estimate: 5-8 days  
Acceptance: Full-arm joints, cables, stops, workspace, retract, E-stop, canonical paths, startup, pause, jog, shutdown, and provenance are validated dry.

### T-710 Validate brush approach, contact, release, and retract

Status: `Blocked`  
Track: Contact/Safety  
Depends on: T-612, T-709  
Owner: Jackson  
Estimate: 4-7 days  
Acceptance: Restricted-energy contact cases validate detection, pressure proxy, release, retraction, emergency recovery, compliance, and no planning-state contact.

### T-711 Define paint and solvent operating procedures

Status: `Blocked`  
Track: Workcell/Safety  
Depends on: T-608, T-701  
Owner: Jackson  
Estimate: 2-3 days  
Acceptance: Supervised loading, cleaning, secured vessels, spill, ventilation, storage, waste, shutdown, and later automation boundaries are documented.

### T-712 Validate physical sensors and fixed-camera observations

Status: `Blocked`  
Track: Sensors/Calibration  
Depends on: AI-201, T-509, T-706, T-709  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Physical sensor modalities have measured units, rates, noise, delay, dropout, camera calibration, fixtures, and sensor-access enforcement.

### T-713 Implement the hardware backend and provenance path

Status: `Blocked`  
Track: Runtime/Hardware  
Depends on: T-103, T-401, T-705, T-709, T-712  
Owner: Jackson/Codex  
Estimate: 5-8 days  
Acceptance: Versioned hardware commands, observations, telemetry, capabilities, timestamps, calibration, safety vetoes, and recorded replay satisfy the common contract.

### T-714 Run fault-injection and recovery tests

Status: `Blocked`  
Track: Safety Validation  
Depends on: T-703, T-705, T-709, T-713  
Owner: Jackson  
Estimate: 3-5 days  
Acceptance: Communication, sensor, command, non-finite, encoder, electrical, thermal, contact, and backend faults produce the intended stop and recovery.

### T-715 Conduct supervised wet painting trials

Status: `Blocked`  
Track: Hardware Validation/Painting  
Depends on: T-710, T-711, T-712, T-714  
Owner: Jackson  
Estimate: 4-7 days  
Acceptance: Scripted black, white, overlap, curve, broad, edge, and cleaning cases validate physical consequence and contamination behavior under supervision.

### T-716 Run sensor-equivalent inference in shadow mode

Status: `Blocked`  
Track: Active-Inference Transfer Validation  
Depends on: M2, T-713, T-715  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Physical observations update inference without command authority or hidden truth, exposing calibrated prediction, VFE, uncertainty, policy, and transfer mismatch.

### T-717 Conduct bounded autonomous research runs

Status: `Blocked`  
Track: Hardware Research/Safety  
Depends on: AI-315, T-714, T-716  
Owner: Jackson  
Estimate: 3-5 days initially  
Acceptance: Bounded supervised runs preserve external vetoes and complete observation-to-execution artifacts and stop on unsupported state.

### T-718 M7 hardware-readiness gate

Status: `Blocked`  
Track: Validation  
Depends on: T-714, T-715, T-716, T-717  
Owner: Jackson  
Estimate: 1 day  
Acceptance: Safety, dry, contact, wet, fault, sensor, backend, shadow, autonomy, provenance, and residual-risk evidence supports only the documented operating mode.

## M8: Research Experiment Program

### T-801 Define the study registry and evidence levels

Status: `Blocked`  
Track: Research Design  
Depends on: AI-115  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Hypotheses, mechanisms, comparisons, outcomes, exclusions, readiness gates, evidence levels, and charter links are registered before confirmatory interpretation.

### T-802 Finalize experiment manifests and inheritance rules

Status: `Blocked`  
Track: Research Ops  
Depends on: T-003, AI-112, T-801  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Complete model, data, learning, proposal, precision, backend, body, sensor, compute, history, and safety provenance determines confirmatory validity.

### T-803 Build the common measurement and analysis pipeline

Status: `Blocked`  
Track: Research Tooling  
Depends on: T-408, T-802  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: Versioned artifacts produce tested predictive, calibration, policy, gaze, embodiment, hierarchy, spatial-temporal, and failure summaries at valid independent-unit levels.

### T-804 Define replication, stopping, and analysis plans

Status: `Blocked`  
Track: Experimental Design  
Depends on: T-801, T-803  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Pilot variance, paired replication, compute ceilings, stopping, failures, multiple comparisons, intervals, and descriptive limits are declared before confirmatory runs.

### T-805 Maintain the capability-readiness matrix

Status: `Blocked`  
Track: Validation/Research Ops  
Depends on: T-801, T-802  
Owner: Jackson/Codex  
Estimate: 1 day initially, then ongoing  
Acceptance: Each study is blocked or released by its exact observation, prediction, embodiment, relation, persistence, policy, and generalization evidence.

### T-806 Publish the formal baseline and approximation report

Status: `Blocked`  
Track: Baseline Research  
Depends on: AI-114, AI-115, T-802, T-803  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: The factorization, access boundary, fixtures, calibration, proposals, preferences, inheritance, runtime, approximations, replicas, and failures define the baseline.

### T-807 Test H1: active foveation

Status: `Blocked`  
Track: Research/Active Vision  
Depends on: AI-304, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs  
Acceptance: Active, random, and uniform gaze are matched and compared for ambiguity, calibration, prediction, fixation pathology, and long-range spatial effects.

### T-808 Test H2: embodied motor prediction

Status: `Blocked`  
Track: Research/Embodiment  
Depends on: AI-308, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs  
Acceptance: Cartesian and body-aware inference are compared under matched canvas alternatives for motor outcomes, policies, geometry, residuals, perturbations, and inactivity pathology.

### T-809 Test H3: slow latent temporal persistence

Status: `Blocked`  
Track: Research/Hierarchy  
Depends on: AI-310, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus runs  
Acceptance: Fast, slow, frozen, shuffled, and reset conditions test predictive information and later policy effects beyond the current canvas.

### T-810 Test H4: relational nonsemantic representation

Status: `Blocked`  
Track: Research/Relational Inference  
Depends on: AI-209, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs  
Acceptance: Local, diagnostic, and uncertain relational models face spatial interventions and held-out separated-event prediction without aesthetic targets.

### T-811 Test H5: epistemic action and pathology

Status: `Blocked`  
Track: Research/Epistemic Policy  
Depends on: AI-309, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs  
Acceptance: Reducible ambiguity, noise, misspecification, reachability, repetition, uncertainty variants, useful information gain, and epistemic pathologies are compared.

### T-812 Test H6: inherited pretrained perception

Status: `Blocked`  
Track: Research/Pretraining  
Depends on: AI-314, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs  
Acceptance: At most two frozen encoders face the same probabilistic adapter, data, policy, and metrics for gain, calibration, cost, sample efficiency, and imported invariance.

### T-813 Test online development and inheritance

Status: `Blocked`  
Track: Research/Continual Learning  
Depends on: AI-312, T-802, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus long runs  
Acceptance: Separate agent histories compare initialization, freezing, online learning, and replay for prediction, drift, forgetting, policy change, and early-output dependence.

### T-814 Run selected interaction studies

Status: `Blocked`  
Track: Research/Interactions  
Depends on: T-807, T-808, T-809, T-810, T-811  
Owner: Jackson/Codex  
Estimate: 6-10 days plus runs  
Acceptance: Only interactions motivated by single-mechanism evidence enter a predeclared affordable comparison matrix.

### T-815 Run simulator and hardware transfer studies

Status: `Blocked`  
Track: Research/Transfer  
Depends on: M2, T-410, T-716, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus hardware time  
Acceptance: Native, MuJoCo, recorded, and live hardware compare matched observations, predictions, calibration, beliefs, VFE/EFE, policies, consequences, and localized mismatch.

### T-816 Define qualitative and conceptual-artifact protocol

Status: `Blocked`  
Track: Research Communication/Art  
Depends on: T-803  
Owner: Jackson  
Estimate: 2-3 days  
Acceptance: Complete, interrupted, failed, and representative canvases have synchronized traces, predeclared selection, no optimization feedback, and honest interpretive limits.

### T-817 M8 synthesis and research-decision gate

Status: `Blocked`  
Track: Research Synthesis  
Depends on: T-806, T-807, T-808, T-809, T-810, T-811, T-812, T-813, T-814, T-815, T-816  
Owner: Jackson  
Estimate: 5-8 days  
Acceptance: Hypothesis evidence, failures, imports, engineering contributions, negative results, ambiguity, and the next feasible high-information direction are reported separately.
