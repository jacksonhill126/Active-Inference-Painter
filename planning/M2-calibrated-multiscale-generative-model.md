# M2: Calibrated Multiscale Generative Model

## Summary

M2 replaces the current oracle-like spatial observation path with a compact,
explicitly probabilistic, sensor-grounded generative model whose hardware
equivalence still requires physical calibration. It also turns the
pixel, canvas, relational, and passage levels into a tested multiscale
hierarchy rather than a collection of loosely coupled predictors.

The central M2 question is:

> Can the agent infer and predict visually and spatially relevant hidden
> structure—and uncertain action-conditioned mark consequences—from
> observations a physical platform could plausibly receive?

The owner-confirmed perceptual boundary is
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`. Detailed material state remains in
the process. The target persistent hierarchy is visual; it does not maintain a
canvas-wide wetness map merely because the process contains one.

M2 builds on the M1 factorization and held-out corpus. It does not add active
gaze yet; the first observation process may expose a fixed full-canvas camera
plus proprioceptive, current, and contact sensors. M3 will restrict that visual
stream through a foveated observation process.

## Scientific Intent

M2 tests the first four capability gates in the research charter:

1. relevant changes are represented;
2. local mark consequences are predicted with calibrated uncertainty;
3. embodied consequences are predicted;
4. separated regions participate in a shared uncertain relational belief.

The milestone must support negative conclusions. If a compact model cannot
predict or represent relevant structure under the available data and compute
budget, the project should know whether the limitation is observation,
capacity, training, uncertainty, or factorization.

## Current Problems Addressed

- Exact simulator material channels are used as observations.
- Observation variance is partly computed from material values that should be
  hidden.
- Material coverage, surface tone, and contrast are deterministic consequences
  but can appear as independent Gaussian channels.
- Bootstrap data uses a different pixel scale from live data.
- Local dynamics are tested more thoroughly than their multi-step rollout
  behavior.
- The global canvas latent only sees a 16 x 16 coarse material field, which
  erases useful angle and continuous-boundary structure.
- The accepted corpus does not retain registered pre/post camera images, so it
  cannot train the selected visual mark-consequence model.
- Relational slots are produced by deterministic clustering with no assignment
  uncertainty.
- Slow beliefs and transition models share a small, highly correlated online
  stream.
- Shared learned parameters and one agent's episodic beliefs are not cleanly
  separated.
- The structural preference can become self-confirming.

## Target Generative Structure

M2 should implement or justify a structure of the following form:

```text
shared parameters theta
        |
process material x_t ---- action a_t ----> process material x_(t+1)
        |                                        |
        v                                        v
registered image o_t                      registered image o_(t+1)
        |                 action, brush          ^
        +----------> ephemeral z_interaction ----+
        |                                        |
        +------> z_visual --> z_mass --> z_relation
                                      |
                                      +---- passage/painting transition

body state b_t ------ motor policy ------> b_(t+1)
        |                                  |
        v                                  v
proprio/current/contact observations
```

The process may expose material arrays to evaluator-only diagnostics, but the
agent-facing hierarchy is grounded in registered observations. A local
interaction latent is inferred freshly for an action/patch and then
marginalized; it is not a persistent inverse-physics canvas. Tone, oriented
boundaries, continuous masses, and relations remain only if they improve
held-out prediction, calibration, or temporal/intervention tests.

## Scope

### Included

- Fixed-camera observation process with declared provisional assumptions and
  a physical-calibration gate.
- Explicit visual observation and action-conditioned visual-transition
  likelihoods.
- Compact amortized or iterative state inference.
- Live-scale local transition learning.
- Multi-step predictive and uncertainty calibration.
- Hierarchical canvas and relational latent redesign.
- Slow-state update schedule and temporal-persistence tests.
- Shared-parameter versus episodic-belief separation.
- Continual-learning and replay evaluation.
- Lower-level parallel pretraining with one canonical online agent.
- Composition-preference implementation from the M1 decision.
- Capacity and pretrained-interface readiness tests.

### Deferred

- Active gaze and foveal policy inference.
- Photorealistic camera simulation.
- Foundation-model fine-tuning.
- Full covariance over all pixels.
- A physically complete oil-paint inverse model.
- Learned painting intentions or demonstrated policies.

## Tasks

### AI-201 Define the sensor-equivalent M2 observation package

Status: `Active`
Track: Sensors/Active Inference  
Depends on: M1  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Choose the initial physical sensor analogues: fixed canvas camera, joint
  encoders, motor current, and contact/force signals where available.
- Define image resolution, channels, sample rates, units, and noise.
- State whether depth or structured-light information is assumed.
- Keep simulator material state available only to the generative process,
  supervised corpus generation where declared, and evaluation.
- Define how material coverage remains a hidden material state inferred from
  sensory consequences rather than a directly observed scalar.
- Retain registered, rectified pre/post camera images, timing, calibration,
  masks, action-aligned crops, brush context, motor realization, and terminal
  provenance in the corpus contract.
- Update the sensor-access ledger from M1.

### AI-202 Implement the fixed-view observation generative process

Status: `Blocked`  
Track: Generative Process/Sensors  
Depends on: AI-201, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Generate the same observation types expected from the future physical
  platform.
- Map material state to visible surface observations without exposing hidden
  thickness, wetness, or pigment arrays to inference.
- Keep observation noise separate from hidden-state process noise.
- Never seed agent forecasts by copying the generative process RNG or future
  sensor-noise continuation.
- Version camera geometry, ground tone, and sensor parameters.
- Add deterministic fixtures for blank, black, white, overlap, and wet-
  blending cases.
- Retain an oracle material mode only as a labeled diagnostic baseline.

### AI-203 Specify and implement the observation likelihood

Status: `Blocked`  
Track: Generative Model  
Depends on: AI-201, AI-202  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Define an image or non-overlapping multiscale-coefficient likelihood with
  explicit support and variance.
- Define the action-conditioned visual transition
  `p(o_(t+1,R) | o_(t,R), action, brush, z_interaction)` and its recognition
  density without exposing process material arrays.
- Define proprioceptive, current, and contact likelihoods separately.
- Predict likelihood variance from permitted conditioning variables or latent
  state, not from inaccessible simulator truth at inference time.
- Avoid counting deterministic transforms as independent evidence.
- Preserve tone and oriented-boundary fidelity; do not count an edge transform
  as independent evidence unless the joint-density approximation is explicit.
- Test likelihood calibration on held-out observations.
- Report what information about white paint, texture, and overlap is lost by
  the chosen sensor.

### AI-204 Build the compact state-inference path

Status: `Active`
Track: Variational Inference  
Depends on: AI-203  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Define `q(z_visual, z_mass, z_relation | observations, prior)` with an
  explicit factorization, plus an ephemeral local
  `q(z_interaction | image crop, action, brush)` where predictive evidence
  justifies it.
- Use a small amortized encoder, iterative update, or hybrid whose
  approximation is documented.
- Fuse transition priors and observations without reading ground-truth
  material state.
- Provide the inferred visual and body initial-state distributions used by
  counterfactual rollouts; do not initialize them from a live process object.
- Preserve visually unresolved material causes as calibrated uncertainty over
  visual outcomes rather than persistent named canvas variables.
- Compare posterior estimates against hidden simulator labels only for
  evaluation.
- Verify VFE terms against M1 fixtures and report inference latency.

Progress, 2026-08-05: `BodyStateEstimator` is connected to the MuJoCo sensor
packet and motor forecasts initialize q/qvel from a frozen posterior revision
with independent future-noise seeds. Forecast thickness, wetness, black mass,
and surface tone now initialize from a separately frozen spatial posterior,
with mean particle zero and diagonal material particles. Compact brush load and
average pigment now initialize from a frozen `BrushLoadBelief`; later particles
also use independent versioned microstructure noise rather than continuing the
live brush RNG. The full material/relational factorization,
substrate-grain/model rollout initialization, held-paint and persistent-bristle
history and contact-to-brush compliance mapping remain open. The
`action-conditioned-camera-update-v0` schedule now propagates the learned
material transition prior, records a post-physics capture boundary, rejects
older frames, and automatically polls the MuJoCo camera until the posterior is
updated. The opt-in `provisional-sensor-simulation-v0` profile now repeats this
cycle without copying the live process into policy forecasts. It uses an
independent fixed-prior MuJoCo/material template and a deliberately bounded
one-step/one-particle/Cartesian-IK profile. This makes AI-204 runnable for
sensor-posterior corpus collection but does not satisfy its calibration or
factorization acceptance criteria; see the canonical status in
`PROJECT_TRACKER.md`.

Direction correction, 2026-08-12: this progress remains evidence that the
sensor boundary, frozen posterior initialization, and independent
counterfactual state work. It is not a commitment to complete a persistent
canvas-wide inverse material posterior. The accepted continuation replaces
that target with visual/body inference plus an optional ephemeral local
interaction latent, as specified in
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

### AI-205 Align local visual-dynamics training with live execution

Status: `Blocked`  
Track: Transition Model/Data  
Depends on: AI-108, AI-202  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Remove bootstrap/live pixel-scale mismatch.
- Make camera registration, brush footprint, action support, and patch margin
  equivalent across corpus and live execution.
- Train local dynamics using the M1 leakage-resistant split.
- Preserve exact identity outside touched support.
- Report visual likelihood and boundary fidelity by action, brush context,
  overlap, tone, and image condition.
- Confirm no simulator outcome labels enter live policy inference.

### AI-206 Validate multi-step visual prediction

Status: `Blocked`  
Track: Transition Model/Validation  
Depends on: AI-204, AI-205  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Evaluate one-, two-, four-, and passage-length rollouts on held-out executed
  trajectories.
- Sample or propagate posterior uncertainty rather than initializing every
  member only at the posterior mean.
- Compare mean-evaluated, particle, and affordable Monte Carlo variants.
- Report visual error growth, normalized likelihood, interval/quantile
  calibration, boundary fidelity, and failure by overlap condition.
- Test that sparse full-canvas scaling does not reward or penalize patch area.
- Choose the least expensive rollout approximation that meets declared error
  and calibration thresholds.

### AI-207 Define the multiscale latent clocks and messages

Status: `Blocked`  
Track: Hierarchical Active Inference  
Depends on: AI-101, AI-204  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Define pixel/mark, tile, passage, and painting clocks.
- Specify priors, likelihood messages, posterior updates, and precision at
  every level.
- Ensure higher levels have slower dynamics and greater temporal depth.
- Prevent slow states from copying lower observations through fast moving
  averages.
- Define what state persists across marks, passages, paintings, and model
  checkpoints.
- Explicitly prohibit persistent canvas-wide wetness in the target hierarchy;
  local interaction latents expire after the action-conditioned prediction.
- Provide a sequence diagram for one complete passage.

### AI-208 Make the global canvas latent predictively necessary

Status: `Blocked`  
Track: Hierarchical Modeling  
Depends on: AI-206, AI-207  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Compare flat, local-only, current 16 x 16 material latent, and one modest
  tone/edge/mass-preserving multiscale visual alternative.
- Evaluate held-out reconstruction and future prediction, not only training
  ELBO.
- Test spatial scrambling, translation, and rotation sensitivity.
- Measure latent effective rank, posterior collapse, and context usage.
- Demonstrate that conditioning on the canvas latent improves predictions of
  separated or later events.
- Verify that oblique and curved boundaries survive the compact
  representation rather than becoming square-cell artifacts.
- Increase capacity only if learning curves show underfitting.

### AI-209 Replace or demote deterministic relational beliefs

Status: `Blocked`  
Track: Relational Inference  
Depends on: AI-207, AI-208  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Decide whether deterministic region slots are observations, engineered
  sufficient statistics, or approximations to latent assignments.
- Implement an affordable uncertain alternative, such as soft assignments,
  sampled segmentation perturbations, or a small learned set posterior.
- Represent uncertainty about region existence, assignment, and relation.
- Test split, merge, occlusion, overlap, and spatial-scramble cases.
- Show whether relational state improves held-out passage prediction.
- If a probabilistic alternative is infeasible, relabel deterministic slots
  and limit claims rather than calling them posterior beliefs.

### AI-210 Validate slow-state temporal persistence

Status: `Blocked`  
Track: Hierarchical Inference/Validation  
Depends on: AI-207, AI-208, AI-209  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Measure how canvas and relational posterior information persists across
  marks and passage boundaries.
- Compare fast-only, slow-transition, shuffled-history, and frozen-slow-state
  baselines.
- Test whether slow latents predict held-out later marks after controlling for
  the current coarse canvas.
- Verify update count and clock behavior directly.
- Detect posterior collapse, stale beliefs, and latest-observation copying.

### AI-211 Implement the M1 structural-preference decision

Status: `Blocked`  
Track: Preferences  
Depends on: AI-110, AI-208, AI-210  
Owner: Jackson/Codex  
Estimate: 2-4 days

Acceptance:

- Implement only the composition formulation approved in M1.
- Keep the evaluator frozen or cross-fitted if required by that decision.
- Log structural model evidence separately from prior preference contribution.
- Preserve a zero-preference ablation.
- Test for blank, noise, memorization, and early-history self-reinforcement.
- Do not tune precision by visual appeal.

### AI-212 Separate shared pretraining from individual development

Status: `Blocked`  
Track: Continual Learning  
Depends on: AI-112, AI-205, AI-210  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Permit pooled or parallel experience for shared low-level likelihood and
  transition parameters.
- Keep each environment's posterior, canvas, passage history, and slow latent
  separate.
- Define the point at which a canonical agent begins its individual online
  history.
- Evaluate forgetting as online updates shift the data distribution.
- Retain a frozen inherited model and online-adapted model for comparison.
- Record inherited data sources in every run manifest.

### AI-213 Build the affordable parallel data and training path

Status: `Blocked`  
Track: Performance/Data  
Depends on: AI-113, AI-205, AI-212  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Generate independent lower-level transitions in parallel using native or S1
  headless simulators.
- Batch frozen observations, local patches, and hierarchy updates efficiently.
- Keep test data and canonical online development outside pooled training.
- Measure throughput before considering Isaac Lab or large GPU simulation.
- Store enough environment context that randomized mechanics are not mistaken
  for irreducible aleatoric noise.
- Document conventional data-generation and SGD work separately from active
  inference at run time.

### AI-214 Evaluate model size and pretrained readiness

Status: `Blocked`  
Track: Feasibility/Modeling  
Depends on: AI-204, AI-205, AI-206, AI-208, AI-213
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Produce data-versus-capacity learning curves for the observation and
  hierarchy models.
- Complete the registered visual local/hierarchy learning-curve continuation
  moved out of M1 AI-109 by the 2026-08-26 gate repair.
- Record trainable parameters, memory, inference latency, and training cost.
- Define a narrow feature-adapter interface that could accept a frozen
  pretrained encoder later.
- Do not integrate a foundation model unless the from-scratch baseline has an
  identified representational limitation.
- Reserve the same probabilistic likelihood and state-space model above every
  candidate encoder.

### AI-215 Run the M2 capability-gate suite

Status: `Blocked`  
Track: Validation  
Depends on: AI-203, AI-204, AI-206, AI-208, AI-209, AI-210, AI-211, AI-212  
Owner: Jackson/Codex  
Estimate: 3 days

Acceptance:

- Observation tests show relevant black, white, overlap, texture, and bodily
  changes are represented.
- Local and multi-step transition tests meet M1 calibration thresholds.
- Embodied outcome distributions distinguish representative motor
  realizations.
- Relational tests respond to separated regions and declared
  transformations.
- Slow-state tests demonstrate temporal persistence beyond current-frame
  summaries.
- Save matched oracle-observation and sensor-equivalent baseline runs.

### AI-216 M2 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-215  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- The live agent no longer depends on privileged material observations for
  research behavior.
- Observation and transition likelihoods have held-out calibration evidence.
- Multiscale and relational levels have explicit probabilistic status.
- Slow latents demonstrate predictive temporal use.
- Structural preference status is explicit.
- M3 receives a stable belief interface suitable for partial foveated
  observations.

## Capability Gate

M2 is complete only when the agent can infer and predict relevant hidden state
from plausible fixed-view observations. A visually interesting canvas does not
waive predictive or calibration failures.

## Feasibility

- Estimated effort: 29-45 focused workdays.
- Solo calendar estimate: 7-11 weeks.
- Target training expenditure: USD 100-250.
- Recommended trainable scale: approximately 1-8 million parameters unless
  learning curves justify more.
- Foundation-model training from scratch is excluded.

The uncertain relational posterior and sensor-equivalent observation model are
the largest research risks. A credible smaller alternative is to keep
deterministic relational statistics explicitly labeled while validating a
probabilistic canvas latent and local dynamics.

## Failure Modes

- Reconstructing hidden simulator material through supervised labels and then
  claiming sensor-only inference without a deployment path.
- Increasing latent dimension until reconstruction improves while future
  prediction and calibration do not.
- Treating a deterministic encoder feature as a likelihood.
- Pooling slow beliefs across parallel environments.
- Letting domain randomization inflate unexplained variance because
  environment context is absent.
- Enabling structural preference before the evaluator is validated.
- Spending most of the compute budget on encoder experiments before the small
  baseline is measured.

## Outputs

- Sensor-equivalent observation specification and renderer.
- Explicit observation likelihood and compact inference model.
- Live-scale local transition model and multi-step calibration report.
- Multiscale clock/message specification.
- Validated canvas and relational latent implementation.
- Slow-state temporal-persistence report.
- Structural-preference implementation or documented disablement.
- Parallel lower-level pretraining pipeline.
- Model-size and pretrained-readiness report.
- M2 capability bundle and lock note.
