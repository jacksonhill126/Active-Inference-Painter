# M2: Calibrated Multiscale Generative Model

## Summary

M2 replaces the current oracle-like spatial observation path with a compact,
explicitly probabilistic, sensor-equivalent generative model. It also turns the
pixel, canvas, relational, and passage levels into a tested multiscale
hierarchy rather than a collection of loosely coupled predictors.

The central M2 question is:

> Can the agent infer and predict materially and spatially relevant hidden
> states from observations a physical platform could plausibly receive?

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
- The global canvas latent only sees a 16 x 16 coarse field.
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
material state x_t  ---- action a_t ----> x_(t+1)
        |                                  |
        v                                  v
canvas observation o_canvas(t)      o_canvas(t+1)

body state b_t ------ motor policy ------> b_(t+1)
        |                                  |
        v                                  v
proprio/current/contact observations

coarse material c_k <---- deterministic coarse-graining of x_t
        |
        +----> uncertain canvas latent z_canvas(k)
        +----> uncertain relational latent z_relation(k)
                        |
                        +---- passage-scale transition
```

Deterministic coarse-graining is permitted. Independently guessed coarse
material state is not. The hierarchy may predict distributions over latent
causes and future observations, but touched coarse fields must remain grounded
in the predicted material state.

## Scope

### Included

- Fixed-camera or idealized sensor-equivalent observation process.
- Explicit visual/material observation likelihood.
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

Status: `Blocked`  
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

- Define `p(o_canvas | x_material, z_canvas)` or a documented feature-space
  likelihood with explicit support and variance.
- Define proprioceptive, current, and contact likelihoods separately.
- Predict likelihood variance from permitted conditioning variables or latent
  state, not from inaccessible simulator truth at inference time.
- Avoid counting deterministic transforms as independent evidence.
- Test likelihood calibration on held-out observations.
- Report what information about white paint, texture, and overlap is lost by
  the chosen sensor.

### AI-204 Build the compact state-inference path

Status: `Blocked`  
Track: Variational Inference  
Depends on: AI-203  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Define `q(x_material, z_canvas, z_relation | observations, prior)` with an
  explicit factorization.
- Use a small amortized encoder, iterative update, or hybrid whose
  approximation is documented.
- Fuse transition priors and observations without reading ground-truth
  material state.
- Preserve uncertainty in unobserved or visually ambiguous material.
- Compare posterior estimates against hidden simulator labels only for
  evaluation.
- Verify VFE terms against M1 fixtures and report inference latency.

### AI-205 Align local dynamics training with live execution

Status: `Blocked`  
Track: Transition Model/Data  
Depends on: AI-108, AI-202  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Remove bootstrap/live pixel-scale mismatch.
- Make brush footprint, grain, and patch margin physically equivalent across
  corpus and live execution.
- Train local dynamics using the M1 leakage-resistant split.
- Preserve exact identity outside touched support.
- Report calibration by action and material condition.
- Confirm no simulator outcome labels enter live policy inference.

### AI-206 Validate multi-step material prediction

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
- Report error growth, interval coverage, and failure by overlap condition.
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
- Provide a sequence diagram for one complete passage.

### AI-208 Make the global canvas latent predictively necessary

Status: `Blocked`  
Track: Hierarchical Modeling  
Depends on: AI-206, AI-207  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Compare flat, local-only, current 16 x 16 latent, and one modest richer
  multiscale alternative.
- Evaluate held-out reconstruction and future prediction, not only training
  ELBO.
- Test spatial scrambling, translation, and rotation sensitivity.
- Measure latent effective rank, posterior collapse, and context usage.
- Demonstrate that conditioning on the canvas latent improves predictions of
  separated or later events.
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
Depends on: AI-109, AI-204, AI-208, AI-213  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Produce data-versus-capacity learning curves for the observation and
  hierarchy models.
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
