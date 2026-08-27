# M1: Formal Baseline And Inference Audit

## Summary

M1 establishes whether the current painter is a defensible active-inference
baseline before increasing model size, adding foveation, or interpreting its
paintings as evidence about composition.

The present implementation contains real active-inference structure: explicit
state posteriors, transition and observation densities, VFE state updates, EFE
policy evaluation, policy priors, terminal preferences, and parameter-
uncertainty approximations. It also contains unresolved approximations that can
change behavior substantially. M1 turns those approximations into a formal
model specification, tractable reference calculations, held-out predictive
benchmarks, and explicit decisions.

M1 does not aim to make the paintings more attractive. It determines what the
current system actually infers, what it merely computes, and which subsequent
failures can be interpreted scientifically.

M1 locks the scientific interpretation, evaluation protocol, and declared
approximations of the baseline. It does not freeze model architecture,
controller implementation, simulator fidelity, or mechanical design. Those
may evolve concurrently behind versioned interfaces, but a changed component
must produce a new manifest rather than silently redefining the baseline.

## Scientific Intent

M1 addresses a prerequisite to every hypothesis in the research charter:

> Before asking whether spatial organization emerges, establish that beliefs,
> uncertainty, preferences, and policy posteriors have the meanings assigned
> to them.

A negative compositional result is not informative if:

- the observation is privileged simulator state;
- the likelihood is mis-scaled or double-counts derived channels;
- the transition ensemble is uncalibrated;
- an EFE sign or entropy convention changes with units;
- candidate frequency silently acts as a policy prior;
- a terminal approximation dominates policy selection;
- online learning overfits the agent's first few paintings.

## Current Problems Addressed

- There is no single factorization that maps every implementation term to a
  random variable and conditional density.
- The spatial estimator receives material fields that a physical robot could
  not observe directly.
- Six spatial channels include derived fields that may be treated as if they
  were independent observations.
- Pixel posteriors are diagonal and transition moments are evaluated at the
  previous mean.
- Observation ambiguity is usually evaluated at predicted means rather than
  integrated over predicted states.
- Ensemble disagreement is an approximate parameter posterior whose
  calibration is only partially tested.
- Terminal coverage is moment-matched to a clamped Beta family.
- The composition hierarchy both learns from and supplies preferences over the
  agent's own outputs.
- The finite proposal generator mixes proposal frequency and declared policy
  prior.
- Bootstrap and live spatial resolutions differ.
- Checkpoints mix shared learned parameters, replay, optimizer state, and
  episodic beliefs without a research-level inheritance policy.

## Scope

### Included

- Formal generative-model and variational-family specification.
- Sensor-access and privileged-state audit.
- Exact or enumerated VFE/EFE reference fixtures.
- Terminal-preference and information-gain approximation stress tests.
- Held-out transition corpus at live spatial scale.
- Calibration and learning-curve baselines.
- Proposal-support and posterior-convergence tests.
- Composition-preference validity decision.
- Online-learning, replay, and checkpoint inheritance protocol.
- Runtime profiling and parallelism measurements.
- Reproducible baseline artifact bundle.

### Deferred

- Replacing the observation model with camera input; this is M2.
- Foveated observation and gaze policies; this is M3.
- Increasing hierarchy size before learning curves establish underfitting.
- MuJoCo and hardware calibration; these are support milestones.
- Declaring a final theory of intrinsic motivation or composition.

## Formal Baseline Target

The model specification must distinguish at least:

- material state `x_material(t)`;
- bodily state `x_body(t)`;
- canvas latent `z_canvas(k)`;
- relational latent `z_relation(k)`;
- passage latent `z_passage(j)`;
- model parameters `theta`;
- canvas, proprioceptive, current, and contact observations;
- painting, gaze, and motor-realization policies;
- prior preferences over outcomes or latent trajectories;
- policy priors;
- proposal distributions used only for computation.

Each variable must have:

- units and support;
- temporal clock;
- whether it is hidden or observed;
- whether a real robot can access it;
- prior, likelihood, or transition factor;
- variational approximation;
- precision source;
- implementation location;
- known approximation.

## Tasks

### AI-101 Write the executable generative-model specification

Status: `Done`
Track: Active Inference/Formalization  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Add a versioned factorization or factor graph covering pixel/material,
  bodily, canvas, relational, passage, and policy levels.
- State the joint density and variational family in mathematical notation.
- Map every logged VFE/EFE contribution to one factor or identify it as an
  approximation requiring a decision.
- Separate prior beliefs, posterior beliefs, prior preferences, policy priors,
  and computational proposal distributions.
- State temporal clocks and message flow between levels.
- State where conventional control, safety, and SGD sit outside exact active
  inference.

Notes:

- Accepted 2026-07-24 in `docs/GENERATIVE_MODEL_SPEC.md` as
  `baseline-oracle-v0`.
- The specification explicitly identifies privileged material observations,
  uncorrected finite proposal distributions, the self-trained composition
  preference, diagonal posterior approximations, and conventional SGD.

### AI-102 Create the variable and sensor-access ledger

Status: `Done`
Track: Active Inference/Sensors  
Depends on: AI-101  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Inventory every value passed from `ArmPainterSim` and `VerticalCanvas` into
  inference, planning, learning, and diagnostics.
- Classify each value as physical sensor observation, derived observation,
  hidden state, simulator-only evaluation label, or execution-only state.
- Identify all paths where exact material, tip, contact, or plant state is
  available to the agent without a likelihood.
- Define the temporary oracle-observation baseline explicitly.
- Produce blocking tasks for any privileged value that affects a research
  claim rather than visualization or evaluation.

Notes:

- Accepted 2026-07-24 in `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md` and the
  machine-readable `planning/variable-sensor-access-ledger.json`.
- The ledger covers every field in the simulator, canvas, plant, motor
  telemetry, contact, brush, kinematics, and execution-forecast boundary
  dataclasses.
- It identifies exact material state, true pose/contact, exact process
  parameters, and copied plant/brush RNG state as research-critical oracle
  access.
- `tests/test_sensor_access_contract.py` enforces field coverage, blocker
  assignment, and the runtime `baseline-oracle-v0` declaration.

### AI-103 Audit observation independence and units

Status: `Done`
Track: Active Inference/Observation Model  
Depends on: AI-101, AI-102  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Determine which of thickness, wetness, pigment, surface tone, contrast, and
  coverage are latent, observed, or deterministic transforms.
- Test whether derived channels are double-counted as independent likelihood
  evidence.
- Record units and normalization for every likelihood and VFE average.
- Test invariance to benign changes of array shape and reporting units where
  the underlying probability model is unchanged.
- Define which observation terms remain provisional until M2.

Notes:

- Accepted 2026-07-24 in `docs/OBSERVATION_FACTOR_AUDIT.md`.
- Spatial likelihood, transition NLL, VFE, and EFE uncertainty terms now use
  thickness, wetness, black mass, and surface tone as the four independent
  material factors.
- Ground contrast and material coverage remain deterministic state views and
  preference outcomes, but are not counted as separate likelihood evidence.
- Eight focused factorization, normalization, and units tests pass.

### AI-104 Verify VFE against tractable reference models

Status: `Done`
Track: Active Inference/Validation  
Depends on: AI-101, AI-103  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Add a scalar and multivariate linear-Gaussian fixture with an analytic
  posterior and analytic VFE.
- Verify posterior mean, variance, complexity, expected log likelihood, and
  total VFE independently.
- Verify the local identity transition outside stroke support.
- Test posterior behavior as likelihood precision and transition precision
  vary over several orders of magnitude.
- Demonstrate that minimizing reported VFE selects the analytic posterior in
  the tractable fixture.
- Keep pixel-diagonal and mean-evaluated approximations explicitly labeled.

Notes:

- Accepted 2026-08-04 in
  `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`.
- Scalar and multivariate conjugate fixtures independently verify posterior
  moments, all VFE components, precision behavior across four orders of
  magnitude, the analytic minimum, and the local outside-patch identity prior.
- The summary VFE reporting-only budget is now the declared 4096 posterior
  samples. Measured maximum absolute error against fine-grid integration over
  seeds 0-4 is 0.02317 nats inside the accepted `+-0.05` nat band. The spatial
  VFE remains within `1e-6` of independent integration.

### AI-105 Verify EFE against enumerated or Monte Carlo references

Status: `Done`
Track: Active Inference/Validation  
Depends on: AI-101, AI-104  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Add a small discrete or linear-Gaussian policy model whose risk, ambiguity,
  information gain, and policy posterior can be enumerated.
- Verify signs and decomposition identities independently of production
  helper functions.
- Test deterministic, purely ambiguous, purely epistemic, and preference-
  dominated cases.
- Verify policy posterior normalization with and without policy priors.
- Identify any production EFE term that lacks a defensible reference-model
  counterpart and either derive, rename, disable, or explicitly defer it.

Notes:

- Accepted 2026-08-04 in
  `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md` after AI-104 closure.
- The enumerated matrix now isolates deterministic, ambiguity-only, purely
  epistemic, and preference-dominated cases and verifies policy posteriors both
  with and without policy priors.
- Terminal, transition, motor, and policy-posterior signs and identities match
  independent references. The earlier motor-ambiguity double count remains
  removed at every summary and spatial total site.
- Low-coverage terminal approximation behavior is assigned to AI-106. The
  composition-gap preference has no accepted normalized reference counterpart
  and is explicitly deferred to AI-110, satisfying this task without approving
  that self-referential preference.

### AI-106 Stress-test terminal coverage and stopping

Status: `Done`
Track: Preferences/Validation  
Depends on: AI-105  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Compare the moment-matched Beta terminal forecast against direct Monte Carlo
  coverage samples across low, medium, and high variance.
- Quantify effects of concentration clamps near zero and one coverage.
- Compare at least one alternative bounded family or sample-based estimate.
- Verify the terminal preference is applied only at `stop`.
- Verify immediate `stop` remains available and proposal-independent.
- Record the terminal coverage and stop prior as declared preferences, not
  learned discoveries.
- Decide whether the current family is acceptable for M2 or must be replaced.

Notes:

- Closed 2026-08-11 in
  `docs/TERMINAL_COVERAGE_STOPPING_ACCEPTANCE_2026-08-11.md` with 48 focused
  tests passing.
- The exact Beta--Beta KL agrees with direct Monte Carlo when the forecast is
  Beta. A broad logit-normal alternative with the same measured first two
  moments differs by 23.6 nats, and the near-blank interior-unimodal floor
  changes represented variance by more than 500x.
- Decision: retain the fixed terminal coverage preference and finite
  always-admissible stop prior, but replace the single moment-matched Beta
  forecast before M2 acceptance with a converged sample-based estimate or a
  calibrated richer bounded family.

### AI-107 Calibrate transition and precision approximations

Status: `Done`
Track: Uncertainty  
Depends on: AI-103, AI-108  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Measure local transition NLL, interval coverage, z-scores, and ensemble
  disagreement on held-out data.
- Stratify calibration by tone, wet-over-wet state, width, length, amount,
  motor realization, canvas region, reach, and patch size.
- Test whether ensemble disagreement increases under meaningful distribution
  shifts.
- Distinguish learned aleatoric variance, ensemble disagreement, fixed
  likelihood variance, and fixed precision multipliers in reports.
- Stop calling fixed precision settings beliefs unless a posterior is inferred.
- Define recalibration and failure thresholds for M2.

Completion record (2026-08-11): trained CNN and conditional patch-cVAE
ensembles were evaluated on AI-108's frozen validation/test trajectories.
Validation-only scalar variance calibration did not repair interval shape:
both nominal 90% intervals covered about 99.4% of held-out residuals. A CNN
trained without dynamic roll increased ensemble disagreement only 1.087x on
held-out dynamic-roll conditions versus the predeclared 1.50x gate. Requested
strata are machine-reported, with wet-over-wet marked structurally unavailable
and patch size marked single-bin. Learned likelihood, cVAE latent, ensemble,
target posterior, fixed camera likelihood, fixed identity likelihood, and EFE
precision terms are kept separate. All precision-ledger entries were
unobserved declared priors. This closes the measurement task with a negative
M2 calibration result; it does not accept either model as calibrated. See
`docs/AI107_UNCERTAINTY_CALIBRATION_TECHNICAL_2026-08-11.md`.
Both models predicted coarse material posteriors. This result does not test the
owner's proposed action-conditioned visual mark-consequence VAE; see
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

### AI-108 Build a leakage-resistant baseline corpus

Status: `Done`
Track: Data/Validation  
Depends on: AI-102, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Generate training, validation, and test transitions at the live 256-pixel
  scale or a physically equivalent pixel scale.
- Split by complete trajectory or canvas episode before extracting patches so
  overlapping pixels cannot leak across splits.
- Cover dry, wet-over-wet, black, white, overlap, edge, broad, narrow, short,
  long, and representative motor cases.
- Record seeds, simulator version, action distribution, material parameters,
  controller/config version, and resolution.
- Keep parallel simulation data separate from the canonical online painting
  history.
- Reserve a test split that is never used for early stopping or model choice.

### AI-109 Establish predictive learning curves

Status: `Done`
Track: Modeling/Validation  
Depends on: AI-107, AI-108  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Train the current local and hierarchy models on increasing data fractions.
- Report train/validation/test likelihood, calibration, and multi-step error.
- Repeat with at least three seeds where affordable.
- Determine whether current models are underfitting, data-limited, or
  overfitting before approving a capacity increase.
- Record wall time, memory, model size, and optimizer steps.
- Establish a baseline that later pretrained or larger models must beat.

Evidence/decision:

- The local branch completed a 27-run matrix over seeds 109/211/307, nested
  3/6/10-trajectory subsets, three CNN capacities, 1/3/5 ensemble members, the
  shadow cVAE, and a normalized identity-plus-consequence likelihood.
- Data helps modestly, larger generic CNN capacity does not, five ensemble
  members improve density, and the material-posterior cVAE does not materially
  improve the base CNN. This does not test the owner's proposed visual cVAE.
  The identity/consequence mixture improves test NLL by 1.392 nats and
  multistep rollout error, but still fails exact mixture-CDF calibration.
- All AI-108 endpoints are fixed-horizon truncations. A 2026-08-12 pilot then
  demonstrated feasible genuine-stop collection with six selected stops and
  two truncations at 192 steps, but it retained coarse final posteriors rather
  than the registered image stream. Truncations must not be manufactured into
  terminal composition labels.
- Closed for M1 on 2026-08-26. The 27-run matrix is the accepted negative/
  inconclusive capacity decision for the old material models. Terminal visual
  collection and visual hierarchy curves move to AI-205, AI-206, AI-208, and
  AI-214 so M1 no longer depends on the M2 architecture it gates.
- See `docs/AI109_PREDICTIVE_LEARNING_CURVES_TECHNICAL_2026-08-12.md` and
  `docs/AI109_PREDICTIVE_LEARNING_CURVES_OWNER_BRIEF_2026-08-12.md`.

### AI-110 Resolve the composition-preference closed loop

Status: `Done`
Track: Active Inference/Preferences  
Depends on: AI-101, AI-105, AI-109  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Apply `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`: resolve the term against a
  predictive visual tone/edge/mass hierarchy, not the provisional 16x16
  material posterior.

- Write the exact probabilistic interpretation of the compression-gap term.
- Determine whether it is a normalized preference, an unnormalized but
  policy-comparable energy, model evidence, or only a diagnostic.
- Test blank, iid noise, spatially shuffled, repeated, transformed, and held-
  out structured fields with a frozen evaluator.
- Test whether rapid online updates cause early accidental outputs to become
  self-reinforcing preferences.
- Compare same-data, cross-fitted, periodically frozen, and disabled variants.
- Either approve one declared preference formulation with limitations or set
  `composition_gap_precision` to zero by default until M2 resolves it.

Decision 2026-08-26: choose the disabled branch. The default profile is
`m1-formal-policy-baseline-v0`; every legacy material-hierarchy preference,
transition-risk precision, passage term, and derived gap-progress stop factor
is disabled. Controlled opt-in runs identify as
`legacy-material-composition-diagnostic-v0` and cannot support a visual
composition claim. See `docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md`.

### AI-111 Separate proposal distributions from policy priors

Status: `Done`
Track: Policy Inference  
Depends on: AI-101, AI-105  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Identify every place candidate frequency changes effective prior mass.
- Define the proposal distribution `r(pi)` separately from the intended policy
  prior `P(pi)`.
- Decide whether mixture weights are genuine priors or computational sampling
  choices.
- Add candidate-count, horizon, seed, and proposal-mixture convergence tests.
- Measure top-action stability and posterior mass as candidate budgets grow.
- Specify correction or explicit-prior treatment for M3.

Notes:

- Work proceeded ahead of AI-105 acceptance as a bounded implementation
  experiment; it does not change the M1 dependency or gate.
- `proposal.py` now defines an amortized, belief-conditioned proposal density
  over mark and passage latents, trained toward the existing base-EFE painting
  posterior. `PolicySampler` can mix its candidates with the hand-written
  proposal, while immediate stop and passage-plan compounds remain outside its
  learned scope.
- This object is a computational proposal only. It is never added to EFE, VFE,
  a prior preference, or the normalized painting-policy posterior, and it does
  not correct finite-candidate bias. The emission mixture defaults to zero.
- `tests/test_proposal.py` now covers declared support and normalization,
  empirical hand-sampler agreement, exact zero-mixture/RNG parity, mixed-source
  attribution, posterior-only training, checkpoint continuation, and unchanged
  EFE under the default gate.
- `tests/test_proposal_convergence.py` and
  `docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md` add the exact equal-EFE
  multiplicity control and a 360-cell grid over candidate count, horizon, seed,
  and learned mixture. The result is negative for proposal invariance: stop mass
  and deep-horizon winning geometry do not converge under the tested budgets.
- Decision: the current implementation reports only
  `Q(pi | sampled candidate set S)`. Mixtures are computational budget splits,
  not policy priors; learned emission stays zero. M3 must declare the full mixed
  discrete/continuous base measure, `P(pi)`, and `r(pi | belief)`, then test a
  `log P - log r` correction before making proposal-invariant claims.

### AI-112 Define online learning and inheritance semantics

Status: `Done`
Track: Continual Learning/Research Ops  
Depends on: AI-101, AI-108  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Separate shared parameters, optimizer state, replay, calibration state,
  episodic posterior beliefs, and canvas history in checkpoint documentation.
- State which components persist across paintings, code versions, and model
  architecture changes.
- Prohibit training on model-imagined rollouts as if they were observations.
- Distinguish parallel pretraining experience from one agent's developmental
  history.
- Define replay retention, held-out monitoring, and catastrophic-forgetting
  checks.

Evidence: checkpoint schema 7 embeds
`online-learning-inheritance-v0`; separate load modes now cover exact
individual resume, parameter-only individual initialization from shared
pretraining, and shared-pretraining continuation. Episode-local body, canvas,
brush, gap, and passage beliefs reset while learned parameters, individual
replay, and developmental calibration follow their declared ownership rules.
All runtime replay APIs reject model-imagined rollouts as observations. See
`docs/ONLINE_LEARNING_INHERITANCE_2026-08-27.md` and its owner brief. The
AI-114 replicas must still instantiate the defined anchor held-out forgetting
check; that is evidence collection, not an open ownership decision.

### AI-113 Profile inference, rollout, and learning separately

Status: `Ready`
Track: Performance/Feasibility  
Depends on: AI-109  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Measure state inference, base EFE, motor forecasting, hierarchy evaluation,
  gradient training, serialization, and rendering separately.
- Report CPU/GPU utilization and memory for representative runs.
- Identify which computations are batchable without changing equations.
- Distinguish parallel policy rollouts, parallel data generation, batched
  training, and independent research replicas.
- Produce a ranked optimization list with expected benefit and scientific
  risk.

### AI-114 Capture reproducible baseline replicas

Status: `Blocked`  
Track: Research Ops  
Depends on: AI-104, AI-105, AI-106, AI-107, AI-109, AI-110, AI-111, AI-112, AI-113
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Run at least three fixed-seed baseline episodes with frozen configuration.
- Save manifests, checkpoints, EFE/VFE traces, calibration summary, policy
  proposal statistics, telemetry, and canvases.
- Include failures and incomplete paintings.
- Record exact code, model, data, and simulator versions.
- Produce a concise baseline report separating verified behavior from
  provisional claims.

### AI-115 M1 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-114  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- The factorization and sensor-access ledger are accepted.
- Tractable VFE/EFE fixtures pass.
- Held-out predictive and calibration baselines exist.
- Terminal, composition, and proposal approximations each have an explicit
  decision.
- Privileged observation mode is visibly labeled.
- M2 receives a bounded blocker list rather than an open-ended architecture
  rewrite.
- The lock applies to interpretation and evaluation, not to the implementation
  architecture.

## Capability Gate

M1 satisfies the research charter's prerequisite gate when:

- inference terms have declared probabilistic meanings;
- exact fixtures support the reported VFE/EFE decompositions;
- held-out tests establish what the current model can predict;
- uncertainty has calibration evidence;
- policy selection is not silently determined by proposal frequency;
- composition preference is either defensible or disabled;
- baseline failure cannot be blamed on undocumented simulator or training
  shortcuts.

M1 does not claim that the agent has a realistic observation model.

## Feasibility

- Estimated effort: 16-24 focused workdays.
- Solo calendar estimate: 4-6 weeks.
- Expected compute expenditure: under USD 100 using the current small models,
  excluding optional cloud replicas.
- Main cost: test design, corpus hygiene, and formal reconciliation, not model
  training.

This is a larger M1 than the previous baseline lock, but it prevents months of
building on quantities whose probabilistic meaning is unsettled.

## Failure Modes

- Producing more diagnostics without independent reference calculations.
- Using the same helper function in both implementation and expected-value
  tests.
- Letting test-set performance influence architecture choices.
- Treating ensemble variance as Bayesian merely because it is nonzero.
- Keeping a questionable composition term enabled because output looks more
  organized.
- Calling proposal weights priors after tuning them only for visual behavior.
- Treating MuJoCo, controller, or mechanical progress as evidence that
  unresolved inference terms are valid.

## Outputs

- Versioned generative-model specification and variable ledger.
- Sensor-access audit.
- Exact VFE/EFE reference fixtures.
- Live-scale held-out transition corpus manifest.
- Predictive and calibration baseline report.
- Terminal preference decision.
- Composition preference decision.
- Proposal/prior decision.
- Online-learning and checkpoint inheritance protocol.
- Runtime profile and parallelism recommendation.
- M1 baseline artifact bundle and lock note.
