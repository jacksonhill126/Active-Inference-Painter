# Project Tracker

This tracker uses milestone-scale tasks, roughly 1-5 day chunks. It is a
planning artifact, not a replacement for the research charter.

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

Status: `Ready`  
Track: Validation  
Depends on: T-003, T-004  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Each project gate lists required tests, logs, and stop conditions.

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

Status: `Blocked`  
Track: Active Inference/Formalization  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Every implemented belief, likelihood, transition, preference, precision, policy prior, proposal, VFE term, and EFE term maps to a declared factor or named approximation.

### AI-102 Create the variable and sensor-access ledger

Status: `Blocked`  
Track: Active Inference/Sensors  
Depends on: AI-101  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Every simulator-to-agent value is classified as physical observation, derived observation, hidden state, evaluation label, or execution-only state.

### AI-103 Audit observation independence and units

Status: `Blocked`  
Track: Active Inference/Observation Model  
Depends on: AI-101, AI-102  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Derived material channels, likelihood independence, units, normalization, and provisional observation terms are explicitly decided.

### AI-104 Verify VFE against tractable reference models

Status: `Blocked`  
Track: Active Inference/Validation  
Depends on: AI-101, AI-103  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Analytic linear-Gaussian fixtures independently verify posterior moments and VFE decomposition across precision regimes.

### AI-105 Verify EFE against enumerated or Monte Carlo references

Status: `Blocked`  
Track: Active Inference/Validation  
Depends on: AI-101, AI-104  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Independent fixtures verify risk, ambiguity, information gain, policy priors, signs, and posterior normalization.

### AI-106 Stress-test terminal coverage and stopping

Status: `Blocked`  
Track: Preferences/Validation  
Depends on: AI-105  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Beta moment matching, clamps, alternatives, terminal-only application, and immediate-stop support have a documented decision.

### AI-108 Build a leakage-resistant baseline corpus

Status: `Blocked`  
Track: Data/Validation  
Depends on: AI-102, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Live-scale train/validation/test transitions are split by trajectory before patch extraction and stratified by material, action, reach, and motor conditions.

### AI-107 Calibrate transition and precision approximations

Status: `Blocked`  
Track: Uncertainty  
Depends on: AI-103, AI-108  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Held-out NLL, interval coverage, z-scores, and OOD disagreement distinguish aleatoric, ensemble, likelihood, and fixed-precision terms.

### AI-109 Establish predictive learning curves

Status: `Blocked`  
Track: Modeling/Validation  
Depends on: AI-107, AI-108  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Data/capacity curves determine whether current local and hierarchy models are underfitting, data-limited, or overfitting.

### AI-110 Resolve the composition-preference closed loop

Status: `Blocked`  
Track: Active Inference/Preferences  
Depends on: AI-101, AI-105, AI-109  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Compression gap is formally approved with frozen/cross-fitted safeguards or disabled as a preference.

### AI-111 Separate proposal distributions from policy priors

Status: `Blocked`  
Track: Policy Inference  
Depends on: AI-101, AI-105  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Candidate frequency, intended policy prior, proposal density, and finite-budget bias have explicit semantics and convergence tests.

### AI-112 Define online learning and inheritance semantics

Status: `Blocked`  
Track: Continual Learning/Research Ops  
Depends on: AI-101, AI-108  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: Shared parameters, optimizer, replay, calibration, episodic beliefs, and canvas history have separate persistence and reporting rules.

### AI-113 Profile inference, rollout, and learning separately

Status: `Blocked`  
Track: Performance/Feasibility  
Depends on: AI-109  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: State inference, EFE, motor forecasting, hierarchy, training, serialization, and rendering have separate timings and a ranked optimization plan.

### AI-114 Capture reproducible baseline replicas

Status: `Blocked`  
Track: Research Ops  
Depends on: AI-104, AI-105, AI-106, AI-107, AI-109, AI-110, AI-111, AI-113  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: At least three fixed-seed runs archive manifests, beliefs, VFE/EFE, calibration, proposal statistics, telemetry, failures, and canvases.

### AI-115 M1 lock decision

Status: `Blocked`  
Track: Validation  
Depends on: AI-114  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: Formal, sensor-access, VFE/EFE, calibration, terminal, composition, proposal, and inheritance decisions are accepted before M2.

## M2: Calibrated Multiscale Generative Model

### AI-201 Define the sensor-equivalent M2 observation package

Status: `Blocked`  
Track: Sensors/Active Inference  
Depends on: M1  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Fixed camera, encoders, current, and contact observations have declared channels, units, rates, noise, and physical analogues.

### AI-202 Implement the fixed-view observation generative process

Status: `Blocked`  
Track: Generative Process/Sensors  
Depends on: AI-201, T-101, T-102  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: The process emits sensor-equivalent observations while hidden material arrays remain evaluation-only.

### AI-203 Specify and implement the observation likelihood

Status: `Blocked`  
Track: Generative Model  
Depends on: AI-201, AI-202  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Canvas, proprioceptive, current, and contact likelihoods are explicit, calibrated, and avoid double-counting deterministic transforms.

### AI-204 Build the compact state-inference path

Status: `Blocked`  
Track: Variational Inference  
Depends on: AI-203  
Owner: Jackson/Codex  
Estimate: 4-6 days  
Acceptance: An explicit compact posterior fuses transition priors with permitted observations and reports calibrated hidden-state uncertainty.

### AI-205 Align local dynamics training with live execution

Status: `Blocked`  
Track: Transition Model/Data  
Depends on: AI-108, AI-202  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Corpus and live execution share physical pixel scale, support geometry, and held-out calibration.

### AI-206 Validate multi-step material prediction

Status: `Blocked`  
Track: Transition Model/Validation  
Depends on: AI-204, AI-205  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: One-mark through passage-length rollouts have measured error growth and uncertainty coverage under an approved approximation.

### AI-207 Define the multiscale latent clocks and messages

Status: `Blocked`  
Track: Hierarchical Active Inference  
Depends on: AI-101, AI-204  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: Pixel, mark, tile, passage, and painting levels have explicit priors, likelihood messages, posterior updates, precision, and persistence.

### AI-208 Make the global canvas latent predictively necessary

Status: `Blocked`  
Track: Hierarchical Modeling  
Depends on: AI-206, AI-207  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: A modest hierarchy improves held-out future prediction and responds correctly to spatial interventions without collapse.

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

### AI-214 Evaluate model size and pretrained readiness

Status: `Blocked`  
Track: Feasibility/Modeling  
Depends on: AI-109, AI-204, AI-208, AI-213  
Owner: Jackson/Codex  
Estimate: 2-3 days  
Acceptance: Learning curves, latency, memory, cost, and a narrow frozen-feature adapter define whether larger or pretrained perception is justified.

### AI-215 Run the M2 capability-gate suite

Status: `Blocked`  
Track: Validation  
Depends on: AI-203, AI-204, AI-206, AI-208, AI-209, AI-210, AI-211, AI-212  
Owner: Jackson/Codex  
Estimate: 3 days  
Acceptance: Observation, local prediction, embodiment, relational representation, and temporal-persistence gates pass under sensor-equivalent input.

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
Acceptance: Gaze state, sensor geometry, fovea, periphery, rates, latency, uncertainty, and accessible samples are explicitly defined.

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

### AI-304 Define gaze policies and their EFE

Status: `Blocked`  
Track: Active Inference/Policy  
Depends on: AI-105, AI-303  
Owner: Jackson/Codex  
Estimate: 3-5 days  
Acceptance: Gaze risk, ambiguity, and information gain are derived, separately logged, and resolve controlled reducible ambiguity without saliency rewards.

### AI-305 Factor gaze, mark, passage, and motor policies

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

Status: `Backlog`  
Track: Simulation  
Depends on: M0  
Owner: TBD  
Estimate: 1 day  
Acceptance: Native simulator constants, limits, home pose, canvas frame, and known shortcuts are documented.

### T-102 Version and protect canvas material invariants

Status: `Backlog`  
Track: Painting Model  
Depends on: M0  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Tests confirm thickness, wetness, pigment mass, visible tone, white-on-white coverage, and material coverage behavior.

### T-103 Define controller, plant, and policy interfaces

Status: `Backlog`  
Track: Control  
Depends on: T-101  
Owner: TBD  
Estimate: 1 day  
Acceptance: `StrokeAction` remains Cartesian/contact intent and IK remains below policy selection.

### T-104 Record full baseline test result

Status: `Backlog`  
Track: Validation  
Depends on: T-101, T-102, T-103  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: Current planner, arm, canvas, and web tests have a recorded baseline result.

### T-105 Capture baseline telemetry and web-runtime behavior

Status: `Backlog`  
Track: Web/Telemetry  
Depends on: T-101, T-103  
Owner: TBD  
Estimate: 1 day  
Acceptance: Default web runtime endpoints, frontend state, canvas image, and telemetry CSV have a recorded baseline.

### T-106 Document known simulator shortcuts and limitations

Status: `Backlog`  
Track: Documentation  
Depends on: T-101, T-102, T-103  
Owner: TBD  
Estimate: 1 day  
Acceptance: Simulator shortcuts are categorized as acceptable baseline, MuJoCo calibration need, or hardware validation need.

### T-107 Define baseline artifact bundle

Status: `Backlog`  
Track: Research Ops  
Depends on: T-003, T-104, T-105  
Owner: TBD  
Estimate: 0.5-1 day  
Acceptance: Baseline bundle contents and location are defined, including test summary, config, telemetry, canvas image, and notes.

### T-108 S0 reference-contract decision

Status: `Backlog`  
Track: Validation  
Depends on: T-104, T-105, T-106, T-107  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S0 is marked locked only if baseline tests pass or failures are documented and judged non-blocking.

## S1: MuJoCo Abstract Clone

### T-201 Match native arm constants in MuJoCo XML

Status: `Active`  
Track: MuJoCo  
Depends on: T-101  
Owner: Jackson/Codex  
Estimate: 1-2 days  
Acceptance: XML matches native joint order, axes, ranges, home pose, link lengths, and canvas frame.

### T-202 Add XML constant tests

Status: `Validate`  
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Tests compare XML constants against `arm_sim.py`.

### T-203 Validate MuJoCo forward kinematics against native kinematics

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-201, T-202  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Representative MuJoCo tip/site poses match native forward kinematics within tolerance.

### T-204 Keep physical housings visual-only

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-201  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: Base and joint geometry do not accidentally constrain the abstract clone.

### T-205 Document exact versus approximate model fields

Status: `Validate`  
Track: Documentation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Model documentation distinguishes simulator-truth fields from visual placeholders.

### T-206 Add MuJoCo load/compile smoke test

Status: `Backlog`  
Track: Validation  
Depends on: T-201  
Owner: TBD  
Estimate: 0.5-1 day  
Acceptance: Optional MuJoCo package loads the XML when installed and skips cleanly when unavailable.

### T-207 Define model version label

Status: `Backlog`  
Track: Operations  
Depends on: T-002, T-201  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: Abstract model version is named and distinguished from future calibrated hardware model versions.

### T-208 Compare model behavior in MuJoCo viewer

Status: `Backlog`  
Track: Manual Validation  
Depends on: T-201, T-206  
Owner: Jackson  
Estimate: 0.5-1 day  
Acceptance: Manual viewer load confirms expected joint sliders, tip/canvas alignment, and triaged discrepancies.

### T-209 S1 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-203, T-204, T-205, T-206, T-208  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S1 is locked only after XML tests, load/compile status, and viewer issues are accepted or triaged.

## S2: MuJoCo Backend Adapter

### T-301 Define common backend surface

Status: `Backlog`  
Track: Architecture  
Depends on: S0, S1  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Python sim and MuJoCo sim expose a shared controller-facing interface.

### T-302 Map `ArmPose` targets to MuJoCo controls

Status: `Backlog`  
Track: Control  
Depends on: T-301  
Owner: TBD  
Estimate: 1 day  
Acceptance: Degree-based controller targets are converted correctly for MuJoCo runtime controls.

### T-303 Read MuJoCo state into existing pose/contact structures

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-301, T-302  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Joint, tip, contact, and telemetry values can be consumed by current runtime code.

### T-304 Reuse `VerticalCanvas` for MuJoCo-driven paint

Status: `Backlog`  
Track: Painting Model  
Depends on: T-303  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: MuJoCo brush contact deposits paint through the existing material model.

### T-305 Add scripted-stroke smoke tests

Status: `Backlog`  
Track: Validation  
Depends on: T-302, T-303, T-304  
Owner: TBD  
Estimate: 1 day  
Acceptance: A scripted MuJoCo stroke reaches the canvas and updates material coverage.

### T-306 Add backend selection to web runtime

Status: `Backlog`  
Track: Web/Runtime  
Depends on: T-301, T-304  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Web runtime can choose native or MuJoCo backend while keeping native as the default.

### T-307 Adapt telemetry for MuJoCo backend

Status: `Backlog`  
Track: Telemetry  
Depends on: T-303, T-306  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Telemetry remains schema-compatible where possible and explicitly marks unavailable MuJoCo fields.

### T-308 Define MuJoCo forecast strategy

Status: `Backlog`  
Track: Planning/Forecasting  
Depends on: T-301, T-305  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Initial MuJoCo live-execution versus forecast-rollout scope is decided and documented.

### T-309 Add backend parity checks

Status: `Backlog`  
Track: Validation  
Depends on: T-305, T-306  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Same scripted stroke can run on native and MuJoCo backends with path/contact/coverage differences recorded.

### T-310 S2 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-305, T-306, T-307, T-308, T-309  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: S2 is locked only after MuJoCo execution, paint update, backend selection, and known gaps are documented.

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

Status: `Blocked`  
Track: Geometry/Architecture  
Depends on: T-101, T-103  
Owner: Jackson/Codex  
Estimate: 2 days  
Acceptance: A versioned SI-unit schema covers frames, joints, limits, transforms, mass properties, collision geometry, brush, canvas, and camera references without controller corrections.

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
