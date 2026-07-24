# M8: Research Experiment Program

## Summary

M8 converts the research charter and M1-M3 capability gates into a staged
experimental program. Experiment infrastructure begins after M1, individual
studies begin when their specific mechanisms pass, and the final crossed
studies wait for the corresponding M3 components. M8 is therefore not a single
ten-week block after all development.

The experiments ask whether active foveation, embodiment, relational belief,
slow latent dynamics, epistemic action, and inherited perception alter
prediction and spatial-temporal organization in interpretable ways. They do
not ask which model makes the most attractive painting.

## Experimental Principles

- Separate exploratory runs, mechanism validation, and confirmatory
  comparisons.
- Declare primary outcomes and exclusions before confirmatory runs.
- Use held-out observations and interventions, not training loss alone.
- Report VFE and EFE decompositions separately.
- Use paired seeds and shared initial conditions where appropriate.
- Preserve failed, interrupted, and visually unappealing runs.
- Report effect sizes, uncertainty intervals, and raw run variation.
- Do not manufacture statistical power with correlated frames or marks.
- Treat each painting/agent history, not each pixel, as the independent unit
  when the claim concerns development or composition.

## Outcome Families

### Predictive and inferential

- Held-out observation and transition log likelihood.
- Multi-step predictive degradation.
- Posterior and interval calibration.
- Policy-posterior stability and proposal convergence.
- Slow-latent predictive information beyond current observation.
- VFE and EFE decomposition sensitivity.

### Embodied and behavioral

- Current, energy, tracking, contact, and execution residuals.
- Gaze allocation and ambiguity reduction.
- Passage duration, continuation, termination, and policy diversity.
- Long-range dependence between separated marks and passages.
- Sensitivity to scramble, rotation, translation, occlusion, and body changes.

### Qualitative and conceptual

- Complete paintings and interrupted runs.
- Synchronized observation, belief, gaze, passage, body, VFE, and EFE traces.
- Failure narratives and architectural interpretation.
- No aggregate aesthetic score or selected-best-output protocol.

## Tasks

### T-801 Define the study registry and evidence levels

Status: `Blocked`  
Track: Research Design  
Depends on: AI-115  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Register each hypothesis, mechanism, comparison, primary outcome, secondary
  outcome, exclusion, and readiness gate.
- Label runs as exploratory, calibration, pilot, confirmatory, transfer, or
  qualitative.
- Define what evidence can support, weaken, or leave each hypothesis
  unresolved.
- Prevent a mechanism failure from being interpreted as a high-level negative
  result.
- Link every study to the relevant research-charter claim.

### T-802 Finalize experiment manifests and inheritance rules

Status: `Blocked`  
Track: Research Ops  
Depends on: T-003, AI-112, T-801  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Record code, config, data, checkpoint, optimizer, replay, precision,
  proposal, backend, geometry, calibration, sensor-access, and random-seed
  provenance.
- Record which learned parameters are inherited and which beliefs are reset.
- Distinguish one agent's developmental history from independent replicas.
- Record wall time, compute device, memory, interruptions, and safety events.
- Make missing provenance invalidate confirmatory status.

### T-803 Build the common measurement and analysis pipeline

Status: `Blocked`  
Track: Research Tooling  
Depends on: T-408, T-802  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Compute predictive, calibration, policy, gaze, embodiment, hierarchy, and
  spatial-temporal outcomes from versioned artifacts.
- Keep raw observations and run-level aggregates available.
- Avoid counting temporally or spatially correlated samples as independent
  agents.
- Produce the same summaries for failed and successful runs.
- Add synthetic fixtures that verify important metrics and interventions.

### T-804 Define replication, stopping, and analysis plans

Status: `Blocked`  
Track: Experimental Design  
Depends on: T-801, T-803  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Use pilot variance to choose a feasible paired-replica count before
  confirmatory evaluation.
- Define sequential stopping, failed-run handling, and compute ceilings.
- Prefer paired differences and interval estimates over isolated mean scores.
- Correct or limit multiple comparisons for secondary outcomes.
- State which conclusions remain descriptive because the affordable sample is
  small.

### T-805 Maintain the capability-readiness matrix

Status: `Blocked`  
Track: Validation/Research Ops  
Depends on: T-801, T-802  
Owner: Jackson/Codex  
Estimate: 1 day initially, then ongoing

Acceptance:

- Map every study to observation, local prediction, embodiment, relation,
  temporal persistence, policy sensitivity, and generalization gates.
- Record pass, fail, provisional, and not-tested states with artifact links.
- Block confirmatory interpretation when a required lower gate fails.
- Permit unrelated experiments to continue when their own gates pass.
- Update the matrix after architecture, sensor, backend, or calibration
  changes.

### T-806 Publish the formal baseline and approximation report

Status: `Blocked`  
Track: Baseline Research  
Depends on: AI-114, AI-115, T-802, T-803  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Report the M1 factorization, sensor boundary, reference fixtures,
  calibration, proposal effects, terminal preference, composition decision,
  checkpoint semantics, and runtime profile.
- Include three or more reproducible baseline replicas where affordable.
- Separate accepted approximations from unresolved defects.
- Include negative and interrupted runs.
- Establish the baseline against which later mechanisms are compared.

### T-807 Test H1: active foveation

Status: `Blocked`  
Track: Research/Active Vision  
Depends on: AI-304, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs

Acceptance:

- Compare active gaze, random gaze, and matched uniform observation.
- Match underlying image process, compute budget, model capacity, inherited
  weights, and terminal preferences.
- Measure ambiguity reduction, posterior calibration, predictive improvement,
  fixation loops, and long-range spatial effects.
- Include scenes where another fixation has no information value.
- Do not interpret saliency or edge preference as organization by itself.

### T-808 Test H2: embodied motor prediction

Status: `Blocked`  
Track: Research/Embodiment  
Depends on: AI-308, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs

Acceptance:

- Compare Cartesian-only and joint/body-aware policy inference.
- Match predicted canvas alternatives while varying motor consequence models.
- Measure energy/current, tracking, contact, mark geometry, policy posterior,
  and model residuals.
- Perturb joint uncertainty, damping, inertia, or reachability in controlled
  tests.
- Test for the trivial-inactivity solution and hidden motor rewards.

### T-809 Test H3: slow latent temporal persistence

Status: `Blocked`  
Track: Research/Hierarchy  
Depends on: AI-310, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus runs

Acceptance:

- Compare fast-only, fast-plus-slow, frozen-slow, shuffled-slow, and reset-slow
  conditions.
- Test whether slow state predicts later marks beyond the current observed
  canvas.
- Intervene at passage boundaries and measure later policy changes.
- Report posterior collapse, latest-observation copying, and stale-state
  failures.
- Treat completed passage count as a diagnostic, not evidence of hierarchy.

### T-810 Test H4: relational nonsemantic representation

Status: `Blocked`  
Track: Research/Relational Inference  
Depends on: AI-209, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs

Acceptance:

- Compare local-only, deterministic-relation diagnostic, and uncertain
  relational models.
- Apply split, merge, scramble, translation, rotation, overlap, and occlusion
  interventions.
- Test held-out prediction of separated and later events.
- Measure assignment uncertainty and relational posterior sensitivity.
- Avoid named aesthetic labels such as balance, harmony, or motif as targets.

### T-811 Test H5: epistemic action and pathology

Status: `Blocked`  
Track: Research/Epistemic Policy  
Depends on: AI-309, AI-313, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs

Acceptance:

- Compare reducible ambiguity, aleatoric noise, model misspecification,
  unreachable states, and repeated observation.
- Test whether information gain declines after uncertainty is resolved.
- Measure fixation loops, novelty chasing, disagreement exploitation, and
  unsafe probing.
- Compare learned, fixed, overconfident, and underconfident uncertainty.
- Report useful and pathological epistemic behavior symmetrically.

### T-812 Test H6: inherited pretrained perception

Status: `Blocked`  
Track: Research/Pretraining  
Depends on: AI-314, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 5-8 days plus runs

Acceptance:

- Compare the accepted from-scratch path with at most two frozen encoders.
- Keep the probabilistic adapter, data splits, policy model, and outcomes
  matched.
- Measure predictive gain, calibration, latency, memory, sample efficiency,
  and imported invariances.
- Cache frozen features where valid.
- Do not use pretrained semantic, aesthetic, or policy outputs.

### T-813 Test online development and inheritance

Status: `Blocked`  
Track: Research/Continual Learning  
Depends on: AI-312, T-802, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus long runs

Acceptance:

- Compare declared shared initialization, blank-slate where feasible, frozen
  parameters, continued online learning, and replay variants.
- Keep agent histories separate.
- Measure predictive improvement, calibration drift, forgetting, policy
  change, and dependence on early accidental output.
- Include multiple painting sequences rather than treating marks as replicas.
- Stop or downgrade the study if runtime makes independent histories
  infeasible.

### T-814 Run selected interaction studies

Status: `Blocked`  
Track: Research/Interactions  
Depends on: T-807, T-808, T-809, T-810, T-811  
Owner: Jackson/Codex  
Estimate: 6-10 days plus runs

Acceptance:

- Select only interactions motivated by single-mechanism results, such as
  foveation by slow hierarchy or embodiment by passage planning.
- Predeclare the reduced comparison matrix and primary contrasts.
- Avoid an unaffordable full factorial architecture sweep.
- Determine whether mechanisms compensate, interfere, or add distinct
  predictive information.
- Preserve the simplest model that answers the scientific question.

### T-815 Run simulator and hardware transfer studies

Status: `Blocked`  
Track: Research/Transfer  
Depends on: M2, T-410, T-716, T-804, T-805  
Owner: Jackson/Codex  
Estimate: 6-10 days plus hardware time

Acceptance:

- Compare native simulation, MuJoCo, recorded hardware, and available live
  hardware under matched sensor-access constraints.
- Compare observation residuals, transition calibration, posterior beliefs,
  VFE/EFE decompositions, policy posteriors, and realized consequences.
- Separate geometry/calibration mismatch from cognitive-model failure.
- Recalibrate through declared data without exposing hidden hardware truth to
  inference.
- Treat transfer failure as a result with a localized cause where possible.

### T-816 Define qualitative and conceptual-artifact protocol

Status: `Blocked`  
Track: Research Communication/Art  
Depends on: T-803  
Owner: Jackson  
Estimate: 2-3 days

Acceptance:

- Preserve complete, interrupted, failed, and representative runs.
- Pair canvases with synchronized gaze, belief, passage, body, VFE, EFE, and
  provenance traces.
- Define selection rules before assembling public comparisons.
- Permit qualitative interpretation without converting it into an undeclared
  optimization target.
- Document where conceptual framing exceeds mechanistic evidence.

### T-817 M8 synthesis and research-decision gate

Status: `Blocked`  
Track: Research Synthesis  
Depends on: T-806, T-807, T-808, T-809, T-810, T-811, T-812, T-813, T-814, T-815, T-816  
Owner: Jackson  
Estimate: 5-8 days

Acceptance:

- Summarize support, contrary evidence, ambiguity, and capability failures for
  each hypothesis.
- Separate active-inference results from supporting engineering results.
- State which conclusions depend on pretrained structure, simulation, or
  hand-declared preferences.
- Publish negative results and unresolved questions.
- Select the next architecture or experiment by information value and
  feasibility, not by the most attractive canvas.

## Concurrency And Staging

- T-801/T-802 begin after M1 and do not wait for M3.
- T-803 through T-806 develop while M2 is implemented.
- T-807 through T-813 begin independently when their exact readiness gates
  pass.
- T-814 waits for single-mechanism evidence.
- T-815 waits for transfer artifacts but does not block simulation studies.
- T-816 can develop throughout the project.

## Feasibility

- Infrastructure and design effort: 14-22 focused workdays.
- Individual study analysis effort: approximately 5-10 focused days each,
  excluding run time.
- Solo calendar estimate: 4-8 months when staged alongside development.
- Target paid compute after M3: no more than USD 200 without a new budget
  decision; use paired pilots to avoid wasting the reserve.
- Smaller credible program: complete T-806 through T-811 and defer pretraining,
  long developmental histories, interactions, and hardware transfer.

## Failure Modes

- Treating pixels, marks, or frames as independent experimental replicas.
- Selecting seeds or paintings after seeing their appearance.
- Running a large factorial sweep without enough replicas to interpret it.
- Calling a capability failure evidence against composition.
- Tuning priors or precisions on confirmatory outcomes.
- Pooling separate agents' online histories.
- Treating statistical significance as scientific importance.
- Hiding ambiguity behind a composite score.

## Outputs

- Study registry, manifests, readiness matrix, and analysis pipeline.
- M1 baseline report.
- H1-H6 mechanism reports.
- Development, interaction, and transfer reports where feasible.
- Complete qualitative and conceptual artifact archive.
- M8 synthesis and next-program decision.
