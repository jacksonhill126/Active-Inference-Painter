# Project Progress

This document is the concise public record of what has been demonstrated, what
has failed, and what comes next. Detailed task state remains in
`planning/PROJECT_TRACKER.md`.

## Current Snapshot

Snapshot date: 2026-08-04.

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
- verify the AI-104/AI-105 VFE/EFE acceptance matrix against independent
  analytic, enumerated, and fine-grid references;
- checkpoint learned state and export telemetry;
- render the live arm and canvas in a browser.

It has not yet demonstrated:

- a complete live painting loop driven only by sensor-equivalent observations;
- calibrated predictive uncertainty at live scale;
- a predictively necessary global or relational latent;
- a proposal-invariant policy posterior or a correction for finite-candidate
  bias; the completed convergence audit was negative;
- a frozen or cross-fitted safeguard for the self-trained composition
  preference;
- MuJoCo parity;
- physical hardware control or sim-to-real transfer;
- emergent composition.

The default `sensor_equivalent` runtime therefore remains deliberately
fail-closed: visualization and scripted execution are available, but painting
policy inference and learning are disabled. The `oracle_material_state` mode is
an explicit diagnostic upper-bound comparator, not a sensor-based agent.

## Verification Snapshot

Local environment: Windows 11, Python 3.14.3, PyTorch 2.11.0+cu126. CUDA was
available, but this test result is not a GPU performance benchmark.

| Check | Result | Interpretation |
| --- | --- | --- |
| Current test collection | 448 tests collected | Includes the learned-proposal, AI-111 convergence, completed AI-104/AI-105 reference, documentation-contract, and MuJoCo forecast-alignment coverage |
| Complete suite, uncontended audit run | 415 passed; 527 seconds observed | Recorded in `docs/DEVELOPMENT_AUDIT.md`; deadlines were not relaxed |
| Independent 2026-08-04 review | 414 passed; one Windows temp-directory setup error | The affected synthetic calibration test body passed separately with 11 usable views and 0.120 px RMS error |
| Proposal and AI-111 focused suites | 20 passed; 8.67 seconds observed | Covers normalized support, parity, training, checkpointing, deterministic convergence metrics, and retained run provenance |
| Current deterministic CI gate | 164 passed; 20.82 seconds observed | Includes both proposal suites and all files listed in `.github/workflows/ci.yml`; two expected obsolete-summary warnings |
| Pre-AI111 complete-suite attempt | No terminal result before the fixed 15-minute limit | Stopped under load with no failure traceback; this is not reported as either a pass or a code failure |
| Source checks | Python compilation and `git diff --check` passed | No truncated source or malformed patch was found |
| MuJoCo forecast alignment | 60 focused motor, plant, MuJoCo, documentation, and runtime tests passed | Selected-plant copying, independent rollout state, live-state non-mutation, contact/material execution, provenance, and the fail-closed sensor boundary are covered |

These timings are local observations, not stable performance claims. Hardware,
operating system, dependency versions, and concurrent load were not yet
captured in a run manifest.

## Current Priorities

1. Stress-test terminal coverage and stopping in AI-106, especially the
   low-coverage regime outside the accepted discrete/continuous KL oracle band.
2. Build the leakage-resistant live-scale baseline corpus in AI-108, then
   measure held-out likelihood and uncertainty calibration in AI-107/AI-109.
3. Resolve the composition-preference closed loop in AI-110 and carry AI-111's
   negative convergence result into an explicit M3 proposal correction. Until
   then, keep the learned mix at zero and label posterior mass as conditional on
   the sampled candidate set.
4. Define checkpoint inheritance and online-learning semantics, profile the
   major runtime phases, and capture three manifested baseline replicas.
5. Make the M1 lock decision before treating ongoing M2 work as an accepted
   sensor model.
6. After M1, connect the body posterior to motor forecasts and continuously
   pair executed-action transition priors with delivered camera updates.

## Progress Log

### 2026-08-04: selected-plant motor forecast alignment

- Removed the compatibility hook that silently replaced MuJoCo with
  `native-abstract-v0` during deep-copied policy forecasts.
- MuJoCo forecast particles now own independent `MjData` under the same
  immutable `mujoco-robstride-electromechanical-v4` model used for execution.
- Proprioceptive forecasts use physical MuJoCo positions, targets, joint limits,
  current, torque, velocity, contact, and per-joint RobStride/MJCF scales.
- Added explicit backend, initialization, and approximation provenance. The
  path remains `baseline-oracle-v0` because it starts from exact process state;
  MuJoCo body-parameter uncertainty and live body-posterior initialization are
  still open.

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
