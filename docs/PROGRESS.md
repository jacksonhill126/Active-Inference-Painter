# Project Progress

This document is the concise public record of what has been demonstrated, what
has failed, and what comes next. Detailed task state remains in
`planning/PROJECT_TRACKER.md`.

## Current Snapshot

Snapshot date: 2026-07-24.

Phase: M1 formal baseline audit with S0 native reference work in parallel.

The current prototype can:

- run the native arm, contact, and oil-paint generative process;
- infer individual-mark and hierarchical passage candidates;
- compare stochastic motor realizations;
- learn local transition and hierarchy models online;
- checkpoint learned state and export telemetry;
- render the live arm and canvas in a browser.

It has not yet demonstrated:

- inference from only sensor-equivalent observations;
- independently verified VFE and EFE decompositions;
- calibrated predictive uncertainty at live scale;
- a predictively necessary global or relational latent;
- MuJoCo parity;
- physical hardware control or sim-to-real transfer;
- emergent composition.

## Verification Snapshot

Local environment: Windows 11, Python 3.14.3, PyTorch 2.11.0+cu126. CUDA was
available, but this test result is not a GPU performance benchmark.

| Check | Result | Interpretation |
| --- | --- | --- |
| Deterministic smoke suite | 59 passed; 11.08 seconds observed | Includes observation-factorization, plant-interface, and sensor-access contracts |
| Native plant/material suite | 55 passed; 2.48 seconds observed | Geometry, reset, and material contracts are green |
| AI-103/T-103 focused contract suite | 12 passed; 3.85 seconds observed | Observation factorization/units and `plant-interface-v1` records are green |
| Complete suite | 252 passed; 349.09 seconds observed; exit 0 | Current baseline and all new M0/AI/S0 contract tests are green |

These timings are local observations, not stable performance claims. Hardware,
operating system, dependency versions, and concurrent load were not yet
captured in a run manifest.

## Current Priorities

1. Verify VFE against independent analytic reference calculations in AI-104.
2. Build the leakage-resistant live-scale baseline corpus in AI-108.
3. Capture baseline web/telemetry behavior in T-105.
4. Document and classify simulator shortcuts in T-106.
5. Complete the T-208 standalone MuJoCo viewer inspection and accept or revise
   the physical-draft/logical-retarget contract.
6. Capture the T-309 matched native/MuJoCo stroke parity artifact.
7. Move provisional MJCF geometry into the backend-neutral T-501
   `RobotGeometrySpec`.

## Progress Log

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
