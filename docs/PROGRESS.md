# Project Progress

This document is the concise public record of what has been demonstrated, what
has failed, and what comes next. Detailed task state remains in
`planning/PROJECT_TRACKER.md`.

## Current Snapshot

Snapshot date: 2026-07-23.

Phase: M0 project operating system and portfolio foundation.

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

Local environment: Windows, Python 3.11-compatible project configuration, CPU
execution.

| Check | Result | Interpretation |
| --- | --- | --- |
| Deterministic smoke suite | 33 passed; 3.94-5.54 seconds observed | Suitable for per-change CI |
| Full suite | 225 passed, 1 failed in 376.42 seconds | Broad coverage, not currently release-green |
| Current full-suite failure | background planning exceeded a 15-second test deadline | Performance-sensitive integration test; tracked rather than hidden |
| Suite excluding driver integration file | 190 passed in 199.70 seconds | Other model tests are also computationally expensive |

These timings are local observations, not stable performance claims. Hardware,
operating system, dependency versions, and concurrent load were not yet
captured in a run manifest.

## Current Priorities

1. Complete M0 validation-gate definitions.
2. Establish the M1 executable generative-model and sensor-access
   specifications.
3. Split deterministic unit tests from slow stochastic and integration tests.
4. Add reproducible baseline runs with measured prediction, calibration,
   policy, execution, and latency outputs.
5. Stabilize the native plant contract before MuJoCo parity work.

## Progress Log

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
