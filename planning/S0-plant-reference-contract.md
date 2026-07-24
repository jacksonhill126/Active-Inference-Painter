# S0: Plant Reference Contract

## Summary

S0 establishes the current Python implementation as a versioned abstract
reference while MuJoCo, CAD, control, and active-inference work continue in
parallel. The goal is to make the simulator, material model, command and
observation schemas, controller boundary, telemetry, and tests explicit enough
that later implementations can be compared against a reproducible reference.

This milestone does not freeze controller gains, trajectory algorithms,
dynamics fidelity, geometry estimates, or mechanical design. Those may change
under new versions. S0 stabilizes interfaces and provenance, documents known
shortcuts, and records a passing reference baseline.

## Reference Contracts

- The Python `ArmPainterSim` is the abstract reference plant until measured
  hardware geometry exists.
- `StrokeAction` remains a Cartesian/contact painting-policy representation.
- IK, trajectory realization, motor dynamics, contact mechanics, and hard
  safety remain below painting policy selection.
- `VerticalCanvas` remains the source of material truth: thickness, wetness,
  bulk pigment mass, visible surface tone, observed tone, ground contrast, and
  material coverage.
- White paint on white ground must increase thickness and material coverage.
- Immediate `stop` and multi-stroke-then-stop policies remain available in
  every planning cycle.

## Tasks

### T-101 Confirm Python arm sim as canonical abstract reference

Status: `Done`
Track: Simulation  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Document arm constants: joint order, joint ranges, home pose, link lengths,
  canvas frame, contact depth, and brush radius behavior.
- Identify which fields are abstract simulator truth versus representative
  placeholders.
- Record that MuJoCo S1 should match these constants before introducing
  measured physical offsets.

Implementation notes:

- Primary files: `src/active_painter/arm_sim.py`,
  `src/active_painter/arm_control.py`, `models/README.md`.
- Relevant tests: `tests/test_native_contract.py`, `tests/test_arm_sim.py`.

Notes:

- Accepted 2026-07-24 in `docs/NATIVE_PLANT_REFERENCE.md` as
  `native-abstract-v0`.
- Focused native contract, canvas, and arm tests passed: 55 tests.

### T-102 Version and protect canvas material invariants

Status: `Done`
Track: Painting Model  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Tests confirm material coverage derives from thickness, not visible tone.
- Tests confirm white-on-white paint increases material coverage.
- Tests confirm repeated paint in already covered cells changes material state
  without increasing covered area.
- Tests confirm wetness persists within a session.
- Tests confirm brush/contact parameters remain configurable.

Implementation notes:

- Primary files: `src/active_painter/arm_sim.py`,
  `src/active_painter/config.py`.
- Relevant tests: `tests/test_canvas.py`, `tests/test_arm_sim.py`,
  `tests/test_spatial_state.py`.

Notes:

- Accepted 2026-07-24 in `docs/NATIVE_PLANT_REFERENCE.md`.
- Added direct tests for repeated-layer coverage, black/white-equivalent
  occupancy, visible-tone independence, clear semantics, configurable
  presence/contact parameters, and the native geometry contract.
- Focused native contract, canvas, and arm tests passed: 55 tests.

### T-103 Define controller, plant, and policy interfaces

Status: `Done`
Track: Control  
Depends on: T-101  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Document that `StrokeAction` is policy intent, not a joint trajectory.
- Document that IK and motor primitives realize selected policies below active
  inference.
- Confirm motor feasibility affects admissibility and predicted consequences,
  not an aesthetic or motor-ease reward.
- Confirm hard joint/current/workspace/watchdog limits remain external safety
  constraints.
- Define the backend-neutral command, sensor, timestamp, and capability fields
  needed by native, MuJoCo, and eventual hardware implementations.
- Separate commanded values, physical sensor samples, inferred state, and
  simulator-only evaluation truth in the interface types.
- Define a conforming forecast interface that cannot accept a live process
  object or copied process RNG state. Counterfactual requests start from a
  declared belief/model snapshot and an independent future-noise seed.
- State explicitly that low-level control laws and gains remain replaceable.

Implementation notes:

- Primary files: `src/active_painter/stroke_execution.py`,
  `src/active_painter/arm_agent_driver.py`,
  `src/active_painter/motor_planning.py`.
- Relevant tests: `tests/test_stroke_execution.py`,
  `tests/test_arm_agent_driver.py`, `tests/test_motor_telemetry.py`,
  `tests/test_motor_reliability.py`.
- Boundary input: `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`, especially the
  true-pose/contact and forecast snapshot leakage findings.

Notes:

- Accepted 2026-07-24 as `plant-interface-v1` in
  `docs/CONTROL_PLANT_POLICY_BOUNDARY.md`.
- `src/active_painter/plant_interface.py` defines separate SI-unit command,
  physical-sensor, posterior-belief, counterfactual, capability, and
  evaluation-truth records.
- Four focused interface tests pass.
- The current native runtime has not migrated to this interface and remains a
  documented oracle baseline. T-109 tracks removal of live simulator and RNG
  state from motor forecasts.

### T-104 Record full baseline test result

Status: `Done`
Track: Validation  
Depends on: T-101, T-102, T-103  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Run `python -m pytest`.
- Record Python version, dependency state, command, pass/fail count, duration,
  and notable skips/failures.
- If full suite is too slow, record the split commands used and why.
- Link any failures to failure-mode entries instead of silently ignoring them.

Baseline commands:

```powershell
python -m pytest
```

Optional focused commands:

```powershell
python -m pytest tests\test_arm_sim.py tests\test_canvas.py tests\test_stroke_execution.py
python -m pytest tests\test_arm_agent_driver.py tests\test_motor_telemetry.py tests\test_motor_reliability.py
python -m pytest tests\test_mujoco_model.py
```

Notes:

- Accepted 2026-07-24 in
  `docs/BASELINE_TEST_RESULT_2026-07-24.md`.
- `python -m pytest -q` completed with 252 passing tests in 349.09 seconds and
  exit status 0 on Python 3.14.3.

### T-105 Capture baseline telemetry and web-runtime behavior

Status: `Ready`
Track: Web/Telemetry  
Depends on: T-101, T-103  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Start the web runtime with default settings.
- Confirm `/api/state`, `/api/canvas.png`, and `/api/telemetry.csv` respond.
- Confirm the Three.js frontend displays arm points, canvas texture, contact
  state, policy diagnostics, and telemetry summaries.
- Save a short baseline telemetry CSV and note the run configuration.

Baseline command:

```powershell
python -m active_painter.web_server --driver-bootstrap-transitions 0 --driver-bootstrap-train-steps 0
```

### T-106 Document known simulator shortcuts and limitations

Status: `Ready`
Track: Documentation  
Depends on: T-101, T-102, T-103  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- List exact simulator state observations that a real robot would need to
  infer through sensors.
- List nonphysical or representative dynamics assumptions.
- List contact/brush simplifications.
- List planning shortcuts such as finite candidate sets, approximated rollout
  densities, and simulator-only summary observations.
- Mark each shortcut as acceptable baseline, needs MuJoCo calibration, or needs
  hardware validation.

Suggested sources:

- `docs/DEVELOPMENT_AUDIT.md`
- `README.md`
- `docs/RESEARCH_CHARTER.md`

### T-107 Define baseline artifact bundle

Status: `Blocked`
Track: Research Ops  
Depends on: T-003, T-104, T-105  
Owner: Jackson/Codex  
Estimate: 0.5-1 day

Acceptance:

- Define where baseline artifacts live, initially `runs/baseline/`.
- Bundle includes test output summary, config snapshot, short telemetry CSV,
  one canvas image, and notes on known limitations.
- Bundle records code version, backend, planner mode, canvas size, and random
  seeds where available.

### T-108 S0 reference-contract decision

Status: `Blocked`
Track: Validation  
Depends on: T-104, T-105, T-106, T-107  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- Accept the S0 reference contract only if baseline tests pass or failures are
  documented and judged non-blocking for the intended comparison.
- State whether S1 MuJoCo work may use the current Python sim as reference.
- List which fields are stable interfaces and which plant/controller details
  remain provisional.
- Record any blocking issues as tracker tasks before moving to S1/S2.

### T-109 Migrate native execution to `plant-interface-v1`

Status: `Blocked`
Track: Control/Inference Boundary
Depends on: T-103, AI-201, AI-203, AI-204
Owner: Jackson/Codex
Estimate: 3-5 days

Acceptance:

- Adapt the native command and physical-sensor paths to `PlantBackend`.
- Build the live body posterior only from permitted sensor packets and model
  assumptions.
- Initialize execution forecasts from `BodyBeliefSnapshot` and an immutable
  model snapshot, never from a copied `ArmPainterSim`.
- Sample future process and sensor noise independently of the generative
  process RNG continuation.
- Keep simulator-only evaluation truth on the separate evaluation interface.
- Demonstrate native forecast and execution behavior with boundary tests and
  update the sensor-access ledger.

## Reference Gate

S0 is complete when:

- The current Python simulator is explicitly documented as the abstract
  reference.
- Canvas material invariants are protected by tests.
- Controller/policy boundary is documented and test-backed.
- A baseline test result is recorded.
- Web runtime and telemetry behavior have a known reference.
- Known shortcuts are documented rather than hidden.
- Subsequent controller, geometry, or dynamics revisions can be represented by
  new versions without changing the painting-policy boundary.

## Failure Modes To Watch

- A MuJoCo or CAD model becomes treated as more authoritative than the current
  Python baseline before measurement.
- Treating the reference contract as a permanent controller or mechanical
  design freeze.
- A controller change silently chooses painting policies instead of realizing
  selected `StrokeAction`s.
- Coverage is inferred from visible tone instead of material thickness.
- Simulator-only observations leak into claims about real robot perception.
- Baseline failures are hand-waved instead of logged with reproduction steps.

## S0 Output Artifacts

- Baseline notes in planning or audit documentation.
- Passing or explicitly triaged test result.
- Optional `runs/baseline/` artifact bundle.
- Versioned plant, command, observation, and capability schemas.
- Tracker updates for any blockers discovered during baseline validation.
