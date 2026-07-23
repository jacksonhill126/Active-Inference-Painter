# M1: Baseline Lock

## Summary

M1 establishes the current Python implementation as the canonical abstract
baseline before MuJoCo, CAD, or hardware become authoritative. The goal is to
make the existing simulator, material model, controller boundary, telemetry,
and tests explicit enough that later robotics work can be compared against a
stable reference.

This milestone does not add new robotics capability. It freezes assumptions,
documents known shortcuts, and records a passing baseline.

## Baseline Contracts

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

Status: `Ready`  
Track: Simulation  
Depends on: M0  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Document arm constants: joint order, joint ranges, home pose, link lengths,
  canvas frame, contact depth, and brush radius behavior.
- Identify which fields are abstract simulator truth versus representative
  placeholders.
- Record that MuJoCo M2 should match these constants before introducing
  measured physical offsets.

Implementation notes:

- Primary files: `src/active_painter/arm_sim.py`,
  `src/active_painter/arm_control.py`, `models/README.md`.
- Relevant tests: `tests/test_arm_sim.py`, `tests/test_mujoco_model.py`.

### T-102 Lock canvas material invariants

Status: `Ready`  
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

### T-103 Lock controller and policy boundary

Status: `Ready`  
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

Implementation notes:

- Primary files: `src/active_painter/stroke_execution.py`,
  `src/active_painter/arm_agent_driver.py`,
  `src/active_painter/motor_planning.py`.
- Relevant tests: `tests/test_stroke_execution.py`,
  `tests/test_arm_agent_driver.py`, `tests/test_motor_telemetry.py`,
  `tests/test_motor_reliability.py`.

### T-104 Record full baseline test result

Status: `Ready`  
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

### T-105 Capture baseline telemetry and web-runtime behavior

Status: `Backlog`  
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

Status: `Backlog`  
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

- `AUDIT.md`
- `README.md`
- `RESEARCH_CHARTER.md`

### T-107 Define baseline artifact bundle

Status: `Backlog`  
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

### T-108 Baseline lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-104, T-105, T-106, T-107  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- Mark M1 as locked only if baseline tests pass or failures are documented and
  judged non-blocking.
- State whether M2 MuJoCo work may use the current Python sim as reference.
- Record any blocking issues as tracker tasks before moving to M2/M3.

## Validation Gate

M1 is complete when:

- The current Python simulator is explicitly documented as the abstract
  reference.
- Canvas material invariants are protected by tests.
- Controller/policy boundary is documented and test-backed.
- A baseline test result is recorded.
- Web runtime and telemetry behavior have a known reference.
- Known shortcuts are documented rather than hidden.

## Failure Modes To Watch

- A MuJoCo or CAD model becomes treated as more authoritative than the current
  Python baseline before measurement.
- A controller change silently chooses painting policies instead of realizing
  selected `StrokeAction`s.
- Coverage is inferred from visible tone instead of material thickness.
- Simulator-only observations leak into claims about real robot perception.
- Baseline failures are hand-waved instead of logged with reproduction steps.

## M1 Output Artifacts

- Baseline notes in planning or audit documentation.
- Passing or explicitly triaged test result.
- Optional `runs/baseline/` artifact bundle.
- Tracker updates for any blockers discovered during baseline validation.

