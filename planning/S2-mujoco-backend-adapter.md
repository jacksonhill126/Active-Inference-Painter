# S2: MuJoCo Backend Adapter

## Summary

S2 connects the existing controller and painting loop to MuJoCo without changing
painting-policy semantics. The backend adapter should let the current runtime
drive either the native Python simulator or MuJoCo through the same controller
surface.

MuJoCo is a generative process backend here. It supplies joint dynamics, body
state, actuator state, and brush tip pose. The existing material model still
owns paint deposition and canvas state.

## Backend Contracts

- `StrokeAction` remains Cartesian/contact intent.
- Existing stroke controllers remain above the backend.
- Backend state must expose enough data for the current web runtime,
  telemetry log, planner diagnostics, and motor forecasts.
- `ArmPose` remains degree-based at the controller boundary.
- MuJoCo runtime state may use radians internally, but conversion must stay in
  the backend adapter.
- The adapter must not introduce painting-level rewards, motor-ease scores, or
  policy preferences.

## Tasks

### T-301 Define common backend surface

Status: `Done`
Track: Architecture  
Depends on: T-103, T-201, T-203
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Define the minimal methods/properties shared by native and MuJoCo backends:
  target setting, stepping, reset, pose, target pose, contact, canvas, plant
  telemetry, kinematics-compatible points, and render points.
- Keep the surface narrow enough that `ArmPainterSim` can satisfy it without a
  large refactor.
- Document which existing call sites are backend-dependent.

Primary call sites:

- `src/active_painter/web_runtime.py`
- `src/active_painter/stroke_execution.py`
- `src/active_painter/telemetry_log.py`

Notes:

- Accepted 2026-07-28 through `plant-interface-v1`,
  `MujocoPlantBackend`, and `MujocoJointPlant`.
- Backend-specific calls remain below `ArmPainterSim`, stroke execution,
  telemetry, and web runtime.

### T-302 Map `ArmPose` targets to MuJoCo controls

Status: `Done`
Track: Control  
Depends on: T-301  
Owner: Jackson/Codex
Estimate: 1 day

Acceptance:

- Degree-based `ArmPose` target values are converted to MuJoCo control units
  exactly once inside the backend.
- Joint order matches M2.
- Controller target values are clipped or rejected consistently with the
  backend safety contract.
- Target pose remains available to telemetry in degrees.

Notes:

- Accepted 2026-07-28 in `MujocoJointPlant.step`; degree/radian conversion,
  joint order, target clipping, encoder state, and telemetry targets have
  focused tests.

### T-303 Read MuJoCo state into existing pose/contact structures

Status: `Done`
Track: MuJoCo  
Depends on: T-301, T-302  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Current MuJoCo qpos is exposed as an `ArmPose`-compatible degree state.
- MuJoCo `tip` site position is exposed in the same world/canvas coordinate
  convention as the native simulator.
- Contact state includes on-canvas, touching, deflection, pressure, force,
  brush width, and brush world position.
- State shape is compatible with `/api/state`.

Notes:

- Accepted 2026-07-28 with direct qpos/encoder state, physical tip/site state,
  exact brush-canvas contact, force/pressure/compression/bend state, and
  runtime payload coverage.

### T-304 Reuse `VerticalCanvas` for MuJoCo-driven paint

Status: `Done`
Track: Painting Model  
Depends on: T-303  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- MuJoCo brush tip/contact drives `VerticalCanvas.paint_at`.
- Existing brush loading, wet blending, bristle texture, material coverage, and
  white-on-white behavior remain unchanged.
- Unloading material changes paint availability without changing arm motion.
- Canvas PNG rendering continues to use the existing material renderer.

Notes:

- Accepted 2026-07-28: brush material is loaded before motion and deposition is
  driven by physical contact/pressure through `VerticalCanvas`.
- Python remains the declared paint-material boundary.

### T-305 Add scripted-stroke smoke tests

Status: `Done`
Track: Validation  
Depends on: T-302, T-303, T-304  
Owner: Jackson/Codex
Estimate: 1 day

Acceptance:

- With optional MuJoCo installed, a scripted stroke moves the tip near the
  canvas and deposits nonzero material coverage.
- Without optional MuJoCo installed, tests skip cleanly.
- Test records basic telemetry: final pose, contact pressure, and coverage.

Notes:

- Accepted 2026-07-28 in `tests/test_mujoco_backend.py`; contact, force,
  pressure, coverage, deterministic snapshot/restore, and a 370-sample
  continuous-contact deposition stroke are covered.

### T-306 Add backend selection to web runtime

Status: `Done`
Track: Web/Runtime  
Depends on: T-301, T-304  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Web runtime can choose native or MuJoCo backend from a command-line flag.
- Default remains native backend until MuJoCo path is validated.
- `/api/state`, `/api/canvas.png`, and `/api/telemetry.csv` keep their current
  contract for the frontend.

Suggested CLI:

```powershell
python -m active_painter.web_server --plant-backend native
python -m active_painter.web_server --plant-backend mujoco
```

Notes:

- Accepted 2026-07-28; state, canvas PNG, telemetry CSV, and XML-driven frontend
  geometry share the runtime.

### T-307 Adapt telemetry for MuJoCo backend

Status: `Done`
Track: Telemetry  
Depends on: T-303, T-306  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Telemetry rows remain schema-compatible where possible.
- Fields unavailable in MuJoCo are zeroed, approximated, or marked explicitly;
  the choice is documented.
- Backend identity and model version are included in runtime diagnostics.

Notes:

- Accepted 2026-07-28 with backend/model identity, joint/actuator/encoder
  state, current, torque, voltage, elastic/backlash/friction/load terms,
  contact, brush-loaded, and actual-deposition fields.
- Declared approximations remain documented.

### T-308 Define MuJoCo forecast strategy

Status: `Done`
Track: Planning/Forecasting  
Depends on: T-301, T-305  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Decide whether initial MuJoCo backend supports live execution only or motor
  forecast rollouts as well.
- If forecasts are deferred, native simulator remains the forecast backend and
  this is documented as an approximation.
- If forecasts are included, simulator copy/reset semantics are defined before
  planner integration.

Default:

- M3 supports live execution first.
- MuJoCo motor forecast rollouts are deferred unless they are cheap and
  deterministic enough to copy/reset.

Notes:

- Initially accepted 2026-07-28 as live MuJoCo execution with a deferred native
  counterfactual approximation.
- Revised 2026-08-04: deep-copied policy particles now preserve
  `mujoco-robstride-electromechanical-v4` with independent `MjData`, matched
  contact/material registration, per-joint RobStride/MJCF normalization, and
  explicit forecast provenance.
- Initialization still uses exact process state in `baseline-oracle-v0`, and
  MuJoCo body-parameter uncertainty is not yet sampled. A sensor-conditioned
  `ExecutionForecaster` must not be claimed.

### T-309 Add backend parity checks

Status: `Active`
Track: Validation  
Depends on: T-305, T-306  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Same scripted stroke can run on native and MuJoCo backends.
- Compare tip path, final pose, contact timing, pressure summary, and material
  coverage.
- Differences are recorded as calibration needs, not hidden as test noise.

Notes:

- Canonical transform, logical retarget, state-shape, contact/deposition, and
  runtime selection tests pass.
- Same-plant counterfactual copying, independent rollout state, live-state
  non-mutation, provenance, and runtime-selection tests pass.
- A versioned matched-stroke artifact comparing path, timing, pressure,
  current, and material coverage remains.

### T-310 S2 lock decision

Status: `Blocked`
Track: Validation  
Depends on: T-305, T-306, T-307, T-308, T-309  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- S2 is locked only if the controller can drive MuJoCo through the backend
  interface and live paint updates through the existing material model.
- Any deferred forecasting or telemetry gaps are documented before M4.

Notes:

- Blocked on the T-309 matched parity artifact and Jackson's explicit S2 lock
  decision. The former native counterfactual substitution is no longer a
  blocker; exact-state oracle initialization remains documented M2 work.

## Validation Gate

S2 is complete when:

- Controller-facing backend surface exists and is used by native and MuJoCo
  paths.
- MuJoCo controls and state conversions are tested.
- MuJoCo brush contact deposits paint through `VerticalCanvas`.
- Web runtime can select backend without breaking existing frontend state.
- Forecasting limitations are explicitly documented.

## Failure Modes To Watch

- Degrees/radians conversion leaks above the backend.
- MuJoCo coordinate corrections are scattered across controllers or web code.
- The adapter changes painting-policy selection instead of plant realization.
- Contact pressure is treated as a global preference instead of an execution
  condition.
- Motor forecast rollouts are added before copy/reset determinism is solved.

## S2 Output Artifacts

- Backend interface or protocol documentation.
- MuJoCo backend implementation plan result.
- Scripted-stroke smoke test result.
- Web backend-selection behavior.
- Telemetry compatibility note.
- S2 lock note or blocker list.
