# M3: MuJoCo Backend Adapter

## Summary

M3 connects the existing controller and painting loop to MuJoCo without changing
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

Status: `Backlog`  
Track: Architecture  
Depends on: M1, M2  
Owner: TBD  
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

### T-302 Map `ArmPose` targets to MuJoCo controls

Status: `Backlog`  
Track: Control  
Depends on: T-301  
Owner: TBD  
Estimate: 1 day

Acceptance:

- Degree-based `ArmPose` target values are converted to MuJoCo control units
  exactly once inside the backend.
- Joint order matches M2.
- Controller target values are clipped or rejected consistently with the
  backend safety contract.
- Target pose remains available to telemetry in degrees.

### T-303 Read MuJoCo state into existing pose/contact structures

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-301, T-302  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- Current MuJoCo qpos is exposed as an `ArmPose`-compatible degree state.
- MuJoCo `tip` site position is exposed in the same world/canvas coordinate
  convention as the native simulator.
- Contact state includes on-canvas, touching, deflection, pressure, force,
  brush width, and brush world position.
- State shape is compatible with `/api/state`.

### T-304 Reuse `VerticalCanvas` for MuJoCo-driven paint

Status: `Backlog`  
Track: Painting Model  
Depends on: T-303  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- MuJoCo brush tip/contact drives `VerticalCanvas.paint_at`.
- Existing brush loading, wet blending, bristle texture, material coverage, and
  white-on-white behavior remain unchanged.
- Painting can be disabled without changing arm motion.
- Canvas PNG rendering continues to use the existing material renderer.

### T-305 Add scripted-stroke smoke tests

Status: `Backlog`  
Track: Validation  
Depends on: T-302, T-303, T-304  
Owner: TBD  
Estimate: 1 day

Acceptance:

- With optional MuJoCo installed, a scripted stroke moves the tip near the
  canvas and deposits nonzero material coverage.
- Without optional MuJoCo installed, tests skip cleanly.
- Test records basic telemetry: final pose, contact pressure, and coverage.

### T-306 Add backend selection to web runtime

Status: `Backlog`  
Track: Web/Runtime  
Depends on: T-301, T-304  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- Web runtime can choose native or MuJoCo backend from a command-line flag.
- Default remains native backend until MuJoCo path is validated.
- `/api/state`, `/api/canvas.png`, and `/api/telemetry.csv` keep their current
  contract for the frontend.

Suggested CLI:

```powershell
python -m active_painter.web_server --backend native
python -m active_painter.web_server --backend mujoco
```

### T-307 Adapt telemetry for MuJoCo backend

Status: `Backlog`  
Track: Telemetry  
Depends on: T-303, T-306  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- Telemetry rows remain schema-compatible where possible.
- Fields unavailable in MuJoCo are zeroed, approximated, or marked explicitly;
  the choice is documented.
- Backend identity and model version are included in runtime diagnostics.

### T-308 Define MuJoCo forecast strategy

Status: `Backlog`  
Track: Planning/Forecasting  
Depends on: T-301, T-305  
Owner: TBD  
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

### T-309 Add backend parity checks

Status: `Backlog`  
Track: Validation  
Depends on: T-305, T-306  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- Same scripted stroke can run on native and MuJoCo backends.
- Compare tip path, final pose, contact timing, pressure summary, and material
  coverage.
- Differences are recorded as calibration needs, not hidden as test noise.

### T-310 M3 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-305, T-306, T-307, T-308, T-309  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- M3 is locked only if the controller can drive MuJoCo through the backend
  interface and live paint updates through the existing material model.
- Any deferred forecasting or telemetry gaps are documented before M4.

## Validation Gate

M3 is complete when:

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

## M3 Output Artifacts

- Backend interface or protocol documentation.
- MuJoCo backend implementation plan result.
- Scripted-stroke smoke test result.
- Web backend-selection behavior.
- Telemetry compatibility note.
- M3 lock note or blocker list.

