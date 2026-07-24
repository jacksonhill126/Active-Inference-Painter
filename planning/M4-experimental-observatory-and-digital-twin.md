# M4: Experimental Observatory And Digital Twin

## Summary

M4 turns the existing web viewer into a backend-neutral experimental
observatory for native simulation, MuJoCo, and later hardware. It keeps the
current Three.js scene and material canvas, but adds explicit time, provenance,
sensor-access, prediction, posterior, execution, and artifact semantics.

M4 is supporting engineering. It does not perform state inference, choose
painting policies, add preferences, or simulate dynamics in the browser.
Python remains authoritative for the generative process, generative model,
policy inference, control, telemetry, and material state.

The milestone is deliberately split:

- Contract, native-runtime, visualization, and artifact work can begin after
  the S0 schemas exist.
- MuJoCo parity work waits only for the corresponding S2 backend artifacts.
- Optional belief and prediction overlays consume the M1 logging contract but
  do not block native/MuJoCo rendering.

## Scientific Intent

M4 does not test the central active-inference hypotheses directly. It makes
those tests inspectable and reproducible by separating:

- hidden simulator truth used only for evaluation;
- observations actually available to the agent;
- prior and posterior beliefs;
- counterfactual predictions;
- selected policies and conventional controller targets;
- realized body, contact, and material consequences.

A polished viewer that blurs these categories is a failure. A plain viewer
that preserves their provenance, clocks, and uncertainty is useful.

## Scope

### Included

- Versioned experimental-state and display-state schemas.
- Explicit simulation, observation, inference, policy, control, telemetry,
  canvas, and wall-clock timestamps.
- State and canvas revision identifiers.
- Native, MuJoCo, and eventual hardware capability declarations.
- Backend-neutral Three.js rendering.
- Material truth, agent observation, and predicted-canvas views with explicit
  labels and access boundaries.
- Prior/posterior/prediction/execution overlays where data exists.
- Deterministic artifact capture for experiments.
- API, schema, synchronization, browser, and parity tests.

### Deferred

- Photorealistic rendering or CAD mesh presentation.
- Browser-side kinematics, contact reconstruction, physics, or inference.
- A general telemetry database or cloud dashboard.
- Camera calibration itself; M5 and M7 own calibration procedures.
- Real-time video streaming unless required by a later hardware experiment.
- Interactive editing of beliefs, EFE terms, or policies from the viewer.
- Treating visualization interpolation as physical or inferential data.

## Boundary Rules

1. `/api/state` and related Python endpoints are authoritative.
2. Display-only smoothing never enters telemetry, inference, paint deposition,
   or experiment artifacts.
3. Missing values are `null` or explicitly unsupported, never meaningful
   zeros.
4. Hidden truth and agent-accessible observations are visually and
   structurally distinct.
5. VFE and EFE decompositions remain separate.
6. Display labels identify approximations rather than presenting them as
   exact Bayesian quantities.
7. Backend selection is recorded at startup and cannot silently fall back.

## Tasks

### T-401 Define the experimental-state contract

Status: `Blocked`  
Track: Architecture/Research Ops  
Depends on: T-003, T-103, T-105  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Define required and optional fields for world geometry, canvas geometry,
  contact, actuator telemetry, controller targets, agent observations, beliefs,
  predictions, policy diagnostics, capabilities, and provenance.
- Declare units, frames, shapes, null semantics, and schema versions.
- Separate authoritative process state, agent-accessible observation,
  evaluation-only truth, and display-only values.
- Preserve existing backend-neutral `/api/state` fields where practical.
- Provide a migration rule for schema changes.

### T-402 Define clocks, revisions, and provenance

Status: `Blocked`  
Track: Research Ops/Telemetry  
Depends on: T-002, T-401  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Define simulation, sensor, inference, policy, control, telemetry, canvas, and
  wall-clock timestamps.
- Add monotonically increasing state, observation, belief, policy, and canvas
  revisions where their rates differ.
- Report code, config, checkpoint, backend, model, calibration, geometry, and
  hardware versions.
- State how dropped, delayed, repeated, or stale samples are represented.
- Make synchronized experiment artifacts refer to exact revisions.

### T-403 Implement the native runtime adapter

Status: `Blocked`  
Track: Runtime/Web  
Depends on: T-401, T-402  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Map the current native runtime into the versioned contract.
- Keep native kinematics and telemetry authoritative in Python.
- Populate capability flags rather than requiring every future field.
- Preserve existing API behavior through an explicit compatibility layer.
- Add no browser-side corrections to physical or inferential state.

### T-404 Reuse the Three.js arm and canvas scene

Status: `Blocked`  
Track: Web  
Depends on: T-403  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Render backend-provided joint points, axes, tip, canvas frame, and contact
  through one code path.
- Retain orbit, pan, zoom, home, and face-canvas controls.
- Keep stable scene dimensions so optional overlays do not shift the layout.
- Render unavailable geometry as unavailable rather than reconstructing it.
- Preserve native visual behavior within screenshot tolerance.

### T-405 Separate material truth, observation, and prediction views

Status: `Blocked`  
Track: Web/Observation Boundary  
Depends on: T-102, T-403  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Keep `/api/canvas.png` as the material-rendering source for process truth.
- Add distinct optional endpoints or payloads for the agent's visual
  observation and predicted canvas statistics.
- Label evaluation truth so it cannot be mistaken for agent input.
- Keep hidden thickness, wetness, and pigment arrays out of agent-view
  payloads.
- Synchronize all images with explicit canvas and observation revisions.

### T-406 Handle capabilities, stale state, and runtime errors

Status: `Blocked`  
Track: Web/Runtime  
Depends on: T-402, T-403, T-404  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Display unsupported fields as unavailable and report capability reasons.
- Detect stale state and retain the last valid scene with a visible timestamp.
- Surface backend initialization and runtime errors without mislabeling the
  run.
- Keep a clear distinction between paused, planning, retracting, executing,
  faulted, and disconnected states.
- Make invalid or non-finite state fail contract validation.

### T-407 Add belief, prediction, and policy overlays

Status: `Blocked`  
Track: Web/Active-Inference Diagnostics  
Depends on: AI-101, T-403, T-405  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Show prior and posterior state summaries without presenting means as full
  distributions.
- Show predicted and realized mark/body consequences using declared units.
- Keep VFE state-inference and EFE policy-inference decompositions separate.
- Identify fixed precisions, inferred precisions, ensemble disagreement, and
  other approximations distinctly.
- Hide unavailable overlays cleanly; diagnostics do not become an M1 blocker.

### T-408 Implement synchronized experiment artifact capture

Status: `Blocked`  
Track: Research Ops  
Depends on: T-003, T-402, T-405, T-407  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Capture state JSON, config and version manifest, telemetry window, canvas
  truth, agent observation, predictions, and screenshots at named events.
- Support mark start/end, passage start/end, stop, fault, and manual snapshot.
- Use deterministic filenames and revision references.
- Record missing capabilities in the artifact rather than silently omitting
  them.
- Keep artifact capture off the policy-selection path.

### T-409 Add contract and synchronization tests

Status: `Blocked`  
Track: Validation  
Depends on: T-403, T-405, T-406, T-407, T-408  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Validate required fields, units, frames, revisions, nulls, and capability
  semantics.
- Test stale, missing, non-finite, out-of-order, and mismatched-revision cases.
- Verify hidden truth does not enter the agent-observation payload.
- Verify VFE and EFE fields cannot overwrite each other.
- Preserve existing web-runtime and canvas tests.

### T-410 Capture native/MuJoCo parity artifacts

Status: `Blocked`  
Track: Validation/Simulation  
Depends on: T-309, T-404, T-405, T-408  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Run matched fixed-pose, free-motion, contact, and scripted-stroke cases.
- Save synchronized state, telemetry, canvas, observation, and viewer
  artifacts for both backends.
- Compare frames, tip path, contact timing, actuator outcomes, and paint
  consequences numerically.
- Classify differences as expected model differences, calibration needs,
  contract defects, or implementation defects.
- Do not require physically different backends to produce identical pixels.

### T-411 Validate responsiveness and visual legibility

Status: `Blocked`  
Track: Web/Manual Validation  
Depends on: T-404, T-405, T-406, T-407  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Verify desktop and mobile layouts without overlapping controls or telemetry.
- Measure state, image, and artifact-capture overhead during planning and
  execution.
- Confirm the viewer does not materially increase brush-stroke latency.
- Capture nonblank screenshots for native and available MuJoCo states.
- Confirm provenance and truth/observation distinctions remain visible.

### T-412 M4 observatory gate

Status: `Blocked`  
Track: Validation  
Depends on: T-409, T-410, T-411  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- Native and MuJoCo use one tested contract.
- Clocks, revisions, capabilities, and provenance are accepted.
- Hidden truth, agent observations, beliefs, predictions, and execution are
  distinguishable.
- Artifact capture is synchronized and reproducible.
- Browser overhead is bounded and all remaining gaps are documented.

## Concurrency And Gate Logic

- T-401 through T-409 and T-411 can proceed against native state before S2 is
  complete.
- T-410 is the only task that requires validated native/MuJoCo parity inputs.
- T-407 may trail M1 formalization without blocking core rendering.
- M4 does not block M2/M3 inference work; it should consume their versioned
  diagnostic outputs incrementally.
- M4 completion is required before viewer artifacts are treated as formal
  experimental evidence, not before model development begins.

## Feasibility

- Estimated effort: 15-23 focused workdays.
- Solo calendar estimate: 4-6 weeks, interleaved with S2 and M1/M2.
- Model-training cost: negligible.
- Main risk: allowing a display schema to become a second architecture rather
  than a faithful, versioned projection of runtime state.

## Failure Modes

- A convenient viewer field becomes an undeclared observation for the agent.
- Display interpolation is exported as physical telemetry.
- Canvas truth and agent observation are visually indistinguishable.
- Missing backend values appear as zeros.
- Screenshots cannot be traced to exact state and canvas revisions.
- Diagnostic computation slows policy inference enough to change behavior.
- MuJoCo progress is interpreted as evidence for active inference.

## Outputs

- Versioned experimental-state schema.
- Native and MuJoCo runtime adapters.
- Backend-neutral Three.js scene.
- Separate truth, observation, belief, prediction, and execution views.
- Synchronized artifact-capture bundle.
- Contract, synchronization, parity, and browser validation results.
- M4 observatory-gate note.
