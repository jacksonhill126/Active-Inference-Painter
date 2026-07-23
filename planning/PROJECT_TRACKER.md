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
### T-000 Task title

Status:
Track:
Depends on:
Owner:
Estimate:
Acceptance:
Notes:
```

## M0: Project Operating System

### T-001 Define tracker conventions

Status: `Ready`  
Track: Operations  
Depends on: none  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: Task fields, status meanings, dependency rules, and validation rules are documented.

### T-002 Define versioning scheme

Status: `Ready`  
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5-1 day  
Acceptance: Version labels exist for code, MuJoCo XML, CAD, calibration, hardware build, and experiment config.

### T-003 Define experiment manifest requirements

Status: `Ready`  
Track: Research Ops  
Depends on: T-001, T-002  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Run metadata and required traces are documented.

### T-004 Define failure-mode log

Status: `Ready`  
Track: Validation  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Failure categories and log fields are documented.

### T-005 Define validation gates

Status: `Backlog`  
Track: Validation  
Depends on: T-003, T-004  
Owner: Jackson/Codex  
Estimate: 1 day  
Acceptance: Each project gate lists required tests, logs, and stop conditions.

### T-006 Create milestone index

Status: `Backlog`  
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5 day  
Acceptance: M0-M8 are indexed with status and dependency summaries.

## M1: Baseline Lock

### T-101 Confirm Python arm sim as canonical abstract reference

Status: `Backlog`  
Track: Simulation  
Depends on: M0  
Owner: TBD  
Estimate: 1 day  
Acceptance: Native simulator constants, limits, home pose, canvas frame, and known shortcuts are documented.

### T-102 Lock canvas material invariants

Status: `Backlog`  
Track: Painting Model  
Depends on: M0  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Tests confirm thickness, wetness, pigment mass, visible tone, white-on-white coverage, and material coverage behavior.

### T-103 Lock controller boundary

Status: `Backlog`  
Track: Control  
Depends on: T-101  
Owner: TBD  
Estimate: 1 day  
Acceptance: `StrokeAction` remains Cartesian/contact intent and IK remains below policy selection.

### T-104 Run existing baseline tests

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

### T-108 Baseline lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-104, T-105, T-106, T-107  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: M1 is marked locked only if baseline tests pass or failures are documented and judged non-blocking.

## M2: MuJoCo Abstract Clone

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

### T-203 Validate MuJoCo forward kinematics

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

### T-209 M2 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-203, T-204, T-205, T-206, T-208  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: M2 is locked only after XML tests, load/compile status, and viewer issues are accepted or triaged.

## M3: MuJoCo Backend Adapter

### T-301 Define common backend surface

Status: `Backlog`  
Track: Architecture  
Depends on: M1, M2  
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
Depends on: T-301  
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

### T-310 M3 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-305, T-306, T-307, T-308, T-309  
Owner: Jackson  
Estimate: 0.5 day  
Acceptance: M3 is locked only after MuJoCo execution, paint update, backend selection, and known gaps are documented.

## M4: Digital Twin Visualization

### T-401 Reuse current Three.js frontend with backend selection

Status: `Backlog`  
Track: Web  
Depends on: M3  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: The web runtime can display either native or MuJoCo-backed state.

### T-402 Keep `/api/canvas.png` as live texture source

Status: `Backlog`  
Track: Web  
Depends on: T-304, T-401  
Owner: TBD  
Estimate: 0.5-1 day  
Acceptance: The canvas texture updates from the same material-state renderer.

### T-403 Show backend and model versions in UI

Status: `Backlog`  
Track: Web/Ops  
Depends on: T-002, T-401  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: UI state reports backend identity, model version, and calibration version.

### T-404 Compare native and MuJoCo scripted stroke output

Status: `Backlog`  
Track: Validation  
Depends on: T-401, T-402  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Same scripted stroke produces comparable arm path and paint footprint.

## M5: Calibration-Ready Geometry

### T-501 Define `RobotGeometrySpec`

Status: `Backlog`  
Track: Geometry  
Depends on: M2  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Spec covers frames, axes, offsets, hard stops, link lengths, masses, COMs, and brush mount.

### T-502 Define measured-frame naming convention

Status: `Backlog`  
Track: Geometry  
Depends on: T-501  
Owner: TBD  
Estimate: 0.5 day  
Acceptance: Base, yaw, shoulder, roll, elbow, wrist, brush tip, and canvas frames have names and units.

### T-503 Create placeholder calibration manifest

Status: `Backlog`  
Track: Calibration  
Depends on: T-501, T-502  
Owner: TBD  
Estimate: 1 day  
Acceptance: Manifest can record unmeasured, measured, and estimated values separately.

## M6: CAD and Prototype Loop

### T-601 Define CAD revision policy

Status: `Backlog`  
Track: CAD  
Depends on: M5  
Owner: TBD  
Estimate: 0.5-1 day  
Acceptance: CAD exports and revision names can be mapped to geometry specs.

### T-602 Map CAD joints to robot frames

Status: `Backlog`  
Track: CAD/Geometry  
Depends on: T-601  
Owner: TBD  
Estimate: 2-4 days  
Acceptance: CAD axes and offsets correspond to `RobotGeometrySpec` frames.

### T-603 Define prototype build checklist

Status: `Backlog`  
Track: Hardware  
Depends on: T-601  
Owner: TBD  
Estimate: 1 day  
Acceptance: Materials, motors, sensors, stops, wiring, brush mount, and canvas fixture are listed.

## M7: Safety and Hardware Bring-Up

### T-701 Define safety envelope

Status: `Backlog`  
Track: Safety  
Depends on: M5  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Joint, current, force, workspace, watchdog, and non-finite-state limits are external to active inference.

### T-702 Specify emergency stop and recovery

Status: `Backlog`  
Track: Safety  
Depends on: T-701  
Owner: TBD  
Estimate: 1 day  
Acceptance: E-stop, manual recovery, and restart behavior are documented before hardware autonomy.

### T-703 Define dry-run validation

Status: `Backlog`  
Track: Hardware Validation  
Depends on: T-701, T-702  
Owner: TBD  
Estimate: 1-2 days  
Acceptance: Hardware can execute safe dry motion before wet contact is attempted.

## M8: Research Experiments

### T-801 Define baseline experiment manifest

Status: `Backlog`  
Track: Research Ops  
Depends on: M0, M1  
Owner: TBD  
Estimate: 1 day  
Acceptance: Experiment runs record required metadata, traces, telemetry, and canvas outputs.

### T-802 Define core ablations

Status: `Backlog`  
Track: Research  
Depends on: T-801  
Owner: TBD  
Estimate: 2-3 days  
Acceptance: Ablations include no foveation, no hierarchy, Cartesian-only, fixed uncertainty, and no motor alternatives.

### T-803 Define sim-to-real comparison protocol

Status: `Backlog`  
Track: Research/Hardware  
Depends on: M3, M7  
Owner: TBD  
Estimate: 2-4 days  
Acceptance: Protocol compares predictive residuals and policy consequences between sim and hardware.
