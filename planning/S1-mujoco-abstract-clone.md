# S1: MuJoCo Physical Draft And Logical Retarget

## Summary

S1 originally planned an exact kinematic clone of the co-located native arm.
That path was deliberately superseded on 2026-07-28 by a versioned,
hardware-oriented MuJoCo draft with separated shoulder anchors, RobStride
direct drives, physical canvas contact, and a half-inch brush. The existing
painting controller retains the native logical coordinate convention through a
named, tested retarget at the backend boundary.

This is still not a calibrated hardware twin. Datasheet-derived, estimated,
visual-only, and unmeasured fields remain explicit. Counterfactual forecasts
now preserve the selected plant: MuJoCo execution predicts through independent
`MjData` under the same immutable model, and its runtime q/qvel initialization
comes from a provisional sensor-conditioned body posterior. Independent
material fields initialize from the frozen spatial posterior. Copied substrate
grain/model context, collapsed held-paint/persistent-bristle history,
contact-to-compliance initialization, and MuJoCo parameter uncertainty remain
declared forecast approximations. Compact brush load/pigment and independent
bristle-scale prior noise now have their own declared forecast boundary.

## Physical-Draft And Retarget Contracts

- Logical joint order remains `yaw`, `pitch`, `roll`, `elbow`.
- Joint axes, signs, command ranges, physical anchors, and keyframes are
  explicit in SI/radians.
- The separated pitch/roll/elbow geometry is intentional and must not be
  corrected by hidden controller offsets.
- The logical canvas frame is mapped to the physical canvas through the named
  retarget tested by the frontend/backend adapters.
- The `tip` site and `bristle_contact` geom define the brush/contact boundary.
- Visual-only, collision-relevant, and contact geometry have explicit roles.
- The version label must distinguish this physical draft from
  `native-abstract-v0` and any future calibrated hardware revision.

## Tasks

### T-201 Define the versioned physical MuJoCo draft and logical command subset

Status: `Done`
Track: MuJoCo  
Depends on: T-101  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- XML defines the four-joint logical command order, explicit axes/signs/ranges,
  separated physical anchors, link geometry, canvas frame, brush contact
  reference, and safe keyframes.
- MuJoCo angle units are explicit.
- Intentional differences from the native logical body are named.

Notes:

- Accepted 2026-07-28 in `models/active_inference_painter.xml` and
  `models/README.md` as `mujoco-robstride-electromechanical-v4`.
- The native command subset is preserved through a named physical retarget
  instead of hidden controller offsets.

### T-202 Add XML geometry, range, actuator, and contact tests

Status: `Done`
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Tests protect joint order/axes/ranges, separated anchors, keyframes, canvas
  dimensions, direct-drive actuator limits, brush geometry/compliance,
  friction, and stable adapter names.

Notes:

- Accepted 2026-07-28 in `tests/test_mujoco_model.py`; the focused XML/model
  suite includes optional MuJoCo compile coverage.

### T-203 Validate physical kinematics and logical retarget transforms

Status: `Done`
Track: MuJoCo  
Depends on: T-201, T-202  
Owner: Jackson/Codex
Estimate: 1-2 days

Acceptance:

- Representative poses verify joint signs, offset kinematics, brush-tip
  transforms, canvas reach, and the logical-canvas-to-physical-target retarget
  within declared tolerances.
- The separated physical shoulder is explicitly distinguished from the
  co-located native abstraction.

Notes:

- Accepted 2026-07-28 through joint-sign/offset-kinematics, reach/keyframe, and
  legacy-canvas retarget tests.

### T-204 Separate visual, collision, and contact geometry

Status: `Done`
Track: MuJoCo  
Depends on: T-201  
Owner: Jackson/Codex
Estimate: 0.5 day

Acceptance:

- Decorative housings remain non-colliding.
- Collision-relevant links and the brush/canvas contact pair are explicit.
- Documentation identifies visual, collision, and contact roles.

Notes:

- Accepted 2026-07-28 in the MJCF collision masks and `models/README.md`;
  compile/contact tests exercise the intended collision pair.

### T-205 Document exact versus approximate model fields

Status: `Done`
Track: Documentation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Model docs list exact simulator-clone fields.
- Model docs list approximate visual fields.
- Model docs list first measurements required for a calibrated twin.

Notes:

- Accepted 2026-07-28 in `models/README.md`, including provisional
  datasheet-derived values, lumped brush compliance, Python-owned paint, native
  counterfactual forecasts, and first calibration measurements.

### T-206 Add MuJoCo load/compile smoke test

Status: `Done`
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex
Estimate: 0.5-1 day

Acceptance:

- If the optional `mujoco` package is installed, a test loads the XML with
  `mujoco.MjModel.from_xml_path`.
- If MuJoCo is not installed, the test skips cleanly with an explicit reason.
- The smoke test confirms joint and actuator counts.

Notes:

- Accepted 2026-07-28 in `tests/test_mujoco_model.py`; the module skips
  explicitly when MuJoCo is unavailable.

### T-207 Define model version label

Status: `Done`
Track: Operations  
Depends on: T-002, T-201  
Owner: Jackson/Codex
Estimate: 0.5 day

Acceptance:

- The model has a runtime-visible version label.
- The label distinguishes the physical draft from the native abstract and
  future calibrated hardware revisions.

Notes:

- Accepted 2026-07-28 as `mujoco-robstride-electromechanical-v4` in MJCF text
  metadata, backend identity, runtime state, and documentation.

### T-208 Compare model behavior in MuJoCo viewer

Status: `Active`
Track: Manual Validation  
Depends on: T-201, T-206  
Owner: Jackson  
Estimate: 0.5-1 day

Acceptance:

- Manual load command is documented.
- Joint sliders move through expected command ranges.
- Brush tip and canvas are visually aligned enough for backend integration.
- Any viewer discrepancy becomes a failure-mode entry or a follow-up task.

Suggested command:

```powershell
simulate "C:\Users\jxnhi\Documents\Active Inference Painter\models\active_inference_painter.xml"
```

Notes:

- The XML-driven Three.js frontend has been inspected through home,
  canvas-facing, top, contact, lower-arm-down, and canvas-edge views.
- A dated standalone MuJoCo viewer inspection record and any discrepancy
  entries remain before acceptance.

### T-209 S1 lock decision

Status: `Blocked`
Track: Validation  
Depends on: T-203, T-204, T-205, T-206, T-208  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- S1 is locked only if XML tests pass, MuJoCo load succeeds or is explicitly
  blocked by missing optional dependency, and manual viewer issues are triaged.
- S2 backend work may begin only after the tip-site coordinate contract is
  accepted.

Notes:

- Blocked only on T-208 manual acceptance and Jackson's explicit
  physical-draft/retarget contract decision.

## Validation Gate

S1 is complete when:

- MuJoCo physical geometry, command subset, and logical retarget are explicit.
- Physical kinematics and retarget transforms are tested for representative
  poses.
- The model loads in MuJoCo or has a documented dependency blocker.
- Geometry roles cannot create undeclared collision/contact behavior.
- The model is versioned as an uncalibrated physical draft, not a hardware
  twin.

## Failure Modes To Watch

- MuJoCo slider units or actuator ranges differ from joint ranges.
- Pitch/roll/elbow signs or offsets diverge from the declared retarget.
- Decorative base or floor geometry creates false physical constraints.
- Canvas frame is shifted enough that IK appears broken when the model is
  actually misaligned.
- Viewer aesthetics get mistaken for measured robot geometry.

## S1 Output Artifacts

- MuJoCo abstract XML model.
- XML constant and optional compile tests.
- Model documentation for exact versus approximate fields.
- Forward-kinematics parity result.
- S1 lock note or blocker list.
