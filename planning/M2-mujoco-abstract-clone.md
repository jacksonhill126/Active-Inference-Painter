# M2: MuJoCo Abstract Clone

## Summary

M2 makes MuJoCo match the current Python arm simulator at the abstract
kinematic and canvas-frame level. This milestone is not a calibrated hardware
model. It exists so the controller can later swap plants without changing the
painting policy.

The Python simulator remains authoritative in M2. MuJoCo must match it before
physical motor offsets, joint housings, hard stops, CAD geometry, or hardware
constraints are introduced.

## Clone Contracts

- Joint order is `yaw`, `pitch`, `roll`, `elbow`.
- Joint ranges match `ArmPose.clipped()`.
- `safe_home` matches `safe_home_pose()`.
- Link lengths match `ArmKinematics`, converted from inches to meters.
- Canvas size and contact plane match `VerticalCanvas`.
- The `tip` site is the brush contact reference for later backend work.
- Base, floor, joint housings, link radii, and brush handle are visual
  placeholders unless explicitly measured.
- Abstract clone geometry must not introduce accidental physical constraints
  that the Python simulator does not have.

## Tasks

### T-201 Match native arm constants in MuJoCo XML

Status: `Active`  
Track: MuJoCo  
Depends on: T-101  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- XML defines the native joint order, axes, ranges, link lengths, canvas frame,
  brush tip site, and `safe_home` keyframe.
- MuJoCo angle units are explicit.
- Position actuators inherit joint ranges or otherwise remain synchronized with
  joint stops.

### T-202 Add XML constant tests

Status: `Validate`  
Track: Validation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Tests compare XML joint constants against `ArmPose.clipped()`.
- Tests compare XML link lengths against `ArmKinematics`.
- Tests compare canvas dimensions and contact plane against `VerticalCanvas`.
- Tests verify actuator ranges stay tied to joint ranges.

### T-203 Validate MuJoCo forward kinematics against native kinematics

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-201, T-202  
Owner: TBD  
Estimate: 1-2 days

Acceptance:

- A representative pose set includes home, straight, near-canvas, roll-positive,
  roll-negative, and elbow-bent configurations.
- For each pose, MuJoCo `tip` site position matches
  `ArmKinematics.tip(ArmPose(...))` within a declared tolerance.
- Any coordinate transform is named and tested rather than corrected ad hoc in
  controller code.

### T-204 Keep physical housings visual-only

Status: `Backlog`  
Track: MuJoCo  
Depends on: T-201  
Owner: TBD  
Estimate: 0.5 day

Acceptance:

- Base, floor, joint marker, and decorative link geoms do not constrain the
  abstract clone.
- Canvas contact remains available for later pressure/contact tests.
- The model documentation states which geoms are collision-relevant.

### T-205 Document exact versus approximate model fields

Status: `Validate`  
Track: Documentation  
Depends on: T-201  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Model docs list exact simulator-clone fields.
- Model docs list approximate visual fields.
- Model docs list first measurements required for a calibrated twin.

### T-206 Add MuJoCo load/compile smoke test

Status: `Backlog`  
Track: Validation  
Depends on: T-201  
Owner: TBD  
Estimate: 0.5-1 day

Acceptance:

- If the optional `mujoco` package is installed, a test loads the XML with
  `mujoco.MjModel.from_xml_path`.
- If MuJoCo is not installed, the test skips cleanly with an explicit reason.
- The smoke test confirms joint and actuator counts.

### T-207 Define model version label

Status: `Backlog`  
Track: Operations  
Depends on: T-002, T-201  
Owner: TBD  
Estimate: 0.5 day

Acceptance:

- Abstract model version is named, initially `mujoco-abstract-v0`.
- Version label appears in model docs and is available for future runtime state.
- The label is distinct from any future calibrated hardware model version.

### T-208 Compare model behavior in MuJoCo viewer

Status: `Backlog`  
Track: Manual Validation  
Depends on: T-201, T-206  
Owner: Jackson  
Estimate: 0.5-1 day

Acceptance:

- Manual load command is documented.
- Joint sliders move through expected abstract ranges.
- Brush tip and canvas are visually aligned enough for backend integration.
- Any viewer discrepancy becomes a failure-mode entry or a follow-up task.

Suggested command:

```powershell
simulate "C:\Users\jxnhi\Documents\Active Inference Painter\models\active_inference_painter.xml"
```

### T-209 M2 lock decision

Status: `Backlog`  
Track: Validation  
Depends on: T-203, T-204, T-205, T-206, T-208  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- M2 is locked only if XML tests pass, MuJoCo load succeeds or is explicitly
  blocked by missing optional dependency, and manual viewer issues are triaged.
- M3 backend work may begin only after the tip-site coordinate contract is
  accepted.

## Validation Gate

M2 is complete when:

- MuJoCo XML constants match the Python simulator.
- Forward kinematics parity is tested for representative poses.
- The model loads in MuJoCo or has a documented dependency blocker.
- Visual-only geometry cannot distort the abstract reference behavior.
- The model is versioned as an abstract clone, not a hardware twin.

## Failure Modes To Watch

- MuJoCo slider units or actuator ranges differ from joint ranges.
- The pitch/roll/elbow sign convention differs from the Python simulator.
- Decorative base or floor geometry creates false physical constraints.
- Canvas frame is shifted enough that IK appears broken when the model is
  actually misaligned.
- Viewer aesthetics get mistaken for measured robot geometry.

## M2 Output Artifacts

- MuJoCo abstract XML model.
- XML constant and optional compile tests.
- Model documentation for exact versus approximate fields.
- Forward-kinematics parity result.
- M2 lock note or blocker list.

