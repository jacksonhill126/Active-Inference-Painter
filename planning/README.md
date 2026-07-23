# Active-Inference Painter Planning

This folder contains project-management artifacts for the robotics program.
The research charter remains the source of scientific intent; these files turn
that intent into milestones, dependencies, validation gates, and execution
tasks.

## Planning Files

- `PROJECT_TRACKER.md`: milestone task tracker and initial backlog.
- `GANTT.md`: rough schedule across research, simulation, CAD, hardware, and validation.
- `M0-operating-system.md`: detailed plan for tracker conventions, manifests, versioning, and failure logs.
- `M1-baseline-lock.md`: detailed plan for locking the Python simulator, material model, controller boundary, and baseline validation.
- `M2-mujoco-abstract-clone.md`: detailed plan for matching the Python arm simulator in MuJoCo.
- `M3-mujoco-backend-adapter.md`: detailed plan for connecting the existing controller and paint model to MuJoCo.

## Project Map

### 1. Research Architecture

Keep painting cognition grounded in active inference. Painting-level decisions
must be traceable to likelihoods, priors, precision beliefs, VFE, EFE, and
policy posteriors. Ordinary control, IK, safety, CAD, and hardware work live
below the selected painting policy.

### 2. Simulation Stack

Use the Python arm simulator as the canonical abstract baseline. Build MuJoCo
first as an abstract clone of that simulator, then later split the simulator
family into an abstract clone, a calibrated digital twin, and a hardware safety
twin.

### 3. Painting and Material Model

Keep paint deposition in the project material model, not in MuJoCo physics.
MuJoCo supplies motion, site positions, contact, and actuator state. The
existing canvas model supplies thickness, wetness, pigment mass, surface tone,
and material coverage.

### 4. Robot Geometry and CAD

Treat current geometry as provisional until measured. Physical offsets, motor
orientations, hard stops, masses, centers of mass, and brush mounts should enter
through versioned geometry/calibration specs rather than controller hacks.

### 5. Control, Safety, and Execution

Keep `StrokeAction` as Cartesian/contact intent. IK, trajectories, servo
control, collision checks, current limits, force limits, workspace limits, and
watchdogs realize selected policies below the active-inference boundary.

### 6. Testing and Validation

Validate low-level capabilities before interpreting painting behavior: geometry
constants, kinematics, contact, paint deposition, telemetry, predictive
calibration, policy sensitivity, safety gates, and sim-to-real residuals.

### 7. Hardware Development

Bring hardware up incrementally: one joint, two-link chain, brush contact rig,
full-arm dry motion, full-arm wet painting, then autonomous research runs.

### 8. Versioning and Operations

Record code version, MuJoCo model version, CAD version, calibration version,
hardware build, experiment config, random seeds, telemetry, and output artifacts
for every meaningful run.

## Milestone Sequence

| Milestone | Name | Purpose |
| --- | --- | --- |
| M0 | Project Operating System | Tracker conventions, manifests, versioning, failure logs, validation gates. |
| M1 | Baseline Lock | Stabilize Python sim, planner tests, canvas model, telemetry, and web view. |
| M2 | MuJoCo Abstract Clone | Match the current abstract arm kinematics, joint ranges, canvas frame, and brush site. |
| M3 | MuJoCo Backend | Drive MuJoCo through the existing controller and feed brush contact into the paint model. |
| M4 | Digital Twin Visualization | Reuse the web frontend for MuJoCo arm state plus live painted canvas texture. |
| M5 | Calibration-Ready Geometry | Define measured frames, offsets, hard stops, inertias, and calibration manifests. |
| M6 | CAD and Prototype Loop | Connect mechanical design revisions to geometry specs and simulation models. |
| M7 | Safety and Hardware Bring-Up | Establish safety envelope, watchdogs, dry tests, wet tests, and failure response. |
| M8 | Research Experiments | Run controlled ablations, sim-to-real comparisons, and documented research runs. |
