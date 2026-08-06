# Active-Inference Painter Planning

This folder contains project-management artifacts for the active-inference
research program and its supporting robotics work. The research charter remains
the source of scientific intent; these files turn that intent into milestones,
dependencies, validation gates, and execution tasks.

## Planning Files

- `PROJECT_TRACKER.md`: milestone task tracker and initial backlog.
- `GANTT.md`: rough schedule across research, simulation, CAD, hardware, and validation.
- `M0-operating-system.md`: detailed plan for tracker conventions, manifests, versioning, and failure logs.
- `VALIDATION_GATES.md`: evidence, pass conditions, and stop conditions for simulation, inference, geometry, hardware, and research gates.
- `variable-sensor-access-ledger.json`: machine-readable observation-boundary inventory.
- `VERSIONING.md`: artifact labels, exact identities, revision rules, and run recording contract.
- `EXPERIMENT_MANIFEST.md`: run provenance, sensor-access, active-inference boundary, and required artifact contract.
- `FAILURE_LOG.md`: append-only failure evidence, categories, severity, reproduction, and resolution contract.
- `MILESTONE_INDEX.md`: milestone status, entry dependencies, exit decisions, and cross-track dependency map.
- `templates/`: machine-readable examples for planning and run artifacts.
- `M1-formal-baseline-and-inference-audit.md`: detailed plan for formalizing and validating the current VFE/EFE baseline.
- `M2-calibrated-multiscale-generative-model.md`: detailed plan for sensor-equivalent observation, calibrated dynamics, and uncertain multiscale beliefs.
- `M3-foveated-hierarchical-policy-inference.md`: detailed plan for active gaze and temporally hierarchical policy inference.
- `M4-experimental-observatory-and-digital-twin.md`: detailed plan for separating truth, observation, belief, prediction, policy, execution, provenance, and synchronized artifacts across backends.
- `M5-versioned-geometry-actuation-and-calibration.md`: detailed plan for physical schemas, frame graphs, actuation choices, uncertainty, sizing, and calibration readiness.
- `M6-cad-and-prototype-iteration.md`: detailed plan for parametric CAD, risk-reducing rigs, fabrication, measurement, and mechanical revision.
- `M7-safety-and-staged-hardware-bring-up.md`: detailed plan for external safety, staged commissioning, sensors, wet trials, shadow inference, and bounded autonomy.
- `M8-research-experiment-program.md`: detailed plan for capability-gated H1-H6 studies, developmental runs, interactions, transfer, and research synthesis.
- `S0-plant-reference-contract.md`: support plan for versioning the Python plant, material, command, observation, control-boundary, and telemetry reference.
- `S1-mujoco-abstract-clone.md`: historical filename for the superseding hardware-oriented MuJoCo physical draft and tested logical retarget plan.
- `S2-mujoco-backend-adapter.md`: support plan for the implemented selectable MuJoCo execution backend and Python paint-process connection.

## Project Map

### 1. Research Architecture

Keep painting cognition grounded in active inference. Painting-level decisions
must be traceable to likelihoods, priors, precision beliefs, VFE, EFE, and
policy posteriors. Ordinary control, IK, safety, CAD, and hardware work live
below the selected painting policy.

### 2. Inference Validity

Formalize the generative model, sensor-access boundary, variational family,
preferences, policy priors, proposal distributions, and approximations before
interpreting output. Use tractable fixtures, held-out likelihoods, calibration,
and matched ablations as capability gates.

### 3. Perception And Temporal Hierarchy

Replace privileged material observations with an explicit observation
likelihood. Preserve uncertainty across pixel, canvas, relational, passage, and
painting timescales. Introduce foveation only after the fixed-view model passes
predictive and calibration gates.

### 4. Simulation Support

Use the Python arm simulator as the versioned `native-abstract-v0` reference.
The original exact-clone plan was superseded by the hardware-oriented
`mujoco-robstride-electromechanical-v4` physical draft, a tested logical
controller retarget, and a selectable execution backend in the `S0-S2` support
track. The draft fixes RobStride 03 at `yaw`/`pitch` and RobStride 02 at
`roll`/`elbow`; it is not yet a calibrated hardware twin. Counterfactual motor
forecasts now preserve the selected plant, including independent MuJoCo rollout
state when MuJoCo is selected. Exact-process initialization remains an explicit
`baseline-oracle-v0` limitation until the body posterior is connected. S0
stabilizes schemas and provenance, not controller gains or mechanical design.
Simulator migration improves the generative process and transfer testing; it
does not by itself validate active inference.

### 5. Painting And Material Model

Keep paint deposition in the project material model, not in MuJoCo physics.
MuJoCo supplies motion, site positions, contact, and actuator state. The
existing canvas model supplies thickness, wetness, pigment mass, surface tone,
and material coverage.

### 6. Robot Geometry And CAD

Treat current geometry as provisional until measured. Physical offsets, motor
orientations, hard stops, masses, centers of mass, and brush mounts should enter
through versioned geometry/calibration specs rather than controller hacks.

### 7. Control, Safety, And Execution

Keep `StrokeAction` as Cartesian/contact intent. IK, trajectories, servo
control, collision checks, current limits, force limits, workspace limits, and
watchdogs realize selected policies below the active-inference boundary.

### 8. Testing And Validation

Validate low-level capabilities before interpreting painting behavior: geometry
constants, kinematics, contact, paint deposition, telemetry, predictive
calibration, policy sensitivity, safety gates, and sim-to-real residuals.

### 9. Experimental Observability

Keep hidden process truth, agent-accessible observation, beliefs, predictions,
policies, controller targets, and realized consequences distinguishable.
Version clocks and revisions so viewer and analysis artifacts can be traced to
the exact runtime state that produced them.

### 10. Hardware Development

Bring hardware up incrementally: one joint, two-link chain, brush contact rig,
full-arm dry motion, full-arm wet painting, then autonomous research runs.

### 11. Versioning And Operations

Record code version, MuJoCo model version, CAD version, calibration version,
hardware build, experiment config, random seeds, telemetry, and output artifacts
for every meaningful run.

## Milestone Sequence

| Milestone | Name | Purpose |
| --- | --- | --- |
| M0 | Project Operating System | Tracker conventions, manifests, versioning, failure logs, validation gates. |
| M1 | Formal Baseline And Inference Audit | Specify the generative model, validate VFE/EFE, audit sensor access, and establish held-out predictive and calibration baselines. |
| M2 | Calibrated Multiscale Generative Model | Infer hidden material and relational state from sensor-equivalent observations with tested uncertainty and temporal hierarchy. |
| M3 | Foveated Hierarchical Policy Inference | Infer gaze, mark, passage, motor, and stopping policies across timescales and test the core research hypotheses. |
| M4 | Experimental Observatory And Digital Twin | Provide backend-neutral, revision-synchronized truth, observation, belief, prediction, execution, provenance, and artifact views. |
| M5 | Versioned Geometry, Actuation, And Calibration | Define physical schemas, uncertainty, work envelopes, actuator choices, sizing, calibration, and exports without freezing design. |
| M6 | CAD And Prototype Iteration | Use parametric CAD and targeted joint/contact rigs to retire fabrication risk and feed measurements back into all models. |
| M7 | Safety And Staged Hardware Bring-Up | Keep safety external while commissioning one joint, linked motion, dry arm, contact, sensors, wet paint, shadow inference, and bounded autonomy. |
| M8 | Research Experiment Program | Run capability-gated H1-H6, developmental, interaction, transfer, and qualitative studies with valid independent units and no aesthetic objective. |

## Simulation Support Sequence

| Support milestone | Name | Purpose |
| --- | --- | --- |
| S0 | Plant Reference Contract | Version the Python plant, material model, command/observation schemas, controller boundary, telemetry, and web reference without freezing implementation. |
| S1 | MuJoCo Physical Draft And Logical Retarget | Version the physical draft, preserve the logical command subset, and test the canvas/brush retarget. |
| S2 | MuJoCo Backend Adapter | Drive MuJoCo through the existing controller and feed contact into the material model. |

Support milestones may proceed when their dependencies are satisfied, but they
must not displace blockers in the active-inference capability gates merely
because simulator work is easier to demonstrate.

## Concurrency Rules

- M1 and S0 begin together. Formal VFE/EFE and policy work does not wait for
  plant completion.
- Corpus and fixed-camera work depend on versioned plant/material schemas, not
  on completion of every S0 task.
- S1/S2, M4, and preliminary M5/M6 work may proceed alongside M1-M3.
- Final parity, calibration, transfer, and hardware gates wait only on the
  specific artifacts they test.
- Interfaces are stabilized early; controllers, models, geometry, and CAD
  remain replaceable and versioned.
