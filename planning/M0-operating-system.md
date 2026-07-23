# M0: Project Operating System

## Summary

M0 establishes the project-management layer used by the rest of the robotics
program. It defines how tasks, versions, experiment manifests, failure logs, and
validation gates are recorded.

This milestone is a planning and operations milestone. It should not modify
runtime behavior.

## New Planning Interfaces

- Task IDs: `T-###`
- Milestone IDs: `M#`
- Statuses: `Backlog`, `Ready`, `Active`, `Blocked`, `Validate`, `Done`
- Artifact types: task tracker, experiment manifest, version manifest, failure log

## Tasks

### T-001 Define tracker conventions

Status: `Ready`  
Track: Operations  
Depends on: none  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- Document task fields: ID, title, status, track, dependencies, owner, estimate, acceptance, notes.
- Document status meanings.
- Document dependency rules.
- Document how completed tasks are validated.

Notes:

- The first implementation lives in `planning/PROJECT_TRACKER.md`.

### T-002 Define versioning scheme

Status: `Ready`  
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5-1 day

Acceptance:

- Define labels for code, MuJoCo XML, CAD, calibration, hardware build, and experiment config.
- Define how versions are recorded in run outputs.
- Include examples such as `mujoco-abstract-v0`, `cad-proto-a`, `calib-none`, and `hardware-none`.

### T-003 Define experiment manifest requirements

Status: `Ready`  
Track: Research Ops  
Depends on: T-001, T-002  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- List required run metadata: code commit, config, random seeds, backend, model version, calibration version, planner mode, canvas size, and output paths.
- List required traces: VFE, EFE, policy posterior, telemetry, canvas snapshots, and failure notes.
- State that manifests must distinguish active-inference terms from conventional support engineering.

### T-004 Define failure-mode log

Status: `Ready`  
Track: Validation  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Define categories: kinematics, contact, paint/material, planner, uncertainty, visualization, MuJoCo, CAD, hardware, and safety.
- Define fields: symptom, reproduction, suspected cause, severity, mitigation, status, and linked task.
- Include the rule that failures are preserved as evidence unless they are irrelevant environment noise.

### T-005 Define validation gates

Status: `Backlog`  
Track: Validation  
Depends on: T-003, T-004  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Define gates for baseline sim, MuJoCo clone, MuJoCo backend, digital twin, calibrated geometry, hardware dry run, hardware wet run, and research-grade experiment.
- Each gate has required tests, logs, and stop conditions.
- Safety gates remain external to active-inference policy selection.

### T-006 Create milestone index

Status: `Backlog`  
Track: Operations  
Depends on: T-001  
Owner: Jackson/Codex  
Estimate: 0.5 day

Acceptance:

- List all milestone planner files to be created later.
- Include status summary for M0-M8.
- Include dependency map from M0 through hardware research runs.

## Validation

- Every M0 task has status, dependency, estimate, and acceptance criteria.
- The operating rules do not introduce rewards or heuristic aesthetics into the active-inference layer.
- Future milestone files can reference M0 conventions without redefining them.

## Defaults

- `planning/` is the canonical folder for planning artifacts.
- `PROJECT_TRACKER.md` is the current source of task status.
- The Python simulator remains canonical until a measured physical model exists.

