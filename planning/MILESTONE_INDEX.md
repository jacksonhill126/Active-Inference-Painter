# Milestone Index

This is the compact index for project milestones and simulation-support
milestones. Detailed task state remains authoritative in
`planning/PROJECT_TRACKER.md`; this file summarizes milestone-level readiness.

Status snapshot: 2026-08-11.

AI-104/AI-105 reference verification is accepted, AI-111 is closed with a
negative proposal-convergence decision, and AI-106 terminal/stopping validation
is closed with a required M2 forecast-family replacement. The posterior remains
conditional on its sampled candidate set and AI-306 must correct finite-
proposal inference before broader claims. M1 remains active until held-out
calibration/learning curves, the composition decision,
inheritance semantics, reproducible replicas, and the AI-115 lock are accepted.
Bounded M2 camera/body, corpus, parallel-training, and conditional patch-cVAE
components were implemented ahead of M1 as explicit provisional experiments;
they do not make M2 active or satisfy their scientific acceptance gates.

AI-108's live-scale leakage-resistant corpus was accepted on 2026-08-11.
AI-107 calibration is also complete, with a negative M2 result: nominal 90%
intervals covered about 99.4% and meaningful dynamic-roll OOD disagreement rose
only 1.087x. The current critical-path task is AI-109 learning curves and
likelihood-family diagnosis.

Current physical-design records are aligned to the fixed RobStride assignment
(03 yaw/pitch, 02 upper-arm roll/elbow) and the selected-but-not-purchased
`provisional-compact-dual-imx296-v1` two-camera rig. Neither plant nor camera
path is hardware calibrated. The 2026-08-11 conditional patch cVAE remains an
offline/shadow material-transition likelihood with no painting-policy
influence. Its 3,000-step AI-107 result did not improve held-out calibration
over the CNN and showed effectively collapsed within-member latent variance.

## Research And Delivery Milestones

| ID | Milestone | Planner | Status | Entry dependency | Exit decision |
| --- | --- | --- | --- | --- | --- |
| M0 | Project Operating System | `M0-operating-system.md` | `Done` | none | all T-001 through T-006 `Done` |
| M1 | Formal Baseline And Inference Audit | `M1-formal-baseline-and-inference-audit.md` | `Active` | M0 | AI-115 |
| M2 | Calibrated Multiscale Generative Model | `M2-calibrated-multiscale-generative-model.md` | `Blocked` | M1 | AI-216 |
| M3 | Foveated Hierarchical Policy Inference | `M3-foveated-hierarchical-policy-inference.md` | `Blocked` | M2 | AI-316 |
| M4 | Experimental Observatory And Digital Twin | `M4-experimental-observatory-and-digital-twin.md` | `Blocked` | T-003, T-103, T-105 | T-412 |
| M5 | Versioned Geometry, Actuation, And Calibration | `M5-versioned-geometry-actuation-and-calibration.md` | `Active` | T-101, T-103 | T-514 |
| M6 | CAD And Prototype Iteration | `M6-cad-and-prototype-iteration.md` | `Blocked` | selected M5 interfaces and measurements | T-615 |
| M7 | Safety And Staged Hardware Bring-Up | `M7-safety-and-staged-hardware-bring-up.md` | `Blocked` | selected M5/M6 hardware artifacts | T-718 |
| M8 | Research Experiment Program | `M8-research-experiment-program.md` | `Blocked` | AI-115 for study registry; study-specific gates thereafter | T-817 |

## Simulation Support Milestones

| ID | Milestone | Planner | Status | Entry dependency | Exit decision |
| --- | --- | --- | --- | --- | --- |
| S0 | Plant Reference Contract | `S0-plant-reference-contract.md` | `Active` | M0 | T-108 |
| S1 | MuJoCo Physical Draft And Logical Retarget | `S1-mujoco-abstract-clone.md` | `Active` | T-101; may overlap later S0 work | T-209 |
| S2 | MuJoCo Backend Adapter | `S2-mujoco-backend-adapter.md` | `Active` | T-103 plus accepted S1 command/retarget tasks | T-310 |

## Dependency Map

The project has a research spine and a supporting embodiment spine. They may
proceed concurrently where their exact task dependencies permit.

```text
M0 -> M1 -> M2 -> M3
 |      |     |     |
 |      +-----+-----+---------------------> M8 research studies
 |
 +-> S0 -> S1 -> S2 ----------------------> M4 backend parity
 |    |                 \                  /
 |    +-----------------> M4 observatory -+
 |    |
 |    +-> M5 -> M6 -> M7 -----------------> M8 hardware transfer
 |              \     ^
 +---------------\----+  M0 safety/gate rules
```

This diagram is deliberately not a claim that every milestone must finish
before the next begins:

- M1 and S0 may begin together after M0.
- S0 evidence T-101 through T-107 is complete. The default fail-closed web/
  telemetry bundle and consolidated shortcut classification were captured on
  2026-08-11; T-108 is ready for Jackson's explicit reference-contract lock.
  T-109 remains a declared native sensor-adapter nonconformance rather than a
  hidden sensor-equivalence claim.
- S1 began from T-101 but deliberately evolved from an exact abstract clone
  into a versioned physical draft with a named logical retarget. T-209 remains
  open until the viewer inspection and contract are accepted.
- S2 live execution and selected-plant counterfactuals are implemented; T-309
  matched-backend evidence and the explicit lock decision remain before T-310.
  M2 now connects the MuJoCo body posterior to forecast q/qvel initialization,
  independent material fields initialize from the spatial posterior, and
  compact brush state initializes from `BrushLoadBelief` plus independent
  microstructure noise. The default/oracle container retains copied substrate
  grain/model context. The opt-in `provisional-sensor-simulation-v0` profile
  now runs a bounded repeated-stroke loop with independent fixed context priors;
  collapsed brush history and missing compliance inference remain explicit.
- M4 contract work may begin when T-003, T-103, and T-105 are complete. Only
  cross-backend parity work waits for S2 outputs.
- M5 and targeted M6 risk-reduction work may proceed alongside M1-M3 when their
  named geometry and interface dependencies are satisfied.
- M8 infrastructure starts after M1, but each study is released by its own
  capability gates. Hardware-transfer studies additionally require M7 evidence.

## Updating This Index

Update the status snapshot when a milestone enters `Active`, reaches its exit
decision, or becomes blocked for a materially different reason. A summary
status never overrides a task status in `PROJECT_TRACKER.md`.
