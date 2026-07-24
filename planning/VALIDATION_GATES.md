# Validation Gates

Version: `validation-gates-v0`

Status: accepted M0 operating contract as of 2026-07-24.

This document defines the evidence required to advance the project through
simulation, inference, geometry, and hardware stages. A gate is a decision
about what the available evidence supports. It is not a claim that the system
is finished, safe in every setting, or biologically equivalent to a human.

## General Gate Rules

Every gate review must:

1. identify the exact code, configuration, model, geometry, calibration,
   hardware, and data versions under review;
2. use an experiment manifest when a runtime result is part of the evidence;
3. preserve failed runs in the failure log unless they are documented as
   irrelevant environment failures;
4. list all required evidence as present, failed, waived, or not applicable;
5. state the operating mode or research claim that is permitted after review;
6. identify residual risks and the next invalidating observation;
7. stop rather than silently substitute simulator truth, missing telemetry,
   unversioned parameters, or hand-selected successful runs.

Automated tests may prepare a gate, but milestone, safety, hardware, and
research-claim gates require Jackson's explicit acceptance. A waiver must name
the missing evidence, why it is non-blocking for the limited decision, and the
task that will close it.

Safety constraints remain external to active-inference painting-policy
selection. No EFE term, preference precision, posterior confidence, or learned
model output can waive a hard safety stop.

## Evidence States

| State | Meaning |
| --- | --- |
| Present | Required artifact exists, is versioned, and passed its declared check. |
| Failed | Evidence was collected and did not meet the criterion. |
| Waived | Named reviewer accepted a limited exception with rationale and follow-up task. |
| Not applicable | The item does not apply to the reviewed operating mode, with rationale. |
| Missing | Required evidence does not exist; the gate cannot pass. |

## G0: Native Baseline Reference

Decision task: `T-108`

Permits:

- treating one named Python plant/material configuration as the reproducible
  abstract reference;
- comparing later MuJoCo and hardware backends against that reference;
- beginning plant-dependent M1 calibration work.

Required tests:

- native arm constant and forward-kinematics contract tests;
- material thickness, wetness, pigment, tone, and coverage invariant tests;
- controller-boundary tests showing that painting policy intent remains above
  IK and motor execution;
- deterministic smoke suite;
- complete integration suite, or a versioned split run with every failure
  entered in the failure log.

Required artifacts and logs:

- native plant reference document and version;
- controller/plant/policy interface document;
- test command, environment, counts, duration, failures, and skips;
- baseline telemetry and web-runtime endpoint capture;
- baseline canvas artifact;
- known-shortcut classification;
- baseline artifact manifest.

Pass conditions:

- constants and interfaces are internally consistent;
- material invariants pass;
- failures are either fixed or explicitly accepted as non-blocking for an
  abstract reference;
- a clean checkout can reproduce the declared smoke baseline.

Stop conditions:

- unexplained nondeterministic failures;
- coverage depends on visible tone rather than material presence;
- simulator-only state is represented as a physical observation;
- controller code is choosing painting content;
- missing revision or configuration identity.

## G1: Formal Active-Inference Baseline

Decision task: `AI-115`

Permits:

- interpreting VFE and EFE diagnostics within the explicitly declared
  approximation boundary;
- beginning the M2 sensor-equivalent model;
- publishing `baseline-v0`.

Required tests:

- analytic linear-Gaussian posterior and VFE fixtures;
- enumerated or Monte Carlo EFE reference fixtures;
- policy posterior normalization and sign tests;
- immediate-stop and terminal-only coverage tests;
- held-out transition likelihood and calibration tests;
- proposal-budget convergence tests;
- checkpoint inheritance and reset-semantics tests.

Required artifacts and logs:

- versioned generative-model factorization and variational family;
- variable and sensor-access ledger;
- VFE and EFE decomposition map;
- approximation register;
- held-out predictive/calibration report;
- terminal preference and composition-preference decisions;
- proposal-distribution versus policy-prior decision;
- online-learning and inheritance contract;
- at least three fixed-seed baseline replicas with manifests.

Pass conditions:

- every decision-relevant quantity maps to a likelihood, transition prior,
  prior preference, precision belief, VFE term, EFE term, or policy prior;
- ordinary proposal bias, controller behavior, and safety constraints are not
  mislabeled as active inference;
- VFE and EFE reference fixtures agree within declared tolerances;
- uncertainty is at least calibrated enough to support the limited baseline
  claims.

Stop conditions:

- unexplained sign or unit dependence in VFE/EFE;
- privileged process state affects a claimed sensor-only result;
- a self-trained preference is evaluated on its own training sample without an
  accepted safeguard;
- finite candidate frequency is reported as a normalized policy prior without
  proposal correction;
- failed calibration is hidden by precision tuning.

## G2: MuJoCo Abstract Clone

Decision task: `T-209`

Permits:

- using the named MuJoCo model as an abstract kinematic clone;
- beginning backend integration;
- making coordinate and kinematic parity claims only.

Required tests:

- MJCF compile/load test;
- exact joint order, axes, ranges, units, link lengths, home pose, canvas frame,
  and tip-site tests;
- representative native-versus-MuJoCo forward-kinematics parity test;
- collision-role test or inspection showing visual placeholders do not create
  unintended constraints.

Required artifacts and logs:

- versioned MJCF and SHA-256;
- field table marking exact, approximate, visual-only, and unmeasured values;
- viewer inspection record;
- parity results with tolerances and failure entries.

Pass conditions:

- declared constants match the native reference;
- tip-site parity passes at home, contact, roll, elbow-bent, and near-limit
  poses;
- model loads in the supported MuJoCo version;
- no physical-calibration claim is made.

Stop conditions:

- coordinate corrections are hidden in controller code;
- actuator or joint ranges diverge silently;
- decorative geometry changes reachable motion;
- the model is described as a calibrated twin.

## G3: MuJoCo Backend

Decision task: `T-310`

Permits:

- executing the existing painting/controller stack against MuJoCo;
- using MuJoCo as a generative-process backend;
- beginning matched backend experiments.

Required tests:

- backend interface conformance;
- command and observation unit conversion;
- scripted free-space and contact strokes;
- canvas deposition driven by MuJoCo tip/contact state;
- reset, copy, replay, and deterministic-seed behavior;
- native-versus-MuJoCo matched-script comparison.

Required artifacts and logs:

- backend capability declaration;
- synchronized command, state, contact, telemetry, and canvas traces;
- exact backend/model versions;
- parity residual report.

Pass conditions:

- frontend and telemetry schemas remain explicit and versioned;
- paint deposition remains owned by the material model;
- backend differences appear as measured residuals rather than hidden
  compensations;
- policy semantics do not change with backend selection.

Stop conditions:

- backend state bypasses the declared observation interface;
- MuJoCo selects painting policies;
- unavailable fields are silently zeroed and interpreted as measurements;
- reset or replay cannot reproduce a scripted case.

## G4: Experimental Observatory And Digital Twin

Decision task: `T-412`

Permits:

- synchronized comparison of process truth, observations, beliefs,
  predictions, policies, controller targets, execution, and artifacts;
- replay-based diagnosis across supported backends.

Required tests:

- clock and revision synchronization;
- process-truth versus agent-observation separation;
- artifact replay and schema migration;
- viewer provenance and stale-state detection;
- native/MuJoCo matched-run observability.

Required artifacts and logs:

- clock-domain and revision contract;
- backend-neutral event schema;
- replay bundle with exact provenance;
- known observability gaps.

Pass conditions:

- every displayed or analyzed quantity names its source and timestamp;
- hidden truth cannot enter inference through the observatory;
- a recorded run can regenerate the relevant diagnostic views.

Stop conditions:

- truth and belief use the same unlabeled field;
- artifacts cannot be tied to one code/model/configuration revision;
- UI state is used as scientific evidence without recorded backing data.

## G5: Calibration-Ready Geometry And Actuation

Decision task: `T-514`

Permits:

- selecting a reversible physical architecture;
- beginning detailed CAD and risk-reduction prototypes;
- using provisional uncertainty bounds for simulation and sizing.

Required tests and analyses:

- frame-graph consistency;
- workspace, singularity, collision, and joint-range analysis;
- actuator torque, speed, thermal, current, and inertia sizing;
- structural load and stiffness estimates;
- sensitivity analysis over uncertain geometry and dynamics;
- calibration observability analysis.

Required artifacts:

- versioned geometry, actuator, transmission, sensor, tool, and workcell
  schemas;
- uncertainty ranges and provenance for every provisional parameter;
- alternatives, margins, cost, and risk register;
- export contract for CAD, MuJoCo, calibration, and hardware.

Pass conditions:

- no placeholder is mistaken for a measurement;
- selected architecture has documented margins and alternatives;
- the planned sensors can identify the parameters needed for control and
  inference;
- high-risk assumptions have prototype tests.

Stop conditions:

- actuator selection lacks thermal/current margin;
- frame or sign conventions are unresolved;
- geometry is frozen before high-risk interfaces can be measured;
- safety-critical behavior depends on the painting agent.

## G6: Hardware Dry Run

Decision tasks: `T-709` for full-arm dry commissioning and `T-718` for the
eventual bounded operating-mode decision.

Permits:

- the specifically reviewed powered dry-motion mode;
- no brush contact or wet painting unless later gates pass.

Required tests:

- independent E-stop, enable, power isolation, and watchdog;
- encoder direction, zero, units, limits, and stale-data response;
- current, voltage, temperature, and communication monitoring;
- single-joint, linked-joint, and full-arm free-space trajectories;
- hold-position, controlled stop, restart, and fault recovery;
- collision/workspace envelope and cable-management inspection;
- injected command, sensor, communication, non-finite, and process faults.

Required artifacts and logs:

- electrical and safety diagrams;
- firmware/driver/software versions;
- calibration record;
- complete command/sensor/current/temperature/fault telemetry;
- signed pre-run checklist and residual-risk record.

Pass conditions:

- every tested fault reaches the intended external safe state;
- dry trajectories remain inside declared limits;
- tracking, current, temperature, oscillation, and hold behavior satisfy their
  provisional envelopes;
- operator can remove power independently of inference software.

Immediate stop conditions:

- E-stop, watchdog, limit, or current protection failure;
- unexpected motion on enable, reset, stale command, or software crash;
- encoder disagreement, non-finite state, thermal limit, loose structure,
  cable snag, or unauthorized mode transition;
- any contact with the canvas or paint equipment during a dry-only review.

## G7: Hardware Wet Run

Decision task: `T-715`; final operating authority remains bounded by `T-718`.

Permits:

- supervised scripted wet-paint trials within the reviewed envelope;
- no autonomous painting until shadow inference and fault gates pass.

Required tests:

- controlled brush approach, contact, release, and retract;
- contact-force/current correlation;
- black, white, overlap, curve, broad-mark, edge, reload, and cleaning cases;
- secured paint and solvent fixtures;
- contamination, drip, spill, and cleanup response;
- camera and physical sensor characterization under wet conditions.

Required artifacts and logs:

- synchronized arm, contact, current, camera, and material-consequence records;
- before/after fixture images;
- sim-versus-physical consequence residuals;
- contamination and maintenance observations;
- updated safety and failure logs.

Pass conditions:

- scripted marks complete without exceeding reviewed force, current, thermal,
  workspace, or contamination limits;
- retraction and stop behavior remain reliable with paint present;
- physical consequences are measured rather than judged only by appearance.

Immediate stop conditions:

- unsecured solvent/paint container, spill, entanglement, unplanned contact,
  loss of camera/sensor provenance, force/current excursion, failed retract,
  or contamination of a safety mechanism;
- autonomous command authority during a supervised-script-only gate.

## G8: Research-Grade Experiment

Decision tasks: study-specific M8 task and final synthesis `T-817`.

Permits:

- making only the preregistered claim supported by the reviewed study;
- publishing the corresponding versioned evidence bundle.

Required tests and design:

- accepted upstream capability gates;
- preregistered hypothesis, mechanism, comparison, outcome, exclusion,
  stopping, replication, and analysis rules;
- matched observation and compute access across conditions;
- valid independent experimental units;
- calibration and failure checks relevant to the claimed mechanism;
- declared exploratory versus confirmatory status.

Required artifacts and logs:

- complete experiment manifests;
- raw and derived artifacts with code/data hashes;
- all included, excluded, interrupted, and failed runs;
- analysis code and uncertainty intervals;
- limitation and negative-result report;
- qualitative artifact selection protocol when canvases are shown.

Pass conditions:

- the result survives its registered controls and capability gates;
- exclusions and stopping follow the registered protocol;
- evidence supports the narrow mechanism claim without importing aesthetic,
  intelligence, creativity, or biological-plausibility claims.

Stop conditions:

- changing preferences, metrics, seeds, or artifact selection after inspecting
  outcomes without labeling the result exploratory;
- missing manifests or irreproducible inheritance history;
- lower-level predictive/calibration failure that invalidates the mechanism
  interpretation;
- treating one long developmental history as many independent samples;
- reporting visual examples selected only because they look successful.

## Gate Review Record

Each completed review should append or link a record containing:

```text
gate_id:
review_date:
reviewer:
decision: pass | fail | limited-pass
permitted_mode_or_claim:
artifact_versions:
evidence_table:
waivers:
residual_risks:
follow_up_tasks:
next_invalidating_observation:
```
