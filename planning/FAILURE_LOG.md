# Failure-Mode Log Contract

The failure log preserves defects, interruptions, invalid evidence, and accepted
limitations as project evidence. It is not a list of only unresolved bugs.

Each meaningful run has `failure-log.jsonl`, even when the file is empty.
Project-wide failures that are not tied to one run use the same record shape in
a milestone or validation artifact bundle.

The initial machine-readable shape is illustrated by
`planning/templates/failure-log.example.jsonl`.

## Categories

Use exactly one primary category and any number of secondary categories.

| Category | Scope |
| --- | --- |
| `kinematics` | frames, transforms, IK/FK, reachability, joint geometry |
| `contact` | touch detection, force/contact state, release, collision |
| `paint_material` | deposition, thickness, pigment, wet mixing, coverage, rendering from material |
| `planner` | policy support, proposal generation, VFE/EFE use, stopping, passage decisions |
| `uncertainty` | likelihood variance, precision, calibration, disagreement, non-finite beliefs |
| `visualization` | viewer state, synchronization, display-only rendering, stale UI |
| `mujoco` | MJCF, MuJoCo dynamics, backend adapter, native parity |
| `cad` | geometry source, fit, tolerances, fabrication exports |
| `hardware` | motors, encoders, drivers, wiring, structure, physical sensors |
| `safety` | hard limits, watchdogs, emergency stop, unsafe state or procedure |

Environmental and tooling failures still use the closest category and record
`failure_domain: environment` or `tooling`.

## Required Fields

Every event contains:

- `schema_version`, `event`, `failure_id`, `event_revision`, and UTC timestamp;
- run ID or `null`, primary/secondary categories, and failure domain;
- a factual symptom separated from suspected causes;
- reproduction status, steps, expected result, observed result, frequency, and
  minimal case when known;
- severity and scientific impact;
- containment, proposed mitigation, and verification method;
- current status and status reason;
- linked task IDs;
- evidence artifact paths and relevant state/canvas/telemetry revisions;
- safety response, when applicable.

Suspected causes are hypotheses, each with evidence and a confidence label.
Do not rewrite a suspicion as an observation.

## Severity

- `diagnostic`: noteworthy anomaly with no demonstrated behavioral or evidence
  impact.
- `minor`: localized defect; run remains usable for its declared purpose.
- `major`: material behavior or evidence is affected; affected claims are
  provisional.
- `blocking`: required capability or validation cannot proceed.
- `safety_critical`: actual or credible risk to people, equipment, or
  surroundings. External safety response takes precedence over inference.

Severity is operational triage, not a reward, prior preference, precision, VFE,
or EFE term. It must never enter painting-policy selection.

Scientific impact is recorded separately as `none`, `visualization_only`,
`measurement_quality`, `model_validity`, `run_validity`, or `safety`.
A visually obvious defect can have no model-validity impact, while a subtle
sensor leak can invalidate a run.

## Failure Domains

`failure_domain` distinguishes:

- `active_inference`: factorization, posterior inference, priors, preferences,
  precision, VFE, EFE, or policy posterior;
- `generative_process`: simulated or physical plant, sensor, canvas, or
  material consequence;
- `support_engineering`: IK, controller, serialization, viewer, conventional
  learning implementation, or backend plumbing;
- `safety`: external safety system or procedure;
- `environment`: operating system, unrelated process, power, or workcell;
- `tooling`: test harness, build tool, CAD tool, or analysis tool.

When classification is uncertain, record all plausible domains in suspected
causes and leave the primary domain `unknown`. Do not call an ordinary
controller failure an active-inference failure without evidence.

## Status And Event History

The JSONL log is append-only. Each line contains a complete current snapshot
for one failure plus an event type:

- `opened`;
- `updated`;
- `mitigated`;
- `validation_requested`;
- `resolved`;
- `accepted_limitation`;
- `classified_environment_noise`;
- `reopened`.

Current status is one of `open`, `investigating`, `mitigated`, `validate`,
`resolved`, `accepted_limitation`, or `irrelevant_environment_noise`.
Increment `event_revision` for every event. Never delete the original event or
silently edit history.

`resolved` requires evidence that the acceptance or reproduction case now
passes. `accepted_limitation` requires a bounded scope and impact statement.
Safety-critical failures require Jackson's explicit acceptance before return to
operation.

## Preservation Rule

Failed, interrupted, non-aesthetic, and negative-result runs remain evidence.
Storage pressure may move bulky artifacts to archival storage, but the manifest,
failure history, hashes, and archive location remain.

An event may be classified as `irrelevant_environment_noise` only when:

- the cause is external to the project behavior under study;
- the rationale and evidence are recorded;
- exclusion does not selectively remove an unfavorable result;
- the entry remains in the append-only log.

The classification permits exclusion from a particular analysis; it does not
erase the event.

## Reproduction And Verification

Prefer the smallest deterministic reproduction that retains the symptom.
Record exact command, configuration, seed, version manifest, and artifact
revision when available. Intermittent issues include observed frequency and
attempt count.

Mitigation and verification are separate:

- containment limits immediate impact;
- proposed mitigation states the intended change;
- verification states the test or observation that could close the issue.

A failure linked to an experiment manifest appears in both
`run.failure_ids` and the failure log. The manifest records run validity; the
failure record explains why.
