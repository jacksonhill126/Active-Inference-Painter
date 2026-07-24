# Experiment Manifest Contract

The experiment manifest is the index and provenance record for one run. It
does not replace telemetry or scientific reports. It states what was run, what
the agent could observe, which probabilistic quantities affected policy
inference, which artifacts were produced, and whether the run is admissible for
its intended use.

The initial machine-readable shape is illustrated by
`planning/templates/experiment-manifest.example.json`.

## Scope

Create a manifest for every run used as:

- debugging or failure evidence;
- a baseline, calibration, pilot, or validation case;
- an exploratory, confirmatory, transfer, or qualitative research run;
- a source for a figure, painting comparison, or public claim.

Smoke tests that produce no retained evidence may omit a manifest. A failed or
interrupted meaningful run must retain one.

## Lifecycle

Write the initial manifest before the first painting action, then update it
atomically as artifacts appear. On termination:

- set status to `completed`, `stopped`, `interrupted`, or `failed`;
- record the end time and termination reason;
- mark each required artifact `present`, `missing`, `unsupported`, or
  `not_applicable`;
- set the evidence level to `provisional` when required provenance or evidence
  is missing.

Archived manifests are immutable. A correction increments
`manifest_revision`, records `supersedes_revision`, and preserves the prior
manifest. Do not rewrite a failed run to look complete.

## Required Metadata

| Section | Required contents |
| --- | --- |
| Identity | schema version, manifest revision, unique run ID, run kind, status, start/end UTC, study ID if any, replica ID, parent run if any |
| Versions | relative path to `version-manifest.json`; exact code, model, backend, geometry, calibration, hardware, config, and checkpoint identities remain in that file |
| Configuration | resolved-config path and SHA-256, planner mode, transition mode, backend, compute device, canvas dimensions, pixel/material scale |
| Randomness | every independently seeded RNG stream, including agent, policy proposal, NumPy, Torch, plant/process noise, brush, and canvas grain where applicable |
| Sensor access | observation mode, permitted observations, derived observations, evaluation-only process truth, unavailable real-platform variables |
| Learning state | whether learning was enabled, checkpoint load result, inherited parameter groups, reset beliefs, replay/optimizer persistence, training-data references |
| Timing | wall time and available inference, planning, execution, learning, serialization, and rendering timings |
| Termination | terminal status, reason, stop source, interruption/fault reference, and safety events |
| Artifacts | paths, formats, synchronization/revision fields where available, SHA-256 values, and availability status |

T-802 may extend learning, replica, and confirmatory-study provenance. It must
preserve these field meanings and treat missing required provenance as
incompatible with confirmatory evidence.

## Active-Inference Boundary

The manifest has separate `active_inference`, `conventional_support`, and
`approximations` sections.

Every decision-relevant painting quantity in `active_inference.terms` declares:

- a stable name;
- its grounding as `likelihood`, `transition_prior`, `prior_preference`,
  `precision_belief`, `fixed_precision`, `vfe_term`, `efe_term`,
  `policy_prior`, or `policy_posterior`;
- the VFE, EFE, or policy-posterior decomposition in which it is reported;
- a factor/specification reference when one exists;
- units and reduction convention;
- a trace artifact;
- whether the implementation is exact, approximate, provisional, or disabled.

Terms that cannot yet be mapped to the generative model are listed under
`approximations` as unresolved. They must not be presented as established
active-inference factors merely because they appear in an EFE calculation.

`conventional_support` lists IK, trajectory interpolation, motor control,
collision checking, hard safety, rendering, serialization, SGD, and other
engineering that realizes or supports a selected painting policy. These
components may be essential, but they are not renamed as active inference.
Hard safety remains external to policy selection.

## Required Artifacts

| Artifact | Requirement |
| --- | --- |
| Resolved configuration | required |
| Version manifest | required |
| VFE trace and decomposition | required when state inference ran |
| EFE trace and decomposition | required when policy inference ran |
| Candidate-policy prior and posterior trace | required when policy inference ran |
| Agent-accessible observation trace or synchronized references | required for research runs |
| Belief and prediction traces | required when claims depend on them |
| Robot/process telemetry | required when execution ran |
| Canvas snapshots | required at initial and terminal state; passage/mark snapshots depend on run protocol |
| Failure log | always required, even when it contains no failure entries |
| Checkpoint | required only when preserving or inheriting learned state |
| Timing/profile trace | required when making performance claims |
| Analysis outputs | required when the run supports a reported result |

VFE and EFE must remain separate traces. A convenience total may be reported,
but it cannot replace either decomposition. Canvas process truth, agent visual
observation, and predicted canvas state must use distinct artifact names.

## Artifact Paths And Missing Data

Paths are relative to the run root unless marked external. Deterministic names
are preferred. An artifact record contains:

- `path`;
- `format`;
- `status`;
- `sha256` when present;
- relevant state, observation, belief, policy, telemetry, or canvas revisions;
- `reason` when missing, unsupported, or not applicable.

Silently omitting an unavailable field is prohibited. Unsupported capability is
not itself a failure, but claiming evidence that requires it is.

## Evidence Levels

- `development`: implementation evidence only.
- `provisional`: useful evidence with a declared missing prerequisite,
  approximation, or incomplete provenance.
- `exploratory`: hypothesis-generating run under a recorded protocol.
- `confirmatory`: permitted only after the relevant capability gates and study
  protocol pass and all required provenance is present.

Run kind and evidence level are separate. For example, a planned confirmatory
run becomes provisional if its observation trace is lost.

The manifest records outputs without judging their appearance. No aesthetic
score, selected-best flag, or reward is part of this contract.
