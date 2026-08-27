# M1 Gate Repair And Formal Policy Baseline

Date: 2026-08-26

Status: implemented gate repair; M1 remains active pending AI-112 through
AI-115

## Decision

The canonical default painting-policy configuration is now
`m1-formal-policy-baseline-v0`.

It retains the admitted M1 likelihood, transition, terminal-coverage,
precision, VFE, EFE, policy-prior, and sampled-candidate posterior terms. It
disables every policy path derived from the earlier coarse-material
composition hierarchy:

- `composition_enabled = false`;
- `composition_gap_precision = 0`;
- `canvas_latent_transition_precision = 0`;
- `relational_transition_precision = 0`;
- `passage_trajectory_enabled = false`; and
- `gap_progress_stop_enabled = false`.

The mechanisms remain executable only through an explicit configuration
opt-in identified at runtime as
`legacy-material-composition-diagnostic-v0`. That profile is a controlled
diagnostic over the provisional `SpatialCanvasState`; it is not an admitted
visual composition model, painting preference, or sensor-equivalent result.

The learned proposal mixture remains zero by default. The policy posterior
therefore retains its already accepted M1 interpretation as
`Q(pi | sampled candidate set S)` rather than a proposal-invariant posterior.

## Why this closes AI-110 for M1

AI-110 allowed either a frozen/cross-fitted accepted structural preference or
disabling the term until M2 supplied an admissible visual hierarchy. The
coarse 16x16 material hierarchy cannot satisfy the visual boundary: it carries
material fields, deterministic region summaries, and no admitted tone/mass/
relation/motif likelihood. Allowing it to affect terminal EFE or the stop
policy prior would also let a self-trained diagnostic shape the data later used
to judge the replacement model.

The M1 decision is therefore **disabled**, not "validated." M2 may propose a
new structural preference only after a predictive visual hierarchy exists and
the frozen/cross-fitted safeguards are tested. That later model needs a new
identity and admission decision; re-enabling this legacy profile is not a
shortcut to acceptance.

## AI-109 scope repair

AI-109's M1 question concerned whether the then-current coarse-material local
and hierarchy models justified more capacity. Its 27-run local branch answered
that question:

- more data helped modestly;
- generic width/depth did not;
- five ensemble members improved density;
- the material cVAE did not materially outperform the CNN; and
- the normalized identity/consequence mixture improved density but failed
  exact predictive-mixture calibration.

That is a valid negative/inconclusive M1 capability result. Requiring AI-109 to
remain open until a terminal registered-image corpus and a new visual
tone/edge/mass hierarchy exist made M1 depend on the M2 architecture that M1
was supposed to unblock.

AI-109 is therefore closed for M1. Its uncompleted visual work is preserved,
not discarded, under:

- AI-205 for registered local visual training/live alignment;
- AI-206 for recursive visual prediction;
- AI-208 for predictively necessary full-canvas visual/mass state; and
- AI-214 for visual and hierarchy data/capacity/seed curves.

Fixed-horizon endpoints remain truncations. The 2026-08-12 stop pilot remains
termination evidence only, and the retained 18 scaled visual trajectories
remain suitable only for an initial local mark-model baseline.

## Checkpoint and run identity

`ArmActiveInferenceDriver` now includes the computed painting-policy profile
in checkpoint architecture metadata and runtime diagnostics. The checkpoint
schema advances from 5 to 6 so a pre-decision checkpoint cannot silently load
as the new formal baseline.

The profile classifier is conservative: any opt-in to the legacy hierarchy,
its transition precisions, passage trajectory, or gap-progress stop prior
changes the run identity to the diagnostic profile.

## Gate state after this change

- AI-109: `Done` for M1; visual successor work moved to M2.
- AI-110: `Done` with the disabled decision.
- AI-112: `Ready`; inheritance/reset semantics still need acceptance.
- AI-113: `Ready`; representative phase profiling is now unblocked.
- AI-114: `Blocked` on AI-112 and AI-113; three manifested replicas do not yet
  exist for this new profile.
- AI-115: `Blocked` on AI-114 and owner lock review.

M1 is not declared complete by this repair. The result is an executable,
non-circular path to the lock decision.

## Validation contract

Tests require the default configuration to identify as
`m1-formal-policy-baseline-v0`, omit the legacy hierarchy and learned proposal,
and expose zero/default-disabled policy terms. Separate positive-control tests
must explicitly opt into the legacy diagnostic and continue to verify its
probabilistic decomposition.

The canonical visual boundary remains
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

## Verification evidence

Observed on the Windows/Python 3.14 development host after the repair:

- the exact deterministic CI file set passed `168` tests with two expected
  obsolete-summary warnings in `63.18 s`;
- the affected composition, hierarchy, precision, stop, driver, and support-
  documentation set passed `123` tests with 32 expected obsolete-summary
  warnings in `78.01 s`; and
- `git diff --check` passed after the documentation cleanup.

These are regression and boundary checks. They do not supply the representative
runtime measurements required by AI-113 or the three manifested painting
replicas required by AI-114.
