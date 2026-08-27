# Online Learning And Inheritance Contract

Date: 2026-08-27

Status: AI-112 complete for `m1-formal-policy-baseline-v0`

## Decision

The runtime now uses the versioned contract
`online-learning-inheritance-v0`. A checkpoint is no longer treated as one
undifferentiated memory. Each component has an owner, a persistence rule, and
an allowed load mode.

The machine-readable source of truth is
`src/active_painter/learning_lifecycle.py`; checkpoint schema 7 embeds that
manifest and the runtime exposes it in diagnostics. A load fails closed when
the checkpoint schema, declared architecture, or inheritance manifest differs.

## Component matrix

| Component | Across paintings | Resume individual development | Initialize an individual from shared pretraining | Continue shared pretraining |
|---|---:|---|---|---|
| Dynamics and composition likelihood parameters, including trained passage-support counters | Yes | Restore | Restore | Restore |
| Generative-model optimizer moments | Yes within the same training history | Restore | Reset | Restore |
| Individual policy-proposal parameters, RNG, and optimizer | Yes within one individual history | Restore | Reset | Reset |
| Observational transition, composition, passage, and passage-step replay | Yes within one individual history, capacity-bounded FIFO | Restore | Reset | Reset and rebuild from the declared training split |
| Motion reliability and learned precision calibration | Yes within one individual history | Restore | Reset | Reset |
| Current canvas/material posterior | No | Reset | Reset | Reset |
| Body posterior and body VFE | No | Reset | Reset | Reset |
| Brush-load beliefs | No | Reset | Reset | Reset |
| Gap-increment belief for the current canvas | No | Reset | Reset | Reset |
| Active passage, passage queue, and slow canvas/relational posterior history | No | Reset | Reset | Reset |

This matrix fixes a concrete prior bug: schema 6 serialized the current
canvas's gap-increment belief and `reset()` did not reset it. Schema 7 omits it,
and both checkpoint loading and new-painting reset now construct the declared
prior.

## Load modes and developmental meaning

`resume_individual_development` is exact continuation of one agent's learning
history. It restores learned parameters, optimizer state, observational
replay, calibration, and the disabled-by-default learned proposal. It does not
restore a painting episode.

`initialize_from_shared_pretraining` requires checkpoint provenance with
`training_role=shared_pretraining`. It imports only shared generative-model
parameters and trained likelihood-support counters. The receiving agent starts
with fresh optimizer state, replay, calibration, proposal, and episodic
beliefs. Parallel simulator trajectories therefore remain pooled training
evidence, not autobiographical experience attributed to the receiving agent.

`continue_shared_pretraining` also requires shared-pretraining provenance. It
restores shared generative parameters and their optimizer moments, but clears
replay and rebuilds it only from the new manifest's training split. The offline
trainer uses this mode.

Code-only changes may cross a checkpoint boundary only when schema,
architecture metadata, and this contract still match exactly. A model
architecture change is rejected unless a future explicit, versioned migration
and its tests are supplied. Run manifests remain responsible for recording the
exact code revision.

## Observation boundary for online learning

Every public runtime replay insertion now requires an evidence-source label.
Accepted sources are a registered post-action camera observation, a physical
sensor observation, or an explicitly labelled oracle diagnostic execution.
`model_imagined_rollout` and unknown labels raise before any replay buffer is
mutated. The action transition prior still predicts without writing replay.

This rule covers per-mark transition replay and the slower composition,
passage, and passage-step buffers. Model counterfactuals remain samples from a
transition prior; they are not likelihood observations and cannot train the
online likelihood as though they happened.

Checkpoint diagnostics report counts for the primary observed-transition
stream by evidence source. Passage buffers are derived views of those same
realized transitions and are not double-counted.

## Replay retention and forgetting protocol

Online replay is capacity-bounded FIFO. Held-out trajectories must never enter
replay, and split assignment remains by whole trajectory before local-patch
extraction.

For any admitted continual-learning run, report two likelihood measurements
beside one another:

1. recent observed-transition NLL, to show adaptation to current experience;
2. fixed anchor held-out NLL, using trajectories and seeds frozen before the
   run, to show retained predictive competence.

The forgetting signal is degradation of anchor held-out NLL beyond a
predeclared seed/bootstrap uncertainty band. Crossing that band fails the
learning-admission report and triggers diagnosis of replay coverage or update
schedule. It must not be converted into a scalar reward, aesthetic score,
prior preference, or EFE term.

This contract defines the monitor and failure rule; AI-114 must instantiate the
fixed anchor set and band in its replica manifests before making an inherited
learning claim.

## Verification

The tests cover:

- parameter-plus-replay continuation for an individual resume;
- parameter-only initialization from a shared-pretraining checkpoint;
- preservation of learned parameters, replay, and precision across a painting
  reset while body/canvas/brush/gap/passage beliefs are cleared;
- rejection of model-imagined replay before mutation;
- persistence of developmental precision alongside reset of episode-local gap
  belief;
- replay provenance reaching the observed-transition callback;
- passage and hierarchy replay requiring realized evidence labels.

No reward, preference, VFE term, EFE term, or policy posterior was added or
renamed by this work.

## Remaining boundary

AI-112 closes the ownership semantics, not continual-learning performance.
The project still lacks an AI-114 anchor-set measurement showing that online
updates avoid catastrophic forgetting across three representative replicas.
That evidence remains a run-level acceptance obligation.
