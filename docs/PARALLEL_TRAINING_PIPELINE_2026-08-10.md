# Parallel Data Collection And Centralized Training Baseline

Date: 2026-08-10; acceptance update 2026-08-11
Status: implemented, tested, and exercised by the accepted AI-108 corpus

## Purpose

The interactive painter learns approximately in real time because physical
execution, camera acquisition, policy inference, and small online gradient
updates are coupled in one loop. That loop remains useful for inspecting one
agent's development, but it is not an affordable way to obtain a large model
corpus.

This baseline separates two jobs without changing the painting decision rule:

1. Independent headless agents collect camera-conditioned experience through
   active-inference policy selection.
2. A central trainer performs conventional batched parameter learning from the
   collected posterior transitions.

The conventional multiprocessing and SGD infrastructure is not relabelled as
active inference. Painting policies inside every worker are still selected by
the existing policy posterior and expected-free-energy decomposition.

## Immediate planning speed change

`learned_proposal_diagnostic_interval_plans` now defaults to 16. The expensive
Monte Carlo comparison between the learned and hand-written proposal is
measured on the first plan and every sixteenth plan thereafter; its last value
is cached between measurements.

This is safe because that proposal comparison is evidence only. No VFE term,
EFE term, preference, precision belief, policy prior, policy posterior, or
control branch reads it. In contrast, the composition-gap calculation still
runs every plan because its increment belief contributes to the declared
stop-policy prior. `planningProfile.proposalDiagnosticFresh` identifies plans
that paid for a fresh comparison.

Set `--proposal-diagnostic-interval-plans 1` on the web server to reproduce the
historical every-plan diagnostic cadence.

## Collection architecture

Run:

```powershell
python -m active_painter.parallel_collect `
  --workers 6 `
  --trajectories 60 `
  --output-dir runs/corpus-20260810
```

Each spawned worker owns an independent:

- `mujoco-robstride-electromechanical-v4` process and counterfactual model;
- camera acquisition clock and camera RNG;
- canvas, brush, and substrate seeds;
- spatial material posterior;
- body posterior;
- replay buffers;
- persistent canvas and relational latents;
- passage history, precision beliefs, and proposal state.

Workers use CPU Torch by default so several collectors do not compete with the
central CUDA training job. They never write a shared checkpoint. The default
maximum of 128 transitions archives an explicitly labelled
`fixed_horizon_truncation` if the policy has not selected terminal `stop` by
then. A policy-selected completion is labelled `policy_selected_stop`; the two
must not be conflated in composition training or reports.

The collector now recycles the entire runtime after one trajectory by default.
This avoids the measured progressive slowdown in long-lived multi-painting
runtimes. Completed shards are still discovered, hashed, split, and audited if
another job fails; the batch is explicitly marked incomplete. The opt-in
`research_full_roll` profile adds dynamic +/-32-degree roll sweeps to the
bounded profile's neutral and fixed +/-24-degree realization support. These
remain conditional motor realizations below the selected painting policy.

This collector uses the explicit `provisional-sensor-simulation-v0` path:
registered MuJoCo camera observations update the spatial posterior after the
action-conditioned capture boundary. It remains an uncalibrated
simulation-only integration baseline, not sensor-equivalent hardware
cognition.

## Corpus record and leakage boundary

`trajectory-posterior-corpus-v2` stores one compressed NPZ file per whole
trajectory. Each transition includes:

- the pre-action camera-derived spatial posterior mean and diagonal log
  variance, at both coarse and pixel training scales;
- the selected eight-dimensional mark, including signed curvature;
- the conditional motor-realization kind;
- the compact inferred pre-stroke brush-load/pigment posterior and an explicit
  availability bit;
- the causally later camera-derived posterior mean and variance;
- posterior revisions and inference-model identifiers;
- final posterior fields and termination provenance.

It does **not** store exact live material state as a training input. Raw camera
frames are also not yet persisted; this first corpus trains the material
transition and composition models from the camera likelihood's posterior. A
future perception corpus must add raw registered products without weakening
the same trajectory split.

Legacy v1 shards remain readable. They carry an unavailable brush-context bit,
not an invented zero-load observation. The separate shadow cVAE trainer uses
this v2 condition; see
`CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md`.

The collector writes `split_manifest.json`. Assignment uses deterministic
greedy multilabel transition balancing over measured posterior/action, reach,
curvature, and conditional motor bins. `corpus_audit.py` can combine multiple
profile roots, writes per-split transition and trajectory-presence counts, and
records a SHA-256 digest for every shard. The split unit is the entire
trajectory. Local patches are constructed by the central trainer only after it
opens the training split. Validation and test trajectories are never added to
a training replay.

This ordering prevents two common leaks:

1. neighboring patches from one curve appearing in train and validation;
2. early and late states from the same evolving canvas appearing in different
   splits.

## Centralized training

Run:

```powershell
python -m active_painter.offline_train `
  --manifest runs/corpus-20260810/split_manifest.json `
  --output-checkpoint runs/checkpoints/shared-pretraining.pt `
  --device cuda `
  --batch-size 64 `
  --dynamics-steps 1000 `
  --composition-steps 200
```

The trainer:

1. verifies that no trajectory identifier occurs in more than one split;
2. reconstructs a canonical model configuration, allowing only declared
   worker-randomness and motor-candidate collection-profile differences;
3. optionally loads shared parameters and optimizer moments from
   `--input-checkpoint`;
4. clears every inherited replay before loading this manifest;
5. extracts local patches from training trajectories only;
6. measures validation/test conditional NLL before and after training;
7. trains the shared local transition likelihood in GPU batches;
8. trains composition only on policy-selected terminal canvases;
9. writes a compatible driver checkpoint plus a separate JSON report.

Deterministic relational region observations are precomputed once per terminal
canvas. They are then passed directly to the composition loss, avoiding the
previous repeated GPU-to-NumPy round trip and hard region clustering inside
every gradient step.

The output is explicitly `shared_pretraining`. Shared model parameters may be
pooled; online canvas posteriors, precision beliefs, brush state, passage
history, and persistent slow latents are not pooled. The normal live reset
constructs those individual states for the receiving agent.

## Scaling benchmark

Run:

```powershell
python -m active_painter.parallel_benchmark `
  --worker-counts 1 4 6 8 `
  --trajectories-per-worker 1 `
  --transitions-per-trajectory 4
```

The benchmark uses labelled fixed-horizon truncations and reports trajectories
per second, transitions per second, speedup relative to the first worker
count, and parallel efficiency. It is a throughput benchmark, not a
painting-quality or learning-capability result.

The 2026-08-10 1/4/6/8 benchmark measured 0.0564/0.2654/0.3976/0.4740
transitions per second. The accepted AI-108 one-runtime-per-trajectory batches
measured 0.492 transitions/s for three-way fixed roll and 0.437 transitions/s
for five-way full roll. See `FULL_BENCHMARK_TRAINING_TECHNICAL_2026-08-10.md`
and `AI108_CORPUS_TECHNICAL_2026-08-11.md`.

## Current limitations and next evidence

- Collect enough policy-selected terminal paintings to support composition
  training and a nontrivial validation/test split.
- Persist registered camera products when training the perception likelihood,
  while preserving whole-trajectory split assignment.
- Add held-out calibration and learning curves by tone, blank/existing
  material, width, curvature, reach region, and motor realization. Do not
  require wet-over-wet sensor strata until a legitimate wetness likelihood
  exists.
- Do not treat repeated simulator rollouts as individual development. They are
  shared pretraining data with explicit provenance.

## Verification completed

Focused tests cover:

- evidence-only diagnostic decimation while composition-gap updates remain at
  every plan;
- lossless trajectory-shard round trip;
- non-overlapping trajectory split assignment;
- centralized train-only patch materialization and checkpoint provenance;
- precomputed relational observations bypassing repeated CPU extraction.

The 2026-08-11 accepted corpus adds explicit tests/evidence for posterior-only
condition labels, process/inference resolution provenance, one-runtime job
construction, compact brush context, multi-profile pooled motor conditioning,
multi-root manifest consumption, and per-split required-condition coverage.

An actual MuJoCo/camera smoke also passed after correcting Windows OpenGL
teardown so the runtime thread that creates the camera renderer closes it. A
one-worker/one-transition command completed in 16.9 s end to end; a
two-worker/two-transition command completed in 18.6 s. For this very small,
startup-dominated sample, observed transition throughput was approximately
1.8 times the one-worker command. The real posterior shard then completed one
central CPU gradient step and wrote a compatible shared-pretraining checkpoint.
These are smoke observations, not stable scaling or learning claims. Later
benchmark and AI-108 results supersede the original pending-throughput note;
held-out calibration and learning curves remain open.
