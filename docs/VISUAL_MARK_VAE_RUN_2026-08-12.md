# Visual Mark VAE Run — 2026-08-12

Status: scaled corpus collection in progress; end-to-end smoke passed

## Purpose

This is the first implementation and training run of the owner's original VAE
proposal: a stochastic, action-conditioned model of **visual mark
consequences**. It is intentionally separate from the earlier coarse-material
patch cVAE.

The model asks: given a fresh registered image of the proposed target region,
the selected stroke, the compact pre-stroke brush belief, camera identity, and
conditional motor realization, what distribution of post-stroke images should
be expected?

## Causal data path

`ArmActiveInferenceDriver` caches the last accepted registered camera bundle.
When execution completes it freezes that bundle with the selected action. The
runtime then registers the earliest permissible post-action exposure. Older
frames are rejected. The first later bundle that produces a camera likelihood
completes one visual transition.

`registered-visual-trajectory-corpus-v1` stores:

- full canvas-registered pre/post grayscale camera products and validity masks;
- exposure sequences and timestamps, camera/product IDs, calibration revision,
  observation model, and registration kind;
- the selected eight-dimensional stroke, conditional motor realization, and
  compact inferred brush belief;
- policy-stop versus fixed-horizon termination provenance;
- complete trajectory identity for split isolation.

It does not store process thickness, wetness, pigment fields, segmentation,
contact truth, joint state, or exact simulator state.

## Model factorization

The shadow model identifier is `action-conditioned-visual-mark-cvae-v0`:

```text
p(z | pre-image, action, brush, camera, motor)
q(z | pre-image, observed post-image, action, brush, camera, motor)
p(post-image | pre-image, action, brush, camera, motor, z)
```

The predictive prior is a learned diagonal Gaussian. The recognition density
may see the observed consequence only during training. Counterfactual sampling
uses the predictive prior. The decoder supplies a normalized pixelwise Beta
likelihood over registered intensity. VFE is logged as reconstruction negative
log likelihood plus latent KL; held-out action ablation and oriented-gradient
diagnostics are reported separately and are not rewards or preferences.

The local crop is an isotropic square around sampled straight/quadratic mark
support plus declared width context. This retains stroke angle instead of
warping every bounding box to a rectangle.

## Smoke evidence

The first real MuJoCo smoke corpus contains 3 complete trajectories and 10
camera examples, split 1/1/1 by whole trajectory. A 32-pixel, 8-latent,
8-base-channel model trained for three epochs. Training negative ELBO decreased
from 0.05735 to 0.04730 nats per observed pixel; validation negative ELBO
decreased from 0.29595 to 0.28337.

This proves the causal collector, loader, model, normalized likelihood,
checkpoint, evaluation, and visualization path run end to end. It is not a
learning-capability result: with only ten camera examples, the model remains
worse than the fresh-image identity baseline on held-out pixel and edge error.

Smoke artifacts are under `data/visual_vae_run_20260812/`:

- `split_manifest.json`
- `collection_report.json`
- `visual_mark_cvae_smoke.pt` and its best checkpoint
- `training_report_smoke.json`
- `prediction_panel_smoke.png`

## Scaled run

The scaled collector uses 96 trajectories, three isolated MuJoCo runtimes,
eight-transition censoring, and whole-trajectory 70/15/15 splitting. Its
artifacts use the `_full` manifest/report names and `corpus_full/` directory.
Every runtime writes completed trajectory shards atomically. Collection
`--resume` audits matching shards, preserves them, and continues each worker at
the next trajectory index, so already closed trajectories survive interruption.
Training checkpoints are written after every epoch and can resume to a
requested total epoch count.

The first scaled attempt was interrupted by a planned Windows Update restart,
not by the collector. Windows System Event 1074 records
`MoUsoCoreWorker.exe` initiating the restart at 2026-08-12 18:14:31 local
time. All three progress files were still advancing through 18:14:44–18:14:47.
Eighteen complete shards survived: 144 executed transitions and 288 paired
camera examples. No temporary shard remained. Windows subsequently performed
two planned update/upgrade restarts; its Windows Update and component-
servicing reboot-pending flags are now clear.

`scripts/run_visual_vae_overnight.ps1` is the durable manual entry point for
the continuation. It resumes the remaining trajectories, fails before training
if collection fails, then starts the 80-epoch CUDA run with an epoch checkpoint,
report, prediction panel, and append-only console log. The terminal must remain
open. On the current Balanced power plan AC sleep is disabled; battery sleep is
15 minutes, so the laptop must remain connected to AC.

## Standing and next evidence

The model is shadow-only. It does not influence policy selection, VFE in the
live camera loop, EFE, composition, or motor realization. Before integration it
must beat deterministic/no-latent baselines on held-out normalized likelihood,
show action/brush/context use, demonstrate calibrated uncertainty, preserve
tone and oriented boundaries, and survive recursive visual rollout tests.
