# Brush-Posterior Forecast Initialization

Date: 2026-08-05

Status: implemented and locally verified

Scope: compact brush state at the counterfactual motor-forecast boundary

## Result

Counterfactual stroke forecasts no longer initialize brush loading or
bristle-scale variation by continuing the hidden live brush state. Planning
freezes one `BrushLoadBelief` revision per pass. The forecast then combines:

- a compact posterior over normalized fresh load and average black-pigment
  fraction; and
- a separately versioned `brush-microstructure-prior-v0` for bristle-scale
  mark variation.

Particle zero uses the compact posterior means and a deterministic
representative microstructure. Later particles sample the posterior's declared
diagonal variance and independent microstructure noise. All future brush noise
is derived from the forecast request seed, not the live brush RNG.

This is a corrected forecast boundary, not a hardware-calibrated brush model.
The named calibration status remains
`provisional_simulation_only_not_hardware_calibrated`.

## Problem That Was Corrected

The earlier forecast deep-copied `ArmPainterSim` and called `Brush.reload`.
That did not generally reuse the current bristle arrays: reload regenerated a
new pattern. However, it regenerated the pattern by continuing the RNG copied
from the live process. It also replaced compact load uncertainty with a single
mean value. Consequently, two hidden facts from the generative process could
affect a policy forecast:

1. the exact point reached in the live brush RNG stream; and
2. the absence of sampling from the declared load/pigment posterior variance.

Adding randomness after copying hidden state would not solve the access
problem. The forecast noise had to come from a declared model and an
independent seed domain.

## Generative-Model Boundary

For the currently selected dedicated brush, the compact approximation is

```text
q(b) = q(load) q(black_fraction)
```

where both factors are bounded scalar Gaussians represented by means and
variances in `BrushLoadBelief`. The belief also carries a revision, inference
model identifier, and calibration status.

Brush preparation remains a policy latent with two conditional transitions:

- `preserve`: retain the frozen compact posterior;
- `reload`: apply `BrushLoadingModel.reload_transition`, producing a full
  selected-color load distribution and a new revision.

This is not actuator selection and not a conventional controller decision. It
is a conditional realization of the already selected painting policy, scored
through the existing mark-outcome EFE terms. A preserve forecast without a
brush belief now fails rather than silently falling back to process truth.

The microstructure factor is currently

```text
p(microstructure | brush configuration,
                       brush-microstructure-prior-v0)
```

It samples bristle offsets, bristle gains, intermittent-streak phases, and
low-order edge-wobble variables through `Brush.reload`, but with an independent
request-derived RNG. Particle zero uses a deterministic representative:
uniform nominal offsets, mean bristle gain, stratified streak phases, and the
zero-mean boundary-wobble effect. This representative is an approximation for
a non-Gaussian nuisance factor, not a claim that every internal variable has a
literal scalar expectation.

## Particle Semantics

For particle index `i`:

```text
i = 0:
    load = E_q[load]
    black_fraction = E_q[black_fraction]
    microstructure = deterministic representative

i > 0:
    load ~ clipped Normal(load_mean, load_variance)
    black_fraction ~ clipped Normal(black_fraction_mean,
                                    black_fraction_variance)
    microstructure ~ brush-microstructure-prior-v0
```

The load and pigment approximation is diagonal. Clipping to `[0, 1]` is a
bounded-support approximation and is named as such; it is not an exact
truncated-Gaussian posterior.

Every motor realization in the same planning pass receives the same frozen
belief and request seed. This preserves common random numbers across motor
alternatives, so differences are attributable to the realization rather than
uncontrolled noise draws. The seed derivation still includes particle index,
brush revision, selected tone, and distinct domains for compact moments and
microstructure.

The aggregate predictive variance continues to use the law of total variance:

```text
Var[o] = E_particle[Var(o | particle)]
         + Var_particle[E(o | particle)]
```

Thus brush posterior and microstructure uncertainty enter the predicted
outcome distribution without being renamed as a reward or mixed into an
unstructured score.

## Planning Integration And Provenance

`ArmActiveInferenceDriver` now freezes the per-brush beliefs when a local or
background planning pass begins. Motor forecast cache keys include:

- revision;
- load mean and variance;
- black-fraction mean and variance;
- inference model identifier; and
- calibration status.

`ExecutionForecast` now reports:

- `brush_posterior_revision`;
- `brush_inference_model_id`;
- `brush_calibration_status`;
- `brush_microstructure_prior_id`; and
- `brush_preparation_kind`.

These fields make it possible to distinguish a preserve forecast from a reload
forecast and to detect results produced under a different brush model.

## Verified Claims

The tests establish that:

- particle zero uses the frozen compact posterior means;
- later particles vary according to declared load/pigment uncertainty and the
  independent microstructure prior;
- changing exact live load, pigment, bristle arrays, or live brush RNG state
  does not change a belief-conditioned forecast under the same request seed;
- the live brush is not mutated by forecasting;
- reload applies the declared transition and increments the revision;
- preserve without a belief fails closed;
- brush provenance reaches `ExecutionForecast`; and
- a planning pass uses its frozen brush revision even if the live belief is
  updated concurrently.

The focused brush, stroke-execution, and driver suite passed 67 tests. Broader
boundary and backend regression results are recorded in `docs/PROGRESS.md`.

## Remaining Nonconformance

The brush posterior is intentionally compact. It does not yet infer:

- held paint volume and held pigment mass from wet pickup;
- persistent bristle realization or wear;
- path-distance phase carried across marks;
- brush tip-lag state;
- brush compliance/deformation conditioned on contact; or
- calibrated physical variation across real brushes and paint mixtures.

Forecast initialization currently collapses held paint and persistent bristle
history into load, average pigment, and a fresh microstructure sample. That is
a documented approximation, not sensor equivalence. The camera/material path
also does not yet compute the local mark-deposition statistic consumed by
`infer_load_from_mark`.

Update, 2026-08-06: the material half of the repeated action/camera loop is now
implemented by `action-conditioned-camera-update-v0`, and the brush depletion
transition advances on the same action clock. The local camera-derived brush
likelihood statistic remains open.

At the wider legacy/oracle rollout boundary, substrate grain and model
parameters still come from the copied container. Native execution still lacks
a `PlantBackend` sensor adapter. MuJoCo body-parameter uncertainty is not
sampled. Contact probability/force is not mapped into a compliance state.

The default remains fail-closed. The later opt-in
`provisional-sensor-simulation-v0` integration profile avoids the live copied
container by using independent fixed grain/model/compliance priors. It can
paint repeatedly in MuJoCo, but the compact brush update described here
remains prior-only and uncalibrated.

## Recommended Next Boundary

The repeated material sensor loop is now wired as:

```text
executed action
    -> action-conditioned material/brush transition prior
    -> delivered camera observation
    -> material likelihood
    -> revised posteriors
    -> next policy inference
```

The remaining brush step is to derive a local mark-deposition likelihood from
the camera evidence without mistaking the material transition prior for an
observation.
Contact-to-brush-compliance initialization should follow only after a
compliance latent and its likelihood have been declared; force alone does not
identify brush deformation. Substrate-grain and parameter posteriors can then
remove the remaining copied rollout context without weakening the fail-closed
boundary.
