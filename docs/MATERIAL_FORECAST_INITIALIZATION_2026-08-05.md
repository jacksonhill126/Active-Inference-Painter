# Material-Posterior Forecast Initialization — 2026-08-05

## Decision

Counterfactual stroke forecasts that receive a `SpatialCanvasState` must not
read their initial thickness, wetness, bulk black-pigment mass, or surface tone
from the copied `ArmPainterSim` process. They initialize those four independent
material factors from one frozen posterior revision for the planning pass.

This is an initial-state likelihood/transition-boundary correction. It does not
add a reward, aesthetic score, or painting-policy heuristic.

## Why This Was Necessary

The forecast implementation uses a deep copy of the live simulator as a
rollout container. Before this change, replacing MuJoCo q/qvel from the body
posterior still left the copied canvas arrays intact. A motor candidate could
therefore receive an unrealistically accurate prediction of its material
consequences even when the painting posterior came only from camera products.

That matters to active inference because the forecasted next material state
enters the spatial expected-free-energy calculation. Hidden process material
was therefore decision-relevant leakage, not merely a diagnostic shortcut.

## Posterior And Particle Semantics

The implemented approximation is a diagonal Gaussian over the four primary
material channels at the posterior's represented spatial resolution:

```text
q(m_t) = product over cells and primary channels
         Normal(mu_t, diagonal variance_t)
```

The Monte Carlo construction is:

1. Particle zero uses `mu_t` exactly.
2. Later particles sample the declared diagonal variance with a request-derived
   seed and a material-specific random-number domain tag.
3. Sampling occurs at posterior-cell resolution.
4. Each sampled cell is expanded piecewise constantly to native canvas pixels.
5. Physical projection enforces nonnegative thickness/wetness/pigment,
   `black_mass <= thickness`, and `0 <= surface_tone <= 1`.
6. Ground contrast and material coverage are recomputed as deterministic
   transforms. They are never sampled as extra independent evidence.

Sampling before upsampling is intentional. Drawing every native canvas pixel
independently would manufacture spatial precision the coarse posterior does not
contain. Piecewise-constant expansion instead says: “all native pixels covered
by this posterior cell share this particle's cell-level uncertainty.”

White paint on white ground remains handled correctly. Surface tone does not
define coverage; material coverage remains the thresholded consequence of
sampled thickness.

## Predictive Uncertainty

Each material particle is propagated through the same selected native or
MuJoCo plant forecast used by execution. Existing forecast aggregation applies
the law of total variance: it combines the within-particle predictive variance
with between-particle dispersion in predicted next states. Material posterior
uncertainty can therefore affect the predictive density used by EFE without
becoming a scalar uncertainty penalty or reward.

## Provenance And Planning Consistency

`SpatialCanvasState` now records:

- `posterior_revision`;
- `inference_model_id`;
- `calibration_status`.

Camera assimilation advances the revision and records the named camera
likelihood. Oracle material transforms are explicitly labeled
`baseline-oracle-v0:exact-material-transform`. Spatial variational updates also
advance and identify their posterior.

The driver freezes one material posterior per planning pass, adds its model and
revision to the motor-forecast cache key, and passes the same frozen snapshot to
all motor alternatives. A newer camera posterior cannot silently mix with an
already-running candidate comparison.

`ExecutionForecast` carries the material provenance alongside body provenance
and names the remaining approximations.

## Verification

The regression test constructs a live simulator canvas whose hidden material
is deliberately far from the supplied posterior. It verifies that:

- particle zero begins at the posterior mean;
- later particles differ according to declared posterior variance;
- samples are reproducible under the same independent seed;
- changing only hidden live material does not change initialized particles or
  predicted moments;
- black pigment never exceeds sampled paint thickness;
- the live process canvas is not mutated;
- material revision/model/calibration provenance reaches the forecast.

Camera and driver tests separately verify posterior provenance and one frozen
material revision per planning pass. In total, 93 affected boundary,
documentation, spatial-state, camera-inference, stroke-execution, and driver
checks passed; the broad run contained 91 tests and two final focused checks
covered provenance/support validation added during review.

## Remaining Nonconformance

Update: the brush realization/RNG item below was corrected later on 2026-08-05
by `docs/BRUSH_FORECAST_INITIALIZATION_2026-08-05.md`. Compact load/pigment now
comes from `BrushLoadBelief`, and bristle-scale variation uses independent
versioned prior noise. Held paint and persistent bristle history remain
collapsed approximations.

This does not make the whole forecast sensor-equivalent. The deep-copied
rollout container still supplies:

- substrate grain;
- held-paint and persistent-bristle history (compact brush state now uses a
  posterior plus independent microstructure prior as noted above);
- plant/model parameters;
- native plant dynamic/RNG state when no `BodyBeliefSnapshot` is supplied.

The body posterior's contact probability and force are not mapped into a
brush-compression/flexure posterior. Doing so from force alone would be
underidentified and could conflict with sampled joint geometry; a declared
compliance-state factor is needed first.

Update, 2026-08-06: `action-conditioned-camera-update-v0` now runs the
material sequence "executed action -> transition prior -> causally later
camera likelihood -> revised posterior." It records a post-physics capture
boundary, rejects older frames, and automatically polls the MuJoCo camera.
The default policy loop remains fail-closed for the other embodiment blockers
named above.

Later update, 2026-08-06: the opt-in
`provisional-sensor-simulation-v0` profile now replaces the live copied
container with an independently constructed fixed-prior MuJoCo/material model
and completes repeated camera-closed strokes. This is an integration baseline,
not acceptance of the missing parameter/compliance posteriors.

## Recommended Next Boundary

The next implementation should derive the local mark statistic used by the
brush likelihood from camera evidence without feeding a transition prediction
back as if it were an observation. Contact-to-compliance mapping should follow only after
the body posterior explicitly represents the corresponding compliance latent.
Substrate-grain and parameter beliefs can then replace the remaining copied
forecast context without allowing either path to fall back to hidden process
material.
