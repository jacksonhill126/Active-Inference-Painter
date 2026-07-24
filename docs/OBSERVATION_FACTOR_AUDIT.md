# Observation Factorization And Units Audit

Status: accepted M1 baseline audit
Task: AI-103
Date: 2026-07-24
Applies to: `baseline-oracle-v0`

## Decision

The spatial material state contains four primary factors and two deterministic
views:

| Field | Current role | Native unit | Independent likelihood evidence? |
| --- | --- | --- | --- |
| thickness | hidden material factor, exposed by the oracle baseline | simulator deposition units per pixel | yes |
| wetness | hidden persistent material factor, exposed by the oracle baseline | simulator wet-material units per pixel | yes |
| black pigment mass | hidden conserved material factor, exposed by the oracle baseline | simulator pigment units per pixel | yes |
| surface tone | hidden top-surface optical factor, exposed by the oracle baseline | dimensionless black fraction in `[0, 1]` | yes |
| ground contrast | deterministic optical view | dimensionless tone difference in `[0, 1]` | no |
| material coverage | deterministic occupancy view | dimensionless indicator in `{0, 1}` | no |

Ground contrast is recomputed from thickness, surface tone, substrate tone, and
the opacity map:

```text
opacity = 1 - exp(-thickness / thickness_scale)
observed_tone = (1 - opacity) * ground_tone + opacity * surface_tone
ground_contrast = abs(observed_tone - ground_tone)
```

Coverage is recomputed as:

```text
material_coverage = 1[thickness >= paint_presence_threshold]
```

Treating either result as another conditionally independent observation of the
same oracle material array would count the same evidence twice. They remain
useful state views and preference outcomes, but they no longer contribute
separate spatial observation VFE, transition NLL, EFE ambiguity, or transition
information gain.

## Probability-Term Inventory

| Term | Variables and reduction | Reported unit | Baseline judgment |
| --- | --- | --- | --- |
| Summary observation `p(o_t | s_t)` | Six correlated global aggregates; Normal log probabilities summed across dimensions and Monte Carlo averaged | nats per six-dimensional summary vector | compatibility-only diagonal pseudo-likelihood |
| Spatial oracle observation `p(o_px | x_px)` | Four primary fields; diagonal Gaussian expectation averaged over field, cell, and channel | nats per independent cell-channel | active oracle baseline |
| Summary transition `p(s_t+1 | s_t, a_t)` | Six correlated aggregates; Gaussian kernel averaged over summary dimensions and bootstrap samples | nats-equivalent per summary dimension, with `0.5 log(2 pi)` omitted | compatibility-only pseudo-likelihood |
| Spatial transition `p(x_t+1 | x_t, a_t)` | Four primary fields inside the dense grid or valid local-patch mask; averaged over primary channels and cells, then bootstrap samples | nats-equivalent per independent cell-channel, with `0.5 log(2 pi)` omitted | active learned transition approximation |
| Spatial posterior VFE | analytic diagonal-Gaussian KL plus expected observation NLL, both averaged over four primary fields and pixels | nats per independent cell-channel | active oracle baseline |
| Spatial transition entropy and information gain in EFE | primary-field Gaussian entropy averaged over four channels and full-canvas area; local support is divided by full-canvas area | nats per independent full-canvas cell-channel | active approximation |
| Spatial observation ambiguity in EFE | excess entropy over the dry-canvas observation baseline, using four primary fields | nats per independent full-canvas cell-channel | active but uncalibrated |
| Terminal coverage preference | moment-matched Beta forecast against the declared terminal Beta preference, evaluated only at `stop` | nats per terminal coverage outcome | active declared preference |
| Passage observation update | diagonal Gaussian precision fusion over six observed passage coordinates plus a beta-Bernoulli tone update | mixed native coordinates; no coherent scalar VFE is currently reported | provisional pseudo-likelihood |
| Canvas hierarchy reconstruction | Gaussian reconstruction and latent KL divided by material cell-channel count | nats per coarse material cell-channel | provisional learned density |
| Relational hierarchy reconstruction | Gaussian reconstruction and latent KL averaged per relational dimension | nats per relational dimension | provisional learned density |
| Hierarchy transition evaluation | diagonal Gaussian KL and cross entropy averaged per canvas or relational latent dimension | nats per latent dimension | provisional learned density |
| Motor EFE likelihood | normalized proprioceptive outcomes with analytic diagonal Gaussian risk, excess likelihood entropy, and mutual information | nats per candidate execution forecast | provisional and oracle-conditioned in the baseline |

The summary state mixes fractions, deposition-unit averages, and maxima. Its
configured scalar standard deviations therefore do not define a single
physical measurement scale. Summary mode is retained for compatibility and
debugging, not as evidence for sensor-equivalent research claims.

## Normalization Rules

1. Spatial observation VFE is a mean over independent material factors and
   pixels. Repeating an identical field over more cells does not change the
   reported value.
2. Local-patch EFE terms are divided by full-canvas area. A larger crop does
   not become preferable merely because more array elements were evaluated.
3. Padded patch cells are excluded by the validity mask.
4. Ensemble/bootstrap reductions average only selected member-sample terms.
5. Deterministic contrast and coverage channels are projected after primary
   state prediction and do not expand the likelihood dimension.
6. Absolute continuous-density NLL depends on coordinate units. If a variable
   is scaled by `k` and its variance by `k^2`, Gaussian KL is invariant while
   NLL shifts by `log(abs(k))` per scaled dimension. Posterior inference and
   comparisons between policies remain unchanged only when that Jacobian
   constant is applied consistently.

The transition trainers omit the policy-independent Gaussian
`0.5 log(2 pi)` constant. Their values are suitable as optimization losses and
relative NLL diagnostics, but must not be compared numerically with the
constant-complete posterior VFE without restoring the omitted term.

## Implemented Corrections

- Declared `INDEPENDENT_MATERIAL_CHANNELS` and
  `DERIVED_MATERIAL_CHANNELS`.
- Restricted dense and local transition training NLL to the four primary
  fields.
- Fixed transition log variance for deterministic output views and continued
  to recompute their means from primary material predictions.
- Restricted spatial posterior fusion and VFE to the four primary fields.
- Restricted spatial EFE ambiguity and transition entropy/information terms to
  the four primary fields.
- Corrected dense entropy reduction to average over both spatial axes and
  channels.
- Changed spatial VFE diagnostics to
  `nats_per_independent_cell_channel`.

## Provisional Until M2

- The oracle directly observes hidden thickness, wetness, pigment, and surface
  tone arrays. A real platform will not.
- The spatial observation variance is a hand-declared diagonal function of
  oracle material state and is not calibrated against camera, encoder,
  current, or force data.
- Pixel independence ignores brush-shaped, optical, and temporal covariance.
- Variance for deterministic contrast and thresholded coverage is only a
  propagated placeholder. A sensor-equivalent model must derive uncertainty
  through the observation mapping rather than attach another likelihood.
- Passage observations use executed action geometry and exact material deltas.
- Summary mode and current hierarchy densities remain correlated
  approximations requiring separate validation.

## Verification

`tests/test_observation_factorization.py` checks:

- the primary/derived factor declaration;
- invariance of posterior VFE to contradictory derived-channel inputs;
- exclusion of derived targets from dense and local transition NLL;
- exclusion of derived channels from EFE ambiguity and entropy;
- invariance under repetition of identical spatial cells;
- Gaussian KL invariance and the expected NLL Jacobian shift under a consistent
  change of units.
