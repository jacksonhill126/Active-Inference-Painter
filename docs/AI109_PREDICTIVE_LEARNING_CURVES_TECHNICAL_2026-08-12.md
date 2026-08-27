# AI-109 Predictive Learning Curves — Technical Record

Date: 2026-08-12
Task: AI-109
Status: M1 task closed 2026-08-26 on the measured material-model result;
registered visual and hierarchy continuation moved to M2
Claim boundary: uncalibrated simulation-only likelihood evidence; no shadow
model is active in painting-policy inference

## 2026-08-26 M1 scope update

This record's original 2026-08-12 conclusion correctly refused to manufacture
terminal labels. It also left AI-109 open until a new registered visual corpus
and hierarchy existed, which made the M1 gate depend on M2. The measured
27-run material-model matrix now closes AI-109's M1 capacity question as a
negative/inconclusive result. The unperformed visual work is preserved under
AI-205, AI-206, AI-208, and AI-214. See
`docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md`.

## Result

The 27-run, three-seed local transition study supports four conclusions:

1. The current Gaussian CNN is somewhat data-sensitive but not simply
   capacity-limited. Increasing the training set from 3 to 10 whole
   trajectories improved mean test negative log likelihood (NLL) by 0.159
   nats per independent material cell-channel, just beyond the declared
   cross-seed threshold. The 6-trajectory point was slightly better than the
   10-trajectory point, so the curve is not yet monotonic.
2. A wider/deeper CNN did not help. The 716,772-parameter model fit both train
   and test data worse than the 137,028-parameter baseline. This is not the
   pattern of classical overfitting; it points to optimization or architecture
   mismatch at this corpus scale.
3. The conditional patch cVAE did not materially improve held-out density,
   remained badly over-dispersed, and accumulated much larger and less stable
   eight-mark rollout error. Its prior-predictive latent variation is still
   effectively collapsed. It remains a rejected shadow candidate for the
   current local likelihood.
4. The normalized identity-plus-consequence mixture is the clear local
   likelihood-family winner. It improved unscaled exact test mixture NLL by
   1.392 nats over the base CNN and retained a low eight-mark rollout RMSE.
   It nevertheless failed calibration: after validation-only component-
   variance scaling, nominal 50% and 90% central intervals covered 90.60% and
   95.75% of test values, respectively. Its mean probability-integral-
   transform (PIT) Kolmogorov-Smirnov distance was 0.387. Better density does
   not make it an admitted generative model.

The hierarchy half of AI-109 was not trained. Every AI-108 trajectory ends in
a fixed-horizon truncation and none ends in a policy-selected `stop`.
Treating those arbitrary cutoffs as terminal paintings would teach the slow
model that a collection timeout is evidence of compositional completion. The
task therefore remains active pending a genuine-stop corpus extension.

## Probabilistic object under test

All three families estimate a local conditional transition likelihood over
the four independent camera-posterior material channels:

```text
p_theta(s'P | sP, logvar(sP), aP, b, m)
```

where `P` is a local patch, `aP` is the rasterized selected mark and motor
realization, `b` is optional compact brush-load belief, and `m` denotes the
model parameters. Coverage and contrast remain derived fields and are not
counted as independent likelihood evidence.

The new shadow family is a normalized mixture for each bootstrap member:

```text
p(s' | s, a, theta_e)
  = pi_0(s,a) N(s'; identity(s), Sigma_0(s,a))
  + pi_1(s,a) N(s'; consequence(s,a), Sigma_1(s,a))
```

The identity component is anchored to projected material persistence. The
continuous consequence mean, both component variances, and the mixture
probability are learned. This is a transition likelihood hypothesis, not a
reward, prior preference, policy prior, or EFE score.

## Corpus and split discipline

Source manifest:
`runs/corpus-ai108-combined-20260811/split_manifest.json`

- 16 independent trajectories and 256 transitions;
- train/validation/test split of 10/3/3 trajectories and 160/48/48
  transitions;
- split by whole trajectory before patch extraction;
- training fractions used nested, condition-diverse subsets of 3, 6, and 10
  trajectories;
- validation and test trajectories remained fixed for every run;
- validation fit one scalar component-variance temperature only;
- the test split was used only for final likelihood, interval, and rollout
  evaluation;
- process material truth was not used as model input or rollout
  initialization.

The nested subset ordering uses only camera-posterior/action condition labels.
It greedily gives rare conditions early representation while preserving
whole-trajectory isolation.

## Experiment matrix

Seeds were 109, 211, and 307.

- CNN data curve: 30%, 60%, and 100% of training trajectories. Gradient steps
  scaled with the fraction: 450, 900, and 1,500.
- CNN capacity curve at full data: 16 channels/1 residual block, 32/2, and
  64/3, each with three ensemble members.
- CNN ensemble curve at full data: 1, 3, and 5 members.
- cVAE family comparison: 32 channels, 2 residual blocks, 16 latent
  dimensions, 3 members, 2,000 gradient steps.
- identity/consequence mixture comparison: 32 channels, 2 residual blocks,
  3 members, 1,500 gradient steps.

The base full-data CNN is reused as the base-capacity and three-member points;
there are 27 unique trained runs, excluding those aliases.

The predeclared material-change rule was:

```text
test-NLL improvement >= max(0.05 nats, either compared cross-seed standard deviation)
```

Calibration retained the AI-107 frozen interval gates. NLL improvement alone
cannot admit a model.

## Cross-seed aggregate results

NLL is unscaled predictive-mixture NLL in nats per independent material
cell-channel. `C90` uses the exact predictive-mixture CDF after a scalar
variance temperature is fit on validation only. Rollout values are test RMSE
inside the cumulative action footprint, recursively predicting from a
camera-posterior state for 1, 2, 4, or 8 marks.

| Family/axis | Level | Train trajectories | Parameters | Train NLL | Validation NLL | Test NLL | Test NLL SD | C90 | H1 RMSE | H2 RMSE | H4 RMSE | H8 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN data | small | 3 | 137,028 | -4.1033 | -3.6127 | -3.8416 | 0.1196 | 0.9920 | 0.0301 | 0.0425 | 0.0581 | 0.0767 |
| CNN data | medium | 6 | 137,028 | -4.2122 | -3.8193 | -4.0435 | 0.0804 | 0.9927 | 0.0242 | 0.0327 | 0.0465 | 0.0712 |
| CNN data | full | 10 | 137,028 | -4.0807 | -3.9201 | -4.0005 | 0.0977 | 0.9837 | 0.0512 | 0.0678 | 0.0781 | 0.0870 |
| CNN capacity | small | 10 | 26,964 | -4.1584 | -3.5257 | -4.0505 | 0.0446 | 0.9948 | 0.0238 | 0.0322 | 0.0451 | 0.0682 |
| CNN capacity | base | 10 | 137,028 | -4.0807 | -3.9201 | -4.0005 | 0.0977 | 0.9837 | 0.0512 | 0.0678 | 0.0781 | 0.0870 |
| CNN capacity | large | 10 | 716,772 | -3.6104 | -3.4869 | -3.5703 | 0.2220 | 0.9679 | 0.0739 | 0.0790 | 0.0818 | 0.0878 |
| CNN ensemble | one | 10 | 45,676 | -4.2490 | -2.9280 | -3.6850 | 0.1918 | 0.9929 | 0.0245 | 0.0329 | 0.0462 | 0.0690 |
| CNN ensemble | three | 10 | 137,028 | -4.0807 | -3.9201 | -4.0005 | 0.0977 | 0.9837 | 0.0512 | 0.0678 | 0.0781 | 0.0870 |
| CNN ensemble | five | 10 | 228,380 | -4.1815 | -3.9495 | -4.1354 | 0.0459 | 0.9919 | 0.0291 | 0.0405 | 0.0549 | 0.0744 |
| cVAE family | full | 10 | 310,020 | -4.1011 | -3.8379 | -4.0573 | 0.0129 | 0.9936 | 0.0320 | 0.0952 | 0.1450 | 0.2250 |
| mixture family | full | 10 | 147,432 | -5.3546 | -5.2441 | -5.3920 | 0.0181 | 0.9575 | 0.0267 | 0.0373 | 0.0524 | 0.0738 |

The full-data CNN's worse rollout result is cross-seed behavior, not a claim
that more evidence is intrinsically harmful. With only ten training and three
test trajectories, different condition mixtures can dominate a nominal
fraction point.

## Exact mixture calibration correction

The original AI-107 evaluator summarized every family by its first two
moments and evaluated Gaussian intervals. That is not correct for an ensemble,
a finite latent mixture, or the explicit two-component likelihood.

AI-109 calibration report v2 therefore evaluates the full predictive
distribution:

- CNN: equal mixture over ensemble Gaussians;
- cVAE: equal finite mixture over ensemble members and prior latent samples;
- explicit mixture: ensemble mixture over each member's learned identity and
  consequence components.

For observation `y`, calibration uses the full mixture CDF `F(y)`. A central
`alpha` interval contains `y` when its PIT lies within
`[(1-alpha)/2, (1+alpha)/2]`. The validation-only scalar is selected by exact
mixture NLL, not moment-Gaussian residual energy.

A regression test now pins component/channel/cell ordering after a NumPy
advanced-indexing defect was found during this correction. For the probe run,
the reconstructed and model-native exact NLLs agree within approximately
`1e-7` nats per element.

### Family calibration shape

| Family | Mean variance scale | Scaled C50 | Scaled C90 | Scaled C95 | Mean PIT KS | Seeds passing gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base CNN | 0.3364 | 0.9510 | 0.9837 | 0.9887 | 0.4519 | 0/3 |
| cVAE | 0.6808 | 0.9716 | 0.9936 | 0.9952 | 0.4111 | 0/3 |
| identity/consequence mixture | 0.0248 | 0.9060 | 0.9575 | 0.9685 | 0.3869 | 0/3 |

The explicit mixture materially improves distribution shape relative to the
other candidates, but the nominal 50% interval still covers more than 90% of
test observations. The small mean variance scale indicates that the learned
component variances are much too large before temperature correction. The
remaining PIT concentration after that correction shows that one global scale
cannot repair component mass and shape.

## Resource observations

| Family/level | Mean training seconds | Peak CUDA memory | Parameters |
| --- | ---: | ---: | ---: |
| CNN small-data | 7.53 | 50.1 MB | 137,028 |
| CNN medium-data | 15.93 | 50.1 MB | 137,028 |
| CNN full/base | 73.47 | 50.1 MB | 137,028 |
| CNN small-capacity | 65.09 | 29.9 MB | 26,964 |
| CNN large-capacity | 52.08 | 108.0 MB | 716,772 |
| CNN one-member | 17.97 | 25.4 MB | 45,676 |
| CNN five-member | 55.23 | 75.2 MB | 228,380 |
| cVAE | 118.17 | 86.6 MB | 310,020 |
| identity/consequence mixture | 88.39 | 45.4 MB | 147,432 |

These are local observations on Windows 11, Python 3.14, PyTorch 2.11 CUDA,
not stable hardware benchmarks. The complete first training/evaluation pass
took about 2,410 seconds. Recomputing the corrected exact-mixture calibration
and multistep reports from saved checkpoints took about 1,032 seconds.

## Diagnosis against AI-109 acceptance

### Data limitation

Partly supported. Full data materially improves test NLL over the smallest
subset, but the medium point is best and the independent sample count remains
small. More trajectories are justified, especially before interpreting
condition-specific or OOD curves.

### Capacity limitation

Not supported. More width/depth degrades both training and held-out density.
Increasing generic CNN size is not the next rational intervention.

### Classical overfitting

Not supported by the declared rule. The large model does not improve training
NLL while worsening test NLL; it fits both worse. The single-member ensemble
has a large train/test gap, while five members materially improve held-out
density.

### Likelihood-family misspecification

Strongly supported. The explicit normalized identity/consequence family
produces the largest, most stable NLL improvement and competitive multistep
rollouts. Its uncertainty parameterization remains misspecified or
insufficiently trained.

### cVAE latent necessity

Not supported **for the declared coarse material-posterior target**. The
cVAE's 0.057-nat mean NLL improvement is below the 0.098-nat seed-variation
threshold, and its multistep error is substantially worse. A latent variable
is not justified for that target merely because it can represent multimodality
in principle.

This experiment did not retain or predict registered camera-image patches. It
therefore provides no evidence for or against the owner's proposed
action-conditioned visual mark-consequence VAE. See
`VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

### Hierarchy/composition

Not testable with the accepted corpus. No hierarchy result may be inferred
from local raster cells or fixed-horizon endpoints.

## Decision and required next work

1. Retain `conditional-local-material-transition-mixture-v0` as the leading
   offline/shadow local likelihood candidate. Do not load it into live EFE or
   describe it as calibrated.
2. Diagnose the mixture's probability mass, not just its mean: report
   component responsibilities and calibration inside/outside the selected
   action footprint, then test whether an explicit spike/hurdle identity event
   or a correlated bounded consequence family is required. Keep every option
   normalized.
3. Collect a trajectory-isolated corpus whose episodes terminate through an
   actually selected `stop`, with immediate-stop support and termination
   provenance preserved. Never relabel a fixed horizon as a stop.
4. Run the hierarchy data/capacity/seed curve only on those genuine terminal
   paintings, with slower temporal targets and fixed held-out episodes.
5. Expand independent trajectory count before making condition-specific or
   OOD uncertainty claims. Three test trajectories are the experimental unit;
   49,152 correlated raster elements are not 49,152 independent trials.

## Verification

The project-wide post-change regression exercised all 520 collected test
cases. The main invocation passed 514 in 499.30 seconds; six fixtures could not
start because the execution sandbox denied pytest's Windows temporary
directory. The same six cases passed in 6.75 seconds when rerun with an
accessible temporary directory, so no test assertion failed. Python source
compilation and `git diff --check` also passed. The focused probabilistic tests
cover exact mixture density reconstruction, full-mixture CDF calibration,
recursive horizons, nested trajectory subsets, resumability, and aggregate
diagnosis.

## Artifacts

- experiment runner: `src/active_painter/learning_curves.py`;
- normalized shadow mixture: `src/active_painter/mixture_transition.py`;
- exact predictive-mixture evaluator and recursive rollouts:
  `src/active_painter/uncertainty_calibration.py`;
- machine aggregate:
  `runs/ai109-learning-curves-20260812/learning_curve_report.json`;
- per-run checkpoints, training reports, calibration reports, and logs:
  `runs/ai109-learning-curves-20260812/runs/`;
- focused tests: `tests/test_learning_curves.py`,
  `tests/test_mixture_transition.py`, and
  `tests/test_uncertainty_calibration.py`.
