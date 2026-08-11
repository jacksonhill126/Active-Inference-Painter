# AI-107 Held-Out Transition And Uncertainty Calibration

Date: 2026-08-11
Status: AI-107 evidence task complete; M2 calibration gate failed
Claim boundary: uncalibrated simulation-only integration evidence

## Question

AI-107 asks whether the local material-transition likelihoods report useful
predictive probabilities and uncertainties on held-out sensor-posterior data.
It is an evaluation task, not a policy or painting-quality task. No metric in
this report is used as a reward, an aesthetic score, a preference, or an EFE
term.

The evaluation distinguishes four quantities that must not be conflated:

1. learned conditional/aleatoric transition variance;
2. ensemble disagreement, used as an epistemic approximation;
3. the fixed camera observation likelihood and fixed identity-transition
   variance;
4. EFE precision multipliers.

The last group is not transition noise. It is inventoried but never inserted
into an NLL, z-score, or predictive interval.

## Evidence artifacts

- accepted corpus manifest:
  `runs/corpus-ai108-combined-20260811/split_manifest.json`
- trained CNN checkpoint: `runs/ai107-calibration-20260811/cnn_combined.pt`
- trained conditional patch-cVAE checkpoint:
  `runs/ai107-calibration-20260811/cvae_combined.pt`
- fixed-condition CNN used for motor OOD:
  `runs/ai107-calibration-20260811/cnn_fixed_only.pt`
- machine-readable result:
  `runs/ai107-calibration-20260811/calibration_report.json`
- evaluator: `src/active_painter/uncertainty_calibration.py`
- focused tests: `tests/test_uncertainty_calibration.py`

The combined corpus contains 160 train, 48 validation, and 48 test
transitions, split by whole trajectory before patch extraction. Validation
and test each contain only three independent trajectories. Pixel/cell counts
are therefore not independent sample counts.

## Protocol

Both models were trained only on the 160-transition training split.

- CNN: 2,500 gradient steps, three-member ensemble.
- cVAE: 3,000 gradient steps, three members, 16-dimensional latent.
- cVAE calibration: 32 prior-latent samples per member.

The validation split fitted one scalar variance temperature

`T = mean((target - prediction)^2 / predicted_variance)`.

That scalar was frozen before test evaluation. No test result selected a
checkpoint, model capacity, stopping point, threshold, or variance scale.

The primary metrics cover all valid independent material cell-channels inside
each local patch. An action-footprint diagnostic, using action-raster channel
zero at a declared threshold of 0.01, was added after the initial primary
result to check whether unchanged background cells were hiding the failure.
It is explicitly non-gating.

The provisional M2 gates were declared before inspecting the test report:

| Metric | Gate |
| --- | ---: |
| absolute signed mean z | at most 0.20 |
| mean squared z | 0.64 to 1.44 |
| nominal 50% interval coverage | 0.40 to 0.60 |
| nominal 90% interval coverage | 0.82 to 0.96 |
| meaningful OOD epistemic ratio | at least 1.50 |
| minimum transitions for a stratum gate | 12 |

These are provisional engineering gates for M2. They are not universal
statistical laws, and AI-109 must test their stability across data size,
capacity, and seed.

## Main held-out results

### Aggregate density and moment calibration

| Model | Test mixture NLL (nats/element) | validation variance scale | signed mean z | mean z^2 | 50% coverage | 90% coverage | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CNN | -4.1597 | 0.6522 | 0.0563 | 0.6410 | 0.9786 | 0.9940 | fail |
| cVAE | -4.1129 | 0.6833 | 0.1033 | 0.6425 | 0.9387 | 0.9941 | fail |

CNN NLL is the exact ensemble Gaussian-mixture density. cVAE NLL is a finite
32-latent-samples-per-member approximation to the ensemble-by-latent Gaussian
mixture. These NLLs include the `0.5 log(2 pi)` constant and should not be
compared directly with the older trainer report, which omits that constant and
averages member NLLs instead of evaluating the ensemble mixture.

Both models fail for the same important reason. A global second moment can be
made superficially plausible, but the interval shape is wrong: the intervals
are much too broad around the majority of near-zero residuals while a small
population of larger residuals carries most squared error. A calibrated 90%
interval should contain roughly 90% of outcomes, not 99.4%.

### Action-footprint diagnostic

Restricting the calculation to cells under the selected mark footprint did
not remove the problem:

| Model | elements | signed mean z | mean z^2 | 50% coverage | 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 8,072 | 0.0573 | 0.4584 | 0.9752 | 0.9929 |
| cVAE | 8,072 | 0.1067 | 0.3321 | 0.9321 | 0.9928 |

This argues against explaining the aggregate failure solely as padding or
unchanged background domination. The local consequence distribution itself
has a strong near-zero-plus-tail structure that one Gaussian per ensemble
member does not capture well.

### Variance decomposition

Mean test variances across valid independent cell-channels were:

| Quantity | CNN | cVAE |
| --- | ---: | ---: |
| learned decoder/transition likelihood | 2.612e-4 | 1.962e-3 |
| within-member latent | 0 | 4.671e-7 |
| between-member ensemble | 1.056e-5 | 8.784e-4 |
| target camera-posterior variance | 1.765e-2 | 1.765e-2 |
| fixed camera likelihood variance | 6.411e-5 | 6.411e-5 |

The cVAE latent contributes almost nothing to prior-predictive variance after
training. Its extra uncertainty is mainly a broader decoder likelihood and
between-member disagreement. On this corpus, the cVAE does not yet provide a
better held-out density or calibration result than the simpler CNN.

Adding target posterior variance to model variance makes intervals still more
conservative. That variant is reported because the target is itself a camera
posterior mean, but it is an approximation and may double-count uncertainty
depending on the intended transition-likelihood semantics. It is not used for
the primary gate.

Per-channel second moments also expose the imbalance. CNN mean squared z was
approximately 0.030 for thickness, 0.111 for wetness, 1.186 for black pigment
mass, and 0.346 for surface tone. A single variance temperature cannot repair
such channel-dependent and non-Gaussian behavior.

## Distribution-shift result

The meaningful OOD test trained a separate CNN only on neutral Cartesian IK
and fixed +/-24-degree upper-arm-roll conditions. It compared held-out samples
from that distribution with held-out dynamic +/-32-degree upper-arm-roll
sweeps, which were absent from the checkpoint's training corpus.

- in-distribution mean ensemble variance: 9.192e-4;
- dynamic-roll OOD mean ensemble variance: 9.994e-4;
- OOD/ID ratio: 1.087;
- required ratio: 1.50;
- result: negative OOD-sensitivity evidence.

The main CNN did increase disagreement 6.59 times when the normalized amount
channel was forced to 1.5, outside its declared [0, 1] support. The cVAE ratio
was only 1.04. The former is a useful numerical stress check but cannot rescue
the meaningful gate: an invalid action tensor is not a physically meaningful
painting-policy outcome.

## Stratification and evidence gaps

The machine report stratifies test calibration by tone, surface condition,
width, length, amount, motor realization, canvas region, normalized reach, and
patch size. Strata with fewer than 12 transitions are marked ineligible for a
provisional gate rather than silently pooled.

Two requested comparisons cannot be made honestly from this corpus:

- **Wet-over-wet:** the current camera likelihood does not identify bulk
  wetness. Exact simulator wetness was not substituted as an oracle label.
- **Patch size:** every held-out patch is in the `small` bin because the live
  camera acquisition is reduced to a 16x16 inference posterior in this
  provisional throughput profile. The report records the stratum but there is
  no size contrast.

Several other test labels have low trajectory or transition counts. The
stratified numbers are diagnostics, not population estimates.

## Fixed likelihood and precision audit

The camera `base_observation_std` and `smear_observation_std` remain fixed
functional likelihood assumptions. The identity warm-up transition variance
is also fixed at `exp(local_identity_logvar)`. Neither is learned aleatoric
variance.

All eight precision-ledger entries in the trained CNN checkpoint have zero
observations. The report therefore labels each one
`declared_prior_unobserved`, not `inferred_gamma_posterior`. Their current
means reproduce declared config priors. The shadow cVAE owns no EFE precision
ledger at all.

## Decision

AI-107 is complete as an evidence task, but the evidence is a failure of the
M2 calibration gate:

- held-out transition density is finite and improved by training;
- Gaussian moment intervals are not calibrated;
- ensemble disagreement is not a reliable detector of the tested meaningful
  motor shift;
- cVAE latent use is effectively collapsed and does not yet justify replacing
  the CNN baseline;
- requested wet-over-wet and patch-size comparisons remain structurally
  unavailable.

No model from this run should be described as calibrated.

## Verification

- `python -m py_compile src/active_painter/uncertainty_calibration.py`: passed.
- Five focused calibration utility tests: passed.
- The broader conditional-cVAE, trajectory-corpus, and synthetic ensemble
  selection executed 14 tests successfully; five `tmp_path` fixtures hit the
  repository's previously observed Windows pytest temporary-directory ACL
  error before their test bodies ran.
- Those same five affected test functions were then executed directly with
  pre-created workspace-local directories: all five passed.
- `git diff --check`: passed, apart from pre-existing line-ending warnings on
  unrelated files.
- The real evaluator completed against all three trained checkpoints and
  atomically wrote the 110,527-byte machine report.

## Required next work

AI-109 should now run data-size, capacity, ensemble, and seed curves using the
same frozen evaluator. It should determine whether the interval/OOD failures
are data-limited, capacity-limited, or structural.

The leading structural likelihood hypothesis is an explicit conditional
mixture: a deposition/change occurrence distribution plus a continuous
material-change distribution, rather than one Gaussian forced to represent
both near-no-change mass and rare larger consequences. That is a generative
likelihood change, not a reward or aesthetic heuristic. It should be tested
against the current CNN and cVAE, not adopted solely because it explains this
one result.

Wet-over-wet calibration requires an evidence path that can identify wetness,
for example a calibrated temporal/appearance likelihood, before it can become
a sensor-corpus stratum. Patch-size calibration requires corpus observations
at more than one genuine inference resolution.
