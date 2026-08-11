# AI-106 Terminal Coverage And Stopping Acceptance

Date: 2026-08-11
Decision: AI-106 complete with a negative M2 forecast-family decision
Scope: terminal material-coverage preference, predicted terminal-coverage
family, and immediate-stop policy prior

## Decision

Retain both of the following as declared priors:

1. the Beta prior preference over **terminal material coverage**, centered at
   0.87 with concentration 110 and evaluated only when a candidate reaches
   `stop`; and
2. the separate finite sigmoid policy prior that makes immediate stopping less
   probable below 0.70 believed coverage without ever removing it from support.

Do **not** approve the current single moment-matched Beta forecast family for
M2. It is acceptable only as the named provisional integration approximation
already in the M1-era runtime. Before M2 acceptance, terminal coverage risk must
be computed from forecast particles/samples or from a calibrated bounded family
that preserves consequential shape or multimodality. This replacement belongs
inside the outcome forecast/likelihood and EFE risk calculation; it does not
change the terminal preference into a learned target.

The decision is negative but AI-106 is complete: the task asked whether to
approve or replace the family, and the evidence requires replacement.

## What was tested

The production approximation turns a predicted coverage mean and variance into
a Beta distribution and computes the exact Beta--Beta KL to the preferred
terminal density. Tests were added in
`tests/test_terminal_coverage_acceptance.py` using two independent checks:

- direct Monte Carlo samples from bounded Beta forecasts, compared with the
  production analytic KL; and
- deterministic Gauss--Hermite integration of a bounded logit-normal forecast,
  whose measured mean and variance are passed to the production Beta moment
  match before comparing the two terminal risks.

The second check asks the scientifically important question: is mean and
variance enough to preserve terminal risk when the true bounded forecast is not
Beta?

## Direct Monte Carlo agreement when the family is correct

All risks are nats. `SE` is the standard error of the sample estimate over
120,000 fixed-seed draws.

| Variance regime | Mean | Variance | Analytic Beta KL | Direct MC KL | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 0.87 | 0.0002 | 0.419011 | 0.416439 | 0.001651 |
| medium | 0.80 | 0.0100 | 3.342295 | 3.336242 | 0.017667 |
| high | 0.65 | 0.0500 | 24.200555 | 24.113650 | 0.101040 |

The analytic implementation is therefore internally correct. The M2 problem
is not an algebra or sign error; it is loss of distributional information when
the predicted outcome is summarized by only two moments and forced into one
Beta shape.

## Alternative bounded-family comparison

The alternative is a logit-normal coverage forecast centered at logit(0.8).
Its moments and KL were integrated independently with 96-point Gauss--Hermite
quadrature. The runtime Beta column uses exactly those independently measured
moments and the configured concentration floor of 1.0.

| Regime | Logit std | Measured mean | Measured variance | Runtime Beta KL | Logit-normal KL | Beta minus alternative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 0.20 | 0.798098 | 0.001032 | 2.070451 | 2.074265 | -0.003814 |
| medium | 0.65 | 0.781565 | 0.011305 | 4.371640 | 4.261075 | +0.110565 |
| high | 2.50 | 0.675474 | 0.101338 | 23.244241 | 46.853089 | **-23.608848** |

At low and medium uncertainty, the approximation is close. At broad
uncertainty, two forecasts with the same first two moments differ by 23.6 nats.
The current family substantially understates how incompatible the broad
logit-normal outcome is with the narrow terminal preference. A calibrated M2
model cannot assume this error away.

## Clamp and boundary effects

The runtime applies four consequential restrictions: coverage means are
clamped to `[1e-4, 1 - 1e-4]`, variances to the bounded Bernoulli maximum and a
`1e-8` minimum, total concentration to `[2, 1e6]`, and then both Beta
concentrations are raised to at least 1 by a common scale factor.

For a near-blank forecast with mean `1e-4` and variance
`(2.29e-3)^2 = 5.2441e-6`:

| Family | alpha | beta | represented variance | terminal KL |
| --- | ---: | ---: | ---: | ---: |
| unrestricted moment match | 0.001807 | 18.065335 | 5.2441e-6 | 53,248.178 |
| runtime interior-unimodal floor | 1.000000 | 9,999.000000 | 9.9980e-9 | 892.250 |

The floor preserves the mean, prevents a digamma singularity, and makes the EFE
numerically usable. It also reduces represented variance by more than 500 times
and changes risk by more than 52,000 nats. Near full coverage the symmetric
restriction changes 7,913.972 nats to 95.555 nats. These are model changes, not
neutral numerical clamps.

Other saturation effects are also explicit:

- variances above the admissible bound at mean 0.5 collapse to the same uniform
  `Beta(1, 1)` forecast; and
- input variances below the minimum/concentration ceiling at mean 0.87 collapse
  to total concentration `1e6`.

The stabilizer remains useful in the provisional runtime because the
unrestricted boundary-spiked Beta is worse: its extreme KL is an artifact of a
family the compact belief does not justify. The evidence nevertheless rules out
calling the stabilized family calibrated or M2-ready.

## Terminal-only and stopping support audit

The control structure passes:

- `Policy` rejects a stop anywhere except the terminal position and requires
  every candidate to end in stop.
- `ExpectedFreeEnergy` accumulates transition/observation terms only over
  non-stop actions, then evaluates terminal coverage once from the final
  predicted state.
- `PolicySampler.sample` inserts `Policy((stop,))` before every learned or
  hand-written continuation branch. It has no learned-proposal record and stays
  present at zero and nonzero learned-proposal mixtures.
- `policy_stop_log_prior` is finite at every tested coverage. It changes prior
  probability but never vetoes immediate stop. Continuations receive zero from
  this factor.
- `tests/test_stop_policy.py`, `tests/test_stop_prior.py`,
  `tests/test_policies.py`, `tests/test_proposal.py`, and the new acceptance
  suite cover these claims.

The terminal preference and stop prior are configuration-owned prior beliefs,
not discoveries learned from paintings. Precision learning can reweight a
modality but cannot change the preferred Beta parameters; the existing
`tests/test_preferences.py` tripwire remains in force.

## M2 replacement requirement

The preferred replacement order is:

1. use the terminal-coverage samples already produced by calibrated
   counterfactual particles to estimate the risk against the fixed preferred
   density, with an explicit entropy/density estimator and convergence test; or
2. if particle cost is prohibitive, fit and validate a bounded mixture or
   logistic-normal family on held-out terminal forecasts, including boundary
   mass and multimodality.

The replacement must preserve separate logging of terminal risk, terminal
entropy, and expected log preference, and must pass sample-count convergence,
boundary, calibration, and zero/preference ablations. A Gaussian clipped to
`[0, 1]` without explicit atoms is not an acceptable silent substitute.

## Reproduction

Focused acceptance command:

```powershell
python -m pytest tests/test_terminal_coverage_acceptance.py tests/test_stop_policy.py tests/test_stop_prior.py tests/test_policies.py tests/test_preferences.py -q
```

Result on 2026-08-11: `48 passed in 3.11s`.
