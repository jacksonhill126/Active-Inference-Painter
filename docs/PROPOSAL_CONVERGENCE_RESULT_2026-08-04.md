# AI-111 Proposal-Convergence Result — 2026-08-04

## Outcome

**Negative for a proposal-invariant policy-posterior interpretation; sufficient
for an explicit candidate-set-conditional interpretation.**

The current production equation is a posterior over the finite candidate set
that happened to be enumerated:

```math
Q(\pi \mid S) = \operatorname{softmax}_{\pi \in S}
\left(\log P_{stop}(\pi) - \gamma G(\pi)\right).
```

It is not presently an importance-sampling approximation to a declared
continuous-policy posterior, because neither the hand-written nor learned
proposal density is divided out. The sampler distributions and their mixture
weights are therefore computational choices `r(pi | belief)`, not policy priors
`P(pi)`. Posterior mass may be reported only as conditional on the sampled set
and its exact candidate configuration.

The evidence below does not support enabling the learned proposal. Keep
`learned_proposal_mix = 0.0` by default. M3 must either implement a complete
base measure, policy prior, and proposal correction, or retain the explicitly
set-conditional interpretation and stop making proposal-invariant posterior
claims.

## Experiment design

The reproducible harness is `active_painter.proposal_convergence`. It ran 360
conditions:

| Variable | Values |
| --- | --- |
| Candidate count | 8, 16, 32, 64, 128 |
| Planning horizon | 1, 3, 5 |
| Sampler seed | 0–7 |
| Learned-proposal mixture | 0.0, 0.25, 0.5 |

Everything else was held fixed:

- model seed `104729`;
- model/proposal state SHA-256
  `58bacb01fc69cb62d6ad7a1a46e40f8ef404f01675747dc74ef1e27b7bf62988`;
- one randomly initialized 8×8 spatial transition ensemble;
- one deterministic partially painted material belief at coverage `0.84375`;
- policy precision `3.0`;
- fixed black tone, no passage-plan branch;
- composition disabled, so AI-110 cannot confound AI-111;
- motor forecasting excluded, so its fixed forecast budget cannot hard-zero
  most candidates;
- learned proposal evaluated at initialization on fixed zero conditioning
  features. Sampling from this fallback is allowed; training on it is not.

This is a mechanistic regression fixture, not evidence that an untrained model
or proposal produces good paintings. “Top action” means the posterior-argmax
first painting action, which is what receding-horizon control would commit to.
The normalized first-action RMS distance is evaluation-only and is never read by
the agent.

## Analytic equal-EFE control

The control fixes coverage at the declared stop-prior midpoint. The immediate
stop candidate therefore has prior mass `0.5`; every equal-EFE continuation has
prior mass `1.0`. No likelihood, preference, precision, or EFE term changes as
candidate count grows.

| Candidates | Continuations | Stop posterior mass | One continuation mass |
| ---: | ---: | ---: | ---: |
| 8 | 7 | 0.066667 | 0.133333 |
| 16 | 15 | 0.032258 | 0.064516 |
| 32 | 31 | 0.015873 | 0.031746 |
| 64 | 63 | 0.007874 | 0.015748 |
| 128 | 127 | 0.003922 | 0.007843 |

This is the candidate-frequency effect in isolation: stop mass falls by 17×
because 120 equal-prior continuation identities were added, not because the
model acquired evidence against stopping.

## Fixed spatial-fixture result

The following is the hand-written baseline (`learned_proposal_mix = 0.0`).
Distances compare the winning first action with the same seed's 128-candidate
winner. Means are over eight sampler seeds.

| Horizon | Candidates | Mean stop mass | Mean top mass | Top-family agreement with 128 | Median first-action distance to 128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.999089 | 0.999089 | 1.000 | 0.000 |
| 1 | 16 | 0.998051 | 0.998051 | 1.000 | 0.000 |
| 1 | 32 | 0.995981 | 0.995981 | 1.000 | 0.000 |
| 1 | 64 | 0.991885 | 0.991885 | 1.000 | 0.000 |
| 1 | 128 | 0.983774 | 0.983774 | 1.000 | 0.000 |
| 3 | 8 | 0.558812 | 0.558812 | 1.000 | 0.000 |
| 3 | 16 | 0.387452 | 0.387452 | 1.000 | 0.000 |
| 3 | 32 | 0.206739 | 0.206739 | 1.000 | 0.000 |
| 3 | 64 | 0.120627 | 0.120627 | 1.000 | 0.000 |
| 3 | 128 | 0.063965 | 0.063965 | 1.000 | 0.000 |
| 5 | 8 | 0.002519 | 0.607396 | 1.000 | 0.3225 |
| 5 | 16 | 0.000171 | 0.412879 | 1.000 | 0.2166 |
| 5 | 32 | 0.000113 | 0.276835 | 1.000 | 0.2135 |
| 5 | 64 | 0.000047 | 0.120578 | 1.000 | 0.2123 |
| 5 | 128 | 0.000024 | 0.064006 | 1.000 | 0.0000 |

The family label is stable within each horizon—stop at horizons 1 and 3,
mark-sequence at horizon 5—but posterior mass is not. At horizon 3, mean stop
mass falls from `0.5588` to `0.0640`. At horizon 5, doubling the budget from 64
to 128 approximately halves mean top mass (`0.1206` to `0.0640`), while the
winning first action remains separated from the 128-candidate reference by a
median normalized RMS distance of `0.2123` (maximum `0.3193`).

Across seeds at 128 candidates and horizon 5, the hand-written proposal's
median pairwise winning-action distance is `0.2948` and its maximum is `0.5357`.
Thus a stable family label does not imply a stable selected geometry.

## Learned-mixture effect at 128 candidates

The proposal is deliberately untrained in this fixture. These values measure
the initial sampling perturbation only, not learned quality.

| Horizon | Learned mix | Mean learned posterior mass | Mean signed stop-mass change vs mix 0 | Top-family agreement | Median first-action distance vs mix 0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.25 | 0.0041 | -0.000010 | 1.000 | 0.0000 |
| 1 | 0.50 | 0.0082 | -0.000041 | 1.000 | 0.0000 |
| 3 | 0.25 | 0.2054 | +0.003697 | 1.000 | 0.0000 |
| 3 | 0.50 | 0.4429 | +0.003274 | 1.000 | 0.0000 |
| 5 | 0.25 | 0.2528 | -0.000001 | 1.000 | 0.0811 |
| 5 | 0.50 | 0.5272 | -0.000000 | 1.000 | 0.2111 |

At horizon 5, a 0.5 learned mixture changes the winning first action by the same
order as doubling the candidate budget. Nothing here supports treating the
mixture as a prior or turning it on before trained-checkpoint evaluation.

## Where candidate frequency changes effective mass

The audit found these current paths:

1. `candidate_policies` creates one immediate-stop atom and N−1 continuation
   identities. Every continuation carries zero base log prior.
2. `passage_proposal_mix` changes how many identities are marks versus passages.
3. `passage_plan_proposal_mix` changes how many identities are compound plans;
   the learned proposal has no density for these.
4. `planning_horizon` changes both available families and the mark-depth/stroke-
   count support.
5. `stroke_tone_prior = None` expands one sampled geometry into black/white
   candidate alternatives.
6. `learned_proposal_mix` reallocates mark/passage slots between hand-written
   and learned samplers.
7. Continuous proposals can place several nearby identities in one region; each
   receives its own posterior entry.
8. Passage-local candidate counts similarly normalize a posterior over the
   enumerated local set, although their explicit transition prior is separate.
9. Motor alternatives are marginalized under a normalized motor prior within a
   painting policy, which avoids a direct realization-count bonus. The separate
   finite motor-forecast budget still changes which painting candidates remain
   active and must not be confused with this base-posterior experiment.

## Decision and required M3 correction

For the present prototype:

- interpret and label the result as `Q(pi | sampled candidate set S)`;
- keep proposal mixtures out of EFE, VFE, preferences, and policy priors;
- keep learned emission at zero by default;
- do not compare posterior mass across candidate configurations as if it were
  proposal invariant;
- log the full candidate count, horizon, seed, family allocation, and mixture
  beside every posterior claim.

Before M3 can claim a proposal-invariant policy posterior, it must:

1. declare a mixed discrete/continuous base measure, including the immediate-
   stop atom;
2. declare normalized `P(pi)` factors for family, depth, tone, geometry, and
   compound plans;
3. provide normalized `r(pi | belief)` densities for every sampled branch,
   including plan support and boundary atoms;
4. use a tested correction such as `log P(pi) - log r(pi | belief)` in a
   self-normalized importance approximation, with effective-sample-size and
   support diagnostics; and
5. repeat nested/frozen-candidate and independently redrawn convergence tests on
   trained, held-out model checkpoints.

Simply adding `log q_proposal` to the current posterior is prohibited: the
proposal was trained toward that posterior and would double-count it.

## Reproduction

```bash
python -m active_painter.proposal_convergence \
  --candidate-counts 8,16,32,64,128 \
  --horizons 1,3,5 \
  --seeds 0,1,2,3,4,5,6,7 \
  --learned-mixtures 0,0.25,0.5 \
  --model-seed 104729 \
  --output runs/ai111-proposal-convergence-2026-08-04-r2/proposal-convergence.json

python -m pytest -q tests/test_proposal.py tests/test_proposal_convergence.py
```

The retained local run bundle contains the initial/final experiment manifest,
resolved config, version and source hashes, append-only failure history, and raw
JSON with all 360 cells, posterior normalization checks, source/family mass, and
aggregate comparisons. The `runs/` directory is intentionally ignored; this
document records the fixed hash, command, and decision-relevant results in
version control.
