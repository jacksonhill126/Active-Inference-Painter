# AI-104 / AI-105 Reference-Model Acceptance

Date: 2026-08-04
Decision: `AI-104 Done`; `AI-105 Done`

This decision accepts the current VFE and EFE arithmetic against tractable,
independent references. It does not accept the sensor model, predictive
calibration, composition preference, finite-proposal semantics, or the M1
milestone as a whole.

## AI-104: Variational Free Energy

`tests/test_reference_oracle.py` now covers the complete acceptance checklist:

- a scalar conjugate Gaussian fixture;
- a three-dimensional diagonal conjugate Gaussian fixture;
- posterior mean and variance, Gaussian complexity, expected negative log
  likelihood, and total VFE;
- an independent fine-grid spatial VFE integration;
- likelihood/transition precision ratios spanning four orders of magnitude;
- the identity transition prior outside the active local stroke patch; and
- the analytic posterior as the unique VFE minimum against mean and variance
  perturbations.

The summary likelihood is heteroscedastic in production, so its reported
expected log likelihood remains an explicitly labeled Monte Carlo
approximation. The reporting-only budget is now the declared
`summary_vfe_report_samples=4096`, replacing an implicit 32 samples. Across
seeds 0 through 4 in the scalar fixture, the measured errors against independent
fine-grid integration were `+0.00613`, `-0.02316`, `-0.01272`, `-0.00398`, and
`+0.01796` nats: maximum absolute error `0.02317` nats and mean error
`-0.00315` nats. The accepted per-seed band is `+-0.05` nats.

This larger budget is used only after optimization to populate the VFE
diagnostic. Its draw runs inside an RNG-state fork, so it does not alter the
posterior optimizer, later stochastic learning, EFE, policy priors, preferences,
or policy posterior.

The spatial estimator remains an explicitly labeled diagonal-Gaussian pixel
approximation with transition moments evaluated at the previous posterior
mean. Its complexity, expected negative log likelihood, and total continue to
match independent fine-grid integration within `1e-6`.

## AI-105: Expected Free Energy

The harness now includes a two-state enumerated acceptance matrix:

| Control | Held fixed | Isolated result |
| --- | --- | --- |
| Deterministic and preferred | deterministic likelihood and matching preference | zero risk and zero ambiguity |
| Pure ambiguity | uniform outcome and uniform preference | `G = log(2)` from ambiguity |
| Pure epistemic contrast | the same uniform predicted outcome and preference | the deterministic likelihood gains `log(2)` nats of information relative to the uniform likelihood |
| Preference dominated | deterministic likelihood | policy ordering comes only from risk |

The production policy posterior matches the independent full posterior both
with a flat policy prior and with an opposing nonuniform policy prior. Existing
reference cases additionally establish:

- terminal coverage risk as a full KL, including forecast entropy, within the
  declared coverage band;
- `transition_risk + transition_ambiguity = -I(theta; s_next)` without a second
  epistemic subtraction;
- motor pragmatic risk minus state/observation information gain and reliability
  parameter novelty, with logged motor ambiguity excluded from the total to
  avoid double counting; and
- identical total assembly at all summary and spatial motor-EFE call sites.

## Explicit Deferrals

The acceptance criterion permits a production EFE term without a defensible
reference counterpart to be derived, renamed, disabled, or explicitly
deferred. Two boundaries remain visible:

- The terminal Beta/discrete oracle is certified only for coverage means in
  `[0.70, 0.90]`. Clamp behavior, low-coverage risk, alternatives, terminal-only
  application, and stopping behavior are the now-ready `AI-106` task.
- The online-trained compression-gap preference is an unnormalized structural
  energy with a self-referential training/evaluation loop. Its arithmetic is
  attributed and tested, but its retention as a preference is not accepted;
  that decision remains explicitly deferred to `AI-110`.

AI-111's negative convergence result is unchanged: the policy posterior is
still conditional on the sampled candidate set, and the learned-proposal mix
remains zero pending an M3 correction.

## Verification

- `tests/test_reference_oracle.py`: 20 passed.
- Reference, observation-factorization, local-spatial, spatial-state, and
  proposal focused group: 76 passed.
- Deterministic CI gate: 164 passed with two expected obsolete-summary warnings.
- Arm-driver and web-runtime integration group: 74 passed with expected
  obsolete-summary warnings.
- Repository collection after this change: 442 tests.
- Python source compilation passed.
