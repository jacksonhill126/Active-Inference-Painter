# Chapter 3 — concept map

Chapter 3 stitches *learning* and *inference* together. The previous chapter
treated the agent's parameters as already known; here we look at how they get
estimated in the first place, and what happens when both the parameters and
the hidden state are unknown at the same time.

## The five recipes

Each is exposed as a configurable building block in `active_inference` and
demonstrated by a thin orchestrator in `chapters/chapter_03/`.

| Recipe | Key API | Orchestrator |
|--------|---------|--------------|
| Closed-form linear regression (MLE) | `mle_linear_regression(X, y)` | `example_3_1_linear_regression_mle.py` |
| Gradient descent over the squared loss | `gd_linear_regression(X, y, learning_rate=...)` | `example_3_2_linear_regression_gd.py` |
| Multiple regression (vectorized) | `mle_linear_regression(X, y)` (batched) | `example_3_3_multiple_regression.py` |
| Multivariate normal anatomy | `mvn_pdf`, `mvn_sample`, `confidence_ellipse` | `example_3_4_multivariate_gaussian.py` |
| Bayesian linear regression | `BayesianLinearRegression(...).fit(X, y)` | `example_3_5_bayesian_linear_regression.py` |
| Linear Gaussian System (sensor fusion) | `LinearGaussianSystem(...).posterior_batch(Y)` | `example_3_6_lgs_food_localization.py` |
| Factor analysis via EM | `fit_factor_analysis(Y, n_factors=...)` | `example_3_7_factor_analysis_em.py` |

## Script inventory

| File | Role |
|---|---|
| `animation_bimodal_emergence.py` | Non-injective generator animation. |
| `animation_blr_predictive_band.py` | Bayesian linear-regression predictive-band animation. |
| `animation_blr_tightening.py` | Parameter posterior tightening animation. |
| `animation_em_convergence.py` | EM log-likelihood and loadings animation. |
| `animation_em_steps.py` | E-step / M-step factor-analysis animation. |
| `animation_lgs_online.py` | Online Linear Gaussian System posterior animation. |
| `animation_precision_sweep.py` | Prior/likelihood precision sweep animation. |
| `animation_sufficient_statistics.py` | Running sufficient-statistics animation. |
| `visualize_calibration.py` | Calibration reliability diagram. |
| `visualize_coverage.py` | Empirical coverage of a fixed 95% LGS credible region across a sample-size (N) sweep. |
| `visualize_posterior_predictive.py` | Posterior-predictive diagnostic figure. |
| `interactive_bayesian_regression.py` | GUI / web-launchable BLR explorer: `N` and prior-precision sliders tighten the ±2σ predictive band; readout reports recovered `β0`/`β1` ± posterior std. |
| `interactive_lgs_localization.py` | GUI / web-launchable 2-D LGS explorer: `(y1, y2)` observation sliders slide the posterior mean ellipse toward the fixed prior; readout reports posterior mean/std and distance from prior/observation. |

## Reusable building blocks

* **`active_inference.core.distributions`** — `mvn_pdf`, `mvn_log_pdf`,
  `mvn_sample`, `mahalanobis_squared`, `isotropic_cov`, `diagonal_cov`. All
  multivariate helpers solve a linear system against the covariance matrix
  (Cholesky for `mvn_log_pdf`/`mvn_sample`, `np.linalg.solve` for
  `mahalanobis_squared`) rather than forming an explicit matrix inverse.
* **`active_inference.core.lgs`** — `LinearGaussianSystem` with
  `posterior(y)` and `posterior_batch(Y)` methods returning a
  `LGSPosterior` (mean, covariance, std, precision).
* **`active_inference.estimators.linear_regression`** —
  `mle_linear_regression`, `gd_linear_regression`,
  `BayesianLinearRegression` (with `fit`, `fit_sequential`, and posterior
  predictive helpers).
* **`active_inference.estimators.em`** — `fit_factor_analysis`,
  plus the individual `factor_analysis_e_step` and
  `factor_analysis_m_step` helpers for didactic walk-throughs.
* **`active_inference.visualizations.animations`** — drop-in helpers for
  `animate_2d_posterior`, `animate_em_convergence`, etc.
  No FFmpeg dependency: GIFs are written via the bundled pillow writer.

## See also

- [`../topics/learning_and_inference.md`](../topics/learning_and_inference.md) —
  the MLE / MAP / Bayesian / EM decision this chapter's Examples 3.1–3.7
  exercise end to end.
- [`../topics/bayesian_inference.md`](../topics/bayesian_inference.md) — the
  prior/likelihood/posterior/evidence machinery behind
  `BayesianLinearRegression` and `LinearGaussianSystem`.
- [`../topics/multivariate_gaussians.md`](../topics/multivariate_gaussians.md) —
  the Cholesky-based MVN density/sampling routines exercised in Example 3.4
  and the LGS posteriors of Example 3.6.
- [`../topics/gradient_descent.md`](../topics/gradient_descent.md) — the
  iterative alternative to the closed-form estimator, demonstrated in
  Example 3.2.

## Where the book takes this next

Chapter 4 generalizes the closed-form posteriors here to settings where the
posterior is no longer Gaussian and replaces the EM machinery with
variational Bayesian inference. The classes here are designed so that those
upgrades plug in naturally — `LinearGaussianMVModel` already exposes
`log_likelihood` and `log_prior`, which is the interface a variational scheme
needs.
