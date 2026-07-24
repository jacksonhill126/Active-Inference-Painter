# Chapter 2 — concept map

Chapter 2 turns the conceptual machinery of Chapter 1 into a concrete
modelling pattern: define a generative process, define a generative model,
choose a grid, infer. The chapter then explores the joint distribution from
several angles and introduces the closed-form and iterative point-estimate
alternatives (MLE / MAP).

## The standard recipe

Every example in this chapter follows the same five steps; the `chapters/`
scripts simply parameterize them differently.

1. **Experimental setting** — pick a domain for the hidden state and a domain
   for the observations.
2. **Generative process** — `LinearGaussianProcess(beta0, beta1, sigma2_y)`,
   optionally with a `nonlinear` ``psi``.
3. **Generative model** — `LinearGaussianModel(...)` with a Gaussian or
   uniform prior over the hidden state.
4. **Inference** — `GridBayesianInference(model, grid).infer(y_obs)`.
5. **Diagnostics / visualizations** — posterior mode, mean, credible
   interval, and the three-panel prior / likelihood / posterior plot.

## What each example explores

| File                                 | Question it answers | Key API |
|--------------------------------------|----------------------|---------|
| `example_2_1_linear_deterministic.py` | What does Bayes do in the noiseless limit? (Answer: it inverts ``g``.) | `LinearGaussianModel(prior_kind="uniform")`, `GridBayesianInference` |
| `example_2_2_linear_probabilistic.py` | How does a Gaussian prior pull the posterior away from the data? | `LinearGaussianProcess`, `LinearGaussianModel`, `GridBayesianInference`, `map_analytic_linear` |
| `example_2_3_precision.py`            | How does the posterior interpolate between the prior and the likelihood as their precisions change? | `LinearGaussianModel`, `GridBayesianInference`, `plot_precision_comparison` |
| `example_2_4_nonlinear_deterministic.py` | What goes wrong with non-injective generators? | `LinearGaussianModel` (quadratic `psi`), `GridBayesianInference`, `plot_generating_function` |
| `example_2_5_nonlinear_probabilistic.py` | Does a localized prior resolve the ambiguity? | `LinearGaussianProcess`, `LinearGaussianModel` (quadratic `psi`), `GridBayesianInference` |
| `example_2_6_imperfect_model.py`      | What happens when the agent's model differs from the true process? | `LinearGaussianProcess(nonlinear=psi)` (true process), `LinearGaussianModel` (agent's linear model), `GridBayesianInference` |
| `example_2_7_multiple_samples.py`     | How does the posterior tighten as N grows, and is sequential = batch? | `orchestrator_workflows.build_example_2_7_multiple_samples` |
| `example_2_8_mle_analytic.py`         | What is the closed-form MLE under linear-Gaussian assumptions? | `mle_analytic_linear`, `mle_loss`, `oracle_agreement`, `GridBayesianInference` |
| `example_2_9_map_analytic.py`         | What is the closed-form MAP, and how does it move with `s2_x`? | `map_analytic_linear`, `mle_analytic_linear`, `map_loss`, `oracle_agreement`, `GridBayesianInference` |
| `example_2_10_gradient_descent.py`    | Can we recover the same answers iteratively? | `orchestrator_workflows.build_example_2_10_gradient_descent` |
| `visualize_generative_model.py`       | What does the joint `p(x, y)` look like as a heatmap and surface? | `GridBayesianInference.joint_density`, `plot_joint_heatmap` |
| `animation_sequential.py` — **animation (GIF)** | How does the posterior tighten as observations accumulate? | `animate_sequential_posterior`, `save_animation` |
| `animation_gradient_descent.py` — **animation (GIF)** | How do the MLE/MAP iterates roll down the objective? | `gradient_descent`, `mle_grad_x`/`mle_loss`, `animate_gradient_descent`, `save_animation` |
| `interactive_explorer.py` — **interactive (GUI, `--mode full\|precision`)** | Drag-and-drop exploration with sliders (`full`); a single precision-ratio slider (`precision`). | `interactive_inference`, `interactive_precision` |
| `interactive_gradient_descent.py` — **Interactive** (GUI / web-launchable) | How does learning rate reshape the descent trajectory, and can we freeze on any step? | `interactive_gradient_descent` |

## Reusable building blocks

* `active_inference.LinearGaussianProcess` — a configurable generative process.
* `active_inference.LinearGaussianModel` — a configurable generative model.
* `active_inference.GridBayesianInference` — one-line exact inference.
* `active_inference.gradient_descent` — drop-in replacement for analytic
  inversion when you want to study optimization dynamics.

## See also

- [`../topics/bayesian_inference.md`](../topics/bayesian_inference.md) — the
  prior/likelihood/posterior/evidence machinery every example in this
  chapter exercises via `GridBayesianInference`.
- [`../topics/generative_models.md`](../topics/generative_models.md) — the
  process/model split (`LinearGaussianProcess` vs `LinearGaussianModel`)
  used throughout the recipe.
- [`../topics/inverse_problem.md`](../topics/inverse_problem.md) — the
  non-injective-generator ambiguity explored in Examples 2.4 and 2.5.
- [`../topics/gradient_descent.md`](../topics/gradient_descent.md) — the
  iterative MLE/MAP alternative to the closed-form estimators, demonstrated
  in Example 2.10 and `animation_gradient_descent.py`.

## Where the book takes this next

Chapter 3 introduces parameter learning (MLE and Bayesian linear regression)
and Chapter 4 generalizes the entire setup with variational Bayes. The
classes here are designed so that those upgrades fit naturally — for example,
`LinearGaussianModel` already exposes both `log_likelihood` and `log_prior`,
which is exactly the interface a variational scheme needs.
