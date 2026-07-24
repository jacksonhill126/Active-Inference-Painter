# Chapter 1 — The Hypothesis-Testing Brain

Chapter 1 of the book sets up the conceptual frame: an agent embedded in an
environment to which it has no direct access. It must reconstruct what is
happening "out there" purely from a stream of noisy sensor readings. The
chapter introduces this through Bayes' theorem — see
[`../topics/bayesian_inference.md`](../topics/bayesian_inference.md) for the
concept-level walkthrough of the four ingredients (prior, likelihood,
evidence, posterior) — and through the generative-process / generative-model
split covered in
[`../topics/generative_models.md`](../topics/generative_models.md).

## The recipe (introduced here, formalized in Chapter 2)

Chapter 1 previews the five-step modeling pattern that Chapter 2 turns into a
systematic recipe. Every `chapters/chapter_01/` script that runs inference
parameterizes it the same way:

1. **Experimental setting** — a single hidden state `x` inside a sealed box;
   the agent never observes it directly, only a stream of noisy readings `y`.
2. **Generative process** — `LinearGaussianProcess(beta0=3.0, beta1=2.0,
   sigma2_y=...)`, the noise-free generator `y = β0 + β1·x` plus Gaussian
   observation noise; `04_inverse_problem.py` and
   `interactive_inverse_problem.py` swap in a non-injective `psi=lambda x: x
   ** 2` to make the generator quadratic.
3. **Generative model** — `LinearGaussianModel(...)` with either a Gaussian
   prior (`02_three_perspectives.py`, `03_bayes_intuition.py`) or a uniform
   prior over a symmetric domain (`04_inverse_problem.py`,
   `interactive_inverse_problem.py`).
4. **Inference** — `GridBayesianInference(model, x_grid).infer(y_obs)`, exact
   1-D grid inference (or its sequential, log-space form in
   `05_belief_from_stream.py`).
5. **Diagnostics / visualizations** — the prior/likelihood/posterior
   triptych (`plot_prior_likelihood_posterior`), the streaming belief
   animation, and the interactive slider explorer.

## Scripts

| Script | Question it answers | Library imports | What it shows |
|---|---|---|---|
| `01_box_scenario.py` | What does the agent actually receive if it is sealed inside the box? | `LinearGaussianProcess`, `get_logger`, `plot_generating_function`, `save_or_show` | The noise-free generator, the sampled stream at a fixed hidden state, and a histogram the agent must invert to recover that state. |
| `02_three_perspectives.py` | What does "modeling" mean under three different vocabularies? | `LinearGaussianModel`, `LinearGaussianProcess`, `GridBayesianInference`, `make_grid`, `get_logger` | The same toy environment rendered as scientific-fit, hypothesis-testing (prediction/error), and statistical (posterior) panels side by side. |
| `03_bayes_intuition.py` | What do the four ingredients of Bayes' theorem look like on a real toy, one factor at a time? | `GridBayesianInference`, `LinearGaussianModel`, `make_grid`, `plot_prior_likelihood_posterior` | Prior/likelihood/posterior overlays, plus the evidence `p(y)` that normalizes the unnormalized posterior. |
| `04_inverse_problem.py` | What happens to the posterior when the generator is non-injective? | `GridBayesianInference`, `LinearGaussianModel`, `make_grid`, `plot_generating_function`, `plot_prior_likelihood_posterior` | A quadratic generator `y = β0 + β1·x²` with a uniform prior over `[-2.5, 2.5]` produces a bi-modal posterior with two annotated modes. |
| `05_belief_from_stream.py` | **Animation** (GIF). How does belief sharpen as evidence accumulates, and does it converge to the right answer? | `LinearGaussianModel`, `LinearGaussianProcess`, `make_grid`, `mle_analytic_linear`, `oracle_agreement`, `animate_stream_belief`, `save_animation` | Sequential log-space posterior updates alongside the arriving stream; the final mode is cross-checked against the closed-form MLE (`oracle_agreement`), and the `σ_n·√N` concentration statistic is logged as an (approximately constant) range over `N`. |
| `interactive_inverse_problem.py` | **Interactive** (GUI / web-launchable). How do `y` and `σ_y²` reshape the bi-modal posterior in real time? | `active_inference.visualizations.interactive_inverse_problem` | Sliders for the observation and likelihood variance drive live readouts of both posterior modes, their separation, and the posterior entropy. Launch from the local web interface (`./run.sh --web`) or run the script directly (opens a GUI window; no `--save` path). |

## Going further

* Try editing `--x-true` and `--n-samples` on `01_box_scenario.py` to see how
  the histogram concentrates around the noise-free generator.
* In `04_inverse_problem.py`, replace the uniform prior with a Gaussian
  prior centred at a positive value and re-run; the bi-modality vanishes.
  This is the cheapest possible "hierarchical model" — Chapter 8 of the
  book formalizes this idea. See also
  [`../topics/inverse_problem.md`](../topics/inverse_problem.md) for why
  non-injective generators break naive inversion and how priors restore
  identifiability.

## Where the book takes this next

Chapter 2 turns the "agent in a box" intuition into the fully specified
recipe — process, model, grid inference — used systematically for the rest
of the book, and adds the closed-form MLE/MAP alternatives to grid
inference.
