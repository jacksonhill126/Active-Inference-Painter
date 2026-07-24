# Chapter 6 — concept map

Chapter 6 opens **Part II**. Chapters 4–5 inferred a *static* hidden state; Chapter 6
makes the state **dynamic** and filters it **online** — one observation arrives per
time step and the agent corrects its belief immediately. The algorithm is *generalized
filtering*, and it is the template for every continuous-time active-inference
simulation in Part II.

> **Implemented:** §6.1 (univariate generalized filter, Example 6.1, Algorithm
> 6.1.1), §6.2 (the multivariate filter, Example 6.2), §6.3–6.5 (generalized
> coordinates of motion + the generalized filter, Example 6.6, Algorithm 6.5.1), and
> §6.6 (correlated embedding orders + Example 6.7).

## Script inventory

| File | Role |
|---|---|
| `example_6_1_generalized_filter.py` | Univariate generalized filtering and the Chapter 6 tracking oracle. |
| `example_6_2_multivariate_filter.py` | Vector Hooke-oscillator generalized filtering. |
| `example_6_6_generalized_coordinates.py` | Generalized-coordinate position and velocity tracking. |
| `example_6_7_multivariate_generalized_coordinates.py` | Vector generalized-coordinate filtering with correlated precision. |
| `visualize_6_6_correlated_embedding_orders.py` | Precision-matrix heatmaps for correlated embedding orders. |

## From static predictive coding to a dynamic state-space model

The static prior mean `m_x` of Chapter 5 is replaced by a **state-transition
function** `f_M` that predicts how the state *flows* in time. The dynamic generative
model (book Eq. 10) is

```
p(μ_x)    = N(μ_x; f_M(μ_x; θ_x), s_x²)     state-transition model
p(y | μ_x) = N(y;  g_M(μ_x; θ_y), σ_y²)     observation model
```

For Example 6.1, `f_M(μ)=θ_x−μ` (a point attractor at `θ_x`) and `g_M(μ)=μ−θ_y`. Under
the Laplace/quadratic approximation the variational free energy keeps the same shape as
Chapter 5 (book Eq. 6, 7a):

| Quantity | Definition |
|----------|-----------|
| state prediction error | `ε_x = μ_x − f_M(μ_x; θ_x)` |
| sensory prediction error | `ε_y = y − g_M(μ_x; θ_y)` |
| free energy | `F = ½(λ_y ε_y² + λ_x ε_x² + log(σ_y² s_x²))` |

## Recognition as a flow integrated by Euler's method

Perception is *gradient flow* on `F`, integrated forward in time (book Eq. 8,
Algorithm 6.1.1):

```
μ̇_x = −κ ∂F/∂μ_x,        μ_x^(t+1) = μ_x^(t) + Δt·μ̇_x
```

> **A derived, not transcribed, gradient.** As in Chapter 5, the book's printed
> gradient (Eq. 7b/14) uses a loose sign convention. The companion *derives* it by the
> chain rule and verifies it against a central finite difference:
> `∂F/∂μ_x = λ_x ε_x (1 − f_M'(μ_x)) − λ_y ε_y g_M'(μ_x)`.

The online loop alternates an **environment step** (the true state transitions and emits
a noisy observation) with an **agent step** (one Euler step of the recognition flow).

## Why the filter tracks — precision is the whole story

For a linear `f_M`, `g_M` the free energy is convex, so the agent's recognition has a
**closed-form fixed point** (the analytical landmark `gf_fixed_point_linear`):

```
μ* = [λ_x(1−a_f)b_f + λ_y a_g(y − b_g)] / [λ_x(1−a_f)² + λ_y a_g²]
```

With **high sensory precision** `λ_y ≫ λ_x` (Example 6.1 uses `λ_y=50`, `λ_x=0.2`) the
sensory term dominates and `μ*` collapses onto the state that explains `y` — which is
exactly `x*` when `y = g_E(x*)`. So the belief tracks the true state. With *equal*
precisions the same machinery gives a *biased* estimate; precision weighting is what
makes perception work. The companion verifies this directly: filtered tracking error
≈ 0.08 versus ≈ 5 for the unfiltered prior.

## The multivariate filter (§6.2) — Hooke's law and perception lag

§6.2 lifts the filter to **vector** states and observations (book Eq. 12–15). The free
energy keeps its shape with precision *matrices* `Π_x`, `Π_y`, and the gradient uses
**Jacobians** (derived, FD-verified):

```
F      = ½(ε_xᵀ Π_x ε_x + ε_yᵀ Π_y ε_y + log|Σ_x Σ_y|)
∂F/∂μ  = (I − J_f)ᵀ Π_x ε_x − J_gᵀ Π_y ε_y
```

Example 6.2 is a mass on a spring — **Hooke's law** `ẋ₁ = x₂`, `ẋ₂ = (k/m)(v₀ − x₁)` — a
linear oscillator orbiting the stable fixed point `(v₀, 0) = (5, 0)`. The vector belief
tracks the orbit, but with a visible **perception lag** at the turning points where the
slope changes fastest: the free energy oscillates rather than staying at its floor. That
lag is exactly what *generalized coordinates of motion* (§6.3) are introduced to fix — by
giving the model the state's velocity and acceleration it can *look ahead* along the
trajectory. The companion verifies the multivariate filter two ways: it **reduces exactly
to the scalar §6.1 filter** on a 1-D problem, and it tracks the oscillator to ≈ 1.1 mean
error versus ≈ 11 for the static prior. The linear case also has a closed-form fixed point
(`mv_gf_fixed_point_linear`, a linear solve).

## Generalized coordinates of motion (§6.3–6.5)

The §6.2 lag has a cure: give the model the state's *motion*. In generalized coordinates
the belief is a whole trajectory `μ̃_x = [μ_x, μ_x', μ_x'', …]` (embedding dimension `M`),
carrying position, velocity, acceleration, … The machinery has three pieces:

* **The derivative shift operator** `D` (`shift_operator`, Eq. 37): `D μ̃_x = μ̃_x'` shifts
  every order up by one and zeros the last — `D·[3,4,2,6,4] = [4,2,6,4,0]`. Formally
  `D = I_C ⊗ S` with `S` the superdiagonal-ones matrix.
* **Embedded functions** (Eq. 30/36, local linearity): `f̃(μ̃) = [f(μ_x), f'(μ_x)μ_x',
  f'(μ_x)μ_x'', …]`, and likewise `g̃`. The generalized errors are then
  `ε̃_x = D μ̃_x − f̃(μ̃_x)` (Eq. 46b) and `ε̃_y = ỹ − g̃(μ̃_x)` (Eq. 46a). The `D` term
  supplies the *actual* generalized motion to compare against the *predicted* flow `f̃` —
  this is what fixes the §6.1 awkwardness (there `ε_x = μ − f(μ)` had no motion to compare).
* **The generalized recognition flow** (Eq. 51): `μ̃̇_x = D μ̃_x − κ ∂F/∂μ̃_x`. The extra
  `D μ̃_x` term moves the belief along its own predicted trajectory — gradient descent
  performed in a *reference frame locked to the state's motion*. At equilibrium
  `μ̃̇_x = D μ̃_x` (the *motion of the expectation equals the expectation of the motion*),
  and `∂F/∂μ̃_x → 0`. The gradient (Eq. 50a),
  `(D − f'(μ_x)I)ᵀ Π̃_x ε̃_x − (g'(μ_x)I)ᵀ Π̃_y ε̃_y`, uses the local-linearity approximation
  — *exact* for linear `f`/`g`, and finite-difference-verified.

**The payoff** (Example 6.6): filtering a drifting object in generalized coordinates, the
order-0 belief `μ_x` recovers the position *and* the order-1 belief `μ_x'` recovers the
true velocity `ẋ*` — the agent infers not just *where* the state is but *how it is moving*.
The companion verifies this: the `D` operator reproduces the book's worked shift, the
gradient matches finite differences exactly (linear case), holding the observation at the
at-rest steady state recovers `[10, 0, 0]`, and on a constant-velocity trajectory a
free-motion model recovers the velocity to ≈ 2 % (with a point-attractor model the velocity
is pulled slightly toward zero — the model's own expectation of rest).

## Correlated embedding orders (§6.6)

The diagonal precision used in §6.5 assumes embedding orders are independent. §6.6
relaxes that assumption: smooth colored noise makes a signal and its derivatives
correlated, so the generalized precision becomes a full matrix. The companion
implements the Gaussian temporal covariance `S(γ)` from the book's derivative formula
and constructs:

```
book/order-major:  Π̃_i(γ) = S(γ)^-1 ⊗ Π_i
repo/state-major:  Π̃_i(γ) = Π_i ⊗ S(γ)^-1
```

Both layouts are tested because the book writes tensors by embedding order while the
code's `D = I_C ⊗ S` uses state-major flattened vectors. Example 6.7 combines this
precision with a vector generalized-coordinate filter for the Hooke oscillator. The
raw exports include `S(γ)`-derived precision matrices, generalized measurements,
belief tensors, prediction errors, and VFE traces.

## Reusable building blocks

* **`active_inference.core.generalized_filtering`** — `DynamicStateSpaceModel`;
  `gf_state_prediction_error`, `gf_sensory_prediction_error`; `gf_free_energy`;
  `gf_free_energy_grad` (analytic, derived) and `gf_free_energy_grad_fd` (numerical
  oracle); `gf_fixed_point_linear` (closed-form steady state).
* **`active_inference.estimators.generalized_filtering`** — `DynamicProcess` +
  `simulate_dynamic_process` (Euler-integrated stochastic environment), and
  `generalized_filter` → `GeneralizedFilterResult` (with `tracking_error`).
* **Multivariate (§6.2)** — `MultivariateDynamicModel` with `VectorFunction` /
  `LinearVectorFunction` / `GenericVectorFunction` (finite-difference Jacobian),
  `mv_gf_free_energy`, `mv_gf_free_energy_grad`(`_fd`), `mv_gf_fixed_point_linear`; and
  `MultivariateDynamicProcess` + `simulate_multivariate_process` +
  `multivariate_generalized_filter` → `MultivariateFilterResult`.
* **Generalized coordinates (§6.3–6.5)** — `shift_operator` (`D`), `embed_flow`,
  `GeneralizedModel`, `generalized_state_error` / `generalized_sensory_error`,
  `generalized_free_energy`, `generalized_free_energy_grad`(`_fd`); and the estimator
  `generalized_filter_gc` → `GeneralizedFilterGCResult` (with `positions` / `velocities`).
* **Correlated/vector generalized coordinates (§6.6)** — `gaussian_temporal_covariance`,
  `correlated_embedding_precision`, `GeneralizedVectorModel`, vector generalized errors
  and VFE gradients, `generalized_measurements_from_series`, and
  `generalized_vector_filter` → `GeneralizedVectorFilterResult`.
* **`active_inference.visualizations.unified.plot_generalized_filter`** — Fig. 6.1.3
  (tracking, prediction errors, free energy) in the shared bold/colourblind-safe style.
* **Ch. 6 educational figures** — `plot_correlated_embedding_precision` and
  `plot_generalized_vector_filter` cover Fig. 6.6.2 and Example 6.7-style traces.

## Verification

* **Gradient** analytic == finite difference (`max|err| ≈ 5e-9`), with a wrong-sign
  falsification check.
* **Closed-form fixed point** zeroes the gradient and matches an independent grid argmin
  across a parameter sweep.
* **Tracking** — the filter's belief tracks an independently-simulated true trajectory
  to `≈ 0.08` mean error, `> 5×` better than the unfiltered prior; the steady-state
  fixed point recovers `x*` under high sensory precision.

## Where the book takes this next

Chapter 7 adds **action** (active generalized filtering), turning this perception filter
into a full action–perception agent.
