# Chapter 9 — concept map

Chapters 6–8 built active inference in *continuous* state-spaces (Gaussian densities,
generalized coordinates). Chapter 9 turns to the **discrete** formulation — the
partially-observable Markov decision process (POMDP). This is the form of active inference
most used in practice (the "pymdp" style): everything is categorical, built from a few
matrices, and exact in the static case.

> **Implemented:** §9.1 — the discrete generative model (`A`/`B`/`C`/`D`) and exact
> hidden-state inference (Example 9.1, verified against Eq. 15); §9.2–§9.3 — dynamic
> forward filtering, prediction rollout, expected observations, and discrete VFE; and
> §9.4–§9.6 — **expected free energy**, policy/action selection, a Grid World planning
> agent, and the exploration/exploitation decomposition (verified against Eq. 63/68 and a
> behavioral goal-reaching oracle).

## Script inventory

| File | Role |
|---|---|
| `example_9_1_state_inference.py` | Static categorical hidden-state inference. |
| `example_9_2_dynamic_filtering.py` | HMM-style forward filtering and per-step discrete VFE. |
| `example_9_3_discrete_vfe.py` | Two-state VFE simplex and surprisal bound. |
| `example_9_4_gridworld.py` | Policy planning in Grid World by expected free energy. |
| `example_9_6_exploration_exploitation.py` | Risk/ambiguity decomposition for policy selection. |
| `animation_belief_filtering.py` | Belief-filtering animation. |
| `animation_efe_tradeoff.py` | Exploration/exploitation trade-off animation. |

## The model is a handful of matrices

| Matrix | Shape | Meaning |
|--------|-------|---------|
| **D** | `(C,)` | state prior `P(s) = Cat(D)` |
| **A** | `(O, C)` | likelihood `P(o\|s) = Cat(A)`; columns (states) sum to 1, `A[o,s] = P(o\|s)` |
| **B** | `(U, C, C)` | per-control transitions `P(s'\|s,u) = Cat(B[u])`; columns sum to 1 |
| **C** | `(O,)` | log-preferences over observations (the goal/utility) |
| **E** | `(n_π,)` | prior over policies |

The static generative model (§9.1) is just `M = {A, D}`, factorizing the joint as
`P(s, o) = P(o|s)·P(s)`. (`B` adds dynamics in §9.2; `C`/`E` enter with action in §9.5.)

```mermaid
flowchart TB
    D["D<br/>state prior"]
    A["A<br/>likelihood P(o|s)"]
    B["B<br/>transition P(s'|s,u)"]
    C["C<br/>observation preferences"]
    S["Belief s<br/>infer_states or forward_filter"]
    P["Policies pi<br/>action sequences"]
    G["Expected free energy G<br/>risk + ambiguity"]
    Q["Policy posterior Q(pi)<br/>softmax(-gamma G)"]
    U["Current action u<br/>action_posterior"]

    D --> S
    A --> S
    B --> S
    S --> P
    P --> G
    B --> G
    A --> G
    C --> G
    G --> Q
    Q --> U
    U --> B
```

## Exact hidden-state inference

An observation is a **one-hot** vector `ô ∈ {0,1}^O`. Selecting the observed row of the
likelihood, `Aᵀô`, gives the "credibility" of each state; multiplying by the prior and
normalizing yields the posterior (book Eq. 12/13):

```
s = (Aᵀô ⊙ D) / Σ(Aᵀô ⊙ D) = σ(log Aᵀô + log D)
```

— likelihood × prior, normalized, equivalently a **softmax** of (log-likelihood +
log-prior). In the static case this is *exact* Bayesian inference (no variational
approximation needed — that comes when we infer a whole state *sequence* in §9.3).

The companion verifies this against the book's exact worked example (Eq. 15): the weather
agent with a uniform prior, on observing **hot**, infers
`P(s) = [0.18, 0.40, 0.36, 0.06]` over (rainy, cloudy, sunny, snowy) — cloudy slightly more
likely than sunny, exactly as the book reports. Validation enforces that every `A`/`B`
column is a proper categorical (≥0, sums to 1).

## Dynamic filtering and discrete VFE (§9.2–§9.3)

Adding `B` turns the static categorical model into a hidden Markov model. Filtering becomes
a two-step loop:

```
prior_t = B · s_{t-1}
s_t     = σ(log Aᵀô_t + log prior_t)
```

`example_9_2_dynamic_filtering.py` shows the loop over a weather sequence: observations can
pull the posterior away from the transition-predicted state, while persistent `B` keeps the
belief from overreacting to a single noisy cue. The same script evaluates the §9.3 discrete
VFE at each filtered posterior:

```
F(s_t) = s_t · (log s_t − log prior_t − log Aᵀô_t)
```

`example_9_3_discrete_vfe.py` reduces this to a two-state simplex and sweeps `q = Q(s=0)`.
The curve is minimized at the exact Bayes posterior and touches the surprisal floor
`−log P(o)`, making the Chapter 4 free-energy bound visible in categorical form.

## Expected free energy — planning as inference (§9.5)

What makes the agent *act* is **expected free energy** (EFE), the utility used to rank
policies (action sequences). For a predicted future state `s` under a policy, EFE
decomposes into exactly two terms (book Eq. 52, the "RO-form"):

```
G = risk + ambiguity
  = o·(log o − log C)              risk        (reward-seeking, Eq. 60), o = A·s
  + (−diag(Aᵀlog A))·s             ambiguity   (information-seeking, Eq. 64)
```

* **Risk** is the KL divergence from the observations the policy is *expected* to produce
  (`o = A·s`) to the agent's *preferred* observations `C` — low when the policy realises
  the goal. It is *reward-seeking* / extrinsic value.
* **Ambiguity** is the expected entropy of the likelihood at the predicted state
  (`H·s`, `H` = per-state observation entropy) — low when the policy leads to states that
  give *unambiguous, informative* observations. It is *information-seeking* / intrinsic value.

A policy's total EFE sums this over its horizon (Eq. 54); the policy posterior is
`Q(π) = σ(−γ G)` (Eq. 55, best policy = lowest EFE = highest probability); and the action
posterior marginalizes policies onto the action taken now, `Q(u) = Σ_π δ(u=V[π,τ])·Q(π)`
(Eq. 69). The agent then takes `argmax Q(u)`.

The companion keeps this decomposition first-class. `efe_components` returns the risk,
ambiguity, expected observation, and optional Chapter 10 novelty term for one predicted
state. `evaluate_policy_components` rolls a full policy forward and returns a
`PolicyEFETrace`: predicted states, predicted observations, per-step terms, and policy
totals. `example_9_6_exploration_exploitation.py` and `animation_efe_tradeoff.py` use this
trace to show the §9.6 point directly: weak preferences let ambiguity reduction make an
exploratory policy competitive; sharper preferences make reward-seeking risk dominate.

> **A first-principles convention note.** The book's general EFE form says `C` is
> softmax-normalized, but its *worked oracle* (Example 9.7, risk = 6.337 nats) consumes `C`
> **raw** (`[1,0]`) with an ε-floor for `log 0`. The companion follows the oracle: `efe_risk`
> uses `C` as supplied. A red-team mutation pass also caught that the symmetric oracle `A`
> failed to bind the `A`-vs-`Aᵀ` orientation, so an asymmetric-A test (risk = 10.8017) was
> added to pin it.

**Grid World (Algorithm 9.5.1).** `make_gridworld` builds a deterministic, fully-observable
grid (identity `A`, four move `B[u]`, one-hot goal `C`); `simulate_pomdp_agent` runs the
receding-horizon loop. With a goal in the opposite corner of a 3×3 grid the agent reaches it
in the minimum **4 steps** — verified, along with the property that a goal-reaching policy
scores below one that never reaches the goal.

## Reusable building blocks

* **`active_inference.core.pomdp`** — `POMDPModel(A, D, B=None, C=None, E=None)` with
  column-stochastic validation; `infer_states` (Eq. 13); `efe_risk` / `efe_ambiguity` /
  `expected_free_energy` (Eq. 52); `efe_components`, `evaluate_policy_components`
  (`PolicyEFETrace`, Eq. 52/54), `evaluate_policy` (Eq. 54), `policy_posterior` (Eq. 55),
  `action_posterior` (Eq. 69), `predict_state`; dynamic helpers `forward_filter`,
  `predict_beliefs`, `expected_observation`, and `discrete_vfe`; primitives `softmax`,
  `one_hot`, `is_stochastic_matrix`.
* **`active_inference.estimators.pomdp`** — `make_gridworld`, `enumerate_policies`,
  `simulate_pomdp_agent` → `GridWorldResult`.
* **`active_inference.visualizations`** — `plot_discrete_belief_sequence` /
  `animate_discrete_beliefs` for Chapter 9 belief trajectories, and
  `plot_policy_efe_decomposition` / `animate_policy_efe_tradeoff` for risk/ambiguity
  policy selection.

## Where the book takes this next

Chapter 10 then *learns* the matrices (`A`, `B`, `D`) from experience, adds habits and
precision, and expands the model into factorial and hierarchical forms.
