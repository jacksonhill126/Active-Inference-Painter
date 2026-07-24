r"""Discrete state-space active inference — the POMDP generative model (book Ch. 9).

Chapters 6–7 used *continuous* states with Gaussian densities. Chapter 9 switches to
*discrete* states and **categorical** distributions, the partially-observable Markov
decision process (POMDP) formulation of active inference. Everything is built from a
handful of matrices (book §9.1–9.2):

* **D** ``∈ [0,1]^C`` — the **state prior** ``P(s) = Cat(D)`` (sums to 1).
* **A** ``∈ [0,1]^{O×C}`` — the **likelihood** ``P(o|s) = Cat(A)``; columns (states) each
  sum to 1, so ``A[o, s] = P(o=o | s=s)``.
* **B** ``∈ [0,1]^{C×C}`` (per control) — the **transition** ``P(s'|s) = Cat(B)``; columns
  (current state) sum to 1, so ``B[s', s] = P(s'=s' | s=s)``.
* **C**, **E** — preferences over observations and a prior over policies (used once we add
  action and expected free energy; defined here for completeness).

This module implements §9.1: the model container and **exact hidden-state inference**. For
a one-hot observation ``ô`` the posterior over states is (book Eq. 12/13)

.. math::
    s = \frac{(A^\top \hat o)\odot D}{\sum_s (A^\top \hat o)\odot D}
      = \sigma\!\left(\log A^\top \hat o + \log D\right),

i.e. likelihood × prior, normalized — equivalently a softmax of (log-likelihood + log-prior).
Inference over a *sequence* with the **B** transitions, and **expected free energy** /
policy selection, build on this in later increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnostics import logsumexp

__all__ = [
    "softmax",
    "one_hot",
    "is_stochastic_matrix",
    "POMDPModel",
    "infer_states",
    # §9.5 — expected free energy + policy/action selection
    "predict_state",
    "EFEComponents",
    "PolicyEFETrace",
    "efe_components",
    "efe_risk",
    "efe_ambiguity",
    "expected_free_energy",
    "evaluate_policy",
    "evaluate_policy_components",
    "policy_posterior",
    "action_posterior",
    # §9.2 — dynamic filtering (HMM forward pass)
    "forward_filter",
    "predict_beliefs",
    "expected_observation",
    # §9.3 — discrete variational free energy
    "discrete_vfe",
    # §10.1 — learning the POMDP parameters (Dirichlet)
    "dirichlet_expected_value",
    "expected_A",
    "expected_B",
    "expected_D",
    "accumulate_a_counts",
    "accumulate_b_counts",
    "accumulate_d_counts",
    "novelty_matrix",
    "parameter_novelty",
    "efe_with_novelty",
    # §10.2 — habit (baseline prior E) + policy precision γ
    "policy_posterior_full",
    "precision_gradient",
    "learn_precision",
    "PrecisionResult",
    # §10.3 — factorial depth (multiple state factors + observation modalities)
    "FactorialPOMDP",
    "factorial_expected_observation",
    "factorial_likelihood_message",
    "factorial_infer_states",
    "factorial_predict_states",
    "factorial_modality_ambiguity",
    "factorial_efe",
    # §10.4 — hierarchical depth (nested POMDP layers)
    "HierarchicalPOMDP",
    "hierarchical_layer_vfe",
    "hierarchical_layer_efe",
    "hierarchical_policy_posterior",
]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""Numerically stable softmax (the Boltzmann normalization ``σ``)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.asarray(1.0)
    axis = int(axis)
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array with ndim={x.ndim}")
    lse = logsumexp(x, axis=axis)
    return np.exp(x - np.expand_dims(lse, axis=axis))


def one_hot(index: int, n: int) -> np.ndarray:
    r"""One-hot observation vector ``ô ∈ {0,1}^n`` with a single 1 at ``index`` (Eq. 8)."""
    n = int(n)
    index = int(index)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if index < 0 or index >= n:
        raise ValueError(f"index must be in [0, {n}), got {index}")
    v = np.zeros(n, dtype=float)
    v[index] = 1.0
    return v


def is_stochastic_matrix(M: np.ndarray, *, atol: float = 1e-9) -> bool:
    """True if every column of ``M`` is a valid categorical (non-negative, sums to 1)."""
    M = np.asarray(M, dtype=float)
    return bool(
        M.ndim == 2
        and np.all(np.isfinite(M))
        and np.all(M >= -atol)
        and np.allclose(M.sum(axis=0), 1.0, atol=atol)
    )


def _probability_vector(
    x: np.ndarray,
    *,
    name: str,
    length: int | None = None,
    atol: float = 1e-9,
) -> np.ndarray:
    """Validate a 1-D categorical vector and return it as ``float``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if length is not None and x.shape[0] != int(length):
        raise ValueError(f"{name} must have length {length}, got {x.shape[0]}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(x < -atol):
        raise ValueError(f"{name} must be non-negative")
    if not np.isclose(x.sum(), 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1, got {x.sum()}")
    return x


def _nonnegative_vector(
    x: np.ndarray,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    """Validate a finite, non-negative 1-D vector with positive total mass."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if length is not None and x.shape[0] != int(length):
        raise ValueError(f"{name} must have length {length}, got {x.shape[0]}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(x < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if x.sum() <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return x


def _observation_vector(obs: "int | np.ndarray", n_obs: int) -> np.ndarray:
    """Convert an observation index or categorical vector to a validated vector."""
    if np.ndim(obs) == 0:
        return one_hot(int(obs), n_obs)
    return _probability_vector(np.asarray(obs, dtype=float), name="obs", length=n_obs)


@dataclass(frozen=True)
class POMDPModel:
    r"""A discrete (categorical) generative model for active inference (book §9.1–9.2).

    Parameters
    ----------
    A : ndarray, shape (O, C)
        Likelihood ``P(o|s)``; each column (a state) is a categorical over observations.
    D : ndarray, shape (C,)
        State prior ``P(s)`` (categorical over states).
    B : ndarray, shape (n_controls, C, C), optional
        Per-control transition tensor; ``B[u]`` has columns summing to 1,
        ``B[u][s', s] = P(s'=s' | s=s, u)``. ``None`` for the static (§9.1) case.
    C : ndarray, shape (O,), optional
        Log-preferences over observations (the goal/utility; used for expected free energy).
    E : ndarray, shape (n_policies,), optional
        Prior over policies.
    """

    A: np.ndarray
    D: np.ndarray
    B: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D (O×C), got shape {A.shape}")
        D = _probability_vector(self.D, name="D", length=A.shape[1])
        if not is_stochastic_matrix(A):
            raise ValueError("A columns must each be a categorical (≥0, sum to 1)")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "D", D)
        if self.B is not None:
            B = np.asarray(self.B, dtype=float)
            if B.ndim == 2:
                B = B[None, :, :]
            if B.ndim != 3:
                raise ValueError(f"B must be 2-D or 3-D, got shape {B.shape}")
            if B.shape[1:] != (A.shape[1], A.shape[1]):
                raise ValueError(
                    f"B slices must have shape (C, C)=({A.shape[1]}, {A.shape[1]}), "
                    f"got {B.shape[1:]}"
                )
            for u in range(B.shape[0]):
                if not is_stochastic_matrix(B[u]):
                    raise ValueError(f"B[{u}] columns must each be a categorical")
            object.__setattr__(self, "B", B)
        if self.C is not None:
            C = _nonnegative_vector(self.C, name="C", length=A.shape[0])
            object.__setattr__(self, "C", C)
        if self.E is not None:
            E = _nonnegative_vector(self.E, name="E")
            object.__setattr__(self, "E", E)

    @property
    def n_states(self) -> int:
        """Return dimensionality metadata for the validated probabilistic model."""
        return int(np.asarray(self.A).shape[1])

    @property
    def n_obs(self) -> int:
        """Return dimensionality metadata for the validated probabilistic model."""
        return int(np.asarray(self.A).shape[0])


def infer_states(model: POMDPModel, obs: "int | np.ndarray", *,
                 prior: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Exact hidden-state inference for a static POMDP (book Eq. 12/13).

    Returns the posterior ``P(s|o) = σ(log Aᵀô + log D)`` for a one-hot observation. ``obs``
    may be an integer index or a one-hot vector. ``prior`` overrides ``D`` (e.g. the
    ``B``-propagated belief from the previous step in a dynamic model).
    """
    A = np.asarray(model.A, dtype=float)
    o = _observation_vector(obs, A.shape[0])
    d = _probability_vector(model.D if prior is None else prior, name="prior", length=A.shape[1])
    likelihood = A.T @ o                      # (C,) — "credibility" of each state (Eq. 9/10)
    log_post = np.log(likelihood + 1e-16) + np.log(d + 1e-16)
    return softmax(log_post)


# ---------------------------------------------------------------------------
# §9.5 — expected free energy, policy and action selection
# ---------------------------------------------------------------------------

_EPS = 1e-16


@dataclass(frozen=True)
class EFEComponents:
    r"""Risk/ambiguity/novelty decomposition of one predicted POMDP state.

    Chapter 9's expected free energy is not just a scalar score. Its educational value is the
    decomposition ``G = risk + ambiguity`` (book Eq. 52): reward-seeking pressure plus
    information-seeking pressure. Chapter 10 adds parameter novelty as an information gain
    that is subtracted. This container makes that decomposition explicit for methods,
    figures, and tests.
    """

    state: np.ndarray
    expected_observation: np.ndarray
    risk: float
    ambiguity: float
    novelty: float = 0.0

    @property
    def total(self) -> float:
        """Total expected free energy ``risk + ambiguity - novelty``."""
        return float(self.risk + self.ambiguity - self.novelty)


@dataclass(frozen=True)
class PolicyEFETrace:
    r"""Per-step expected-free-energy trace for one policy (book Eq. 54).

    ``states[t]`` is the predicted belief after applying the ``t``-th action in ``policy``;
    ``risks[t]`` and ``ambiguities[t]`` are the two Chapter 9 EFE terms at that prediction.
    ``novelties[t]`` is zero unless the Chapter 10 Dirichlet novelty term was requested.
    """

    policy: np.ndarray
    states: np.ndarray
    expected_observations: np.ndarray
    risks: np.ndarray
    ambiguities: np.ndarray
    novelties: np.ndarray

    @property
    def totals_per_step(self) -> np.ndarray:
        """Per-step ``G_t = risk_t + ambiguity_t - novelty_t``."""
        return self.risks + self.ambiguities - self.novelties

    @property
    def total(self) -> float:
        """Policy EFE ``G(π) = Σ_t G_t`` (book Eq. 54)."""
        return float(np.sum(self.totals_per_step))

    @property
    def risk_total(self) -> float:
        """Policy-total risk term."""
        return float(np.sum(self.risks))

    @property
    def ambiguity_total(self) -> float:
        """Policy-total ambiguity term."""
        return float(np.sum(self.ambiguities))

    @property
    def novelty_total(self) -> float:
        """Policy-total parameter novelty term."""
        return float(np.sum(self.novelties))


def predict_state(model: POMDPModel, s: np.ndarray, u: int) -> np.ndarray:
    r"""One-step state prediction under control ``u``: ``s' = B[u]·s`` (book §9.4/Eq 44).

    With the null/empty observation the likelihood term is uniform, so the predicted
    belief is simply the ``B``-propagated current belief (already normalized).
    """
    if model.B is None:
        raise ValueError("model has no B (transition) tensor")
    u = int(u)
    if u < 0 or u >= model.B.shape[0]:
        raise ValueError(f"u must be in [0, {model.B.shape[0]}), got {u}")
    s = _probability_vector(s, name="s", length=model.n_states)
    return np.asarray(model.B[u], dtype=float) @ s


def efe_components(
    model: POMDPModel,
    s: np.ndarray,
    C: np.ndarray,
    *,
    a: Optional[np.ndarray] = None,
) -> EFEComponents:
    r"""Decompose EFE at one predicted state into book terms.

    Returns risk (reward-seeking, Eq. 60), ambiguity (information-seeking, Eq. 64), and
    optionally Chapter 10 parameter novelty (Eq. 13/15) when Dirichlet counts ``a`` are
    supplied. The scalar used for policy ranking is ``components.total``.
    """
    A = np.asarray(model.A, dtype=float)
    s = _probability_vector(s, name="s", length=model.n_states)
    C = _nonnegative_vector(C, name="C", length=model.n_obs)
    o = A @ s
    risk = float(o @ (np.log(o + _EPS) - np.log(C + _EPS)))
    H = -np.sum(A * np.log(A + _EPS), axis=0)
    ambiguity = float(H @ s)
    novelty = 0.0 if a is None else parameter_novelty(a, s)
    return EFEComponents(
        state=s.copy(),
        expected_observation=o,
        risk=risk,
        ambiguity=ambiguity,
        novelty=float(novelty),
    )


def efe_risk(model: POMDPModel, s: np.ndarray, C: np.ndarray) -> float:
    r"""Risk / reward-seeking term of EFE (book Eq. 60): ``o·(log o − log C)``, ``o = A s``.

    A KL-like divergence from the predicted observations to the preferences ``C`` — low
    when the policy is expected to realise preferred observations. ``C`` is used as supplied
    (raw preference vector, ε-floored for ``log``), matching the book's Example 9.7 oracle.
    """
    return efe_components(model, s, C).risk


def efe_ambiguity(model: POMDPModel, s: np.ndarray) -> float:
    r"""Ambiguity / information-seeking term of EFE (book Eq. 64/67): ``H·s``.

    ``H = −diag(Aᵀ log A)`` is the per-state expected entropy of the likelihood (positive);
    the dot with the predicted state gives the expected observation entropy under the policy.
    Minimizing it drives the agent toward states that yield *unambiguous* (informative)
    observations.
    """
    C_uniform = np.ones(model.n_obs) / model.n_obs
    return efe_components(model, s, C_uniform).ambiguity


def expected_free_energy(model: POMDPModel, s: np.ndarray, C: np.ndarray) -> float:
    r"""Expected free energy at a predicted state (book Eq. 52): ``G = risk + ambiguity``."""
    return efe_components(model, s, C).total


def evaluate_policy(model: POMDPModel, s0: np.ndarray, policy: "list[int] | np.ndarray",
                    C: np.ndarray) -> float:
    r"""Total expected free energy of a policy (book Eq. 54): ``G(π) = Σ_τ G^{[π,τ]}``.

    Propagates the belief forward under the policy's action sequence (``s ← B[u]·s``) and
    sums the per-step EFE over the planning horizon. Lower ``G`` = better policy.
    """
    return evaluate_policy_components(model, s0, policy, C).total


def evaluate_policy_components(
    model: POMDPModel,
    s0: np.ndarray,
    policy: "list[int] | np.ndarray",
    C: np.ndarray,
    *,
    a: Optional[np.ndarray] = None,
) -> PolicyEFETrace:
    r"""Per-step EFE decomposition for a policy rollout (book Eq. 52/54).

    This is the method-level object behind the Chapter 9 exploration/exploitation
    figures: it keeps the policy's predicted states, expected observations, and the two
    additive EFE terms instead of collapsing everything immediately to one scalar.
    """
    policy_arr = np.asarray(policy, dtype=int)
    if policy_arr.ndim != 1:
        raise ValueError(f"policy must be a 1-D action sequence, got {policy_arr.shape}")
    s = _probability_vector(s0, name="s0", length=model.n_states)
    states: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    risks: list[float] = []
    ambiguities: list[float] = []
    novelties: list[float] = []
    for u in policy_arr:
        s = predict_state(model, s, int(u))
        comp = efe_components(model, s, C, a=a)
        states.append(comp.state)
        observations.append(comp.expected_observation)
        risks.append(comp.risk)
        ambiguities.append(comp.ambiguity)
        novelties.append(comp.novelty)
    return PolicyEFETrace(
        policy=policy_arr.copy(),
        states=np.asarray(states, dtype=float),
        expected_observations=np.asarray(observations, dtype=float),
        risks=np.asarray(risks, dtype=float),
        ambiguities=np.asarray(ambiguities, dtype=float),
        novelties=np.asarray(novelties, dtype=float),
    )


def policy_posterior(G: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    r"""Policy posterior ``Q(π) = σ(−γ G)`` (book Eq. 55) — best (lowest-EFE) policy most likely."""
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    return softmax(-gamma * np.asarray(G, dtype=float))


def action_posterior(policy_post: np.ndarray, policies: "list[list[int]] | np.ndarray",
                     n_controls: int, *, tau: int = 0) -> np.ndarray:
    r"""Action posterior at step ``tau`` (book Eq. 69): ``Q(u) = Σ_π δ(u = V[π,τ]) Q(π)``.

    Marginalizes the policy posterior onto the action each policy takes at time ``tau``.
    """
    policy_post = _probability_vector(policy_post, name="policy_post")
    policies = np.asarray(policies, dtype=int)
    if policies.ndim != 2:
        raise ValueError(f"policies must be 2-D (n_policies, horizon), got {policies.shape}")
    if policies.shape[0] != policy_post.shape[0]:
        raise ValueError("policy_post length must match number of policies")
    tau = int(tau)
    if tau < 0 or tau >= policies.shape[1]:
        raise ValueError(f"tau must be in [0, {policies.shape[1]}), got {tau}")
    q_u = np.zeros(int(n_controls), dtype=float)
    for p, pol in enumerate(policies):
        u = int(pol[tau])
        if u < 0 or u >= n_controls:
            raise ValueError(f"policy action must be in [0, {n_controls}), got {u}")
        q_u[u] += policy_post[p]
    return q_u


# ---------------------------------------------------------------------------
# §9.2 — Inference in a dynamic environment (HMM forward filtering, Alg. 9.2.1)
# ---------------------------------------------------------------------------


def forward_filter(
    model: POMDPModel,
    observations: "list[int] | np.ndarray",
    *,
    B: "np.ndarray | None" = None,
) -> np.ndarray:
    r"""Forward filtering for a hidden Markov model (book Alg. 9.2.1).

    Processes a sequence of observations one at a time, maintaining a rolling
    belief about the hidden state:

    * ``t = 0``: ``s^{(0)} = σ(log A^\top \hat o^{(0)} + log D)``
    * ``t > 0``: ``s^{(t)} = σ(log A^\top \hat o^{(t)} + log B s^{(t-1)})``

    The ``B`` matrix propagates the previous belief to a prior for the current step
    (the discrete analogue of the Kalman predict step).

    Args:
        model: :class:`POMDPModel` with ``A`` and ``D`` matrices.
        observations: ``(T,)`` int sequence of observation indices *or* ``(T, O)``
            one-hot array.  Passed directly to :func:`infer_states`.
        B: ``(C, C)`` column-stochastic transition matrix.  If ``None``, uses
            ``model.B[0]`` (the first control slice of the stored tensor).

    Returns:
        beliefs: ``(T, C)`` array; row ``t`` is the posterior belief at step ``t``.
    """
    obs_arr = np.asarray(observations)
    T = len(obs_arr)
    if B is None:
        _B = None if model.B is None else np.asarray(model.B[0], dtype=float)
    else:
        _B = np.asarray(B, dtype=float)
        if _B.shape != (model.n_states, model.n_states) or not is_stochastic_matrix(_B):
            raise ValueError("B must be a column-stochastic (C, C) transition matrix")
    if T > 1 and _B is None:
        raise ValueError("forward_filter needs B or model.B for sequences longer than one")
    beliefs = np.zeros((T, model.n_states))
    for t in range(T):
        obs_t = int(obs_arr[t]) if obs_arr.ndim == 1 else obs_arr[t]
        if t == 0:
            beliefs[t] = infer_states(model, obs_t)
        else:
            assert _B is not None
            prior_t = _B @ beliefs[t - 1]
            beliefs[t] = infer_states(model, obs_t, prior=prior_t)
    return beliefs


def predict_beliefs(
    model: POMDPModel,
    s: np.ndarray,
    horizon: int,
    *,
    B: "np.ndarray | None" = None,
) -> np.ndarray:
    r"""Prediction rollout: propagate belief forward with no observation (book Alg. 9.2.2).

    At each future step ``τ > t`` there is no sensory data, so the likelihood term
    ``log A · o^∅`` is omitted (``o^∅ ∈ \{0\}^D``).  The update reduces to

    .. math::
        s^{(\tau)} = \sigma(\log B s^{(\tau-1)})

    which is equivalent to column-normalising ``B s`` (since ``Bs`` is already a
    valid probability vector), but the log-space path is retained for numerical
    stability with near-zero entries.

    Args:
        model: :class:`POMDPModel`.
        s: ``(C,)`` current (or seed) belief vector.
        horizon: number of future steps to predict.
        B: ``(C, C)`` transition matrix; if ``None`` uses ``model.B[0]``.

    Returns:
        beliefs: ``(horizon, C)`` array of predicted future beliefs.
    """
    horizon = int(horizon)
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if B is None:
        if model.B is None:
            raise ValueError("predict_beliefs needs B or model.B")
        _B = np.asarray(model.B[0], dtype=float)
    else:
        _B = np.asarray(B, dtype=float)
        if _B.shape != (model.n_states, model.n_states) or not is_stochastic_matrix(_B):
            raise ValueError("B must be a column-stochastic (C, C) transition matrix")
    beliefs = np.zeros((horizon, model.n_states))
    s_prev = _probability_vector(s, name="s", length=model.n_states)
    for i in range(horizon):
        log_prior = np.log(_B @ s_prev + _EPS)
        beliefs[i] = softmax(log_prior)
        s_prev = beliefs[i]
    return beliefs


def expected_observation(model: POMDPModel, s: np.ndarray) -> np.ndarray:
    r"""Expected observation distribution ``o = A s`` (book Alg. 9.2.2, line 11).

    Given a belief ``s`` over hidden states, returns the expected distribution over
    observations under the likelihood ``A``.

    Args:
        model: :class:`POMDPModel` with ``A``.
        s: ``(C,)`` belief vector.

    Returns:
        ``(O,)`` expected observation distribution (sums to 1 if ``s`` does).
    """
    s = _probability_vector(s, name="s", length=model.n_states)
    return np.asarray(model.A, dtype=float) @ s


# ---------------------------------------------------------------------------
# §9.3 — Variational inference for hidden-state estimation
# ---------------------------------------------------------------------------


def discrete_vfe(
    model: POMDPModel,
    s: np.ndarray,
    obs: "int | np.ndarray",
    prior: np.ndarray,
) -> float:
    r"""Discrete variational free energy — **G-form** (book Eq. 29).

    .. math::
        \mathcal{F} = s \cdot (\log s - \log \mathrm{prior} - \log A^\top \hat o)

    This is the discrete :math:`D_{KL}[Q(s) \| P(o, s)]` where the variational
    density :math:`Q(s) = s` and the joint :math:`P(o, s) = P(o|s)P(s)` is
    approximated by the likelihood column :math:`A^\top \hat o` times the prior.

    At the minimum (:math:`s` = exact posterior),
    :math:`\mathcal{F} \approx -\log P(o)` (the surprisal / negative log model
    evidence).

    Args:
        model: :class:`POMDPModel` with ``A``.
        s: ``(C,)`` variational belief (any categorical, not necessarily posterior).
        obs: observation — an integer index *or* a ``(O,)`` one-hot vector.
        prior: ``(C,)`` prior over states: ``D`` at ``t = 0``,
            ``B s^{(t-1)}`` for ``t > 0``.

    Returns:
        float: variational free energy (≥ surprisal, with equality at the posterior).
    """
    A = np.asarray(model.A, dtype=float)
    s_arr = _probability_vector(s, name="s", length=model.n_states)
    prior_arr = _probability_vector(prior, name="prior", length=model.n_states)
    obs_vec = _observation_vector(obs, model.n_obs)
    log_lik = np.log(A.T @ obs_vec + _EPS)    # (C,) log P(o | s) per state
    log_prior = np.log(prior_arr + _EPS)        # (C,) log P(s)
    log_s = np.log(s_arr + _EPS)               # (C,) log Q(s)
    return float(s_arr @ (log_s - log_prior - log_lik))


# ---------------------------------------------------------------------------
# §10.1 — Learning the POMDP model parameters (Dirichlet concentration updates)
# ---------------------------------------------------------------------------
#
# Chapter 9 assumed the arrays A, B, D were *known*. Chapter 10 makes them
# unknown random variables with **Dirichlet** priors. A Dirichlet is the
# conjugate prior of the categorical, so learning is just *counting*: each
# array gets a matrix/vector of concentration parameters (pseudocounts) ``a``,
# ``b``, ``d``; over a trial the agent accumulates evidence (state–observation
# and state–state co-occurrences), adds it to the pseudocounts at the end of the
# trial, and reads off the updated probability arrays as the **expected value**
# of the resulting Dirichlet (book Eq. 4–6).


def dirichlet_expected_value(counts: np.ndarray, *, axis: int = 0) -> np.ndarray:
    r"""Expected value of a Dirichlet from its concentration parameters (book Eq. 5).

    The mean of ``Dir(α)`` is ``α / Σα``. For the POMDP arrays the normalization runs
    over the *output* axis of each categorical: columns (``axis=0``) for the
    likelihood ``A`` (``A_ij = a_ij / Σ_i a_ij``) and each transition slice ``B[u]``,
    and the whole vector for the state prior ``D``. The result is column-stochastic
    (or, for a 1-D vector, sums to 1).
    """
    counts = np.asarray(counts, dtype=float)
    if not np.all(np.isfinite(counts)):
        raise ValueError("Dirichlet counts must contain only finite values")
    if np.any(counts < 0.0):
        raise ValueError("Dirichlet counts must be non-negative")
    totals = counts.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Dirichlet counts must have positive mass along the normalized axis")
    return counts / (totals + _EPS)


def expected_A(a: np.ndarray) -> np.ndarray:
    r"""Likelihood ``A = E[Dir(a)]`` — column-normalize the ``(O, C)`` pseudocounts."""
    return dirichlet_expected_value(np.asarray(a, dtype=float), axis=0)


def expected_B(b: np.ndarray) -> np.ndarray:
    r"""Transition ``B = E[Dir(b)]`` — column-normalize ``(C, C)`` or each ``(U, C, C)`` slice."""
    b = np.asarray(b, dtype=float)
    axis = 1 if b.ndim == 3 else 0      # normalize the "from-state" (column) axis
    return dirichlet_expected_value(b, axis=axis)


def expected_D(d: np.ndarray) -> np.ndarray:
    r"""State prior ``D = E[Dir(d)]`` — normalize the ``(C,)`` pseudocount vector."""
    return dirichlet_expected_value(np.asarray(d, dtype=float), axis=0)


def accumulate_a_counts(o: np.ndarray, s: np.ndarray) -> np.ndarray:
    r"""Single-step likelihood evidence ``o ∘ s`` (outer product, book Eq. 4a term).

    ``o`` is the observation (one-hot or belief, ``(O,)``) and ``s`` the state belief
    ``(C,)``; the outer product ``o sᵀ`` is the ``(O, C)`` table of state–observation
    co-occurrence mass added to the ``a`` pseudocounts. Summing it over a trial and
    adding to ``a`` is the ``A``-learning update.
    """
    return np.outer(np.asarray(o, dtype=float), np.asarray(s, dtype=float))


def accumulate_b_counts(s_curr: np.ndarray, s_prev: np.ndarray) -> np.ndarray:
    r"""Single-step transition evidence ``s^{(τ)} ∘ s^{(τ-1)}`` (book Eq. 4b/6 term).

    The ``(C, C)`` outer product ``s_curr s_prevᵀ`` counts the (next-state, current-state)
    transition mass; ``[i, j]`` is evidence for moving *to* ``i`` *from* ``j`` — matching the
    column-stochastic convention ``B[i, j] = P(s'=i | s=j)``.
    """
    return np.outer(np.asarray(s_curr, dtype=float), np.asarray(s_prev, dtype=float))


def accumulate_d_counts(s0: np.ndarray) -> np.ndarray:
    r"""Single-step state-prior evidence — the initial-state belief ``s^{(0)}`` (book Eq. 4c)."""
    return np.asarray(s0, dtype=float)


def novelty_matrix(a: np.ndarray) -> np.ndarray:
    r"""Parameter-novelty matrix ``W ≈ ½(1/a − 1/a₀)`` (book Eq. 12).

    The KL-divergence between the current Dirichlet belief about ``A`` and the belief
    *after* a hypothetical extra count, simplified. ``a₀`` has every entry of a column
    set to that column's sum, so ``W`` is large where a concentration parameter is small
    (few counts ⇒ low confidence ⇒ much to learn) and shrinks as pseudocounts grow.
    """
    a = np.asarray(a, dtype=float)
    a0 = a.sum(axis=0, keepdims=True)            # column sums, broadcast down each column
    return 0.5 * (1.0 / (a + _EPS) - 1.0 / (a0 + _EPS))


def parameter_novelty(a: np.ndarray, s: np.ndarray) -> float:
    r"""Expected parameter information gain about ``A`` for a predicted state (book Eq. 13b/19).

    ``novelty = o · (W s)`` with ``o = A s`` (``A = E[Dir(a)]``) and ``W`` the novelty matrix.
    It blends the per-pair KL-divergences ``W`` by how likely each state/observation is under
    the policy — the expected information gain from visiting this state. Larger when the
    agent is uncertain about the relevant state–observation mappings.
    """
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    A = expected_A(a)
    o = A @ s
    W = novelty_matrix(a)
    return float(o @ (W @ s))


def efe_with_novelty(model: POMDPModel, s: np.ndarray, C: np.ndarray, a: np.ndarray) -> float:
    r"""Novelty-augmented expected free energy (book Eq. 15): ``G = risk + ambiguity − novelty``.

    Extends the Chapter 9 EFE (:func:`expected_free_energy`) with the **parameter
    information-gain** term :func:`parameter_novelty`. Because novelty is an information
    *gain* it is **subtracted** — a naive agent with low pseudocounts is driven to visit
    state–observation pairs it knows least about (novelty-seeking exploration) before
    pursuing its preferences.
    """
    return efe_components(model, s, C, a=a).total


# ---------------------------------------------------------------------------
# §10.2 — Habit learning (baseline prior E) and policy precision (γ)
# ---------------------------------------------------------------------------
#
# Chapter 9 set the policy posterior equal to the EFE-derived prior,
# ``Q(π) = σ(−γ G)``. Chapter 10 §10.2 augments policy selection with three
# extra ingredients (book Eq. 20–22): the policy-dependent **variational free
# energy** ``F^{[p]}`` (evidence each policy actually accrued after seeing data),
# a **baseline prior / habit** ``E`` (a learned bias toward certain policies),
# and a **policy precision** ``γ`` (confidence in the EFE estimate, itself
# learnable from a Gamma prior, Eq. 23–25).


def policy_posterior_full(
    G: np.ndarray,
    *,
    F: Optional[np.ndarray] = None,
    E: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    r"""Full variational policy posterior (book Eq. 20–22): ``Q(π) = σ(log E − F − γ G)``.

    Generalizes the Chapter 9 :func:`policy_posterior` (which is the special case
    ``F = 0``, ``E`` uniform). The three terms:

    * ``G`` — expected free energy per policy (the future-looking EFE score, lower = better);
    * ``F`` — policy-dependent **variational** free energy (past/present evidence; ``None`` ⇒
      omitted, Eq. 20 reduces to the EFE-only prior);
    * ``E`` — **baseline prior / habit** over policies (``None`` ⇒ uniform, no bias);
    * ``gamma`` — **policy precision** scaling the EFE term (confidence in ``G``).

    Reproduces the book's Example 10.5 (Fig 10.2.3) exactly: with uniform ``E`` and ``F = 0``,
    raising ``γ`` from 0 (uniform) concentrates probability on the lowest-EFE policy; a
    non-uniform habit ``E`` shifts mass toward the preferred policies.
    """
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    G = np.asarray(G, dtype=float)
    if G.ndim != 1 or not np.all(np.isfinite(G)):
        raise ValueError("G must be a finite 1-D vector")
    n = G.shape[0]
    F_arr = np.zeros(n) if F is None else np.asarray(F, dtype=float)
    if F_arr.shape != (n,) or not np.all(np.isfinite(F_arr)):
        raise ValueError(f"F must be a finite vector of shape ({n},)")
    if E is None:
        log_E = np.zeros(n)                       # uniform habit ⇒ constant, drops out of σ
    else:
        E = _nonnegative_vector(E, name="E", length=n)
        log_E = np.log(E / E.sum() + _EPS)
    return softmax(log_E - F_arr - gamma * G)


def precision_gradient(
    G: np.ndarray,
    F: np.ndarray,
    gamma: float,
    *,
    E: Optional[np.ndarray] = None,
    beta0: float = 1.0,
) -> float:
    r"""Free-energy gradient w.r.t. the policy precision (book Eq. 23).

    .. math::
        \frac{\partial \mathcal F}{\partial \gamma}
          = (\beta - \beta_0) + (\pi - \pi_0)\cdot(-G),

    where ``β = 1/γ`` is the Gamma rate, ``π₀ = σ(log E − γ G)`` is the policy *prior*
    (before seeing data) and ``π = σ(log E − F − γ G)`` the policy *posterior* (after). The
    gradient tracks the mismatch between prior and posterior policy beliefs: when the
    observed evidence ``F`` agrees with the EFE ordering ``G`` the precision rises, when it
    disagrees the precision falls.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if beta0 <= 0.0:
        raise ValueError(f"beta0 must be positive, got {beta0}")
    G = np.asarray(G, dtype=float)
    beta = 1.0 / (gamma + _EPS)
    pi0 = policy_posterior_full(G, E=E, gamma=gamma)
    pi = policy_posterior_full(G, F=F, E=E, gamma=gamma)
    return float((beta - beta0) + (pi - pi0) @ (-G))


@dataclass(frozen=True)
class PrecisionResult:
    """Store learned policy precision, beta rate, iteration trace, and residual."""

    gamma: float                # learned policy precision γ = 1/β
    beta: float                 # learned Gamma rate β
    gamma_trace: np.ndarray     # γ per iteration
    converged: bool
    grad_final: float           # |∂F/∂γ| at the fixed point (self-consistency residual)


def learn_precision(
    G: np.ndarray,
    F: np.ndarray,
    *,
    E: Optional[np.ndarray] = None,
    beta0: float = 1.0,
    kappa: float = 0.25,
    n_iter: int = 64,
    tol: float = 1e-10,
) -> PrecisionResult:
    r"""Learn the policy precision ``γ`` by gradient descent on VFE (book Eq. 23–25).

    Iterates the Gamma-rate update (Eq. 24) ``β ← β − κ_γ ∂F/∂γ`` and reads off the precision
    as the Gamma mean (Eq. 25, with shape ``α = 1``) ``γ = 1/β``. The fixed point is
    self-consistent: ``∂F/∂γ → 0`` (asserted via ``grad_final``).

    The precision rises when the policy-dependent VFE ``F`` is *close* to the EFE ``G`` (the
    agent's expectations were confirmed ⇒ high confidence in ``G``) and falls when ``F`` is
    *far* from ``G`` (book Example 10.6). The exact magnitude depends on the Gamma prior rate
    ``beta0``; the relative ordering (close ⇒ higher γ than far) is convention-independent.
    """
    if beta0 <= 0.0:
        raise ValueError(f"beta0 must be positive, got {beta0}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    G = np.asarray(G, dtype=float)
    F = np.asarray(F, dtype=float)
    if G.shape != F.shape or G.ndim != 1:
        raise ValueError("G and F must be 1-D vectors with matching shape")
    beta = float(beta0)
    trace = [1.0 / beta]
    grad = float("inf")
    converged = False
    for _ in range(int(n_iter)):
        gamma = 1.0 / (beta + _EPS)
        grad = precision_gradient(G, F, gamma, E=E, beta0=beta0)
        beta = max(beta - kappa * grad, _EPS)     # β > 0 (Gamma rate constraint)
        trace.append(1.0 / beta)
        if abs(grad) < tol:
            converged = True
            break
    return PrecisionResult(gamma=1.0 / beta, beta=beta, gamma_trace=np.array(trace),
                           converged=converged, grad_final=abs(grad))


# ---------------------------------------------------------------------------
# §10.3 — Factorial depth (multiple state factors + observation modalities)
# ---------------------------------------------------------------------------
#
# Chapter 9 used a single hidden-state vector and a single observation. Real
# problems usually have several *independent* hidden **state factors** (e.g. "which
# machine pays" and "what did I just choose") that jointly generate several
# **observation modalities** (e.g. a hint, a reward, a proprioceptive echo). §10.3
# generalizes the POMDP to sets of arrays (book Eq. 32–38):
#
# * ``A = {A^(0), …, A^(M)}`` — one likelihood per modality; each ``A^(m)`` has shape
#   ``(O_m, C_0, …, C_N)`` and is conditioned on *all* state factors (axis 0 = the
#   observation, axes 1.. = the factors), column-normalized over the observation axis.
# * ``B = {B^(0), …, B^(N)}`` — one transition per factor, ``B^(n)`` of shape
#   ``(C_n, C_n, U_n)`` (or ``(C_n, C_n)`` for an uncontrollable factor).
# * ``C = {C^(0), …, C^(M)}`` — preferences per modality; ``D = {D^(0), …, D^(N)}`` priors.
#
# Inference is mean-field over the factors (Eq. 35): ``Q(s) = ∏_n Q(s^(n))``. Each
# factor's posterior is updated from its prior plus a likelihood message that averages
# ``log A^(m)`` over the *other* factors' current beliefs (Eq. 36/37) — variational
# message passing. Everything reduces exactly to the Chapter 9 single-factor case when
# ``N = M = 0`` (verified in the tests against the book's Eq. 15 weather posterior).


def _factor_letters(n: int) -> str:
    """Support validated numerical mechanics for the core inference model."""
    if n > 20:
        raise ValueError("factorial POMDP supports at most 20 state factors")
    return "abcdefghijklmnopqrst"[:n]


def factorial_expected_observation(A_m: np.ndarray, states: "list[np.ndarray]") -> np.ndarray:
    r"""Expected observation for one modality (book §10.3): contract ``A^(m)`` with all factors.

    ``A_m`` has shape ``(O_m, C_0, …, C_N)``; ``states`` is the list of per-factor belief
    vectors. Returns ``o^(m) = Σ_{c} A^(m)[:, c] ∏_n s^(n)[c_n]`` — the predicted observation
    distribution over modality ``m``. Reduces to ``A · s`` for a single factor.
    """
    out = np.asarray(A_m, dtype=float)
    for s in states:
        out = np.tensordot(out, np.asarray(s, dtype=float), axes=([1], [0]))
    return out


def factorial_likelihood_message(
    A_m: np.ndarray,
    obs_m: np.ndarray,
    states: "list[np.ndarray]",
    factor: int,
) -> np.ndarray:
    r"""Variational likelihood message into one state factor (book Eq. 36 likelihood term).

    Computes ``(E_{s∖n}[log A^(m)]) · o^(m)`` — averages ``log A^(m)`` over every state factor
    *except* ``factor`` (using their current marginals), then dots the resulting
    ``(O_m, C_factor)`` matrix with the observation ``obs_m`` to give a ``(C_factor,)`` message.
    This is the per-factor evidence that modality ``m`` contributes to factor ``factor``.
    """
    logA = np.log(np.asarray(A_m, dtype=float) + _EPS)
    n_factors = len(states)
    letters = _factor_letters(n_factors)
    in_sub = "z" + letters
    out_sub = "z" + letters[factor]
    ops = [logA]
    subs = [in_sub]
    for k in range(n_factors):
        if k != factor:
            ops.append(np.asarray(states[k], dtype=float))
            subs.append(letters[k])
    msg_matrix = np.einsum(",".join(subs) + "->" + out_sub, *ops)   # (O_m, C_factor)
    return np.asarray(obs_m, dtype=float) @ msg_matrix


def factorial_infer_states(
    model: "FactorialPOMDP",
    obs: "list[int | np.ndarray]",
    *,
    priors: "list[np.ndarray] | None" = None,
    n_iter: int = 64,
    tol: float = 1e-10,
) -> "list[np.ndarray]":
    r"""Mean-field hidden-state inference for a factorial POMDP (book Eq. 35–37).

    Coordinate-ascent variational message passing: each factor's posterior is
    ``s^(n) = σ(log prior^(n) + Σ_m message_m→n)`` (the fixed point of the Eq. 37 gradient
    descent), iterated across factors until convergence. ``obs`` is a list of per-modality
    observations (int index or one-hot/belief vector). ``priors`` overrides ``D`` (e.g. the
    ``B``-propagated beliefs in a dynamic model). Returns the list of factor posteriors.
    """
    A = model.A
    n_factors = model.n_factors
    obs_vecs = [
        one_hot(int(o), model.obs_sizes[m]) if np.ndim(o) == 0 else np.asarray(o, dtype=float)
        for m, o in enumerate(obs)
    ]
    base = priors if priors is not None else model.D
    s = [np.asarray(p, dtype=float).copy() for p in base]
    log_prior = [np.log(p + _EPS) for p in s]
    for _ in range(int(n_iter)):
        max_delta = 0.0
        for n in range(n_factors):
            msg = np.zeros(model.factor_sizes[n])
            for m in range(model.n_modalities):
                msg = msg + factorial_likelihood_message(A[m], obs_vecs[m], s, n)
            new = softmax(log_prior[n] + msg)
            max_delta = max(max_delta, float(np.max(np.abs(new - s[n]))))
            s[n] = new
        if max_delta < tol:
            break
    return s


def factorial_predict_states(
    model: "FactorialPOMDP",
    states: "list[np.ndarray]",
    actions: "list[int]",
) -> "list[np.ndarray]":
    r"""Propagate each state factor one step under its action (book §10.3, per-factor ``B``).

    For factor ``n`` with transition ``B^(n)`` and action ``actions[n]`` returns
    ``B^(n)[:, :, u] · s^(n)``. Uncontrollable factors (2-D ``B^(n)`` or a single action
    slice) ignore the action. Returns the list of predicted factor beliefs.
    """
    if model.B is None:
        raise ValueError("model has no B (transition) set")
    out = []
    for n, s in enumerate(states):
        Bn = np.asarray(model.B[n], dtype=float)
        if Bn.ndim == 2:
            out.append(Bn @ np.asarray(s, dtype=float))
        else:
            u = int(actions[n]) if n < len(actions) else 0
            out.append(Bn[:, :, u] @ np.asarray(s, dtype=float))
    return out


def factorial_modality_ambiguity(A_m: np.ndarray, states: "list[np.ndarray]") -> float:
    r"""Expected observation entropy for one modality (book Eq. 38a ambiguity term).

    ``Σ_{joint} s(joint) · H[A^(m)[:, joint]]`` with ``H`` the entropy over the observation
    axis — the expected ambiguity of modality ``m`` given the factorized state belief.
    Computed by contracting the per-joint-state entropy with the factor marginals.
    """
    A_m = np.asarray(A_m, dtype=float)
    H = -np.sum(A_m * np.log(A_m + _EPS), axis=0)   # (C_0,…,C_N) per-joint entropy
    out = H
    for s in states:
        out = np.tensordot(out, np.asarray(s, dtype=float), axes=([0], [0]))
    return float(out)


def factorial_efe(
    model: "FactorialPOMDP",
    states: "list[np.ndarray]",
    *,
    C: "list[np.ndarray] | None" = None,
) -> float:
    r"""Expected free energy for a factorial POMDP at one predicted step (book Eq. 38a).

    Sums **risk** (reward-seeking) and **ambiguity** (information-seeking) over every
    observation modality:

    .. math::
        G = \sum_m \Big[ o^{(m)}\cdot(\log o^{(m)} - \log C^{(m)}) + H^{(m)}\cdot s \Big],

    with ``o^(m)`` the expected observation (:func:`factorial_expected_observation`) and the
    ambiguity from :func:`factorial_modality_ambiguity`. Preferences ``C`` default to the
    model's; each ``C^(m)`` is used raw (ε-floored), matching the Chapter 9 convention.
    Reduces to :func:`expected_free_energy` for a single factor and modality.
    """
    prefs = C if C is not None else model.C
    if prefs is None:
        raise ValueError("factorial EFE needs preferences C (per modality)")
    total = 0.0
    for m in range(model.n_modalities):
        o = factorial_expected_observation(model.A[m], states)
        Cm = np.asarray(prefs[m], dtype=float)
        risk = float(o @ (np.log(o + _EPS) - np.log(Cm + _EPS)))
        ambiguity = factorial_modality_ambiguity(model.A[m], states)
        total += risk + ambiguity
    return total


@dataclass
class FactorialPOMDP:
    r"""A factorial discrete generative model — sets of arrays per factor/modality (book §10.3).

    Parameters
    ----------
    A : list of ndarray
        One likelihood per observation modality; ``A[m]`` has shape ``(O_m, C_0, …, C_N)``,
        normalized over the observation axis (axis 0) for every combination of state factors.
    D : list of ndarray
        One prior per state factor; ``D[n]`` is a categorical of length ``C_n``.
    B : list of ndarray, optional
        One transition per state factor; ``B[n]`` of shape ``(C_n, C_n, U_n)`` (controllable)
        or ``(C_n, C_n)`` (uncontrollable). ``None`` for the static case.
    C : list of ndarray, optional
        One preference vector per modality (used for expected free energy).
    """

    A: "list[np.ndarray]"
    D: "list[np.ndarray]"
    B: "list[np.ndarray] | None" = None
    C: "list[np.ndarray] | None" = None

    def __post_init__(self) -> None:
        if len(self.D) == 0:
            raise ValueError("D must contain at least one state-factor prior")
        if len(self.A) == 0:
            raise ValueError("A must contain at least one observation modality")
        self.A = [np.asarray(a, dtype=float) for a in self.A]
        self.D = [_probability_vector(d, name=f"D[{n}]") for n, d in enumerate(self.D)]
        n_factors = len(self.D)
        for m, a in enumerate(self.A):
            if a.ndim != n_factors + 1:
                raise ValueError(
                    f"A[{m}] must have ndim = n_factors+1 = {n_factors + 1}, got {a.ndim}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"A[{m}] must contain only finite values")
            if np.any(a < 0.0):
                raise ValueError(f"A[{m}] must be non-negative")
            if not np.allclose(a.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"A[{m}] must be normalized over the observation axis (axis 0)")
            for n in range(n_factors):
                if a.shape[n + 1] != self.D[n].shape[0]:
                    raise ValueError(f"A[{m}] axis {n + 1} must match factor {n} size")
        if self.B is not None:
            if len(self.B) != n_factors:
                raise ValueError(f"B must contain one transition per factor ({n_factors})")
            self.B = [np.asarray(b, dtype=float) for b in self.B]
            for n, b in enumerate(self.B):
                c = self.D[n].shape[0]
                if b.ndim == 2:
                    if b.shape != (c, c) or not is_stochastic_matrix(b):
                        raise ValueError(f"B[{n}] must be a column-stochastic ({c}, {c}) matrix")
                elif b.ndim == 3:
                    if b.shape[:2] != (c, c):
                        raise ValueError(f"B[{n}] first two axes must be ({c}, {c})")
                    for u in range(b.shape[2]):
                        if not is_stochastic_matrix(b[:, :, u]):
                            raise ValueError(f"B[{n}][:, :, {u}] must be column-stochastic")
                else:
                    raise ValueError(f"B[{n}] must be 2-D or 3-D, got shape {b.shape}")
        if self.C is not None:
            if len(self.C) != len(self.A):
                raise ValueError(f"C must contain one preference vector per modality ({len(self.A)})")
            self.C = [
                _nonnegative_vector(c, name=f"C[{m}]", length=self.A[m].shape[0])
                for m, c in enumerate(self.C)
            ]

    @property
    def n_factors(self) -> int:
        """Return dimensionality metadata for the validated probabilistic model."""
        return len(self.D)

    @property
    def n_modalities(self) -> int:
        """Return dimensionality metadata for the validated probabilistic model."""
        return len(self.A)

    @property
    def factor_sizes(self) -> "list[int]":
        """Return dimensionality metadata for the validated probabilistic model."""
        return [int(d.shape[0]) for d in self.D]

    @property
    def obs_sizes(self) -> "list[int]":
        """Return dimensionality metadata for the validated probabilistic model."""
        return [int(a.shape[0]) for a in self.A]


# ---------------------------------------------------------------------------
# §10.4 — Hierarchical depth (nested POMDP layers)
# ---------------------------------------------------------------------------
#
# §10.4 stacks POMDPs into ``L+1`` layers (book Eq. 39–50). A higher layer's state
# sets the *initial-state prior* of the layer below (Eq. 42, ``P_D(s^[0,l] | s^[1,l+1])``),
# so slow high-level states contextualize fast low-level dynamics — "nested time scales".
# Each layer keeps its own ``A^[l]``, ``B^[l]``, ``C^[l]`` and is inverted by the same
# machinery as a flat POMDP. We implement the most generic constraint (higher-level states
# constraining the layer below, no cross-layer learning), with per-layer VFE (Eq. 50),
# per-layer EFE (Eq. 43), and the layer-wise policy posterior (Eq. 49).


def hierarchical_layer_vfe(
    s: np.ndarray,
    obs: "int | np.ndarray",
    A: np.ndarray,
    *,
    prior: np.ndarray,
) -> float:
    r"""Per-layer variational free energy for a hierarchical POMDP layer (book Eq. 50).

    .. math::
        F^{[\pi,\tau,l]} = s \cdot \big(\log s - \log B[u]^{[l]} s^{[\tau-1]}
                                        - \log A^{[l]} \cdot o\big),

    here written with the (already ``B``-propagated) ``prior`` standing in for
    ``B[u]^{[l]} s^{[τ-1]}`` (``D^{[l]}`` at ``τ=0``). Identical in form to the flat discrete
    VFE (:func:`discrete_vfe`); the hierarchy enters through how ``prior`` is supplied by the
    layer above. ``A`` is the layer likelihood and ``obs`` the layer observation.
    """
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)
    prior = np.asarray(prior, dtype=float)
    o = one_hot(int(obs), A.shape[0]) if np.ndim(obs) == 0 else np.asarray(obs, dtype=float)
    log_lik = np.log(A.T @ o + _EPS)
    return float(s @ (np.log(s + _EPS) - np.log(prior + _EPS) - log_lik))


def hierarchical_layer_efe(A: np.ndarray, s: np.ndarray, C: np.ndarray) -> float:
    r"""Per-layer expected free energy for a hierarchical layer (book Eq. 43).

    ``G^{[π,τ,l]} = o·(log o − log C^{[l]}) + H^{[l]}·s`` with ``o = A^{[l]} s`` — the same
    risk + ambiguity decomposition as the flat case (:func:`expected_free_energy`), evaluated
    per layer. Lower ``G`` ⇒ the layer's policy is preferred.
    """
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)
    o = A @ s
    C = np.asarray(C, dtype=float)
    risk = float(o @ (np.log(o + _EPS) - np.log(C + _EPS)))
    H = -np.sum(A * np.log(A + _EPS), axis=0)
    ambiguity = float(H @ s)
    return risk + ambiguity


def hierarchical_policy_posterior(
    G: np.ndarray,
    *,
    F: "np.ndarray | None" = None,
    E: "np.ndarray | None" = None,
) -> np.ndarray:
    r"""Layer-wise variational policy posterior (book Eq. 49): ``σ(log E − Σ_τ F − Σ_τ G)``.

    ``G`` and ``F`` are per-policy totals already summed over the layer's planning horizon
    (pass the summed vectors). ``E`` is the layer's habit prior (uniform if ``None``); ``F``
    the per-policy VFE evidence (``None`` ⇒ EFE-only, the Eq. 45 prior). Thin wrapper over
    :func:`policy_posterior_full` that documents the hierarchical layer semantics.
    """
    return policy_posterior_full(np.asarray(G, dtype=float), F=F, E=E, gamma=1.0)


@dataclass
class HierarchicalPOMDP:
    r"""A stack of POMDP layers with top-down state→prior linking (book §10.4, Eq. 39–50).

    Parameters
    ----------
    layers : list of POMDPModel
        Layer ``0`` is the fastest/lowest; layer ``L`` the slowest/highest. Each layer is an
        ordinary :class:`POMDPModel` (its own ``A``/``B``/``C``/``D``).
    link : list of ndarray, optional
        ``link[l]`` has shape ``(C_l, C_{l+1})``: a column-stochastic map from a state of
        layer ``l+1`` to the **initial-state prior** of layer ``l`` (book Eq. 42). With a
        higher-layer belief ``s^{[l+1]}`` the lower layer's prior is ``link[l] · s^{[l+1]}``.
        ``None`` ⇒ each layer simply uses its own ``D`` (independent layers).
    """

    layers: "list[POMDPModel]"
    link: "list[np.ndarray] | None" = None

    def __post_init__(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("layers must contain at least one POMDPModel")
        if self.link is not None:
            if len(self.link) != max(self.n_layers - 1, 0):
                raise ValueError("link must contain one map per adjacent layer pair")
            self.link = [np.asarray(m, dtype=float) for m in self.link]
            for ell, m in enumerate(self.link):
                if not is_stochastic_matrix(m):
                    raise ValueError(f"link[{ell}] must be column-stochastic")
                if m.shape[0] != self.layers[ell].n_states:
                    raise ValueError(f"link[{ell}] rows must match layer {ell} states")
                if m.shape[1] != self.layers[ell + 1].n_states:
                    raise ValueError(f"link[{ell}] cols must match layer {ell + 1} states")

    @property
    def n_layers(self) -> int:
        """Return dimensionality metadata for the validated probabilistic model."""
        return len(self.layers)

    def layer_prior(self, ell: int, higher_belief: "np.ndarray | None" = None) -> np.ndarray:
        r"""Initial-state prior for layer ``ell`` (book Eq. 42).

        Returns ``link[ell] · higher_belief`` when a higher-layer belief and link are present
        (top-down contextualization), else the layer's own ``D``.
        """
        layer = self.layers[ell]
        if higher_belief is None or self.link is None or ell >= len(self.link):
            return np.asarray(layer.D, dtype=float)
        prior = np.asarray(self.link[ell], dtype=float) @ np.asarray(higher_belief, dtype=float)
        return prior / prior.sum()
