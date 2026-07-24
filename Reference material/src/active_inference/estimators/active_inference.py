r"""The active generalized filter — the action-perception loop (book Algorithm 7.2.1).

Unlike Chapter 6 (where the observation stream is pre-generated), here the agent's
**action feeds back into the generative process**, so the environment and agent must
be simulated together in one online loop:

1. the environment steps forward, its dynamics driven by the agent's current action;
2. the agent perceives (updates ``μ_x``) and acts (updates ``a``), both by descending
   variational free energy;
3. the new action is applied to the environment on the next step.

With action enabled, the true external state is driven to the agent's preferred
set-point ``v``; with action disabled it follows its uncontrolled (exogenous) dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.active_inference import (
    ActiveInferenceAgent,
    MultivariateActiveInferenceAgent,
    action_gradient,
    multivariate_action_gradient,
    multivariate_ai_free_energy,
    multivariate_perception_flow,
    perception_gradient,
)
from ..core.generalized_filtering import (
    VectorFunction,
    flatten_generalized_coordinates,
    generalized_vector_sensory_error,
    gf_free_energy,
    gf_sensory_prediction_error,
    unflatten_generalized_coordinates,
)
from ..core.predictive_coding import GenerativeFunction
from .generalized_filtering import generalized_measurements_from_series

__all__ = [
    "ActiveEnvironment",
    "ActiveInferenceResult",
    "simulate_active_inference",
    "MultivariateActiveEnvironment",
    "MultivariateActiveInferenceResult",
    "simulate_multivariate_active_inference",
]


@dataclass(frozen=True)
class ActiveEnvironment:
    r"""The generative process for active inference (book Eq. 1, 15, 18).

    The state evolves by its own drift plus the agent's action force,
    ``x*^(t+1) = x*^(t) + Δt·(drift(x*) + a) + ω_x``, and emits ``y = g_E(x*) + ω_y``.
    The ``drift`` (e.g. ``v* − x*``, a point attractor at the exogenous force ``v*``) is
    what the agent must counteract to reach its own preference.
    """

    drift: GenerativeFunction
    g: GenerativeFunction
    omega_x: float = 0.05
    omega_y: float = 0.05


@dataclass(frozen=True)
class ActiveInferenceResult:
    """Store coupled perception-action trajectory, observations, errors, and free energies."""

    xs: np.ndarray             # (T+1,) true external state x*
    mus: np.ndarray            # (T+1,) belief μ_x
    actions: np.ndarray        # (T+1,) action a
    ys: np.ndarray             # (T+1,) observations
    free_energies: np.ndarray  # (T+1,)
    eps_y: np.ndarray          # (T+1,) sensory prediction error
    action_start: int

    def settled_state(self, *, tail: int = 200) -> float:
        """Mean true state over the last ``tail`` steps (the controlled set-point)."""
        return float(np.mean(self.xs[-tail:]))

    def settled_action(self, *, tail: int = 200) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(np.mean(self.actions[-tail:]))


def simulate_active_inference(
    agent: ActiveInferenceAgent,
    env: ActiveEnvironment,
    *,
    x0: float,
    mu0: float,
    n_steps: int = 6000,
    dt: float = 0.01,
    action_start: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> ActiveInferenceResult:
    r"""Run the coupled environment–agent loop (Algorithm 7.2.1).

    Parameters
    ----------
    action_start : int
        Step at which action turns on (before it, the agent only perceives — useful for
        showing the contrast). Set to ``n_steps`` (or pass ``kappa_a=0``) for a
        perception-only run.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = n_steps + 1
    xs = np.empty(n)
    mus = np.empty(n)
    acts = np.empty(n)
    ys = np.empty(n)
    fes = np.empty(n)
    eys = np.empty(n)

    x, mu, a = float(x0), float(mu0), 0.0
    for t in range(n):
        # Environment emits an observation from the current true state.
        y = float(np.asarray(env.g(x))) + env.omega_y * rng.standard_normal()
        xs[t], mus[t], acts[t], ys[t] = x, mu, a, y
        eys[t] = gf_sensory_prediction_error(agent.perception_model, y, mu)
        fes[t] = gf_free_energy(agent.perception_model, y, mu)

        # Agent step: perception (always) and action (after action_start).
        mu = mu + dt * perception_gradient(agent, y, mu)
        if t >= action_start:
            a = a + dt * action_gradient(agent, y, mu)

        # Environment step: drift + the agent's action force.
        drift = float(np.asarray(env.drift(x)))
        x = x + dt * (drift + a) + env.omega_x * rng.standard_normal() * np.sqrt(dt)

    return ActiveInferenceResult(xs=xs, mus=mus, actions=acts, ys=ys, free_energies=fes,
                                 eps_y=eys, action_start=action_start)


@dataclass(frozen=True)
class MultivariateActiveEnvironment:
    r"""Vector generative process for active generalized filtering (§7.5).

    The state follows ``x* ← x* + Δt·(drift(x*) + B_a a)`` and emits
    ``y = g_E(x*) + ω_y``. ``action_matrix`` is the explicit mapping from action
    coordinates to environmental state flow.
    """

    drift: VectorFunction
    g: VectorFunction
    action_matrix: np.ndarray
    omega_x: float = 0.0
    omega_y: float = 0.0


@dataclass(frozen=True)
class MultivariateActiveInferenceResult:
    """Store vector active generalized-filtering trajectories, actions, errors, and VFE."""

    xs: np.ndarray             # (T+1, C)
    mus: np.ndarray            # (T+1, M, C)
    actions: np.ndarray        # (T+1, A)
    ys: np.ndarray             # (T+1, D)
    y_tildes: np.ndarray       # (T+1, M, D)
    free_energies: np.ndarray  # (T+1,)
    eps_y: np.ndarray          # (T+1, M, D)
    action_start: int

    def settled_state(self, *, tail: int = 200) -> np.ndarray:
        """Mean true vector state over the last ``tail`` steps."""
        return np.mean(self.xs[-tail:], axis=0)

    def settled_action(self, *, tail: int = 200) -> np.ndarray:
        """Mean action vector over the last ``tail`` steps."""
        return np.mean(self.actions[-tail:], axis=0)

    def preference_error(self, preference: np.ndarray, *, tail: int = 200) -> float:
        """Euclidean distance between settled state and preferred state."""
        return float(np.linalg.norm(self.settled_state(tail=tail) - np.asarray(preference, dtype=float)))


def simulate_multivariate_active_inference(
    agent: MultivariateActiveInferenceAgent,
    env: MultivariateActiveEnvironment,
    *,
    x0: np.ndarray,
    mu0_tilde: Optional[np.ndarray] = None,
    n_steps: int = 2000,
    dt: float = 0.01,
    action_start: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> MultivariateActiveInferenceResult:
    r"""Run Algorithm 7.5.1 with vector states and generalized coordinates.

    Observations arrive as ordinary vector samples. The agent constructs generalized
    measurements online from the observation history, then performs one perception
    Euler step and, after ``action_start``, one action Euler step.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be finite positive, got {dt!r}")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("x0 must be a vector")
    C = x.shape[0]
    action_matrix = np.asarray(env.action_matrix, dtype=float)
    if action_matrix.shape != (C, agent.action_dim):
        raise ValueError(f"action_matrix shape {action_matrix.shape} must be ({C}, {agent.action_dim})")
    if mu0_tilde is None:
        mu = flatten_generalized_coordinates(np.zeros((agent.perception_model.embedding_dim, agent.perception_model.dim_x)))
    else:
        mu = flatten_generalized_coordinates(mu0_tilde)
    if mu.shape[0] != agent.perception_model.embedding_dim * agent.perception_model.dim_x:
        raise ValueError("mu0_tilde has incompatible shape for agent perception model")

    y0 = np.asarray(env.g(x), dtype=float)
    D = y0.shape[0]
    if D != agent.perception_model.dim_y:
        raise ValueError(f"environment observation dimension {D} != model dim_y {agent.perception_model.dim_y}")

    n = n_steps + 1
    M = agent.perception_model.embedding_dim
    xs = np.empty((n, C), dtype=float)
    mus = np.empty((n, M, agent.perception_model.dim_x), dtype=float)
    actions = np.empty((n, agent.action_dim), dtype=float)
    ys = np.empty((n, D), dtype=float)
    y_tildes = np.empty((n, M, D), dtype=float)
    fes = np.empty(n, dtype=float)
    eps_y = np.empty((n, M, D), dtype=float)

    a = np.zeros(agent.action_dim, dtype=float)
    y_history: list[np.ndarray] = []
    for t in range(n):
        y = np.asarray(env.g(x), dtype=float) + env.omega_y * rng.standard_normal(D)
        y_history.append(y)
        y_tilde = generalized_measurements_from_series(np.vstack(y_history), embedding_dim=M, dt=dt)[-1]
        y_flat = flatten_generalized_coordinates(y_tilde)

        xs[t] = x
        mus[t] = unflatten_generalized_coordinates(mu, M, agent.perception_model.dim_x)
        actions[t] = a
        ys[t] = y
        y_tildes[t] = y_tilde
        fes[t] = multivariate_ai_free_energy(agent, y_flat, mu)
        eps_y[t] = unflatten_generalized_coordinates(
            generalized_vector_sensory_error(agent.perception_model, y_flat, mu),
            M,
            D,
        )

        mu = mu + dt * multivariate_perception_flow(agent, y_flat, mu)
        if t >= action_start:
            a = a + dt * multivariate_action_gradient(agent, y_flat, mu)

        drift = np.asarray(env.drift(x), dtype=float)
        x = x + dt * (drift + action_matrix @ a) + env.omega_x * rng.standard_normal(C) * np.sqrt(dt)

    return MultivariateActiveInferenceResult(
        xs=xs,
        mus=mus,
        actions=actions,
        ys=ys,
        y_tildes=y_tildes,
        free_energies=fes,
        eps_y=eps_y,
        action_start=action_start,
    )
