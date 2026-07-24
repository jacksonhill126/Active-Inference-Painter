r"""Active generalized filtering — continuous-state active inference (book Ch. 7).

Chapter 6 gave us *perception*: an agent that updates its belief ``μ_x`` to track a
hidden state. Chapter 7 adds **action** ``a``: the agent now also *changes the world*
so that reality conforms to its preferences. The two halves minimize the same
variational free energy:

.. math::
    \dot\mu_x = -\kappa_x\,\partial\mathcal F/\partial\mu_x      \quad\text{(perception)}\\
    \dot a    = -\kappa_a\,\partial\mathcal F/\partial a         \quad\text{(action)}

Free energy is not directly a function of ``a`` (action is not in the generative
model), so action descends free energy **through the sensory channel** by the chain
rule (book Eq. 7–9, 11):

.. math::
    \frac{\partial\mathcal F}{\partial a}
        = \frac{\partial y(a)}{\partial a}\,\frac{\partial\mathcal F}{\partial y}
        = \frac{\partial y(a)}{\partial a}\,\lambda_y\,\varepsilon_y
    \;\Rightarrow\;
    \dot a = -\kappa_a\,\frac{\partial y(a)}{\partial a}\,\lambda_y\,\varepsilon_y .

``∂y(a)/∂a`` is the **forward model** — the agent's belief about the sensory
consequence of acting. The simplest forward model (used here and in the book) is a
constant gain/sign; action simply pushes the generative process to cancel the sensory
prediction error.

The agent's **preference prior** ``v = η`` is a set-point encoded as the point attractor
of the state-transition model ``f_M(μ_x) = v − μ_x``: the agent *expects* to be at ``v``,
and acts until its sensations match that expectation (a Bayesian thermostat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generalized_filtering import (
    DynamicStateSpaceModel,
    GeneralizedVectorModel,
    flatten_generalized_coordinates,
    gf_free_energy,
    gf_free_energy_grad,
    gf_sensory_prediction_error,
    generalized_vector_free_energy,
    generalized_vector_free_energy_grad,
    generalized_vector_sensory_error,
)

__all__ = [
    "ActiveInferenceAgent",
    "action_gradient",
    "perception_gradient",
    "ai_free_energy",
    "MultivariateActiveInferenceAgent",
    "multivariate_perception_flow",
    "multivariate_action_gradient",
    "multivariate_ai_free_energy",
]


@dataclass(frozen=True)
class ActiveInferenceAgent:
    r"""A continuous-state active-inference agent (book §7.1–7.4).

    Parameters
    ----------
    perception_model : DynamicStateSpaceModel
        The agent's generative model. Its state-transition ``f`` encodes the
        **preference** as a point attractor, e.g. ``f_M(μ)=v−μ`` is
        ``LinearFunction(slope=-1, intercept=v)`` with set-point ``v``; ``g`` is the
        observation model. Precisions live on the model (``lambda_x``, ``lambda_y``).
    forward_model : float
        The forward model ``∂y/∂a`` — the (constant) sensory consequence of action.
    kappa_x : float
        Perception learning rate (the ``μ_x`` flow).
    kappa_a : float
        Action learning rate (the ``a`` flow).
    """

    perception_model: DynamicStateSpaceModel
    forward_model: float = 1.0
    kappa_x: float = 0.2
    kappa_a: float = 0.4

    @property
    def preference(self) -> float:
        """The set-point ``v`` (the intercept of the attractor ``f_M(μ)=v−μ``)."""
        return float(getattr(self.perception_model.f, "intercept", 0.0))


def perception_gradient(agent: ActiveInferenceAgent, y: float, mu: float) -> float:
    r"""Perception flow ``μ̇_x = −κ_x ∂F/∂μ_x`` (book Eq. 10a), via the §6.1 gradient."""
    return -agent.kappa_x * gf_free_energy_grad(agent.perception_model, y, mu)


def action_gradient(agent: ActiveInferenceAgent, y: float, mu: float) -> float:
    r"""Action flow ``ȧ = −κ_a (∂y/∂a) λ_y ε_y`` (book Eq. 9/11/17).

    Action descends free energy through the sensory channel: the forward model maps
    the sensory prediction error onto a control signal that nudges the world to cancel
    that error.
    """
    eps_y = gf_sensory_prediction_error(agent.perception_model, y, mu)
    return -agent.kappa_a * agent.forward_model * agent.perception_model.lambda_y * eps_y


def ai_free_energy(agent: ActiveInferenceAgent, y: float, mu: float) -> float:
    """The variational free energy the agent minimizes (perception objective, Eq. 7a)."""
    return gf_free_energy(agent.perception_model, y, mu)


@dataclass(frozen=True)
class MultivariateActiveInferenceAgent:
    r"""Vector active generalized-filtering agent in generalized coordinates (§7.5).

    Parameters
    ----------
    perception_model : GeneralizedVectorModel
        Multivariate generalized-coordinate generative model from Chapter 6.
    forward_model : ndarray, shape (M·D, A)
        The generalized sensory consequence ``∂ỹ/∂a`` of an action vector. It maps
        action dimensions into the state-major generalized sensory vector.
    kappa_x : float
        Perception learning rate for ``μ̃_x``.
    kappa_a : float
        Action learning rate for ``a``.
    """

    perception_model: GeneralizedVectorModel
    forward_model: np.ndarray
    kappa_x: float = 0.2
    kappa_a: float = 0.4

    def __post_init__(self) -> None:
        forward = np.asarray(self.forward_model, dtype=float)
        expected_rows = self.perception_model.embedding_dim * self.perception_model.dim_y
        if forward.ndim != 2 or forward.shape[0] != expected_rows:
            raise ValueError(f"forward_model shape {forward.shape} must be ({expected_rows}, action_dim)")
        if not np.all(np.isfinite(forward)):
            raise ValueError("forward_model must contain only finite values")
        if not (np.isfinite(self.kappa_x) and self.kappa_x >= 0.0):
            raise ValueError(f"kappa_x must be finite non-negative, got {self.kappa_x!r}")
        if not (np.isfinite(self.kappa_a) and self.kappa_a >= 0.0):
            raise ValueError(f"kappa_a must be finite non-negative, got {self.kappa_a!r}")

    @property
    def action_dim(self) -> int:
        """Return the dimensionality of the action vector."""
        return int(np.asarray(self.forward_model).shape[1])


def multivariate_perception_flow(
    agent: MultivariateActiveInferenceAgent, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Perception flow ``Dμ̃_x − κ_x ∂F/∂μ̃_x`` for Algorithm 7.5.1."""
    mu_flat = flatten_generalized_coordinates(mu_tilde)
    return (
        agent.perception_model.D @ mu_flat
        - agent.kappa_x * generalized_vector_free_energy_grad(agent.perception_model, y_tilde, mu_flat)
    )


def multivariate_action_gradient(
    agent: MultivariateActiveInferenceAgent, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Action flow ``ȧ = −κ_a (∂ỹ/∂a)^T Π̃_y ε̃_y`` for §7.5."""
    eps_y = generalized_vector_sensory_error(agent.perception_model, y_tilde, mu_tilde)
    forward = np.asarray(agent.forward_model, dtype=float)
    return -agent.kappa_a * (forward.T @ agent.perception_model.Pi_y @ eps_y)


def multivariate_ai_free_energy(
    agent: MultivariateActiveInferenceAgent, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> float:
    """Return the generalized VFE minimized by vector perception and action."""
    return generalized_vector_free_energy(agent.perception_model, y_tilde, mu_tilde)
