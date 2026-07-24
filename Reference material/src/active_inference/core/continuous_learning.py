r"""Chapter 8 continuous-state learning, attention, and hierarchical messages.

Chapter 8 extends generalized filtering from perception/action to the **triple
estimation** problem: infer hidden states, first-order model parameters, and
second-order precision parameters from the same sensory stream. The simplified
Laplace models here keep the book's essential structure while remaining small
enough for chapter examples:

* ``mu_x`` tracks the hidden state.
* ``mu_theta`` tracks a first-order observation parameter.
* ``mu_zeta`` is a log precision, so the learned variance is always positive.
* hierarchical continuous models pass prediction errors upward and predictions
  downward between two layers.

All gradients are derived from the scalar free-energy functions and are checked
against finite differences in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LearningAttentionModel",
    "LearningAttentionState",
    "LearningAttentionComponents",
    "LearningAttentionGradient",
    "log_precision_to_precision",
    "log_precision_to_variance",
    "learning_attention_free_energy",
    "learning_attention_grad",
    "learning_attention_grad_fd",
    "ContinuousHierarchyLayer",
    "HierarchicalContinuousModel",
    "HierarchicalMessageTerms",
    "hierarchical_continuous_free_energy",
    "hierarchical_continuous_grad",
    "hierarchical_continuous_grad_fd",
    "hierarchical_message_terms",
]


def _require_finite_scalar(name: str, value: float) -> float:
    """Validate a scalar precondition before numerical inference proceeds."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _require_positive_scalar(name: str, value: float) -> float:
    """Validate a scalar precondition before numerical inference proceeds."""
    out = _require_finite_scalar(name, value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Validate a scalar precondition before numerical inference proceeds."""
    out = _require_finite_scalar(name, value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return out


def log_precision_to_precision(zeta: np.ndarray | float) -> np.ndarray | float:
    r"""Convert log precision ``ζ`` to precision ``λ = exp(ζ)``.

    The log parameterization is the Chapter 8 attention device: optimizing ``ζ``
    can never produce a negative precision.
    """
    arr = np.asarray(zeta, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("zeta must contain only finite values")
    out = np.exp(arr)
    return float(out) if out.ndim == 0 else out


def log_precision_to_variance(zeta: np.ndarray | float) -> np.ndarray | float:
    r"""Convert log precision ``ζ`` to variance ``σ² = exp(-ζ)``."""
    arr = np.asarray(zeta, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("zeta must contain only finite values")
    out = np.exp(-arr)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class LearningAttentionState:
    r"""Variational means for Chapter 8 triple estimation.

    ``mu_x`` is the hidden-state mean, ``mu_theta`` is a first-order observation
    parameter, and ``mu_zeta`` is a second-order log precision.
    """

    mu_x: float
    mu_theta: float
    mu_zeta: float

    def as_vector(self) -> np.ndarray:
        """Return ``[mu_x, mu_theta, mu_zeta]`` for vector-gradient tests."""
        return np.array([self.mu_x, self.mu_theta, self.mu_zeta], dtype=float)

    @classmethod
    def from_vector(cls, values: np.ndarray) -> "LearningAttentionState":
        """Build the state dataclass from a length-three vector."""
        arr = np.asarray(values, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError("LearningAttentionState vector must have shape (3,) and be finite")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class LearningAttentionGradient:
    """Gradient of Chapter 8 free energy with respect to ``(mu_x, mu_theta, mu_zeta)``."""

    d_mu_x: float
    d_mu_theta: float
    d_mu_zeta: float

    def as_vector(self) -> np.ndarray:
        """Return the variational state as a numeric vector."""
        return np.array([self.d_mu_x, self.d_mu_theta, self.d_mu_zeta], dtype=float)


@dataclass(frozen=True)
class LearningAttentionModel:
    r"""Simplified Chapter 8 generative model for learning and attention.

    The observation model is ``g_M(mu_x, mu_theta) = mu_x - mu_theta``. The hidden
    state is regularized toward ``state_attractor`` and its precision is learned via
    the log precision ``mu_zeta``.
    """

    state_attractor: float
    theta_prior_mean: float = 0.0
    zeta_prior_mean: float = 0.0
    sigma2_y: float = 1.0
    sigma2_theta: float = 1.0
    sigma2_zeta: float = 1.0

    def __post_init__(self) -> None:
        _require_finite_scalar("state_attractor", self.state_attractor)
        _require_finite_scalar("theta_prior_mean", self.theta_prior_mean)
        _require_finite_scalar("zeta_prior_mean", self.zeta_prior_mean)
        _require_positive_scalar("sigma2_y", self.sigma2_y)
        _require_positive_scalar("sigma2_theta", self.sigma2_theta)
        _require_positive_scalar("sigma2_zeta", self.sigma2_zeta)

    @property
    def lambda_y(self) -> float:
        """Return the precision parameter implied by the validated variance."""
        return 1.0 / self.sigma2_y

    @property
    def lambda_theta(self) -> float:
        """Return the precision parameter implied by the validated variance."""
        return 1.0 / self.sigma2_theta

    @property
    def lambda_zeta(self) -> float:
        """Return the precision parameter implied by the validated variance."""
        return 1.0 / self.sigma2_zeta

    def predict_obs(self, state: LearningAttentionState) -> float:
        """Observation prediction ``g_M(mu_x, mu_theta) = mu_x - mu_theta``."""
        return float(state.mu_x - state.mu_theta)


@dataclass(frozen=True)
class LearningAttentionComponents:
    """Store Chapter 8 free-energy terms, errors, precision, and learned variance."""

    free_energy: float
    eps_y: float
    eps_x: float
    eps_theta: float
    eps_zeta: float
    precision_x: float
    variance_x: float
    sensory_term: float
    state_term: float
    theta_term: float
    zeta_term: float


def learning_attention_free_energy(
    model: LearningAttentionModel,
    y: float,
    state: LearningAttentionState,
) -> LearningAttentionComponents:
    r"""Evaluate Chapter 8 VFE for hidden-state, parameter, and precision means."""
    y = _require_finite_scalar("y", y)
    if not np.all(np.isfinite(state.as_vector())):
        raise ValueError("state values must be finite")
    precision_x = float(log_precision_to_precision(state.mu_zeta))
    variance_x = float(log_precision_to_variance(state.mu_zeta))
    eps_y = y - model.predict_obs(state)
    eps_x = state.mu_x - model.state_attractor
    eps_theta = state.mu_theta - model.theta_prior_mean
    eps_zeta = state.mu_zeta - model.zeta_prior_mean
    sensory = model.lambda_y * eps_y**2
    state_term = precision_x * eps_x**2
    theta_term = model.lambda_theta * eps_theta**2
    zeta_term = model.lambda_zeta * eps_zeta**2
    log_terms = (
        np.log(model.sigma2_y)
        - state.mu_zeta
        + np.log(model.sigma2_theta)
        + np.log(model.sigma2_zeta)
    )
    free_energy = 0.5 * (sensory + state_term + theta_term + zeta_term + log_terms)
    return LearningAttentionComponents(
        free_energy=float(free_energy),
        eps_y=float(eps_y),
        eps_x=float(eps_x),
        eps_theta=float(eps_theta),
        eps_zeta=float(eps_zeta),
        precision_x=precision_x,
        variance_x=variance_x,
        sensory_term=float(0.5 * sensory),
        state_term=float(0.5 * state_term),
        theta_term=float(0.5 * theta_term),
        zeta_term=float(0.5 * zeta_term),
    )


def learning_attention_grad(
    model: LearningAttentionModel,
    y: float,
    state: LearningAttentionState,
) -> np.ndarray:
    r"""Return analytic gradient dF/d(mu_x, mu_theta, mu_zeta) for triple estimation."""
    comp = learning_attention_free_energy(model, y, state)
    d_mu_x = -model.lambda_y * comp.eps_y + comp.precision_x * comp.eps_x
    d_mu_theta = model.lambda_y * comp.eps_y + model.lambda_theta * comp.eps_theta
    d_mu_zeta = 0.5 * (comp.precision_x * comp.eps_x**2 - 1.0) + (
        model.lambda_zeta * comp.eps_zeta
    )
    return LearningAttentionGradient(d_mu_x, d_mu_theta, d_mu_zeta).as_vector()


def learning_attention_grad_fd(
    model: LearningAttentionModel,
    y: float,
    state: LearningAttentionState,
    eps: float = 1e-5,
) -> np.ndarray:
    """Return central finite-difference gradient for checking Chapter 8 analytics."""
    eps = _require_positive_scalar("eps", eps)
    x = state.as_vector()
    grad = np.empty_like(x)
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        hi = learning_attention_free_energy(
            model, y, LearningAttentionState.from_vector(x + d)
        ).free_energy
        lo = learning_attention_free_energy(
            model, y, LearningAttentionState.from_vector(x - d)
        ).free_energy
        grad[j] = (hi - lo) / (2.0 * eps)
    return grad


@dataclass(frozen=True)
class ContinuousHierarchyLayer:
    """Lower continuous layer ``y = mu_x - obs_offset`` with a sensory precision."""

    obs_offset: float = 0.0
    sensory_precision: float = 1.0

    def __post_init__(self) -> None:
        _require_finite_scalar("obs_offset", self.obs_offset)
        _require_positive_scalar("sensory_precision", self.sensory_precision)

    def predict_obs(self, mu_x: float) -> float:
        """Evaluate the generative prediction for the supplied state."""
        return float(mu_x - self.obs_offset)


@dataclass(frozen=True)
class HierarchicalContinuousModel:
    r"""Two-layer continuous hierarchy for Chapter 8 message passing.

    ``belief = [mu_x, mu_v]``. The upper context ``mu_v`` predicts the lower hidden
    state through ``link_gain * mu_v``. Setting link/context precisions to zero gives
    the single-layer sensory model.
    """

    lower: ContinuousHierarchyLayer
    link_precision: float = 1.0
    context_prior_mean: float = 0.0
    context_precision: float = 1.0
    link_gain: float = 1.0

    def __post_init__(self) -> None:
        _require_nonnegative_scalar("link_precision", self.link_precision)
        _require_finite_scalar("context_prior_mean", self.context_prior_mean)
        _require_nonnegative_scalar("context_precision", self.context_precision)
        _require_finite_scalar("link_gain", self.link_gain)


@dataclass(frozen=True)
class HierarchicalMessageTerms:
    """Forward/backward message terms for a two-layer continuous hierarchy."""

    sensory_error: float
    link_error: float
    context_error: float
    bottom_up_error: float
    top_down_error: float
    context_error_weighted: float
    top_down_prediction: float
    gradient: np.ndarray


def _hier_belief(belief: np.ndarray) -> tuple[float, float]:
    """Support validated numerical mechanics for the core inference model."""
    arr = np.asarray(belief, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError("belief must be a finite vector with shape (2,)")
    return float(arr[0]), float(arr[1])


def hierarchical_message_terms(
    model: HierarchicalContinuousModel,
    y: float,
    belief: np.ndarray,
) -> HierarchicalMessageTerms:
    """Return forward errors and backward precision-weighted hierarchy messages."""
    y = _require_finite_scalar("y", y)
    mu_x, mu_v = _hier_belief(belief)
    sensory_error = y - model.lower.predict_obs(mu_x)
    top_down_prediction = model.link_gain * mu_v
    link_error = mu_x - top_down_prediction
    context_error = mu_v - model.context_prior_mean
    bottom_up_error = model.lower.sensory_precision * sensory_error
    top_down_error = model.link_precision * link_error
    context_weighted = model.context_precision * context_error
    grad_x = -bottom_up_error + top_down_error
    grad_v = -model.link_gain * top_down_error + context_weighted
    return HierarchicalMessageTerms(
        sensory_error=float(sensory_error),
        link_error=float(link_error),
        context_error=float(context_error),
        bottom_up_error=float(bottom_up_error),
        top_down_error=float(top_down_error),
        context_error_weighted=float(context_weighted),
        top_down_prediction=float(top_down_prediction),
        gradient=np.array([grad_x, grad_v], dtype=float),
    )


def hierarchical_continuous_free_energy(
    model: HierarchicalContinuousModel,
    y: float,
    belief: np.ndarray,
) -> float:
    """Return Laplace free energy for sensory, link, and context hierarchy errors."""
    terms = hierarchical_message_terms(model, y, belief)
    return float(0.5 * (
        model.lower.sensory_precision * terms.sensory_error**2
        + model.link_precision * terms.link_error**2
        + model.context_precision * terms.context_error**2
    ))


def hierarchical_continuous_grad(
    model: HierarchicalContinuousModel,
    y: float,
    belief: np.ndarray,
) -> np.ndarray:
    """Return analytic gradient of two-layer hierarchy free energy with respect to beliefs."""
    return hierarchical_message_terms(model, y, belief).gradient


def hierarchical_continuous_grad_fd(
    model: HierarchicalContinuousModel,
    y: float,
    belief: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Return central finite-difference gradient for validating hierarchy derivatives."""
    eps = _require_positive_scalar("eps", eps)
    arr = np.asarray(belief, dtype=float)
    grad = np.empty_like(arr)
    for j in range(2):
        d = np.zeros(2)
        d[j] = eps
        grad[j] = (
            hierarchical_continuous_free_energy(model, y, arr + d)
            - hierarchical_continuous_free_energy(model, y, arr - d)
        ) / (2.0 * eps)
    return grad
