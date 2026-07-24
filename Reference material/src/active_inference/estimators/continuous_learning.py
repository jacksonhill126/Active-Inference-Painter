"""Chapter 8 estimators for continuous-state learning and attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.continuous_learning import (
    LearningAttentionModel,
    LearningAttentionState,
    learning_attention_free_energy,
    learning_attention_grad,
    log_precision_to_variance,
)

__all__ = [
    "LearningAttentionResult",
    "simulate_learning_attention",
]


@dataclass(frozen=True)
class LearningAttentionResult:
    """Store Chapter 8 learning-attention state, parameter, precision, and VFE traces."""

    mus: np.ndarray
    mu_thetas: np.ndarray
    mu_zetas: np.ndarray
    theta_velocities: np.ndarray
    zeta_velocities: np.ndarray
    predicted_observations: np.ndarray
    free_energies: np.ndarray
    eps_y: np.ndarray
    eps_x: np.ndarray
    variances_x: np.ndarray
    ys: np.ndarray
    dt: float

    @property
    def final_state(self) -> LearningAttentionState:
        """Return the final value of the stored inference trajectory."""
        return LearningAttentionState(
            float(self.mus[-1]),
            float(self.mu_thetas[-1]),
            float(self.mu_zetas[-1]),
        )

    @property
    def final_variance_x(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.variances_x[-1])

    def tracking_error(self, truth: np.ndarray, *, burn_in: int = 0) -> float:
        """Mean absolute hidden-state tracking error after ``burn_in``."""
        truth = np.asarray(truth, dtype=float)
        if truth.shape != self.mus.shape:
            raise ValueError(f"truth shape {truth.shape} != mus shape {self.mus.shape}")
        return float(np.mean(np.abs(self.mus[burn_in:] - truth[burn_in:])))


def _require_positive(name: str, value: float) -> float:
    """Validate a scalar precondition before numerical inference proceeds."""
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite positive, got {value!r}")
    return out


def _require_nonnegative(name: str, value: float) -> float:
    """Validate a scalar precondition before numerical inference proceeds."""
    out = float(value)
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be finite non-negative, got {value!r}")
    return out


def simulate_learning_attention(
    model: LearningAttentionModel,
    ys: np.ndarray,
    *,
    initial: LearningAttentionState,
    dt: float = 0.01,
    kappa_x: float = 0.5,
    kappa_theta: float = 0.5,
    kappa_zeta: float = 0.1,
    damping: float = 1.0,
    learning_gain: float = 4.0,
    attention_gain: float = 12.0,
) -> LearningAttentionResult:
    r"""Run Chapter 8 perception, learning, and attention on an observation stream.

    Hidden state perception uses a first-order gradient flow. The parameter and
    log-precision means use damped velocity flows, a compact implementation of the
    slower time scales described in Chapter 8.
    """
    ys = np.asarray(ys, dtype=float)
    if ys.ndim != 1 or ys.size == 0 or not np.all(np.isfinite(ys)):
        raise ValueError("ys must be a non-empty finite 1-D array")
    dt = _require_positive("dt", dt)
    kappa_x = _require_positive("kappa_x", kappa_x)
    kappa_theta = _require_nonnegative("kappa_theta", kappa_theta)
    kappa_zeta = _require_nonnegative("kappa_zeta", kappa_zeta)
    damping = _require_nonnegative("damping", damping)
    learning_gain = _require_positive("learning_gain", learning_gain)
    attention_gain = _require_positive("attention_gain", attention_gain)

    n = ys.shape[0]
    mus = np.empty(n)
    mu_thetas = np.empty(n)
    mu_zetas = np.empty(n)
    theta_vels = np.empty(n)
    zeta_vels = np.empty(n)
    mu_ys = np.empty(n)
    fes = np.empty(n)
    eps_y = np.empty(n)
    eps_x = np.empty(n)
    variances = np.empty(n)

    state = initial
    theta_v = 0.0
    zeta_v = 0.0
    for t, y in enumerate(ys):
        comp = learning_attention_free_energy(model, float(y), state)
        grad = learning_attention_grad(model, float(y), state)

        mus[t] = state.mu_x
        mu_thetas[t] = state.mu_theta
        mu_zetas[t] = state.mu_zeta
        theta_vels[t] = theta_v
        zeta_vels[t] = zeta_v
        mu_ys[t] = model.predict_obs(state)
        fes[t] = comp.free_energy
        eps_y[t] = comp.eps_y
        eps_x[t] = comp.eps_x
        variances[t] = comp.variance_x

        mu_x = state.mu_x - dt * kappa_x * grad[0]
        theta_v = theta_v + dt * (-damping * theta_v - learning_gain * kappa_theta * grad[1])
        zeta_v = zeta_v + dt * (-damping * zeta_v - attention_gain * kappa_zeta * grad[2])
        state = LearningAttentionState(
            mu_x=float(mu_x),
            mu_theta=float(state.mu_theta + dt * theta_v),
            mu_zeta=float(state.mu_zeta + dt * zeta_v),
        )

    return LearningAttentionResult(
        mus=mus,
        mu_thetas=mu_thetas,
        mu_zetas=mu_zetas,
        theta_velocities=theta_vels,
        zeta_velocities=zeta_vels,
        predicted_observations=mu_ys,
        free_energies=fes,
        eps_y=eps_y,
        eps_x=eps_x,
        variances_x=np.asarray(log_precision_to_variance(mu_zetas), dtype=float),
        ys=ys,
        dt=dt,
    )
