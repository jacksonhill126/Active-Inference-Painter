r"""Online generalized filtering for perception (book §6.1, Algorithm 6.1.1).

A dynamic environment is simulated by Euler-integrating a stochastic state-space
*generative process*; the agent then filters the resulting observation stream by
taking one Euler step down the variational free energy at each time point. With a
high sensory precision relative to the prior precision, the belief ``μ_x`` tracks
the true external state ``x*`` — *passive perception in a dynamic world*.

See :mod:`active_inference.core.generalized_filtering` for the model, free energy,
and (derived, finite-difference-verified) gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.generalized_filtering import (
    DynamicStateSpaceModel,
    gf_free_energy,
    gf_free_energy_grad,
    gf_sensory_prediction_error,
    gf_state_prediction_error,
)
from ..core.predictive_coding import GenerativeFunction

__all__ = [
    "DynamicProcess",
    "simulate_dynamic_process",
    "GeneralizedFilterResult",
    "generalized_filter",
    "generalized_measurements_from_series",
    "GeneralizedVectorFilterResult",
    "generalized_vector_filter",
]


@dataclass(frozen=True)
class DynamicProcess:
    r"""A univariate stochastic generative process ``E`` (book Eq. 9).

    ``x'* = f_E(x*) + ω_x``  (state dynamics) and ``y = g_E(x*) + ω_y``
    (observation generation), with zero-mean Gaussian noise of standard deviation
    ``omega_x`` / ``omega_y``.
    """

    f: GenerativeFunction
    g: GenerativeFunction
    omega_x: float = 0.1
    omega_y: float = 0.1


@dataclass(frozen=True)
class DynamicProcessTrace:
    """Group validated active-inference quantities used by this module."""
    xs: np.ndarray   # (T+1,) true external state x*
    ys: np.ndarray   # (T+1,) observations
    dt: float


def simulate_dynamic_process(
    process: DynamicProcess,
    x0: float,
    *,
    n_steps: int,
    dt: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> DynamicProcessTrace:
    r"""Euler-integrate the generative process, returning ``(x*, y)`` trajectories.

    State update (Euler): ``x*^(t+1) = x*^(t) + Δt·f_E(x*^(t)) + ω_x·ξ`` with
    ``ξ ~ N(0,1)``; observation ``y^(t) = g_E(x*^(t)) + ω_y·ξ'``. The point attractor
    encoded in ``f_E`` (e.g. ``θ_x − x``) draws ``x*`` toward its fixed point.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    xs = np.empty(n_steps + 1, dtype=float)
    ys = np.empty(n_steps + 1, dtype=float)
    x = float(x0)
    for t in range(n_steps + 1):
        xs[t] = x
        ys[t] = float(np.asarray(process.g(x))) + process.omega_y * rng.standard_normal()
        flow = float(np.asarray(process.f(x)))
        x = x + dt * flow + process.omega_x * rng.standard_normal() * np.sqrt(dt)
    return DynamicProcessTrace(xs=xs, ys=ys, dt=dt)


@dataclass(frozen=True)
class GeneralizedFilterResult:
    """Store univariate generalized-filtering beliefs, predictions, errors, and VFE."""

    mus: np.ndarray            # (T+1,) belief mean μ_x over time
    mu_ys: np.ndarray          # (T+1,) predicted observation g_M(μ_x)
    free_energies: np.ndarray  # (T+1,) VFE over time
    eps_x: np.ndarray          # (T+1,) state prediction error
    eps_y: np.ndarray          # (T+1,) sensory prediction error
    ys: np.ndarray             # (T+1,) the observation stream filtered

    @property
    def final_mu(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.mus[-1])

    def tracking_error(self, truth: np.ndarray, *, burn_in: int = 0) -> float:
        """Mean absolute tracking error ``|μ_x − x*|`` after ``burn_in`` steps."""
        truth = np.asarray(truth, dtype=float)
        return float(np.mean(np.abs(self.mus[burn_in:] - truth[burn_in:])))


def generalized_filter(
    model: DynamicStateSpaceModel,
    ys: np.ndarray,
    *,
    dt: float = 0.01,
    kappa: float = 0.1,
    mu0: float = 0.0,
) -> GeneralizedFilterResult:
    r"""Filter an observation stream online (Algorithm 6.1.1, agent step).

    At each time point the belief takes ONE Euler step of the recognition flow
    ``μ̇_x = −κ ∂F/∂μ_x``::

        μ_x^(t+1) = μ_x^(t) + Δt·μ̇_x        (Eq. 8, descent convention)

    Parameters
    ----------
    model : DynamicStateSpaceModel
        The agent's dynamic generative model.
    ys : ndarray
        Observation stream (e.g. from :func:`simulate_dynamic_process`).
    dt : float
        Euler time-step (matches the process discretization).
    kappa : float
        Learning rate of the recognition flow.
    mu0 : float
        Initial belief ``μ_x^(0)``.
    """
    ys = np.asarray(ys, dtype=float)
    n = ys.shape[0]
    mus = np.empty(n, dtype=float)
    mu_ys = np.empty(n, dtype=float)
    fes = np.empty(n, dtype=float)
    ex = np.empty(n, dtype=float)
    ey = np.empty(n, dtype=float)

    mu = float(mu0)
    for t in range(n):
        y = float(ys[t])
        mus[t] = mu
        mu_ys[t] = model.predict_obs(mu)
        ex[t] = gf_state_prediction_error(model, mu)
        ey[t] = gf_sensory_prediction_error(model, y, mu)
        fes[t] = gf_free_energy(model, y, mu)
        # Recognition: one Euler step of μ̇_x = −κ ∂F/∂μ_x.
        mu = mu - dt * kappa * gf_free_energy_grad(model, y, mu)
    return GeneralizedFilterResult(mus=mus, mu_ys=mu_ys, free_energies=fes,
                                   eps_x=ex, eps_y=ey, ys=ys)


# ---------------------------------------------------------------------------
# §6.2 — multivariate generative process + online filter (vector states)
# ---------------------------------------------------------------------------

from ..core.generalized_filtering import (  # noqa: E402
    MultivariateDynamicModel,
    VectorFunction,
    mv_gf_free_energy,
    mv_gf_free_energy_grad,
)


@dataclass(frozen=True)
class MultivariateDynamicProcess:
    """A multivariate stochastic generative process ``E`` (book §6.2)."""

    f: VectorFunction
    g: VectorFunction
    omega_x: float = 0.1
    omega_y: float = 0.1


@dataclass(frozen=True)
class MultivariateProcessTrace:
    """Group validated active-inference quantities used by this module."""
    xs: np.ndarray   # (T+1, C) true external state x*
    ys: np.ndarray   # (T+1, D) observations
    dt: float


def simulate_multivariate_process(
    process: MultivariateDynamicProcess,
    x0: np.ndarray,
    *,
    n_steps: int,
    dt: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> MultivariateProcessTrace:
    """Euler-integrate a vector generative process (e.g. the Hooke's-law oscillator)."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x0, dtype=float).copy()
    C = x.shape[0]
    y0 = np.asarray(process.g(x), dtype=float)
    D = y0.shape[0]
    xs = np.empty((n_steps + 1, C), dtype=float)
    ys = np.empty((n_steps + 1, D), dtype=float)
    for t in range(n_steps + 1):
        xs[t] = x
        ys[t] = np.asarray(process.g(x), dtype=float) + process.omega_y * rng.standard_normal(D)
        flow = np.asarray(process.f(x), dtype=float)
        x = x + dt * flow + process.omega_x * rng.standard_normal(C) * np.sqrt(dt)
    return MultivariateProcessTrace(xs=xs, ys=ys, dt=dt)


@dataclass(frozen=True)
class MultivariateFilterResult:
    """Store multivariate generalized-filtering belief, observation, and VFE trajectories."""

    mus: np.ndarray            # (T+1, C)
    mu_ys: np.ndarray          # (T+1, D)
    free_energies: np.ndarray  # (T+1,)
    ys: np.ndarray             # (T+1, D)

    def tracking_error(self, truth: np.ndarray, *, burn_in: int = 0) -> float:
        """Mean Euclidean tracking error ``‖μ_x − x*‖`` after ``burn_in`` steps."""
        truth = np.asarray(truth, dtype=float)
        diff = self.mus[burn_in:] - truth[burn_in:]
        return float(np.mean(np.linalg.norm(diff, axis=1)))


def multivariate_generalized_filter(
    model: MultivariateDynamicModel,
    ys: np.ndarray,
    *,
    dt: float = 0.01,
    kappa: float = 1.0,
    mu0: Optional[np.ndarray] = None,
) -> MultivariateFilterResult:
    r"""Online multivariate generalized filtering (book §6.2, Eq. 14/15).

    One Euler step of ``μ̇_x = −κ ∂F/∂μ_x`` per time point, with the vector belief
    ``μ_x ∈ ℝ^C`` and Jacobian-based gradient.
    """
    ys = np.asarray(ys, dtype=float)
    n, D = ys.shape
    C = model.dim_x
    mu = np.zeros(C) if mu0 is None else np.asarray(mu0, dtype=float).copy()
    mus = np.empty((n, C), dtype=float)
    mu_ys = np.empty((n, D), dtype=float)
    fes = np.empty(n, dtype=float)
    for t in range(n):
        y = ys[t]
        mus[t] = mu
        mu_ys[t] = np.asarray(model.g(mu), dtype=float)
        fes[t] = mv_gf_free_energy(model, y, mu)
        mu = mu - dt * kappa * mv_gf_free_energy_grad(model, y, mu)
    return MultivariateFilterResult(mus=mus, mu_ys=mu_ys, free_energies=fes, ys=ys)


# ---------------------------------------------------------------------------
# §6.3–6.5 — generalized filtering in generalized coordinates
# ---------------------------------------------------------------------------

from ..core.generalized_filtering import (  # noqa: E402
    GeneralizedVectorModel,
    GeneralizedModel,
    flatten_generalized_coordinates,
    generalized_free_energy,
    generalized_free_energy_grad,
    generalized_vector_free_energy,
    generalized_vector_free_energy_grad,
    generalized_vector_sensory_error,
    generalized_vector_state_error,
    generalized_sensory_error,
    generalized_state_error,
    unflatten_generalized_coordinates,
)


@dataclass(frozen=True)
class GeneralizedFilterGCResult:
    """Trace of :func:`generalized_filter_gc` (one row per time point)."""

    mus: np.ndarray            # (T, M) generalized belief μ̃_x over time
    free_energies: np.ndarray  # (T,)
    eps_x: np.ndarray          # (T, M) generalized state prediction error
    eps_y: np.ndarray          # (T, M) generalized sensory prediction error
    ys: np.ndarray             # (T, M) generalized observations filtered

    @property
    def positions(self) -> np.ndarray:
        """The order-0 belief ``μ_x^[0]`` (the inferred position) over time."""
        return self.mus[:, 0]

    @property
    def velocities(self) -> np.ndarray:
        """The order-1 belief ``μ_x^[1]`` (the inferred velocity) over time."""
        return self.mus[:, 1]


def generalized_filter_gc(
    model: GeneralizedModel,
    ys_tilde: np.ndarray,
    *,
    dt: float = 0.01,
    kappa: float = 1.0,
    mu0_tilde: Optional[np.ndarray] = None,
) -> GeneralizedFilterGCResult:
    r"""Generalized filtering in generalized coordinates (book §6.5, Algorithm 6.5.1).

    Each time point the belief follows the generalized recognition flow (Eq. 51/52)::

        μ̃̇_x = D μ̃_x − κ ∂F/∂μ̃_x,     μ̃_x ← μ̃_x + Δt·μ̃̇_x

    The ``D μ̃_x`` term moves the belief along its own generalized motion (a reference
    frame locked to the state's trajectory); the gradient term corrects toward the data.

    Parameters
    ----------
    ys_tilde : ndarray, shape (T, M)
        Generalized observations (each row ``[y, y', y'', …]`` for one time point).
    mu0_tilde : ndarray, shape (M,), optional
        Initial generalized belief (defaults to zeros).
    """
    ys_tilde = np.asarray(ys_tilde, dtype=float)
    n, M = ys_tilde.shape
    D = model.D
    mu = np.zeros(M) if mu0_tilde is None else np.asarray(mu0_tilde, dtype=float).copy()
    mus = np.empty((n, M), dtype=float)
    fes = np.empty(n, dtype=float)
    ex = np.empty((n, M), dtype=float)
    ey = np.empty((n, M), dtype=float)
    for t in range(n):
        yt = ys_tilde[t]
        mus[t] = mu
        ex[t] = generalized_state_error(model, mu)
        ey[t] = generalized_sensory_error(model, yt, mu)
        fes[t] = generalized_free_energy(model, yt, mu)
        flow = D @ mu - kappa * generalized_free_energy_grad(model, yt, mu)
        mu = mu + dt * flow
    return GeneralizedFilterGCResult(mus=mus, free_energies=fes, eps_x=ex, eps_y=ey, ys=ys_tilde)


def generalized_measurements_from_series(ys: np.ndarray, *, embedding_dim: int, dt: float) -> np.ndarray:
    r"""Estimate generalized measurements ``ỹ`` from an observation time series (§6.3).

    Parameters
    ----------
    ys : ndarray, shape (T, D) or (T,)
        Observation samples ordered by time.
    embedding_dim : int
        Number of embedding orders to estimate, including order 0.
    dt : float
        Sampling interval used for finite-difference derivative estimates.

    Returns
    -------
    ndarray, shape (T, embedding_dim, D)
        ``ỹ[t, m, d]`` stores the ``m``-th temporal derivative estimate for
        observation dimension ``d``. Order 0 is the original observation.
    """
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim!r}")
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be finite positive, got {dt!r}")
    arr = np.asarray(ys, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"ys must have shape (T,) or (T, D), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("ys must contain only finite values")
    T, D = arr.shape
    out = np.zeros((T, embedding_dim, D), dtype=float)
    out[:, 0, :] = arr
    current = arr
    for order in range(1, embedding_dim):
        if T == 1:
            derivative = np.zeros_like(current)
        else:
            derivative = np.gradient(current, dt, axis=0, edge_order=1)
        out[:, order, :] = derivative
        current = derivative
    return out


@dataclass(frozen=True)
class GeneralizedVectorFilterResult:
    """Trace of multivariate generalized filtering in generalized coordinates."""

    mus: np.ndarray            # (T, M, C)
    free_energies: np.ndarray  # (T,)
    eps_x: np.ndarray          # (T, M, C)
    eps_y: np.ndarray          # (T, M, D)
    ys: np.ndarray             # (T, M, D)

    @property
    def positions(self) -> np.ndarray:
        """Return order-0 beliefs ``μ_x^[0]`` with shape ``(T, C)``."""
        return self.mus[:, 0, :]

    def tracking_error(self, truth: np.ndarray, *, burn_in: int = 0) -> float:
        """Mean Euclidean tracking error of order-0 beliefs after ``burn_in``."""
        truth = np.asarray(truth, dtype=float)
        diff = self.positions[burn_in:] - truth[burn_in:]
        return float(np.mean(np.linalg.norm(diff, axis=1)))


def generalized_vector_filter(
    model: GeneralizedVectorModel,
    ys_tilde: np.ndarray,
    *,
    dt: float = 0.01,
    kappa: float = 1.0,
    mu0_tilde: Optional[np.ndarray] = None,
) -> GeneralizedVectorFilterResult:
    r"""Run vector generalized filtering with correlated embedding orders (§6.6).

    The Euler update is the vector analogue of Eq. 51/52:
    ``μ̃ ← μ̃ + Δt·(D μ̃ − κ ∂F/∂μ̃)``. Public arrays are stored as
    ``(time, embedding_order, variable)`` tensors for readability.
    """
    ys_tilde = np.asarray(ys_tilde, dtype=float)
    expected_shape_tail = (model.embedding_dim, model.dim_y)
    if ys_tilde.ndim != 3 or ys_tilde.shape[1:] != expected_shape_tail:
        raise ValueError(f"ys_tilde shape {ys_tilde.shape} must be (T, {expected_shape_tail[0]}, {expected_shape_tail[1]})")
    if not np.all(np.isfinite(ys_tilde)):
        raise ValueError("ys_tilde must contain only finite values")
    if mu0_tilde is None:
        mu = flatten_generalized_coordinates(np.zeros((model.embedding_dim, model.dim_x)))
    else:
        mu = flatten_generalized_coordinates(mu0_tilde)
    if mu.shape[0] != model.embedding_dim * model.dim_x:
        raise ValueError("mu0_tilde has incompatible shape for model")

    n = ys_tilde.shape[0]
    mus = np.empty((n, model.embedding_dim, model.dim_x), dtype=float)
    fes = np.empty(n, dtype=float)
    ex = np.empty((n, model.embedding_dim, model.dim_x), dtype=float)
    ey = np.empty((n, model.embedding_dim, model.dim_y), dtype=float)
    for t in range(n):
        yt_flat = flatten_generalized_coordinates(ys_tilde[t])
        mus[t] = unflatten_generalized_coordinates(mu, model.embedding_dim, model.dim_x)
        ex[t] = unflatten_generalized_coordinates(
            generalized_vector_state_error(model, mu),
            model.embedding_dim,
            model.dim_x,
        )
        ey[t] = unflatten_generalized_coordinates(
            generalized_vector_sensory_error(model, yt_flat, mu),
            model.embedding_dim,
            model.dim_y,
        )
        fes[t] = generalized_vector_free_energy(model, yt_flat, mu)
        flow = model.D @ mu - kappa * generalized_vector_free_energy_grad(model, yt_flat, mu)
        mu = mu + dt * flow
    return GeneralizedVectorFilterResult(mus=mus, free_energies=fes, eps_x=ex, eps_y=ey, ys=ys_tilde)
