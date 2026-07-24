r"""Generalized filtering for perception — the dynamic state-space model (book §6.1).

Chapter 6 moves predictive coding (Chapter 5) from a *static* hidden state to a
*dynamic* one that is filtered online: at each time point the environment
transitions and emits an observation, and the agent updates its belief ``μ_x`` by
one Euler step down the variational free energy. This module provides the
univariate machinery of §6.1 (Example 6.1); the generative model reuses the
:class:`~active_inference.core.predictive_coding.GenerativeFunction` abstraction
from Chapter 5.

The dynamic generative model :math:`\mathcal M` (book Eq. 10) is

.. math::
    p(\mu_x) = \mathcal N(\mu_x;\, f_{\mathcal M}(\mu_x;\theta_x),\, s_x^2)
        \qquad
    p(y\mid\mu_x) = \mathcal N(y;\, g_{\mathcal M}(\mu_x;\theta_y),\, \sigma_y^2)

i.e. the static prior mean ``m_x`` of Chapter 5 is replaced by a **state-transition
function** ``f_M`` that predicts the state's flow. The Laplace/quadratic free energy
and its prediction errors (Eq. 6, 7a) are

.. math::
    \varepsilon_x = \mu_x - f_{\mathcal M}(\mu_x;\theta_x), \qquad
    \varepsilon_y = y - g_{\mathcal M}(\mu_x;\theta_y), \\
    \mathcal F = \tfrac12\left(\lambda_y\varepsilon_y^2 + \lambda_x\varepsilon_x^2
        + \log(\sigma_y^2 s_x^2)\right).

**Derived gradient (not transcribed).** As in Chapter 5 the book's printed gradient
sign convention (Eq. 7b) is loose, so we derive ``∂F/∂μ_x`` by the chain rule and
verify it against a finite difference in the tests:

.. math::
    \frac{\partial\mathcal F}{\partial\mu_x}
        = \lambda_x\varepsilon_x\,(1 - f_{\mathcal M}'(\mu_x))
        - \lambda_y\varepsilon_y\,g_{\mathcal M}'(\mu_x).

Recognition descends this gradient as a *flow* ``μ̇_x = −κ ∂F/∂μ_x`` and integrates
it with Euler's method, ``μ_x ← μ_x + Δt·μ̇_x`` (Eq. 8, Algorithm 6.1.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from .predictive_coding import GenerativeFunction, LinearFunction

__all__ = [
    "DynamicStateSpaceModel",
    "gf_state_prediction_error",
    "gf_sensory_prediction_error",
    "gf_free_energy",
    "gf_free_energy_grad",
    "gf_free_energy_grad_fd",
    "gf_fixed_point_linear",
    # §6.2 — multivariate generalized filter
    "VectorFunction",
    "LinearVectorFunction",
    "GenericVectorFunction",
    "MultivariateDynamicModel",
    "mv_gf_free_energy",
    "mv_gf_free_energy_grad",
    "mv_gf_free_energy_grad_fd",
    "mv_gf_fixed_point_linear",
    # §6.3–6.5 — generalized coordinates of motion
    "shift_operator",
    "embed_flow",
    "gaussian_temporal_covariance",
    "correlated_embedding_precision",
    "flatten_generalized_coordinates",
    "unflatten_generalized_coordinates",
    "GeneralizedModel",
    "generalized_state_error",
    "generalized_sensory_error",
    "generalized_free_energy",
    "generalized_free_energy_grad",
    "generalized_free_energy_grad_fd",
    "GeneralizedVectorModel",
    "embed_vector_flow",
    "generalized_vector_state_error",
    "generalized_vector_sensory_error",
    "generalized_vector_free_energy",
    "generalized_vector_free_energy_grad",
    "generalized_vector_free_energy_grad_fd",
]


@dataclass(frozen=True)
class DynamicStateSpaceModel:
    r"""A univariate dynamic generative model for generalized filtering (§6.1).

    Parameters
    ----------
    f : GenerativeFunction
        State-transition (flow) function ``f_M(μ_x; θ_x)`` predicting the state's
        temporal derivative. For Example 6.1, ``f_M(μ)=θ_x-μ`` is a point attractor
        — ``LinearFunction(slope=-1, intercept=θ_x)``.
    g : GenerativeFunction
        Observation function ``g_M(μ_x; θ_y)``. For Example 6.1, ``g_M(μ)=μ-θ_y`` —
        ``LinearFunction(slope=1, intercept=-θ_y)``.
    s2_x : float
        State variance ``s_x²`` (> 0); inverse is the state precision ``λ_x``.
    sigma2_y : float
        Observation variance ``σ_y²`` (> 0); inverse is the sensory precision ``λ_y``.
    """

    f: GenerativeFunction
    g: GenerativeFunction
    s2_x: float = 1.0
    sigma2_y: float = 1.0

    def __post_init__(self) -> None:
        if not (self.s2_x > 0) or not np.isfinite(self.s2_x):
            raise ValueError(f"s2_x must be finite positive, got {self.s2_x!r}")
        if not (self.sigma2_y > 0) or not np.isfinite(self.sigma2_y):
            raise ValueError(f"sigma2_y must be finite positive, got {self.sigma2_y!r}")

    @property
    def lambda_x(self) -> float:
        """State precision ``λ_x = 1/s_x²``."""
        return 1.0 / self.s2_x

    @property
    def lambda_y(self) -> float:
        """Sensory precision ``λ_y = 1/σ_y²``."""
        return 1.0 / self.sigma2_y

    def predict_state(self, mu: float) -> float:
        """Predicted flow ``f_M(μ_x)``."""
        return float(np.asarray(self.f(float(mu))))

    def predict_obs(self, mu: float) -> float:
        """Predicted observation ``μ_y = g_M(μ_x)``."""
        return float(np.asarray(self.g(float(mu))))


def gf_state_prediction_error(model: DynamicStateSpaceModel, mu: float) -> float:
    r"""State prediction error ``ε_x = μ_x − f_M(μ_x)`` (book Eq. 6)."""
    return float(mu) - model.predict_state(mu)


def gf_sensory_prediction_error(model: DynamicStateSpaceModel, y: float, mu: float) -> float:
    r"""Sensory prediction error ``ε_y = y − g_M(μ_x)`` (book Eq. 6)."""
    return float(y) - model.predict_obs(mu)


@dataclass(frozen=True)
class _GFFreeEnergy:
    """Group validated active-inference quantities used by this module."""
    free_energy: float
    eps_x: float
    eps_y: float


def gf_free_energy(model: DynamicStateSpaceModel, y: float, mu: float) -> float:
    r"""Laplace/quadratic variational free energy ``F`` at ``μ_x`` (book Eq. 7a)."""
    eps_x = gf_state_prediction_error(model, mu)
    eps_y = gf_sensory_prediction_error(model, y, mu)
    log_term = np.log(model.sigma2_y * model.s2_x)
    return 0.5 * (model.lambda_y * eps_y**2 + model.lambda_x * eps_x**2 + log_term)


def gf_free_energy_grad(model: DynamicStateSpaceModel, y: float, mu: float) -> float:
    r"""Analytic ``∂F/∂μ_x`` (derived by chain rule; verified vs finite difference).

    .. math::
        \frac{\partial\mathcal F}{\partial\mu_x}
            = \lambda_x\varepsilon_x(1 - f_{\mathcal M}'(\mu_x))
            - \lambda_y\varepsilon_y g_{\mathcal M}'(\mu_x).
    """
    eps_x = gf_state_prediction_error(model, mu)
    eps_y = gf_sensory_prediction_error(model, y, mu)
    fprime = float(np.asarray(model.f.derivative(float(mu))))
    gprime = float(np.asarray(model.g.derivative(float(mu))))
    return model.lambda_x * eps_x * (1.0 - fprime) - model.lambda_y * eps_y * gprime


def gf_free_energy_grad_fd(
    model: DynamicStateSpaceModel, y: float, mu: float, eps: float = 1e-5
) -> float:
    """Return scalar central finite-difference gradient for validating ``gf_free_energy``."""
    f_hi = gf_free_energy(model, y, mu + eps)
    f_lo = gf_free_energy(model, y, mu - eps)
    return (f_hi - f_lo) / (2.0 * eps)


def gf_fixed_point_linear(model: DynamicStateSpaceModel, y: float) -> float:
    r"""Closed-form recognition fixed point ``μ*`` for **linear** ``f_M`` and ``g_M``.

    With ``f_M(μ)=a_f μ + b_f`` and ``g_M(μ)=a_g μ + b_g`` the free energy is convex in
    ``μ`` and ``∂F/∂μ=0`` gives

    .. math::
        \mu^* = \frac{\lambda_x(1-a_f)b_f + \lambda_y a_g (y - b_g)}
                     {\lambda_x(1-a_f)^2 + \lambda_y a_g^2}.

    This is the steady-state of the agent's inner relaxation at a held observation —
    the analytical landmark the tests verify the filter against. Raises ``TypeError``
    for nonlinear ``f``/``g``.
    """
    if not (isinstance(model.f, LinearFunction) and isinstance(model.g, LinearFunction)):
        raise TypeError(
            "gf_fixed_point_linear requires LinearFunction f and g; for nonlinear "
            "components use the grid argmin of gf_free_energy."
        )
    a_f, b_f = model.f.slope, model.f.intercept
    a_g, b_g = model.g.slope, model.g.intercept
    lx, ly = model.lambda_x, model.lambda_y
    num = lx * (1.0 - a_f) * b_f + ly * a_g * (float(y) - b_g)
    den = lx * (1.0 - a_f) ** 2 + ly * a_g**2
    return num / den


# ---------------------------------------------------------------------------
# §6.2 — the multivariate generalized filter (vector states / observations)
# ---------------------------------------------------------------------------


class VectorFunction:
    """A differentiable map ``ℝ^C → ℝ^k`` with a Jacobian (for multivariate ``f``/``g``)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Return the local derivative used by gradient-based inference."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearVectorFunction(VectorFunction):
    r"""Affine map ``h(x) = A x + b`` with constant Jacobian ``A``."""

    A: np.ndarray
    b: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(x, dtype=float) + self.b

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return the local derivative used by gradient-based inference."""
        return np.asarray(self.A, dtype=float)


@dataclass(frozen=True)
class GenericVectorFunction(VectorFunction):
    """Arbitrary vector map with a central finite-difference Jacobian fallback."""

    fn: object
    out_dim: int
    eps: float = 1e-5

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(x, dtype=float)), dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return the local derivative used by gradient-based inference."""
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        J = np.empty((self.out_dim, n), dtype=float)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = self.eps
            J[:, j] = (self(x + dx) - self(x - dx)) / (2.0 * self.eps)
        return J


def _as_precision_matrix(p, dim: int) -> np.ndarray:
    """Coerce a scalar / (d,) vector / (d,d) matrix into a (dim, dim) precision matrix."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        out = np.eye(dim) * float(arr)
        _validate_precision_matrix(out, name="precision")
        return out
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"precision vector length {arr.shape[0]} != dim {dim}")
        out = np.diag(arr)
        _validate_precision_matrix(out, name="precision")
        return out
    if arr.shape != (dim, dim):
        raise ValueError(f"precision matrix shape {arr.shape} != ({dim}, {dim})")
    _validate_precision_matrix(arr, name="precision")
    return arr


def _validate_precision_matrix(matrix: np.ndarray, *, name: str) -> None:
    """Validate that ``matrix`` is finite, symmetric, and positive definite."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _infer_precision_dim(p) -> int:
    """Infer the base random-variable dimension from a scalar, vector, or matrix precision."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        return 1
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return int(arr.shape[0])
    raise ValueError(f"precision must be scalar, vector, or square matrix, got shape {arr.shape}")


def _as_generalized_precision_matrix(p, embedding_dim: int, dim: int) -> np.ndarray:
    """Coerce generalized-coordinate precision into state-major ``(dim*M, dim*M)`` form."""
    total = embedding_dim * dim
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        return _as_precision_matrix(arr, total)
    if arr.ndim == 1:
        if arr.shape[0] == total:
            return _as_precision_matrix(arr, total)
        if arr.shape[0] == embedding_dim:
            return _as_precision_matrix(np.kron(np.eye(dim), np.diag(arr)), total)
        if arr.shape[0] == dim:
            return _as_precision_matrix(np.kron(np.diag(arr), np.eye(embedding_dim)), total)
        raise ValueError(
            f"generalized precision vector length {arr.shape[0]} must be one of "
            f"{embedding_dim}, {dim}, or {total}"
        )
    if arr.ndim != 2:
        raise ValueError(f"generalized precision must be scalar, vector, or matrix, got {arr.ndim}D")
    if arr.shape == (total, total):
        return _as_precision_matrix(arr, total)
    if arr.shape == (embedding_dim, embedding_dim):
        return _as_precision_matrix(np.kron(np.eye(dim), arr), total)
    if arr.shape == (dim, dim):
        return _as_precision_matrix(np.kron(arr, np.eye(embedding_dim)), total)
    raise ValueError(
        f"generalized precision matrix shape {arr.shape} must be "
        f"({embedding_dim}, {embedding_dim}), ({dim}, {dim}), or ({total}, {total})"
    )


@dataclass(frozen=True)
class MultivariateDynamicModel:
    r"""A multivariate dynamic generative model (book §6.2, Eq. 12).

    ``f`` is the vector state-transition (flow) map ``ℝ^C → ℝ^C`` and ``g`` the
    observation map ``ℝ^C → ℝ^D``. ``precision_x`` / ``precision_y`` may be given as a
    scalar, a diagonal vector, or a full matrix; they are coerced to ``Π_x`` (C×C) and
    ``Π_y`` (D×D).
    """

    f: VectorFunction
    g: VectorFunction
    precision_x: np.ndarray
    precision_y: np.ndarray
    dim_x: int = 2
    dim_y: int = 2

    @property
    def Pi_x(self) -> np.ndarray:
        """Return the precision parameter implied by the validated variance."""
        return _as_precision_matrix(self.precision_x, self.dim_x)

    @property
    def Pi_y(self) -> np.ndarray:
        """Return the precision parameter implied by the validated variance."""
        return _as_precision_matrix(self.precision_y, self.dim_y)

    def errors(self, y: np.ndarray, mu: np.ndarray):
        """Return prediction-error terms for the current generalized state."""
        mu = np.asarray(mu, dtype=float)
        eps_x = mu - self.f(mu)
        eps_y = np.asarray(y, dtype=float) - self.g(mu)
        return eps_x, eps_y


def mv_gf_free_energy(model: MultivariateDynamicModel, y: np.ndarray, mu: np.ndarray) -> float:
    r"""Multivariate Laplace free energy ``F`` (book Eq. 12)."""
    eps_x, eps_y = model.errors(y, mu)
    Pi_x, Pi_y = model.Pi_x, model.Pi_y
    quad = eps_x @ Pi_x @ eps_x + eps_y @ Pi_y @ eps_y
    # log|Σ_x Σ_y| = −log|Π_x| − log|Π_y|
    logdet = -np.linalg.slogdet(Pi_x)[1] - np.linalg.slogdet(Pi_y)[1]
    return 0.5 * (quad + logdet)


def mv_gf_free_energy_grad(
    model: MultivariateDynamicModel, y: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    r"""Analytic ``∂F/∂μ_x`` via Jacobians (derived; verified vs finite difference).

    .. math::
        \frac{\partial\mathcal F}{\partial\mu_x}
            = (I - J_f)^\top \Pi_x \varepsilon_x - J_g^\top \Pi_y \varepsilon_y.
    """
    mu = np.asarray(mu, dtype=float)
    eps_x, eps_y = model.errors(y, mu)
    J_f = model.f.jacobian(mu)
    J_g = model.g.jacobian(mu)
    eye = np.eye(mu.shape[0])
    return (eye - J_f).T @ model.Pi_x @ eps_x - J_g.T @ model.Pi_y @ eps_y


def mv_gf_free_energy_grad_fd(
    model: MultivariateDynamicModel, y: np.ndarray, mu: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Return vector central finite-difference gradient for multivariate GF validation."""
    mu = np.asarray(mu, dtype=float)
    grad = np.empty_like(mu)
    for j in range(mu.shape[0]):
        d = np.zeros_like(mu)
        d[j] = eps
        grad[j] = (mv_gf_free_energy(model, y, mu + d) - mv_gf_free_energy(model, y, mu - d)) / (2 * eps)
    return grad


def mv_gf_fixed_point_linear(model: MultivariateDynamicModel, y: np.ndarray) -> np.ndarray:
    r"""Closed-form recognition fixed point for **linear** ``f``/``g`` (the oracle).

    With ``f(x)=A_f x + b_f``, ``g(x)=A_g x + b_g`` and ``M = I − A_f`` the stationary
    ``∂F/∂μ=0`` solves the linear system

    .. math::
        (M^\top\Pi_x M + A_g^\top\Pi_y A_g)\,x
            = M^\top\Pi_x b_f + A_g^\top\Pi_y (y - b_g).
    """
    if not (isinstance(model.f, LinearVectorFunction) and isinstance(model.g, LinearVectorFunction)):
        raise TypeError("mv_gf_fixed_point_linear requires LinearVectorFunction f and g.")
    y = np.asarray(y, dtype=float)
    A_f, b_f = np.asarray(model.f.A, float), np.asarray(model.f.b, float)
    A_g, b_g = np.asarray(model.g.A, float), np.asarray(model.g.b, float)
    Pi_x, Pi_y = model.Pi_x, model.Pi_y
    M = np.eye(A_f.shape[0]) - A_f
    lhs = M.T @ Pi_x @ M + A_g.T @ Pi_y @ A_g
    rhs = M.T @ Pi_x @ b_f + A_g.T @ Pi_y @ (y - b_g)
    return np.linalg.solve(lhs, rhs)


# ---------------------------------------------------------------------------
# §6.3–6.5 — generalized coordinates of motion (the D operator + embedding)
# ---------------------------------------------------------------------------


def shift_operator(embedding_dim: int, n_states: int = 1) -> np.ndarray:
    r"""The derivative shift operator ``D`` (book Eq. 37).

    ``D μ̃_x = μ̃_x'`` shifts a generalized-coordinate vector *up* one embedding order
    and places a zero for the last order. Formally ``D = I_C ⊗ S`` where ``S`` is the
    ``M×M`` matrix with ones on the superdiagonal (``M = embedding_dim``). For the
    book's example ``D·[3,4,2,6,4] = [4,2,6,4,0]``.
    """
    S = np.diag(np.ones(embedding_dim - 1), k=1)
    if n_states == 1:
        return S
    return np.kron(np.eye(n_states), S)


def gaussian_temporal_covariance(embedding_dim: int, gamma: float) -> np.ndarray:
    r"""Temporal covariance ``S(γ)`` for Gaussian-correlated embedding orders (§6.6).

    The book's §6.6 forms ``S`` from derivatives of the Gaussian autocorrelation
    function ``ρ(t;γ)=exp(-γ t² / 4)`` at zero. Entry ``S[i,j]`` is
    ``(-1)^j ρ^(i+j)(0)``: odd derivatives vanish, while even derivatives produce the
    alternating powers that yield the printed leading block
    ``[[1, 0, -γ/2], [0, γ/2, 0], [-γ/2, 0, 3γ²/4]]``.

    Parameters
    ----------
    embedding_dim : int
        Number of generalized-coordinate orders, including order 0.
    gamma : float
        Positive finite smoothness/roughness parameter controlling inter-order
        correlations.

    Returns
    -------
    ndarray, shape (embedding_dim, embedding_dim)
        Symmetric positive-definite temporal covariance matrix.
    """
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim!r}")
    gamma = float(gamma)
    if not (np.isfinite(gamma) and gamma > 0.0):
        raise ValueError(f"gamma must be finite positive, got {gamma!r}")
    a = gamma / 4.0
    out = np.zeros((embedding_dim, embedding_dim), dtype=float)
    for i in range(embedding_dim):
        for j in range(embedding_dim):
            order = i + j
            if order % 2:
                continue
            half = order // 2
            derivative = ((-1.0) ** half) * factorial(order) / factorial(half) * (a**half)
            out[i, j] = ((-1.0) ** j) * derivative
    _validate_precision_matrix(out, name="temporal covariance")
    return out


def correlated_embedding_precision(
    precision,
    embedding_dim: int,
    *,
    gamma: float,
    layout: str = "state_major",
) -> np.ndarray:
    r"""Generalized precision ``Π̃_i(γ)`` for correlated embedding orders (§6.6).

    ``precision`` is the base precision ``Π_i`` over state or observation variables.
    For the book/order-major layout this returns ``S(γ)^-1 ⊗ Π_i``.  The repository's
    generalized-coordinate vectors are state-major to match :func:`shift_operator`,
    so the default returns the equivalent ``Π_i ⊗ S(γ)^-1``.
    """
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim!r}")
    base_dim = _infer_precision_dim(precision)
    base = _as_precision_matrix(precision, base_dim)
    temporal_precision = np.linalg.inv(gaussian_temporal_covariance(embedding_dim, gamma))
    if layout == "order_major":
        out = np.kron(temporal_precision, base)
    elif layout == "state_major":
        out = np.kron(base, temporal_precision)
    else:
        raise ValueError("layout must be 'state_major' or 'order_major'")
    out = 0.5 * (out + out.T)
    _validate_precision_matrix(out, name="correlated embedding precision")
    return out


def flatten_generalized_coordinates(values: np.ndarray) -> np.ndarray:
    """Flatten educational ``(embedding_order, variable)`` arrays into state-major vectors."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if not np.all(np.isfinite(arr)):
            raise ValueError("generalized-coordinate vector must contain finite values")
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError(f"generalized coordinates must be 1-D or 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("generalized-coordinate matrix must contain finite values")
    return arr.T.reshape(-1)


def unflatten_generalized_coordinates(values: np.ndarray, embedding_dim: int, dim: int) -> np.ndarray:
    """Unflatten a state-major vector into educational ``(embedding_order, variable)`` form."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (embedding_dim, dim):
            raise ValueError(f"generalized matrix shape {arr.shape} != ({embedding_dim}, {dim})")
        if not np.all(np.isfinite(arr)):
            raise ValueError("generalized-coordinate matrix must contain finite values")
        return arr.copy()
    if arr.ndim != 1 or arr.shape[0] != embedding_dim * dim:
        raise ValueError(
            f"state-major generalized vector shape {arr.shape} must be ({embedding_dim * dim},)"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("generalized-coordinate vector must contain finite values")
    return arr.reshape(dim, embedding_dim).T


def embed_flow(f_val: float, f_prime: float, mu_tilde: np.ndarray) -> np.ndarray:
    r"""Embed a (state-transition or observation) function into generalized coordinates.

    Under the local-linearity approximation (book Eq. 30/36), the generalized function
    ``f̃(μ̃)`` is ``[f(μ_x), f'(μ_x)·μ_x', f'(μ_x)·μ_x'', …]`` — order 0 is the function
    value, every higher order is the slope times that order's motion.
    """
    mu_tilde = np.asarray(mu_tilde, dtype=float)
    out = f_prime * mu_tilde.copy()
    out[0] = f_val
    return out


@dataclass(frozen=True)
class GeneralizedModel:
    r"""Univariate dynamic model in generalized coordinates (book §6.3–6.5).

    The belief ``μ̃_x ∈ ℝ^{M}`` carries the state and its higher-order motions
    (``embedding_dim = M``). Precisions are per-embedding-order diagonal vectors
    ``precision_x`` / ``precision_y`` (length ``M``); §6.6 generalizes to correlated
    embedding orders.
    """

    f: GenerativeFunction
    g: GenerativeFunction
    precision_x: np.ndarray
    precision_y: np.ndarray
    embedding_dim: int = 3

    @property
    def D(self) -> np.ndarray:
        """Return the generalized-coordinate shift operator matrix."""
        return shift_operator(self.embedding_dim, 1)

    @property
    def Pi_x(self) -> np.ndarray:
        """Return the precision parameter implied by the validated variance."""
        return _as_generalized_precision_matrix(self.precision_x, self.embedding_dim, 1)

    @property
    def Pi_y(self) -> np.ndarray:
        """Return the precision parameter implied by the validated variance."""
        return _as_generalized_precision_matrix(self.precision_y, self.embedding_dim, 1)

    def embed_f(self, mu_tilde: np.ndarray) -> np.ndarray:
        """Embed a scalar function into generalized coordinates."""
        mu0 = float(mu_tilde[0])
        return embed_flow(float(np.asarray(self.f(mu0))),
                          float(np.asarray(self.f.derivative(mu0))), mu_tilde)

    def embed_g(self, mu_tilde: np.ndarray) -> np.ndarray:
        """Embed a scalar function into generalized coordinates."""
        mu0 = float(mu_tilde[0])
        return embed_flow(float(np.asarray(self.g(mu0))),
                          float(np.asarray(self.g.derivative(mu0))), mu_tilde)


def generalized_state_error(model: GeneralizedModel, mu_tilde: np.ndarray) -> np.ndarray:
    r"""Generalized state prediction error ``ε̃_x = D μ̃_x − f̃_M(μ̃_x)`` (book Eq. 46b)."""
    mu_tilde = np.asarray(mu_tilde, dtype=float)
    return model.D @ mu_tilde - model.embed_f(mu_tilde)


def generalized_sensory_error(
    model: GeneralizedModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Generalized sensory prediction error ``ε̃_y = ỹ − g̃_M(μ̃_x)`` (book Eq. 46a)."""
    return np.asarray(y_tilde, dtype=float) - model.embed_g(np.asarray(mu_tilde, dtype=float))


def generalized_free_energy(
    model: GeneralizedModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> float:
    r"""Generalized variational free energy ``F`` (book Eq. 45)."""
    eps_x = generalized_state_error(model, mu_tilde)
    eps_y = generalized_sensory_error(model, y_tilde, mu_tilde)
    Pi_x, Pi_y = model.Pi_x, model.Pi_y
    quad = eps_x @ Pi_x @ eps_x + eps_y @ Pi_y @ eps_y
    logdet = -np.linalg.slogdet(Pi_x)[1] - np.linalg.slogdet(Pi_y)[1]
    return 0.5 * (quad + logdet)


def generalized_free_energy_grad(
    model: GeneralizedModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Analytic ``∂F/∂μ̃_x`` in generalized coordinates (book Eq. 50a; derived).

    .. math::
        \frac{\partial\mathcal F}{\partial\tilde\mu_x}
            = (\mathcal D - f_{\mathcal M}'(\mu_x) I)^\top \tilde\Pi_x \tilde\varepsilon_x
            - (g_{\mathcal M}'(\mu_x) I)^\top \tilde\Pi_y \tilde\varepsilon_y.

    Uses the local-linearity approximation (the Jacobian of the embedded function is
    ``f'(μ_x) I``), which is *exact* for linear ``f``/``g`` — verified against a finite
    difference of :func:`generalized_free_energy` in the tests.
    """
    mu_tilde = np.asarray(mu_tilde, dtype=float)
    mu0 = float(mu_tilde[0])
    fprime = float(np.asarray(model.f.derivative(mu0)))
    gprime = float(np.asarray(model.g.derivative(mu0)))
    eye = np.eye(model.embedding_dim)
    eps_x = generalized_state_error(model, mu_tilde)
    eps_y = generalized_sensory_error(model, y_tilde, mu_tilde)
    return (model.D - fprime * eye).T @ model.Pi_x @ eps_x - (gprime * eye).T @ model.Pi_y @ eps_y


def generalized_free_energy_grad_fd(
    model: GeneralizedModel, y_tilde: np.ndarray, mu_tilde: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Return central finite-difference gradient for generalized-coordinate free energy."""
    mu_tilde = np.asarray(mu_tilde, dtype=float)
    grad = np.empty_like(mu_tilde)
    for j in range(mu_tilde.shape[0]):
        d = np.zeros_like(mu_tilde)
        d[j] = eps
        grad[j] = (generalized_free_energy(model, y_tilde, mu_tilde + d)
                   - generalized_free_energy(model, y_tilde, mu_tilde - d)) / (2 * eps)
    return grad


@dataclass(frozen=True)
class GeneralizedVectorModel:
    r"""Multivariate dynamic model in generalized coordinates (§6.6 / Example 6.7).

    The public educational representation is a matrix ``(M, C)`` for hidden states
    and ``(M, D)`` for observations, where rows index embedding order. Internally the
    model uses state-major vectors ``[x0, x0', ..., x1, x1', ...]`` so the derivative
    shift operator remains ``I_C ⊗ S`` as implemented by :func:`shift_operator`.
    Precisions may be scalar, diagonal vectors, base matrices, or full generalized
    matrices; correlated orders should be supplied by :func:`correlated_embedding_precision`.
    """

    f: VectorFunction
    g: VectorFunction
    precision_x: np.ndarray
    precision_y: np.ndarray
    embedding_dim: int = 3
    dim_x: int = 2
    dim_y: int = 2

    @property
    def D(self) -> np.ndarray:
        """Return the state-major generalized-coordinate shift operator."""
        return shift_operator(self.embedding_dim, self.dim_x)

    @property
    def Pi_x(self) -> np.ndarray:
        """Return the generalized hidden-state precision matrix."""
        return _as_generalized_precision_matrix(self.precision_x, self.embedding_dim, self.dim_x)

    @property
    def Pi_y(self) -> np.ndarray:
        """Return the generalized sensory precision matrix."""
        return _as_generalized_precision_matrix(self.precision_y, self.embedding_dim, self.dim_y)

    def embed_f(self, mu_tilde: np.ndarray) -> np.ndarray:
        """Embed the vector state-transition function ``f`` under local linearity."""
        return embed_vector_flow(self.f, mu_tilde, self.embedding_dim, self.dim_x)

    def embed_g(self, mu_tilde: np.ndarray) -> np.ndarray:
        """Embed the vector observation function ``g`` under local linearity."""
        return embed_vector_flow(self.g, mu_tilde, self.embedding_dim, self.dim_x)


def embed_vector_flow(
    fn: VectorFunction,
    mu_tilde: np.ndarray,
    embedding_dim: int,
    dim_x: int,
) -> np.ndarray:
    r"""Embed a vector function into state-major generalized coordinates.

    Under the local-linearity approximation, order 0 is ``fn(μ^[0])`` and each
    higher order is ``J_fn(μ^[0]) @ μ^[m]``. The returned vector uses the same
    state-major flattening as :func:`flatten_generalized_coordinates`.
    """
    mu_matrix = unflatten_generalized_coordinates(mu_tilde, embedding_dim, dim_x)
    mu0 = mu_matrix[0]
    f0 = np.asarray(fn(mu0), dtype=float)
    J = np.asarray(fn.jacobian(mu0), dtype=float)
    if J.shape[1] != dim_x:
        raise ValueError(f"function Jacobian input dimension {J.shape[1]} != {dim_x}")
    out_dim = f0.shape[0]
    out = np.empty((embedding_dim, out_dim), dtype=float)
    out[0] = f0
    for order in range(1, embedding_dim):
        out[order] = J @ mu_matrix[order]
    return flatten_generalized_coordinates(out)


def _embedded_vector_jacobian(fn: VectorFunction, mu_tilde: np.ndarray, embedding_dim: int, dim_x: int) -> np.ndarray:
    """Return the local-linear generalized Jacobian for a vector function."""
    mu_matrix = unflatten_generalized_coordinates(mu_tilde, embedding_dim, dim_x)
    J = np.asarray(fn.jacobian(mu_matrix[0]), dtype=float)
    return np.kron(J, np.eye(embedding_dim))


def generalized_vector_state_error(model: GeneralizedVectorModel, mu_tilde: np.ndarray) -> np.ndarray:
    r"""Vector state prediction error ``ε̃_x = D μ̃_x − f̃_M(μ̃_x)`` (§6.6)."""
    mu_flat = flatten_generalized_coordinates(mu_tilde)
    if mu_flat.shape[0] != model.embedding_dim * model.dim_x:
        raise ValueError("mu_tilde has incompatible shape for model")
    return model.D @ mu_flat - model.embed_f(mu_flat)


def generalized_vector_sensory_error(
    model: GeneralizedVectorModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Vector sensory prediction error ``ε̃_y = ỹ − g̃_M(μ̃_x)`` (§6.6)."""
    y_flat = flatten_generalized_coordinates(y_tilde)
    if y_flat.shape[0] != model.embedding_dim * model.dim_y:
        raise ValueError("y_tilde has incompatible shape for model")
    return y_flat - model.embed_g(flatten_generalized_coordinates(mu_tilde))


def generalized_vector_free_energy(
    model: GeneralizedVectorModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> float:
    r"""Full-matrix generalized VFE for vector states with correlated orders (Eq. 61)."""
    eps_x = generalized_vector_state_error(model, mu_tilde)
    eps_y = generalized_vector_sensory_error(model, y_tilde, mu_tilde)
    Pi_x, Pi_y = model.Pi_x, model.Pi_y
    quad = eps_x @ Pi_x @ eps_x + eps_y @ Pi_y @ eps_y
    logdet = -np.linalg.slogdet(Pi_x)[1] - np.linalg.slogdet(Pi_y)[1]
    return 0.5 * (quad + logdet)


def generalized_vector_free_energy_grad(
    model: GeneralizedVectorModel, y_tilde: np.ndarray, mu_tilde: np.ndarray
) -> np.ndarray:
    r"""Analytic ``∂F/∂μ̃_x`` for vector generalized filtering (Eq. 50/61)."""
    mu_flat = flatten_generalized_coordinates(mu_tilde)
    eps_x = generalized_vector_state_error(model, mu_flat)
    eps_y = generalized_vector_sensory_error(model, y_tilde, mu_flat)
    J_f = _embedded_vector_jacobian(model.f, mu_flat, model.embedding_dim, model.dim_x)
    J_g = _embedded_vector_jacobian(model.g, mu_flat, model.embedding_dim, model.dim_x)
    return (model.D - J_f).T @ model.Pi_x @ eps_x - J_g.T @ model.Pi_y @ eps_y


def generalized_vector_free_energy_grad_fd(
    model: GeneralizedVectorModel, y_tilde: np.ndarray, mu_tilde: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Return central finite-difference gradient for vector generalized VFE."""
    mu_flat = flatten_generalized_coordinates(mu_tilde)
    grad = np.empty_like(mu_flat)
    for j in range(mu_flat.shape[0]):
        d = np.zeros_like(mu_flat)
        d[j] = eps
        grad[j] = (
            generalized_vector_free_energy(model, y_tilde, mu_flat + d)
            - generalized_vector_free_energy(model, y_tilde, mu_flat - d)
        ) / (2.0 * eps)
    return grad
