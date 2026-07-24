r"""Predictive coding — the core methods of Chapter 5.

Predictive coding re-derives variational inference (Chapter 4) as the minimization
of **precision-weighted prediction errors**.  The starting point is the E-form of
variational free energy (Chapter 4, Eq. 29) under two simplifications the book
makes in §5.1:

* the variational density is **fixed-form Gaussian**, and
* we care only about its *mean* :math:`\mu_x` (a Dirac-:math:`\delta` / Laplace
  point mass), so the entropy term vanishes.

What remains is the **MAP/Laplace free energy** (book Eq. 3):

.. math::

    \mathcal F_{\mathrm{MAP}} = -\log p(y \mid x=\mu_x) - \log p(x=\mu_x)

which, for Gaussian likelihood and prior, expands (book Eq. 5/7a) to

.. math::

    \mathcal F_{\mathrm{MAP}} = \tfrac12\!\left(\log\sigma_y^2
        + \frac{\varepsilon_y^2}{\sigma_y^2}
        + \log s_x^2 + \frac{\varepsilon_x^2}{s_x^2}\right) + \text{const}

with the two **prediction errors** (book Eq. 6)

.. math::

    \varepsilon_y = y - g(\mu_x) \quad\text{(sensory)}, \qquad
    \varepsilon_x = \mu_x - m_x \quad\text{(state)}.

Here ``g`` is the observation **generating function** (e.g. the linear
``g(x)=β₁x+β₀`` of Chapters 3–4, or the nonlinear ``g(x)=x²+b`` of Example 5.3).

**Sign discipline (important).** The book's gradient (Eq. 16) is written with a
sign convention that wobbles between Eq. 6b, the p.283 text, and Eq. 16.  Rather
than transcribe a sign, we *derive* the gradient by the chain rule and **verify it
against a central finite difference of the loss** (:func:`pc_free_energy_grad` is
checked against :func:`predictive_coding_free_energy` in the tests).  With
:math:`\varepsilon_y = y - g(\mu_x)` and :math:`\varepsilon_x = \mu_x - m_x`,

.. math::

    \frac{\partial \mathcal F_{\mathrm{MAP}}}{\partial \mu_x}
        = \lambda_x\,\varepsilon_x - \lambda_y\,\varepsilon_y\,g'(\mu_x),
    \qquad \lambda \equiv 1/\sigma^2 .

Gradient descent ``μ ← μ − κ ∂F/∂μ`` therefore moves the belief *up* the Jacobian-
weighted sensory error and *down* the state error — precision-weighted prediction-
error message passing, the simplest perception equation in continuous active
inference (book Eq. 16c).

**Cross-chapter oracle.** For a *linear* ``g`` and Gaussian likelihood/prior the
MAP estimate equals the exact posterior *mean*; so the fixed point of predictive
coding is exactly Chapter 4's :class:`~active_inference.core.inference.GridBayesianInference`
posterior mode/mean.  That identity is the verification oracle for the whole
chapter (see ``tests/estimators/test_predictive_coding.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Generating functions  g(x)  with derivatives  g'(x)
# ---------------------------------------------------------------------------


class GenerativeFunction:
    r"""An observation generating function ``g(x)`` with its derivative ``g'(x)``.

    Predictive coding needs the Jacobian :math:`\partial g/\partial \mu_x` to turn a
    sensory prediction error into a state update (book Eq. 16).  Subclasses provide
    an analytic derivative; :class:`GenericFunction` falls back to a central finite
    difference so *any* differentiable callable can be used.
    """

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the local derivative used by gradient-based inference."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearFunction(GenerativeFunction):
    r"""``g(x) = slope · x + intercept`` with constant derivative ``slope``.

    This is the Chapters 3–4 observation model: with ``slope=β₁`` and
    ``intercept=β₀`` it is ``g(x)=β₁x+β₀`` (Example 5.1 uses ``g(x)=2x+3``).
    """

    slope: float = 1.0
    intercept: float = 0.0

    def __call__(self, x):
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def derivative(self, x):
        """Return the local derivative used by gradient-based inference."""
        return np.full_like(np.asarray(x, dtype=float), self.slope)


@dataclass(frozen=True)
class QuadraticFunction(GenerativeFunction):
    r"""``g(x) = a · x² + b`` with derivative ``g'(x) = 2 a x`` (Example 5.3 uses ``x²+1``)."""

    a: float = 1.0
    b: float = 0.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return self.a * x**2 + self.b

    def derivative(self, x):
        """Return the local derivative used by gradient-based inference."""
        return 2.0 * self.a * np.asarray(x, dtype=float)


@dataclass(frozen=True)
class GenericFunction(GenerativeFunction):
    r"""Wrap any callable ``fn`` (and optional analytic ``dfn``).

    When ``dfn`` is omitted the derivative is a central finite difference with step
    ``eps`` — faithful to the book's "any differentiable generating function" remark
    (after Eq. 16c) while keeping the package autodiff-free.
    """

    fn: Callable[[np.ndarray], np.ndarray]
    dfn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    eps: float = 1e-5

    def __call__(self, x):
        return np.asarray(self.fn(np.asarray(x, dtype=float)), dtype=float)

    def derivative(self, x):
        """Return the local derivative used by gradient-based inference."""
        if self.dfn is not None:
            return np.asarray(self.dfn(np.asarray(x, dtype=float)), dtype=float)
        x = np.asarray(x, dtype=float)
        return (self(x + self.eps) - self(x - self.eps)) / (2.0 * self.eps)


# ---------------------------------------------------------------------------
# Predictive-coding model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictiveCodingModel:
    r"""A univariate predictive-coding generative model (book §5.1–5.2).

    Pairs a generating function ``g`` with Gaussian likelihood/prior variances and a
    prior mean, matching the book's :math:`\mathcal M` (Eq. 14):

    * Likelihood ``p(y|x) = N(y; g(x), σ_y²)``
    * Prior ``p(x) = N(x; m_x, s_x²)``

    Parameters
    ----------
    g : GenerativeFunction
        Observation generating function with a derivative.
    sigma2_y : float
        Likelihood (sensory) variance ``σ_y²`` (> 0). Its inverse is the sensory
        precision ``λ_y``.
    m_x : float
        Prior mean ``m_x``.
    s2_x : float
        Prior (state) variance ``s_x²`` (> 0). Its inverse is the state precision
        ``λ_x``.
    """

    g: GenerativeFunction
    sigma2_y: float = 0.25
    m_x: float = 4.0
    s2_x: float = 0.25

    def __post_init__(self) -> None:
        if not (self.sigma2_y > 0) or not np.isfinite(self.sigma2_y):
            raise ValueError(f"sigma2_y must be finite positive, got {self.sigma2_y!r}")
        if not (self.s2_x > 0) or not np.isfinite(self.s2_x):
            raise ValueError(f"s2_x must be finite positive, got {self.s2_x!r}")

    @property
    def lambda_y(self) -> float:
        """Sensory precision ``λ_y = 1/σ_y²``."""
        return 1.0 / self.sigma2_y

    @property
    def lambda_x(self) -> float:
        """State precision ``λ_x = 1/s_x²``."""
        return 1.0 / self.s2_x

    def predict(self, mu: float) -> float:
        """Expected observation ``μ_y = g(μ_x)`` (the model's prediction)."""
        return float(np.asarray(self.g(float(mu))))


# ---------------------------------------------------------------------------
# Prediction errors + free energy
# ---------------------------------------------------------------------------


def sensory_prediction_error(model: PredictiveCodingModel, y: float, mu: float) -> float:
    r"""Sensory prediction error ``ε_y = y − g(μ_x)`` (book Eq. 6a)."""
    return float(y) - float(np.asarray(model.g(float(mu))))


def state_prediction_error(model: PredictiveCodingModel, mu: float) -> float:
    r"""State prediction error ``ε_x = μ_x − m_x`` (book Eq. 6b)."""
    return float(mu) - model.m_x


@dataclass(frozen=True)
class PCFreeEnergy:
    r"""Predictive-coding free energy and its prediction-error decomposition.

    Attributes
    ----------
    free_energy : float
        :math:`\mathcal F_{\mathrm{MAP}}` (book Eq. 7a), the Laplace/MAP variational
        free energy (an upper bound on surprisal, tight at the posterior mode).
    eps_y, eps_x : float
        Sensory and state prediction errors (Eq. 6).
    weighted_eps_y, weighted_eps_x : float
        Precision-weighted squared prediction errors ``λ ε²`` (the terms VFE is
        built from; book Eq. 7b).
    mu_y : float
        The model's expected observation ``g(μ_x)``.
    """

    free_energy: float
    eps_y: float
    eps_x: float
    weighted_eps_y: float
    weighted_eps_x: float
    mu_y: float

    def summary(self, ndigits: int = 4) -> str:
        """Return key diagnostic quantities as a compact dictionary."""
        return (
            f"PCFreeEnergy(F={round(self.free_energy, ndigits)}, "
            f"eps_y={round(self.eps_y, ndigits)}, eps_x={round(self.eps_x, ndigits)})"
        )


def predictive_coding_free_energy(
    model: PredictiveCodingModel, y: float, mu: float
) -> PCFreeEnergy:
    r"""MAP/Laplace variational free energy at belief mean ``μ_x`` (book Eq. 7a).

    .. math::
        \mathcal F = \tfrac12\big(\log\sigma_y^2 + \lambda_y\varepsilon_y^2
                                 + \log s_x^2  + \lambda_x\varepsilon_x^2\big)

    (constants from the Gaussian normalizers are dropped, as in the book). Returns
    the full :class:`PCFreeEnergy` decomposition.
    """
    eps_y = sensory_prediction_error(model, y, mu)
    eps_x = state_prediction_error(model, mu)
    w_eps_y = model.lambda_y * eps_y**2
    w_eps_x = model.lambda_x * eps_x**2
    fe = 0.5 * (np.log(model.sigma2_y) + w_eps_y + np.log(model.s2_x) + w_eps_x)
    return PCFreeEnergy(
        free_energy=float(fe),
        eps_y=eps_y,
        eps_x=eps_x,
        weighted_eps_y=float(w_eps_y),
        weighted_eps_x=float(w_eps_x),
        mu_y=float(np.asarray(model.g(float(mu)))),
    )


def pc_free_energy_grad(model: PredictiveCodingModel, y: float, mu: float) -> float:
    r"""Analytic gradient ``∂F/∂μ_x`` of the MAP free energy (book Eq. 16).

    Derived by the chain rule (NOT transcribed — the book's sign convention varies):
    with :math:`\varepsilon_y = y-g(\mu_x)`, :math:`\varepsilon_x = \mu_x-m_x`,

    .. math::
        \frac{\partial\mathcal F}{\partial\mu_x}
            = \lambda_x\varepsilon_x - \lambda_y\,\varepsilon_y\,g'(\mu_x).

    The companion tests assert this matches a central finite difference of
    :func:`predictive_coding_free_energy` to < 1e-5 — so the recognition dynamics
    are provably descent directions regardless of the book's notation.
    """
    eps_y = sensory_prediction_error(model, y, mu)
    eps_x = state_prediction_error(model, mu)
    gprime = float(np.asarray(model.g.derivative(float(mu))))
    return model.lambda_x * eps_x - model.lambda_y * eps_y * gprime


def pc_free_energy_grad_fd(
    model: PredictiveCodingModel, y: float, mu: float, eps: float = 1e-5
) -> float:
    """Central finite-difference gradient of :func:`predictive_coding_free_energy`.

    The numerical oracle the analytic :func:`pc_free_energy_grad` is verified against.
    """
    f_hi = predictive_coding_free_energy(model, y, mu + eps).free_energy
    f_lo = predictive_coding_free_energy(model, y, mu - eps).free_energy
    return (f_hi - f_lo) / (2.0 * eps)


# ---------------------------------------------------------------------------
# Analytical landmarks — closed forms used as oracles and figure annotations
# ---------------------------------------------------------------------------


def pc_curvature_linear(model: PredictiveCodingModel) -> float:
    r"""Local curvature ``L = λ_x + g'² λ_y`` of the MAP free energy.

    For a :class:`LinearFunction` ``g`` (constant slope) this is the *exact*
    second derivative ``∂²F/∂μ²`` everywhere, so ``F`` is a parabola of width
    ``1/L``. Fixed-step recognition dynamics ``μ ← μ − κ ∂F/∂μ`` converge iff
    ``κ < 2/L`` and the residual contracts by ``|1 − κL|`` per step — the book's
    κ-sensitivity warning made precise. For a nonlinear ``g`` it is the curvature
    evaluated at the prior mean (a local approximation).
    """
    gprime = float(np.asarray(model.g.derivative(model.m_x)))
    return model.lambda_x + gprime**2 * model.lambda_y


def pc_linear_fixed_point(model: PredictiveCodingModel, y: float) -> float:
    r"""Closed-form recognition fixed point ``μ*`` for a **linear** ``g``.

    Setting ``∂F/∂μ = λ_x(μ−m_x) − λ_y(y−aμ−b)a = 0`` for ``g(x)=ax+b`` gives

    .. math::
        \mu^* = \frac{\lambda_x m_x + \lambda_y\,a\,(y-b)}{\lambda_x + \lambda_y a^2}.

    This is exactly the Gaussian posterior mean (the MAP estimate coincides with
    the posterior mean for a linear-Gaussian model), so it is both the analytical
    landmark the figures annotate and an independent oracle for
    :func:`~active_inference.estimators.predictive_coding.predictive_coding_inference`.

    Raises
    ------
    TypeError
        If ``model.g`` is not a :class:`LinearFunction` (no closed form otherwise;
        use a grid argmin of the free energy instead).
    """
    if not isinstance(model.g, LinearFunction):
        raise TypeError(
            "pc_linear_fixed_point requires a LinearFunction g; for nonlinear g "
            "use the grid argmin of predictive_coding_free_energy."
        )
    a, b = model.g.slope, model.g.intercept
    num = model.lambda_x * model.m_x + model.lambda_y * a * (float(y) - b)
    return num / (model.lambda_x + model.lambda_y * a**2)
