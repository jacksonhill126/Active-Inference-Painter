"""Variational inference algorithms — Chapter 4 §4.2, §4.5, §4.6.

Three ways to minimize variational free energy (VFE), in increasing
sophistication, all reproduced from the book:

* :func:`coordinate_search_vfe` — **Algorithm 4.2.1**, Example 4.1. Zero-order
  *coordinate search*: from the current ``(mu, var)`` evaluate VFE at the eight
  neighbours one step ``kappa`` away and jump to the lowest. Pedagogical; shows
  VFE *is* a usable loss without any gradient.
* :func:`fixed_form_vi` — **Algorithm 4.6.1**, Example 4.7. *Fixed-form* VI:
  assume ``q`` is Gaussian and follow the (numerical) gradient of VFE w.r.t. its
  parameters ``(mu, var)`` (Eq. 47). The book used PyTorch autodiff; this companion
  stays dependency-light and uses central finite differences — same method, no torch.
* :func:`free_form_cavi` — **Algorithm 4.5.1**, Examples 4.5/4.6. *Free-form*
  mean-field coordinate-ascent VI (CAVI) on the three-unknown model
  ``(x, beta0, beta1)`` of Eq. 32–34, using the fundamental theorem of mean-field
  VI (Eq. 43): ``q(theta_s) ∝ exp(E_{q(theta_\\s)}[log p(y, theta)])``. For this
  conjugate linear-Gaussian model each update is closed-form Gaussian.

Every routine returns a trace dataclass so orchestrators can plot the descent and
tests can assert convergence to the exact grid posterior and VFE monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.generative_model import GenerativeModel
from ..core.inference import GridBayesianInference
from ..core.variational import (
    GaussianBelief,
    VFEComponents,
    variational_free_energy,
)

_LOG_2PI = float(np.log(2.0 * np.pi))


# ===========================================================================
# §4.2 — Coordinate search (Algorithm 4.2.1, Example 4.1)
# ===========================================================================


@dataclass
class CoordinateSearchResult:
    """Trace of :func:`coordinate_search_vfe`.

    Attributes
    ----------
    mus, vars_, free_energies : ndarray
        Per-iteration belief mean, variance, and VFE (including the initialization
        at index 0).
    belief : GaussianBelief
        Final variational density.
    converged : bool
        Whether the ``|ΔF| < tol`` convergence test fired before ``n_iter``.
    n_iter_run : int
        Number of accepted update steps actually taken.
    """

    mus: np.ndarray
    vars_: np.ndarray
    free_energies: np.ndarray
    belief: GaussianBelief
    converged: bool
    n_iter_run: int

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])


def _neighbours(mu: float, var: float, kappa: float) -> list[tuple[float, float]]:
    """The eight ``(mu ± kappa, var ± kappa)`` grid neighbours (Fig. 4.2.1)."""
    out = []
    for dmu in (-kappa, 0.0, kappa):
        for dvar in (-kappa, 0.0, kappa):
            if dmu == 0.0 and dvar == 0.0:
                continue
            out.append((mu + dmu, var + dvar))
    return out


def coordinate_search_vfe(
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    mu0: Optional[float] = None,
    var0: Optional[float] = None,
    kappa: float = 0.01,
    n_iter: int = 20,
    tol: float = 0.0,
    min_var: float = 1e-4,
) -> CoordinateSearchResult:
    """Minimize VFE by coordinate search (Algorithm 4.2.1).

    Starting from ``(mu0, var0)`` — by default the model's prior ``(m_x, s2_x)``
    per Eq. 15 — repeatedly evaluate VFE at the eight neighbours a step ``kappa``
    away and move to the lowest. Stops after ``n_iter`` steps or when no neighbour
    improves on the current VFE by more than ``tol``.

    Parameters mirror the book: ``kappa`` is the step size :math:`\\kappa`,
    ``n_iter`` the iteration budget :math:`\\mathcal T`. ``min_var`` clamps the
    variance to stay positive.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if mu0 is None:
        mu0 = float(getattr(model, "m_x", 0.0))
    if var0 is None:
        var0 = float(getattr(model, "s2_x", 1.0))

    # Precompute the oracle once; reuse across every VFE evaluation.
    result = GridBayesianInference(model=model, x_grid=x_grid).infer(y)
    log_evidence = float(result.log_evidence)
    posterior = np.asarray(result.posterior, dtype=float)

    def vfe(mu: float, var: float) -> float:
        """Return the variational free-energy objective for current beliefs."""
        return variational_free_energy(
            GaussianBelief(mu, max(var, min_var)),
            model,
            y,
            x_grid,
            log_evidence=log_evidence,
            posterior=posterior,
        ).free_energy

    mu, var = float(mu0), max(float(var0), min_var)
    mus = [mu]
    vars_ = [var]
    fes = [vfe(mu, var)]
    converged = False
    n_run = 0

    for _ in range(int(n_iter)):
        best_mu, best_var, best_fe = mu, var, fes[-1]
        for nmu, nvar in _neighbours(mu, var, kappa):
            if nvar <= 0:
                continue
            fe = vfe(nmu, nvar)
            if fe < best_fe:
                best_mu, best_var, best_fe = nmu, nvar, fe
        delta = fes[-1] - best_fe  # improvement (>= 0)
        mu, var = best_mu, max(best_var, min_var)
        mus.append(mu)
        vars_.append(var)
        fes.append(best_fe)
        n_run += 1
        if delta <= tol:
            converged = True
            break

    return CoordinateSearchResult(
        mus=np.asarray(mus),
        vars_=np.asarray(vars_),
        free_energies=np.asarray(fes),
        belief=GaussianBelief(mu, var),
        converged=converged,
        n_iter_run=n_run,
    )


# ===========================================================================
# §4.6 — Fixed-form variational inference (Algorithm 4.6.1, Example 4.7)
# ===========================================================================


@dataclass
class FixedFormResult:
    """Trace of :func:`fixed_form_vi` (gradient descent on VFE parameters)."""

    mus: np.ndarray
    vars_: np.ndarray
    free_energies: np.ndarray
    components: list[VFEComponents]
    belief: GaussianBelief
    converged: bool
    n_iter_run: int

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])


def fixed_form_vi(
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    mu0: Optional[float] = None,
    var0: Optional[float] = None,
    lr: float = 5e-3,
    n_iter: int = 2000,
    tol: float = 1e-9,
    min_var: float = 1e-4,
    fd_eps: float = 1e-4,
) -> FixedFormResult:
    """Minimize VFE by gradient descent on ``(mu, var)`` (Algorithm 4.6.1, Eq. 47).

    The variational density is fixed to Gaussian; we follow the gradient of VFE
    w.r.t. its mean and variance::

        mu  <- mu  - lr * dF/dmu
        var <- var - lr * dF/dvar

    Gradients are estimated by central finite differences (``fd_eps``), keeping the
    package free of an autodiff dependency while remaining faithful to Eq. 47.
    ``lr`` is the learning rate :math:`\\kappa`. For numerical safety ``mu`` is
    clamped to the span of ``x_grid`` (so the belief never slides off the support
    used for the trapezoid integral) and ``var`` is floored at ``min_var``.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    grid_lo, grid_hi = float(x_grid[0]), float(x_grid[-1])
    if mu0 is None:
        mu0 = float(getattr(model, "m_x", 0.0))
    if var0 is None:
        var0 = float(getattr(model, "s2_x", 1.0))

    result = GridBayesianInference(model=model, x_grid=x_grid).infer(y)
    log_evidence = float(result.log_evidence)
    posterior = np.asarray(result.posterior, dtype=float)

    def components(mu: float, var: float) -> VFEComponents:
        """Return the variational free-energy objective for current beliefs."""
        return variational_free_energy(
            GaussianBelief(mu, max(var, min_var)),
            model,
            y,
            x_grid,
            log_evidence=log_evidence,
            posterior=posterior,
        )

    def vfe(mu: float, var: float) -> float:
        """Return the variational free-energy objective for current beliefs."""
        return components(mu, var).free_energy

    mu, var = float(mu0), max(float(var0), min_var)
    comp0 = components(mu, var)
    mus = [mu]
    vars_ = [var]
    fes = [comp0.free_energy]
    comps = [comp0]
    converged = False
    n_run = 0

    for _ in range(int(n_iter)):
        g_mu = (vfe(mu + fd_eps, var) - vfe(mu - fd_eps, var)) / (2 * fd_eps)
        v_hi, v_lo = var + fd_eps, max(var - fd_eps, min_var)
        g_var = (vfe(mu, v_hi) - vfe(mu, v_lo)) / (v_hi - v_lo)
        mu = float(np.clip(mu - lr * g_mu, grid_lo, grid_hi))
        var = max(var - lr * g_var, min_var)
        comp = components(mu, var)
        mus.append(mu)
        vars_.append(var)
        fes.append(comp.free_energy)
        comps.append(comp)
        n_run += 1
        if abs(fes[-2] - fes[-1]) < tol:
            converged = True
            break

    return FixedFormResult(
        mus=np.asarray(mus),
        vars_=np.asarray(vars_),
        free_energies=np.asarray(fes),
        components=comps,
        belief=GaussianBelief(mu, var),
        converged=converged,
        n_iter_run=n_run,
    )


# ===========================================================================
# §4.5 — Free-form mean-field CAVI (Algorithm 4.5.1, Examples 4.5/4.6)
# ===========================================================================


@dataclass
class MeanFieldConfig:
    """Generative model of Eq. 32–34: ``y = beta0 + beta1 * x + noise`` with
    Gaussian priors on the three unknowns ``(x, beta0, beta1)``.

    Defaults are the book's prior settings ``phi`` (Eq. 34).
    """

    sigma2_y: float = 0.25
    m_x: float = 4.0
    s2_x: float = 0.25
    m_b0: float = 0.0
    s2_b0: float = 0.25
    m_b1: float = 0.0
    s2_b1: float = 0.25

    def __post_init__(self) -> None:
        for name in ("sigma2_y", "s2_x", "s2_b0", "s2_b1"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive")


@dataclass
class CAVIResult:
    """Store mean-field CAVI beliefs, free energies, convergence, and sweep count."""

    q_x: GaussianBelief
    q_b0: GaussianBelief
    q_b1: GaussianBelief
    free_energies: np.ndarray            # one entry per completed sweep (+ init)
    mu_x: np.ndarray
    mu_b0: np.ndarray
    mu_b1: np.ndarray
    converged: bool
    n_sweeps_run: int

    @property
    def final_free_energy(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(self.free_energies[-1])


def _mean_field_vfe(cfg: MeanFieldConfig, y: float, qx, qb0, qb1) -> float:
    """Closed-form mean-field VFE ``F = -Σ H[q_i] - E_q[log p(y, theta)]`` for Eq. 33.

    Uses the factorized expectation of the bilinear log-joint; correct under the
    mean-field independence assumption (Eq. 37–38).
    """
    mx, vx = qx.mu, qx.var
    mb0, vb0 = qb0.mu, qb0.var
    mb1, vb1 = qb1.mu, qb1.var
    s2y = cfg.sigma2_y

    # E_q[(y - b0 - b1 x)^2] under mean-field independence.
    e_b1x = mb1 * mx
    e_b1sq = mb1 ** 2 + vb1
    e_xsq = mx ** 2 + vx
    e_resid_sq = (
        y ** 2
        - 2 * y * mb0
        - 2 * y * e_b1x
        + (mb0 ** 2 + vb0)
        + 2 * mb0 * e_b1x
        + e_b1sq * e_xsq
    )
    e_log_lik = -0.5 * (_LOG_2PI + np.log(s2y)) - e_resid_sq / (2 * s2y)

    def e_log_prior(m, v, m0, s2):
        """Return the expected log-probability contribution for this factor."""
        return -0.5 * (_LOG_2PI + np.log(s2)) - ((m - m0) ** 2 + v) / (2 * s2)

    e_log_joint = (
        e_log_lik
        + e_log_prior(mx, vx, cfg.m_x, cfg.s2_x)
        + e_log_prior(mb0, vb0, cfg.m_b0, cfg.s2_b0)
        + e_log_prior(mb1, vb1, cfg.m_b1, cfg.s2_b1)
    )
    entropy = sum(0.5 * (_LOG_2PI + 1.0 + np.log(v)) for v in (vx, vb0, vb1))
    return float(-entropy - e_log_joint)


def free_form_cavi(
    y: float,
    cfg: Optional[MeanFieldConfig] = None,
    *,
    n_sweeps: int = 50,
    tol: float = 1e-9,
) -> CAVIResult:
    """Mean-field coordinate-ascent VI on ``(x, beta0, beta1)`` (Algorithm 4.5.1).

    Each partition is updated in turn (order ``x → beta0 → beta1``) using the
    fundamental theorem of mean-field VI (Eq. 43). For this linear-Gaussian model
    the optimal ``q(theta_s)`` is Gaussian with the closed-form precision/mean::

        tau_x  = E[b1^2]/s2y + 1/s2_x      mu_x  = (E[b1](y - E[b0])/s2y + m_x/s2_x)/tau_x
        tau_b0 = 1/s2y     + 1/s2_b0        mu_b0 = ((y - E[b1]E[x])/s2y + m_b0/s2_b0)/tau_b0
        tau_b1 = E[x^2]/s2y + 1/s2_b1       mu_b1 = (E[x](y - E[b0])/s2y + m_b1/s2_b1)/tau_b1

    where ``E[b1^2]=mu_b1^2+var_b1`` and ``E[x^2]=mu_x^2+var_x``. Each partition is
    initialised at its prior (Eq. 45). VFE is recorded after every full sweep and
    is guaranteed non-increasing (CAVI monotonicity).
    """
    cfg = MeanFieldConfig() if cfg is None else cfg
    s2y = cfg.sigma2_y

    # Initialise each partition at its prior (Eq. 45).
    qx = GaussianBelief(cfg.m_x, cfg.s2_x)
    qb0 = GaussianBelief(cfg.m_b0, cfg.s2_b0)
    qb1 = GaussianBelief(cfg.m_b1, cfg.s2_b1)

    fes = [_mean_field_vfe(cfg, y, qx, qb0, qb1)]
    mu_x_hist, mu_b0_hist, mu_b1_hist = [qx.mu], [qb0.mu], [qb1.mu]
    converged = False
    n_run = 0

    for _ in range(int(n_sweeps)):
        # --- update q(x) (freeze b0, b1) ---
        e_b1 = qb1.mu
        e_b1sq = qb1.mu ** 2 + qb1.var
        tau_x = e_b1sq / s2y + 1.0 / cfg.s2_x
        mu_x = (e_b1 * (y - qb0.mu) / s2y + cfg.m_x / cfg.s2_x) / tau_x
        qx = GaussianBelief(mu_x, 1.0 / tau_x)

        # --- update q(beta0) (freeze x, b1) ---
        tau_b0 = 1.0 / s2y + 1.0 / cfg.s2_b0
        mu_b0 = ((y - qb1.mu * qx.mu) / s2y + cfg.m_b0 / cfg.s2_b0) / tau_b0
        qb0 = GaussianBelief(mu_b0, 1.0 / tau_b0)

        # --- update q(beta1) (freeze x, b0) ---
        e_xsq = qx.mu ** 2 + qx.var
        tau_b1 = e_xsq / s2y + 1.0 / cfg.s2_b1
        mu_b1 = (qx.mu * (y - qb0.mu) / s2y + cfg.m_b1 / cfg.s2_b1) / tau_b1
        qb1 = GaussianBelief(mu_b1, 1.0 / tau_b1)

        fes.append(_mean_field_vfe(cfg, y, qx, qb0, qb1))
        mu_x_hist.append(qx.mu)
        mu_b0_hist.append(qb0.mu)
        mu_b1_hist.append(qb1.mu)
        n_run += 1
        if abs(fes[-2] - fes[-1]) < tol:
            converged = True
            break

    return CAVIResult(
        q_x=qx,
        q_b0=qb0,
        q_b1=qb1,
        free_energies=np.asarray(fes),
        mu_x=np.asarray(mu_x_hist),
        mu_b0=np.asarray(mu_b0_hist),
        mu_b1=np.asarray(mu_b1_hist),
        converged=converged,
        n_sweeps_run=n_run,
    )
