"""Static figures for Chapter 4 — variational free energy.

Reproduces the chapter's key plots from reusable, configurable helpers:

* :func:`plot_vfe_contour` — VFE as a function of the variational parameters
  ``(mu, var)`` with an optional optimization-path overlay (Fig. 4.2.2 left,
  Fig. 4.6.1 left).
* :func:`plot_density_evolution` — the variational density ``q(x)`` over the
  iterations of inference, converging onto the posterior (Fig. 4.2.2 right,
  Fig. 4.6.1 right).
* :func:`plot_vfe_decomposition` — the G / C / E decompositions of VFE traced
  across iterations (Fig. 4.4.1, Fig. 4.6.2).
* :func:`plot_surprisal_relationship` — model evidence ``p(y)`` versus surprisal
  ``-log p(y)`` (Fig. 4.3.1).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..core.generative_model import GenerativeModel
from ..core.inference import GridBayesianInference
from ..core.variational import GaussianBelief, VFEComponents, variational_free_energy
from .style import COLORS


def vfe_surface(
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    mu_lo: float,
    mu_hi: float,
    var_lo: float,
    var_hi: float,
    n_mu: int = 60,
    n_var: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate VFE on a ``(mu, var)`` grid; returns ``(MU, VAR, F)`` meshes.

    The exact posterior + log-evidence oracle is computed once and reused for
    every cell, so an ``n_mu × n_var`` surface is cheap.
    """
    result = GridBayesianInference(model=model, x_grid=x_grid).infer(y)
    log_ev, post = float(result.log_evidence), np.asarray(result.posterior)
    mus = np.linspace(mu_lo, mu_hi, n_mu)
    vars_ = np.linspace(var_lo, var_hi, n_var)
    MU, VAR = np.meshgrid(mus, vars_)
    F = np.empty_like(MU)
    for i in range(VAR.shape[0]):
        for j in range(MU.shape[1]):
            F[i, j] = variational_free_energy(
                GaussianBelief(MU[i, j], VAR[i, j]),
                model, y, x_grid, log_evidence=log_ev, posterior=post,
            ).free_energy
    return MU, VAR, F


def plot_vfe_contour(
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    mu_lo: float = 0.0,
    mu_hi: float = 5.0,
    var_lo: float = 0.02,
    var_hi: float = 2.0,
    n_mu: int = 60,
    n_var: int = 60,
    path_mus: Optional[Sequence[float]] = None,
    path_vars: Optional[Sequence[float]] = None,
    truth: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Variational free energy over (μ, σ²)",
) -> plt.Figure:
    """Filled contour of VFE over the variational parameters (Fig. 4.2.2 left).

    Overlays the optimization path (``path_mus``/``path_vars``, red markers) and
    the true minimum ``truth = (mu*, var*)`` (green ``+``) when supplied.
    """
    MU, VAR, F = vfe_surface(
        model, y, x_grid, mu_lo=mu_lo, mu_hi=mu_hi, var_lo=var_lo, var_hi=var_hi,
        n_mu=n_mu, n_var=n_var,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure
    cf = ax.contourf(MU, VAR, F, levels=20, cmap="viridis")
    ax.contour(MU, VAR, F, levels=10, colors="black", linewidths=0.5, alpha=0.4)
    fig.colorbar(cf, ax=ax, label=r"$\mathcal{F}$")
    if path_mus is not None and path_vars is not None:
        ax.plot(path_mus, path_vars, color=COLORS["likelihood"], lw=1.5,
                marker="v", ms=5, mfc=COLORS["likelihood"], mec="white",
                label="optimization path")
    if truth is not None:
        ax.plot(truth[0], truth[1], marker="+", color=COLORS["posterior"],
                ms=16, mew=3, label="true minimum")
    ax.set_xlabel(r"$\mu_x$")
    ax.set_ylabel(r"$\sigma_x^2$")
    ax.set_title(title)
    if path_mus is not None or truth is not None:
        ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_density_evolution(
    x_grid: np.ndarray,
    beliefs: Sequence[GaussianBelief],
    *,
    posterior: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    title: str = "Variational density q(x) over iterations",
    xlabel: str = "hidden state x",
) -> plt.Figure:
    """Plot ``q(x)`` at each iteration, coloured by iteration (Fig. 4.2.2 right).

    The exact ``posterior`` (if given) is drawn as a dashed black overlay so the
    convergence ``q → p(x|y)`` is visible.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    n = len(beliefs)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(n, 1)))
    for i, belief in enumerate(beliefs):
        ax.plot(x_grid, belief.pdf(x_grid), color=colors[i], lw=1.2, alpha=0.9)
    if posterior is not None:
        ax.plot(x_grid, posterior, color="black", ls="--", lw=2.0,
                label="exact posterior p(x|y)")
        ax.legend(loc="upper right", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
    fig.colorbar(sm, ax=ax, label="iteration")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    return fig


def plot_vfe_decomposition(
    components: Sequence[VFEComponents],
    *,
    fig: Optional[plt.Figure] = None,
    title: str = "Decompositions of variational free energy",
) -> plt.Figure:
    """Three-panel trace of the G / C / E decompositions (Fig. 4.4.1, Fig. 4.6.2).

    * Left (G/D): ``F``, divergence ``KL(q‖p(x|y))``, and constant surprisal.
    * Middle (C): accuracy ``E_q[log p(y|x)]`` and complexity ``KL(q‖p(x))``.
    * Right (E): average energy ``E_q[log p(x,y)]`` and entropy ``H[q]``.
    """
    F = np.array([c.free_energy for c in components])
    divergence = np.array([c.divergence for c in components])
    surprisal = np.array([c.surprisal for c in components])
    complexity = np.array([c.complexity for c in components])
    accuracy = np.array([c.accuracy for c in components])
    energy = np.array([c.energy for c in components])
    entropy = np.array([-c.neg_entropy for c in components])
    it = np.arange(len(components))

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    else:
        axes = fig.subplots(1, 3)
    ax_g, ax_c, ax_e = axes

    ax_g.plot(it, F, color=COLORS["prior"], label=r"$\mathcal{F}$")
    ax_g.plot(it, divergence, color=COLORS["posterior"], label="divergence")
    ax_g.plot(it, surprisal, color=COLORS["likelihood"], label="surprisal")
    ax_g.set_title("G / D form")
    ax_g.set_xlabel("iteration")
    ax_g.legend(fontsize=9)

    ax_c.plot(it, accuracy, color=COLORS["prior"], label="accuracy")
    ax_c.plot(it, complexity, color=COLORS["posterior"], label="complexity")
    ax_c.set_title("C form")
    ax_c.set_xlabel("iteration")
    ax_c.legend(fontsize=9)

    ax_e.plot(it, energy, color=COLORS["prior"], label="energy")
    ax_e.plot(it, entropy, color=COLORS["posterior"], label="entropy")
    ax_e.set_title("E form")
    ax_e.set_xlabel("iteration")
    ax_e.legend(fontsize=9)

    fig.suptitle(title)
    return fig


def plot_surprisal_relationship(
    *,
    mu: float = 0.0,
    sigma2: float = 0.64,
    y_lo: float = -2.0,
    y_hi: float = 2.0,
    n: int = 400,
    title: str = "Model evidence and surprisal",
) -> plt.Figure:
    """Reproduce Fig. 4.3.1: ``p(y)``, ``-log p(y)`` vs ``y``, and vs ``p(y)``."""
    from ..core.distributions import gaussian_pdf

    y = np.linspace(y_lo, y_hi, n)
    py = gaussian_pdf(y, mu, sigma2)
    surprisal = -np.log(py)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    a0.plot(y, py, color=COLORS["prior"])
    a0.set_xlabel("y")
    a0.set_ylabel("p(y)")
    a1.plot(y, surprisal, color="#34495e")
    a1.set_xlabel("y")
    a1.set_ylabel(r"$-\log p(y)$")
    p_axis = np.linspace(0.01, 1.0, n)
    a2.plot(p_axis, -np.log(p_axis), color="#9b59b6")
    a2.set_xlabel("p(y)")
    a2.set_ylabel(r"$-\log p(y)$")
    fig.suptitle(title)
    return fig
