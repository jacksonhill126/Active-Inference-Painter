r"""Unified, streamlined visualization layer for Chapters 4–10.

The per-chapter plot modules (``plotting.py``, ``variational.py``, ``diagnostics.py``)
each re-specify figure sizes, colours, grids and legends.  This module provides a
small shared vocabulary so inference *results* — whatever produced them — render
through one consistent visual language:

* :func:`panel_grid` / :func:`finalize` — the styling primitives (palette, grid,
  legend, spines) every panel routes through, so nothing hard-codes ``figsize`` or
  hex colours.
* :func:`plot_recognition_dynamics` — **one** descent panel that accepts *either* a
  Chapter 4 :class:`~active_inference.estimators.variational.FixedFormResult` *or* a
  Chapter 5 :class:`~active_inference.estimators.predictive_coding.PredictiveCodingResult`
  (duck-typed on ``.mus`` / ``.free_energies``), with optional truth / oracle lines.
* :func:`plot_prediction_errors` — Figure 5.1.2 (VFE with true/min/prior markers and
  the two squared precision-weighted prediction errors).
* :func:`plot_hierarchical_pc` — Figure 5.4.4 (per-layer ``μ``, ``ε`` and free energy).

All colours come from :data:`~active_inference.visualizations.style.COLORS` or a
colormap helper; output goes through :func:`~active_inference.visualizations.plotting.save_or_show`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .plotting import save_or_show  # noqa: F401  (re-exported for orchestrators)
from .style import COLORS, annotate_point, annotate_stat_box

__all__ = [
    "panel_grid",
    "finalize",
    "layer_colors",
    "plot_recognition_dynamics",
    "plot_prediction_errors",
    "plot_multivariate_pc",
    "plot_hierarchical_pc",
    "plot_generalized_filter",
    "plot_correlated_embedding_precision",
    "plot_generalized_vector_filter",
    "plot_multivariate_active_inference",
    "plot_learning_attention",
    "plot_hierarchical_message_passing",
    "plot_discrete_belief_sequence",
    "plot_policy_efe_decomposition",
    "plot_parameter_learning",
    "plot_two_armed_bandit",
    "plot_factorial_likelihood",
    "plot_hierarchical_timescales",
    "save_or_show",
]


# ---------------------------------------------------------------------------
# Styling primitives
# ---------------------------------------------------------------------------


def panel_grid(
    n: int,
    *,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    rows: int = 1,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a row (or grid) of ``n`` panels with the shared default size.

    Returns ``(fig, axes)`` where ``axes`` is always a flat 1-D array of length
    ``n`` (so callers can index uniformly). ``figsize`` defaults to a width that
    scales with the number of columns.
    """
    cols = int(np.ceil(n / rows))
    if figsize is None:
        figsize = (4.4 * cols, 4.0 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for extra in axes[n:]:
        extra.set_visible(False)
    if title:
        fig.suptitle(title)
    return fig, axes[:n]


def finalize(
    ax: plt.Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    legend_loc: str = "best",
) -> plt.Axes:
    """Apply the shared aesthetic to a single axis (grid, labels, legend)."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, fontsize=9)
    return ax


def layer_colors(n: int):
    """``n`` perceptually-ordered colours for hierarchical layers (a style helper)."""
    cmap = plt.get_cmap("viridis")
    return [cmap(t) for t in np.linspace(0.15, 0.85, max(n, 1))]


# ---------------------------------------------------------------------------
# Chapter 8 — learning, attention, hierarchy, and message passing
# ---------------------------------------------------------------------------


def plot_learning_attention(
    result,
    *,
    truth_x: Optional[np.ndarray] = None,
    truth_theta: Optional[float] = None,
    title: str = "Fig. 8.1 · perception, learning, and attention",
):
    """Visualize Chapter 8 triple estimation through VFE minimization.

    The panels separate fast hidden-state perception, slow first-order parameter
    learning, second-order precision/variance learning, and the free-energy/error
    traces so the time-scale separation is visible.
    """
    mus = np.asarray(result.mus, dtype=float)
    thetas = np.asarray(result.mu_thetas, dtype=float)
    zetas = np.asarray(result.mu_zetas, dtype=float)
    variances = np.asarray(result.variances_x, dtype=float)
    fes = np.asarray(result.free_energies, dtype=float)
    eps_y = np.asarray(result.eps_y, dtype=float)
    eps_x = np.asarray(result.eps_x, dtype=float)
    if not (mus.ndim == thetas.ndim == zetas.ndim == variances.ndim == fes.ndim == 1):
        raise TypeError("learning-attention result traces must be 1-D")
    if not (mus.shape == thetas.shape == zetas.shape == variances.shape == fes.shape):
        raise ValueError("learning-attention result traces must have the same length")

    t = np.arange(mus.shape[0]) * float(getattr(result, "dt", 1.0))
    fig, axes = panel_grid(4, title=title, figsize=(16.5, 8.4), rows=2)

    ax = axes[0]
    ax.plot(t, mus, color=COLORS["posterior"], label=r"belief $\mu_x$")
    if truth_x is not None:
        tx = np.asarray(truth_x, dtype=float)
        ax.plot(t, tx, color=COLORS["truth"], lw=1.6, alpha=0.75, label=r"true $x^*$")
    annotate_stat_box(ax, f"final μx={mus[-1]:.3f}", loc="lower right")
    finalize(ax, xlabel="time", ylabel="state", title="fast hidden-state perception")

    ax = axes[1]
    ax.plot(t, thetas, color=COLORS["likelihood"], label=r"$\mu_\theta$")
    if truth_theta is not None:
        ax.axhline(truth_theta, color=COLORS["truth"], ls="--", lw=1.8,
                   label=rf"true $\theta={truth_theta:.2f}$")
    ax.plot(t, zetas, color=COLORS["neutral"], ls=":", label=r"$\mu_\zeta$")
    annotate_stat_box(ax, f"final θ={thetas[-1]:.3f}\nfinal ζ={zetas[-1]:.3f}",
                      loc="lower right")
    finalize(ax, xlabel="time", ylabel="parameter mean", title="slow learning and attention")

    ax = axes[2]
    ax.plot(t, variances, color=COLORS["state"], label=r"learned $s_x^2=e^{-\zeta}$")
    annotate_stat_box(ax, f"final var={variances[-1]:.4f}", loc="upper right")
    finalize(ax, xlabel="time", ylabel="variance", title="log precision keeps variance positive")

    ax = axes[3]
    ax.plot(t, fes, color=COLORS["data"], label=r"$\mathcal{F}$")
    ax.plot(t, np.abs(eps_y), color=COLORS["sensory"], alpha=0.8, label=r"$|\varepsilon_y|$")
    ax.plot(t, np.abs(eps_x), color=COLORS["state"], alpha=0.8, label=r"$|\varepsilon_x|$")
    finalize(ax, xlabel="time", ylabel="nats / error", title="single objective drives all updates")
    return fig


def plot_hierarchical_message_passing(
    model,
    *,
    y: float,
    belief: np.ndarray,
    title: str = "Fig. 8.5 · forward and backward message passing",
):
    """Draw a compact two-layer Chapter 8 message-passing diagram."""
    from active_inference.core.continuous_learning import hierarchical_message_terms

    terms = hierarchical_message_terms(model, y, belief)
    fig, axes = panel_grid(1, title=title, figsize=(13.2, 6.4))
    ax = axes[0]
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.4, 6.0)
    ax.axis("off")

    nodes = {
        "sensory data": dict(pos=(0.8, 2.6), symbol="y", color=COLORS["sensory"],
                             label_offset=(0.0, -0.58)),
        "sensory error": dict(pos=(2.8, 4.45), symbol=r"$\epsilon_y$",
                              color=COLORS["state"], label_offset=(0.0, 0.58)),
        "lower state": dict(pos=(5.0, 2.6), symbol=r"$\mu_x$",
                            color=COLORS["posterior"], label_offset=(0.0, -0.58)),
        "link error": dict(pos=(7.2, 4.45), symbol=r"$\epsilon_v$",
                           color=COLORS["state"], label_offset=(0.0, 0.58)),
        "upper context": dict(pos=(9.2, 2.6), symbol=r"$\mu_v$",
                              color=COLORS["prior"], label_offset=(0.0, -0.58)),
    }
    for label, spec in nodes.items():
        x, y0 = spec["pos"]
        color = spec["color"]
        ax.scatter([x], [y0], s=1150, color=color, edgecolor="white", linewidth=2.2, zorder=3)
        ax.text(x, y0, spec["symbol"], ha="center", va="center", fontsize=17,
                fontweight="bold", color="white" if label != "sensory data" else "black",
                zorder=4)
        dx, dy = spec["label_offset"]
        ax.text(x + dx, y0 + dy, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color="#111111")

    def arrow(
        a: str,
        b: str,
        text: str,
        color: str,
        *,
        rad: float = 0.0,
        label_xy: tuple[float, float],
    ) -> None:
        """Draw one labeled message-passing arrow in data coordinates."""
        xa, ya = nodes[a]["pos"]
        xb, yb = nodes[b]["pos"]
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops=dict(arrowstyle="->", lw=2.4, color=color,
                            connectionstyle=f"arc3,rad={rad}"),
        )
        ax.text(*label_xy, text, color=color, ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.5),
                zorder=5)

    arrow("sensory data", "sensory error", "forward sample y", COLORS["sensory"],
          rad=0.16, label_xy=(1.55, 3.95))
    arrow("lower state", "sensory error", r"prediction $g(\mu_x)$", COLORS["posterior"],
          rad=-0.14, label_xy=(3.9, 4.95))
    arrow("sensory error", "lower state", rf"bottom-up $\xi_y={terms.bottom_up_error:.2f}$",
          COLORS["state"], rad=-0.24, label_xy=(3.9, 2.25))
    arrow("upper context", "link error", "top-down prior", COLORS["prior"],
          rad=0.16, label_xy=(8.45, 3.95))
    arrow("lower state", "link error", "state prediction", COLORS["posterior"],
          rad=-0.14, label_xy=(6.1, 4.95))
    arrow("link error", "upper context", rf"backward $\xi_v={terms.top_down_error:.2f}$",
          COLORS["state"], rad=0.24, label_xy=(8.1, 2.25))

    ax.text(5.0, 5.62,
            "Prediction errors compare forward messages with top-down predictions.",
            ha="center", va="center", fontsize=11, color="#333333")

    stat = (
        rf"$\epsilon_y={terms.sensory_error:.2f}$"
        "\n"
        rf"$\epsilon_v={terms.link_error:.2f}$"
        "\n"
        rf"$\nabla F=({terms.gradient[0]:.2f}, {terms.gradient[1]:.2f})$"
    )
    annotate_stat_box(ax, stat, loc="lower left")
    return fig


# ---------------------------------------------------------------------------
# Unified recognition-dynamics panel (Chapter 4 fixed-form OR Chapter 5 PC)
# ---------------------------------------------------------------------------


def _trace(result, name: str):
    """Duck-typed accessor: return ``result.<name>`` as an array, or ``None``."""
    val = getattr(result, name, None)
    return None if val is None else np.asarray(val, dtype=float)


def _empirical_rate(fes: np.ndarray) -> Optional[float]:
    """Median per-step contraction of the residual ``|F_k − F_∞|`` (the linear
    convergence rate). Returns ``None`` for traces too short to estimate."""
    fes = np.asarray(fes, dtype=float)
    if fes.shape[0] < 4:
        return None
    resid = np.abs(fes - fes[-1])[:-1]
    resid = resid[resid > 1e-15]
    if resid.shape[0] < 3:
        return None
    ratios = resid[1:] / resid[:-1]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0) & (ratios < 1.5)]
    return float(np.median(ratios)) if ratios.size else None


def plot_recognition_dynamics(
    result,
    *,
    truth: Optional[float] = None,
    oracle: Optional[float] = None,
    surprisal: Optional[float] = None,
    title: Optional[str] = None,
    label: str = r"$\mu_x$",
):
    r"""Unified descent figure for a recognition / inference run.

    Works on any result exposing a 1-D ``mus`` belief-mean trace and a
    ``free_energies`` trace — i.e. Chapter 4's ``FixedFormResult`` *and* Chapter 5's
    ``PredictiveCodingResult``.  Renders up to three panels:

    1. belief-mean ``μ_x`` vs iteration, with optional ``truth`` (blue) and
       ``oracle`` (green, e.g. the grid posterior mean) reference lines;
    2. prediction errors vs iteration (``ε_x``, ``ε_y``) — only if the result
       carries them (Chapter 5);
    3. free energy vs iteration, with an optional ``surprisal`` floor.

    Returns the :class:`matplotlib.figure.Figure`.
    """
    mus = _trace(result, "mus")
    fes = _trace(result, "free_energies")
    if mus is None or fes is None:
        raise TypeError("result must expose 1-D `mus` and `free_energies` traces")
    eps_x = _trace(result, "eps_x")
    eps_y = _trace(result, "eps_y")
    has_errors = eps_x is not None and eps_y is not None

    n_panels = 3 if has_errors else 2
    fig, axes = panel_grid(n_panels, title=title, figsize=(5.2 * n_panels, 4.8))
    it = np.arange(mus.shape[0])
    mu0, mu_star = float(mus[0]), float(mus[-1])
    converged = bool(getattr(result, "converged", True))
    n_iter = int(getattr(result, "n_iter_run", mus.shape[0] - 1))

    # Panel 1 — belief mean trajectory, with the fixed point annotated.
    ax = axes[0]
    ax.plot(it, mus, color=COLORS["posterior"], lw=2.6, label=label, zorder=3)
    if truth is not None:
        ax.axhline(truth, color=COLORS["prior"], ls="--", lw=1.8,
                   label=rf"truth $x^*={truth:.3g}$")
    if oracle is not None:
        ax.axhline(oracle, color=COLORS["likelihood"], ls=":", lw=2.0,
                   label=rf"oracle $={oracle:.4g}$")
    annotate_point(ax, it[-1], mu_star, rf"$\mu^*={mu_star:.4f}$",
                   color=COLORS["posterior"],
                   dx=-0.30 * len(it), dy=0.10 * (mu0 - mu_star + 1e-9))
    stats = [f"μ₀     = {mu0:.4f}", f"μ*     = {mu_star:.4f}"]
    if oracle is not None:
        stats.append(f"oracle = {oracle:.4f}")
        stats.append(f"|μ*−o| = {abs(mu_star - oracle):.2e}")
    if truth is not None:
        stats.append(f"x*     = {truth:.4f}")
    stats.append(f"iters  = {n_iter} ({'conv' if converged else 'max'})")
    annotate_stat_box(ax, "\n".join(stats), loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=label, title="belief mean (recognition)")

    # Panel 2 — prediction errors (Chapter 5 only).
    idx = 1
    if has_errors:
        ax = axes[idx]
        ax.plot(it, eps_x, color=COLORS["state"], lw=2.6, label=r"$\varepsilon_x$ (state)")
        ax.plot(it, eps_y, color=COLORS["sensory"], lw=2.6, label=r"$\varepsilon_y$ (sensory)")
        ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
        annotate_stat_box(
            ax,
            f"εx→ {float(eps_x[-1]):+.3e}\nεy→ {float(eps_y[-1]):+.3e}",
            loc="upper right",
        )
        finalize(ax, xlabel="iteration", ylabel="prediction error",
                 title="precision-weighted prediction errors")
        idx += 1

    # Panel 3 — free energy descent, with the floor and decrease annotated.
    ax = axes[idx]
    ax.plot(it, fes, color=COLORS["data"], lw=2.6, label=r"$\mathcal{F}$", zorder=3)
    f0, f_star = float(fes[0]), float(fes[-1])
    annotate_point(ax, it[-1], f_star, rf"$\mathcal{{F}}^*={f_star:.3f}$",
                   color=COLORS["data"], dx=-0.30 * len(it),
                   dy=0.18 * (f0 - f_star + 1e-9))
    rate = _empirical_rate(fes)
    fstats = [f"F₀ = {f0:.3f}", f"F* = {f_star:.3f}", f"ΔF = {f0 - f_star:.3f}"]
    if rate is not None:
        fstats.append(f"rate≈ {rate:.3f}")
    if surprisal is not None:
        ax.axhline(surprisal, color=COLORS["likelihood"], ls="--", lw=1.8,
                   label=rf"$-\log p(y)={surprisal:.3f}$")
        fstats.append(f"−logp(y)= {surprisal:.3f}")
    annotate_stat_box(ax, "\n".join(fstats), loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=r"$\mathcal{F}$", title="free energy descent")
    return fig


# ---------------------------------------------------------------------------
# Figure 5.1.2 — VFE and prediction errors over the belief mean
# ---------------------------------------------------------------------------


def plot_prediction_errors(
    model,
    y: float,
    mu_grid: np.ndarray,
    *,
    truth: Optional[float] = None,
    title: str = "Fig. 5.1.2 · free energy and prediction errors",
):
    r"""Reproduce Figure 5.1.2 for a :class:`PredictiveCodingModel`.

    **Left:** the MAP free energy ``F(μ_x)`` over a grid of belief means, with the
    minimum (red), the true state ``x^*`` (blue) and the prior mean ``m_x`` (green)
    marked.  **Right:** the two squared precision-weighted prediction errors
    ``λ_x ε_x²`` and ``λ_y ε_y²``, which vanish at ``m_x`` and at the data-consistent
    state respectively.
    """
    from ..core.predictive_coding import (
        LinearFunction,
        pc_curvature_linear,
        pc_linear_fixed_point,
        predictive_coding_free_energy,
    )

    mu_grid = np.asarray(mu_grid, dtype=float)
    comps = [predictive_coding_free_energy(model, y, float(m)) for m in mu_grid]
    F = np.array([c.free_energy for c in comps])
    w_eps_x = np.array([c.weighted_eps_x for c in comps])
    w_eps_y = np.array([c.weighted_eps_y for c in comps])
    mu_min = float(mu_grid[int(np.argmin(F))])

    # Analytical landmark: for a linear g the minimizer has a closed form.
    mu_analytic = None
    if isinstance(model.g, LinearFunction):
        mu_analytic = pc_linear_fixed_point(model, y)

    fig, axes = panel_grid(2, title=title, figsize=(13, 5.0))

    ax = axes[0]
    ax.plot(mu_grid, F, color=COLORS["prior"], lw=2.6, label=r"$\mathcal{F}(\mu_x)$")
    annotate_point(ax, mu_min, F.min(), rf"min $\mathcal{{F}}$ @ {mu_min:.3f}",
                   color=COLORS["likelihood"],
                   dx=0.12 * (mu_grid[-1] - mu_grid[0]),
                   dy=0.12 * (F.max() - F.min()))
    if truth is not None:
        ax.axvline(truth, color=COLORS["prior"], ls="--", lw=1.6,
                   label=rf"$x^*={truth:.3g}$")
    ax.axvline(model.m_x, color=COLORS["posterior"], ls=":", lw=1.8,
               label=rf"prior $m_x={model.m_x:.3g}$")
    fstats = [f"argmin μ = {mu_min:.4f}", f"min F    = {F.min():.4f}"]
    if mu_analytic is not None:
        fstats.append(f"analytic = {mu_analytic:.4f}")
        fstats.append(f"L (curv) = {pc_curvature_linear(model):.3f}")
    annotate_stat_box(ax, "\n".join(fstats), loc="upper center")
    finalize(ax, xlabel=r"$\mu_x$", ylabel=r"$\mathcal{F}$",
             title="MAP free energy")

    ax = axes[1]
    ax.plot(mu_grid, w_eps_x, color=COLORS["state"], lw=2.8,
            label=r"$\lambda_x\varepsilon_x^2$ (state)")
    ax.plot(mu_grid, w_eps_y, color=COLORS["sensory"], lw=2.8,
            label=r"$\lambda_y\varepsilon_y^2$ (sensory)")
    ax.axvline(mu_min, color=COLORS["likelihood"], ls="-.", lw=1.6,
               label=rf"balance @ {mu_min:.3f}")
    # The free energy is the sum of the two weighted errors (up to log consts):
    ax.plot(mu_grid, 0.5 * (w_eps_x + w_eps_y), color=COLORS["neutral"], lw=1.6,
            ls=":", label=r"$\frac{1}{2}(\lambda_x\varepsilon_x^2+\lambda_y\varepsilon_y^2)$")
    annotate_stat_box(
        ax,
        f"min F at μ = {mu_min:.4f}\nstate err 0 at m_x={model.m_x:.3g}",
        loc="lower right",
    )
    finalize(ax, xlabel=r"$\mu_x$", ylabel="squared weighted error",
             title="prediction errors trade off", legend_loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# §5.3 — multivariate predictive coding (vector state, Jacobian, precision matrices)
# ---------------------------------------------------------------------------


def plot_multivariate_pc(
    result,
    *,
    truth: Optional[Sequence[float]] = None,
    oracle: Optional[Sequence[float]] = None,
    title: str = "§5.3 · multivariate predictive coding",
):
    r"""Unified descent figure for :func:`multivariate_predictive_coding` (book §5.3).

    The vector analogue of :func:`plot_recognition_dynamics`: it brings the §5.3
    figure into the same annotated vocabulary as the rest of Chapter 5, with three
    panels —

    1. each belief component ``μ_c`` vs iteration, with optional ``truth`` (dashed)
       and ``oracle`` (dotted, e.g. the closed-form
       :func:`~active_inference.estimators.predictive_coding.pc_multivariate_linear_fixed_point`)
       reference lines and a μ* / ‖μ*−oracle‖ stat box;
    2. the **prediction-error magnitudes** ``‖ε_x‖`` (state) and ``‖ε_y‖`` (sensory)
       evolving — the sensory error ``‖ε_y‖`` falls as the belief explains the data,
       while the state error ``‖ε_x‖`` reflects the prior↔likelihood trade-off (it
       can *grow* under a near-flat prior as the belief leaves ``m_x`` to fit the
       data); drawn only if the result carries ``eps_x`` / ``eps_y`` traces;
    3. the free energy ``𝓕`` falling, with ``F₀ → F*``, ``ΔF`` and the empirical
       convergence rate.

    Returns the :class:`matplotlib.figure.Figure`.
    """
    mus = np.asarray(result.mus, dtype=float)            # (T, C)
    fes = np.asarray(result.free_energies, dtype=float)  # (T,)
    if mus.ndim != 2:
        raise TypeError("multivariate result must expose a 2-D `mus` (T, C) trace")
    eps_x = _trace(result, "eps_x")
    eps_y = _trace(result, "eps_y")
    has_errors = (eps_x is not None and eps_x.size and
                  eps_y is not None and eps_y.size)

    T, C = mus.shape
    it = np.arange(T)
    mu_star = mus[-1]
    truth_arr = None if truth is None else np.asarray(truth, dtype=float)
    oracle_arr = None if oracle is None else np.asarray(oracle, dtype=float)
    converged = bool(getattr(result, "converged", True))
    n_iter = int(getattr(result, "n_iter_run", T - 1))
    comp_colors = layer_colors(C)

    n_panels = 3 if has_errors else 2
    fig, axes = panel_grid(n_panels, title=title, figsize=(5.2 * n_panels, 4.8))

    # Panel 1 — per-component belief trajectories with truth / oracle landmarks.
    ax = axes[0]
    for c in range(C):
        ax.plot(it, mus[:, c], color=comp_colors[c], lw=2.6, label=rf"$\mu_{{{c + 1}}}$")
        if truth_arr is not None:
            ax.axhline(truth_arr[c], color=comp_colors[c], ls="--", lw=1.3)
        if oracle_arr is not None:
            ax.axhline(oracle_arr[c], color=comp_colors[c], ls=":", lw=1.6)
    # Document the reference-line styles once in the legend (no per-component clutter).
    if truth_arr is not None:
        ax.plot([], [], color=COLORS["neutral"], ls="--", lw=1.3, label="truth $x^*$")
    if oracle_arr is not None:
        ax.plot([], [], color=COLORS["neutral"], ls=":", lw=1.6, label="oracle")
    mu_str = "[" + ", ".join(f"{v:.3g}" for v in mu_star) + "]"
    stats = [f"μ* = {mu_str}"]
    if oracle_arr is not None:
        stats.append(f"‖μ*−o‖ = {float(np.linalg.norm(mu_star - oracle_arr)):.2e}")
    stats.append(f"iters = {n_iter} ({'conv' if converged else 'max'})")
    annotate_stat_box(ax, "\n".join(stats), loc="center right")
    finalize(ax, xlabel="iteration", ylabel=r"$\mu$", title="state estimate")

    # Panel 2 — prediction-error magnitudes decaying (the §5.3 concept).
    idx = 1
    if has_errors:
        ax = axes[idx]
        ex = np.linalg.norm(eps_x, axis=1)
        ey = np.linalg.norm(eps_y, axis=1)
        ax.plot(it, ex, color=COLORS["state"], lw=2.6, label=r"$\|\varepsilon_x\|$ (state)")
        ax.plot(it, ey, color=COLORS["sensory"], lw=2.6, label=r"$\|\varepsilon_y\|$ (sensory)")
        ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
        annotate_stat_box(
            ax,
            f"‖εx‖→ {float(ex[-1]):.3e}\n‖εy‖→ {float(ey[-1]):.3e}",
            loc="upper right",
        )
        finalize(ax, xlabel="iteration", ylabel="prediction-error magnitude",
                 title="prediction errors")
        idx += 1

    # Panel 3 — free-energy descent.
    ax = axes[idx]
    ax.plot(it, fes, color=COLORS["data"], lw=2.6, label=r"$\mathcal{F}$", zorder=3)
    f0, f_star = float(fes[0]), float(fes[-1])
    annotate_point(ax, it[-1], f_star, rf"$\mathcal{{F}}^*={f_star:.3f}$",
                   color=COLORS["data"], dx=-0.30 * len(it),
                   dy=0.18 * (f0 - f_star + 1e-9))
    fstats = [f"F₀ = {f0:.3f}", f"F* = {f_star:.3f}", f"ΔF = {f0 - f_star:.3f}"]
    rate = _empirical_rate(fes)
    if rate is not None:
        fstats.append(f"rate≈ {rate:.3f}")
    annotate_stat_box(ax, "\n".join(fstats), loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=r"$\mathcal{F}$", title="free energy descent")
    return fig


# ---------------------------------------------------------------------------
# Figure 5.4.4 — hierarchical predictive coding
# ---------------------------------------------------------------------------


def plot_hierarchical_pc(
    result,
    *,
    truth: Optional[Sequence[float]] = None,
    title: str = "Fig. 5.4.4 · hierarchical predictive coding",
):
    r"""Reproduce Figure 5.4.4 from a :class:`HierarchicalPCResult`.

    Three panels: (left) each node ``μ^{[l]}`` over iterations (node 0 is the fixed
    sensory data), (middle) each layer-wise prediction error ``ε^{[l]}`` decaying to
    zero, (right) per-layer free energy and the summed total ``Σ F^{[l]}``.
    """
    mus = np.asarray(result.mus, dtype=float)        # (T, L+1)
    errs = np.asarray(result.errors, dtype=float)    # (T, L+1)
    lfes = np.asarray(result.layer_free_energies, dtype=float)  # (T, L+1)
    fes = np.asarray(result.free_energies, dtype=float)
    T, n_nodes = mus.shape
    it = np.arange(T)
    colors = layer_colors(n_nodes)
    mu_star = mus[-1]
    n_iter = int(getattr(result, "n_iter_run", T - 1))

    fig, axes = panel_grid(3, title=title, figsize=(15.5, 4.8))

    ax = axes[0]
    for lvl in range(n_nodes):
        lab = rf"$y=\mu^{{[0]}}\!\to{mu_star[lvl]:.3g}$" if lvl == 0 \
            else rf"$\mu^{{[{lvl}]}}\!\to{mu_star[lvl]:.3g}$"
        ax.plot(it, mus[:, lvl], color=colors[lvl], lw=2.4, label=lab)
        annotate_point(ax, it[-1], mu_star[lvl], f"{mu_star[lvl]:.2f}",
                       color=colors[lvl], dx=-0.16 * T, dy=0.0, arrow=False, ms=7)
    if truth is not None:
        for lvl, t in enumerate(truth):
            ax.axhline(t, color=colors[lvl], ls="--", lw=1.2)
    mu_str = "[" + ", ".join(f"{v:.3g}" for v in mu_star) + "]"
    annotate_stat_box(ax, f"μ* = {mu_str}\niters = {n_iter}", loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=r"$\mu^{[l]}$",
             title="layer beliefs (top-down)")

    ax = axes[1]
    for lvl in range(n_nodes):
        ax.plot(it, errs[:, lvl], color=colors[lvl], lw=2.4,
                label=rf"$\varepsilon^{{[{lvl}]}}$")
    ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
    annotate_stat_box(ax, f"max|ε| → {float(np.max(np.abs(errs[-1]))):.2e}",
                      loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=r"$\varepsilon^{[l]}$",
             title="layer prediction errors → 0")

    ax = axes[2]
    for lvl in range(n_nodes):
        ax.plot(it, lfes[:, lvl], color=colors[lvl], lw=1.8,
                label=rf"$\mathcal{{F}}^{{[{lvl}]}}$")
    ax.plot(it, fes, color=COLORS["likelihood"], ls="--", lw=2.6,
            label=r"$\sum_l \mathcal{F}^{[l]}$")
    annotate_stat_box(ax, f"ΣF₀ = {float(fes[0]):.3f}\nΣF* = {float(fes[-1]):.3e}",
                      loc="upper right")
    finalize(ax, xlabel="iteration", ylabel=r"$\mathcal{F}$",
             title="per-layer & total free energy")
    return fig


# ---------------------------------------------------------------------------
# Figure 6.1.3 — generalized filtering for perception (Chapter 6)
# ---------------------------------------------------------------------------


def plot_generalized_filter(
    result,
    *,
    truth: Optional[Sequence[float]] = None,
    dt: float = 1.0,
    title: str = "Fig. 6.1.3 · generalized filtering for perception",
):
    r"""Reproduce Figure 6.1.3 from a ``GeneralizedFilterResult``.

    Three panels, sharing the package's bold style: (left) the belief ``μ_x`` (and
    predicted observation ``μ_y``) tracking the true external state ``x^*`` and the
    observation ``y`` over time; (middle) the state and sensory prediction errors;
    (right) the variational free energy falling as the filter locks on.
    """
    mus = np.asarray(result.mus, dtype=float)
    t = np.arange(mus.shape[0]) * float(dt)
    fig, axes = panel_grid(3, title=title, figsize=(15.5, 4.8))

    ax = axes[0]
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        ax.plot(t, truth, color=COLORS["truth"], lw=2.0, label=r"true $x^*$")
        terr = float(np.mean(np.abs(mus[len(mus) // 3:] - truth[len(truth) // 3:])))
        annotate_stat_box(ax, f"track err = {terr:.3f}\nμ₀ = {mus[0]:.2f}",
                          loc="lower right")
    ax.plot(t, np.asarray(result.ys, dtype=float), color=COLORS["sensory"], lw=1.4,
            alpha=0.8, label=r"obs $y$")
    ax.plot(t, mus, color=COLORS["posterior"], lw=2.4, label=r"belief $\mu_x$")
    finalize(ax, xlabel="time", ylabel="state", title="tracking the hidden state")

    ax = axes[1]
    ax.plot(t, np.asarray(result.eps_x, dtype=float), color=COLORS["state"], lw=2.0,
            label=r"$\varepsilon_x$ (state)")
    ax.plot(t, np.asarray(result.eps_y, dtype=float), color=COLORS["sensory"], lw=2.0,
            label=r"$\varepsilon_y$ (sensory)")
    ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
    finalize(ax, xlabel="time", ylabel="prediction error", title="prediction errors")

    ax = axes[2]
    fes = np.asarray(result.free_energies, dtype=float)
    ax.plot(t, fes, color=COLORS["data"], lw=2.4, label=r"$\mathcal{F}$")
    annotate_stat_box(ax, f"F₀ = {fes[0]:.2f}\nF* = {fes[-1]:.2f}", loc="upper right")
    finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$", title="free energy")
    return fig


def plot_correlated_embedding_precision(
    precisions: Sequence[np.ndarray],
    gammas: Sequence[float],
    *,
    title: str = "Fig. 6.6.2 · correlated embedding-order precision",
):
    """Visualize how ``γ`` changes the generalized precision over embedding orders."""
    n = len(precisions)
    fig, axes = panel_grid(n, title=title, figsize=(4.2 * n, 4.4))
    vmax = max(float(np.max(np.abs(p))) for p in precisions)
    for ax, precision, gamma in zip(axes, precisions, gammas):
        im = ax.imshow(precision, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(rf"$\gamma={float(gamma):.2f}$")
        ax.set_xlabel("embedding order")
        ax.set_ylabel("embedding order")
        ax.set_xticks(np.arange(precision.shape[0]))
        ax.set_yticks(np.arange(precision.shape[0]))
        ax.grid(False)
    fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02, label="precision")
    return fig


def plot_generalized_vector_filter(
    result,
    truth: np.ndarray,
    *,
    ordinary=None,
    dt: float = 1.0,
    title: str = "Fig. 6.6.4 · multivariate generalized filtering in generalized coordinates",
):
    """Plot Example 6.7 vector generalized filtering with optional ordinary-filter baseline."""
    truth = np.asarray(truth, dtype=float)
    positions = np.asarray(result.positions, dtype=float)
    t = np.arange(positions.shape[0]) * float(dt)
    fig, axes = panel_grid(4, title=title, figsize=(16.8, 8.4), rows=2)
    colors = (COLORS["posterior"], COLORS["state"])
    burn = positions.shape[0] // 3
    err = float(np.mean(np.linalg.norm(positions[burn:] - truth[burn:], axis=1)))

    ax = axes[0]
    for dim, color in enumerate(colors):
        ax.plot(t, truth[:, dim], color=COLORS["truth"], lw=1.2, alpha=0.45)
        ax.plot(t, positions[:, dim], color=color, lw=2.2, label=rf"$\mu_x^{{[{dim}]}}$")
        if ordinary is not None:
            ax.plot(t, np.asarray(ordinary.mus)[:, dim], color=color, ls="--", lw=1.3, alpha=0.65)
    annotate_stat_box(ax, f"GC err={err:.3f}\ndashed=ordinary filter", loc="lower right")
    finalize(ax, xlabel="time", ylabel="state", title="state tracking")

    ax = axes[1]
    ys0 = np.asarray(result.ys)[:, 0, :]
    eps0 = np.asarray(result.eps_y)[:, 0, :]
    for dim, color in enumerate(colors):
        ax.plot(
            t,
            ys0[:, dim],
            color=COLORS["neutral"],
            lw=1.0,
            alpha=0.4,
            label=rf"$y^{{[{dim}]}}$",
        )
        ax.plot(t, eps0[:, dim], color=color, lw=1.8, label=rf"$\varepsilon_y^{{[{dim}]}}$")
    ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.0)
    finalize(ax, xlabel="time", ylabel="observation / error", title="order-0 sensory error")

    ax = axes[2]
    velocities = np.asarray(result.mus)[:, 1, :]
    for dim, color in enumerate(colors):
        ax.plot(t, velocities[:, dim], color=color, lw=2.0, label=rf"$\mu_x^{{\prime,[{dim}]}}$")
    finalize(ax, xlabel="time", ylabel="velocity belief", title="recovered first-order motion")

    ax = axes[3]
    fes = np.asarray(result.free_energies, dtype=float)
    ax.plot(t, fes, color=COLORS["data"], lw=2.2, label=r"$\mathcal{F}$")
    annotate_stat_box(ax, f"F0={fes[0]:.2f}\nFtail={float(fes[-100:].mean()):.2f}",
                      loc="upper right")
    finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$", title="free energy")
    return fig


def plot_multivariate_active_inference(
    result,
    *,
    preference: np.ndarray,
    exogenous: Optional[np.ndarray] = None,
    baseline=None,
    dt: float = 1.0,
    title: str = "Fig. 7.5 · multivariate active generalized filtering",
):
    """Plot vector active inference: state, belief, action, sensory error, and VFE."""
    xs = np.asarray(result.xs, dtype=float)
    mus = np.asarray(result.mus, dtype=float)[:, 0, :]
    actions = np.asarray(result.actions, dtype=float)
    t = np.arange(xs.shape[0]) * float(dt)
    fig, axes = panel_grid(4, title=title, figsize=(16.8, 8.4), rows=2)
    colors = (COLORS["posterior"], COLORS["state"])
    preference = np.asarray(preference, dtype=float)
    action_time = result.action_start * float(dt)

    ax = axes[0]
    ax.plot(xs[:, 0], xs[:, 1], color=COLORS["truth"], lw=2.0, label=r"active $x^*$ path")
    ax.plot(mus[:, 0], mus[:, 1], color=COLORS["posterior"], lw=1.6, alpha=0.75,
            label=r"belief $\mu_x^{[0]}$")
    if baseline is not None:
        base_xs = np.asarray(baseline.xs, dtype=float)
        ax.plot(base_xs[:, 0], base_xs[:, 1], color=COLORS["neutral"], ls="--", lw=1.6,
                label="no-action baseline")
    ax.scatter(preference[0], preference[1], marker="*", s=180, color=COLORS["likelihood"],
               edgecolor="black", linewidth=0.8, label="preference")
    if exogenous is not None:
        ex = np.asarray(exogenous, dtype=float)
        ax.scatter(ex[0], ex[1], marker="x", s=110, color=COLORS["neutral"], label="exogenous attractor")
    ax.set_aspect("equal", adjustable="datalim")
    annotate_stat_box(ax, f"settled err={result.preference_error(preference):.3f}",
                      loc="upper right")
    finalize(ax, xlabel=r"$x_0$", ylabel=r"$x_1$", title="2-D action-perception path")

    ax = axes[1]
    for dim, color in enumerate(colors):
        ax.plot(t, xs[:, dim], color=color, lw=2.0, label=rf"$x^*_{{{dim}}}$")
        ax.plot(t, mus[:, dim], color=color, ls="--", lw=1.4, label=rf"$\mu_{{{dim}}}$")
        ax.axhline(preference[dim], color=color, ls=":", lw=1.1)
    ax.axvline(action_time, color="black", ls="--", lw=1.2, label="action on")
    finalize(ax, xlabel="time", ylabel="state", title="state and belief components")

    ax = axes[2]
    for dim, color in enumerate(colors):
        ax.plot(t, actions[:, dim], color=color, lw=2.0, label=rf"$a_{{{dim}}}$")
    ax.axvline(action_time, color="black", ls="--", lw=1.2)
    finalize(ax, xlabel="time", ylabel="action", title="control cancels exogenous drift")

    ax = axes[3]
    eps0 = np.asarray(result.eps_y, dtype=float)[:, 0, :]
    for dim, color in enumerate(colors):
        ax.plot(t, np.abs(eps0[:, dim]), color=color, lw=1.7, label=rf"$|\epsilon_y^{dim}|$")
    ax.plot(t, np.asarray(result.free_energies, dtype=float), color=COLORS["data"], lw=2.2,
            label=r"$\mathcal{F}$")
    ax.axvline(action_time, color="black", ls="--", lw=1.2)
    finalize(ax, xlabel="time", ylabel="error / VFE", title="prediction error and free energy")
    return fig


# ---------------------------------------------------------------------------
# Figure 9.2/9.3 — dynamic discrete filtering and variational free energy
# ---------------------------------------------------------------------------


def plot_discrete_belief_sequence(
    beliefs: np.ndarray,
    *,
    observations: Optional[Sequence[str | int]] = None,
    state_labels: Optional[Sequence[str]] = None,
    free_energies: Optional[np.ndarray] = None,
    title: str = "Fig. 9.2 · dynamic POMDP filtering",
):
    r"""Visualize a dynamic POMDP forward-filtering sequence (Chapter 9 §9.2–9.3).

    The first panel is a state-by-time heatmap of filtered beliefs. The second panel
    plots each hidden-state marginal as a trajectory. If ``free_energies`` is supplied,
    a third panel shows the discrete variational free energy at each filtered posterior,
    making the §9.3 connection explicit: filtering moves from a prior prediction to the
    posterior that minimizes the per-step categorical VFE.
    """
    beliefs = np.asarray(beliefs, dtype=float)
    if beliefs.ndim != 2 or beliefs.shape[0] < 1:
        raise ValueError("beliefs must be a non-empty (T, C) array")
    if not np.all(np.isfinite(beliefs)):
        raise ValueError("beliefs must contain only finite values")
    if not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each belief row must sum to 1")
    T, C = beliefs.shape
    labels = list(state_labels) if state_labels is not None else [f"s{k}" for k in range(C)]
    if len(labels) != C:
        raise ValueError(f"state_labels must have length {C}")
    obs_labels = None if observations is None else [str(o) for o in observations]
    if obs_labels is not None and len(obs_labels) != T:
        raise ValueError(f"observations must have length {T}")
    fes = None if free_energies is None else np.asarray(free_energies, dtype=float)
    if fes is not None and fes.shape != (T,):
        raise ValueError(f"free_energies must have shape ({T},)")

    n_panels = 3 if fes is not None else 2
    fig, axes = panel_grid(n_panels, title=title, figsize=(5.2 * n_panels, 4.8))
    t = np.arange(T)
    colors = layer_colors(C)

    ax = axes[0]
    im = ax.imshow(beliefs.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="belief")
    ax.set_yticks(np.arange(C))
    ax.set_yticklabels(labels)
    ax.set_xticks(t)
    xlabel = "time"
    if obs_labels is not None and T <= 8:
        ax.set_xticklabels([f"{i}\n{o}" for i, o in enumerate(obs_labels)])
        xlabel = "time / observation"
    annotate_stat_box(ax, f"final: {labels[int(np.argmax(beliefs[-1]))]}\n"
                          f"P = {beliefs[-1].max():.2f}",
                      loc="upper right")
    finalize(ax, xlabel=xlabel, ylabel="state",
             title="filtered posterior P(s_t | o_1:t)", legend=False)

    ax = axes[1]
    for k in range(C):
        ax.plot(t, beliefs[:, k], color=colors[k], lw=2.4, marker="o", ms=4,
                label=labels[k])
    ax.set_ylim(-0.04, 1.04)
    finalize(ax, xlabel="time", ylabel="probability", title="state belief trajectories")

    if fes is not None:
        ax = axes[2]
        ax.plot(t, fes, color=COLORS["data"], lw=2.6, marker="o", ms=5,
                label=r"$\mathcal{F}(s_t)$")
        annotate_stat_box(ax, f"F min = {float(fes.min()):.3f}\n"
                              f"F final = {float(fes[-1]):.3f}",
                          loc="upper right")
        finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$",
                 title="per-step discrete VFE at posterior")
    return fig


# ---------------------------------------------------------------------------
# Figure 9.5/9.6 — risk + ambiguity policy decomposition
# ---------------------------------------------------------------------------


def plot_policy_efe_decomposition(
    traces: Sequence[object],
    *,
    policy_labels: Optional[Sequence[str]] = None,
    posterior: Optional[np.ndarray] = None,
    title: str = "Fig. 9.6 · risk/ambiguity trade-off in policy selection",
):
    r"""Visualize policy EFE as the book's constituent terms.

    Each trace is expected to expose ``risk_total``, ``ambiguity_total``,
    ``novelty_total`` and ``totals_per_step`` (for example
    :class:`active_inference.core.pomdp.PolicyEFETrace`). Left panel: policy totals split
    into reward-seeking risk, information-seeking ambiguity, and optional Chapter 10
    novelty. Right panel: per-step ``G`` across each policy horizon. If a policy posterior is
    supplied, the selected policy probability is annotated.
    """
    traces = list(traces)
    if not traces:
        raise ValueError("traces must contain at least one policy trace")
    n = len(traces)
    labels = list(policy_labels) if policy_labels is not None else [f"π{k}" for k in range(n)]
    if len(labels) != n:
        raise ValueError(f"policy_labels must have length {n}")
    risks = np.array([float(t.risk_total) for t in traces])
    ambiguities = np.array([float(t.ambiguity_total) for t in traces])
    novelties = np.array([float(t.novelty_total) for t in traces])
    totals = np.array([float(t.total) for t in traces])
    q = None if posterior is None else np.asarray(posterior, dtype=float)
    if q is not None and q.shape != (n,):
        raise ValueError(f"posterior must have shape ({n},)")

    fig, axes = panel_grid(2, title=title, figsize=(12.8, 4.9))
    x = np.arange(n)

    ax = axes[0]
    ax.bar(x, risks, color=COLORS["prior"], label="risk (reward-seeking)")
    ax.bar(x, ambiguities, bottom=risks, color=COLORS["sensory"],
           label="ambiguity (information-seeking)")
    if np.any(novelties > 1e-12):
        ax.bar(x, -novelties, color=COLORS["posterior"], label="novelty gain")
    ax.plot(x, totals, "o", color=COLORS["data"], ms=9, label="total G")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    best = int(np.argmin(totals))
    stat = f"min G: {labels[best]} = {totals[best]:.3f}"
    if q is not None:
        stat += f"\nQ({labels[int(np.argmax(q))]}) = {float(np.max(q)):.2f}"
    annotate_stat_box(ax, stat, loc="upper right")
    finalize(ax, xlabel="policy", ylabel="nats", title="G = risk + ambiguity − novelty")

    ax = axes[1]
    colors = layer_colors(n)
    max_horizon = max(len(np.asarray(tr.totals_per_step, dtype=float)) for tr in traces)
    for i, tr in enumerate(traces):
        step = np.asarray(tr.totals_per_step, dtype=float)
        ax.plot(np.arange(1, len(step) + 1), step, color=colors[i], lw=2.5,
                marker="o", ms=5, label=labels[i])
    ax.set_xticks(np.arange(1, max_horizon + 1))
    if max_horizon == 1:
        ax.set_xlim(0.5, 1.5)
    else:
        ax.set_xlim(0.8, max_horizon + 0.2)
    finalize(ax, xlabel="prediction step", ylabel=r"$G_t$",
             title="per-step EFE across the planning horizon")
    return fig


# ---------------------------------------------------------------------------
# Figures 10.1.2/10.1.3/10.1.4 — Dirichlet parameter learning (Chapter 10)
# ---------------------------------------------------------------------------


def _entry_colors(n: int):
    """``n`` distinct colours for matrix entries being learned (style helper)."""
    cmap = plt.get_cmap("viridis")
    return [cmap(t) for t in np.linspace(0.1, 0.9, max(n, 1))]


def plot_parameter_learning(
    history: np.ndarray,
    confidence: np.ndarray,
    *,
    truth: Optional[np.ndarray] = None,
    symbol: str = "A",
    title: str = "Fig. 10.1.3 · learning a POMDP array",
):
    r"""Reproduce the book's parameter-learning figures (Figs 10.1.3/10.1.4).

    A duck-typed plotter for any learned array (``A`` or ``B``): pass the per-trial
    **probability** history and the **concentration** (pseudocount) history. Left panel:
    each matrix entry's learned probability across trials, converging on its true value
    (dots); right panel: the matching concentration parameters growing linearly — the
    "confidence" that makes the estimate progressively harder to move (book §10.1).

    Args:
        history: ``(n_trials+1, ...)`` per-trial expected probability arrays.
        confidence: ``(n_trials+1, ...)`` per-trial raw concentration parameters.
        truth: optional true array (same trailing shape) — drawn as target dots.
        symbol: array name for labels (``"A"`` or ``"B"``).
        title: figure suptitle.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    history = np.asarray(history, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    n_trials = history.shape[0] - 1
    trials = np.arange(history.shape[0])
    flat = history.reshape(history.shape[0], -1)          # (T+1, n_entries)
    conf_flat = confidence.reshape(confidence.shape[0], -1)
    n_entries = flat.shape[1]
    colors = _entry_colors(n_entries)
    truth_flat = None if truth is None else np.asarray(truth, dtype=float).ravel()

    fig, axes = panel_grid(2, title=title, figsize=(12.5, 4.8))

    ax = axes[0]
    idx = list(np.ndindex(history.shape[1:]))
    for k in range(n_entries):
        lbl = rf"${symbol}_{{{idx[k][0]}{idx[k][1]}}}$" if len(idx[k]) == 2 else f"{symbol}[{k}]"
        ax.plot(trials, flat[:, k], color=colors[k], lw=2.4, label=lbl)
        if truth_flat is not None:
            ax.plot(n_trials, truth_flat[k], "o", color=colors[k], ms=9,
                    markeredgecolor="black", zorder=5)
    if truth_flat is not None:
        final_err = float(np.max(np.abs(flat[-1] - truth_flat)))
        annotate_stat_box(ax, f"max |{symbol}−{symbol}*| = {final_err:.3f}\n"
                              f"dots = true values", loc="center right")
    finalize(ax, xlabel="trial", ylabel="probability",
             title=f"learned {symbol} entries converge on the truth", legend=True)

    ax = axes[1]
    for k in range(n_entries):
        ax.plot(trials, conf_flat[:, k], color=colors[k], lw=2.4)
    annotate_stat_box(ax, f"Σ pseudocounts: {conf_flat[0].sum():.0f} → {conf_flat[-1].sum():.0f}\n"
                          "confidence grows with counts", loc="upper left")
    finalize(ax, xlabel="trial", ylabel="concentration (pseudocount)",
             title="confidence (Dirichlet pseudocounts) grows", legend=False)
    return fig


# ---------------------------------------------------------------------------
# Figures 10.3.4/10.3.6 — factorial depth: the two-armed bandit (Chapter 10 §10.3)
# ---------------------------------------------------------------------------

_TAB_ACTIONS = ["start", "hint", "left", "right"]


def plot_two_armed_bandit(result, *, title="Fig. 10.3.6 · two-armed bandit (factorial AIF)"):
    r"""Reproduce the two-armed bandit results figure (book Figs 10.3.6/10.3.7).

    Three panels from a ``TwoArmedBanditResult``: (left) the agent's **context belief**
    (which machine is better) converging on the truth; (middle) the **policy posterior** over
    the four choice actions across time — the explore/exploit trace; (right) the **reward
    outcomes** (win/lose/start) the agent accrued. A stat box reports wins and hints taken.
    """
    ctx = np.asarray(result.context_belief, dtype=float)
    pol = np.asarray(result.policy_posterior, dtype=float)
    rew = np.asarray(result.reward_obs, dtype=int)
    t_ctx = np.arange(ctx.shape[0])
    t_pol = np.arange(pol.shape[0])
    truth = int(result.true_context)
    fig, axes = panel_grid(3, title=title, figsize=(15.5, 4.8))

    ax = axes[0]
    ax.plot(t_ctx, ctx[:, 0], color=COLORS["prior"], lw=2.6, label="L-better")
    ax.plot(t_ctx, ctx[:, 1], color=COLORS["likelihood"], lw=2.6, label="R-better")
    ax.axhline(1.0, color=COLORS["neutral"], ls="--", lw=1.0)
    annotate_stat_box(ax, f"true: {'right' if truth else 'left'}-better\n"
                          f"final P = {ctx[-1][truth]:.2f}", loc="center right")
    finalize(ax, xlabel="time", ylabel="probability", title="context belief → truth")

    ax = axes[1]
    colors = _entry_colors(pol.shape[1])
    for a in range(pol.shape[1]):
        ax.plot(t_pol, pol[:, a], color=colors[a], lw=2.4, label=_TAB_ACTIONS[a])
    finalize(ax, xlabel="time", ylabel=r"$Q(\pi)$", title="policy posterior (explore/exploit)")

    ax = axes[2]
    labels = {0: "start", 1: "lose", 2: "win"}
    rew_colors = {0: COLORS["neutral"], 1: COLORS["likelihood"], 2: COLORS["posterior"]}
    ax.bar(t_pol, np.ones_like(rew), color=[rew_colors[r] for r in rew], alpha=0.9)
    ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=rew_colors[k]) for k in (0, 1, 2)]
    ax.legend(handles, [labels[k] for k in (0, 1, 2)], loc="upper right", fontsize=9)
    annotate_stat_box(ax, f"wins: {result.n_wins}/{len(rew)}\nhints: {result.n_hints}",
                      loc="upper left")
    finalize(ax, xlabel="time", ylabel="", title="reward outcomes", legend=False)
    return fig


def plot_factorial_likelihood(model, *, title="Fig. 10.3.4 · factorial likelihood A (per modality)"):
    r"""Visualize the factorial likelihood arrays ``A`` (book Fig 10.3.4).

    One heatmap per observation modality. Because each ``A^(m)`` is conditioned on *all* state
    factors, the conditioning factors are flattened along the x-axis (joint state index) and
    the observations along the y-axis — showing how every joint state maps to an observation
    distribution. Reproduces the structure of the two-armed bandit ``A`` set.
    """
    A = model.A
    n = len(A)
    fig, axes = panel_grid(n, title=title, figsize=(5.0 * n, 4.4))
    for m in range(n):
        Am = np.asarray(A[m], dtype=float)
        flat = Am.reshape(Am.shape[0], -1)            # (O_m, joint states)
        ax = axes[m]
        im = ax.imshow(flat, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(flat.shape[1]))
        ax.set_yticks(range(flat.shape[0]))
        finalize(ax, xlabel="joint state (factors flattened)", ylabel="observation",
                 title=f"modality {m}: P(o | factors)", legend=False)
    return fig


# ---------------------------------------------------------------------------
# Figure 10.4.1 — hierarchical depth: nested time scales (Chapter 10 §10.4)
# ---------------------------------------------------------------------------


def plot_hierarchical_timescales(result, *, title="Fig. 10.4.1 · hierarchical POMDP (nested time scales)"):
    r"""Visualize a hierarchical POMDP run (book §10.4, Fig 10.4.1).

    From a ``HierarchicalResult``: (left) the **slow top-layer belief** over macro-steps
    converging on the true regime; (middle) the **top-down prior** pushed into the fast bottom
    layer each macro-step (how the high level contextualizes the low level); (right) the
    **bottom-layer belief** trajectory across the nested fast steps. Demonstrates how a slow
    high-level state steers fast low-level dynamics.
    """
    top = np.asarray(result.top_belief, dtype=float)
    priors = np.asarray(result.bottom_priors, dtype=float)
    bottom = np.asarray(result.bottom_belief, dtype=float)   # (n_macro, inner, C_bottom)
    fig, axes = panel_grid(3, title=title, figsize=(15.5, 4.8))

    ax = axes[0]
    tcolors = _entry_colors(top.shape[1])
    for k in range(top.shape[1]):
        ax.plot(np.arange(top.shape[0]), top[:, k], color=tcolors[k], lw=2.6,
                label=f"regime {k}")
    annotate_stat_box(ax, f"true regime: {result.true_top}", loc="center right")
    finalize(ax, xlabel="macro-step", ylabel="probability", title="slow top-layer belief")

    ax = axes[1]
    bcolors = _entry_colors(priors.shape[1])
    for k in range(priors.shape[1]):
        ax.plot(np.arange(priors.shape[0]), priors[:, k], color=bcolors[k], lw=2.4,
                marker="o", ms=5, label=f"state {k}")
    finalize(ax, xlabel="macro-step", ylabel="prior probability",
             title="top-down prior into bottom layer")

    ax = axes[2]
    n_macro, inner, C = bottom.shape
    flat = bottom.reshape(n_macro * inner, C)
    for k in range(C):
        ax.plot(np.arange(flat.shape[0]), flat[:, k], color=bcolors[k], lw=2.0,
                label=f"state {k}")
    for k in range(1, n_macro):
        ax.axvline(k * inner - 0.5, color=COLORS["neutral"], ls=":", lw=1.0)
    finalize(ax, xlabel="fast step (nested)", ylabel="probability",
             title="fast bottom-layer belief")
    return fig
