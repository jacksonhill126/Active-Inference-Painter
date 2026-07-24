"""Ergodic-density and entropy-bound helpers for FEP extras topics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_vector(name: str, value: np.ndarray | list[float]) -> np.ndarray:
    """Validate and return a finite one-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(name: str, value: int) -> int:
    """Validate and return a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive")
    return ivalue


@dataclass(frozen=True)
class EntropyBound:
    """Entropy quantity, upper bound, and non-negative residual gap."""

    entropy: float
    upper_bound: float
    gap: float

    def __post_init__(self) -> None:
        """Validate scalar fields after dataclass initialization."""
        entropy = float(self.entropy)
        upper = float(self.upper_bound)
        gap = float(self.gap)
        if not np.isfinite(entropy) or not np.isfinite(upper) or not np.isfinite(gap):
            raise ValueError("entropy bound values must be finite")
        if gap < -1e-10:
            raise ValueError("gap must be non-negative")
        object.__setattr__(self, "entropy", entropy)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "gap", max(0.0, gap))


def ergodic_density(
    trajectory: np.ndarray | list[float],
    *,
    bins: int = 80,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a normalized one-dimensional ergodic density from samples."""
    values = _finite_vector("trajectory", trajectory)
    n_bins = _positive_int("bins", bins)
    if bounds is None:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("bounds must be finite and increasing")
    density, edges = np.histogram(values, bins=n_bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mass = float(np.trapezoid(density, centers))
    if mass > 0.0:
        density = density / mass
    return centers, density


def density_entropy(x_grid: np.ndarray | list[float], density: np.ndarray | list[float]) -> float:
    """Return differential entropy ``-int p(x) log p(x) dx`` on a grid."""
    x = _finite_vector("x_grid", x_grid)
    p = _finite_vector("density", density)
    if x.shape != p.shape:
        raise ValueError("x_grid and density must have the same shape")
    if np.any(p < 0.0):
        raise ValueError("density must be non-negative")
    mass = float(np.trapezoid(p, x))
    if mass <= 0.0:
        raise ValueError("density must have positive mass")
    p = p / mass
    terms = np.zeros_like(p)
    mask = p > 0.0
    terms[mask] = -p[mask] * np.log(p[mask])
    return float(np.trapezoid(terms, x))


def entropy_upper_bound_from_vfe(entropy: float, vfe_upper_bound: float) -> EntropyBound:
    """Return the gap for an entropy quantity bounded above by a VFE-like value."""
    ent = float(entropy)
    upper = float(vfe_upper_bound)
    if not np.isfinite(ent) or not np.isfinite(upper):
        raise ValueError("entropy and vfe_upper_bound must be finite")
    if upper + 1e-10 < ent:
        raise ValueError("vfe_upper_bound must be greater than or equal to entropy")
    return EntropyBound(entropy=ent, upper_bound=upper, gap=upper - ent)


def ergodic_ou_trajectory(
    *,
    n_steps: int = 500,
    drift: float = 0.08,
    diffusion: float = 0.25,
    initial: float = 2.5,
) -> np.ndarray:
    """Return a deterministic-noise Ornstein-Uhlenbeck teaching trajectory.

    The sinusoidal forcing keeps the example reproducible while still producing
    a nontrivial trajectory for ergodic-density and entropy-bound demos.
    """
    steps = _positive_int("n_steps", n_steps)
    drift_v = float(drift)
    diffusion_v = float(diffusion)
    if not np.isfinite(drift_v) or drift_v <= 0.0:
        raise ValueError("drift must be finite and positive")
    if not np.isfinite(diffusion_v) or diffusion_v < 0.0:
        raise ValueError("diffusion must be finite and non-negative")
    xs = np.empty(steps, dtype=float)
    xs[0] = float(initial)
    phases = np.linspace(0.0, 16.0 * np.pi, steps)
    forcing = np.sin(phases) + 0.5 * np.sin(2.7 * phases + 0.2)
    for t in range(1, steps):
        xs[t] = xs[t - 1] - drift_v * xs[t - 1] + diffusion_v * forcing[t]
    return xs


__all__ = [
    "EntropyBound",
    "ergodic_density",
    "density_entropy",
    "entropy_upper_bound_from_vfe",
    "ergodic_ou_trajectory",
]
