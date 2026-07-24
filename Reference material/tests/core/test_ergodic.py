"""Tests for ergodic-density and entropy-bound helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    EntropyBound,
    density_entropy,
    entropy_upper_bound_from_vfe,
    ergodic_density,
    ergodic_ou_trajectory,
)


def test_ergodic_density_integrates_to_one() -> None:
    """Histogram-based ergodic density is normalized on its grid."""
    trajectory = ergodic_ou_trajectory(n_steps=300)
    centers, density = ergodic_density(trajectory, bins=40)
    assert centers.shape == density.shape
    assert np.trapezoid(density, centers) == pytest.approx(1.0, rel=5e-2)


def test_density_entropy_accepts_normalized_density() -> None:
    """Differential entropy helper returns a finite scalar."""
    x = np.linspace(-3.0, 3.0, 200)
    density = np.exp(-0.5 * x**2)
    density = density / np.trapezoid(density, x)
    assert np.isfinite(density_entropy(x, density))


def test_entropy_upper_bound_returns_non_negative_gap() -> None:
    """Entropy-bound helper validates upper-bound orientation."""
    bound = entropy_upper_bound_from_vfe(1.25, 2.0)
    assert isinstance(bound, EntropyBound)
    assert bound.gap == pytest.approx(0.75)
    with pytest.raises(ValueError):
        entropy_upper_bound_from_vfe(2.0, 1.0)
