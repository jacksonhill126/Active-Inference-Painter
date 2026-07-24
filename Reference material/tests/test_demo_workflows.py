"""Unit tests for application demo workflows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from active_inference.demo_workflows import (
    build_bicycle_demo,
    build_drone_flight_demo,
    build_eye_saccades_demo,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close matplotlib figures after each test."""
    yield
    plt.close("all")


class TestEyeSaccadesDemo:
    def test_build_returns_finite_arrays(self) -> None:
        result = build_eye_saccades_demo()
        assert "grid_path" in result.arrays
        assert result.arrays["grid_path"].size >= 2
        assert np.all(np.isfinite(result.arrays["best_first_action_efe"]))
        assert "visualize_eye_saccades" in result.figures


class TestBicycleDemo:
    def test_build_returns_active_trace(self) -> None:
        result = build_bicycle_demo(seed=3, n_steps=80)
        assert result.arrays["active_xs"].shape[0] == 81
        assert np.all(np.isfinite(result.arrays["active_free_energy"]))
        assert result.summary["active_preference_error"] <= result.summary["passive_preference_error"]


class TestDroneFlightDemo:
    def test_build_returns_plan_and_posterior(self) -> None:
        result = build_drone_flight_demo(seed=5)
        assert result.arrays["plan_states"].size >= 2
        assert result.arrays["posterior_mean"].shape == (2,)
        assert np.all(np.isfinite(result.arrays["posterior_std"]))
        assert result.summary["plan_reached"] is True
