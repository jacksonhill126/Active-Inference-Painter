"""Tests for Chapter 8 continuous learning and attention estimators."""

from __future__ import annotations

import numpy as np

from active_inference.core.continuous_learning import LearningAttentionModel, LearningAttentionState
from active_inference.estimators.continuous_learning import simulate_learning_attention


class TestLearningAttentionSimulation:
    def test_learning_reduces_tracking_error_and_free_energy(self) -> None:
        rng = np.random.default_rng(8)
        true_x = 5.0 + 0.15 * np.sin(np.linspace(0, 5, 360))
        theta_true = 3.0
        ys = true_x - theta_true + rng.normal(0.0, 0.08, size=true_x.shape)
        model = LearningAttentionModel(
            state_attractor=5.0,
            theta_prior_mean=0.0,
            zeta_prior_mean=0.0,
            sigma2_y=0.02,
            sigma2_theta=50.0,
            sigma2_zeta=50.0,
        )

        result = simulate_learning_attention(
            model,
            ys,
            initial=LearningAttentionState(mu_x=1.5, mu_theta=0.0, mu_zeta=0.0),
            dt=0.03,
            kappa_x=0.7,
            kappa_theta=2.0,
            kappa_zeta=0.25,
            damping=1.2,
        )

        baseline_error = float(np.mean(np.abs(1.5 - true_x[-120:])))
        assert result.tracking_error(true_x, burn_in=240) < baseline_error * 0.35
        assert abs(result.final_state.mu_theta - theta_true) < 0.4

    def test_parameter_precision_and_energy_converge(self) -> None:
        rng = np.random.default_rng(9)
        true_x = np.full(420, 5.0)
        theta_true = 3.0
        ys = true_x - theta_true + rng.normal(0.0, 0.10, size=true_x.shape)
        model = LearningAttentionModel(
            state_attractor=5.0,
            theta_prior_mean=0.0,
            zeta_prior_mean=0.0,
            sigma2_y=0.02,
            sigma2_theta=50.0,
            sigma2_zeta=50.0,
        )
        result = simulate_learning_attention(
            model,
            ys,
            initial=LearningAttentionState(mu_x=2.0, mu_theta=0.0, mu_zeta=0.0),
            dt=0.03,
            kappa_x=0.7,
            kappa_theta=2.0,
            kappa_zeta=0.25,
            damping=1.2,
        )

        assert abs(result.final_state.mu_theta - theta_true) < 0.35
        assert abs(result.final_variance_x - 0.01) < 0.08
        assert np.mean(result.free_energies[-80:]) < np.mean(result.free_energies[:80])
