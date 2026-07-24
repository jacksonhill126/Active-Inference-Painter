"""Tests for Chapter 8 continuous learning, attention, and hierarchy primitives."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.continuous_learning import (
    ContinuousHierarchyLayer,
    HierarchicalContinuousModel,
    LearningAttentionModel,
    LearningAttentionState,
    hierarchical_continuous_free_energy,
    hierarchical_continuous_grad,
    hierarchical_continuous_grad_fd,
    hierarchical_message_terms,
    learning_attention_free_energy,
    learning_attention_grad,
    learning_attention_grad_fd,
    log_precision_to_precision,
    log_precision_to_variance,
)


def _learning_model() -> LearningAttentionModel:
    return LearningAttentionModel(
        state_attractor=5.0,
        theta_prior_mean=0.0,
        zeta_prior_mean=0.0,
        sigma2_y=0.05,
        sigma2_theta=25.0,
        sigma2_zeta=25.0,
    )


class TestLogPrecisionTransforms:
    def test_precision_and_variance_are_inverse_positive(self) -> None:
        zeta = np.array([-2.0, 0.0, 2.0])
        precision = log_precision_to_precision(zeta)
        variance = log_precision_to_variance(zeta)
        np.testing.assert_allclose(precision * variance, np.ones_like(zeta))
        assert np.all(precision > 0)
        assert np.all(variance > 0)

    def test_rejects_nonfinite_log_precision(self) -> None:
        with pytest.raises(ValueError):
            log_precision_to_precision(np.array([0.0, np.inf]))


class TestLearningAttentionFreeEnergy:
    def test_components_match_hand_computed_formula(self) -> None:
        model = _learning_model()
        state = LearningAttentionState(mu_x=4.0, mu_theta=1.0, mu_zeta=np.log(2.0))
        components = learning_attention_free_energy(model, y=2.5, state=state)

        eps_y = 2.5 - (4.0 - 1.0)
        eps_x = 4.0 - 5.0
        eps_theta = 1.0
        eps_zeta = np.log(2.0)
        expected = 0.5 * (
            model.lambda_y * eps_y**2
            + 2.0 * eps_x**2
            + model.lambda_theta * eps_theta**2
            + model.lambda_zeta * eps_zeta**2
            + np.log(model.sigma2_y)
            - np.log(2.0)
            + np.log(model.sigma2_theta)
            + np.log(model.sigma2_zeta)
        )

        assert components.eps_y == pytest.approx(eps_y)
        assert components.eps_x == pytest.approx(eps_x)
        assert components.precision_x == pytest.approx(2.0)
        assert components.free_energy == pytest.approx(expected)

    def test_gradient_matches_finite_difference(self) -> None:
        model = _learning_model()
        for state in (
            LearningAttentionState(mu_x=2.0, mu_theta=-1.0, mu_zeta=-0.4),
            LearningAttentionState(mu_x=6.0, mu_theta=2.5, mu_zeta=1.2),
        ):
            analytic = learning_attention_grad(model, y=2.0, state=state)
            fd = learning_attention_grad_fd(model, y=2.0, state=state)
            np.testing.assert_allclose(analytic, fd, atol=1e-5)

    def test_model_rejects_invalid_variances(self) -> None:
        with pytest.raises(ValueError):
            LearningAttentionModel(state_attractor=0.0, sigma2_y=0.0)
        with pytest.raises(ValueError):
            LearningAttentionModel(state_attractor=0.0, sigma2_theta=-1.0)


class TestHierarchicalContinuousModel:
    def _hierarchy(self) -> HierarchicalContinuousModel:
        return HierarchicalContinuousModel(
            lower=ContinuousHierarchyLayer(obs_offset=3.0, sensory_precision=20.0),
            link_precision=2.0,
            context_prior_mean=5.0,
            context_precision=0.5,
            link_gain=1.0,
        )

    def test_gradient_matches_finite_difference(self) -> None:
        model = self._hierarchy()
        for belief in (np.array([4.0, 2.0]), np.array([6.0, 7.0])):
            analytic = hierarchical_continuous_grad(model, y=2.0, belief=belief)
            fd = hierarchical_continuous_grad_fd(model, y=2.0, belief=belief)
            np.testing.assert_allclose(analytic, fd, atol=1e-5)

    def test_reduces_to_single_layer_when_hierarchy_disabled(self) -> None:
        model = HierarchicalContinuousModel(
            lower=ContinuousHierarchyLayer(obs_offset=3.0, sensory_precision=10.0),
            link_precision=0.0,
            context_prior_mean=0.0,
            context_precision=0.0,
        )
        belief = np.array([4.5, 99.0])
        eps_y = 2.0 - (4.5 - 3.0)
        expected = 0.5 * 10.0 * eps_y**2
        assert hierarchical_continuous_free_energy(model, y=2.0, belief=belief) == pytest.approx(expected)

    def test_message_terms_decompose_gradient(self) -> None:
        model = self._hierarchy()
        belief = np.array([4.0, 2.0])
        terms = hierarchical_message_terms(model, y=2.0, belief=belief)
        grad = hierarchical_continuous_grad(model, y=2.0, belief=belief)
        np.testing.assert_allclose(terms.gradient, grad)
        assert terms.sensory_error == pytest.approx(1.0)
        assert terms.top_down_prediction == pytest.approx(2.0)
        assert terms.bottom_up_error == pytest.approx(20.0)
