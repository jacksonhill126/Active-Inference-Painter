"""Tests for ``core.active_inference`` — Chapter 7 action machinery.

Oracle discipline: the action gradient is checked against the closed-form
`−κ_a·(∂y/∂a)·λ_y·ε_y`, and the perception gradient against the §6.1 gradient.
"""

from __future__ import annotations

import pytest

from active_inference.core.predictive_coding import LinearFunction
from active_inference.core.generalized_filtering import (
    DynamicStateSpaceModel,
    gf_free_energy_grad,
    gf_sensory_prediction_error,
)
from active_inference.core.active_inference import (
    ActiveInferenceAgent,
    action_gradient,
    perception_gradient,
)


def agent(**kw) -> ActiveInferenceAgent:
    # Preference v=0 encoded as f_M(μ)=v−μ = LinearFunction(-1, 0); g_M(μ)=μ−θ_y.
    model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 0.0), g=LinearFunction(1.0, -3.0),
                                   s2_x=1.0, sigma2_y=0.05)
    p = dict(perception_model=model, forward_model=1.0, kappa_x=0.2, kappa_a=0.4)
    p.update(kw)
    return ActiveInferenceAgent(**p)


class TestAgent:
    def test_preference_is_attractor_intercept(self) -> None:
        assert agent().preference == pytest.approx(0.0)
        ag = agent(perception_model=DynamicStateSpaceModel(
            f=LinearFunction(-1.0, 5.0), g=LinearFunction(1.0, -3.0), s2_x=1.0, sigma2_y=0.05))
        assert ag.preference == pytest.approx(5.0)


class TestGradients:
    def test_perception_gradient_matches_gf(self) -> None:
        ag = agent()
        for mu in (2.0, 5.0, 8.0):
            assert perception_gradient(ag, 7.0, mu) == pytest.approx(
                -ag.kappa_x * gf_free_energy_grad(ag.perception_model, 7.0, mu))

    def test_action_gradient_closed_form(self) -> None:
        ag = agent()
        for mu in (2.0, 5.0, 8.0):
            eps_y = gf_sensory_prediction_error(ag.perception_model, 7.0, mu)
            expected = -ag.kappa_a * ag.forward_model * ag.perception_model.lambda_y * eps_y
            assert action_gradient(ag, 7.0, mu) == pytest.approx(expected)

    def test_action_vanishes_at_zero_sensory_error(self) -> None:
        # When the observation matches the prediction, no action is needed.
        ag = agent()
        # g_M(μ)=μ−3, so y=μ−3 gives ε_y=0.
        assert action_gradient(ag, 5.0 - 3.0, 5.0) == pytest.approx(0.0)


class TestMultivariateGeneralizedActiveInference:
    def _agent(self):
        import numpy as np

        from active_inference.core.active_inference import MultivariateActiveInferenceAgent
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            correlated_embedding_precision,
        )

        M = 2
        model = GeneralizedVectorModel(
            f=LinearVectorFunction(-np.eye(2), np.zeros(2)),
            g=LinearVectorFunction(np.eye(2), np.zeros(2)),
            precision_x=correlated_embedding_precision(np.eye(2), M, gamma=2.0),
            precision_y=correlated_embedding_precision(np.eye(2) * 20.0, M, gamma=2.0),
            embedding_dim=M,
            dim_x=2,
            dim_y=2,
        )
        forward = np.zeros((M * 2, 2))
        forward[[0, 2], :] = np.eye(2)
        return MultivariateActiveInferenceAgent(model, forward_model=forward, kappa_x=0.4, kappa_a=0.5)

    def test_vector_action_vanishes_at_zero_generalized_error(self) -> None:
        import numpy as np

        from active_inference.core.active_inference import multivariate_action_gradient
        from active_inference.core.generalized_filtering import flatten_generalized_coordinates

        ag = self._agent()
        mu = flatten_generalized_coordinates(np.zeros((2, 2)))
        y = flatten_generalized_coordinates(np.zeros((2, 2)))
        np.testing.assert_allclose(multivariate_action_gradient(ag, y, mu), np.zeros(2))

    def test_vector_action_gradient_matches_finite_difference_through_forward_model(self) -> None:
        import numpy as np

        from active_inference.core.active_inference import multivariate_action_gradient
        from active_inference.core.generalized_filtering import (
            flatten_generalized_coordinates,
            generalized_vector_free_energy,
        )

        ag = self._agent()
        mu = flatten_generalized_coordinates(np.array([[0.2, -0.1], [0.0, 0.0]]))
        y = flatten_generalized_coordinates(np.array([[1.0, -0.5], [0.1, 0.0]]))
        analytic_flow = multivariate_action_gradient(ag, y, mu)

        def energy(a: np.ndarray) -> float:
            return generalized_vector_free_energy(ag.perception_model, y + ag.forward_model @ a, mu)

        eps = 1e-6
        fd_grad = np.empty(2)
        for i in range(2):
            da = np.zeros(2)
            da[i] = eps
            fd_grad[i] = (energy(da) - energy(-da)) / (2.0 * eps)
        np.testing.assert_allclose(analytic_flow, -ag.kappa_a * fd_grad, atol=1e-5)
