"""Tests for ``core.generalized_filtering`` — the Chapter 6 §6.1 dynamic model.

Oracle discipline (anti-theater): the analytic free-energy gradient is checked
against a central finite difference at off-equilibrium points, and the closed-form
recognition fixed point is verified to zero the gradient and to match an independent
grid argmin across a parameter sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.predictive_coding import LinearFunction, QuadraticFunction
from active_inference.core.generalized_filtering import (
    DynamicStateSpaceModel,
    gf_fixed_point_linear,
    gf_free_energy,
    gf_free_energy_grad,
    gf_free_energy_grad_fd,
    gf_sensory_prediction_error,
    gf_state_prediction_error,
)


def example_model(**kw) -> DynamicStateSpaceModel:
    # Example 6.1: f_M(μ)=θ_x−μ, g_M(μ)=μ−θ_y, book precisions λ_x=0.2, λ_y=50.
    p = dict(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
             s2_x=5.0, sigma2_y=0.02)
    p.update(kw)
    return DynamicStateSpaceModel(**p)


class TestModelValidation:
    def test_rejects_nonpositive_variance(self) -> None:
        with pytest.raises(ValueError):
            DynamicStateSpaceModel(f=LinearFunction(-1, 10), g=LinearFunction(1, -3), s2_x=0.0)
        with pytest.raises(ValueError):
            DynamicStateSpaceModel(f=LinearFunction(-1, 10), g=LinearFunction(1, -3), sigma2_y=-1.0)

    def test_precisions(self) -> None:
        m = example_model()
        assert m.lambda_x == pytest.approx(0.2)
        assert m.lambda_y == pytest.approx(50.0)


class TestPredictionErrors:
    def test_state_and_sensory_errors(self) -> None:
        m = example_model()
        # f_M(4)=10-4=6 -> ε_x = 4-6 = -2
        assert gf_state_prediction_error(m, 4.0) == pytest.approx(-2.0)
        # g_M(4)=4-3=1 -> ε_y = y - 1
        assert gf_sensory_prediction_error(m, 7.0, 4.0) == pytest.approx(6.0)


class TestFreeEnergyGradient:
    def test_analytic_grad_matches_fd(self) -> None:
        # ISC: derived ∂F/∂μ matches central finite difference off-equilibrium.
        for m in (example_model(),
                  DynamicStateSpaceModel(f=QuadraticFunction(0.5, 1.0),
                                         g=LinearFunction(1.0, -3.0), s2_x=5.0, sigma2_y=0.02)):
            for mu in (1.0, 3.0, 5.0, 8.0, 11.0):
                an = gf_free_energy_grad(m, 7.0, mu)
                fd = gf_free_energy_grad_fd(m, 7.0, mu)
                assert an == pytest.approx(fd, abs=1e-5)

    def test_wrong_sign_grad_fails_fd(self) -> None:
        m = example_model()
        # A deliberately negated gradient must NOT match the finite difference.
        mu = 8.0
        assert abs(-gf_free_energy_grad(m, 7.0, mu) - gf_free_energy_grad_fd(m, 7.0, mu)) > 1.0


class TestClosedFormFixedPoint:
    def test_fixed_point_zeroes_gradient(self) -> None:
        for kw in (dict(), dict(s2_x=1.0, sigma2_y=1.0),
                   dict(f=LinearFunction(-0.5, 4.0), g=LinearFunction(2.0, 1.0))):
            m = example_model(**kw)
            mu_star = gf_fixed_point_linear(m, 6.5)
            assert gf_free_energy_grad(m, 6.5, mu_star) == pytest.approx(0.0, abs=1e-9)

    def test_fixed_point_matches_grid_argmin(self) -> None:
        m = example_model()
        for y in (2.0, 6.888, 10.0):
            fp = gf_fixed_point_linear(m, y)
            grid = np.linspace(fp - 1.0, fp + 1.0, 40001)
            gmin = grid[int(np.argmin([gf_free_energy(m, y, float(g)) for g in grid]))]
            assert fp == pytest.approx(float(gmin), abs=1e-3)

    def test_high_sensory_precision_recovers_state(self) -> None:
        # With λ_y ≫ λ_x the fixed point ≈ the state that explains y (μ = y + θ_y),
        # which equals x* when y = x* − θ_y. This is WHY the filter tracks.
        m = example_model()  # λ_y=50 ≫ λ_x=0.2
        x_star = 10.0
        y = x_star - 3.0  # g_E(x*) = x* − θ_y*, θ_y*=3
        assert gf_fixed_point_linear(m, y) == pytest.approx(x_star, abs=0.1)

    def test_rejects_nonlinear(self) -> None:
        m = DynamicStateSpaceModel(f=QuadraticFunction(1.0, 1.0), g=LinearFunction(1.0, -3.0))
        with pytest.raises(TypeError):
            gf_fixed_point_linear(m, 7.0)


# ---------------------------------------------------------------------------
# §6.2 — multivariate generalized filter
# ---------------------------------------------------------------------------


class TestMultivariateGeneralizedFilter:
    def _hooke_model(self, **kw):
        from active_inference.core.generalized_filtering import (
            MultivariateDynamicModel, LinearVectorFunction)
        k, m, v0, th = 4.0, 3.0, 5.0, 3.0
        A_f = np.array([[0.0, 1.0], [-k / m, 0.0]])
        b_f = np.array([0.0, (k / m) * v0])
        p = dict(f=LinearVectorFunction(A_f, b_f), g=LinearVectorFunction(np.eye(2), np.array([-th, -th])),
                 precision_x=0.5, precision_y=10.0, dim_x=2, dim_y=2)
        p.update(kw)
        return MultivariateDynamicModel(**p)

    def test_reduces_to_scalar_filter(self) -> None:
        # ISC: a 1-D multivariate model's gradient equals the scalar §6.1 gradient.
        from active_inference.core.generalized_filtering import (
            DynamicStateSpaceModel, MultivariateDynamicModel, LinearVectorFunction,
            gf_free_energy_grad, mv_gf_free_energy_grad)
        sc = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                                    s2_x=5.0, sigma2_y=0.02)
        mv = MultivariateDynamicModel(
            f=LinearVectorFunction(np.array([[-1.0]]), np.array([10.0])),
            g=LinearVectorFunction(np.array([[1.0]]), np.array([-3.0])),
            precision_x=np.array([0.2]), precision_y=np.array([50.0]), dim_x=1, dim_y=1)
        for mu in (1.0, 4.0, 8.0):
            gs = gf_free_energy_grad(sc, 7.0, mu)
            gm = mv_gf_free_energy_grad(mv, np.array([7.0]), np.array([mu]))[0]
            assert gs == pytest.approx(gm, abs=1e-9)

    def test_jacobian_grad_matches_fd(self) -> None:
        from active_inference.core.generalized_filtering import (
            mv_gf_free_energy_grad, mv_gf_free_energy_grad_fd)
        model = self._hooke_model()
        y = np.array([4.0, 1.0])
        for mu in ([8.0, 8.0], [5.0, 0.0], [2.0, -3.0]):
            an = mv_gf_free_energy_grad(model, y, np.array(mu))
            fd = mv_gf_free_energy_grad_fd(model, y, np.array(mu))
            np.testing.assert_allclose(an, fd, atol=1e-5)

    def test_closed_form_fixed_point_zeroes_gradient(self) -> None:
        from active_inference.core.generalized_filtering import (
            mv_gf_free_energy_grad, mv_gf_fixed_point_linear)
        model = self._hooke_model()
        y = np.array([4.0, 1.0])
        fp = mv_gf_fixed_point_linear(model, y)
        np.testing.assert_allclose(mv_gf_free_energy_grad(model, y, fp), 0.0, atol=1e-9)

    def test_generic_jacobian_matches_analytic(self) -> None:
        from active_inference.core.generalized_filtering import GenericVectorFunction
        A = np.array([[0.0, 1.0], [-4.0 / 3.0, 0.0]])
        b = np.array([0.0, 20.0 / 3.0])
        gen = GenericVectorFunction(lambda x: A @ x + b, out_dim=2)
        np.testing.assert_allclose(gen.jacobian(np.array([3.0, 2.0])), A, atol=1e-6)


# ---------------------------------------------------------------------------
# §6.3–6.5 — generalized coordinates of motion
# ---------------------------------------------------------------------------


class TestGeneralizedCoordinates:
    def _model(self, **kw):
        from active_inference.core.generalized_filtering import GeneralizedModel
        p = dict(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                 precision_x=np.array([0.2, 0.2, 0.2]), precision_y=np.array([50.0, 50.0, 50.0]),
                 embedding_dim=3)
        p.update(kw)
        return GeneralizedModel(**p)

    def test_shift_operator_book_example(self) -> None:
        # Book Eq. 37 worked example: D·[3,4,2,6,4] = [4,2,6,4,0].
        from active_inference.core.generalized_filtering import shift_operator
        D = shift_operator(5)
        np.testing.assert_allclose(D @ np.array([3.0, 4, 2, 6, 4]), [4, 2, 6, 4, 0])

    def test_shift_operator_multivariate_kron(self) -> None:
        from active_inference.core.generalized_filtering import shift_operator
        D = shift_operator(3, n_states=2)
        assert D.shape == (6, 6)
        # block-diagonal of two 3x3 superdiagonal-shift blocks.
        np.testing.assert_allclose(D[:3, :3], np.diag([1.0, 1.0], k=1))
        np.testing.assert_allclose(D[3:, 3:], np.diag([1.0, 1.0], k=1))

    def test_embed_flow(self) -> None:
        # f̃ = [f(μ0), f'·μ1, f'·μ2]; for f=θ_x−μ at [3,2,1]: f(3)=7, f'=-1 → [7,-2,-1].
        from active_inference.core.generalized_filtering import embed_flow
        np.testing.assert_allclose(embed_flow(7.0, -1.0, np.array([3.0, 2.0, 1.0])), [7.0, -2.0, -1.0])

    def test_generalized_grad_matches_fd_linear_exact(self) -> None:
        # For linear f/g the local-linearity gradient is EXACT vs finite difference.
        from active_inference.core.generalized_filtering import (
            generalized_free_energy_grad, generalized_free_energy_grad_fd)
        m = self._model()
        yt = np.array([7.0, 0.0, 0.0])
        for mu in ([5.0, 1.0, 0.0], [8.0, -2.0, 1.0], [10.0, 0.0, 0.0]):
            an = generalized_free_energy_grad(m, yt, np.array(mu))
            fd = generalized_free_energy_grad_fd(m, yt, np.array(mu))
            np.testing.assert_allclose(an, fd, atol=1e-5)

    def test_generalized_state_error_uses_D_operator(self) -> None:
        # ε̃_x = D μ̃ − f̃: the D operator supplies the *actual* generalized motion.
        from active_inference.core.generalized_filtering import generalized_state_error
        m = self._model()
        # at μ̃=[4,3,2]: Dμ̃=[3,2,0]; f̃=[10-4, -3, -2]=[6,-3,-2]; ε=[-3,5,2].
        np.testing.assert_allclose(generalized_state_error(m, np.array([4.0, 3.0, 2.0])),
                                   [3 - 6, 2 - (-3), 0 - (-2)])


# ---------------------------------------------------------------------------
# §6.6 — correlated embedding orders and multivariate generalized coordinates
# ---------------------------------------------------------------------------


class TestCorrelatedEmbeddingOrders:
    def test_gaussian_temporal_covariance_matches_book_matrix(self) -> None:
        from active_inference.core.generalized_filtering import gaussian_temporal_covariance

        S = gaussian_temporal_covariance(3, gamma=2.0)
        expected = np.array(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 3.0],
            ]
        )
        np.testing.assert_allclose(S, expected, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(S) > 0.0)

    def test_generalized_precision_orderings_match_kron_formulas(self) -> None:
        from active_inference.core.generalized_filtering import (
            correlated_embedding_precision,
            gaussian_temporal_covariance,
        )

        Pi = np.diag([2.0, 3.0])
        S_inv = np.linalg.inv(gaussian_temporal_covariance(3, gamma=1.7))
        order_major = correlated_embedding_precision(Pi, 3, gamma=1.7, layout="order_major")
        state_major = correlated_embedding_precision(Pi, 3, gamma=1.7, layout="state_major")
        np.testing.assert_allclose(order_major, np.kron(S_inv, Pi))
        np.testing.assert_allclose(state_major, np.kron(Pi, S_inv))
        assert order_major.shape == state_major.shape == (6, 6)
        assert np.all(np.linalg.eigvalsh(state_major) > 0.0)

    def test_rejects_invalid_smoothness_and_layout(self) -> None:
        from active_inference.core.generalized_filtering import (
            correlated_embedding_precision,
            gaussian_temporal_covariance,
        )

        for gamma in (0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError):
                gaussian_temporal_covariance(3, gamma=gamma)
        with pytest.raises(ValueError):
            correlated_embedding_precision(1.0, 3, gamma=1.0, layout="column_major")

    def test_scalar_generalized_model_accepts_full_precision_matrix(self) -> None:
        from active_inference.core.generalized_filtering import (
            GeneralizedModel,
            correlated_embedding_precision,
            generalized_free_energy_grad,
            generalized_free_energy_grad_fd,
        )

        precision = correlated_embedding_precision(0.2, 3, gamma=1.5)
        model = GeneralizedModel(
            f=LinearFunction(-1.0, 10.0),
            g=LinearFunction(1.0, -3.0),
            precision_x=precision,
            precision_y=correlated_embedding_precision(50.0, 3, gamma=1.5),
            embedding_dim=3,
        )
        np.testing.assert_allclose(model.Pi_x, precision)
        y_tilde = np.array([7.0, 0.2, -0.1])
        mu_tilde = np.array([8.0, -1.0, 0.5])
        np.testing.assert_allclose(
            generalized_free_energy_grad(model, y_tilde, mu_tilde),
            generalized_free_energy_grad_fd(model, y_tilde, mu_tilde),
            atol=1e-5,
        )

    def test_vector_generalized_gradient_matches_finite_difference(self) -> None:
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            correlated_embedding_precision,
            flatten_generalized_coordinates,
            generalized_vector_free_energy_grad,
            generalized_vector_free_energy_grad_fd,
        )

        M = 3
        A_f = np.array([[0.0, 1.0], [-4.0 / 3.0, 0.0]])
        b_f = np.array([0.0, 20.0 / 3.0])
        model = GeneralizedVectorModel(
            f=LinearVectorFunction(A_f, b_f),
            g=LinearVectorFunction(np.eye(2), np.array([-3.0, -3.0])),
            precision_x=correlated_embedding_precision(np.eye(2) * 0.5, M, gamma=1.8),
            precision_y=correlated_embedding_precision(np.eye(2) * 10.0, M, gamma=1.8),
            embedding_dim=M,
            dim_x=2,
            dim_y=2,
        )
        mu = flatten_generalized_coordinates(np.array([[5.0, 0.0], [1.0, -1.0], [0.2, 0.1]]))
        y = flatten_generalized_coordinates(np.array([[2.0, -3.0], [0.5, -0.2], [0.0, 0.1]]))
        np.testing.assert_allclose(
            generalized_vector_free_energy_grad(model, y, mu),
            generalized_vector_free_energy_grad_fd(model, y, mu),
            atol=1e-5,
        )
