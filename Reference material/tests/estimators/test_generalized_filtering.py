"""Tests for ``estimators.generalized_filtering`` — online Algorithm 6.1.1.

Oracle discipline: the filter is verified to TRACK an independently-simulated true
state (mean tracking error far below the unfiltered baseline), and its relaxation
at a held observation is verified against the closed-form fixed point.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.predictive_coding import LinearFunction
from active_inference.core.generalized_filtering import (
    DynamicStateSpaceModel,
    gf_fixed_point_linear,
)
from active_inference.estimators.generalized_filtering import (
    DynamicProcess,
    GeneralizedFilterResult,
    generalized_filter,
    simulate_dynamic_process,
)


def example_process() -> DynamicProcess:
    # E: f_E=θ_x*−x* (attractor 10), g_E=x*−θ_y* (θ_y*=3).
    return DynamicProcess(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                          omega_x=0.1, omega_y=0.1)


def example_model() -> DynamicStateSpaceModel:
    return DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                                  s2_x=5.0, sigma2_y=0.02)


class TestSimulateProcess:
    def test_shapes_and_attractor(self) -> None:
        tr = simulate_dynamic_process(example_process(), x0=5.0, n_steps=1000, dt=0.01,
                                      rng=np.random.default_rng(0))
        assert tr.xs.shape == (1001,) and tr.ys.shape == (1001,)
        assert tr.xs[0] == pytest.approx(5.0)
        # The point attractor at 10 draws x* toward it.
        assert tr.xs[-50:].mean() == pytest.approx(10.0, abs=0.5)


class TestGeneralizedFilter:
    def test_returns_result(self) -> None:
        tr = simulate_dynamic_process(example_process(), x0=5.0, n_steps=200, dt=0.01,
                                      rng=np.random.default_rng(1))
        res = generalized_filter(example_model(), tr.ys, dt=0.01, kappa=0.1, mu0=15.0)
        assert isinstance(res, GeneralizedFilterResult)
        assert res.mus.shape == tr.ys.shape

    def test_tracks_true_state(self) -> None:
        # ISC: belief μ_x tracks the true external state x* far better than the prior.
        rng = np.random.default_rng(0)
        tr = simulate_dynamic_process(example_process(), x0=5.0, n_steps=1000, dt=0.01, rng=rng)
        res = generalized_filter(example_model(), tr.ys, dt=0.01, kappa=0.1, mu0=15.0)
        tracked = res.tracking_error(tr.xs, burn_in=300)
        unfiltered = float(np.mean(np.abs(15.0 - tr.xs[300:])))  # belief held at the prior
        assert tracked < 0.5             # close tracking
        assert tracked < 0.2 * unfiltered  # at least 5x better than not filtering

    def test_relaxation_matches_closed_form_fixed_point(self) -> None:
        # Holding y fixed, the agent's iterated Euler step converges to the closed form.
        model = example_model()
        y = 6.888
        ys = np.full(4000, y)
        res = generalized_filter(model, ys, dt=0.01, kappa=0.1, mu0=15.0)
        assert res.final_mu == pytest.approx(gf_fixed_point_linear(model, y), abs=1e-3)

    def test_free_energy_decreases(self) -> None:
        ys = np.full(2000, 6.888)
        res = generalized_filter(example_model(), ys, dt=0.01, kappa=0.1, mu0=15.0)
        # Relaxation at a fixed observation is a descent — F at the end < F at the start.
        assert res.free_energies[-1] < res.free_energies[0]


class TestMultivariateFilter:
    def _hooke(self):
        from active_inference.core.generalized_filtering import (
            MultivariateDynamicModel, LinearVectorFunction)
        from active_inference.estimators.generalized_filtering import MultivariateDynamicProcess
        k, m, v0, th = 4.0, 3.0, 5.0, 3.0
        A_f = np.array([[0.0, 1.0], [-k / m, 0.0]])
        b_f = np.array([0.0, (k / m) * v0])
        proc = MultivariateDynamicProcess(
            f=LinearVectorFunction(A_f, b_f), g=LinearVectorFunction(np.eye(2), np.array([-th, -th])),
            omega_x=0.1, omega_y=0.1)
        model = MultivariateDynamicModel(
            f=LinearVectorFunction(A_f, b_f), g=LinearVectorFunction(np.eye(2), np.array([-th, -th])),
            precision_x=0.5, precision_y=10.0, dim_x=2, dim_y=2)
        return proc, model

    def test_tracks_oscillator(self) -> None:
        # ISC: the belief follows the Hooke's-law orbit far better than the static prior.
        # (A residual error remains — the perception LAG the book describes, motivating
        #  generalized coordinates in §6.3.)
        from active_inference.estimators.generalized_filtering import (
            simulate_multivariate_process, multivariate_generalized_filter)
        proc, model = self._hooke()
        tr = simulate_multivariate_process(proc, x0=np.array([0.0, 5.0]), n_steps=1000,
                                           dt=0.01, rng=np.random.default_rng(0))
        res = multivariate_generalized_filter(model, tr.ys, dt=0.01, kappa=1.0,
                                              mu0=np.array([8.0, 8.0]))
        te = res.tracking_error(tr.xs, burn_in=300)
        unfiltered = float(np.mean(np.linalg.norm(np.array([8.0, 8.0]) - tr.xs[300:], axis=1)))
        assert te < 2.0                      # close tracking of an amplitude-~7 oscillator
        assert te < 0.2 * unfiltered         # >5x better than the static prior
        # the belief actually oscillates (not stuck at the centre).
        assert res.mus[300:, 0].std() > 1.0

    def test_reduces_to_scalar_on_1d(self) -> None:
        # A 1-D multivariate filter reproduces the scalar §6.1 belief trace.
        from active_inference.core.generalized_filtering import (
            DynamicStateSpaceModel, MultivariateDynamicModel, LinearVectorFunction)
        from active_inference.estimators.generalized_filtering import (
            generalized_filter, multivariate_generalized_filter)
        ys = np.full(500, 6.888)
        sc = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                                    s2_x=5.0, sigma2_y=0.02)
        mv = MultivariateDynamicModel(
            f=LinearVectorFunction(np.array([[-1.0]]), np.array([10.0])),
            g=LinearVectorFunction(np.array([[1.0]]), np.array([-3.0])),
            precision_x=np.array([0.2]), precision_y=np.array([50.0]), dim_x=1, dim_y=1)
        r_sc = generalized_filter(sc, ys, dt=0.01, kappa=0.1, mu0=15.0)
        r_mv = multivariate_generalized_filter(mv, ys.reshape(-1, 1), dt=0.01, kappa=0.1,
                                               mu0=np.array([15.0]))
        np.testing.assert_allclose(r_mv.mus[:, 0], r_sc.mus, atol=1e-9)


class TestGeneralizedCoordinatesFilter:
    def _model(self, **kw):
        from active_inference.core.generalized_filtering import GeneralizedModel
        p = dict(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                 precision_x=np.array([0.2, 0.2, 0.2]), precision_y=np.array([50.0, 50.0, 50.0]),
                 embedding_dim=3)
        p.update(kw)
        return GeneralizedModel(**p)

    def test_recovers_position_at_rest(self) -> None:
        # Holding the generalized obs at the at-rest steady state recovers [10,0,0]:
        # position 10, zero velocity/acceleration. (D-operator equilibrium μ̃̇=Dμ̃.)
        from active_inference.estimators.generalized_filtering import generalized_filter_gc
        ys = np.tile([7.0, 0.0, 0.0], (3000, 1))
        r = generalized_filter_gc(self._model(), ys, dt=0.01, kappa=1.0,
                                  mu0_tilde=np.array([5.0, 0.0, 0.0]))
        np.testing.assert_allclose(r.mus[-1], [10.0, 0.0, 0.0], atol=1e-2)

    def test_recovers_velocity_free_motion(self) -> None:
        # The headline payoff: in generalized coordinates the belief recovers the
        # higher-order MOTION. With a free-motion model (no attractor pulling
        # velocity to zero) the order-1 belief recovers the true velocity.
        from active_inference.core.generalized_filtering import GeneralizedModel
        from active_inference.estimators.generalized_filtering import generalized_filter_gc
        m = GeneralizedModel(f=LinearFunction(0.0, 0.0), g=LinearFunction(1.0, 0.0),
                             precision_x=np.array([1.0, 1.0, 1.0]),
                             precision_y=np.array([100.0, 100.0, 100.0]), embedding_dim=3)
        dt, v, T = 0.01, 2.0, 4000
        xstar = v * np.arange(T) * dt
        ys = np.stack([xstar, np.full(T, v), np.zeros(T)], axis=1)
        r = generalized_filter_gc(m, ys, dt=dt, kappa=1.0, mu0_tilde=np.zeros(3))
        assert np.mean(r.velocities[2000:]) == pytest.approx(2.0, abs=0.05)
        assert np.mean(np.abs(r.positions[1000:] - xstar[1000:])) < 0.01

    def test_reduces_to_nongeneralized_when_D_zero(self) -> None:
        # Footnote 22: without generalized coordinates Dμ̃=0 and the update is just
        # −κ∂F. With embedding_dim=1 the D operator is the 1×1 zero matrix.
        from active_inference.core.generalized_filtering import shift_operator
        np.testing.assert_allclose(shift_operator(1), np.zeros((1, 1)))


class TestVectorGeneralizedCoordinatesFilter:
    def _hooke(self, *, embedding_dim: int = 3):
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            MultivariateDynamicModel,
            correlated_embedding_precision,
        )

        k, m, v0, th = 4.0, 3.0, 5.0, 3.0
        A_f = np.array([[0.0, 1.0], [-k / m, 0.0]])
        b_f = np.array([0.0, (k / m) * v0])
        g = LinearVectorFunction(np.eye(2), np.array([-th, -th]))
        base = MultivariateDynamicModel(
            f=LinearVectorFunction(A_f, b_f),
            g=g,
            precision_x=0.5,
            precision_y=10.0,
            dim_x=2,
            dim_y=2,
        )
        model = GeneralizedVectorModel(
            f=LinearVectorFunction(A_f, b_f),
            g=g,
            precision_x=correlated_embedding_precision(np.eye(2) * 0.5, embedding_dim, gamma=2.0),
            precision_y=correlated_embedding_precision(np.eye(2) * 10.0, embedding_dim, gamma=2.0),
            embedding_dim=embedding_dim,
            dim_x=2,
            dim_y=2,
        )
        return base, model

    def test_generalized_measurements_are_time_embedding_tensor(self) -> None:
        from active_inference.estimators.generalized_filtering import generalized_measurements_from_series

        dt = 0.1
        time = np.arange(10) * dt
        ys = np.column_stack([2.0 * time, -time])
        y_tilde = generalized_measurements_from_series(ys, embedding_dim=3, dt=dt)
        assert y_tilde.shape == (10, 3, 2)
        np.testing.assert_allclose(y_tilde[-1, 0], ys[-1])
        np.testing.assert_allclose(y_tilde[-1, 1], [2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(y_tilde[-1, 2], [0.0, 0.0], atol=1e-12)

    def test_vector_generalized_filter_improves_over_ordinary_filter(self) -> None:
        from active_inference.estimators.generalized_filtering import (
            MultivariateDynamicProcess,
            generalized_measurements_from_series,
            generalized_vector_filter,
            multivariate_generalized_filter,
            simulate_multivariate_process,
        )

        base, model = self._hooke()
        proc = MultivariateDynamicProcess(f=base.f, g=base.g, omega_x=0.0, omega_y=0.0)
        trace = simulate_multivariate_process(
            proc,
            x0=np.array([0.0, 5.0]),
            n_steps=900,
            dt=0.01,
            rng=np.random.default_rng(2),
        )
        y_tilde = generalized_measurements_from_series(trace.ys, embedding_dim=3, dt=0.01)
        ordinary = multivariate_generalized_filter(
            base,
            trace.ys,
            dt=0.01,
            kappa=1.0,
            mu0=np.array([8.0, 8.0]),
        )
        generalized = generalized_vector_filter(
            model,
            y_tilde,
            dt=0.01,
            kappa=1.0,
            mu0_tilde=np.zeros((3, 2)),
        )
        assert generalized.tracking_error(trace.xs, burn_in=250) < ordinary.tracking_error(trace.xs, burn_in=250)
        assert generalized.free_energies[-100:].mean() < generalized.free_energies[:100].mean()
