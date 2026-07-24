"""Tests for ``core.predictive_coding`` — prediction errors and MAP free energy.

The running model is the book's Chapter 5 linear example (also the Chapter 3–4
model): ``g(x)=2x+3``, prior ``N(4, 0.25)``, ``σ_y²=0.25``, ``y=7``. Its MAP
estimate equals the exact posterior mean ``2.4`` (the cross-chapter oracle). The
nonlinear example (Example 5.3) uses ``g(x)=x²+1``, ``m_x=3``, ``y=5.84``.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.diagnostics import (
    ConvergenceReport,
    OracleAgreement,
    convergence_report,
    gradient_check,
    oracle_agreement,
)
from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.predictive_coding import (
    GenericFunction,
    LinearFunction,
    PCFreeEnergy,
    PredictiveCodingModel,
    QuadraticFunction,
    pc_curvature_linear,
    pc_free_energy_grad,
    pc_free_energy_grad_fd,
    pc_linear_fixed_point,
    predictive_coding_free_energy,
    sensory_prediction_error,
    state_prediction_error,
)
from active_inference.core.variational import vfe_map_form

Y_OBS = 7.0


def linear_model(**kw) -> PredictiveCodingModel:
    p = dict(g=LinearFunction(2.0, 3.0), sigma2_y=0.25, m_x=4.0, s2_x=0.25)
    p.update(kw)
    return PredictiveCodingModel(**p)


# ---------------------------------------------------------------------------
# Generating functions + their derivatives (ISC-41, ISC-42)
# ---------------------------------------------------------------------------


class TestGenerativeFunctions:
    def test_linear_value_and_derivative(self) -> None:
        g = LinearFunction(2.0, 3.0)
        assert float(g(2.0)) == pytest.approx(7.0)
        assert float(np.asarray(g.derivative(2.0))) == pytest.approx(2.0)

    def test_quadratic_value_and_derivative(self) -> None:
        g = QuadraticFunction(1.0, 1.0)
        assert float(g(3.0)) == pytest.approx(10.0)        # 3²+1
        assert float(np.asarray(g.derivative(3.0))) == pytest.approx(6.0)  # 2·3

    def test_generic_with_analytic_derivative(self) -> None:
        g = GenericFunction(fn=lambda x: np.sin(x), dfn=lambda x: np.cos(x))
        assert float(np.asarray(g.derivative(0.5))) == pytest.approx(np.cos(0.5))

    @pytest.mark.parametrize(
        "g,xs",
        [
            (LinearFunction(2.0, 3.0), [-2.0, 0.0, 2.5, 5.0]),
            (QuadraticFunction(1.0, 1.0), [-3.0, 0.0, 2.2, 4.0]),
            (GenericFunction(fn=lambda x: np.exp(0.3 * x)), [-1.0, 0.0, 1.0, 2.0]),
        ],
    )
    def test_derivative_matches_finite_difference(self, g, xs) -> None:
        # ISC-42: analytic (or fallback) derivative ≈ central finite difference.
        for x in xs:
            fd = (float(g(x + 1e-5)) - float(g(x - 1e-5))) / 2e-5
            assert float(np.asarray(g.derivative(x))) == pytest.approx(fd, abs=1e-5)


class TestModelValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_bad_variances(self, bad: float) -> None:
        with pytest.raises(ValueError):
            PredictiveCodingModel(g=LinearFunction(), sigma2_y=bad)
        with pytest.raises(ValueError):
            PredictiveCodingModel(g=LinearFunction(), s2_x=bad)

    def test_precisions(self) -> None:
        m = linear_model()
        assert m.lambda_y == pytest.approx(4.0)
        assert m.lambda_x == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Prediction errors + free energy (ISC-43, ISC-44)
# ---------------------------------------------------------------------------


class TestPredictionErrors:
    def test_sensory_error(self) -> None:
        m = linear_model()
        # ε_y = y − g(μ); at μ=2 → 7 − 7 = 0.
        assert sensory_prediction_error(m, Y_OBS, 2.0) == pytest.approx(0.0)
        assert sensory_prediction_error(m, Y_OBS, 4.0) == pytest.approx(7.0 - 11.0)

    def test_state_error(self) -> None:
        m = linear_model()
        assert state_prediction_error(m, 4.0) == pytest.approx(0.0)   # μ=m_x
        assert state_prediction_error(m, 2.4) == pytest.approx(2.4 - 4.0)

    def test_free_energy_formula(self) -> None:
        m = linear_model()
        c = predictive_coding_free_energy(m, Y_OBS, 3.0)
        eps_y, eps_x = 7.0 - 9.0, 3.0 - 4.0
        expected = 0.5 * (np.log(0.25) + eps_y**2 / 0.25 + np.log(0.25) + eps_x**2 / 0.25)
        assert c.free_energy == pytest.approx(expected, abs=1e-12)
        assert isinstance(c, PCFreeEnergy)
        assert c.weighted_eps_y == pytest.approx(eps_y**2 / 0.25)
        assert c.weighted_eps_x == pytest.approx(eps_x**2 / 0.25)
        assert c.mu_y == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Cross-chapter equivalence + gradient (ISC-45, ISC-46)
# ---------------------------------------------------------------------------


class TestEquivalenceAndGradient:
    def test_equals_ch4_map_form_up_to_constant(self) -> None:
        # ISC-45: F_MAP and −vfe_map_form differ only by a μ-independent constant.
        m = linear_model()
        lgm = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
        diffs = [
            predictive_coding_free_energy(m, Y_OBS, mu).free_energy - (-vfe_map_form(lgm, Y_OBS, mu))
            for mu in (1.0, 2.0, 2.4, 3.0, 4.0)
        ]
        assert max(diffs) - min(diffs) == pytest.approx(0.0, abs=1e-9)
        # the constant is the dropped log(2π) (two Gaussian normalizers, ½·2·log2π).
        assert diffs[0] == pytest.approx(-np.log(2 * np.pi), abs=1e-6)

    @pytest.mark.parametrize("model", [
        linear_model(),
        PredictiveCodingModel(g=QuadraticFunction(1.0, 1.0), sigma2_y=0.25, m_x=3.0, s2_x=0.25),
    ])
    def test_analytic_grad_matches_fd(self, model: PredictiveCodingModel) -> None:
        # ISC-46: derived ∂F/∂μ matches a central finite difference (sign resolved).
        y = 5.84 if isinstance(model.g, QuadraticFunction) else Y_OBS
        for mu in (0.5, 1.5, 2.2, 3.0, 4.0):
            an = pc_free_energy_grad(model, y, mu)
            fd = pc_free_energy_grad_fd(model, y, mu)
            assert an == pytest.approx(fd, abs=1e-5)

    def test_gradient_is_zero_at_minimum(self) -> None:
        # Linear MAP minimum is 2.4 (the posterior mean) → gradient vanishes there.
        m = linear_model()
        assert pc_free_energy_grad(m, Y_OBS, 2.4) == pytest.approx(0.0, abs=1e-9)


class TestAnalyticalLandmarks:
    """Closed forms used as oracles and figure annotations (new methods)."""

    def test_linear_fixed_point_equals_known_value(self) -> None:
        assert pc_linear_fixed_point(linear_model(), Y_OBS) == pytest.approx(2.4)

    def test_linear_fixed_point_zeroes_the_gradient(self) -> None:
        # The closed form must be a true stationary point of F for several configs.
        for kw in [dict(), dict(m_x=-1.0, s2_x=2.0),
                   dict(g=LinearFunction(-1.5, 0.5), sigma2_y=1.0)]:
            m = linear_model(**kw)
            mu_star = pc_linear_fixed_point(m, Y_OBS)
            assert pc_free_energy_grad(m, Y_OBS, mu_star) == pytest.approx(0.0, abs=1e-9)

    def test_linear_fixed_point_matches_grid_argmin(self) -> None:
        m = linear_model()
        grid = np.linspace(-5.0, 10.0, 30001)
        fvals = [predictive_coding_free_energy(m, Y_OBS, float(g)).free_energy for g in grid]
        assert pc_linear_fixed_point(m, Y_OBS) == pytest.approx(
            float(grid[int(np.argmin(fvals))]), abs=1e-3)

    def test_curvature_is_second_derivative(self) -> None:
        # For linear g, L is the exact constant ∂²F/∂μ² — check vs finite difference.
        m = linear_model()
        h, mu = 1e-3, 1.7
        f = lambda x: predictive_coding_free_energy(m, Y_OBS, x).free_energy  # noqa: E731
        d2 = (f(mu + h) - 2 * f(mu) + f(mu - h)) / h**2
        assert pc_curvature_linear(m) == pytest.approx(d2, rel=1e-4)
        assert pc_curvature_linear(m) == pytest.approx(20.0)  # λ_x + slope²·λ_y = 4+16

    def test_fixed_point_rejects_nonlinear(self) -> None:
        with pytest.raises(TypeError):
            pc_linear_fixed_point(
                PredictiveCodingModel(g=QuadraticFunction(1.0, 1.0)), Y_OBS)


# ---------------------------------------------------------------------------
# Statistics / validation helpers (ISC-55, ISC-56, ISC-57)
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_gradient_check_small_for_correct_grad(self) -> None:
        m = linear_model()
        err = gradient_check(
            lambda mu: predictive_coding_free_energy(m, Y_OBS, mu).free_energy,
            lambda mu: pc_free_energy_grad(m, Y_OBS, mu),
            3.0,
        )
        assert err < 1e-5

    def test_gradient_check_large_for_wrong_grad(self) -> None:
        m = linear_model()
        err = gradient_check(
            lambda mu: predictive_coding_free_energy(m, Y_OBS, mu).free_energy,
            lambda mu: -pc_free_energy_grad(m, Y_OBS, mu),  # deliberately wrong sign
            3.0,
        )
        assert err > 1.0

    def test_convergence_report_on_geometric_sequence(self) -> None:
        # Geometric decay with ratio 0.5 → empirical rate ≈ 0.5, monotone. (The
        # rate is measured against the trace's *final* value, so the tail residuals
        # bias it slightly below 0.5 on a finite trace — hence the tolerance.)
        trace = 0.5 ** np.arange(30)
        rep = convergence_report(trace)
        assert isinstance(rep, ConvergenceReport)
        assert rep.monotone
        assert rep.rate == pytest.approx(0.5, abs=0.05)
        assert 0.0 < rep.rate < 1.0          # linear convergence
        assert rep.max_increase == pytest.approx(0.0)
        assert rep.total_decrease == pytest.approx(trace[0] - trace[-1])

    def test_convergence_report_flags_increase(self) -> None:
        rep = convergence_report(np.array([1.0, 0.5, 0.7, 0.2]))
        assert not rep.monotone
        assert rep.max_increase == pytest.approx(0.2, abs=1e-12)

    def test_oracle_agreement(self) -> None:
        ok = oracle_agreement(2.401, 2.4, tol=1e-2)
        assert isinstance(ok, OracleAgreement)
        assert ok.passed and ok.abs_error == pytest.approx(0.001, abs=1e-9)
        assert not oracle_agreement(2.5, 2.4, tol=1e-2).passed
