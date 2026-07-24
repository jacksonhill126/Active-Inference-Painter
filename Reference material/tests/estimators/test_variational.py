"""Tests for ``estimators.variational`` — the three Chapter 4 VI algorithms.

* ``coordinate_search_vfe``  — Algorithm 4.2.1 (zero-order coordinate search)
* ``fixed_form_vi``          — Algorithm 4.6.1 (gradient descent on (mu, var))
* ``free_form_cavi``         — Algorithm 4.5.1 (mean-field CAVI on x, beta0, beta1)

Oracle: the exact grid posterior of Example 4.1 is ``N(2.4, 0.05)`` with
``y = 7``; every algorithm's VFE trace must be (weakly) monotone non-increasing
and bounded below by the surprisal ``-log p(y)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.inference import GridBayesianInference
from active_inference.estimators.variational import (
    CAVIResult,
    CoordinateSearchResult,
    FixedFormResult,
    MeanFieldConfig,
    coordinate_search_vfe,
    fixed_form_vi,
    free_form_cavi,
)

Y_OBS = 7.0
POST_MEAN = 2.4
POST_VAR = 0.05


@pytest.fixture
def x_grid() -> np.ndarray:
    return np.linspace(-6.0, 12.0, 2001)


def make_model(**overrides) -> LinearGaussianModel:
    kwargs = dict(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25,
                  prior_kind="gaussian")
    kwargs.update(overrides)
    return LinearGaussianModel(**kwargs)


def _is_monotone_non_increasing(fes: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(np.diff(fes) <= tol))


# ---------------------------------------------------------------------------
# §4.2 — coordinate search (Algorithm 4.2.1)
# ---------------------------------------------------------------------------


class TestCoordinateSearch:
    def test_returns_result_type(self, x_grid: np.ndarray) -> None:
        res = coordinate_search_vfe(make_model(), Y_OBS, x_grid)
        assert isinstance(res, CoordinateSearchResult)
        assert len(res.mus) == len(res.vars_) == len(res.free_energies)

    def test_monotone_non_increasing(self, x_grid: np.ndarray) -> None:
        res = coordinate_search_vfe(make_model(), Y_OBS, x_grid,
                                    kappa=0.01, n_iter=20)
        assert _is_monotone_non_increasing(res.free_energies)

    def test_moves_from_prior_toward_posterior(self, x_grid: np.ndarray) -> None:
        # Book params (kappa=0.01, 20 it) deliberately stop short of the minimum
        # (§4.4); we only require monotone motion from prior 4.0 toward 2.4.
        res = coordinate_search_vfe(make_model(), Y_OBS, x_grid,
                                    kappa=0.01, n_iter=20)
        assert res.mus[0] == pytest.approx(4.0)        # starts at the prior mean
        assert res.belief.mu < res.mus[0]              # moved toward 2.4
        assert POST_MEAN <= res.belief.mu <= 4.0

    def test_stays_above_surprisal(self, x_grid: np.ndarray) -> None:
        model = make_model()
        surprisal = -float(
            GridBayesianInference(model=model, x_grid=x_grid).infer(Y_OBS).log_evidence)
        res = coordinate_search_vfe(model, Y_OBS, x_grid)
        assert np.all(res.free_energies >= surprisal - 1e-6)

    def test_extended_run_reaches_minimum(self, x_grid: np.ndarray) -> None:
        # Larger step + budget (the orchestrator's --extended path) converges to 2.4.
        res = coordinate_search_vfe(make_model(), Y_OBS, x_grid,
                                    kappa=0.05, n_iter=400)
        assert res.belief.mu == pytest.approx(POST_MEAN, abs=0.1)

    def test_custom_init_respected(self, x_grid: np.ndarray) -> None:
        res = coordinate_search_vfe(make_model(), Y_OBS, x_grid,
                                    mu0=1.0, var0=0.3, kappa=0.01, n_iter=5)
        assert res.mus[0] == pytest.approx(1.0)
        assert res.vars_[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# §4.6 — fixed-form VI (Algorithm 4.6.1)
# ---------------------------------------------------------------------------


class TestFixedFormVI:
    def test_returns_result_type(self, x_grid: np.ndarray) -> None:
        res = fixed_form_vi(make_model(), Y_OBS, x_grid)
        assert isinstance(res, FixedFormResult)
        assert len(res.components) == len(res.mus)

    def test_converges_to_exact_posterior(self, x_grid: np.ndarray) -> None:
        res = fixed_form_vi(make_model(), Y_OBS, x_grid, lr=5e-3, n_iter=3000)
        assert res.belief.mu == pytest.approx(POST_MEAN, abs=1e-2)
        assert res.belief.var == pytest.approx(POST_VAR, abs=1e-2)

    def test_monotone_non_increasing(self, x_grid: np.ndarray) -> None:
        res = fixed_form_vi(make_model(), Y_OBS, x_grid, lr=5e-3, n_iter=3000)
        assert _is_monotone_non_increasing(res.free_energies)

    def test_reaches_surprisal_bound(self, x_grid: np.ndarray) -> None:
        model = make_model()
        surprisal = -float(
            GridBayesianInference(model=model, x_grid=x_grid).infer(Y_OBS).log_evidence)
        res = fixed_form_vi(model, Y_OBS, x_grid, lr=5e-3, n_iter=3000)
        # At the posterior the bound is tight, so final F ~ surprisal.
        assert res.final_free_energy == pytest.approx(surprisal, abs=1e-2)
        assert res.final_free_energy >= surprisal - 1e-6

    def test_mu_stays_on_grid(self, x_grid: np.ndarray) -> None:
        res = fixed_form_vi(make_model(), Y_OBS, x_grid, lr=5e-3, n_iter=3000)
        assert np.all(res.mus >= x_grid[0] - 1e-9)
        assert np.all(res.mus <= x_grid[-1] + 1e-9)
        assert np.all(res.vars_ > 0)

    def test_converged_flag_set_on_plateau(self, x_grid: np.ndarray) -> None:
        res = fixed_form_vi(make_model(), Y_OBS, x_grid, lr=5e-3, n_iter=20000,
                            tol=1e-9)
        assert res.converged
        assert res.n_iter_run < 20000


# ---------------------------------------------------------------------------
# §4.5 — free-form mean-field CAVI (Algorithm 4.5.1)
# ---------------------------------------------------------------------------


class TestFreeFormCAVI:
    def test_returns_result_type(self) -> None:
        res = free_form_cavi(Y_OBS)
        assert isinstance(res, CAVIResult)
        assert len(res.mu_x) == len(res.free_energies)

    def test_monotone_non_increasing(self) -> None:
        res = free_form_cavi(Y_OBS, n_sweeps=50)
        assert _is_monotone_non_increasing(res.free_energies)

    def test_converges_and_reconstructs_observation(self) -> None:
        # At the fixed point the posterior means should reconstruct y:
        # beta0 + beta1 * x ~ y = 7.
        res = free_form_cavi(Y_OBS, n_sweeps=200, tol=1e-12)
        assert res.converged
        recon = res.q_b0.mu + res.q_b1.mu * res.q_x.mu
        assert recon == pytest.approx(Y_OBS, abs=0.5)

    def test_partitions_move_from_priors(self) -> None:
        cfg = MeanFieldConfig()
        res = free_form_cavi(Y_OBS, cfg=cfg, n_sweeps=100)
        # x starts at prior mean 4 and rises; beta0/beta1 start at 0 and grow.
        assert res.mu_x[0] == pytest.approx(cfg.m_x)
        assert res.mu_b0[0] == pytest.approx(cfg.m_b0)
        assert res.mu_b1[0] == pytest.approx(cfg.m_b1)
        assert res.q_b1.mu > 0.0          # positive slope learned
        assert res.q_x.mu > cfg.m_x       # x pulled upward to explain y

    def test_all_posterior_variances_positive(self) -> None:
        res = free_form_cavi(Y_OBS, n_sweeps=100)
        assert res.q_x.var > 0
        assert res.q_b0.var > 0
        assert res.q_b1.var > 0

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            MeanFieldConfig(sigma2_y=0.0)
        with pytest.raises(ValueError):
            MeanFieldConfig(s2_x=-1.0)
