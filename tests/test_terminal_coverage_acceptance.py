"""AI-106 acceptance tests for terminal coverage and stopping.

These tests deliberately use estimators that are independent of the analytic
Beta--Beta expression used by the planner.  They pin both the part that works
(the analytic expression agrees with direct samples *when* the forecast really
is Beta) and the limitation that matters for M2 (two bounded families with the
same first two moments need not have the same terminal risk).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.distributions import Beta, kl_divergence

from active_painter.config import PainterConfig
from active_painter.efe_common import coverage_beta_approximation
from active_painter.preferences import TerminalCoveragePreference


def _target(dtype: torch.dtype = torch.float64) -> Beta:
    config = PainterConfig()
    return Beta(
        torch.tensor(config.target_coverage * config.terminal_concentration, dtype=dtype),
        torch.tensor(
            (1.0 - config.target_coverage) * config.terminal_concentration,
            dtype=dtype,
        ),
    )


@pytest.mark.parametrize(
    ("mean", "variance"),
    [
        (0.87, 0.0002),  # narrow forecast in the preferred band
        (0.80, 0.0100),  # moderate bounded uncertainty
        (0.65, 0.0500),  # broad, asymmetric bounded uncertainty
    ],
)
def test_moment_matched_beta_risk_agrees_with_direct_monte_carlo(
    mean: float,
    variance: float,
) -> None:
    """Direct q samples recover the analytic KL across three variance regimes."""

    q = coverage_beta_approximation(
        torch.tensor(mean, dtype=torch.float64),
        torch.tensor(variance, dtype=torch.float64),
        concentration_floor=0.0,
    )
    target = _target()
    exact = kl_divergence(q, target)

    # torch.distributions does not accept a Generator.  The seed makes this a
    # deterministic statistical acceptance test; its tolerance is six measured
    # standard errors with a small numerical floor.
    torch.manual_seed(1060 + int(round(mean * 100)))
    samples = q.sample((120_000,))
    log_ratio = q.log_prob(samples) - target.log_prob(samples)
    estimate = log_ratio.mean()
    standard_error = log_ratio.std(unbiased=True) / math.sqrt(log_ratio.numel())

    assert float(estimate) == pytest.approx(
        float(exact),
        abs=max(0.03, 6.0 * float(standard_error)),
    )


def _logit_normal_quadrature(
    logit_mean: float,
    logit_std: float,
    nodes: int = 96,
) -> tuple[float, float, float]:
    """Return coverage mean, variance, and KL(logit-normal || target Beta).

    Gauss--Hermite quadrature is deterministic and independent of the planner's
    Beta moment match.  Stable log-sigmoid identities avoid clipping the
    bounded samples near zero or one.
    """

    z, weights = np.polynomial.hermite.hermgauss(nodes)
    y = float(logit_mean) + math.sqrt(2.0) * float(logit_std) * z
    normalized_weights = weights / math.sqrt(math.pi)
    log_coverage = -np.logaddexp(0.0, -y)
    log_uncovered = -np.logaddexp(0.0, y)
    coverage = np.exp(log_coverage)
    mean = float(np.sum(normalized_weights * coverage))
    variance = float(np.sum(normalized_weights * (coverage - mean) ** 2))

    log_q = (
        -0.5 * ((y - float(logit_mean)) / float(logit_std)) ** 2
        - math.log(float(logit_std))
        - 0.5 * math.log(2.0 * math.pi)
        - log_coverage
        - log_uncovered
    )
    config = PainterConfig()
    alpha = config.target_coverage * config.terminal_concentration
    beta = (1.0 - config.target_coverage) * config.terminal_concentration
    log_target = (
        (alpha - 1.0) * log_coverage
        + (beta - 1.0) * log_uncovered
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )
    risk = float(np.sum(normalized_weights * (log_q - log_target)))
    return mean, variance, risk


def test_beta_moment_match_is_not_family_invariant_at_high_uncertainty() -> None:
    """A bounded logit-normal exposes information lost by two-moment matching.

    At low uncertainty the Beta approximation is excellent.  At broad
    uncertainty, the alternative has the same measured mean/variance passed to
    the planner but a materially different shape and terminal KL.  This is the
    quantitative reason AI-106 does not approve the family for M2.
    """

    logit_mean = math.log(0.8 / 0.2)
    differences: list[float] = []
    for logit_std in (0.2, 0.65, 2.5):
        mean, variance, alternative_risk = _logit_normal_quadrature(logit_mean, logit_std)
        beta_forecast = coverage_beta_approximation(
            torch.tensor(mean, dtype=torch.float64),
            torch.tensor(variance, dtype=torch.float64),
            concentration_floor=PainterConfig().terminal_forecast_concentration_floor,
        )
        beta_risk = float(kl_divergence(beta_forecast, _target()))
        differences.append(beta_risk - alternative_risk)

    assert abs(differences[0]) < 0.01
    assert abs(differences[1]) < 0.25
    assert abs(differences[2]) > 20.0


def test_boundary_floor_preserves_mean_but_collapses_near_blank_variance() -> None:
    """Quantify the declared floor's consequential cold-start behavior."""

    mean = torch.tensor(1.0e-4, dtype=torch.float64)
    variance = torch.tensor(2.29e-3**2, dtype=torch.float64)
    unrestricted = coverage_beta_approximation(mean, variance, concentration_floor=0.0)
    runtime = coverage_beta_approximation(mean, variance, concentration_floor=1.0)

    assert float(runtime.mean) == pytest.approx(float(unrestricted.mean), abs=1e-15)
    assert float(runtime.variance) < float(unrestricted.variance) / 500.0
    assert float(unrestricted.concentration1) < 0.01
    assert float(runtime.concentration1) == pytest.approx(1.0, abs=1e-12)

    unrestricted_risk = float(kl_divergence(unrestricted, _target()))
    runtime_risk = float(kl_divergence(runtime, _target()))
    assert unrestricted_risk > 50_000.0
    assert 800.0 < runtime_risk < 1_000.0


def test_boundary_clamps_discard_extreme_variance_information() -> None:
    """Very broad and very narrow inputs saturate at declared family bounds."""

    # Above the admissible Bernoulli variance at mean 0.5, all inputs collapse
    # to the concentration-minimum uniform Beta(1, 1).
    broad_a = coverage_beta_approximation(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.30, dtype=torch.float64),
        concentration_floor=1.0,
    )
    broad_b = coverage_beta_approximation(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(3.00, dtype=torch.float64),
        concentration_floor=1.0,
    )
    assert float(broad_a.concentration1) == pytest.approx(1.0)
    assert float(broad_a.concentration0) == pytest.approx(1.0)
    assert float(broad_b.concentration1) == pytest.approx(1.0)
    assert float(broad_b.concentration0) == pytest.approx(1.0)

    # Below the variance floor/concentration ceiling, distinct tiny variances
    # likewise produce the same capped concentration.
    narrow_a = coverage_beta_approximation(
        torch.tensor(0.87, dtype=torch.float64),
        torch.tensor(1.0e-16, dtype=torch.float64),
        concentration_floor=1.0,
    )
    narrow_b = coverage_beta_approximation(
        torch.tensor(0.87, dtype=torch.float64),
        torch.tensor(1.0e-12, dtype=torch.float64),
        concentration_floor=1.0,
    )
    assert float(narrow_a.concentration1 + narrow_a.concentration0) == pytest.approx(1.0e6)
    assert float(narrow_b.concentration1 + narrow_b.concentration0) == pytest.approx(1.0e6)


def test_terminal_preference_parameters_remain_declared_constants() -> None:
    """AI-106 records a prior preference, never a learned coverage discovery."""

    config = PainterConfig()
    preference = TerminalCoveragePreference(config)
    assert float(preference.alpha) == pytest.approx(
        config.target_coverage * config.terminal_concentration,
        rel=1e-6,
    )
    assert float(preference.beta) == pytest.approx(
        (1.0 - config.target_coverage) * config.terminal_concentration,
        rel=1e-6,
    )
