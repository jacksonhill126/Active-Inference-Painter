from __future__ import annotations

import numpy as np

from active_painter.uncertainty_calibration import (
    calibration_metrics,
    ood_disagreement_summary,
    precision_inventory,
    validation_variance_scale,
)


def test_calibration_metrics_report_signed_z_and_interval_coverage() -> None:
    residual = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    variance = np.ones_like(residual)

    metrics = calibration_metrics(residual, variance)

    assert metrics["element_count"] == 4
    assert metrics["finite"] is True
    assert np.isclose(metrics["signed_mean_z"], 0.5)
    assert np.isclose(metrics["mean_squared_z"], 1.5)
    assert np.isclose(metrics["coverage_50"], 0.25)
    assert np.isclose(metrics["coverage_90"], 0.75)


def test_validation_variance_scale_is_fit_from_standardized_residual_energy() -> None:
    residual = np.asarray([2.0, -2.0], dtype=np.float64)
    variance = np.asarray([2.0, 2.0], dtype=np.float64)

    assert np.isclose(validation_variance_scale(residual, variance), 2.0)


def test_precision_inventory_does_not_call_unobserved_prior_a_posterior() -> None:
    config = {
        "precision_beliefs_enabled": True,
        "modality_precision_beliefs_enabled": True,
        "transition_precision": 3.0,
    }
    ledger = {
        "transition": {
            "alpha": 1.0,
            "beta": 1.0 / 3.0,
            "observations": 0.0,
        }
    }

    inventory = precision_inventory(config, ledger)
    transition = inventory["terms"]["transition"]

    assert transition["status"] == "declared_prior_unobserved"
    assert transition["reported_mean"] == 3.0
    assert transition["used_in_predictive_intervals_or_transition_nll"] is False


def test_precision_inventory_distinguishes_inferred_and_fixed_terms() -> None:
    inferred = precision_inventory(
        {
            "precision_beliefs_enabled": True,
            "modality_precision_beliefs_enabled": True,
            "transition_precision": 2.0,
        },
        {"transition": {"alpha": 2.0, "beta": 0.5, "observations": 5}},
    )["terms"]["transition"]
    fixed = precision_inventory(
        {
            "precision_beliefs_enabled": False,
            "modality_precision_beliefs_enabled": False,
            "transition_precision": 2.0,
        },
        {"transition": {"alpha": 2.0, "beta": 0.5, "observations": 5}},
    )["terms"]["transition"]

    assert inferred["status"] == "inferred_gamma_posterior"
    assert inferred["reported_mean"] == 4.0
    assert fixed["status"] == "fixed_multiplier"
    assert fixed["reported_mean"] == 2.0


def test_ood_summary_applies_declared_ratio_gate() -> None:
    result = ood_disagreement_summary([1.0, 1.0], [2.0, 2.0], threshold=1.5)

    assert np.isclose(result["ood_to_in_distribution_ratio"], 2.0)
    assert result["passes_provisional_m2_gate"] is True
