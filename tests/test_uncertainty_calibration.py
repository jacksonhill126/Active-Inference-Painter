from __future__ import annotations

import numpy as np
import torch

from active_painter.uncertainty_calibration import (
    PredictionRecord,
    _masked_component_numpy,
    calibration_metrics,
    mixture_variance_scale,
    ood_disagreement_summary,
    predictive_mixture_calibration,
    precision_inventory,
    validation_variance_scale,
)


def _prediction_record(component_residual: np.ndarray) -> PredictionRecord:
    residual = np.asarray(component_residual[0], dtype=np.float64)
    return PredictionRecord(
        labels={},
        trajectory_id="test",
        residual=residual,
        learned_likelihood_variance=np.ones_like(residual),
        latent_variance=np.zeros_like(residual),
        ensemble_variance=np.zeros_like(residual),
        target_posterior_variance=np.zeros_like(residual),
        fixed_camera_likelihood_variance=np.ones_like(residual),
        mixture_nll=np.zeros_like(residual),
        action_support=np.ones(residual.shape[-1], dtype=bool),
        component_residual=np.asarray(component_residual, dtype=np.float64),
        component_variance=np.ones_like(component_residual, dtype=np.float64),
        component_log_weight=np.zeros_like(component_residual, dtype=np.float64),
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


def test_predictive_mixture_coverage_uses_full_cdf_not_moment_collapse() -> None:
    probabilities = torch.linspace(0.0005, 0.9995, 1000, dtype=torch.float64)
    quantiles = (2.0**0.5 * torch.erfinv(2.0 * probabilities - 1.0)).numpy()
    record = _prediction_record(quantiles.reshape(1, 1, -1))

    metrics = predictive_mixture_calibration([record])

    assert abs(metrics["coverage_50"] - 0.50) < 0.01
    assert abs(metrics["coverage_90"] - 0.90) < 0.01
    assert metrics["pit_ks_distance_from_uniform"] < 0.01


def test_mixture_variance_scale_optimizes_component_density_on_validation() -> None:
    probabilities = torch.linspace(0.001, 0.999, 999, dtype=torch.float64)
    residual = (
        2.0 * (2.0**0.5 * torch.erfinv(2.0 * probabilities - 1.0))
    ).numpy()
    record = _prediction_record(residual.reshape(1, 1, -1))

    fitted = mixture_variance_scale([record])

    assert abs(fitted - 4.0) < 0.1


def test_masked_component_extraction_preserves_component_channel_cell_order() -> None:
    values = torch.arange(2 * 2 * 4 * 2 * 3, dtype=torch.float64).reshape(
        2, 2, 4, 2, 3
    )
    mask = torch.tensor(
        [
            [[[True, False, True], [False, True, False]]],
            [[[False, True, False], [True, False, True]]],
        ]
    )

    extracted = _masked_component_numpy(values, mask)

    assert len(extracted) == 2
    assert extracted[0].shape == (2, 4, 3)
    np.testing.assert_array_equal(
        extracted[0], values[:, 0][:, :, mask[0, 0]].numpy()
    )
    np.testing.assert_array_equal(
        extracted[1], values[:, 1][:, :, mask[1, 0]].numpy()
    )
