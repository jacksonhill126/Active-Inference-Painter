"""Tests for Part III free-energy form helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    FreeEnergyForm,
    bethe_free_energy_form,
    expected_free_energy_form,
    free_energy_of_future,
    free_energy_variant_table,
    generalized_free_energy_form,
    observed_predicted_free_energy,
    renyi_bound,
    renyi_limit_energy,
)


def test_expected_free_energy_form_adds_risk_ambiguity_and_information() -> None:
    """EFE helper exposes the signed risk/ambiguity/epistemic decomposition."""
    form = expected_free_energy_form(risk=2.0, ambiguity=0.5, epistemic_value=0.25)
    assert isinstance(form, FreeEnergyForm)
    assert form.total == pytest.approx(2.25)
    assert form.term_vector(("risk", "ambiguity", "epistemic_value")).tolist() == pytest.approx(
        [2.0, 0.5, -0.25]
    )


def test_future_and_generalized_forms_are_transparent_sums() -> None:
    """Future-facing teaching forms keep their additive terms explicit."""
    fef = free_energy_of_future(1.0, 2.5)
    gfe = generalized_free_energy_form(1.0, 2.5, information_gain=0.5)
    assert fef.total == pytest.approx(3.5)
    assert gfe.total == pytest.approx(3.0)


def test_observed_predicted_form_validates_weight() -> None:
    """Observed/predicted blends reject invalid interpolation weights."""
    form = observed_predicted_free_energy(1.0, 3.0, observation_weight=0.25)
    assert form.total == pytest.approx(2.5)
    with pytest.raises(ValueError):
        observed_predicted_free_energy(1.0, 3.0, observation_weight=1.5)


def test_bethe_form_is_energy_minus_entropy_plus_penalty() -> None:
    """Bethe helper preserves the energy, entropy, and consistency penalty identity."""
    form = bethe_free_energy_form(4.0, 1.25, consistency_penalty=0.5)
    assert form.total == pytest.approx(3.25)


def test_renyi_bound_limit_matches_expected_energy() -> None:
    """Renyi certainty-equivalent approaches expected energy near alpha one."""
    probs = np.array([0.2, 0.3, 0.5])
    energies = np.array([1.0, 2.0, 4.0])
    assert renyi_limit_energy(probs, energies) == pytest.approx(2.8)
    assert renyi_bound(probs, energies, alpha=0.999) == pytest.approx(2.8, rel=2e-3)
    with pytest.raises(ValueError):
        renyi_bound(probs, energies, alpha=1.0)


def test_variant_table_shapes_and_identity() -> None:
    """Policy-indexed variant table returns aligned arrays and EFE identity."""
    table = free_energy_variant_table([1.0, 2.0], [0.5, 0.25], [0.2, 0.1])
    assert set(table) >= {"risk", "ambiguity", "epistemic_value", "expected_free_energy"}
    assert table["expected_free_energy"].shape == (2,)
    assert table["expected_free_energy"].tolist() == pytest.approx([1.3, 2.15])
