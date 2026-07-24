"""Tests for thermodynamic / FEP bridge helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    GaussianBelief,
    LinearGaussianModel,
    ThermodynamicState,
    boltzmann_entropy,
    canonical_probabilities,
    enthalpy,
    expected_energy,
    gibbs_free_energy,
    helmholtz_free_energy,
    variational_free_energy,
    vfe_thermodynamic_state,
)


class TestCanonicalProbabilities:
    def test_lower_energy_state_gets_higher_probability(self) -> None:
        """Canonical probabilities normalize stabilized Boltzmann weights."""
        energies = np.array([0.0, 1.0, 2.0])
        probs = canonical_probabilities(energies, temperature=1.0)

        assert probs.shape == energies.shape
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] > probs[1] > probs[2]

    def test_temperature_flattens_distribution(self) -> None:
        """Higher temperature reduces the spread between state probabilities."""
        energies = np.array([0.0, 2.0])
        cold = canonical_probabilities(energies, temperature=0.5)
        hot = canonical_probabilities(energies, temperature=5.0)

        assert cold[0] - cold[1] > hot[0] - hot[1]

    def test_temperature_must_be_positive(self) -> None:
        """Canonical probabilities reject non-physical non-positive temperatures."""
        with pytest.raises(ValueError, match="temperature"):
            canonical_probabilities([0.0, 1.0], temperature=0.0)


class TestThermodynamicPotentials:
    def test_scalar_potentials_match_hand_computed_values(self) -> None:
        """Free-energy, enthalpy, and Gibbs helpers expose the textbook algebra."""
        assert expected_energy([0.25, 0.75], [2.0, 6.0]) == pytest.approx(5.0)
        assert boltzmann_entropy([0.25, 0.75]) == pytest.approx(
            -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
        )
        assert helmholtz_free_energy(energy=5.0, entropy=1.5, temperature=2.0) == 2.0
        assert enthalpy(energy=5.0, pressure=3.0, volume=4.0) == 17.0
        assert gibbs_free_energy(
            energy=5.0,
            entropy=1.5,
            temperature=2.0,
            pressure=3.0,
            volume=4.0,
        ) == 14.0

    def test_state_properties_share_one_convention(self) -> None:
        """ThermodynamicState derives H, A, and G from one validated state."""
        state = ThermodynamicState(energy=5.0, entropy=1.5, temperature=2.0,
                                   pressure=3.0, volume=4.0)

        assert state.helmholtz_free_energy == pytest.approx(2.0)
        assert state.enthalpy == pytest.approx(17.0)
        assert state.gibbs_free_energy == pytest.approx(14.0)

    def test_probabilities_must_match_energies(self) -> None:
        """Expected energy requires one probability per energy level."""
        with pytest.raises(ValueError, match="same shape"):
            expected_energy([1.0], [1.0, 2.0])


class TestVFEThermodynamicBridge:
    def test_vfe_state_equals_variational_free_energy_at_unit_temperature(self) -> None:
        """At T=1 and pV=0, U - TS equals the existing VFE calculation."""
        model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                    m_x=4.0, s2_x=0.25)
        x_grid = np.linspace(-6.0, 12.0, 2001)
        q = GaussianBelief(mu=2.2, var=0.18)
        components = variational_free_energy(q, model, 7.0, x_grid)

        state = vfe_thermodynamic_state(q, model, 7.0, x_grid)

        assert state.energy == pytest.approx(-components.energy)
        assert state.entropy == pytest.approx(-components.neg_entropy)
        assert state.helmholtz_free_energy == pytest.approx(components.free_energy)

    def test_temperature_changes_entropy_weight_without_changing_energy(self) -> None:
        """Temperature changes the thermodynamic potential, not the VFE ingredients."""
        model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                    m_x=4.0, s2_x=0.25)
        x_grid = np.linspace(-6.0, 12.0, 2001)
        q = GaussianBelief(mu=2.0, var=0.25)

        unit = vfe_thermodynamic_state(q, model, 7.0, x_grid, temperature=1.0)
        hot = vfe_thermodynamic_state(q, model, 7.0, x_grid, temperature=2.0)

        assert hot.energy == pytest.approx(unit.energy)
        assert hot.entropy == pytest.approx(unit.entropy)
        assert hot.helmholtz_free_energy == pytest.approx(
            unit.energy - 2.0 * unit.entropy
        )
