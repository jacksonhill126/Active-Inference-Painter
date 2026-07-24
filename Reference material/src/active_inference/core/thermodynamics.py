"""Thermodynamic bridge helpers for active-inference free-energy terms.

The functions in this module are intentionally small and explicit. They expose
standard statistical-physics potentials while documenting the active-inference
analogy used by the extras topic orchestrators:

``U = E_q[-log p(x, y)]`` and ``S = H[q]``.

At unit temperature and zero pressure-volume contribution, ``U - T S`` equals
the variational free energy computed by :func:`active_inference.core.variational
.variational_free_energy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generative_model import GenerativeModel
from .variational import QDensity, variational_free_energy


def _positive_temperature(temperature: float) -> float:
    """Validate and return a finite positive temperature."""
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return value


def _finite_scalar(name: str, value: float) -> float:
    """Validate and return a finite scalar thermodynamic input."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _probability_vector(probabilities: np.ndarray | list[float]) -> np.ndarray:
    """Validate and return a normalized one-dimensional probability vector."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D vector")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    probs = probs / total
    return probs


@dataclass(frozen=True)
class ThermodynamicState:
    """Validated thermodynamic state used for FEP analogy calculations.

    Parameters
    ----------
    energy : float
        Internal energy ``U``. In the VFE bridge this is
        ``E_q[-log p(x, y)]``.
    entropy : float
        Entropy ``S``. In the VFE bridge this is differential entropy ``H[q]``.
    temperature : float, default=1.0
        Positive entropy multiplier ``T``.
    pressure : float, default=0.0
        Optional analogy pressure ``p``.
    volume : float, default=0.0
        Optional analogy volume ``V``.
    """

    energy: float
    entropy: float
    temperature: float = 1.0
    pressure: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        """Validate scalar fields after dataclass initialization."""
        object.__setattr__(self, "energy", _finite_scalar("energy", self.energy))
        object.__setattr__(self, "entropy", _finite_scalar("entropy", self.entropy))
        object.__setattr__(self, "temperature", _positive_temperature(self.temperature))
        object.__setattr__(self, "pressure", _finite_scalar("pressure", self.pressure))
        object.__setattr__(self, "volume", _finite_scalar("volume", self.volume))

    @property
    def helmholtz_free_energy(self) -> float:
        """Return Helmholtz free energy ``A = U - T S``."""
        return helmholtz_free_energy(self.energy, self.entropy, self.temperature)

    @property
    def enthalpy(self) -> float:
        """Return enthalpy ``H = U + p V``."""
        return enthalpy(self.energy, self.pressure, self.volume)

    @property
    def gibbs_free_energy(self) -> float:
        """Return Gibbs free energy ``G = H - T S``."""
        return gibbs_free_energy(
            self.energy,
            self.entropy,
            self.temperature,
            self.pressure,
            self.volume,
        )


def canonical_probabilities(
    energies: np.ndarray | list[float],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return canonical Boltzmann probabilities for finite energy levels.

    The returned vector is proportional to ``exp(-energy / temperature)`` and
    normalized with a max-shift for numerical stability.
    """
    temp = _positive_temperature(temperature)
    e = np.asarray(energies, dtype=float)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("energies must be a non-empty 1-D vector")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    log_weights = -e / temp
    log_weights = log_weights - float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def expected_energy(
    probabilities: np.ndarray | list[float],
    energies: np.ndarray | list[float],
) -> float:
    """Return ``E_p[energy]`` for matching probability and energy vectors."""
    probs = _probability_vector(probabilities)
    e = np.asarray(energies, dtype=float)
    if e.shape != probs.shape:
        raise ValueError("probabilities and energies must have the same shape")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    return float(probs @ e)


def boltzmann_entropy(probabilities: np.ndarray | list[float]) -> float:
    """Return discrete Boltzmann/Shannon entropy ``-sum p log p`` in nats."""
    probs = _probability_vector(probabilities)
    terms = np.zeros_like(probs)
    mask = probs > 0.0
    terms[mask] = -probs[mask] * np.log(probs[mask])
    return float(np.sum(terms))


def helmholtz_free_energy(
    energy: float,
    entropy: float,
    temperature: float = 1.0,
) -> float:
    """Return Helmholtz free energy ``A = U - T S``."""
    return (
        _finite_scalar("energy", energy)
        - _positive_temperature(temperature) * _finite_scalar("entropy", entropy)
    )


def enthalpy(energy: float, pressure: float = 0.0, volume: float = 0.0) -> float:
    """Return enthalpy ``H = U + p V``."""
    return (
        _finite_scalar("energy", energy)
        + _finite_scalar("pressure", pressure) * _finite_scalar("volume", volume)
    )


def gibbs_free_energy(
    energy: float,
    entropy: float,
    temperature: float = 1.0,
    pressure: float = 0.0,
    volume: float = 0.0,
) -> float:
    """Return Gibbs free energy ``G = H - T S``."""
    return enthalpy(energy, pressure, volume) - (
        _positive_temperature(temperature) * _finite_scalar("entropy", entropy)
    )


def vfe_thermodynamic_state(
    q: QDensity,
    model: GenerativeModel,
    y: float,
    x_grid: np.ndarray,
    *,
    temperature: float = 1.0,
    pressure: float = 0.0,
    volume: float = 0.0,
) -> ThermodynamicState:
    """Map variational free-energy components into a thermodynamic state.

    The bridge uses ``U = E_q[-log p(x,y)]`` and ``S = H[q]``. Therefore at
    ``temperature=1``, ``pressure=0``, and ``volume=0``, the state's
    ``helmholtz_free_energy`` property equals the existing variational free
    energy value.
    """
    components = variational_free_energy(q, model, y, x_grid)
    return ThermodynamicState(
        energy=-components.energy,
        entropy=-components.neg_entropy,
        temperature=temperature,
        pressure=pressure,
        volume=volume,
    )
