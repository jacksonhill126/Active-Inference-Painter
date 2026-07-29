from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .plant_interface import (
    BodyBeliefSnapshot,
    PhysicalSensorPacket,
    PlantCapabilities,
)


BODY_INFERENCE_VERSION = "body-inference-v0"
_EPSILON = 1e-12


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def _probability(name: str, value: float, *, open_interval: bool = False) -> None:
    valid = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not math.isfinite(value) or not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}.")


@dataclass(frozen=True, slots=True)
class BodyLikelihoodSpec:
    """Versioned, explicit body likelihood and transition-prior assumptions.

    Values must come from a declared calibration or simulation experiment.
    There are intentionally no numerical defaults: silently inventing sensor
    precision would make posterior confidence look more established than it is.
    """

    model_id: str
    calibration_status: str
    encoder_position_std_rad: float
    encoder_velocity_std_rad_s: float
    position_process_std_rad_sqrt_s: float
    velocity_process_std_rad_s_per_sqrt_s: float
    initial_velocity_std_rad_s: float
    initial_contact_probability: float
    contact_onset_rate_hz: float
    contact_break_rate_hz: float
    contact_switch_true_positive: float
    contact_switch_false_positive: float
    contact_force_std_n: float
    initial_contact_force_std_n: float
    contact_force_process_std_n_sqrt_s: float

    def __post_init__(self) -> None:
        if not self.model_id or not self.calibration_status:
            raise ValueError("model_id and calibration_status must be non-empty.")
        for name in (
            "encoder_position_std_rad",
            "encoder_velocity_std_rad_s",
            "position_process_std_rad_sqrt_s",
            "velocity_process_std_rad_s_per_sqrt_s",
            "initial_velocity_std_rad_s",
            "contact_force_std_n",
            "initial_contact_force_std_n",
            "contact_force_process_std_n_sqrt_s",
        ):
            _positive_finite(name, float(getattr(self, name)))
        for name in ("contact_onset_rate_hz", "contact_break_rate_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        _probability(
            "initial_contact_probability",
            self.initial_contact_probability,
            open_interval=True,
        )
        _probability(
            "contact_switch_true_positive",
            self.contact_switch_true_positive,
            open_interval=True,
        )
        _probability(
            "contact_switch_false_positive",
            self.contact_switch_false_positive,
            open_interval=True,
        )
        if self.contact_switch_true_positive <= self.contact_switch_false_positive:
            raise ValueError(
                "contact_switch_true_positive must exceed "
                "contact_switch_false_positive."
            )


@dataclass(frozen=True, slots=True)
class BodyVFEFactor:
    """One likelihood factor's contribution to variational free energy."""

    name: str
    complexity: float
    negative_log_likelihood: float

    @property
    def total(self) -> float:
        return self.complexity + self.negative_log_likelihood


@dataclass(frozen=True, slots=True)
class BodyVFEComponents:
    """Separately logged VFE decomposition for the body posterior."""

    total: float
    complexity: float
    negative_log_likelihood: float
    expected_log_likelihood: float
    factors: tuple[BodyVFEFactor, ...]
    used_observations: tuple[str, ...]
    unassimilated_observations: tuple[str, ...]
    units: str = "nats"
    approximation: str = (
        "exact conjugate mean-field updates for diagonal Gaussian joint/force "
        "factors and a Bernoulli contact-switch factor; current, voltage, "
        "temperature, deflection, and fault channels are not assimilated"
    )


def _normal_update(
    name: str,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, BodyVFEFactor]:
    posterior_variance = 1.0 / (
        1.0 / prior_variance + 1.0 / observation_variance
    )
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + observation / observation_variance
    )
    complexity = 0.5 * np.sum(
        np.log(prior_variance / posterior_variance)
        + (
            posterior_variance
            + np.square(posterior_mean - prior_mean)
        )
        / prior_variance
        - 1.0
    )
    negative_log_likelihood = 0.5 * np.sum(
        np.log(2.0 * math.pi * observation_variance)
        + (
            np.square(observation - posterior_mean) + posterior_variance
        )
        / observation_variance
    )
    return (
        posterior_mean,
        posterior_variance,
        BodyVFEFactor(
            name=name,
            complexity=float(complexity),
            negative_log_likelihood=float(negative_log_likelihood),
        ),
    )


def _bernoulli_switch_update(
    prior_contact_probability: float,
    observed_contact: bool,
    spec: BodyLikelihoodSpec,
) -> tuple[float, BodyVFEFactor]:
    probability_if_contact = (
        spec.contact_switch_true_positive
        if observed_contact
        else 1.0 - spec.contact_switch_true_positive
    )
    probability_if_clear = (
        spec.contact_switch_false_positive
        if observed_contact
        else 1.0 - spec.contact_switch_false_positive
    )
    evidence = (
        prior_contact_probability * probability_if_contact
        + (1.0 - prior_contact_probability) * probability_if_clear
    )
    posterior = (
        prior_contact_probability * probability_if_contact
    ) / max(evidence, _EPSILON)
    prior = float(np.clip(prior_contact_probability, _EPSILON, 1.0 - _EPSILON))
    posterior = float(np.clip(posterior, _EPSILON, 1.0 - _EPSILON))
    complexity = (
        posterior * math.log(posterior / prior)
        + (1.0 - posterior)
        * math.log((1.0 - posterior) / (1.0 - prior))
    )
    negative_log_likelihood = -(
        posterior * math.log(max(probability_if_contact, _EPSILON))
        + (1.0 - posterior)
        * math.log(max(probability_if_clear, _EPSILON))
    )
    return posterior, BodyVFEFactor(
        name="contact_switch",
        complexity=complexity,
        negative_log_likelihood=negative_log_likelihood,
    )


class BodyStateEstimator:
    """Infer a body posterior from permitted physical sensor samples.

    This estimator is a compact M2 building block, not a painting controller.
    It neither reads simulator truth nor selects policies. Its transition prior
    is a diagonal constant-velocity approximation, and every assimilated
    observation has an explicit likelihood factor.
    """

    def __init__(
        self,
        capabilities: PlantCapabilities,
        likelihood: BodyLikelihoodSpec,
    ) -> None:
        self.capabilities = capabilities
        self.likelihood = likelihood
        self.posterior: BodyBeliefSnapshot | None = None
        self.last_transition_prior: BodyBeliefSnapshot | None = None
        self.last_vfe: BodyVFEComponents | None = None

    def reset(self) -> None:
        self.posterior = None
        self.last_transition_prior = None
        self.last_vfe = None

    def _initial_prior(self, packet: PhysicalSensorPacket) -> BodyBeliefSnapshot:
        limits = np.asarray(self.capabilities.position_limits_rad, dtype=np.float64)
        position_mean = np.mean(limits, axis=1)
        # A moment-matched Gaussian approximation to a uniform joint-limit
        # prior. Hard joint limits remain an external safety constraint.
        position_variance = np.square(limits[:, 1] - limits[:, 0]) / 12.0
        joint_count = len(packet.joint_names)
        return BodyBeliefSnapshot(
            monotonic_time_s=packet.monotonic_time_s,
            joint_names=packet.joint_names,
            joint_position_mean_rad=tuple(float(value) for value in position_mean),
            joint_position_variance_rad2=tuple(
                float(value) for value in position_variance
            ),
            joint_velocity_mean_rad_s=(0.0,) * joint_count,
            joint_velocity_variance_rad2_s2=(
                self.likelihood.initial_velocity_std_rad_s**2,
            )
            * joint_count,
            contact_probability=self.likelihood.initial_contact_probability,
            contact_force_mean_n=0.0 if packet.contact_force_n is not None else None,
            contact_force_variance_n2=(
                self.likelihood.initial_contact_force_std_n**2
                if packet.contact_force_n is not None
                else None
            ),
            posterior_revision=0,
        )

    def _transition_prior(
        self,
        packet: PhysicalSensorPacket,
    ) -> BodyBeliefSnapshot:
        previous = self.posterior
        if previous is None:
            return self._initial_prior(packet)
        if packet.joint_names != previous.joint_names:
            raise ValueError("Sensor joint order changed after estimator initialization.")
        dt = packet.monotonic_time_s - previous.monotonic_time_s
        if dt < -1e-12:
            raise ValueError("Sensor packet time moved backwards.")
        dt = max(0.0, dt)
        position_mean = np.asarray(previous.joint_position_mean_rad)
        position_variance = np.asarray(previous.joint_position_variance_rad2)
        velocity_mean = np.asarray(previous.joint_velocity_mean_rad_s)
        velocity_variance = np.asarray(previous.joint_velocity_variance_rad2_s2)
        predicted_position_mean = position_mean + dt * velocity_mean
        predicted_position_variance = (
            position_variance
            + dt * dt * velocity_variance
            + self.likelihood.position_process_std_rad_sqrt_s**2 * dt
        )
        predicted_velocity_variance = (
            velocity_variance
            + self.likelihood.velocity_process_std_rad_s_per_sqrt_s**2 * dt
        )

        total_rate = (
            self.likelihood.contact_onset_rate_hz
            + self.likelihood.contact_break_rate_hz
        )
        if total_rate > 0.0:
            equilibrium = self.likelihood.contact_onset_rate_hz / total_rate
            contact_probability = equilibrium + (
                previous.contact_probability - equilibrium
            ) * math.exp(-total_rate * dt)
        else:
            contact_probability = previous.contact_probability

        force_mean = previous.contact_force_mean_n
        force_variance = previous.contact_force_variance_n2
        if force_mean is None and packet.contact_force_n is not None:
            force_mean = 0.0
            force_variance = self.likelihood.initial_contact_force_std_n**2
        elif force_variance is not None:
            force_variance += (
                self.likelihood.contact_force_process_std_n_sqrt_s**2 * dt
            )

        return BodyBeliefSnapshot(
            monotonic_time_s=packet.monotonic_time_s,
            joint_names=previous.joint_names,
            joint_position_mean_rad=tuple(
                float(value) for value in predicted_position_mean
            ),
            joint_position_variance_rad2=tuple(
                float(value) for value in predicted_position_variance
            ),
            joint_velocity_mean_rad_s=previous.joint_velocity_mean_rad_s,
            joint_velocity_variance_rad2_s2=tuple(
                float(value) for value in predicted_velocity_variance
            ),
            contact_probability=float(contact_probability),
            contact_force_mean_n=force_mean,
            contact_force_variance_n2=force_variance,
            posterior_revision=previous.posterior_revision + 1,
        )

    def update(self, packet: PhysicalSensorPacket) -> BodyBeliefSnapshot:
        if packet.joint_names != self.capabilities.joint_names:
            raise ValueError(
                "Sensor packet joint names/order do not match plant capabilities."
            )
        prior = self._transition_prior(packet)
        self.last_transition_prior = prior
        factors: list[BodyVFEFactor] = []
        used = ["encoder_position_rad", "encoder_velocity_rad_s"]

        position_mean, position_variance, factor = _normal_update(
            "encoder_position",
            np.asarray(prior.joint_position_mean_rad),
            np.asarray(prior.joint_position_variance_rad2),
            np.asarray(packet.encoder_position_rad),
            np.full(
                len(packet.joint_names),
                self.likelihood.encoder_position_std_rad**2,
            ),
        )
        factors.append(factor)
        velocity_mean, velocity_variance, factor = _normal_update(
            "encoder_velocity",
            np.asarray(prior.joint_velocity_mean_rad_s),
            np.asarray(prior.joint_velocity_variance_rad2_s2),
            np.asarray(packet.encoder_velocity_rad_s),
            np.full(
                len(packet.joint_names),
                self.likelihood.encoder_velocity_std_rad_s**2,
            ),
        )
        factors.append(factor)

        contact_probability = prior.contact_probability
        if packet.contact_switch is not None:
            contact_probability, factor = _bernoulli_switch_update(
                contact_probability,
                packet.contact_switch,
                self.likelihood,
            )
            factors.append(factor)
            used.append("contact_switch")

        force_mean = prior.contact_force_mean_n
        force_variance = prior.contact_force_variance_n2
        if packet.contact_force_n is not None:
            if force_mean is None or force_variance is None:
                raise RuntimeError("Contact-force prior was not initialized.")
            updated_mean, updated_variance, factor = _normal_update(
                "contact_force",
                np.asarray([force_mean]),
                np.asarray([force_variance]),
                np.asarray([packet.contact_force_n]),
                np.asarray([self.likelihood.contact_force_std_n**2]),
            )
            force_mean = float(updated_mean[0])
            force_variance = float(updated_variance[0])
            factors.append(factor)
            used.append("contact_force_n")

        unassimilated = ["motor_current_a", "bus_voltage_v"]
        if packet.motor_temperature_c is not None:
            unassimilated.append("motor_temperature_c")
        if packet.tool_deflection_m is not None:
            unassimilated.append("tool_deflection_m")
        if packet.fault_flags:
            # Faults remain inputs to hard safety, outside the painting model.
            unassimilated.append("fault_flags")

        posterior = BodyBeliefSnapshot(
            monotonic_time_s=packet.monotonic_time_s,
            joint_names=packet.joint_names,
            joint_position_mean_rad=tuple(float(value) for value in position_mean),
            joint_position_variance_rad2=tuple(
                float(value) for value in position_variance
            ),
            joint_velocity_mean_rad_s=tuple(float(value) for value in velocity_mean),
            joint_velocity_variance_rad2_s2=tuple(
                float(value) for value in velocity_variance
            ),
            contact_probability=contact_probability,
            contact_force_mean_n=force_mean,
            contact_force_variance_n2=force_variance,
            posterior_revision=prior.posterior_revision,
        )
        complexity = float(sum(factor.complexity for factor in factors))
        negative_log_likelihood = float(
            sum(factor.negative_log_likelihood for factor in factors)
        )
        self.last_vfe = BodyVFEComponents(
            total=complexity + negative_log_likelihood,
            complexity=complexity,
            negative_log_likelihood=negative_log_likelihood,
            expected_log_likelihood=-negative_log_likelihood,
            factors=tuple(factors),
            used_observations=tuple(used),
            unassimilated_observations=tuple(unassimilated),
        )
        self.posterior = posterior
        return posterior
