from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from active_painter.body_inference import (
    BODY_INFERENCE_VERSION,
    MUJOCO_BODY_LIKELIHOOD,
    BodyLikelihoodSpec,
    BodyStateEstimator,
)
from active_painter.plant_interface import PhysicalSensorPacket, PlantCapabilities


JOINTS = ("yaw", "pitch", "roll", "elbow")


def _capabilities() -> PlantCapabilities:
    return PlantCapabilities(
        backend_id="body-inference-test",
        joint_names=JOINTS,
        position_limits_rad=(
            (-1.0, 1.0),
            (-1.5, 1.5),
            (-3.0, 3.0),
            (-1.0, 2.0),
        ),
        command_modes=("position",),
        physical_sensor_fields=(
            "encoder_position_rad",
            "encoder_velocity_rad_s",
            "motor_current_a",
            "bus_voltage_v",
            "contact_force_n",
            "tool_deflection_m",
            "contact_switch",
        ),
        nominal_control_period_s=0.001,
        supports_deterministic_step=True,
        supports_evaluation_truth=True,
    )


def _spec(*, encoder_position_std_rad: float = 0.02) -> BodyLikelihoodSpec:
    return BodyLikelihoodSpec(
        model_id="test-body-likelihood-v0",
        calibration_status="synthetic_test_only",
        encoder_position_std_rad=encoder_position_std_rad,
        encoder_velocity_std_rad_s=0.08,
        position_process_std_rad_sqrt_s=0.03,
        velocity_process_std_rad_s_per_sqrt_s=0.20,
        initial_velocity_std_rad_s=2.0,
        initial_contact_probability=0.1,
        contact_onset_rate_hz=0.2,
        contact_break_rate_hz=2.0,
        contact_switch_true_positive=0.97,
        contact_switch_false_positive=0.02,
        contact_force_std_n=0.25,
        initial_contact_force_std_n=5.0,
        contact_force_process_std_n_sqrt_s=1.0,
    )


def _packet(
    *,
    time_s: float = 0.0,
    position: tuple[float, ...] = (0.4, -0.6, 0.2, 1.0),
    velocity: tuple[float, ...] = (0.1, -0.2, 0.0, 0.3),
    contact_switch: bool | None = True,
    contact_force_n: float | None = 2.5,
) -> PhysicalSensorPacket:
    return PhysicalSensorPacket(
        sequence=int(round(time_s * 1000.0)),
        monotonic_time_s=time_s,
        joint_names=JOINTS,
        encoder_position_rad=position,
        encoder_velocity_rad_s=velocity,
        motor_current_a=(0.2, 0.3, 0.1, 0.1),
        bus_voltage_v=(48.0, 48.0, 48.0, 48.0),
        contact_force_n=contact_force_n,
        tool_deflection_m=0.001 if contact_force_n is not None else None,
        contact_switch=contact_switch,
        fault_flags=("watchdog_example",),
    )


def test_body_posterior_fuses_transition_prior_with_permitted_observations() -> None:
    estimator = BodyStateEstimator(_capabilities(), _spec())

    first = estimator.update(_packet())
    second = estimator.update(
        _packet(
            time_s=0.01,
            position=(0.42, -0.61, 0.21, 1.02),
            velocity=(0.12, -0.19, 0.03, 0.28),
        )
    )

    assert BODY_INFERENCE_VERSION == "body-inference-v0"
    assert first.posterior_revision == 0
    assert second.posterior_revision == 1
    assert first.inference_model_id == (
        "body-inference-v0:test-body-likelihood-v0"
    )
    assert first.calibration_status == "synthetic_test_only"
    assert second.inference_model_id == first.inference_model_id
    assert second.calibration_status == first.calibration_status
    assert second.monotonic_time_s == pytest.approx(0.01)
    assert second.joint_names == JOINTS
    second_observation = np.asarray((0.42, -0.61, 0.21, 1.02))
    assert estimator.last_transition_prior is not None
    prior_mean = np.asarray(
        estimator.last_transition_prior.joint_position_mean_rad
    )
    posterior_mean = np.asarray(second.joint_position_mean_rad)
    assert np.linalg.norm(posterior_mean - second_observation) < np.linalg.norm(
        prior_mean - second_observation
    )
    assert np.all(
        (posterior_mean - prior_mean)
        * (second_observation - posterior_mean)
        >= -1e-12
    )
    assert all(value > 0.0 for value in second.joint_position_variance_rad2)
    assert 0.0 < second.contact_probability < 1.0
    assert second.contact_force_mean_n == pytest.approx(2.5, abs=0.02)

    vfe = estimator.last_vfe
    assert vfe is not None
    assert vfe.total == pytest.approx(
        vfe.complexity + vfe.negative_log_likelihood
    )
    assert vfe.expected_log_likelihood == pytest.approx(
        -vfe.negative_log_likelihood
    )
    assert vfe.total == pytest.approx(sum(factor.total for factor in vfe.factors))
    assert {factor.name for factor in vfe.factors} == {
        "encoder_position",
        "encoder_velocity",
        "contact_switch",
        "contact_force",
    }
    assert vfe.used_observations == (
        "encoder_position_rad",
        "encoder_velocity_rad_s",
        "contact_switch",
        "contact_force_n",
    )
    assert set(vfe.unassimilated_observations) == {
        "motor_current_a",
        "bus_voltage_v",
        "tool_deflection_m",
        "fault_flags",
    }


def test_encoder_likelihood_precision_controls_posterior_precision() -> None:
    packet = _packet(contact_switch=None, contact_force_n=None)
    loose = BodyStateEstimator(
        _capabilities(),
        _spec(encoder_position_std_rad=0.20),
    ).update(packet)
    precise = BodyStateEstimator(
        _capabilities(),
        _spec(encoder_position_std_rad=0.005),
    ).update(packet)
    observation = np.asarray(packet.encoder_position_rad)
    loose_error = np.linalg.norm(
        np.asarray(loose.joint_position_mean_rad) - observation
    )
    precise_error = np.linalg.norm(
        np.asarray(precise.joint_position_mean_rad) - observation
    )

    assert max(precise.joint_position_variance_rad2) < min(
        loose.joint_position_variance_rad2
    )
    assert precise_error < loose_error


def test_contact_switch_posterior_and_vfe_match_bernoulli_model_evidence() -> None:
    spec = _spec()
    touching_estimator = BodyStateEstimator(_capabilities(), spec)
    clear_estimator = BodyStateEstimator(_capabilities(), spec)
    touching = touching_estimator.update(
        _packet(contact_switch=True, contact_force_n=None)
    )
    clear = clear_estimator.update(
        _packet(contact_switch=False, contact_force_n=None)
    )

    expected_touching = (
        spec.initial_contact_probability * spec.contact_switch_true_positive
    ) / (
        spec.initial_contact_probability * spec.contact_switch_true_positive
        + (1.0 - spec.initial_contact_probability)
        * spec.contact_switch_false_positive
    )
    assert touching.contact_probability == pytest.approx(expected_touching)
    assert touching.contact_probability > spec.initial_contact_probability
    assert clear.contact_probability < spec.initial_contact_probability

    vfe = touching_estimator.last_vfe
    assert vfe is not None
    contact_factor = next(
        factor for factor in vfe.factors if factor.name == "contact_switch"
    )
    evidence = (
        spec.initial_contact_probability * spec.contact_switch_true_positive
        + (1.0 - spec.initial_contact_probability)
        * spec.contact_switch_false_positive
    )
    assert contact_factor.total == pytest.approx(-math.log(evidence))


def test_missing_optional_contact_channels_leave_contact_at_transition_prior() -> None:
    estimator = BodyStateEstimator(_capabilities(), _spec())
    posterior = estimator.update(
        _packet(contact_switch=None, contact_force_n=None)
    )

    assert posterior.contact_probability == pytest.approx(
        estimator.likelihood.initial_contact_probability
    )
    assert posterior.contact_force_mean_n is None
    assert posterior.contact_force_variance_n2 is None
    assert estimator.last_vfe is not None
    assert {
        factor.name for factor in estimator.last_vfe.factors
    } == {"encoder_position", "encoder_velocity"}


def test_body_likelihood_requires_declared_nondefault_precision() -> None:
    with pytest.raises(ValueError, match="encoder_position_std_rad"):
        replace(_spec(), encoder_position_std_rad=0.0)


def test_mujoco_body_likelihood_is_explicitly_provisional_simulation_only() -> None:
    spec = MUJOCO_BODY_LIKELIHOOD

    assert spec.model_id == "mujoco-ideal-sensor-body-likelihood-v0"
    assert spec.calibration_status == (
        "provisional_simulation_only_not_hardware_calibrated"
    )
    assert spec.encoder_position_std_rad > 0.0
    assert spec.encoder_velocity_std_rad_s > 0.0
    assert spec.position_process_std_rad_sqrt_s > 0.0
    assert spec.velocity_process_std_rad_s_per_sqrt_s > 0.0
