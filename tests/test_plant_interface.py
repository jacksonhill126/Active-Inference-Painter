from dataclasses import fields

import pytest

from active_painter.env import StrokeAction
from active_painter.plant_interface import (
    BodyBeliefSnapshot,
    CounterfactualRolloutRequest,
    PhysicalSensorPacket,
    PlantCapabilities,
    PlantCommand,
    SimulatorEvaluationTruth,
)


JOINTS = ("yaw", "pitch", "roll", "elbow")


def _capabilities() -> PlantCapabilities:
    return PlantCapabilities(
        backend_id="test",
        joint_names=JOINTS,
        position_limits_rad=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0), (0.0, 2.5)),
        command_modes=("position",),
        physical_sensor_fields=(
            "encoder_position_rad",
            "encoder_velocity_rad_s",
            "motor_current_a",
            "bus_voltage_v",
        ),
        nominal_control_period_s=0.005,
        supports_deterministic_step=True,
        supports_evaluation_truth=True,
    )


def _belief() -> BodyBeliefSnapshot:
    return BodyBeliefSnapshot(
        monotonic_time_s=1.25,
        joint_names=JOINTS,
        joint_position_mean_rad=(0.0, -0.4, 0.0, 1.2),
        joint_position_variance_rad2=(0.01, 0.01, 0.02, 0.01),
        joint_velocity_mean_rad_s=(0.0, 0.0, 0.0, 0.0),
        joint_velocity_variance_rad2_s2=(0.02, 0.02, 0.03, 0.02),
        contact_probability=0.1,
        contact_force_mean_n=None,
        contact_force_variance_n2=None,
        posterior_revision=7,
    )


def test_command_validation_uses_declared_joint_order_and_si_units() -> None:
    command = PlantCommand(
        sequence=2,
        monotonic_time_s=1.5,
        position_target_rad=(0.0, -0.5, 0.1, 1.3),
        velocity_target_rad_s=(0.0, 0.0, 0.0, 0.0),
    )

    command.validate_for(_capabilities())

    with pytest.raises(ValueError, match="one value per declared joint"):
        PlantCommand(3, 1.6, (0.0, -0.5)).validate_for(_capabilities())


def test_sensor_packet_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="motor_current_a"):
        PhysicalSensorPacket(
            sequence=0,
            monotonic_time_s=0.0,
            joint_names=JOINTS,
            encoder_position_rad=(0.0, 0.0, 0.0, 0.0),
            encoder_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
            motor_current_a=(0.0,),
            bus_voltage_v=(24.0, 24.0, 24.0, 24.0),
        )


def test_agent_facing_records_exclude_simulator_truth_and_process_objects() -> None:
    agent_facing_types = (
        PhysicalSensorPacket,
        PlantCommand,
        BodyBeliefSnapshot,
        CounterfactualRolloutRequest,
    )
    forbidden = {
        "actual_pose",
        "actual_position_rad",
        "exact_contact",
        "material",
        "canvas",
        "plant",
        "sim",
        "process_rng",
        "rng_state",
    }

    for record_type in agent_facing_types:
        assert forbidden.isdisjoint(field.name for field in fields(record_type))

    truth_fields = {field.name for field in fields(SimulatorEvaluationTruth)}
    assert {"actual_position_rad", "exact_contact"}.issubset(truth_fields)


def test_counterfactual_request_starts_from_belief_and_independent_noise_seed() -> None:
    request = CounterfactualRolloutRequest(
        request_id="forecast-12",
        initial_belief=_belief(),
        stroke_intent=StrokeAction(0.2, 0.3, 0.7, 0.8, 0.05, 0.8, 1.0),
        motor_primitive_kind="elbow_pivot",
        model_snapshot_id="native-abstract-v0:weights-9",
        sample_count=8,
        independent_noise_seed=912,
    )

    assert request.initial_belief.posterior_revision == 7
    assert request.independent_noise_seed == 912
    assert not hasattr(request, "sim")
    assert not hasattr(request, "rng_state")
