import copy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from active_painter.arm_control import ik_pose_for_canvas_point
from active_painter.arm_sim import ArmPainterSim, JointPlant
from active_painter.config import PainterConfig
from active_painter.mujoco_backend import MujocoJointPlant, MujocoPlantBackend
from active_painter.plant_interface import (
    EvaluationTruthProvider,
    PlantBackend,
    PlantCommand,
    SteppablePlantBackend,
)
from active_painter.web_runtime import WebSimRuntime


def test_mujoco_backend_implements_si_plant_and_truth_contracts() -> None:
    backend = MujocoPlantBackend()

    assert isinstance(backend, PlantBackend)
    assert isinstance(backend, SteppablePlantBackend)
    assert isinstance(backend, EvaluationTruthProvider)
    assert backend.capabilities.joint_names == ("yaw", "pitch", "roll", "elbow")
    assert backend.capabilities.backend_id == "mujoco-robstride-direct-v3"
    assert backend.capabilities.nominal_control_period_s == pytest.approx(0.001)

    contact_q = backend.keyframe_qpos("contact_probe")[:4]
    backend.set_state(contact_q, control_rad=contact_q)
    backend.send_command(
        PlantCommand(
            sequence=1,
            monotonic_time_s=0.0,
            position_target_rad=tuple(float(value) for value in contact_q),
        )
    )
    backend.step(0.1)
    packet = backend.read_sensors()
    truth = backend.read_evaluation_truth()

    assert packet.contact_switch is True
    assert packet.contact_force_n is not None and packet.contact_force_n > 0.0
    assert packet.tool_deflection_m is not None and packet.tool_deflection_m > 0.0
    assert np.isfinite(packet.encoder_position_rad).all()
    assert np.isfinite(packet.encoder_velocity_rad_s).all()
    assert np.isfinite(packet.motor_current_a).all()
    assert truth.exact_contact is True
    assert truth.exact_contact_force_n == pytest.approx(packet.contact_force_n)
    # Named MuJoCo sensors are sampled before the final integration stage, so
    # they may trail privileged qpos truth by one 1 ms model timestep.
    assert truth.actual_position_rad == pytest.approx(
        packet.encoder_position_rad,
        abs=1e-3,
    )


def test_mujoco_backend_snapshot_restore_is_deterministic() -> None:
    backend = MujocoPlantBackend()
    home = backend.keyframe_qpos("safe_home")[:4]
    contact = backend.keyframe_qpos("contact_probe")[:4]
    backend.set_state(home, control_rad=home)
    backend.send_command(
        PlantCommand(
            sequence=2,
            monotonic_time_s=0.0,
            position_target_rad=tuple(float(value) for value in contact),
        )
    )
    snapshot = backend.state_snapshot()

    backend.step(0.075)
    first_q = backend.joint_position_rad()
    first_v = backend.joint_velocity_rad_s()
    backend.restore_state(snapshot)
    backend.step(0.075)

    assert backend.joint_position_rad() == pytest.approx(first_q, abs=1e-12)
    assert backend.joint_velocity_rad_s() == pytest.approx(first_v, abs=1e-12)


def test_mujoco_execution_drives_existing_vertical_canvas_paint_process() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=64))
    sim.plant = MujocoJointPlant()
    sim.reset_pose()
    sim.set_target(ik_pose_for_canvas_point(0.0, 0.0, sim.canvas.distance + 0.2))
    sim.paint_enabled = True
    sim.intended_contact_pressure = 0.5

    for _ in range(360):
        sim.step(1.0 / 240.0)

    assert sim.plant.backend.exact_brush_canvas_contact()
    assert sim.contact.on_canvas
    assert sim.contact.force > 0.0
    assert sim.contact.pressure > 0.0
    assert abs(float(sim.contact.brush_world[0])) < 0.2
    assert abs(float(sim.contact.brush_world[2])) < 0.6
    assert sim.canvas.material_coverage() > 0.0
    assert float(sim.canvas.thickness.max()) > 0.0


def test_mujoco_actual_execution_keeps_native_counterfactual_approximation() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=32))
    sim.plant = MujocoJointPlant()
    sim.reset_pose()

    forecast_copy = copy.deepcopy(sim)

    assert isinstance(sim.plant, MujocoJointPlant)
    assert isinstance(forecast_copy.plant, JointPlant)
    assert sim.plant.counterfactual_backend_id == "native-abstract-v0 approximation"
    assert forecast_copy.actual_pose == sim.actual_pose
    assert np.array_equal(forecast_copy.canvas.thickness, sim.canvas.thickness)


def test_web_runtime_can_select_direct_mujoco_state() -> None:
    runtime = WebSimRuntime(
        canvas_size=32,
        plant_backend="mujoco",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    runtime.agent_enabled = False
    runtime.agent_driver.enabled = False
    for _ in range(8):
        runtime._advance_one_step(1.0 / 240.0)

    state = runtime.state()
    runtime._record_telemetry(force=True)
    telemetry = runtime.telemetry_log.recent(1)[0]

    assert state["plantBackend"] == "mujoco"
    assert state["counterfactualPlantBackend"] == "native-abstract-v0 approximation"
    assert state["robot"]["mode"] == "mujoco_direct"
    assert state["robot"]["backendId"] == "mujoco-robstride-direct-v3"
    assert len(state["robot"]["jointPositionDeg"]) == 4
    assert len(state["robot"]["tipM"]) == 3
    assert np.isfinite(state["robot"]["tipM"]).all()
    for name in ("yaw", "pitch", "roll", "elbow"):
        assert telemetry[f"position_{name}_deg"] == pytest.approx(
            state["robot"]["jointPositionDeg"][name]
        )
        assert telemetry[f"target_{name}_deg"] == pytest.approx(
            runtime.sim.plant.telemetry_joint_target_deg()[name]
        )
