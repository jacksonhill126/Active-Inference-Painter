import numpy as np
import pytest

from active_painter.arm_sim import (
    JOINT_NAMES,
    ArmKinematics,
    ArmPainterSim,
    ArmPose,
    VerticalCanvas,
    safe_home_pose,
)
from active_painter.config import PainterConfig


def test_native_abstract_v0_joint_order_ranges_and_home_pose() -> None:
    assert JOINT_NAMES == ("yaw", "pitch", "roll", "elbow")
    assert ArmPose(-999.0, -999.0, -999.0, -999.0).clipped() == ArmPose(
        yaw=-90.0,
        pitch=-90.0,
        roll=-180.0,
        elbow=0.0,
    )
    assert ArmPose(999.0, 999.0, 999.0, 999.0).clipped() == ArmPose(
        yaw=90.0,
        pitch=90.0,
        roll=180.0,
        elbow=150.0,
    )
    assert safe_home_pose() == ArmPose(yaw=0.0, pitch=-50.0, roll=0.0, elbow=100.0)


def test_native_abstract_v0_link_and_canvas_geometry() -> None:
    kinematics = ArmKinematics()
    canvas = VerticalCanvas(PainterConfig(canvas_size=33))

    assert kinematics.upper_arm == pytest.approx(13.0)
    assert kinematics.lower_arm == pytest.approx(13.0)
    assert canvas.width == pytest.approx(20.0)
    assert canvas.height == pytest.approx(20.0)
    assert canvas.distance == pytest.approx(17.0)
    assert canvas.bushing_travel == pytest.approx(0.5)
    assert canvas.contact_stiffness == pytest.approx(55.0)
    assert canvas.world_to_pixel(0.0, 0.0) == pytest.approx((16.0, 16.0))


def test_native_abstract_v0_kinematic_point_order_and_link_lengths() -> None:
    kinematics = ArmKinematics()
    points = kinematics.joint_points(ArmPose())

    assert points.shape == (3, 3)
    assert np.array_equal(points[0], np.zeros(3))
    assert np.linalg.norm(points[1] - points[0]) == pytest.approx(13.0)
    assert np.linalg.norm(points[2] - points[1]) == pytest.approx(13.0)


def test_reset_pose_preserves_canvas_but_realigns_arm_state() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=32))
    point = np.asarray([0.0, sim.canvas.distance, 0.0])
    sim.canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 30.0)
    material_before = sim.canvas.thickness.copy()
    sim.actual_pose = ArmPose(yaw=20.0, pitch=-20.0, roll=10.0, elbow=70.0)

    sim.reset_pose()

    assert sim.actual_pose == safe_home_pose()
    assert sim.target_pose == safe_home_pose()
    assert np.array_equal(sim.canvas.thickness, material_before)
    assert not sim.paint_enabled
