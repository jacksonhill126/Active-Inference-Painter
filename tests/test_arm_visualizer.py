import numpy as np

from active_painter.arm_sim import ArmKinematics, VerticalCanvas
from active_painter.arm_control import scripted_contact_pressure, scripted_pose
from active_painter.config import PainterConfig


def test_scripted_pose_sweeps_across_canvas_area() -> None:
    kin = ArmKinematics()
    tips = np.asarray([kin.tip(scripted_pose(t)) for t in np.linspace(0.0, 36.0, 80)])
    assert tips[:, 0].max() - tips[:, 0].min() > 9.0
    assert tips[:, 2].max() - tips[:, 2].min() > 6.0
    assert 16.3 < tips[:, 1].mean() < 17.6


def test_scripted_pose_has_lift_and_pressure_variation() -> None:
    kin = ArmKinematics()
    canvas = VerticalCanvas(PainterConfig(canvas_size=32))
    pressures = np.asarray(
        [
            canvas.contact_from_tip(kin.tip(scripted_pose(t)), scripted_contact_pressure(t)).pressure
            for t in np.linspace(0.0, 18.0, 160)
        ]
    )
    assert pressures.min() == 0.0
    assert pressures.max() > 0.25
    assert pressures.std() > 0.08
