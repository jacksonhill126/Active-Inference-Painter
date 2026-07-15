import numpy as np

from active_painter.action_encoding import encoded_action_vector
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.policies import MotorPrimitiveLatent
from active_painter.spatial_state import rasterize_stroke_action


def test_summary_action_encoding_adds_motor_realization_condition() -> None:
    cfg = PainterConfig()
    action = StrokeAction(0.1, 0.2, 0.7, 0.8, 0.08, 0.5, 1.0)

    cartesian = encoded_action_vector(action, cfg)
    elbow = encoded_action_vector(action, cfg, MotorPrimitiveLatent("elbow_pivot"))
    roll_positive = encoded_action_vector(action, cfg, MotorPrimitiveLatent("upper_arm_roll_positive"))
    roll_negative = encoded_action_vector(action, cfg, MotorPrimitiveLatent("upper_arm_roll_negative"))

    assert cartesian.shape == (cfg.action_dim,)
    assert elbow.shape == (cfg.action_dim,)
    assert np.allclose(cartesian[:7], elbow[:7])
    assert np.allclose(cartesian[7:], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(elbow[7:], [0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(roll_positive[7:], [0.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(roll_negative[7:], [0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(encoded_action_vector(StrokeAction.stop_action(), cfg), 0.0)
    assert cfg.action_dim == 7 + len(cfg.motor_realization_kinds)
    assert cfg.spatial_action_channels == 6 + len(cfg.motor_realization_kinds)


def test_spatial_action_raster_adds_constant_motor_realization_fields() -> None:
    cfg = PainterConfig(spatial_grid_size=8)
    action = StrokeAction(0.1, 0.2, 0.7, 0.8, 0.08, 0.5, 1.0)

    raster = rasterize_stroke_action(
        action,
        cfg.spatial_grid_size,
        motor_primitive=MotorPrimitiveLatent("joint_spline"),
        config=cfg,
    )

    assert raster.shape == (cfg.spatial_action_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    assert np.allclose(raster[6], 0.0)
    assert np.allclose(raster[7], 1.0)
    assert np.allclose(raster[8], 0.0)
    assert np.allclose(raster[9], 0.0)
    assert np.allclose(raster[10], 0.0)
    assert np.allclose(
        rasterize_stroke_action(StrokeAction.stop_action(), cfg.spatial_grid_size, config=cfg),
        0.0,
    )
