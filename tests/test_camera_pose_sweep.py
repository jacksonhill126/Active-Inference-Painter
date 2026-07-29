from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from active_painter.camera_geometry import load_camera_rig
from active_painter.camera_pose_sweep import (
    evaluate_pose_sweep,
    generate_contact_poses,
)


MODEL_PATH = Path(__file__).parents[1] / "models" / "active_inference_painter.xml"


def test_contact_pose_grid_reaches_declared_canvas_targets() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rig = load_camera_rig(MODEL_PATH)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

    poses = generate_contact_poses(
        model,
        rig,
        grid_size=3,
        rolls_deg=(-32.0, 0.0, 32.0),
    )

    assert len(poses) == 27
    assert max(pose.solve_error_m for pose in poses) < 1e-6
    for pose in poses:
        data.qpos[:] = pose.qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        assert data.site_xpos[tip_id] == pytest.approx(
            np.asarray(pose.tip_position_m),
            abs=1e-6,
        )


def test_representative_contact_grid_has_continuous_tip_view() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    rig = load_camera_rig(MODEL_PATH)
    poses = generate_contact_poses(
        model,
        rig,
        grid_size=2,
        rolls_deg=(0.0,),
    )

    metrics = evaluate_pose_sweep(
        model,
        rig,
        poses,
        ray_grid_size=3,
    )
    contact_cameras = {
        camera.name
        for camera in rig.cameras
        if camera.role == "contact_tracking"
        and camera.availability == "continuous"
    }
    for pose in poses:
        selected = [
            metric
            for metric in metrics
            if metric.pose_label == pose.label
            and metric.camera_name in contact_cameras
        ]
        assert selected
        assert any(metric.tip_visible for metric in selected)
