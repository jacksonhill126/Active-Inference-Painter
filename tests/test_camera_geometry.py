from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from active_painter.camera_geometry import (
    camera_pixels_to_canvas_uv,
    canvas_axial_depth_range_m,
    canvas_frustum_mask,
    canvas_incidence_angle_deg,
    canvas_uv_to_camera_homography,
    load_camera_rig,
    project_canvas_uv,
    project_world_points,
    rectify_canvas_image,
)


MODEL_PATH = Path(__file__).parents[1] / "models" / "active_inference_painter.xml"


def _pil_perspective_coefficients(matrix: np.ndarray) -> tuple[float, ...]:
    normalized = matrix / matrix[2, 2]
    return (
        normalized[0, 0],
        normalized[0, 1],
        normalized[0, 2],
        normalized[1, 0],
        normalized[1, 1],
        normalized[1, 2],
        normalized[2, 0],
        normalized[2, 1],
    )


def test_mjcf_camera_rig_has_versioned_roles_and_canonical_projection() -> None:
    rig = load_camera_rig(MODEL_PATH)

    assert rig.version == "provisional-multiview-v2"
    assert rig.geometry_model == "ideal_pinhole_without_distortion"
    assert rig.calibration_status == "provisional_simulation_geometry"
    assert rig.normalization == "role_dependent_v1"
    assert rig.focus_model == "fixed_manual_pending_lens_selection"
    assert rig.observation_encoding == (
        "linear_grayscale_float32_normalized_0_1"
    )
    assert rig.shutter_model == "global_shutter"
    assert [camera.name for camera in rig.cameras] == [
        "canvas_right_oblique",
        "canvas_left_oblique",
        "canvas_inspection_deployed",
        "brush_standoff_overhead",
    ]
    assert [camera.role for camera in rig.cameras] == [
        "contact_tracking",
        "contact_tracking",
        "canvas_inspection",
        "brush_standoff",
    ]
    assert [camera.availability for camera in rig.cameras] == [
        "continuous",
        "continuous",
        "park_only",
        "continuous",
    ]
    assert {camera.channels for camera in rig.cameras} == {"grayscale"}
    assert [camera.registration for camera in rig.cameras] == [
        "canvas_plane_homography",
        "canvas_plane_homography",
        "canvas_plane_homography",
        "canvas_edge_profile",
    ]

    expected_incidence = (
        31.7548481366,
        32.5724697676,
        0.0,
        75.5092986784,
    )
    expected_depth = (
        (1.3495300677, 1.7143020165),
        (1.3567525452, 1.7346980916),
        (0.3826, 0.3826),
        (0.646, 1.154),
    )
    canvas_uv = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
    )
    for index, camera in enumerate(rig.cameras):
        rotation = camera.rotation
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-11)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-11)
        assert canvas_incidence_angle_deg(camera, rig.canvas) == pytest.approx(
            expected_incidence[index]
        )
        assert canvas_axial_depth_range_m(camera, rig.canvas) == pytest.approx(
            expected_depth[index]
        )

        if camera.registration == "canvas_plane_homography":
            assert camera.resolution_px == (1024, 1024)
            assert camera.model_input_resolution_px == (512, 512)
            pixels = project_canvas_uv(camera, rig.canvas, canvas_uv)
            reconstructed = camera_pixels_to_canvas_uv(
                camera, rig.canvas, pixels
            )
            assert reconstructed == pytest.approx(canvas_uv, abs=1e-12)
            assert pixels[-1] == pytest.approx((512.0, 512.0), abs=1e-8)
            assert canvas_frustum_mask(camera, rig.canvas, (64, 64)).all()
        else:
            assert camera.resolution_px == (640, 480)
            assert camera.model_input_resolution_px == (640, 480)


def test_rectification_correlates_all_camera_views_in_canvas_uv() -> None:
    rig = load_camera_rig(MODEL_PATH)
    canonical_size = (96, 96)
    camera_size = (256, 256)
    height, width = canonical_size[1], canonical_size[0]
    yy, xx = np.mgrid[:height, :width]
    canonical_array = np.stack(
        (
            xx * 255.0 / (width - 1),
            yy * 255.0 / (height - 1),
            (xx + yy) * 127.5 / (width - 1),
        ),
        axis=-1,
    ).astype(np.uint8)
    canonical = Image.fromarray(canonical_array, mode="RGB")
    canonical_pixels_to_uv = np.diag(
        (1.0 / (width - 1), 1.0 / (height - 1), 1.0)
    )

    for camera in (
        camera
        for camera in rig.cameras
        if camera.registration == "canvas_plane_homography"
    ):
        canonical_to_camera = canvas_uv_to_camera_homography(
            camera,
            rig.canvas,
            camera_size,
        ) @ canonical_pixels_to_uv
        camera_to_canonical = np.linalg.inv(canonical_to_camera)
        simulated_camera = canonical.transform(
            camera_size,
            Image.Transform.PERSPECTIVE,
            _pil_perspective_coefficients(camera_to_canonical),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        rectified = rectify_canvas_image(
            simulated_camera,
            camera,
            rig.canvas,
            canonical_size,
        )

        error = np.abs(
            np.asarray(rectified, dtype=np.float64)
            - canonical_array.astype(np.float64)
        )
        interior = error[3:-3, 3:-3]
        assert float(np.mean(interior)) < 1.1
        assert float(np.max(interior)) <= 2.0


def test_standoff_camera_observes_normal_separation_without_canvas_warp() -> None:
    rig = load_camera_rig(MODEL_PATH)
    camera = rig.camera("brush_standoff_overhead")

    with pytest.raises(ValueError, match="not canvas-plane homography"):
        canvas_uv_to_camera_homography(camera, rig.canvas)

    points = np.stack(
        (
            rig.canvas.center,
            rig.canvas.center + np.asarray((0.0, -0.001, 0.0)),
        )
    )
    pixels, depth = project_world_points(
        camera,
        points,
        camera.model_input_resolution_px,
    )
    assert depth == pytest.approx((0.9, 0.9))
    assert pixels[0] == pytest.approx((320.0, 90.2543801))
    assert pixels[1, 1] - pixels[0, 1] == pytest.approx(0.64379028)


def test_camera_clear_park_has_unoccluded_canvas_rays_for_all_sensor_views() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rig = load_camera_rig(MODEL_PATH)
    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "camera_clear_park"
    )
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    canvas_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "canvas_surface"
    )
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    assert data.site_xpos[tip_id] == pytest.approx(
        (0.075, 0.4064, 0.096), abs=2e-9
    )
    assert data.ncon == 0
    geom_groups = np.ones(6, dtype=np.uint8)
    for camera in rig.cameras:
        for v in np.linspace(0.001, 0.999, 9):
            for u in np.linspace(0.001, 0.999, 9):
                canvas_point = rig.canvas.world_from_uv((u, v))
                direction = canvas_point - camera.position
                direction /= np.linalg.norm(direction)
                hit_geom = np.full(1, -1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    model,
                    data,
                    camera.position,
                    direction,
                    geom_groups,
                    True,
                    -1,
                    hit_geom,
                )
                assert distance > 0.0
                assert int(hit_geom[0]) == canvas_geom_id
