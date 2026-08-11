from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from active_painter.camera_observation import (  # noqa: E402
    FOVEA_CANVAS_PRODUCT,
    GLOBAL_CANVAS_PRODUCT,
    NATIVE_SENSOR_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
    CameraObservationProcess,
    FoveaRequest,
)
from active_painter.camera_geometry import project_canvas_uv  # noqa: E402


MODEL_PATH = Path(__file__).parents[1] / "models" / "active_inference_painter.xml"
TEST_NATIVE_RESOLUTIONS = {
    "canvas_right_oblique": (640, 480),
    "canvas_left_oblique": (640, 480),
}


def _process(seed: int) -> CameraObservationProcess:
    return CameraObservationProcess(
        MODEL_PATH,
        random_seed=seed,
        native_resolution_overrides=TEST_NATIVE_RESOLUTIONS,
    )


def _keyframe_qpos(process: CameraObservationProcess, name: str) -> np.ndarray:
    import mujoco

    key_id = mujoco.mj_name2id(process.model, mujoco.mjtObj.mjOBJ_KEY, name)
    return np.asarray(process.model.key_qpos[key_id], dtype=np.float64).copy()


def _request(
    request_id: str = "look-1",
    *,
    camera_name: str = "canvas_right_oblique",
    center: tuple[float, float] = (0.5, 0.5),
    span: tuple[float, float] = (0.2, 0.2),
) -> FoveaRequest:
    return FoveaRequest(
        request_id=request_id,
        camera_name=camera_name,
        requested_time_s=0.0,
        expires_time_s=0.1,
        center_canvas_uv=center,
        span_canvas_uv=span,
        selection_basis="sensor_posterior",
        selection_revision="test-observation-posterior-1",
    )


def test_camera_product_contract_is_read_only_and_non_oracle() -> None:
    image = np.full((8, 9), 0.5, dtype=np.float64)
    frame = CameraFrame(
        camera_name="left",
        role="contact_tracking",
        sequence=3,
        product_kind=GLOBAL_CANVAS_PRODUCT,
        product_id="global",
        capture_time_s=1.0,
        available_time_s=1.02,
        calibration_revision="test-rig",
        observation_model="test-observation",
        registration="canvas_plane_homography",
        sampling_kind="native_to_canvas_homography",
        source_resolution_px=(90, 80),
        declared_acquisition_resolution_px=(3840, 2160),
        grayscale=image,
        calibration_validity=np.ones_like(image, dtype=np.bool_),
    )

    assert frame.grayscale.dtype == np.float32
    assert frame.grayscale.shape == (8, 9)
    assert not frame.grayscale.flags.writeable
    assert not frame.calibration_validity.flags.writeable
    forbidden = {
        "segmentation",
        "exact_visibility",
        "exact_contact",
        "material",
        "canvas",
        "pose",
        "qpos",
    }
    for record_type in (CameraFrame, CameraObservationBundle, FoveaRequest):
        assert forbidden.isdisjoint(field.name for field in fields(record_type))


def test_fovea_request_is_explicit_bounded_observation_space_metadata() -> None:
    request = _request(center=(0.0, 1.0), span=(0.25, 0.4))

    assert request.center_canvas_uv == (0.0, 1.0)
    assert request.selection_basis == "sensor_posterior"
    assert request.selection_revision == "test-observation-posterior-1"
    with pytest.raises(ValueError, match="center"):
        _request(center=(-0.01, 0.5))
    with pytest.raises(ValueError, match="span"):
        _request(span=(0.0, 0.2))
    with pytest.raises(ValueError, match="selection_basis"):
        FoveaRequest(
            request_id="oracle",
            camera_name="canvas_right_oblique",
            requested_time_s=0.0,
            expires_time_s=0.1,
            center_canvas_uv=(0.5, 0.5),
            span_canvas_uv=(0.2, 0.2),
            selection_basis="exact_simulator_pose",
            selection_revision="forbidden",
        )


def test_native_global_and_requested_fovea_share_one_capture() -> None:
    process = _process(7)
    try:
        qpos = _keyframe_qpos(process, "safe_home")
        products = process.render_products_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=qpos,
            canvas_grayscale=np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64),
            fovea_requests=(_request(),),
        )
    finally:
        process.close()

    by_kind = {product.product_kind: product for product in products}
    assert set(by_kind) == {
        NATIVE_SENSOR_PRODUCT,
        GLOBAL_CANVAS_PRODUCT,
        FOVEA_CANVAS_PRODUCT,
    }
    native = by_kind[NATIVE_SENSOR_PRODUCT]
    global_view = by_kind[GLOBAL_CANVAS_PRODUCT]
    fovea = by_kind[FOVEA_CANVAS_PRODUCT]
    assert native.grayscale.shape == (480, 640)
    assert native.source_resolution_px == (640, 480)
    assert native.declared_acquisition_resolution_px == (1456, 1088)
    assert global_view.grayscale.shape == (512, 512)
    assert fovea.grayscale.shape == (256, 256)
    assert fovea.fovea_request_id == "look-1"
    assert fovea.center_canvas_uv == (0.5, 0.5)
    assert fovea.span_canvas_uv == (0.2, 0.2)
    assert fovea.sampling_kind == "native_to_requested_canvas_uv"
    assert {product.sequence for product in products} == {0}
    assert {product.capture_time_s for product in products} == {0.0}
    assert {product.source_resolution_px for product in products} == {(640, 480)}


def test_no_fovea_is_created_without_an_explicit_request() -> None:
    process = _process(8)
    try:
        products = process.render_products_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=_keyframe_qpos(process, "safe_home"),
            canvas_grayscale=np.ones((32, 32), dtype=np.float64),
        )
    finally:
        process.close()

    assert [product.product_kind for product in products] == [
        NATIVE_SENSOR_PRODUCT,
        GLOBAL_CANVAS_PRODUCT,
    ]


def test_fovea_samples_native_frame_directly_not_global_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(18)
    width, height = TEST_NATIVE_RESOLUTIONS["canvas_right_oblique"]
    native = np.broadcast_to(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        (height, width),
    ).copy()
    monkeypatch.setattr(
        process,
        "_render_native",
        lambda camera, canvas, rng: native.copy(),
    )
    request = _request(center=(0.55, 0.45), span=(0.18, 0.14))
    try:
        products = process.render_products_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=_keyframe_qpos(process, "safe_home"),
            canvas_grayscale=np.ones((32, 32), dtype=np.float64),
            fovea_requests=(request,),
        )
        camera = process.rig.camera("canvas_right_oblique")
        fovea = next(
            product
            for product in products
            if product.product_kind == FOVEA_CANVAS_PRODUCT
        )
        fovea_width, fovea_height = camera.foveal_resolution_px or (0, 0)
        u = np.linspace(0.55 - 0.09, 0.55 + 0.09, fovea_width)
        v = np.linspace(0.45 - 0.07, 0.45 + 0.07, fovea_height)
        uu, vv = np.meshgrid(u, v)
        pixels = project_canvas_uv(
            camera,
            process.rig.canvas,
            np.stack((uu, vv), axis=-1),
            (width, height),
        )
    finally:
        process.close()

    expected = pixels[..., 0] / (width - 1)
    assert fovea.grayscale[fovea.calibration_validity] == pytest.approx(
        expected[fovea.calibration_validity], abs=2e-6
    )


def test_edge_fovea_has_partial_validity_instead_of_oracle_fill() -> None:
    process = _process(9)
    try:
        products = process.render_products_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=_keyframe_qpos(process, "safe_home"),
            canvas_grayscale=np.ones((32, 32), dtype=np.float64),
            fovea_requests=(
                _request(center=(0.0, 0.0), span=(0.3, 0.3)),
            ),
        )
    finally:
        process.close()

    fovea = next(
        product
        for product in products
        if product.product_kind == FOVEA_CANVAS_PRODUCT
    )
    assert 0 < np.count_nonzero(fovea.calibration_validity) < 256 * 256
    assert np.all(fovea.grayscale[~fovea.calibration_validity] == 0.0)


def test_rendered_canvas_composite_changes_paint_but_keeps_occluders() -> None:
    blank_process = _process(12)
    painted_process = _process(12)
    try:
        qpos = _keyframe_qpos(blank_process, "safe_home")
        blank = np.ones((64, 64), dtype=np.float64)
        black = np.zeros((64, 64), dtype=np.float64)
        blank_frame = blank_process.render_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=qpos,
            canvas_grayscale=blank,
        )
        painted_frame = painted_process.render_immediate(
            "canvas_right_oblique",
            monotonic_time_s=0.0,
            qpos=qpos,
            canvas_grayscale=black,
        )
    finally:
        blank_process.close()
        painted_process.close()

    assert blank_frame.product_kind == GLOBAL_CANVAS_PRODUCT
    difference = blank_frame.grayscale - painted_frame.grayscale
    assert float(np.quantile(difference, 0.9)) > 0.45
    # Identical native noise seeds leave non-canvas arm/background pixels
    # unchanged after both products are rectified.
    assert int(np.count_nonzero(np.abs(difference) < 1e-7)) > 100


def test_multirate_delivery_groups_native_and_derived_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(2)
    monkeypatch.setattr(
        process,
        "_render_native",
        lambda camera, canvas, rng: np.full(
            (
                TEST_NATIVE_RESOLUTIONS[camera.name][1],
                TEST_NATIVE_RESOLUTIONS[camera.name][0],
            ),
            0.5,
            dtype=np.float32,
        ),
    )
    qpos = _keyframe_qpos(process, "safe_home")
    canvas = np.ones((16, 16), dtype=np.float64)
    try:
        assert process.observe(
            0.0,
            qpos=qpos,
            canvas_grayscale=canvas,
            inspection_available=False,
            fovea_requests=(_request(),),
        ).frames == ()

        first_delivery = process.observe(
            1.0 / 60.0,
            qpos=qpos,
            canvas_grayscale=canvas,
            inspection_available=False,
        )
        assert {
            (frame.camera_name, frame.product_kind)
            for frame in first_delivery.frames
        } == {
            ("canvas_right_oblique", NATIVE_SENSOR_PRODUCT),
            ("canvas_right_oblique", GLOBAL_CANVAS_PRODUCT),
            ("canvas_right_oblique", FOVEA_CANVAS_PRODUCT),
            ("canvas_left_oblique", NATIVE_SENSOR_PRODUCT),
            ("canvas_left_oblique", GLOBAL_CANVAS_PRODUCT),
        }
    finally:
        process.close()


def test_provisional_specular_residual_is_bounded_and_visible_on_black() -> None:
    process = _process(4)
    try:
        appearance = process._surface_appearance(
            np.zeros(3, dtype=np.float64),
            np.asarray((0.5, 0.5, 0.8), dtype=np.float64),
        )
    finally:
        process.close()

    assert appearance[:2] == pytest.approx((0.0, 0.0))
    assert 0.0 < appearance[2] <= process.rig.provisional_specular_strength
