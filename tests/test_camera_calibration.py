from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from active_painter.camera_calibration import (
    CalibrationQualitySpec,
    CheckerboardSpec,
    calibrate_camera_images,
)


def test_default_checkerboard_fits_a3_and_has_metric_svg() -> None:
    checkerboard = CheckerboardSpec()

    assert checkerboard.page_size_mm == pytest.approx((376.0, 292.0))
    assert checkerboard.object_points_mm().shape == (88, 3)
    svg = checkerboard.to_svg()
    assert 'width="376mm" height="292mm"' in svg
    assert "11x8 inner corners; 28 mm squares" in svg
    assert svg.count('fill="#000"/>') == 54


def _write_synthetic_calibration_views(
    output_dir: Path,
    checkerboard: CheckerboardSpec,
) -> tuple[Path, ...]:
    cv2 = pytest.importorskip("cv2")
    image_size = (960, 540)
    camera_matrix = np.asarray(
        ((820.0, 0.0, 480.0), (0.0, 815.0, 270.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    square_pixels = 72
    texture = np.full(
        (
            checkerboard.square_rows * square_pixels,
            checkerboard.square_columns * square_pixels,
        ),
        255,
        dtype=np.uint8,
    )
    for row in range(checkerboard.square_rows):
        for column in range(checkerboard.square_columns):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    texture,
                    (column * square_pixels, row * square_pixels),
                    ((column + 1) * square_pixels, (row + 1) * square_pixels),
                    0,
                    thickness=-1,
                )
    texture_corners = np.asarray(
        (
            (0.0, 0.0),
            (texture.shape[1] - 1.0, 0.0),
            (texture.shape[1] - 1.0, texture.shape[0] - 1.0),
            (0.0, texture.shape[0] - 1.0),
        ),
        dtype=np.float32,
    )
    square = checkerboard.square_size_mm
    outer_corners = np.asarray(
        (
            (-square, -square, 0.0),
            (checkerboard.inner_columns * square, -square, 0.0),
            (
                checkerboard.inner_columns * square,
                checkerboard.inner_rows * square,
                0.0,
            ),
            (-square, checkerboard.inner_rows * square, 0.0),
        ),
        dtype=np.float32,
    )
    views: list[Path] = []
    poses = (
        (-0.18, -0.16, -0.08, -120.0, -70.0, 730.0),
        (-0.12, 0.14, 0.08, 55.0, -75.0, 760.0),
        (0.14, -0.16, 0.12, -110.0, 45.0, 750.0),
        (0.18, 0.13, -0.10, 55.0, 45.0, 780.0),
        (-0.22, 0.04, 0.16, -35.0, -20.0, 690.0),
        (0.20, -0.03, -0.16, -30.0, -15.0, 720.0),
        (-0.08, -0.21, 0.05, 5.0, -80.0, 790.0),
        (0.08, 0.21, -0.05, 0.0, 50.0, 790.0),
        (-0.24, -0.08, 0.12, -105.0, -10.0, 800.0),
        (0.24, 0.08, -0.12, 45.0, -10.0, 800.0),
        (-0.14, 0.22, 0.03, -40.0, 30.0, 740.0),
        (0.14, -0.22, -0.03, -45.0, -55.0, 740.0),
        (-0.05, 0.08, 0.22, -55.0, -30.0, 680.0),
        (0.05, -0.08, -0.22, -45.0, -30.0, 700.0),
        (-0.17, 0.17, 0.10, -80.0, 20.0, 770.0),
        (0.17, -0.17, -0.10, 15.0, -55.0, 770.0),
    )
    for index, (rx, ry, rz, tx, ty, tz) in enumerate(poses):
        rotation_vector = np.asarray((rx, ry, rz), dtype=np.float64)
        translation_vector = np.asarray((tx, ty, tz), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            outer_corners,
            rotation_vector,
            translation_vector,
            camera_matrix,
            np.zeros(5),
        )
        homography = cv2.getPerspectiveTransform(
            texture_corners, projected.reshape(-1, 2).astype(np.float32)
        )
        image = cv2.warpPerspective(
            texture,
            homography,
            image_size,
            flags=cv2.INTER_NEAREST,
            borderValue=180,
        )
        path = output_dir / f"view_{index:02d}.png"
        assert cv2.imwrite(str(path), image)
        views.append(path)
    return tuple(views)


def test_synthetic_intrinsic_calibration_recovers_pinhole_camera(
    tmp_path: Path,
) -> None:
    checkerboard = CheckerboardSpec()
    views = _write_synthetic_calibration_views(tmp_path, checkerboard)

    result = calibrate_camera_images(
        camera_name="synthetic_camera",
        image_inputs=[str(path) for path in views],
        checkerboard=checkerboard,
        quality=CalibrationQualitySpec(
            minimum_used_views=10,
            maximum_rms_error_px=0.5,
            maximum_single_view_error_px=0.8,
            minimum_corner_coverage_fraction=0.60,
            minimum_center_span_fraction=0.20,
        ),
    )

    assert len(result.used_images) >= 10
    assert result.rms_reprojection_error_px < 0.5
    assert result.camera_matrix_px[0][0] == pytest.approx(820.0, rel=0.03)
    assert result.camera_matrix_px[1][1] == pytest.approx(815.0, rel=0.03)
    assert result.field_of_view_deg == pytest.approx(
        (60.686, 36.662), abs=1.0
    )
    assert max(abs(value) for value in result.distortion_coefficients) < 0.1
