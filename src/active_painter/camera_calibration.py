"""Physical fixed-camera intrinsic calibration utilities.

This module deliberately keeps calibration truth below the painting-policy
boundary.  It measures the image formation process; it does not expose board
poses, exact camera poses, or simulator geometry to the painting model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CALIBRATION_SCHEMA_VERSION = "camera-intrinsics-v1"
DEFAULT_INNER_COLUMNS = 11
DEFAULT_INNER_ROWS = 8
DEFAULT_SQUARE_SIZE_MM = 28.0
DEFAULT_MARGIN_MM = 20.0


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "Camera calibration requires the optional calibration dependency; "
            "install with `python -m pip install -e .[calibration]`."
        ) from exc
    return cv2


@dataclass(frozen=True, slots=True)
class CheckerboardSpec:
    """Metric checkerboard definition using OpenCV inner-corner convention."""

    inner_columns: int = DEFAULT_INNER_COLUMNS
    inner_rows: int = DEFAULT_INNER_ROWS
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM
    margin_mm: float = DEFAULT_MARGIN_MM

    def __post_init__(self) -> None:
        if self.inner_columns < 3 or self.inner_rows < 3:
            raise ValueError("checkerboard needs at least 3 x 3 inner corners")
        if self.square_size_mm <= 0.0 or not np.isfinite(self.square_size_mm):
            raise ValueError("square_size_mm must be finite and positive")
        if self.margin_mm < 0.0 or not np.isfinite(self.margin_mm):
            raise ValueError("margin_mm must be finite and non-negative")

    @property
    def square_columns(self) -> int:
        return self.inner_columns + 1

    @property
    def square_rows(self) -> int:
        return self.inner_rows + 1

    @property
    def page_size_mm(self) -> tuple[float, float]:
        return (
            self.square_columns * self.square_size_mm + 2.0 * self.margin_mm,
            self.square_rows * self.square_size_mm + 2.0 * self.margin_mm,
        )

    def object_points_mm(self) -> np.ndarray:
        points = np.zeros(
            (self.inner_columns * self.inner_rows, 3), dtype=np.float32
        )
        points[:, :2] = np.mgrid[
            0 : self.inner_columns, 0 : self.inner_rows
        ].T.reshape(-1, 2)
        points[:, :2] *= self.square_size_mm
        return points

    def to_svg(self) -> str:
        width_mm, height_mm = self.page_size_mm
        rectangles = [
            (
                f'  <rect x="{self.margin_mm + column * self.square_size_mm:g}" '
                f'y="{self.margin_mm + row * self.square_size_mm:g}" '
                f'width="{self.square_size_mm:g}" '
                f'height="{self.square_size_mm:g}" fill="#000"/>'
            )
            for row in range(self.square_rows)
            for column in range(self.square_columns)
            if (row + column) % 2 == 0
        ]
        metadata = (
            f"{self.inner_columns}x{self.inner_rows} inner corners; "
            f"{self.square_size_mm:g} mm squares"
        )
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width_mm:g}mm" height="{height_mm:g}mm" '
                f'viewBox="0 0 {width_mm:g} {height_mm:g}">'
            ),
            f"  <title>{metadata}</title>",
            f'  <rect width="{width_mm:g}" height="{height_mm:g}" fill="#fff"/>',
            *rectangles,
            (
                f'  <rect x="{self.margin_mm:g}" y="{self.margin_mm:g}" '
                f'width="{self.square_columns * self.square_size_mm:g}" '
                f'height="{self.square_rows * self.square_size_mm:g}" '
                f'fill="none" stroke="#000" stroke-width="0.2"/>'
            ),
            "</svg>",
            "",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CalibrationQualitySpec:
    minimum_used_views: int = 15
    maximum_rms_error_px: float = 0.8
    maximum_single_view_error_px: float = 1.5
    minimum_corner_coverage_fraction: float = 0.70
    minimum_center_span_fraction: float = 0.35
    minimum_tilt_span_deg: float = 15.0


@dataclass(frozen=True, slots=True)
class CameraCalibrationResult:
    schema_version: str
    camera_name: str
    created_utc: str
    image_size_px: tuple[int, int]
    checkerboard: dict[str, float | int]
    used_images: tuple[str, ...]
    rejected_images: tuple[dict[str, str], ...]
    camera_matrix_px: tuple[tuple[float, float, float], ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    rms_reprojection_error_px: float
    mean_per_view_error_px: float
    maximum_per_view_error_px: float
    per_view_error_px: tuple[float, ...]
    field_of_view_deg: tuple[float, float]
    corner_coverage_fraction: tuple[float, float]
    view_center_span_fraction: tuple[float, float]
    board_tilt_range_deg: tuple[float, float]
    quality_gates: dict[str, bool]
    status: str
    approximations: tuple[str, ...]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _expand_image_inputs(inputs: Sequence[str]) -> tuple[Path, ...]:
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths: set[Path] = set()
    for value in inputs:
        candidate = Path(value)
        if candidate.is_dir():
            paths.update(
                path.resolve()
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in supported
            )
            continue
        matches = tuple(Path(match) for match in glob.glob(value))
        if matches:
            paths.update(
                match.resolve()
                for match in matches
                if match.is_file() and match.suffix.lower() in supported
            )
        elif candidate.is_file() and candidate.suffix.lower() in supported:
            paths.add(candidate.resolve())
    return tuple(sorted(paths))


def _per_view_reprojection_errors(
    cv2: Any,
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    rotation_vectors: Sequence[np.ndarray],
    translation_vectors: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, ...]:
    errors: list[float] = []
    for object_view, image_view, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_view,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )
        residual = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return tuple(errors)


def _coverage_metrics(
    image_points: Sequence[np.ndarray], image_size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    width, height = image_size
    all_corners = np.concatenate(
        [points.reshape(-1, 2) for points in image_points], axis=0
    )
    corner_span = np.ptp(all_corners, axis=0) / np.asarray((width, height))
    centers = np.asarray(
        [points.reshape(-1, 2).mean(axis=0) for points in image_points]
    )
    center_span = np.ptp(centers, axis=0) / np.asarray((width, height))
    return (
        (float(corner_span[0]), float(corner_span[1])),
        (float(center_span[0]), float(center_span[1])),
    )


def _tilt_range_deg(cv2: Any, rotations: Sequence[np.ndarray]) -> tuple[float, float]:
    tilts: list[float] = []
    for rotation_vector in rotations:
        rotation, _ = cv2.Rodrigues(rotation_vector)
        normal = rotation[:, 2]
        cosine = float(np.clip(abs(normal[2]), 0.0, 1.0))
        tilts.append(float(np.degrees(np.arccos(cosine))))
    return (min(tilts), max(tilts))


def calibrate_camera_images(
    *,
    camera_name: str,
    image_inputs: Sequence[str],
    checkerboard: CheckerboardSpec = CheckerboardSpec(),
    quality: CalibrationQualitySpec = CalibrationQualitySpec(),
) -> CameraCalibrationResult:
    """Detect a checkerboard and solve Brown-Conrady intrinsics."""

    cv2 = _require_cv2()
    paths = _expand_image_inputs(image_inputs)
    if not paths:
        raise ValueError("no supported calibration images matched the input")

    object_template = checkerboard.object_points_mm()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    rejected_images: list[dict[str, str]] = []
    image_size: tuple[int, int] | None = None
    pattern_size = (checkerboard.inner_columns, checkerboard.inner_rows)
    flags = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected_images.append({"path": str(path), "reason": "decode_failed"})
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected_images.append(
                {"path": str(path), "reason": "resolution_mismatch"}
            )
            continue
        detected, corners = cv2.findChessboardCornersSB(
            image, pattern_size, flags=flags
        )
        if not detected or corners is None:
            rejected_images.append(
                {"path": str(path), "reason": "checkerboard_not_detected"}
            )
            continue
        object_points.append(object_template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        used_images.append(str(path))

    if image_size is None:
        raise ValueError("none of the supplied calibration images could be decoded")
    if len(used_images) < 3:
        raise ValueError(
            "at least three detected checkerboard views are required to solve intrinsics"
        )

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_view = _per_view_reprojection_errors(
        cv2,
        object_points,
        image_points,
        rotations,
        translations,
        camera_matrix,
        distortion,
    )
    corner_coverage, center_span = _coverage_metrics(image_points, image_size)
    tilt_range = _tilt_range_deg(cv2, rotations)
    width, height = image_size
    fov = (
        float(np.degrees(2.0 * np.arctan(width / (2.0 * camera_matrix[0, 0])))),
        float(np.degrees(2.0 * np.arctan(height / (2.0 * camera_matrix[1, 1])))),
    )
    quality_gates = {
        "minimum_used_views": len(used_images) >= quality.minimum_used_views,
        "rms_reprojection_error": float(rms) <= quality.maximum_rms_error_px,
        "maximum_single_view_error": max(per_view)
        <= quality.maximum_single_view_error_px,
        "horizontal_corner_coverage": corner_coverage[0]
        >= quality.minimum_corner_coverage_fraction,
        "vertical_corner_coverage": corner_coverage[1]
        >= quality.minimum_corner_coverage_fraction,
        "horizontal_center_span": center_span[0]
        >= quality.minimum_center_span_fraction,
        "vertical_center_span": center_span[1]
        >= quality.minimum_center_span_fraction,
        "board_tilt_span": (tilt_range[1] - tilt_range[0])
        >= quality.minimum_tilt_span_deg,
    }
    coefficients = tuple(float(value) for value in distortion.reshape(-1)[:5])
    return CameraCalibrationResult(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        camera_name=camera_name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        image_size_px=image_size,
        checkerboard={
            "inner_columns": checkerboard.inner_columns,
            "inner_rows": checkerboard.inner_rows,
            "square_size_mm": checkerboard.square_size_mm,
        },
        used_images=tuple(used_images),
        rejected_images=tuple(rejected_images),
        camera_matrix_px=tuple(
            tuple(float(value) for value in row) for row in camera_matrix
        ),
        distortion_model="opencv_brown_conrady_5_v1",
        distortion_coefficients=coefficients,
        rms_reprojection_error_px=float(rms),
        mean_per_view_error_px=float(np.mean(per_view)),
        maximum_per_view_error_px=max(per_view),
        per_view_error_px=per_view,
        field_of_view_deg=fov,
        corner_coverage_fraction=corner_coverage,
        view_center_span_fraction=center_span,
        board_tilt_range_deg=tilt_range,
        quality_gates=quality_gates,
        status=("accepted" if all(quality_gates.values()) else "needs_more_data"),
        approximations=(
            "Brown-Conrady five-coefficient lens model.",
            "Calibration is valid only for the captured resolution, crop mode, focus, and stabilization settings.",
            "Reprojection error measures checkerboard fit, not rolling-shutter or photometric accuracy.",
        ),
    )


def _checkerboard_from_args(args: argparse.Namespace) -> CheckerboardSpec:
    return CheckerboardSpec(
        inner_columns=args.columns,
        inner_rows=args.rows,
        square_size_mm=args.square_mm,
        margin_mm=getattr(args, "margin_mm", DEFAULT_MARGIN_MM),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and solve fixed-camera checkerboard calibration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    target = subparsers.add_parser("generate-target")
    target.add_argument("--output", type=Path, required=True)
    target.add_argument("--columns", type=int, default=DEFAULT_INNER_COLUMNS)
    target.add_argument("--rows", type=int, default=DEFAULT_INNER_ROWS)
    target.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_SIZE_MM)
    target.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--camera-name", required=True)
    calibrate.add_argument("--images", nargs="+", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--columns", type=int, default=DEFAULT_INNER_COLUMNS)
    calibrate.add_argument("--rows", type=int, default=DEFAULT_INNER_ROWS)
    calibrate.add_argument(
        "--square-mm", type=float, default=DEFAULT_SQUARE_SIZE_MM
    )
    calibrate.add_argument("--minimum-views", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checkerboard = _checkerboard_from_args(args)
    if args.command == "generate-target":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(checkerboard.to_svg(), encoding="utf-8")
        width_mm, height_mm = checkerboard.page_size_mm
        print(
            f"Wrote {args.output} at {width_mm:g} x {height_mm:g} mm; "
            "print at 100% / actual size."
        )
        return 0

    result = calibrate_camera_images(
        camera_name=args.camera_name,
        image_inputs=args.images,
        checkerboard=checkerboard,
        quality=CalibrationQualitySpec(minimum_used_views=args.minimum_views),
    )
    result.write_json(args.output)
    print(
        json.dumps(
            {
                "camera_name": result.camera_name,
                "status": result.status,
                "used_views": len(result.used_images),
                "rejected_views": len(result.rejected_images),
                "rms_reprojection_error_px": result.rms_reprojection_error_px,
                "field_of_view_deg": result.field_of_view_deg,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result.status == "accepted" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
