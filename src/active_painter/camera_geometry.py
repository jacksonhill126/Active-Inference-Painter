from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import numpy as np
from PIL import Image


ROBOT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "active_inference_painter.xml"
)


def _numbers(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(item) for item in value.split())


def _vec3(value: str | None) -> np.ndarray:
    values = _numbers(value)
    if not values:
        return np.zeros(3, dtype=np.float64)
    if len(values) != 3:
        raise ValueError(f"expected three values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("camera axis has zero length")
    return vector / norm


def _camera_rotation(element: ElementTree.Element) -> np.ndarray:
    """Return MuJoCo camera-local axes as columns in world coordinates."""

    xyaxes = _numbers(element.get("xyaxes"))
    if len(xyaxes) != 6:
        raise ValueError(
            f"sensor camera {element.get('name')!r} must declare explicit xyaxes"
        )
    x_axis = _unit(np.asarray(xyaxes[:3], dtype=np.float64))
    raw_y = np.asarray(xyaxes[3:], dtype=np.float64)
    y_axis = _unit(raw_y - float(np.dot(raw_y, x_axis)) * x_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


@dataclass(frozen=True, slots=True)
class CanvasPlaneSpec:
    """Registered planar canvas with normalized UV origin at its top-left."""

    center_m: tuple[float, float, float]
    width_m: float
    height_m: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.center_m, dtype=np.float64)

    @property
    def front_normal(self) -> np.ndarray:
        return np.asarray((0.0, -1.0, 0.0), dtype=np.float64)

    def world_from_uv(self, uv: np.ndarray | Iterable[float]) -> np.ndarray:
        coordinates = np.asarray(uv, dtype=np.float64)
        if coordinates.shape[-1] != 2:
            raise ValueError("canvas UV coordinates must end in two components")
        result = np.empty((*coordinates.shape[:-1], 3), dtype=np.float64)
        result[..., 0] = (
            self.center_m[0] + (coordinates[..., 0] - 0.5) * self.width_m
        )
        result[..., 1] = self.center_m[1]
        result[..., 2] = (
            self.center_m[2] + (0.5 - coordinates[..., 1]) * self.height_m
        )
        return result

    def corners_world(self) -> np.ndarray:
        return self.world_from_uv(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        )


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    role: str
    availability: str
    channels: str
    registration: str
    position_m: tuple[float, float, float]
    rotation_camera_to_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    fovy_deg: float
    resolution_px: tuple[int, int]
    model_input_resolution_px: tuple[int, int]
    sample_rate_hz: float

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.position_m, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.rotation_camera_to_world, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CameraRigSpec:
    version: str
    geometry_model: str
    calibration_status: str
    normalization: str
    focus_model: str
    observation_encoding: str
    shutter_model: str
    canvas: CanvasPlaneSpec
    cameras: tuple[CameraSpec, ...]

    def camera(self, name: str) -> CameraSpec:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(name)


def load_camera_rig(
    model_path: Path | str = ROBOT_MODEL_PATH,
) -> CameraRigSpec:
    """Load the provisional sensor-camera contract from the authoritative MJCF."""

    root = ElementTree.parse(Path(model_path)).getroot()
    custom = root.find("custom")
    worldbody = root.find("worldbody")
    if custom is None or worldbody is None:
        raise ValueError("MuJoCo model must contain custom and worldbody elements")
    text = {
        element.get("name", ""): element.get("data", "")
        for element in custom.findall("text")
    }
    camera_names = text["sensor_camera_order"].split()
    roles = text["sensor_camera_roles"].split()
    availability = text["sensor_camera_availability"].split()
    channels = text["sensor_camera_channels"].split()
    registration = text["sensor_camera_registration"].split()
    aligned_fields = (roles, availability, channels, registration)
    if any(len(values) != len(camera_names) for values in aligned_fields):
        raise ValueError("sensor camera metadata must align with camera order")
    numeric = {
        element.get("name", ""): _numbers(element.get("data"))
        for element in custom.findall("numeric")
    }
    input_resolution_values = numeric["sensor_camera_model_resolution_px"]
    sample_rates = numeric["sensor_camera_sample_rate_hz"]
    if len(input_resolution_values) != 2 * len(camera_names):
        raise ValueError("sensor camera model resolutions must align with camera order")
    if len(sample_rates) != len(camera_names):
        raise ValueError("sensor camera sample rates must align with camera order")

    canvas_body = worldbody.find("./body[@name='canvas']")
    if canvas_body is None:
        raise ValueError("MuJoCo model has no world canvas body")
    canvas_geom = canvas_body.find("./geom[@name='canvas_surface']")
    if canvas_geom is None:
        raise ValueError("MuJoCo model has no canvas_surface geom")
    body_position = _vec3(canvas_body.get("pos"))
    geom_position = _vec3(canvas_geom.get("pos"))
    half_size = _numbers(canvas_geom.get("size"))
    if len(half_size) != 3:
        raise ValueError("canvas_surface must declare three box half-sizes")
    canvas = CanvasPlaneSpec(
        center_m=(
            float(body_position[0] + geom_position[0]),
            float(body_position[1] + geom_position[1] - half_size[1]),
            float(body_position[2] + geom_position[2]),
        ),
        width_m=2.0 * half_size[0],
        height_m=2.0 * half_size[2],
    )

    cameras: list[CameraSpec] = []
    for index, (name, role, mode, channel, registration_mode) in enumerate(
        zip(
            camera_names,
            roles,
            availability,
            channels,
            registration,
            strict=True,
        )
    ):
        element = worldbody.find(f"./camera[@name='{name}']")
        if element is None:
            raise ValueError(f"MuJoCo model has no world camera named {name!r}")
        rotation = _camera_rotation(element)
        resolution = tuple(int(value) for value in _numbers(element.get("resolution")))
        if len(resolution) != 2 or min(resolution) <= 1:
            raise ValueError(f"sensor camera {name!r} needs a usable resolution")
        cameras.append(
            CameraSpec(
                name=name,
                role=role,
                availability=mode,
                channels=channel,
                registration=registration_mode,
                position_m=tuple(float(value) for value in _vec3(element.get("pos"))),
                rotation_camera_to_world=tuple(
                    tuple(float(value) for value in row) for row in rotation
                ),
                fovy_deg=float(element.get("fovy", "45")),
                resolution_px=(resolution[0], resolution[1]),
                model_input_resolution_px=(
                    int(input_resolution_values[2 * index]),
                    int(input_resolution_values[2 * index + 1]),
                ),
                sample_rate_hz=float(sample_rates[index]),
            )
        )

    return CameraRigSpec(
        version=text["camera_rig_version"],
        geometry_model=text["camera_geometry_model"],
        calibration_status=text["camera_calibration_status"],
        normalization=text["camera_normalization"],
        focus_model=text["camera_focus_model"],
        observation_encoding=text["camera_observation_encoding"],
        shutter_model=text["camera_shutter_model"],
        canvas=canvas,
        cameras=tuple(cameras),
    )


def camera_intrinsics(
    camera: CameraSpec,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Ideal pinhole intrinsics for a width/height image with square pixels."""

    width, height = image_size or camera.resolution_px
    if width <= 1 or height <= 1:
        raise ValueError("camera image dimensions must be greater than one")
    focal_px = 0.5 * height / np.tan(np.deg2rad(camera.fovy_deg) * 0.5)
    return np.asarray(
        (
            (focal_px, 0.0, 0.5 * width),
            (0.0, focal_px, 0.5 * height),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def canvas_uv_to_camera_homography(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map normalized top-left canvas UV coordinates into camera pixels."""

    if camera.registration != "canvas_plane_homography":
        raise ValueError(
            f"camera {camera.name!r} uses {camera.registration!r}, "
            "not canvas-plane homography"
        )
    origin = canvas.world_from_uv((0.0, 0.0))
    canvas_affine = np.column_stack(
        (
            np.asarray((canvas.width_m, 0.0, 0.0)),
            np.asarray((0.0, 0.0, -canvas.height_m)),
            origin - camera.position,
        )
    )
    # MuJoCo cameras look along local -z and use local +y as image-up.
    camera_to_cv = np.diag((1.0, -1.0, -1.0))
    homography = (
        camera_intrinsics(camera, image_size)
        @ camera_to_cv
        @ camera.rotation.T
        @ canvas_affine
    )
    if abs(float(homography[2, 2])) <= 1e-12:
        raise ValueError(f"camera {camera.name!r} has a degenerate canvas view")
    return homography / homography[2, 2]


def project_world_points(
    camera: CameraSpec,
    points_world: np.ndarray | Iterable[float],
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project arbitrary world points and return pixels plus positive depth."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("world points must end in three components")
    flat = points.reshape(-1, 3)
    local = (flat - camera.position) @ camera.rotation
    cv_coordinates = local @ np.diag((1.0, -1.0, -1.0))
    depth = cv_coordinates[:, 2]
    homogeneous = (camera_intrinsics(camera, image_size) @ cv_coordinates.T).T
    pixels = homogeneous[:, :2] / depth[:, None]
    return (
        pixels.reshape((*points.shape[:-1], 2)),
        depth.reshape(points.shape[:-1]),
    )


def project_canvas_uv(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
    uv: np.ndarray | Iterable[float],
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    coordinates = np.asarray(uv, dtype=np.float64)
    if coordinates.shape[-1] != 2:
        raise ValueError("canvas UV coordinates must end in two components")
    flat = coordinates.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
    projected = (
        canvas_uv_to_camera_homography(camera, canvas, image_size)
        @ homogeneous.T
    ).T
    pixels = projected[:, :2] / projected[:, 2, None]
    return pixels.reshape(coordinates.shape)


def camera_pixels_to_canvas_uv(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
    pixels: np.ndarray | Iterable[float],
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    coordinates = np.asarray(pixels, dtype=np.float64)
    if coordinates.shape[-1] != 2:
        raise ValueError("camera pixels must end in two components")
    flat = coordinates.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
    projected = (
        np.linalg.inv(canvas_uv_to_camera_homography(camera, canvas, image_size))
        @ homogeneous.T
    ).T
    uv = projected[:, :2] / projected[:, 2, None]
    return uv.reshape(coordinates.shape)


def canvas_incidence_angle_deg(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
) -> float:
    canvas_to_camera = _unit(camera.position - canvas.center)
    cosine = float(np.clip(np.dot(canvas.front_normal, canvas_to_camera), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def canvas_axial_depth_range_m(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
) -> tuple[float, float]:
    relative = canvas.corners_world() - camera.position
    camera_coordinates = relative @ camera.rotation
    depth = -camera_coordinates[:, 2]
    if np.any(depth <= 0.0):
        raise ValueError(f"canvas lies behind camera {camera.name!r}")
    return float(np.min(depth)), float(np.max(depth))


def canvas_frustum_mask(
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
    output_size: tuple[int, int],
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return canonical pixels geometrically inside the ideal camera image."""

    output_width, output_height = output_size
    if output_width <= 1 or output_height <= 1:
        raise ValueError("canonical image dimensions must be greater than one")
    source_width, source_height = image_size or camera.resolution_px
    x = np.linspace(0.0, 1.0, output_width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, output_height, dtype=np.float64)
    uu, vv = np.meshgrid(x, y)
    pixels = project_canvas_uv(
        camera,
        canvas,
        np.stack((uu, vv), axis=-1),
        image_size,
    )
    return (
        (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= source_width - 1.0)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= source_height - 1.0)
    )


def rectify_canvas_image(
    image: Image.Image,
    camera: CameraSpec,
    canvas: CanvasPlaneSpec,
    output_size: tuple[int, int],
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    fillcolor: int | tuple[int, ...] = 0,
) -> Image.Image:
    """Rectify an already-undistorted ideal camera frame into canvas UV space.

    This transform normalizes only projective geometry. Occlusion, focus,
    exposure, glare, and camera-specific likelihood precision remain separate
    observation metadata.
    """

    output_width, output_height = output_size
    if output_width <= 1 or output_height <= 1:
        raise ValueError("canonical image dimensions must be greater than one")
    uv_from_output = np.asarray(
        (
            (1.0 / (output_width - 1.0), 0.0, 0.0),
            (0.0, 1.0 / (output_height - 1.0), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    output_to_input = canvas_uv_to_camera_homography(
        camera,
        canvas,
        image.size,
    ) @ uv_from_output
    output_to_input /= output_to_input[2, 2]
    coefficients = (
        output_to_input[0, 0],
        output_to_input[0, 1],
        output_to_input[0, 2],
        output_to_input[1, 0],
        output_to_input[1, 1],
        output_to_input[1, 2],
        output_to_input[2, 0],
        output_to_input[2, 1],
    )
    return image.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        data=coefficients,
        resample=resample,
        fillcolor=fillcolor,
    )
