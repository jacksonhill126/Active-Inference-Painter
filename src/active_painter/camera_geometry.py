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
    hardware_baseline: str
    hardware_status: str
    lens_status: str
    capture_mode: str
    transport: str
    shutter_model: str
    position_m: tuple[float, float, float]
    rotation_camera_to_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    fovy_deg: float
    resolution_px: tuple[int, int]
    acquisition_resolution_px: tuple[int, int]
    model_input_resolution_px: tuple[int, int]
    foveal_resolution_px: tuple[int, int] | None
    focal_length_mm: float | None
    active_sensor_width_mm: float | None
    full_frame_equivalent_focal_length_mm: float | None
    sample_rate_hz: float
    read_noise_std: float
    signal_noise_std: float
    likelihood_model_error_std: float
    likelihood_inlier_probability: float
    likelihood_outlier_std: float
    latency_s: float
    dropout_probability: float
    quantization_bits: int

    def __post_init__(self) -> None:
        if min(self.acquisition_resolution_px) <= 1:
            raise ValueError("camera acquisition resolution must be usable")
        if self.foveal_resolution_px is not None and min(
            self.foveal_resolution_px
        ) <= 1:
            raise ValueError("camera foveal resolution must be usable when present")
        optical_values = (
            self.focal_length_mm,
            self.active_sensor_width_mm,
            self.full_frame_equivalent_focal_length_mm,
        )
        if any(
            value is not None and (value <= 0.0 or not np.isfinite(value))
            for value in optical_values
        ):
            raise ValueError("known camera optical dimensions must be positive")
        if self.sample_rate_hz <= 0.0 or not np.isfinite(self.sample_rate_hz):
            raise ValueError("camera sample_rate_hz must be finite and positive")
        if (
            self.read_noise_std < 0.0
            or self.signal_noise_std < 0.0
            or not np.isfinite(self.read_noise_std)
            or not np.isfinite(self.signal_noise_std)
        ):
            raise ValueError(
                "camera noise parameters must be finite and non-negative"
            )
        if (
            self.likelihood_model_error_std <= 0.0
            or self.likelihood_outlier_std <= 0.0
            or not np.isfinite(self.likelihood_model_error_std)
            or not np.isfinite(self.likelihood_outlier_std)
        ):
            raise ValueError("camera likelihood standard deviations must be finite and positive")
        if not 0.0 < self.likelihood_inlier_probability < 1.0:
            raise ValueError("camera likelihood inlier probability must lie in (0, 1)")
        if self.latency_s < 0.0 or not np.isfinite(self.latency_s):
            raise ValueError("camera latency_s must be finite and non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("camera dropout_probability must lie in [0, 1]")
        if not 1 <= self.quantization_bits <= 16:
            raise ValueError("camera quantization_bits must lie in [1, 16]")

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
    observation_model: str
    product_contract: str
    fovea_addressing: str
    fovea_selection_boundary: str
    noise_model: str
    noise_status: str
    likelihood_model: str
    likelihood_status: str
    specular_model: str
    provisional_specular_strength: float
    canvas: CanvasPlaneSpec
    cameras: tuple[CameraSpec, ...]

    def __post_init__(self) -> None:
        if not self.cameras or len({camera.name for camera in self.cameras}) != len(
            self.cameras
        ):
            raise ValueError("camera names must be non-empty and unique")
        if (
            self.provisional_specular_strength < 0.0
            or self.provisional_specular_strength > 1.0
            or not np.isfinite(self.provisional_specular_strength)
        ):
            raise ValueError("provisional specular strength must lie in [0, 1]")

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
    hardware = text["sensor_camera_hardware_baseline"].split()
    hardware_status = text["sensor_camera_hardware_status"].split()
    lens_status = text["sensor_camera_lens_status"].split()
    capture_modes = text["sensor_camera_capture_mode"].split()
    transport = text["sensor_camera_transport"].split()
    shutter_models = text["sensor_camera_shutter_model"].split()
    aligned_fields = (
        roles,
        availability,
        channels,
        registration,
        hardware,
        hardware_status,
        lens_status,
        capture_modes,
        transport,
        shutter_models,
    )
    if any(len(values) != len(camera_names) for values in aligned_fields):
        raise ValueError("sensor camera metadata must align with camera order")
    numeric = {
        element.get("name", ""): _numbers(element.get("data"))
        for element in custom.findall("numeric")
    }
    input_resolution_values = numeric["sensor_camera_model_resolution_px"]
    acquisition_resolution_values = numeric[
        "sensor_camera_acquisition_resolution_px"
    ]
    foveal_resolution_values = numeric["sensor_camera_foveal_resolution_px"]
    focal_lengths = numeric["sensor_camera_focal_length_mm"]
    active_sensor_widths = numeric["sensor_camera_active_width_mm"]
    equivalent_focal_lengths = numeric[
        "sensor_camera_full_frame_equivalent_focal_length_mm"
    ]
    sample_rates = numeric["sensor_camera_sample_rate_hz"]
    read_noise = numeric["sensor_camera_read_noise_std"]
    signal_noise = numeric["sensor_camera_signal_noise_std"]
    likelihood_model_error = numeric[
        "sensor_camera_likelihood_model_error_std"
    ]
    likelihood_inlier_probability = numeric[
        "sensor_camera_likelihood_inlier_probability"
    ]
    likelihood_outlier_std = numeric[
        "sensor_camera_likelihood_outlier_std"
    ]
    latencies = numeric["sensor_camera_latency_s"]
    dropout = numeric["sensor_camera_dropout_probability"]
    quantization_bits = numeric["sensor_camera_quantization_bits"]
    if len(input_resolution_values) != 2 * len(camera_names):
        raise ValueError("sensor camera model resolutions must align with camera order")
    if len(acquisition_resolution_values) != 2 * len(camera_names):
        raise ValueError(
            "sensor camera acquisition resolutions must align with camera order"
        )
    if len(foveal_resolution_values) != 2 * len(camera_names):
        raise ValueError(
            "sensor camera foveal resolutions must align with camera order"
        )
    per_camera_numeric = {
        "sample rates": sample_rates,
        "focal lengths": focal_lengths,
        "active sensor widths": active_sensor_widths,
        "equivalent focal lengths": equivalent_focal_lengths,
        "read noise": read_noise,
        "signal noise": signal_noise,
        "likelihood model error": likelihood_model_error,
        "likelihood inlier probabilities": likelihood_inlier_probability,
        "likelihood outlier standard deviations": likelihood_outlier_std,
        "latencies": latencies,
        "dropout probabilities": dropout,
        "quantization bits": quantization_bits,
    }
    for label, values in per_camera_numeric.items():
        if len(values) != len(camera_names):
            raise ValueError(f"sensor camera {label} must align with camera order")

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
    for index, (
        name,
        role,
        mode,
        channel,
        registration_mode,
        hardware_model,
        status,
        lens_state,
        capture_mode,
        transport_kind,
        shutter,
    ) in enumerate(
        zip(
            camera_names,
            roles,
            availability,
            channels,
            registration,
            hardware,
            hardware_status,
            lens_status,
            capture_modes,
            transport,
            shutter_models,
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
                hardware_baseline=hardware_model,
                hardware_status=status,
                lens_status=lens_state,
                capture_mode=capture_mode,
                transport=transport_kind,
                shutter_model=shutter,
                position_m=tuple(float(value) for value in _vec3(element.get("pos"))),
                rotation_camera_to_world=tuple(
                    tuple(float(value) for value in row) for row in rotation
                ),
                fovy_deg=float(element.get("fovy", "45")),
                resolution_px=(resolution[0], resolution[1]),
                acquisition_resolution_px=(
                    int(acquisition_resolution_values[2 * index]),
                    int(acquisition_resolution_values[2 * index + 1]),
                ),
                model_input_resolution_px=(
                    int(input_resolution_values[2 * index]),
                    int(input_resolution_values[2 * index + 1]),
                ),
                foveal_resolution_px=(
                    (
                        int(foveal_resolution_values[2 * index]),
                        int(foveal_resolution_values[2 * index + 1]),
                    )
                    if foveal_resolution_values[2 * index] > 0
                    and foveal_resolution_values[2 * index + 1] > 0
                    else None
                ),
                focal_length_mm=(
                    float(focal_lengths[index])
                    if focal_lengths[index] > 0.0
                    else None
                ),
                active_sensor_width_mm=(
                    float(active_sensor_widths[index])
                    if active_sensor_widths[index] > 0.0
                    else None
                ),
                full_frame_equivalent_focal_length_mm=(
                    float(equivalent_focal_lengths[index])
                    if equivalent_focal_lengths[index] > 0.0
                    else None
                ),
                sample_rate_hz=float(sample_rates[index]),
                read_noise_std=float(read_noise[index]),
                signal_noise_std=float(signal_noise[index]),
                likelihood_model_error_std=float(likelihood_model_error[index]),
                likelihood_inlier_probability=float(
                    likelihood_inlier_probability[index]
                ),
                likelihood_outlier_std=float(likelihood_outlier_std[index]),
                latency_s=float(latencies[index]),
                dropout_probability=float(dropout[index]),
                quantization_bits=int(quantization_bits[index]),
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
        observation_model=text["camera_observation_model"],
        product_contract=text["camera_product_contract"],
        fovea_addressing=text["camera_fovea_addressing"],
        fovea_selection_boundary=text["camera_fovea_selection_boundary"],
        noise_model=text["camera_noise_model"],
        noise_status=text["camera_noise_status"],
        likelihood_model=text["camera_likelihood_model"],
        likelihood_status=text["camera_likelihood_status"],
        specular_model=text["camera_specular_model"],
        provisional_specular_strength=float(
            numeric["camera_provisional_specular_strength"][0]
        ),
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
