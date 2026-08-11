from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

try:
    import mujoco
except ImportError:  # pragma: no cover - exercised in base installs
    mujoco = None  # type: ignore[assignment]

from .camera_geometry import (
    ROBOT_MODEL_PATH,
    CameraRigSpec,
    CameraSpec,
    camera_pixels_to_canvas_uv,
    canvas_frustum_mask,
    load_camera_rig,
    project_canvas_uv,
    rectify_canvas_image,
)


CAMERA_OBSERVATION_INTERFACE_VERSION = "camera-observation-interface-v1"
SENSOR_GEOM_GROUP = 2
NATIVE_SENSOR_PRODUCT = "native_sensor"
GLOBAL_CANVAS_PRODUCT = "global_canvas"
FOVEA_CANVAS_PRODUCT = "fovea_canvas"
EDGE_PROFILE_PRODUCT = "edge_profile"
CAMERA_PRODUCT_KINDS = frozenset(
    (
        NATIVE_SENSOR_PRODUCT,
        GLOBAL_CANVAS_PRODUCT,
        FOVEA_CANVAS_PRODUCT,
        EDGE_PROFILE_PRODUCT,
    )
)
FOVEA_SELECTION_BASES = frozenset(
    ("sensor_posterior", "policy_prediction", "operator_diagnostic")
)


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype,
    ndim: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _finite_pair(
    value: tuple[float, float], *, name: str
) -> tuple[float, float]:
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True, slots=True)
class FoveaRequest:
    """An observation-space request for one native-derived canvas fovea.

    The camera process never constructs this request from MuJoCo pose,
    segmentation, contact, or material truth. A perception/policy component
    must supply the request from its permitted posterior or prediction and
    name that basis explicitly.
    """

    request_id: str
    camera_name: str
    requested_time_s: float
    expires_time_s: float
    center_canvas_uv: tuple[float, float]
    span_canvas_uv: tuple[float, float]
    selection_basis: str
    selection_revision: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.camera_name or not self.selection_revision:
            raise ValueError(
                "request_id, camera_name, and selection_revision must be non-empty"
            )
        if self.selection_basis not in FOVEA_SELECTION_BASES:
            raise ValueError(
                f"selection_basis must be one of {sorted(FOVEA_SELECTION_BASES)}"
            )
        if not (
            math.isfinite(self.requested_time_s)
            and math.isfinite(self.expires_time_s)
            and self.expires_time_s >= self.requested_time_s
        ):
            raise ValueError("fovea request timestamps must be finite and ordered")
        center = _finite_pair(self.center_canvas_uv, name="center_canvas_uv")
        span = _finite_pair(self.span_canvas_uv, name="span_canvas_uv")
        if any(component < 0.0 or component > 1.0 for component in center):
            raise ValueError("fovea center must lie inside normalized canvas UV")
        if any(component <= 0.0 or component > 1.0 for component in span):
            raise ValueError("fovea span must lie in (0, 1] per canvas axis")
        object.__setattr__(self, "center_canvas_uv", center)
        object.__setattr__(self, "span_canvas_uv", span)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One read-only image product derived from a physical camera capture.

    The record intentionally contains no simulator segmentation, exact
    visibility, material arrays, contact labels, or pose truth. Occluders are
    retained as ordinary camera pixels. Multiple products may share one
    camera/sequence when they derive from the same native acquisition.
    """

    camera_name: str
    role: str
    sequence: int
    product_kind: str
    product_id: str
    capture_time_s: float
    available_time_s: float
    calibration_revision: str
    observation_model: str
    registration: str
    sampling_kind: str
    source_resolution_px: tuple[int, int]
    declared_acquisition_resolution_px: tuple[int, int]
    grayscale: np.ndarray
    calibration_validity: np.ndarray
    fovea_request_id: str | None = None
    center_canvas_uv: tuple[float, float] | None = None
    span_canvas_uv: tuple[float, float] | None = None
    selection_basis: str | None = None
    selection_revision: str | None = None
    interface_version: str = CAMERA_OBSERVATION_INTERFACE_VERSION

    def __post_init__(self) -> None:
        if not self.camera_name or not self.role or not self.product_id:
            raise ValueError("camera_name, role, and product_id must be non-empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.product_kind not in CAMERA_PRODUCT_KINDS:
            raise ValueError(f"unsupported camera product {self.product_kind!r}")
        if not (
            math.isfinite(self.capture_time_s)
            and math.isfinite(self.available_time_s)
            and self.available_time_s >= self.capture_time_s
        ):
            raise ValueError("camera timestamps must be finite and ordered")
        for name, resolution in (
            ("source_resolution_px", self.source_resolution_px),
            (
                "declared_acquisition_resolution_px",
                self.declared_acquisition_resolution_px,
            ),
        ):
            if len(resolution) != 2 or min(resolution) <= 1:
                raise ValueError(f"{name} must be a usable width/height pair")
        grayscale = _readonly_array(
            self.grayscale,
            dtype=np.dtype(np.float32),
            ndim=2,
            name="grayscale",
        )
        if not np.isfinite(grayscale).all():
            raise ValueError("grayscale must contain only finite values")
        if np.any(grayscale < 0.0) or np.any(grayscale > 1.0):
            raise ValueError("grayscale must lie in [0, 1]")
        validity = _readonly_array(
            self.calibration_validity,
            dtype=np.dtype(np.bool_),
            ndim=2,
            name="calibration_validity",
        )
        if validity.shape != grayscale.shape:
            raise ValueError("calibration_validity must match grayscale shape")
        fovea_metadata = (
            self.fovea_request_id,
            self.center_canvas_uv,
            self.span_canvas_uv,
            self.selection_basis,
            self.selection_revision,
        )
        if self.product_kind == FOVEA_CANVAS_PRODUCT:
            if any(value is None for value in fovea_metadata):
                raise ValueError("fovea products require complete request metadata")
            center = _finite_pair(
                self.center_canvas_uv,  # type: ignore[arg-type]
                name="center_canvas_uv",
            )
            span = _finite_pair(
                self.span_canvas_uv,  # type: ignore[arg-type]
                name="span_canvas_uv",
            )
            if any(component < 0.0 or component > 1.0 for component in center):
                raise ValueError("fovea center must lie inside normalized canvas UV")
            if any(component <= 0.0 or component > 1.0 for component in span):
                raise ValueError("fovea span must lie in (0, 1] per canvas axis")
            if self.selection_basis not in FOVEA_SELECTION_BASES:
                raise ValueError(
                    f"selection_basis must be one of {sorted(FOVEA_SELECTION_BASES)}"
                )
            object.__setattr__(self, "center_canvas_uv", center)
            object.__setattr__(self, "span_canvas_uv", span)
        elif any(value is not None for value in fovea_metadata):
            raise ValueError("only fovea products may contain fovea request metadata")
        if self.product_kind == NATIVE_SENSOR_PRODUCT and grayscale.shape != (
            self.source_resolution_px[1],
            self.source_resolution_px[0],
        ):
            raise ValueError("native product shape must match source resolution")
        object.__setattr__(self, "grayscale", grayscale)
        object.__setattr__(self, "calibration_validity", validity)


@dataclass(frozen=True, slots=True)
class CameraObservationBundle:
    """Camera products delivered together at one observation-loop instant."""

    monotonic_time_s: float
    frames: tuple[CameraFrame, ...]
    interface_version: str = CAMERA_OBSERVATION_INTERFACE_VERSION

    def __post_init__(self) -> None:
        if not math.isfinite(self.monotonic_time_s):
            raise ValueError("monotonic_time_s must be finite")
        keys = tuple(
            (frame.camera_name, frame.sequence, frame.product_id)
            for frame in self.frames
        )
        if len(set(keys)) != len(keys):
            raise ValueError("bundle camera/sequence/product keys must be unique")
        if any(
            frame.available_time_s > self.monotonic_time_s + 1e-12
            for frame in self.frames
        ):
            raise ValueError("bundles cannot expose products before availability")


def _sensor_scene_option() -> mujoco.MjvOption:
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is required for CameraObservationProcess; "
            "install the project's mujoco extra"
        )
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Housing envelopes are not finalized mounts and must not occlude their
    # own simulated optical views.
    option.geomgroup[SENSOR_GEOM_GROUP] = 0
    return option


def _bilinear_sample_normalized(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape
    pixels = np.empty_like(uv, dtype=np.float64)
    pixels[..., 0] = uv[..., 0] * (width - 1)
    pixels[..., 1] = uv[..., 1] * (height - 1)
    return _bilinear_sample_pixels(image, pixels)


def _bilinear_sample_pixels(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(pixels[..., 0], 0.0, width - 1.0)
    y = np.clip(pixels[..., 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


class CameraObservationProcess:
    """MuJoCo-backed native/global/foveal camera generative process.

    MuJoCo supplies geometry, lighting, and occlusion. The caller supplies
    only current superficial grayscale canvas appearance. Segmentation is
    process-internal and discarded after compositing. Acquisition noise is
    applied once in the native frame; global and foveal products are derived
    independently from that same noisy capture.

    ``native_resolution_overrides`` exists only for bounded tests and resource-
    constrained diagnostics. An operational sensor-equivalent process must
    omit it and render the XML-declared acquisition resolution.
    """

    def __init__(
        self,
        model_path: Path | str = ROBOT_MODEL_PATH,
        *,
        random_seed: int = 0,
        native_resolution_overrides: Mapping[str, tuple[int, int]] | None = None,
        identity_canvas_appearance: bool = False,
    ) -> None:
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is required for CameraObservationProcess; "
                "install the project's mujoco extra"
            )
        self.model_path = Path(model_path)
        # Bounded sensor-simulation profiles may opt into a photometrically
        # matched canvas composite. Geometry/occlusion and acquisition noise
        # remain in the camera process, but the canvas pixel intensity is the
        # same superficial-gray quantity used by the declared likelihood.
        self.identity_canvas_appearance = bool(identity_canvas_appearance)
        self.rig: CameraRigSpec = load_camera_rig(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._option = _sensor_scene_option()
        overrides = dict(native_resolution_overrides or {})
        unknown = set(overrides).difference(camera.name for camera in self.rig.cameras)
        if unknown:
            raise KeyError(f"unknown camera resolution overrides: {sorted(unknown)}")
        self._native_resolution_px = {
            camera.name: tuple(overrides.get(camera.name, camera.acquisition_resolution_px))
            for camera in self.rig.cameras
        }
        if any(
            len(resolution) != 2 or min(resolution) <= 1
            for resolution in self._native_resolution_px.values()
        ):
            raise ValueError("native resolution overrides must be usable")
        self._renderers = {
            resolution: mujoco.Renderer(
                self.model,
                height=resolution[1],
                width=resolution[0],
            )
            for resolution in set(self._native_resolution_px.values())
        }
        self._canvas_geom_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "canvas_surface",
            )
        )
        if self._canvas_geom_id < 0:
            raise ValueError("MuJoCo model has no canvas_surface geom")
        seed_sequence = np.random.SeedSequence(random_seed)
        self._rng = {
            camera.name: np.random.default_rng(child)
            for camera, child in zip(
                self.rig.cameras,
                seed_sequence.spawn(len(self.rig.cameras)),
                strict=True,
            )
        }
        self._global_validity = {
            camera.name: self._global_calibration_validity(camera)
            for camera in self.rig.cameras
        }
        self._next_capture_time_s = {
            camera.name: 0.0 for camera in self.rig.cameras
        }
        self._sequence = {camera.name: 0 for camera in self.rig.cameras}
        self._pending: list[CameraFrame] = []
        self._pending_fovea_requests: dict[str, FoveaRequest] = {}
        self._last_observe_time_s = -math.inf

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def reset(self) -> None:
        self._next_capture_time_s = {
            camera.name: 0.0 for camera in self.rig.cameras
        }
        self._sequence = {camera.name: 0 for camera in self.rig.cameras}
        self._pending.clear()
        self._pending_fovea_requests.clear()
        self._last_observe_time_s = -math.inf

    def observe(
        self,
        monotonic_time_s: float,
        *,
        qpos: np.ndarray,
        canvas_grayscale: np.ndarray,
        inspection_available: bool = False,
        fovea_requests: Sequence[FoveaRequest] = (),
    ) -> CameraObservationBundle:
        """Capture due views and deliver products after configured latency."""

        if not math.isfinite(monotonic_time_s):
            raise ValueError("monotonic_time_s must be finite")
        if monotonic_time_s + 1e-12 < self._last_observe_time_s:
            raise ValueError("camera observation time cannot move backwards")
        self._last_observe_time_s = monotonic_time_s
        qpos_array, canvas = self._validated_process_inputs(qpos, canvas_grayscale)
        self._ingest_fovea_requests(monotonic_time_s, fovea_requests)
        self._discard_expired_fovea_requests(monotonic_time_s)

        self.data.qpos[:] = qpos_array
        self.data.time = monotonic_time_s
        mujoco.mj_forward(self.model, self.data)

        for camera in self.rig.cameras:
            available = camera.availability == "continuous" or (
                camera.availability == "park_only" and inspection_available
            )
            if not available:
                continue
            if monotonic_time_s + 1e-12 < self._next_capture_time_s[camera.name]:
                continue
            self._next_capture_time_s[camera.name] = (
                monotonic_time_s + 1.0 / camera.sample_rate_hz
            )
            rng = self._rng[camera.name]
            sequence = self._sequence[camera.name]
            self._sequence[camera.name] += 1
            if float(rng.random()) < camera.dropout_probability:
                continue
            native = self._render_native(camera, canvas, rng)
            requests = tuple(
                request
                for request in self._pending_fovea_requests.values()
                if request.camera_name == camera.name
                and request.requested_time_s <= monotonic_time_s + 1e-12
                and request.expires_time_s >= monotonic_time_s - 1e-12
            )
            self._pending.extend(
                self._products_from_native(
                    camera,
                    native,
                    sequence=sequence,
                    capture_time_s=monotonic_time_s,
                    available_time_s=monotonic_time_s + camera.latency_s,
                    fovea_requests=requests,
                )
            )
            for request in requests:
                del self._pending_fovea_requests[request.request_id]

        ready = tuple(
            frame
            for frame in self._pending
            if frame.available_time_s <= monotonic_time_s + 1e-12
        )
        self._pending = [
            frame
            for frame in self._pending
            if frame.available_time_s > monotonic_time_s + 1e-12
        ]
        return CameraObservationBundle(monotonic_time_s=monotonic_time_s, frames=ready)

    def render_immediate(
        self,
        camera_name: str,
        *,
        monotonic_time_s: float,
        qpos: np.ndarray,
        canvas_grayscale: np.ndarray,
        product_kind: str | None = None,
        fovea_request: FoveaRequest | None = None,
    ) -> CameraFrame:
        """Render one selected product without the rate/latency queue."""

        products = self.render_products_immediate(
            camera_name,
            monotonic_time_s=monotonic_time_s,
            qpos=qpos,
            canvas_grayscale=canvas_grayscale,
            fovea_requests=(() if fovea_request is None else (fovea_request,)),
        )
        camera = self.rig.camera(camera_name)
        selected_kind = product_kind or (
            GLOBAL_CANVAS_PRODUCT
            if camera.registration == "canvas_plane_homography"
            else EDGE_PROFILE_PRODUCT
        )
        selected = tuple(
            product for product in products if product.product_kind == selected_kind
        )
        if len(selected) != 1:
            raise ValueError(
                f"camera {camera_name!r} produced {len(selected)} "
                f"products of kind {selected_kind!r}"
            )
        return selected[0]

    def render_products_immediate(
        self,
        camera_name: str,
        *,
        monotonic_time_s: float,
        qpos: np.ndarray,
        canvas_grayscale: np.ndarray,
        fovea_requests: Sequence[FoveaRequest] = (),
    ) -> tuple[CameraFrame, ...]:
        """Render native/global/requested-foveal products for diagnostics."""

        if not math.isfinite(monotonic_time_s):
            raise ValueError("monotonic_time_s must be finite")
        camera = self.rig.camera(camera_name)
        qpos_array, canvas = self._validated_process_inputs(qpos, canvas_grayscale)
        for request in fovea_requests:
            self._validate_fovea_request(request, monotonic_time_s)
            if request.camera_name != camera_name:
                raise ValueError("immediate fovea request targets another camera")
        self.data.qpos[:] = qpos_array
        self.data.time = monotonic_time_s
        mujoco.mj_forward(self.model, self.data)
        sequence = self._sequence[camera.name]
        self._sequence[camera.name] += 1
        native = self._render_native(camera, canvas, self._rng[camera.name])
        return self._products_from_native(
            camera,
            native,
            sequence=sequence,
            capture_time_s=monotonic_time_s,
            available_time_s=monotonic_time_s,
            fovea_requests=tuple(fovea_requests),
        )

    def _validated_process_inputs(
        self, qpos: np.ndarray, canvas_grayscale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},)")
        canvas = np.asarray(canvas_grayscale, dtype=np.float64)
        if canvas.ndim != 2 or min(canvas.shape) <= 1:
            raise ValueError("canvas_grayscale must be a nontrivial 2-D image")
        if not np.isfinite(canvas).all() or np.any(canvas < 0.0) or np.any(canvas > 1.0):
            raise ValueError("canvas_grayscale must be finite and lie in [0, 1]")
        return qpos_array, canvas

    def _ingest_fovea_requests(
        self,
        monotonic_time_s: float,
        requests: Sequence[FoveaRequest],
    ) -> None:
        seen: set[str] = set()
        for request in requests:
            self._validate_fovea_request(request, monotonic_time_s)
            if request.request_id in seen or request.request_id in self._pending_fovea_requests:
                raise ValueError(f"duplicate pending fovea request {request.request_id!r}")
            seen.add(request.request_id)
            self._pending_fovea_requests[request.request_id] = request

    def _validate_fovea_request(
        self, request: FoveaRequest, monotonic_time_s: float
    ) -> None:
        camera = self.rig.camera(request.camera_name)
        if camera.registration != "canvas_plane_homography":
            raise ValueError("foveae require a canvas-plane camera")
        if camera.foveal_resolution_px is None:
            raise ValueError(f"camera {camera.name!r} has no foveal product")
        if request.requested_time_s > monotonic_time_s + 1e-12:
            raise ValueError("cannot submit a fovea request from the future")

    def _discard_expired_fovea_requests(self, monotonic_time_s: float) -> None:
        expired = tuple(
            request_id
            for request_id, request in self._pending_fovea_requests.items()
            if request.expires_time_s < monotonic_time_s - 1e-12
        )
        for request_id in expired:
            del self._pending_fovea_requests[request_id]

    def _global_calibration_validity(self, camera: CameraSpec) -> np.ndarray:
        width, height = camera.model_input_resolution_px
        if camera.registration == "canvas_plane_homography":
            return canvas_frustum_mask(
                camera,
                self.rig.canvas,
                camera.model_input_resolution_px,
                self._native_resolution_px[camera.name],
            )
        return np.ones((height, width), dtype=np.bool_)

    def _render_native(
        self,
        camera: CameraSpec,
        canvas: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        native_size = self._native_resolution_px[camera.name]
        renderer = self._renderers[native_size]
        renderer.update_scene(self.data, camera=camera.name, scene_option=self._option)
        rgb = renderer.render().astype(np.float64) / 255.0
        scene_gray = np.clip(
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2],
            0.0,
            1.0,
        )

        if camera.registration == "canvas_plane_homography":
            renderer.enable_segmentation_rendering()
            renderer.update_scene(
                self.data,
                camera=camera.name,
                scene_option=self._option,
            )
            segmentation = renderer.render()
            renderer.disable_segmentation_rendering()
            canvas_visible = (
                (segmentation[..., 0] == self._canvas_geom_id)
                & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            )
            yy, xx = np.nonzero(canvas_visible)
            if len(xx):
                camera_pixels = np.column_stack((xx, yy)).astype(np.float64)
                canvas_uv = camera_pixels_to_canvas_uv(
                    camera,
                    self.rig.canvas,
                    camera_pixels,
                    native_size,
                )
                surface = _bilinear_sample_normalized(canvas, canvas_uv)
                raw = scene_gray.copy()
                raw[yy, xx] = (
                    surface
                    if self.identity_canvas_appearance
                    else self._surface_appearance(surface, scene_gray[yy, xx])
                )
            else:
                raw = scene_gray
        else:
            raw = scene_gray
        return self._apply_acquisition_approximation(raw, camera, rng).astype(
            np.float32
        )

    def _products_from_native(
        self,
        camera: CameraSpec,
        native: np.ndarray,
        *,
        sequence: int,
        capture_time_s: float,
        available_time_s: float,
        fovea_requests: Sequence[FoveaRequest],
    ) -> tuple[CameraFrame, ...]:
        native_size = (native.shape[1], native.shape[0])
        products = [
            self._make_frame(
                camera,
                sequence=sequence,
                product_kind=NATIVE_SENSOR_PRODUCT,
                product_id="native",
                capture_time_s=capture_time_s,
                available_time_s=available_time_s,
                registration="native_sensor",
                sampling_kind="direct_acquisition",
                source_resolution_px=native_size,
                grayscale=native,
                validity=np.ones_like(native, dtype=np.bool_),
            )
        ]
        if camera.registration == "canvas_plane_homography":
            global_image = rectify_canvas_image(
                Image.fromarray(native, mode="F"),
                camera,
                self.rig.canvas,
                camera.model_input_resolution_px,
                fillcolor=0,
            )
            products.append(
                self._make_frame(
                    camera,
                    sequence=sequence,
                    product_kind=GLOBAL_CANVAS_PRODUCT,
                    product_id="global",
                    capture_time_s=capture_time_s,
                    available_time_s=available_time_s,
                    registration=camera.registration,
                    sampling_kind="native_to_canvas_homography",
                    source_resolution_px=native_size,
                    grayscale=np.clip(
                        np.asarray(global_image, dtype=np.float32), 0.0, 1.0
                    ),
                    validity=self._global_validity[camera.name],
                )
            )
            products.extend(
                self._fovea_frame(
                    camera,
                    native,
                    request,
                    sequence=sequence,
                    capture_time_s=capture_time_s,
                    available_time_s=available_time_s,
                )
                for request in fovea_requests
            )
        else:
            resized = Image.fromarray(native, mode="F").resize(
                camera.model_input_resolution_px,
                resample=Image.Resampling.BILINEAR,
            )
            edge = np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)
            products.append(
                self._make_frame(
                    camera,
                    sequence=sequence,
                    product_kind=EDGE_PROFILE_PRODUCT,
                    product_id="edge_profile",
                    capture_time_s=capture_time_s,
                    available_time_s=available_time_s,
                    registration=camera.registration,
                    sampling_kind="native_to_model_resolution",
                    source_resolution_px=native_size,
                    grayscale=edge,
                    validity=self._global_validity[camera.name],
                )
            )
        return tuple(products)

    def _fovea_frame(
        self,
        camera: CameraSpec,
        native: np.ndarray,
        request: FoveaRequest,
        *,
        sequence: int,
        capture_time_s: float,
        available_time_s: float,
    ) -> CameraFrame:
        if camera.foveal_resolution_px is None:
            raise ValueError(f"camera {camera.name!r} has no foveal resolution")
        width, height = camera.foveal_resolution_px
        center_u, center_v = request.center_canvas_uv
        span_u, span_v = request.span_canvas_uv
        u = np.linspace(center_u - span_u / 2.0, center_u + span_u / 2.0, width)
        v = np.linspace(center_v - span_v / 2.0, center_v + span_v / 2.0, height)
        uu, vv = np.meshgrid(u, v)
        canvas_uv = np.stack((uu, vv), axis=-1)
        native_size = (native.shape[1], native.shape[0])
        native_pixels = project_canvas_uv(
            camera,
            self.rig.canvas,
            canvas_uv,
            native_size,
        )
        validity = (
            (uu >= 0.0)
            & (uu <= 1.0)
            & (vv >= 0.0)
            & (vv <= 1.0)
            & (native_pixels[..., 0] >= 0.0)
            & (native_pixels[..., 0] <= native_size[0] - 1.0)
            & (native_pixels[..., 1] >= 0.0)
            & (native_pixels[..., 1] <= native_size[1] - 1.0)
        )
        fovea = _bilinear_sample_pixels(native, native_pixels).astype(np.float32)
        fovea[~validity] = 0.0
        return self._make_frame(
            camera,
            sequence=sequence,
            product_kind=FOVEA_CANVAS_PRODUCT,
            product_id=f"fovea:{request.request_id}",
            capture_time_s=capture_time_s,
            available_time_s=available_time_s,
            registration=camera.registration,
            sampling_kind="native_to_requested_canvas_uv",
            source_resolution_px=native_size,
            grayscale=fovea,
            validity=validity,
            fovea_request=request,
        )

    def _make_frame(
        self,
        camera: CameraSpec,
        *,
        sequence: int,
        product_kind: str,
        product_id: str,
        capture_time_s: float,
        available_time_s: float,
        registration: str,
        sampling_kind: str,
        source_resolution_px: tuple[int, int],
        grayscale: np.ndarray,
        validity: np.ndarray,
        fovea_request: FoveaRequest | None = None,
    ) -> CameraFrame:
        return CameraFrame(
            camera_name=camera.name,
            role=camera.role,
            sequence=sequence,
            product_kind=product_kind,
            product_id=product_id,
            capture_time_s=capture_time_s,
            available_time_s=available_time_s,
            calibration_revision=self.rig.version,
            observation_model=self.rig.observation_model,
            registration=registration,
            sampling_kind=sampling_kind,
            source_resolution_px=source_resolution_px,
            declared_acquisition_resolution_px=camera.acquisition_resolution_px,
            grayscale=grayscale,
            calibration_validity=validity,
            fovea_request_id=(
                fovea_request.request_id if fovea_request is not None else None
            ),
            center_canvas_uv=(
                fovea_request.center_canvas_uv if fovea_request is not None else None
            ),
            span_canvas_uv=(
                fovea_request.span_canvas_uv if fovea_request is not None else None
            ),
            selection_basis=(
                fovea_request.selection_basis if fovea_request is not None else None
            ),
            selection_revision=(
                fovea_request.selection_revision
                if fovea_request is not None
                else None
            ),
        )

    def _surface_appearance(
        self,
        surface: np.ndarray,
        rendered_canvas: np.ndarray,
    ) -> np.ndarray:
        if not len(surface):
            return surface
        median = float(np.median(rendered_canvas))
        denominator = max(median, 1e-6)
        diffuse_gain = np.clip(rendered_canvas / denominator, 0.8, 1.2)
        highlight = np.clip(
            (rendered_canvas - median) / max(1.0 - median, 1e-6),
            0.0,
            1.0,
        )
        # Approximation: a weak camera-visible lighting residual, not a wet-
        # paint BRDF and not a hidden wetness observation.
        return np.clip(
            surface * diffuse_gain
            + self.rig.provisional_specular_strength * highlight,
            0.0,
            1.0,
        )

    @staticmethod
    def _apply_acquisition_approximation(
        image: np.ndarray,
        camera: CameraSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sigma = np.sqrt(
            camera.read_noise_std**2
            + camera.signal_noise_std**2 * np.clip(image, 0.0, 1.0)
        )
        noisy = np.clip(image + rng.normal(0.0, sigma), 0.0, 1.0)
        levels = float(2**camera.quantization_bits - 1)
        return np.rint(noisy * levels) / levels
