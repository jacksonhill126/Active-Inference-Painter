from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mujoco
import numpy as np

from .camera_geometry import (
    CameraRigSpec,
    CameraSpec,
    load_camera_rig,
    project_world_points,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "active_inference_painter.xml"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "figures"
    / "camera_observability"
)
CONTROLLED_JOINTS = ("yaw", "pitch", "roll", "elbow")
ROLL_SWEEP_DEG = (-32.0, 0.0, 32.0)
SENSOR_GEOM_GROUP = 2


@dataclass(frozen=True, slots=True)
class ContactPose:
    label: str
    u: float
    v: float
    roll_deg: float
    qpos: tuple[float, ...]
    tip_position_m: tuple[float, float, float]
    solve_error_m: float
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class CameraPoseMetric:
    camera_name: str
    pose_label: str
    roll_deg: float
    row: int
    column: int
    tip_in_frame: bool
    tip_visible: bool
    canvas_visible_fraction: float


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, kind, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model has no {name!r}")
    return object_id


def _set_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: Iterable[float],
) -> None:
    data.qpos[:] = np.asarray(tuple(qpos), dtype=np.float64)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def _joint_addresses(
    model: mujoco.MjModel,
) -> tuple[dict[str, int], dict[str, int]]:
    joint_ids = {
        name: _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in CONTROLLED_JOINTS
    }
    qpos_addresses = {
        name: int(model.jnt_qposadr[joint_id])
        for name, joint_id in joint_ids.items()
    }
    dof_addresses = {
        name: int(model.jnt_dofadr[joint_id])
        for name, joint_id in joint_ids.items()
    }
    return qpos_addresses, dof_addresses


def solve_contact_pose(
    model: mujoco.MjModel,
    target_m: np.ndarray,
    roll_deg: float,
    *,
    initial_qpos: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Solve fixed-roll contact IK using the MuJoCo site Jacobian.

    This is conventional geometric IK used only to sample the plant workspace.
    It does not select painting policies.
    """

    data = mujoco.MjData(model)
    tip_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    qpos_addresses, dof_addresses = _joint_addresses(model)
    variable_names = ("yaw", "pitch", "elbow")
    variable_qpos = np.asarray(
        [qpos_addresses[name] for name in variable_names], dtype=np.int32
    )
    variable_dofs = np.asarray(
        [dof_addresses[name] for name in variable_names], dtype=np.int32
    )
    joint_ids = np.asarray(
        [
            _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in variable_names
        ],
        dtype=np.int32,
    )
    lower = model.jnt_range[joint_ids, 0].astype(np.float64)
    upper = model.jnt_range[joint_ids, 1].astype(np.float64)
    roll_rad = float(np.deg2rad(roll_deg))

    seeds: list[np.ndarray] = []
    if initial_qpos is not None:
        seeds.append(np.asarray(initial_qpos, dtype=np.float64)[variable_qpos])
    seeds.extend(
        np.deg2rad(
            np.asarray(
                (
                    (0.0, -50.0, 95.0),
                    (28.0, -45.0, 80.0),
                    (-28.0, -45.0, 80.0),
                    (0.0, -10.0, -55.0),
                    (35.0, -20.0, -65.0),
                    (-35.0, -20.0, -65.0),
                ),
                dtype=np.float64,
            )
        )
    )

    best_qpos: np.ndarray | None = None
    best_error = float("inf")
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    for seed in seeds:
        q = np.clip(seed, lower, upper)
        for _ in range(80):
            data.qpos[:] = 0.0
            data.qpos[variable_qpos] = q
            data.qpos[qpos_addresses["roll"]] = roll_rad
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            residual = target_m - data.site_xpos[tip_id]
            if float(np.linalg.norm(residual)) < 2e-7:
                break
            mujoco.mj_jacSite(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                tip_id,
            )
            jacobian = jacobian_position[:, variable_dofs]
            damping = 2e-6
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                residual,
            )
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 0.18:
                delta *= 0.18 / delta_norm
            q = np.clip(q + delta, lower, upper)

        data.qpos[:] = 0.0
        data.qpos[variable_qpos] = q
        data.qpos[qpos_addresses["roll"]] = roll_rad
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        error = float(np.linalg.norm(target_m - data.site_xpos[tip_id]))
        if error < best_error:
            best_error = error
            best_qpos = data.qpos.copy()

    if best_qpos is None or best_error > 2e-4:
        raise RuntimeError(
            f"contact IK failed at target {target_m.tolist()}, "
            f"roll={roll_deg:.1f} deg, residual={best_error:.6f} m"
        )
    return best_qpos, best_error


def generate_contact_poses(
    model: mujoco.MjModel,
    rig: CameraRigSpec,
    *,
    grid_size: int = 9,
    rolls_deg: tuple[float, ...] = ROLL_SWEEP_DEG,
    margin: float = 0.035,
) -> tuple[ContactPose, ...]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    coordinates = np.linspace(margin, 1.0 - margin, grid_size)
    poses: list[ContactPose] = []
    previous_qpos: np.ndarray | None = None
    for roll_deg in rolls_deg:
        for row, v in enumerate(coordinates):
            columns = tuple(enumerate(coordinates))
            if row % 2:
                columns = tuple(reversed(columns))
            for column, u in columns:
                target = rig.canvas.world_from_uv((u, v))
                qpos, error = solve_contact_pose(
                    model,
                    target,
                    roll_deg,
                    initial_qpos=previous_qpos,
                )
                previous_qpos = qpos
                poses.append(
                    ContactPose(
                        label=(
                            f"u{u:.3f}_v{v:.3f}_roll{roll_deg:+.0f}"
                        ),
                        u=float(u),
                        v=float(v),
                        roll_deg=float(roll_deg),
                        qpos=tuple(float(value) for value in qpos),
                        tip_position_m=tuple(float(value) for value in target),
                        solve_error_m=error,
                        row=row,
                        column=column,
                    )
                )
    return tuple(poses)


def _sensor_scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Camera housings are sizing aids. Mounts and the inspection stow mechanism
    # are not modeled, so they are excluded from sensor-equivalent visibility.
    option.geomgroup[SENSOR_GEOM_GROUP] = 0
    return option


def _canvas_visible_fraction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rig: CameraRigSpec,
    camera: CameraSpec,
    *,
    ray_grid_size: int = 9,
) -> float:
    canvas_geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "canvas_surface"
    )
    geom_groups = np.ones(6, dtype=np.uint8)
    geom_groups[SENSOR_GEOM_GROUP] = 0
    hit_geom = np.full(1, -1, dtype=np.int32)
    visible = 0
    total = 0
    for v in np.linspace(0.01, 0.99, ray_grid_size):
        for u in np.linspace(0.01, 0.99, ray_grid_size):
            canvas_point = rig.canvas.world_from_uv((u, v))
            direction = canvas_point - camera.position
            direction /= np.linalg.norm(direction)
            hit_geom[0] = -1
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
            total += 1
            if distance > 0.0 and int(hit_geom[0]) == canvas_geom_id:
                visible += 1
    return visible / total


def _tip_visibility_from_segmentation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: CameraSpec,
    renderer: mujoco.Renderer,
    option: mujoco.MjvOption,
) -> tuple[bool, bool]:
    tip_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    bristle_geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "bristle_contact"
    )
    width, height = camera.model_input_resolution_px
    tip_pixels, tip_depth = project_world_points(
        camera,
        data.site_xpos[tip_id],
        (width, height),
    )
    tip_x, tip_y = (float(value) for value in tip_pixels)
    in_frame = bool(
        float(tip_depth) > 0.0
        and 0.0 <= tip_x < width
        and 0.0 <= tip_y < height
    )
    if not in_frame:
        return False, False

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera.name, scene_option=option)
    segmentation = renderer.render()
    renderer.disable_segmentation_rendering()
    bristle_pixels = np.argwhere(
        (segmentation[..., 0] == bristle_geom_id)
        & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    )
    if not len(bristle_pixels):
        return True, False
    squared_distance = (
        (bristle_pixels[:, 1] - tip_x) ** 2
        + (bristle_pixels[:, 0] - tip_y) ** 2
    )
    radius_px = max(7.0, 0.018 * min(width, height))
    return True, bool(float(np.min(squared_distance)) <= radius_px**2)


def evaluate_pose_sweep(
    model: mujoco.MjModel,
    rig: CameraRigSpec,
    poses: tuple[ContactPose, ...],
    *,
    ray_grid_size: int = 9,
) -> tuple[CameraPoseMetric, ...]:
    data = mujoco.MjData(model)
    option = _sensor_scene_option()
    renderers = {
        camera.name: mujoco.Renderer(
            model,
            height=camera.model_input_resolution_px[1],
            width=camera.model_input_resolution_px[0],
        )
        for camera in rig.cameras
    }
    metrics: list[CameraPoseMetric] = []
    try:
        for pose in poses:
            _set_qpos(model, data, pose.qpos)
            for camera in rig.cameras:
                in_frame, visible = _tip_visibility_from_segmentation(
                    model,
                    data,
                    camera,
                    renderers[camera.name],
                    option,
                )
                metrics.append(
                    CameraPoseMetric(
                        camera_name=camera.name,
                        pose_label=pose.label,
                        roll_deg=pose.roll_deg,
                        row=pose.row,
                        column=pose.column,
                        tip_in_frame=in_frame,
                        tip_visible=visible,
                        canvas_visible_fraction=_canvas_visible_fraction(
                            model,
                            data,
                            rig,
                            camera,
                            ray_grid_size=ray_grid_size,
                        ),
                    )
                )
    finally:
        for renderer in renderers.values():
            renderer.close()
    return tuple(metrics)


def _render_grayscale(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: CameraSpec,
    option: mujoco.MjvOption,
) -> np.ndarray:
    width, height = camera.model_input_resolution_px
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera.name, scene_option=option)
        rgb = renderer.render().astype(np.float64)
    finally:
        renderer.close()
    return np.clip(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        0.0,
        255.0,
    ).astype(np.uint8)


def _nearest_pose(
    poses: tuple[ContactPose, ...],
    u: float,
    v: float,
    roll_deg: float,
) -> ContactPose:
    return min(
        poses,
        key=lambda pose: (
            abs(pose.roll_deg - roll_deg)
            + abs(pose.u - u)
            + abs(pose.v - v)
        ),
    )


def render_representative_plate(
    model: mujoco.MjModel,
    rig: CameraRigSpec,
    poses: tuple[ContactPose, ...],
    output_path: Path,
) -> None:
    key_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_KEY, "camera_clear_park"
    )
    park_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, park_data, key_id)
    mujoco.mj_forward(model, park_data)
    representative = (
        ("Park / inspection", None),
        ("Upper left", _nearest_pose(poses, 0.035, 0.035, 0.0)),
        ("Upper right, +32° roll", _nearest_pose(poses, 0.965, 0.035, 32.0)),
        ("Center, −32° roll", _nearest_pose(poses, 0.5, 0.5, -32.0)),
        ("Lower left, −32° roll", _nearest_pose(poses, 0.035, 0.965, -32.0)),
        ("Lower right", _nearest_pose(poses, 0.965, 0.965, 0.0)),
    )
    option = _sensor_scene_option()
    data = mujoco.MjData(model)
    figure, axes = plt.subplots(
        len(rig.cameras),
        len(representative),
        figsize=(18.0, 10.5),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#f4f1ea")
    tip_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    for row, camera in enumerate(rig.cameras):
        for column, (label, pose) in enumerate(representative):
            if pose is None:
                data.qpos[:] = park_data.qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            else:
                _set_qpos(model, data, pose.qpos)
            image = _render_grayscale(model, data, camera, option)
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            pixels, depth = project_world_points(
                camera,
                data.site_xpos[tip_id],
                camera.model_input_resolution_px,
            )
            x, y = (float(value) for value in pixels)
            if (
                float(depth) > 0.0
                and 0.0 <= x < image.shape[1]
                and 0.0 <= y < image.shape[0]
            ):
                axis.scatter(
                    [x],
                    [y],
                    s=42,
                    facecolors="none",
                    edgecolors="#ff8c42",
                    linewidths=1.4,
                )
                axis.plot(
                    [x - 6, x + 6],
                    [y, y],
                    color="#ff8c42",
                    linewidth=0.8,
                )
                axis.plot(
                    [x, x],
                    [y - 6, y + 6],
                    color="#ff8c42",
                    linewidth=0.8,
                )
            axis.set_xticks(())
            axis.set_yticks(())
            for spine in axis.spines.values():
                spine.set_color("#c6c0b5")
                spine.set_linewidth(0.7)
            if row == 0:
                axis.set_title(label, fontsize=10.5, color="#17252a", pad=7)
            if column == 0:
                availability = camera.availability.replace("_", " ")
                axis.set_ylabel(
                    f"{camera.name.replace('_', ' ')}\n"
                    f"{camera.role.replace('_', ' ')} · {availability}",
                    fontsize=9.5,
                    color="#17252a",
                    labelpad=10,
                )
            if (
                camera.availability == "park_only"
                and pose is not None
            ):
                axis.text(
                    0.5,
                    0.96,
                    "HYPOTHETICAL — STOWED WHILE PAINTING",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.8,
                    color="#ffb184",
                    bbox={
                        "facecolor": "#17252a",
                        "edgecolor": "none",
                        "alpha": 0.84,
                        "pad": 2.5,
                    },
                )
    figure.suptitle(
        "Active-Inference Painter · Grayscale Camera Pose Sweep",
        fontsize=18,
        color="#17252a",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.006,
        "Orange reticle: projected brush tip. Compact camera housings are "
        "hidden so the plots describe optical observability, not mount CAD.",
        ha="center",
        fontsize=9,
        color="#44545a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_visibility_figure(
    rig: CameraRigSpec,
    metrics: tuple[CameraPoseMetric, ...],
    output_path: Path,
    *,
    grid_size: int,
) -> None:
    camera_names = tuple(camera.name for camera in rig.cameras)
    contact_camera_names = tuple(
        camera.name
        for camera in rig.cameras
        if camera.role == "contact_tracking"
        and camera.availability == "continuous"
    )
    columns = (*camera_names, "continuous contact union")
    figure, axes = plt.subplots(
        len(ROLL_SWEEP_DEG),
        len(columns),
        figsize=(15.5, 8.8),
    )
    figure.patch.set_facecolor("#f4f1ea")
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.865,
        bottom=0.11,
        wspace=0.38,
        hspace=0.52,
    )
    cmap = ListedColormap(("#8f3b3b", "#d8ede9"))
    metric_lookup = {
        (metric.camera_name, metric.roll_deg, metric.row, metric.column): metric
        for metric in metrics
    }
    for row_index, roll_deg in enumerate(ROLL_SWEEP_DEG):
        for column_index, name in enumerate(columns):
            if name == "continuous contact union":
                visible = np.zeros((grid_size, grid_size), dtype=np.float64)
                canvas_fraction = np.ones(
                    (grid_size, grid_size), dtype=np.float64
                )
                for row in range(grid_size):
                    for column in range(grid_size):
                        selected = [
                            metric_lookup[
                                (camera_name, roll_deg, row, column)
                            ]
                            for camera_name in contact_camera_names
                        ]
                        visible[row, column] = float(
                            any(metric.tip_visible for metric in selected)
                        )
                        canvas_fraction[row, column] = max(
                            metric.canvas_visible_fraction
                            for metric in selected
                        )
            else:
                selected = [
                    metric
                    for metric in metrics
                    if metric.camera_name == name
                    and metric.roll_deg == roll_deg
                ]
                visible = np.zeros((grid_size, grid_size), dtype=np.float64)
                canvas_fraction = np.zeros(
                    (grid_size, grid_size), dtype=np.float64
                )
                for metric in selected:
                    visible[metric.row, metric.column] = float(
                        metric.tip_visible
                    )
                    canvas_fraction[
                        metric.row, metric.column
                    ] = metric.canvas_visible_fraction
            axis = axes[row_index, column_index]
            axis.imshow(
                visible,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                interpolation="nearest",
            )
            axis.set_xticks((0, grid_size - 1), ("left", "right"))
            axis.set_yticks((0, grid_size - 1), ("top", "bottom"))
            axis.tick_params(labelsize=7.5, colors="#44545a")
            visible_fraction = float(np.mean(visible))
            camera = (
                None
                if name == "continuous contact union"
                else rig.camera(name)
            )
            if name == "continuous contact union":
                detail = (
                    f"tip visible {100.0 * visible_fraction:.1f}% · "
                    f"best-view canvas {100.0 * np.min(canvas_fraction):.1f}%"
                )
            elif camera is not None and camera.registration == "canvas_edge_profile":
                detail = (
                    f"tip visible {100.0 * visible_fraction:.1f}% · "
                    "edge profile only"
                )
            else:
                detail = (
                    f"tip visible {100.0 * visible_fraction:.1f}% · "
                    f"min canvas {100.0 * np.min(canvas_fraction):.1f}%"
                )
            axis.text(
                0.5,
                -0.22,
                detail,
                transform=axis.transAxes,
                ha="center",
                fontsize=7.3,
                color="#44545a",
            )
            if row_index == 0:
                axis.set_title(
                    name.replace("_", " "),
                    fontsize=9.5,
                    color="#17252a",
                    pad=8,
                )
            if column_index == 0:
                axis.set_ylabel(
                    f"roll {roll_deg:+.0f}°",
                    fontsize=10,
                    color="#17252a",
                )
            for spine in axis.spines.values():
                spine.set_color("#c6c0b5")
                spine.set_linewidth(0.7)
    figure.suptitle(
        "Brush-Tip Observability Across the Contact Workspace",
        fontsize=18,
        color="#17252a",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Green: the bristle geometry is visible at the projected contact point. "
        "Red: out of frame or occluded. Park-only inspection views are "
        "diagnostic and excluded from the continuous union.",
        ha="center",
        fontsize=8.8,
        color="#44545a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def _summary_payload(
    rig: CameraRigSpec,
    poses: tuple[ContactPose, ...],
    metrics: tuple[CameraPoseMetric, ...],
    *,
    grid_size: int,
    ray_grid_size: int,
) -> dict[str, object]:
    cameras: dict[str, object] = {}
    for camera in rig.cameras:
        selected = [
            metric for metric in metrics if metric.camera_name == camera.name
        ]
        cameras[camera.name] = {
            "role": camera.role,
            "availability": camera.availability,
            "registration": camera.registration,
            "channels": camera.channels,
            "hardware_baseline": camera.hardware_baseline,
            "hardware_status": camera.hardware_status,
            "lens_status": camera.lens_status,
            "capture_mode": camera.capture_mode,
            "transport": camera.transport,
            "shutter_model": camera.shutter_model,
            "focal_length_mm": camera.focal_length_mm,
            "active_sensor_width_mm": camera.active_sensor_width_mm,
            "full_frame_equivalent_focal_length_mm": (
                camera.full_frame_equivalent_focal_length_mm
            ),
            "vertical_field_of_view_deg": camera.fovy_deg,
            "acquisition_resolution_px": list(
                camera.acquisition_resolution_px
            ),
            "model_input_resolution_px": list(
                camera.model_input_resolution_px
            ),
            "foveal_resolution_px": (
                list(camera.foveal_resolution_px)
                if camera.foveal_resolution_px is not None
                else None
            ),
            "sample_rate_hz": camera.sample_rate_hz,
            "tip_in_frame_fraction": float(
                np.mean([metric.tip_in_frame for metric in selected])
            ),
            "tip_visible_fraction": float(
                np.mean([metric.tip_visible for metric in selected])
            ),
            "canvas_visible_fraction_min": float(
                min(metric.canvas_visible_fraction for metric in selected)
            ),
            "canvas_visible_fraction_mean": float(
                np.mean(
                    [
                        metric.canvas_visible_fraction
                        for metric in selected
                    ]
                )
            ),
            "by_roll": {
                f"{roll_deg:+.0f}": {
                    "tip_visible_fraction": float(
                        np.mean(
                            [
                                metric.tip_visible
                                for metric in selected
                                if metric.roll_deg == roll_deg
                            ]
                        )
                    ),
                    "canvas_visible_fraction_min": float(
                        min(
                            metric.canvas_visible_fraction
                            for metric in selected
                            if metric.roll_deg == roll_deg
                        )
                    ),
                }
                for roll_deg in ROLL_SWEEP_DEG
            },
        }
    contact_camera_names = {
        camera.name
        for camera in rig.cameras
        if camera.role == "contact_tracking"
        and camera.availability == "continuous"
    }
    by_pose: dict[str, list[CameraPoseMetric]] = {}
    for metric in metrics:
        if metric.camera_name in contact_camera_names:
            by_pose.setdefault(metric.pose_label, []).append(metric)
    continuous_union = float(
        np.mean(
            [
                any(metric.tip_visible for metric in pose_metrics)
                for pose_metrics in by_pose.values()
            ]
        )
    )
    return {
        "schema_version": "camera-pose-sweep-v1",
        "model_path": str(MODEL_PATH),
        "camera_rig_version": rig.version,
        "grid_size": grid_size,
        "roll_sweep_deg": list(ROLL_SWEEP_DEG),
        "contact_pose_count": len(poses),
        "ray_grid_size": ray_grid_size,
        "max_ik_residual_m": max(pose.solve_error_m for pose in poses),
        "continuous_contact_tip_visible_union_fraction": continuous_union,
        "cameras": cameras,
        "approximations": [
            "Camera and mount visual geoms are excluded from sensor rays.",
            "Only the selected compact left/right views participate; no "
            "inspection or overhead camera is assumed.",
            "MuJoCo grayscale shading does not include paint appearance, lens "
            "distortion, glare, blur, exposure dynamics, or sensor noise.",
            "Bristle visibility is segmentation-based at model input "
            "resolution; it is not a learned detector score.",
        ],
    }


def _write_detail_tables(
    model: mujoco.MjModel,
    output_dir: Path,
    poses: tuple[ContactPose, ...],
    metrics: tuple[CameraPoseMetric, ...],
) -> None:
    qpos_addresses, _ = _joint_addresses(model)
    with (output_dir / "contact_poses.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "pose_label",
                "canvas_u",
                "canvas_v",
                "grid_row",
                "grid_column",
                "roll_sweep_deg",
                "yaw_rad",
                "pitch_rad",
                "roll_rad",
                "elbow_rad",
                "tip_x_m",
                "tip_y_m",
                "tip_z_m",
                "ik_residual_m",
            ),
        )
        writer.writeheader()
        for pose in poses:
            writer.writerow(
                {
                    "pose_label": pose.label,
                    "canvas_u": pose.u,
                    "canvas_v": pose.v,
                    "grid_row": pose.row,
                    "grid_column": pose.column,
                    "roll_sweep_deg": pose.roll_deg,
                    "yaw_rad": pose.qpos[qpos_addresses["yaw"]],
                    "pitch_rad": pose.qpos[qpos_addresses["pitch"]],
                    "roll_rad": pose.qpos[qpos_addresses["roll"]],
                    "elbow_rad": pose.qpos[qpos_addresses["elbow"]],
                    "tip_x_m": pose.tip_position_m[0],
                    "tip_y_m": pose.tip_position_m[1],
                    "tip_z_m": pose.tip_position_m[2],
                    "ik_residual_m": pose.solve_error_m,
                }
            )
    with (output_dir / "camera_pose_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "camera_name",
                "pose_label",
                "roll_deg",
                "grid_row",
                "grid_column",
                "tip_in_frame",
                "tip_visible",
                "canvas_visible_fraction",
            ),
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "camera_name": metric.camera_name,
                    "pose_label": metric.pose_label,
                    "roll_deg": metric.roll_deg,
                    "grid_row": metric.row,
                    "grid_column": metric.column,
                    "tip_in_frame": metric.tip_in_frame,
                    "tip_visible": metric.tip_visible,
                    "canvas_visible_fraction": (
                        metric.canvas_visible_fraction
                    ),
                }
            )


def run_pose_sweep(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    grid_size: int = 9,
    ray_grid_size: int = 9,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    rig = load_camera_rig(MODEL_PATH)
    poses = generate_contact_poses(model, rig, grid_size=grid_size)
    metrics = evaluate_pose_sweep(
        model,
        rig,
        poses,
        ray_grid_size=ray_grid_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    render_representative_plate(
        model,
        rig,
        poses,
        output_dir / "camera_pose_sweep_plate.png",
    )
    render_visibility_figure(
        rig,
        metrics,
        output_dir / "camera_visibility_maps.png",
        grid_size=grid_size,
    )
    payload = _summary_payload(
        rig,
        poses,
        metrics,
        grid_size=grid_size,
        ray_grid_size=ray_grid_size,
    )
    _write_detail_tables(model, output_dir, poses, metrics)
    (output_dir / "camera_pose_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and quantify the MuJoCo camera contact-pose sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--ray-grid-size", type=int, default=9)
    arguments = parser.parse_args()
    payload = run_pose_sweep(
        arguments.output_dir,
        grid_size=arguments.grid_size,
        ray_grid_size=arguments.ray_grid_size,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
