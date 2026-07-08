from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm_sim import ArmPainterSim, VerticalCanvas
from .config import PainterConfig
from .env import StrokeAction


MATERIAL_CHANNELS = ("thickness", "wetness", "black_mass", "observed_tone", "ground_contrast", "material_coverage")
ACTION_CHANNELS = ("footprint", "start", "end", "width", "amount", "tone")


@dataclass(frozen=True, slots=True)
class MaterialPyramidLevel:
    """One coarse-grained material field level derived from the pixel canvas."""

    name: str
    grid_size: int
    material: np.ndarray

    @property
    def dimensions(self) -> int:
        return int(np.prod(self.material.shape))

    def coverage(self, thickness_scale: float) -> np.ndarray:
        if self.material.shape[0] > 5:
            return self.material[5]
        return coverage_from_thickness(self.material[0], thickness_scale)


@dataclass(frozen=True, slots=True)
class SpatialCanvasState:
    """Explicit spatial material belief substrate for the painting planner.

    The first implementation intentionally uses material fields, not learned
    aesthetic variables: local thickness, wetness, and black pigment mass.
    Coverage is derived from thickness.
    """

    material: np.ndarray
    logvar: np.ndarray
    pyramid: tuple[MaterialPyramidLevel, ...] = ()

    @property
    def grid_size(self) -> int:
        return int(self.material.shape[-1])

    def coverage(self, thickness_scale: float) -> np.ndarray:
        if self.material.shape[0] > 5:
            return self.material[5]
        return coverage_from_thickness(self.material[0], thickness_scale)

    def material_coverage_mean(self, thickness_scale: float) -> float:
        return float(self.coverage(thickness_scale).mean())

    def flatten_mean(self) -> np.ndarray:
        return self.material.astype(np.float32).reshape(-1)

    def flatten_logvar(self) -> np.ndarray:
        return self.logvar.astype(np.float32).reshape(-1)


def spatial_canvas_state(
    sim: ArmPainterSim,
    config: PainterConfig | None = None,
    *,
    logvar_value: float = -8.0,
) -> SpatialCanvasState:
    cfg = config or sim.config
    material = material_grid_from_canvas(sim.canvas, cfg.spatial_grid_size, cfg.spatial_material_channels)
    return SpatialCanvasState(
        material=material,
        logvar=np.full_like(material, logvar_value, dtype=np.float32),
        pyramid=material_pyramid_from_canvas(sim.canvas, cfg),
    )


def material_pyramid_from_canvas(canvas: VerticalCanvas, config: PainterConfig) -> tuple[MaterialPyramidLevel, ...]:
    native_grid = int(canvas.thickness.shape[0])
    requested = (native_grid, *config.material_pyramid_levels, config.spatial_grid_size)
    grid_sizes = sorted({int(size) for size in requested if 0 < int(size) <= native_grid}, reverse=True)
    levels: list[MaterialPyramidLevel] = []
    for grid_size in grid_sizes:
        if grid_size == native_grid:
            name = "pixel"
        elif grid_size == config.spatial_grid_size:
            name = "planner"
        else:
            name = f"tile_{grid_size}"
        levels.append(
            MaterialPyramidLevel(
                name=name,
                grid_size=grid_size,
                material=material_grid_from_canvas(canvas, grid_size, config.spatial_material_channels),
            )
        )
    return tuple(levels)


def material_grid_from_canvas(canvas: VerticalCanvas, grid_size: int, channel_count: int | None = None) -> np.ndarray:
    count = len(MATERIAL_CHANNELS) if channel_count is None else int(channel_count)
    if count < 3 or count > len(MATERIAL_CHANNELS):
        raise ValueError(f"spatial material channel count must be between 3 and {len(MATERIAL_CHANNELS)}.")
    thickness = downsample_mean(canvas.thickness, grid_size)
    wetness = downsample_mean(canvas.wetness, grid_size)
    black_mass = downsample_mean(canvas.black_mass, grid_size)
    material_coverage = downsample_mean(canvas.coverage_field(), grid_size)
    observed_tone = downsample_mean(canvas.observed_tone(), grid_size)
    ground_contrast = downsample_mean(canvas.ground_contrast_field(), grid_size)
    return np.stack(
        [
            thickness,
            wetness,
            black_mass,
            observed_tone,
            ground_contrast,
            material_coverage,
        ],
        axis=0,
    ).astype(np.float32)[:count]


def downsample_mean(field: np.ndarray, grid_size: int) -> np.ndarray:
    if field.ndim != 2:
        raise ValueError("Expected a 2-D canvas field.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    height, width = field.shape
    row_edges = np.linspace(0, height, grid_size + 1, dtype=np.int64)
    col_edges = np.linspace(0, width, grid_size + 1, dtype=np.int64)
    if np.all(np.diff(row_edges) > 0) and np.all(np.diff(col_edges) > 0):
        sums = np.add.reduceat(np.add.reduceat(field.astype(np.float64), row_edges[:-1], axis=0), col_edges[:-1], axis=1)
        counts = np.outer(np.diff(row_edges), np.diff(col_edges)).astype(np.float64)
        return (sums / counts).astype(np.float32)
    # Fallback for grids finer than the canvas, where some blocks are empty.
    out = np.zeros((grid_size, grid_size), dtype=np.float32)
    for row in range(grid_size):
        r0, r1 = int(row_edges[row]), int(row_edges[row + 1])
        for col in range(grid_size):
            c0, c1 = int(col_edges[col]), int(col_edges[col + 1])
            patch = field[r0:max(r0 + 1, r1), c0:max(c0 + 1, c1)]
            out[row, col] = float(patch.mean()) if patch.size else 0.0
    return out


def coverage_from_thickness(thickness: np.ndarray, thickness_scale: float) -> np.ndarray:
    return (1.0 - np.exp(-np.clip(thickness, 0.0, None) / max(1e-8, thickness_scale))).astype(np.float32)


def rasterize_stroke_action(
    action: StrokeAction,
    grid_size: int,
) -> np.ndarray:
    """Rasterize a StrokeAction into deterministic action-conditioning fields."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    x = (xx.astype(np.float32) + 0.5) / grid_size
    y = (yy.astype(np.float32) + 0.5) / grid_size

    if action.stop:
        return np.zeros((len(ACTION_CHANNELS), grid_size, grid_size), dtype=np.float32)

    ax, ay = float(action.x0), float(action.y0)
    bx, by = float(action.x1), float(action.y1)
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy + 1e-8
    t = np.clip(((x - ax) * vx + (y - ay) * vy) / denom, 0.0, 1.0)
    px = ax + t * vx
    py = ay + t * vy
    d2 = (x - px) ** 2 + (y - py) ** 2
    sigma = max(0.006, float(action.width) / 2.355)
    footprint = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
    footprint[footprint < 1e-4] = 0.0

    blob_sigma = max(0.018, float(action.width) * 0.45)
    start = np.exp(-0.5 * ((x - ax) ** 2 + (y - ay) ** 2) / (blob_sigma * blob_sigma)).astype(np.float32)
    end = np.exp(-0.5 * ((x - bx) ** 2 + (y - by) ** 2) / (blob_sigma * blob_sigma)).astype(np.float32)
    start[start < 1e-4] = 0.0
    end[end < 1e-4] = 0.0

    return np.stack(
        [
            footprint,
            start,
            end,
            np.full_like(footprint, float(action.width), dtype=np.float32),
            np.full_like(footprint, float(action.amount), dtype=np.float32),
            np.full_like(footprint, float(action.tone), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def spatial_state_diagnostics(state: SpatialCanvasState, config: PainterConfig) -> dict[str, object]:
    coverage = state.coverage(config.thickness_scale)
    return {
        "kind": "spatial_material",
        "gridSize": state.grid_size,
        "materialChannels": list(MATERIAL_CHANNELS[: state.material.shape[0]]),
        "meanThickness": float(state.material[0].mean()),
        "meanWetness": float(state.material[1].mean()),
        "meanBlackMass": float(state.material[2].mean()),
        "meanObservedTone": float(state.material[3].mean()) if state.material.shape[0] > 3 else None,
        "meanGroundContrast": float(state.material[4].mean()) if state.material.shape[0] > 4 else None,
        "meanMaterialCoverageField": float(state.material[5].mean()) if state.material.shape[0] > 5 else None,
        "coverageMean": float(coverage.mean()),
        "coverageMax": float(coverage.max(initial=0.0)),
        "materialPyramid": material_pyramid_diagnostics(state.pyramid, config),
    }


def material_pyramid_diagnostics(
    pyramid: tuple[MaterialPyramidLevel, ...],
    config: PainterConfig,
) -> list[dict[str, object]]:
    return [
        {
            "name": level.name,
            "gridSize": level.grid_size,
            "dimensions": level.dimensions,
            "meanThickness": float(level.material[0].mean()),
            "meanWetness": float(level.material[1].mean()),
            "meanBlackMass": float(level.material[2].mean()),
            "coverageMean": float(level.coverage(config.thickness_scale).mean()),
        }
        for level in pyramid
    ]
