from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import PainterConfig
from .spatial_state import SpatialCanvasState


@dataclass(frozen=True, slots=True)
class MarkEventSlot:
    active_probability: float
    center_x: float
    center_y: float
    covariance_xx: float
    covariance_xy: float
    covariance_yy: float
    mass: float
    mean_thickness: float
    mean_wetness: float
    mean_observed_tone: float
    mean_ground_contrast: float
    mean_material_coverage: float


@dataclass(frozen=True, slots=True)
class MarkEventBelief:
    slots: tuple[MarkEventSlot, ...]
    active_count: int
    residual_coverage_mass: float
    approximation: str = (
        "connected-component posterior summary over spatial material coverage; "
        "not a policy preference or reward term"
    )

    def diagnostics(self) -> dict[str, object]:
        return {
            "activeCount": self.active_count,
            "residualCoverageMass": self.residual_coverage_mass,
            "approximation": self.approximation,
            "slots": [asdict(slot) for slot in self.slots],
        }

    def feature_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [
                    slot.active_probability,
                    slot.center_x,
                    slot.center_y,
                    slot.covariance_xx,
                    slot.covariance_xy,
                    slot.covariance_yy,
                    slot.mass,
                    slot.mean_thickness,
                    slot.mean_wetness,
                    slot.mean_observed_tone,
                    slot.mean_ground_contrast,
                    slot.mean_material_coverage,
                ]
                for slot in self.slots
            ],
            dtype=np.float32,
        )


def infer_mark_event_belief(state: SpatialCanvasState, config: PainterConfig) -> MarkEventBelief:
    coverage = state.coverage(config.thickness_scale)
    active = coverage >= config.mark_activation_coverage
    components = _connected_components(active)
    slots = [
        _slot_from_component(state, coverage, component)
        for component in components
    ]
    slots.sort(key=lambda slot: slot.mass, reverse=True)
    selected = slots[: config.mark_slot_count]
    selected_mass = sum(slot.mass for slot in selected)
    total_mass = float(coverage.sum() / coverage.size)
    padded = tuple(selected + [_empty_slot() for _ in range(max(0, config.mark_slot_count - len(selected)))])
    return MarkEventBelief(
        slots=padded,
        active_count=len(selected),
        residual_coverage_mass=max(0.0, total_mass - selected_mass),
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    if mask.ndim != 2:
        raise ValueError("Expected a 2-D mark activation mask.")
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=np.int64))
    return components


def _slot_from_component(
    state: SpatialCanvasState,
    coverage: np.ndarray,
    component: np.ndarray,
) -> MarkEventSlot:
    rows = component[:, 0]
    cols = component[:, 1]
    grid_size = max(1, state.grid_size)
    weights = coverage[rows, cols].astype(np.float64)
    mass_sum = float(weights.sum())
    if mass_sum <= 0.0:
        return _empty_slot()
    x = (cols.astype(np.float64) + 0.5) / grid_size
    y = (rows.astype(np.float64) + 0.5) / grid_size
    center_x = float(np.dot(weights, x) / mass_sum)
    center_y = float(np.dot(weights, y) / mass_sum)
    dx = x - center_x
    dy = y - center_y
    covariance_xx = float(np.dot(weights, dx * dx) / mass_sum)
    covariance_xy = float(np.dot(weights, dx * dy) / mass_sum)
    covariance_yy = float(np.dot(weights, dy * dy) / mass_sum)
    observed_tone = state.material[3] if state.material.shape[0] > 3 else np.zeros_like(coverage)
    ground_contrast = state.material[4] if state.material.shape[0] > 4 else np.zeros_like(coverage)
    material_coverage = state.material[5] if state.material.shape[0] > 5 else coverage
    return MarkEventSlot(
        active_probability=1.0,
        center_x=center_x,
        center_y=center_y,
        covariance_xx=covariance_xx,
        covariance_xy=covariance_xy,
        covariance_yy=covariance_yy,
        mass=float(mass_sum / coverage.size),
        mean_thickness=float(np.dot(weights, state.material[0, rows, cols]) / mass_sum),
        mean_wetness=float(np.dot(weights, state.material[1, rows, cols]) / mass_sum),
        mean_observed_tone=float(np.dot(weights, observed_tone[rows, cols]) / mass_sum),
        mean_ground_contrast=float(np.dot(weights, ground_contrast[rows, cols]) / mass_sum),
        mean_material_coverage=float(np.dot(weights, material_coverage[rows, cols]) / mass_sum),
    )


def _empty_slot() -> MarkEventSlot:
    return MarkEventSlot(
        active_probability=0.0,
        center_x=0.0,
        center_y=0.0,
        covariance_xx=0.0,
        covariance_xy=0.0,
        covariance_yy=0.0,
        mass=0.0,
        mean_thickness=0.0,
        mean_wetness=0.0,
        mean_observed_tone=0.0,
        mean_ground_contrast=0.0,
        mean_material_coverage=0.0,
    )
