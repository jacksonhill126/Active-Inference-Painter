from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import PainterConfig
from .env import StrokeAction
from .policies import MotorPrimitiveLatent
from .spatial_state import SpatialCanvasState, rasterize_stroke_action


@dataclass(frozen=True, slots=True)
class LocalPatchBounds:
    """Pixel-grid crop bounds for one local transition likelihood."""

    row0: int
    row1: int
    col0: int
    col1: int
    grid_size: int

    def __post_init__(self) -> None:
        if not (0 <= self.row0 < self.row1 <= self.grid_size):
            raise ValueError("Invalid row bounds for local patch.")
        if not (0 <= self.col0 < self.col1 <= self.grid_size):
            raise ValueError("Invalid column bounds for local patch.")

    @property
    def height(self) -> int:
        return self.row1 - self.row0

    @property
    def width(self) -> int:
        return self.col1 - self.col0

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def area_fraction(self) -> float:
        return float(self.area / max(1, self.grid_size * self.grid_size))

    def slices(self) -> tuple[slice, slice]:
        return slice(self.row0, self.row1), slice(self.col0, self.col1)


@dataclass(frozen=True, slots=True)
class LocalPatchTransition:
    """One cropped pixel-level training example for p(s_patch_next | s_patch, a_patch)."""

    bounds: LocalPatchBounds
    material: np.ndarray
    action: np.ndarray
    next_material: np.ndarray


@dataclass(slots=True)
class LocalPatchBatch:
    material: torch.Tensor
    action: torch.Tensor
    next_material: torch.Tensor
    mask: torch.Tensor


class LocalPatchReplayBuffer:
    """Variable-size local transition replay with padded masked batches."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.data: deque[LocalPatchTransition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: LocalPatchTransition) -> None:
        self.data.append(transition)

    def add_from_states(
        self,
        state: SpatialCanvasState,
        action: StrokeAction,
        next_state: SpatialCanvasState,
        config: PainterConfig,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        transition = local_patch_transition_from_states(state, action, next_state, config, motor_primitive)
        if transition is not None:
            self.add(transition)

    def sample(self, batch_size: int, device: torch.device) -> LocalPatchBatch:
        indices = self.rng.integers(0, len(self.data), size=batch_size)
        transitions = [self.data[int(index)] for index in indices]
        max_h = max(transition.bounds.height for transition in transitions)
        max_w = max(transition.bounds.width for transition in transitions)
        channels = transitions[0].material.shape[0]
        action_channels = transitions[0].action.shape[0]
        material = np.zeros((batch_size, channels, max_h, max_w), dtype=np.float32)
        action = np.zeros((batch_size, action_channels, max_h, max_w), dtype=np.float32)
        next_material = np.zeros_like(material)
        mask = np.zeros((batch_size, 1, max_h, max_w), dtype=np.float32)
        for row, transition in enumerate(transitions):
            h, w = transition.bounds.height, transition.bounds.width
            material[row, :, :h, :w] = transition.material
            action[row, :, :h, :w] = transition.action
            next_material[row, :, :h, :w] = transition.next_material
            mask[row, :, :h, :w] = 1.0
        return LocalPatchBatch(
            material=torch.tensor(material, device=device),
            action=torch.tensor(action, device=device),
            next_material=torch.tensor(next_material, device=device),
            mask=torch.tensor(mask, device=device),
        )


def pixel_material_from_state(state: SpatialCanvasState) -> np.ndarray:
    if state.pyramid:
        level = max(state.pyramid, key=lambda item: item.grid_size)
        return level.material.astype(np.float32)
    return state.material.astype(np.float32)


def pixel_logvar_from_state(state: SpatialCanvasState, config: PainterConfig) -> np.ndarray:
    material = pixel_material_from_state(state)
    if state.logvar.shape == material.shape:
        return state.logvar.astype(np.float32)
    value = float(np.clip(state.logvar.mean() if state.logvar.size else config.local_identity_logvar, -30.0, 20.0))
    return np.full_like(material, value, dtype=np.float32)


def local_patch_bounds_for_action(
    action: StrokeAction,
    grid_size: int,
    config: PainterConfig,
) -> LocalPatchBounds | None:
    if action.stop:
        return None
    raster = rasterize_stroke_action(action, grid_size, config=config)
    support = np.any(raster[:3] > 0.0, axis=0)
    if not np.any(support):
        return None
    rows, cols = np.nonzero(support)
    margin = max(0, int(config.local_patch_margin_cells))
    row0 = max(0, int(rows.min()) - margin)
    row1 = min(grid_size, int(rows.max()) + margin + 1)
    col0 = max(0, int(cols.min()) - margin)
    col1 = min(grid_size, int(cols.max()) + margin + 1)
    row0, row1 = _expand_interval_to_minimum(row0, row1, grid_size, int(config.local_patch_min_cells))
    col0, col1 = _expand_interval_to_minimum(col0, col1, grid_size, int(config.local_patch_min_cells))
    return LocalPatchBounds(row0=row0, row1=row1, col0=col0, col1=col1, grid_size=grid_size)


def crop_patch(field: np.ndarray, bounds: LocalPatchBounds) -> np.ndarray:
    row_slice, col_slice = bounds.slices()
    return field[..., row_slice, col_slice].astype(np.float32, copy=True)


def paste_patch(field: np.ndarray, patch: np.ndarray, bounds: LocalPatchBounds) -> np.ndarray:
    out = field.astype(np.float32, copy=True)
    row_slice, col_slice = bounds.slices()
    out[..., row_slice, col_slice] = patch.astype(np.float32)
    return out


def local_patch_transition_from_states(
    state: SpatialCanvasState,
    action: StrokeAction,
    next_state: SpatialCanvasState,
    config: PainterConfig,
    motor_primitive: MotorPrimitiveLatent | None = None,
) -> LocalPatchTransition | None:
    material = pixel_material_from_state(state)
    next_material = pixel_material_from_state(next_state)
    grid_size = int(material.shape[-1])
    bounds = local_patch_bounds_for_action(action, grid_size, config)
    if bounds is None:
        return None
    action_raster = rasterize_stroke_action(action, grid_size, motor_primitive=motor_primitive, config=config)
    return LocalPatchTransition(
        bounds=bounds,
        material=crop_patch(material, bounds),
        action=crop_patch(action_raster, bounds),
        next_material=crop_patch(next_material, bounds),
    )


def _expand_interval_to_minimum(start: int, end: int, limit: int, minimum: int) -> tuple[int, int]:
    minimum = min(max(1, minimum), limit)
    size = end - start
    if size >= minimum:
        return start, end
    missing = minimum - size
    before = missing // 2
    after = missing - before
    start = max(0, start - before)
    end = min(limit, end + after)
    if end - start < minimum:
        if start == 0:
            end = min(limit, minimum)
        else:
            start = max(0, limit - minimum)
            end = limit
    return start, end
