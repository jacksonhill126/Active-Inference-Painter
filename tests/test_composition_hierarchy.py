"""Tests for the compression-gap composition hierarchy.

The declared claim: gap(s) = ELBO_hier(s) - log p_flat(s) scores spatially
structured canvases above cell-shuffled ones (same marginals, structure
destroyed) and above blank ones, once the hierarchy has trained on structured
examples. Blank canvases never score positive: the flat code is already
perfect there and the hierarchy pays for its latent code.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from active_painter.composition import CompositionHierarchy
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.policies import Policy
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_state import SpatialCanvasState, rasterize_stroke_action


def structured_fields(rng: np.random.Generator, count: int, channels: int = 6, grid: int = 16) -> torch.Tensor:
    """Half-plane compositions: strong spatial structure, coherent channels."""

    fields = np.zeros((count, channels, grid, grid), dtype=np.float32)
    channel_scales = np.asarray([1.0, 0.5, 0.25, 0.8, 0.6, 0.9], dtype=np.float32)[:channels]
    for index in range(count):
        split = int(rng.integers(4, grid - 4))
        high, low = float(rng.uniform(0.25, 0.4)), float(rng.uniform(0.0, 0.05))
        pattern = np.full((grid, grid), low, dtype=np.float32)
        if rng.uniform() < 0.5:
            pattern[:, split:] = high
        else:
            pattern[split:, :] = high
        for channel in range(channels):
            fields[index, channel] = pattern * channel_scales[channel]
    fields += rng.normal(0.0, 0.01, fields.shape).astype(np.float32)
    return torch.tensor(fields)


def shuffle_cells(fields: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Permute cells per sample (same permutation across channels).

    Preserves each image's per-channel marginals exactly — the flat code is
    unchanged — while destroying the spatial structure the hierarchy exploits.
    """

    count, channels, grid, _ = fields.shape
    shuffled = fields.clone().reshape(count, channels, grid * grid)
    for index in range(count):
        perm = torch.tensor(rng.permutation(grid * grid))
        shuffled[index] = shuffled[index][:, perm]
    return shuffled.reshape(count, channels, grid, grid)


def trained_hierarchy(train_steps: int = 400) -> tuple[CompositionHierarchy, np.random.Generator]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = PainterConfig()
    model = CompositionHierarchy(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(train_steps):
        batch = structured_fields(rng, 64)
        loss = model.training_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, rng


def test_compression_gap_prefers_structure_over_shuffled_and_blank() -> None:
    model, rng = trained_hierarchy()
    held_out = structured_fields(rng, 32)
    shuffled = shuffle_cells(held_out, rng)
    blank = torch.zeros(8, 6, 16, 16)

    gap_structured = float(model.compression_gap(held_out).mean())
    gap_shuffled = float(model.compression_gap(shuffled).mean())
    gap_blank = float(model.compression_gap(blank).mean())

    assert gap_structured > gap_shuffled + 0.2
    assert gap_structured > gap_blank
    assert gap_structured > 0.0


def test_blank_canvas_never_scores_positive_gap() -> None:
    # The flat code describes a blank canvas perfectly at the shared
    # quantization floor; the hierarchy can at best match it and still pays
    # for its latent code, trained or not.
    torch.manual_seed(3)
    untrained = CompositionHierarchy(PainterConfig())
    blank = torch.zeros(4, 6, 16, 16)
    assert float(untrained.compression_gap(blank).max()) <= 0.01


class RightHalfMassGap:
    """Stub composition model: gap proportional to right-half thickness mass."""

    def compression_gap(self, fields: torch.Tensor) -> torch.Tensor:
        grid = fields.shape[-1]
        return 50.0 * fields[:, 0, :, grid // 2 :].mean(dim=(-2, -1))


class DeterministicFootprintDynamics:
    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        footprint = action_raster[:, 0:1]
        amount = action_raster[:, 4:5]
        delta = torch.zeros_like(material)
        delta[:, 0:1] = 0.012 * amount * footprint
        next_material = material + delta
        aleatoric = torch.full_like(next_material, 2e-5)
        return next_material, aleatoric, torch.zeros_like(next_material)


def test_composition_gap_enters_spatial_efe_as_declared_terminal_preference() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16, composition_gap_precision=1.0)
    material = np.zeros((6, 16, 16), dtype=np.float32)
    belief = SpatialCanvasState(material=material, logvar=np.full_like(material, -12.0))
    efe = SpatialExpectedFreeEnergy(
        cfg,
        DeterministicFootprintDynamics(),
        TerminalCoveragePreference(cfg),
        composition=RightHalfMassGap(),
    )

    left_stroke = Policy((StrokeAction(0.1, 0.3, 0.35, 0.7, 0.12, 0.8, 1.0), StrokeAction.stop_action()))
    right_stroke = Policy((StrokeAction(0.65, 0.3, 0.9, 0.7, 0.12, 0.8, 1.0), StrokeAction.stop_action()))

    left = efe.evaluate(belief, left_stroke)
    right = efe.evaluate(belief, right_stroke)

    assert right.composition_gap > left.composition_gap
    assert right.composition_risk < left.composition_risk
    assert right.total < left.total
    assert right.total == pytest.approx(
        right.terminal_risk
        + right.ambiguity
        + right.transition_risk
        + right.transition_ambiguity
        + right.composition_risk
    )


def test_spatial_agent_trains_composition_hierarchy_online() -> None:
    cfg = PainterConfig(
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_hidden_channels=8,
        composition_latent_dim=8,
        batch_size=4,
    )
    agent = SpatialActiveInferencePainter(cfg, seed=5, device="cpu")
    assert agent.composition is not None

    rng = np.random.default_rng(2)
    stroke = StrokeAction(0.2, 0.2, 0.8, 0.8, 0.1, 0.5, 1.0)
    for _ in range(6):
        material = rng.uniform(0.0, 0.2, (cfg.spatial_material_channels, 8, 8)).astype(np.float32)
        state = SpatialCanvasState(material=material, logvar=np.full_like(material, -8.0))
        next_state = SpatialCanvasState(material=material + 0.01, logvar=np.full_like(material, -8.0))
        agent.add_transition(state, stroke, next_state)

    agent.train_dynamics(gradient_steps=1)

    assert agent.last_composition_loss is not None
    assert np.isfinite(agent.last_composition_loss)
    assert agent.belief_composition_gap() is not None
