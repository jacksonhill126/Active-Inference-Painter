"""Tests for the compression-gap composition hierarchy.

The declared claim: gap(s) = ELBO_hier(s) - max_m log p_m(s), where the
opponent is the BEST member of a declared, hand-written, parameter-free
baseline family (an iid-cell Gaussian and a 3x3 hollow-neighbourhood local
Markov code). Four invariants are pinned here:

- a blank canvas never scores a positive gap (both members sit exactly on the
  shared quantization floor, so the baseline is already perfect);
- iid noise never scores a positive gap (the iid member wins the family max);
- a structured canvas the hierarchy trained on scores strictly above a
  cell-shuffled version of itself (same marginals, structure destroyed);
- a LOCALLY SMOOTH but globally unstructured canvas scores <= 0, which the old
  iid-only baseline failed.

Note on scale: `structured_fields` and the helpers beside it use channel_scales
[1.0, 0.5, 0.25, 0.8, 0.6, 0.9], which are NOT the realistic per-channel scales
the recorded 16x16x6 / 1500-step measurements use (thickness-like channels there
live at ~0.002-0.008). The local member gains materially more on these larger
amplitudes, so gap numbers from this file are not comparable to those recorded
measurements.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from active_painter import composition
from active_painter.canvas_hierarchy import HierarchicalCanvasModel
from active_painter.composition import (
    BASELINE_CEILING_NATS,
    CompositionHierarchy,
    baseline_log_likelihood,
    flat_log_likelihood,
    local_smoothness_log_likelihood,
)
from active_painter.config import (
    LEGACY_MATERIAL_HIERARCHY_DIAGNOSTIC_ID,
    M1_FORMAL_POLICY_BASELINE_ID,
    PainterConfig,
    painting_policy_profile_id,
)
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


CHANNEL_SCALES = np.asarray([1.0, 0.5, 0.25, 0.8, 0.6, 0.9], dtype=np.float32)


def soft_blob_fields(
    rng: np.random.Generator, count: int, channels: int = 6, grid: int = 16
) -> torch.Tensor:
    """Locally smooth, globally unstructured: one soft Gaussian blob per image.

    Every cell is well predicted by its immediate neighbours, so the local
    baseline member explains this almost perfectly, but there is no long-range
    organisation for a composition code to earn credit from.
    """

    rows, cols = np.mgrid[0:grid, 0:grid]
    fields = np.zeros((count, channels, grid, grid), dtype=np.float32)
    for index in range(count):
        center_row = float(rng.uniform(4, grid - 4))
        center_col = float(rng.uniform(4, grid - 4))
        sigma = float(rng.uniform(2.5, 4.0))
        pattern = 0.4 * np.exp(
            -((rows - center_row) ** 2 + (cols - center_col) ** 2) / (2.0 * sigma**2)
        )
        for channel in range(channels):
            fields[index, channel] = pattern * CHANNEL_SCALES[channel]
    fields += rng.normal(0.0, 0.01, fields.shape).astype(np.float32)
    return torch.tensor(fields)


def lowpass_noise_fields(
    rng: np.random.Generator, count: int, channels: int = 6, grid: int = 16
) -> torch.Tensor:
    """Locally smooth, globally unstructured: low-pass-filtered noise.

    Three 5x5 box passes with replicate padding leave strong local correlation
    and no long-range structure, which is exactly the case the iid-only
    baseline used to award a positive gap.
    """

    pattern = rng.uniform(0.0, 0.4, (count, grid, grid)).astype(np.float32)
    for _ in range(3):
        padded = np.pad(pattern, ((0, 0), (2, 2), (2, 2)), mode="edge")
        smoothed = np.zeros_like(pattern)
        for row_offset in range(5):
            for col_offset in range(5):
                smoothed += padded[
                    :, row_offset : row_offset + grid, col_offset : col_offset + grid
                ]
        pattern = smoothed / 25.0
    low, high = float(pattern.min()), float(pattern.max())
    pattern = (pattern - low) / max(high - low, 1e-8) * 0.4

    fields = np.zeros((count, channels, grid, grid), dtype=np.float32)
    for channel in range(channels):
        fields[:, channel] = pattern * CHANNEL_SCALES[channel]
    fields += rng.normal(0.0, 0.01, fields.shape).astype(np.float32)
    return torch.tensor(fields)


def shuffle_cells(fields: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Permute cells per sample (same permutation across channels).

    Preserves each image's per-channel marginals exactly, so the iid member is
    provably invariant and wins the family maximum; the local member's score
    DROPS because the neighbourhood prediction is destroyed. The baseline as a
    whole is therefore unchanged, which is what makes this a valid
    marginal-preserving null model.
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


@pytest.fixture(scope="module")
def mixture_trained_hierarchy() -> tuple[CompositionHierarchy, np.random.Generator]:
    """One hierarchy trained on BOTH bands and soft blobs, shared by two tests.

    Training on the blobs too is the adversarial case: the hierarchy has every
    opportunity to model them, so a non-positive blob gap cannot be dismissed
    as the hierarchy simply never having seen that input. 600 steps, not 400:
    at 400 the true-positive bands gap is too close to the separation margin.
    """

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = CompositionHierarchy(PainterConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(600):
        batch = torch.cat([structured_fields(rng, 32), soft_blob_fields(rng, 32)], dim=0)
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

    # The gap must be earned by spatial organisation, not by per-channel
    # marginals, and the shuffle must remain a valid null model under the
    # richer family. Three properties establish that. The local member wins the
    # family maximum on structured input:
    assert float(baseline_log_likelihood(held_out).mean()) > float(flat_log_likelihood(held_out).mean())
    # it LOSES on the shuffle, because the neighbourhood prediction is destroyed:
    assert float(local_smoothness_log_likelihood(shuffled).mean()) < float(
        flat_log_likelihood(shuffled).mean()
    )
    # and the iid member is exactly invariant under a permutation of cells, so
    # the family maximum falls back to it and the baseline is unmoved by the
    # shuffle (up to per-image tie-breaking, measured at 7.7e-5 nats).
    assert float(flat_log_likelihood(shuffled).mean()) == pytest.approx(
        float(flat_log_likelihood(held_out).mean()), abs=1e-5
    )
    assert float(baseline_log_likelihood(shuffled).mean()) == pytest.approx(
        float(flat_log_likelihood(shuffled).mean()), abs=1e-3
    )


def test_blank_canvas_never_scores_positive_gap() -> None:
    # The baseline family describes a blank canvas perfectly at the shared
    # quantization floor; the hierarchy can at best match it and still pays
    # for its latent code, trained or not.
    torch.manual_seed(3)
    untrained = CompositionHierarchy(PainterConfig())
    blank = torch.zeros(4, 6, 16, 16)
    assert float(untrained.compression_gap(blank).max()) <= 0.01

    # Both members saturate identically on a blank canvas, so adding the local
    # member cannot move the blank invariant: it is a member tie exactly on the
    # shared ceiling. The same holds for any spatially constant field, which is
    # why "spatially trivial" is exact here rather than approximate.
    assert float(flat_log_likelihood(blank).mean()) == pytest.approx(BASELINE_CEILING_NATS)
    assert float(local_smoothness_log_likelihood(blank).mean()) == pytest.approx(BASELINE_CEILING_NATS)
    assert float(baseline_log_likelihood(blank).mean()) == pytest.approx(
        float(flat_log_likelihood(blank).mean())
    )
    constant = torch.full((4, 6, 16, 16), 0.5)
    assert float(local_smoothness_log_likelihood(constant).mean()) == pytest.approx(BASELINE_CEILING_NATS)


def test_iid_noise_never_scores_positive_gap() -> None:
    """Declared invariant 2: nothing local or global to exploit in iid noise."""

    model, rng = trained_hierarchy()
    scatter = torch.tensor(
        (rng.uniform(0.0, 0.4, (16, 6, 16, 16)) * CHANNEL_SCALES[None, :, None, None]).astype(
            np.float32
        )
    )

    assert float(model.compression_gap(scatter).max()) <= 0.0
    # The local member LOSES on iid input (a neighbourhood mean predicts
    # nothing there), so the family maximum falls back to the iid member on
    # every image and this invariant is inherited exactly, not approximately,
    # from the previous baseline (measured difference 0.0).
    assert float(local_smoothness_log_likelihood(scatter).mean()) < float(
        flat_log_likelihood(scatter).mean()
    )
    assert float(baseline_log_likelihood(scatter).mean()) == pytest.approx(
        float(flat_log_likelihood(scatter).mean()), abs=1e-6
    )


def test_locally_smooth_but_globally_unstructured_canvas_scores_non_positive(
    mixture_trained_hierarchy: tuple[CompositionHierarchy, np.random.Generator],
) -> None:
    """The point of the baseline family: smoothness alone must not earn the gap.

    A soft blob and low-pass noise are locally predictable but carry no
    long-range organisation. The local member explains them, so the hierarchy
    earns no evidence for them even though it was trained on blobs.
    """

    model, _ = mixture_trained_hierarchy
    eval_rng = np.random.default_rng(101)
    blob = soft_blob_fields(eval_rng, 32)
    lowpass = lowpass_noise_fields(eval_rng, 32)

    assert float(model.compression_gap(blob).mean()) <= 0.0
    assert float(model.compression_gap(blob).max()) <= 0.01
    assert float(model.compression_gap(lowpass).mean()) <= 0.0

    # Necessity proof: the old iid-only baseline awards this canvas a large
    # positive gap, i.e. it credits the composition preference for mere local
    # smoothness. On this mixture-trained hierarchy the old gap also ranks the
    # blob at or above structured bands, which is the finding that motivates
    # the family (measured, not asserted: the bands margin is too tight to pin).
    with torch.no_grad():
        old_gap_blob = model.elbo(blob, sample=False) - flat_log_likelihood(blob)
    assert float(old_gap_blob.mean()) > 0.5


def test_new_baseline_still_separates_long_range_structure_from_local_smoothness(
    mixture_trained_hierarchy: tuple[CompositionHierarchy, np.random.Generator],
) -> None:
    """The stronger opponent must not collapse the true positive."""

    model, _ = mixture_trained_hierarchy
    eval_rng = np.random.default_rng(202)
    bands = structured_fields(eval_rng, 32)
    blob = soft_blob_fields(eval_rng, 32)

    gap_bands = float(model.compression_gap(bands).mean())
    gap_blob = float(model.compression_gap(blob).mean())

    assert gap_bands > 0.0
    assert gap_bands > gap_blob + 0.2


def test_production_hierarchy_shares_the_baseline_family() -> None:
    """The gap the live agent actually uses is the family gap, not the iid gap.

    Nothing in src/ instantiates CompositionHierarchy; the spatial agent builds
    HierarchicalCanvasModel. Both must resolve to the one shared baseline
    implementation, or every other invariant here could pass while the
    production preference kept the old opponent.
    """

    torch.manual_seed(7)
    cfg = PainterConfig(spatial_grid_size=8, composition_hidden_channels=8)
    model = HierarchicalCanvasModel(cfg)
    rng = np.random.default_rng(4)
    fields = torch.tensor(rng.uniform(0.0, 0.3, (3, 6, 8, 8)).astype(np.float32))

    assert float(model.baseline_log_likelihood(fields).mean()) == pytest.approx(
        float(composition.baseline_log_likelihood(fields).mean())
    )
    with torch.no_grad():
        expected = model.canvas_elbo(fields) - composition.baseline_log_likelihood(fields)
    assert float(model.compression_gap(fields).mean()) == pytest.approx(float(expected.mean()))
    assert float(model.compression_gap(torch.zeros(4, 6, 8, 8)).max()) <= 0.01


def test_local_baseline_flag_restores_iid_only_baseline() -> None:
    """The hand-written alternative stays available and the difference is measurable."""

    torch.manual_seed(11)
    on_cfg = PainterConfig(spatial_grid_size=16, composition_hidden_channels=8)
    off_cfg = PainterConfig(
        spatial_grid_size=16,
        composition_hidden_channels=8,
        composition_local_baseline_enabled=False,
    )
    on_model = HierarchicalCanvasModel(on_cfg)
    off_model = HierarchicalCanvasModel(off_cfg)
    # Identical weights, so only the reference code differs.
    off_model.load_state_dict(on_model.state_dict())

    blob = soft_blob_fields(np.random.default_rng(9), 8)
    with torch.no_grad():
        elbo = on_model.canvas_elbo(blob)
    gap_on = float(on_model.compression_gap(blob).mean())
    gap_off = float(off_model.compression_gap(blob).mean())

    assert gap_off == pytest.approx(float((elbo - flat_log_likelihood(blob)).mean()))
    assert gap_on == pytest.approx(float((elbo - composition.baseline_log_likelihood(blob)).mean()))
    assert gap_on < gap_off

    # The new flag is strictly narrower than the existing precision gate: it
    # must never resurrect the composition term when the precision is zero.
    zero_precision = PainterConfig(
        canvas_size=32,
        spatial_grid_size=16,
        composition_gap_precision=0.0,
        composition_local_baseline_enabled=True,
    )
    material = np.zeros((6, 16, 16), dtype=np.float32)
    belief = SpatialCanvasState(material=material, logvar=np.full_like(material, -12.0))
    efe = SpatialExpectedFreeEnergy(
        zero_precision,
        DeterministicFootprintDynamics(),
        TerminalCoveragePreference(zero_precision),
        composition=RightHalfMassGap(),
    )
    stroke = Policy((StrokeAction(0.65, 0.3, 0.9, 0.7, 0.12, 0.8, 1.0), StrokeAction.stop_action()))
    components = efe.evaluate(belief, stroke)
    assert components.composition_gap == 0.0
    assert components.composition_risk == 0.0


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
    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=16,
        composition_enabled=True,
        composition_gap_precision=1.0,
    )
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
    # Attribution: the stored (post-weighted) composition risk must be exactly
    # -(raw gap) * gamma_composition * normalizer_composition, read back off the
    # component's own recorded weight fields. Without this the dataclass could
    # record a precision it did not actually apply.
    assert right.composition_gap_precision == pytest.approx(cfg.composition_gap_precision)
    assert right.composition_gap_normalizer == 1.0
    assert right.composition_gap_normalizer_name == "nats_per_cell_channel_all_material_channels"
    assert right.composition_risk == pytest.approx(
        -right.composition_gap
        * right.composition_gap_precision
        * right.composition_gap_normalizer
    )


def test_composition_structural_flag_is_separate_from_its_precision() -> None:
    """A learned precision must never construct or destroy the hierarchy.

    `composition_enabled` is declared structure; `composition_gap_precision` is a
    belief's prior mean. With the structural flag off the modality is exactly
    0.0 no matter what the precision says.
    """

    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=16,
        composition_gap_precision=1.0,
        composition_enabled=False,
    )
    material = np.zeros((6, 16, 16), dtype=np.float32)
    belief = SpatialCanvasState(material=material, logvar=np.full_like(material, -12.0))
    efe = SpatialExpectedFreeEnergy(
        cfg,
        DeterministicFootprintDynamics(),
        TerminalCoveragePreference(cfg),
        composition=RightHalfMassGap(),
    )
    stroke = Policy((StrokeAction(0.65, 0.3, 0.9, 0.7, 0.12, 0.8, 1.0), StrokeAction.stop_action()))
    components = efe.evaluate(belief, stroke)
    assert components.composition_gap == 0.0
    assert components.composition_risk == 0.0


def test_default_m1_policy_baseline_disables_legacy_material_hierarchy_terms() -> None:
    cfg = PainterConfig()

    assert painting_policy_profile_id(cfg) == M1_FORMAL_POLICY_BASELINE_ID
    assert not cfg.composition_enabled
    assert cfg.composition_gap_precision == 0.0
    assert cfg.canvas_latent_transition_precision == 0.0
    assert cfg.relational_transition_precision == 0.0
    assert not cfg.passage_trajectory_enabled
    assert not cfg.gap_progress_stop_enabled

    agent = SpatialActiveInferencePainter(cfg, seed=3, device="cpu")
    assert agent.composition is None
    assert agent.policy_proposal is None


def test_legacy_material_hierarchy_requires_an_explicit_diagnostic_opt_in() -> None:
    cfg = PainterConfig(
        composition_enabled=True,
        composition_gap_precision=1.0,
        canvas_latent_transition_precision=0.30,
        relational_transition_precision=0.30,
        passage_trajectory_enabled=True,
        gap_progress_stop_enabled=True,
    )

    assert (
        painting_policy_profile_id(cfg)
        == LEGACY_MATERIAL_HIERARCHY_DIAGNOSTIC_ID
    )
    agent = SpatialActiveInferencePainter(cfg, seed=3, device="cpu")
    assert agent.composition is not None
    assert agent.policy_proposal is not None


def test_spatial_agent_trains_composition_hierarchy_online() -> None:
    cfg = PainterConfig(
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_hidden_channels=8,
        composition_latent_dim=8,
        composition_enabled=True,
        composition_gap_precision=1.0,
        canvas_latent_transition_precision=0.30,
        relational_transition_precision=0.30,
        passage_trajectory_enabled=True,
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
