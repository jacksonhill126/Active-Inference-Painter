import numpy as np
import torch

from active_painter.arm_sim import VerticalCanvas
from active_painter.config import PainterConfig
from active_painter.efe_common import project_summary_transition_support
from active_painter.env import PaintCanvasEnv, StrokeAction
from active_painter.models import ObservationModel


def test_white_paint_increases_material_coverage() -> None:
    env = PaintCanvasEnv(PainterConfig(canvas_size=32), seed=1)
    before = env.latent_state()[0]
    white = StrokeAction(0.1, 0.5, 0.9, 0.5, 0.12, 0.6, tone=0.0)
    env.step(white)
    after = env.latent_state()[0]
    assert after > before
    assert env.latent_state()[5] > 0.01


def test_wet_thick_canvas_has_higher_observation_noise() -> None:
    env = PaintCanvasEnv(PainterConfig(canvas_size=32), seed=1)
    low = env.observation_std().mean()
    action = StrokeAction(0.1, 0.5, 0.9, 0.5, 0.2, 1.3, tone=1.0)
    for _ in range(5):
        env.step(action)
    high = env.observation_std().mean()
    assert high > low


def test_summary_canvas_wetness_persists_between_marks() -> None:
    env = PaintCanvasEnv(PainterConfig(canvas_size=32), seed=1)
    wet_mark = StrokeAction(0.1, 0.5, 0.9, 0.5, 0.2, 0.8, tone=1.0)
    dry_region_mark = StrokeAction(0.1, 0.1, 0.9, 0.1, 0.05, 0.1, tone=0.0)
    env.step(wet_mark)
    before = env.wetness.copy()

    env.step(dry_region_mark)

    assert np.all(env.wetness >= before)


def test_summary_transition_support_does_not_predict_spontaneous_drying() -> None:
    current = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.2, 0.1]])
    proposed = torch.tensor([[0.3, 0.4, 0.5, 0.1, 0.3, 0.2]])

    projected = project_summary_transition_support(current, proposed)

    assert projected[0, 3] == current[0, 3]


def test_observation_model_ambiguity_is_excess_entropy() -> None:
    cfg = PainterConfig()
    model = ObservationModel(cfg)
    dry = torch.zeros(cfg.state_dim)
    wet_thick = torch.tensor([0.4, 0.7, 1.2, 0.8, 0.5, 0.2])
    assert model.ambiguity(dry).item() == 0.0
    assert model.ambiguity(wet_thick).item() > model.ambiguity(dry).item()


def test_repeated_layers_increase_thickness_without_counting_coverage_twice() -> None:
    cfg = PainterConfig(canvas_size=48, paint_presence_threshold=1e-8)
    canvas = VerticalCanvas(cfg)
    point = np.asarray([0.0, canvas.distance, 0.0])

    canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 30.0)
    first_coverage = canvas.coverage_field().copy()
    first_thickness = canvas.thickness.copy()
    canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 30.0)

    assert np.array_equal(canvas.coverage_field(), first_coverage)
    assert np.all(canvas.thickness >= first_thickness)
    assert canvas.thickness.sum() > first_thickness.sum()


def test_white_and_black_have_identical_material_coverage_semantics() -> None:
    cfg = PainterConfig(canvas_size=48)
    white = VerticalCanvas(cfg)
    black = VerticalCanvas(cfg)
    point = np.asarray([0.0, white.distance, 0.0])

    white.paint_at(point, pressure=0.8, tone=0.0, dt=1.0 / 30.0)
    black.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 30.0)

    assert np.array_equal(white.coverage_field(), black.coverage_field())
    assert white.material_coverage() == black.material_coverage()
    assert white.visible_tone().mean() < black.visible_tone().mean()


def test_visible_tone_without_material_does_not_create_coverage() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=24))
    canvas.surface_tone.fill(1.0)

    assert canvas.visible_tone().mean() == 1.0
    assert canvas.material_coverage() == 0.0


def test_canvas_clear_resets_material_but_preserves_fixed_grain() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=32, canvas_grain_seed=7))
    point = np.asarray([0.0, canvas.distance, 0.0])
    canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 30.0)
    grain_before = canvas.grain.copy()

    canvas.clear()

    assert not canvas.thickness.any()
    assert not canvas.wetness.any()
    assert not canvas.black_mass.any()
    assert not canvas.surface_tone.any()
    assert np.array_equal(canvas.grain, grain_before)


def test_material_presence_threshold_and_contact_stiffness_are_configurable() -> None:
    low_threshold = VerticalCanvas(
        PainterConfig(canvas_size=32, paint_presence_threshold=1e-8),
        contact_stiffness=20.0,
    )
    high_threshold = VerticalCanvas(
        PainterConfig(canvas_size=32, paint_presence_threshold=1.0),
        contact_stiffness=80.0,
    )
    point = np.asarray([0.0, low_threshold.distance, 0.0])
    for canvas in (low_threshold, high_threshold):
        canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 120.0)

    assert low_threshold.material_coverage() > high_threshold.material_coverage()
    tip = np.asarray([0.0, low_threshold.distance + 0.1, 0.0])
    assert high_threshold.contact_from_tip(tip).force > low_threshold.contact_from_tip(tip).force
