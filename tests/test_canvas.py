import numpy as np
import torch

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
