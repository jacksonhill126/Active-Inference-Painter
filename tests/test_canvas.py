import torch

from active_painter.config import PainterConfig
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


def test_observation_model_ambiguity_is_excess_entropy() -> None:
    cfg = PainterConfig()
    model = ObservationModel(cfg)
    dry = torch.zeros(cfg.state_dim)
    wet_thick = torch.tensor([0.4, 0.7, 1.2, 0.8, 0.5, 0.2])
    assert model.ambiguity(dry).item() == 0.0
    assert model.ambiguity(wet_thick).item() > model.ambiguity(dry).item()
