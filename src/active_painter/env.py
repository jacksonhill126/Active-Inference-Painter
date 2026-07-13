from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .config import PainterConfig


@dataclass(frozen=True, slots=True)
class StrokeAction:
    """Painting-level policy primitive.

    Coordinates and width are normalized to the canvas. `tone` is 0 for white
    and 1 for black. `amount` controls deposited material, not desired pressure.
    Pressure/contact should later be inferred conditionally by a body/contact
    generative model when this policy is realized by the robot.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    amount: float
    tone: float
    stop: bool = False

    @staticmethod
    def stop_action() -> "StrokeAction":
        return StrokeAction(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, stop=True)

    def vector(self) -> np.ndarray:
        if self.stop:
            return np.zeros(7, dtype=np.float32)
        return np.asarray(
            [self.x0, self.y0, self.x1, self.y1, self.width, self.amount, self.tone],
            dtype=np.float32,
        )


class PaintCanvasEnv:
    """Stochastic generative process for paint deposition.

    The hidden physical canvas separates thickness, persistent wetness,
    conserved bulk pigment, and top-layer surface tone. Material coverage is
    derived from thickness and is therefore not visible black/white tone.
    """

    STATE_NAMES: Final[tuple[str, ...]] = (
        "coverage",
        "mean_thickness",
        "max_thickness",
        "mean_wetness",
        "overlap_fraction",
        "mean_ground_contrast",
    )

    def __init__(self, config: PainterConfig, seed: int = 0) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        n = config.canvas_size
        self.thickness = np.zeros((n, n), dtype=np.float32)
        self.wetness = np.zeros((n, n), dtype=np.float32)
        self.black_mass = np.zeros((n, n), dtype=np.float32)
        self.surface_tone = np.zeros((n, n), dtype=np.float32)
        self.done = False
        yy, xx = np.mgrid[0:n, 0:n]
        self._xx = xx.astype(np.float32) / max(1, n - 1)
        self._yy = yy.astype(np.float32) / max(1, n - 1)

    def reset(self) -> np.ndarray:
        self.thickness.fill(0)
        self.wetness.fill(0)
        self.black_mass.fill(0)
        self.surface_tone.fill(0.0)
        self.done = False
        return self.observe()

    def coverage_field(self) -> np.ndarray:
        return 1.0 - np.exp(-self.thickness / self.cfg.thickness_scale)

    def visible_tone(self) -> np.ndarray:
        return np.clip(self.surface_tone, 0.0, 1.0)

    def observed_tone(self) -> np.ndarray:
        coverage = self.coverage_field()
        return np.clip(
            (1.0 - coverage) * self.cfg.canvas_ground_tone + coverage * self.visible_tone(),
            0.0,
            1.0,
        )

    def ground_contrast_field(self) -> np.ndarray:
        return np.abs(self.observed_tone() - self.cfg.canvas_ground_tone).astype(np.float32)

    def latent_state(self) -> np.ndarray:
        coverage = self.coverage_field()
        painted = self.thickness > 0.02
        overlap = self.thickness > self.cfg.thickness_scale
        return np.asarray(
            [
                float(coverage.mean()),
                float(self.thickness.mean()),
                float(self.thickness.max(initial=0.0)),
                float(self.wetness.mean()),
                float(overlap.mean()),
                float((self.ground_contrast_field() * painted).mean()),
            ],
            dtype=np.float32,
        )

    def observation_std(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.latent_state() if state is None else state
        smear = np.clip(0.55 * s[1] + 0.75 * s[3] + 0.35 * s[4], 0.0, 2.0)
        base = self.cfg.base_observation_std
        extra = self.cfg.smear_observation_std * smear
        scales = np.asarray([0.55, 0.7, 1.0, 0.8, 0.8, 0.9], dtype=np.float32)
        return base + extra * scales

    def observe(self) -> np.ndarray:
        state = self.latent_state()
        return state + self.rng.normal(0.0, self.observation_std(state)).astype(np.float32)

    def _stroke_footprint(self, action: StrokeAction) -> np.ndarray:
        # Distance from each pixel to the finite line segment.
        ax, ay, bx, by = action.x0, action.y0, action.x1, action.y1
        vx, vy = bx - ax, by - ay
        denom = vx * vx + vy * vy + 1e-8
        t = np.clip(((self._xx - ax) * vx + (self._yy - ay) * vy) / denom, 0.0, 1.0)
        px = ax + t * vx
        py = ay + t * vy
        d2 = (self._xx - px) ** 2 + (self._yy - py) ** 2
        sigma = max(0.006, action.width / 2.355)
        return np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)

    def step(self, action: StrokeAction) -> tuple[np.ndarray, bool, dict[str, float]]:
        if self.done:
            raise RuntimeError("Episode is finished. Call reset().")
        if action.stop:
            self.done = True
            state = self.latent_state()
            return self.observe(), True, {"coverage": float(state[0]), "stopped": 1.0}

        footprint = self._stroke_footprint(action)
        local_weight = footprint / (footprint.sum() + 1e-8)
        local_thickness = float((local_weight * self.thickness).sum())
        local_wetness = float((local_weight * self.wetness).sum())

        # Thick/wet paint makes the generative process more stochastic.
        smear_scale = 0.025 + 0.22 * np.tanh(local_thickness + 1.4 * local_wetness)
        field_noise = self.rng.normal(0.0, smear_scale, footprint.shape).astype(np.float32)
        deposited = action.amount * footprint * np.clip(1.0 + field_noise, 0.1, 2.2)

        previous_tone = self.surface_tone.copy()
        incoming_tone = float(action.tone >= 0.5)
        surface_alpha = 1.0 - np.exp(
            -deposited / max(1e-8, float(self.cfg.oil_surface_opacity_thickness))
        )
        wet_pickup = np.clip(
            float(self.cfg.oil_wet_pickup_fraction)
            * self.wetness
            / np.maximum(self.wetness + deposited, 1e-6),
            0.0,
            0.75,
        )
        loaded_tone = (1.0 - wet_pickup) * incoming_tone + wet_pickup * previous_tone
        self.surface_tone[:] = np.clip(
            (1.0 - surface_alpha) * previous_tone + surface_alpha * loaded_tone,
            0.0,
            1.0,
        )
        self.thickness += deposited
        self.black_mass += deposited * incoming_tone
        self.wetness += 0.8 * deposited
        self.wetness[:] = np.clip(self.wetness, 0.0, 3.0)

        state = self.latent_state()
        return self.observe(), False, {
            "coverage": float(state[0]),
            "mean_thickness": float(state[1]),
            "mean_wetness": float(state[3]),
            "stopped": 0.0,
        }
