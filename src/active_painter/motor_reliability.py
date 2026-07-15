from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PainterConfig


def execution_error_ratio_sq(realized: float, predicted: float, floor: float = 1e-4) -> float:
    """Squared ratio of realized to body-model-predicted execution error.

    This is the reliability observation: 1 means the body model's forecast of
    this motion's jitter was exact, >1 means real execution is jitterier than
    the model believes. Bounded so a single freak stroke cannot destroy a
    posterior built from many observations.
    """

    ratio = float(realized) / max(float(predicted), floor)
    return float(np.clip(ratio * ratio, 1e-2, 25.0))


@dataclass(slots=True)
class MotionReliabilityBelief:
    """Inverse-gamma precision belief over one motion family's execution jitter.

    Models the variance-inflation ratio r^2 between realized and forecast
    tracking error for a motor realization kind. The posterior mean multiplies
    that kind's forecast outcome variance inside motor expected free energy --
    a declared precision belief, not a reward -- so reliable motions are
    preferred and unproven ones carry both extra risk and resolvable
    uncertainty (information gain from trying them).
    """

    alpha: float
    beta: float

    @classmethod
    def from_prior(cls, mean: float, strength: float) -> "MotionReliabilityBelief":
        # `strength` counts pseudo-observations; alpha > 2 keeps the variance
        # of the prior finite so the epistemic term starts finite too.
        alpha = 2.0 + max(0.5, float(strength)) / 2.0
        beta = max(1e-3, float(mean)) * (alpha - 1.0)
        return cls(alpha=alpha, beta=beta)

    def expected_inflation(self) -> float:
        return float(self.beta / max(self.alpha - 1.0, 1e-6))

    def epistemic_nats(self) -> float:
        """Resolvable uncertainty about this kind's jitter, in nats.

        Half the log of one plus the squared coefficient of variation of the
        inverse-gamma posterior: Var[r^2]/E[r^2]^2 = 1/(alpha - 2).
        """

        if self.alpha <= 2.0 + 1e-6:
            return float(0.5 * np.log1p(1.0))
        return float(0.5 * np.log1p(1.0 / (self.alpha - 2.0)))

    def update(self, ratio_sq: float, weight: float = 1.0) -> None:
        w = max(0.0, float(weight))
        self.alpha += 0.5 * w
        self.beta += 0.5 * w * float(ratio_sq)

    def snapshot(self) -> dict[str, float]:
        return {"alpha": float(self.alpha), "beta": float(self.beta)}


@dataclass(slots=True)
class MotionReliabilityLedger:
    """Per-motion-kind reliability beliefs, learned from executed strokes."""

    config: PainterConfig
    beliefs: dict[str, MotionReliabilityBelief] = field(default_factory=dict)

    def belief(self, kind: str) -> MotionReliabilityBelief:
        key = str(kind or "cartesian_ik")
        found = self.beliefs.get(key)
        if found is None:
            found = MotionReliabilityBelief.from_prior(
                self.config.motor_reliability_prior_mean,
                self.config.motor_reliability_prior_strength,
            )
            self.beliefs[key] = found
        return found

    def expected_inflation(self, kind: str) -> float:
        if not self.config.motor_reliability_enabled:
            return 1.0
        return self.belief(kind).expected_inflation()

    def epistemic_nats(self, kind: str) -> float:
        if not self.config.motor_reliability_enabled:
            return 0.0
        return self.belief(kind).epistemic_nats()

    def observe(self, kind: str, ratio_sq: float, weight: float = 1.0) -> None:
        if not self.config.motor_reliability_enabled:
            return
        self.belief(kind).update(ratio_sq, weight)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            kind: {
                "expected_inflation": belief.expected_inflation(),
                "epistemic_nats": belief.epistemic_nats(),
                "observations": max(0.0, 2.0 * (belief.alpha - 2.0) - self.config.motor_reliability_prior_strength),
            }
            for kind, belief in sorted(self.beliefs.items())
        }

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {kind: belief.snapshot() for kind, belief in self.beliefs.items()}

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        for kind, state in payload.items():
            if isinstance(state, dict) and "alpha" in state and "beta" in state:
                self.beliefs[str(kind)] = MotionReliabilityBelief(
                    alpha=float(state["alpha"]),
                    beta=float(state["beta"]),
                )
