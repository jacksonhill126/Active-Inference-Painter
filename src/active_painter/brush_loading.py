from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import PainterConfig
from .env import StrokeAction
from .inference import VFEComponents
from .policies import BrushPreparationPolicy


BRUSH_LOADING_MODEL_VERSION = "brush-loading-belief-v0"
BRUSH_LOADING_CALIBRATION_STATUS = (
    "provisional_simulation_only_not_hardware_calibrated"
)
BRUSH_MICROSTRUCTURE_PRIOR_VERSION = "brush-microstructure-prior-v0"


@dataclass(frozen=True, slots=True)
class BrushLoadBelief:
    """Compact posterior over one dedicated physical brush.

    Load is normalized to [0, 1]. Black fraction is also in [0, 1], so the
    white fraction is its complement. These are latent beliefs; exact process
    brush state is not part of this record.
    """

    load_mean: float
    load_variance: float
    black_fraction_mean: float
    black_fraction_variance: float
    revision: int = 0
    inference_model_id: str = BRUSH_LOADING_MODEL_VERSION
    calibration_status: str = BRUSH_LOADING_CALIBRATION_STATUS

    def __post_init__(self) -> None:
        if not 0.0 <= self.load_mean <= 1.0:
            raise ValueError("load_mean must lie in [0, 1].")
        if not 0.0 <= self.black_fraction_mean <= 1.0:
            raise ValueError("black_fraction_mean must lie in [0, 1].")
        if self.load_variance < 0.0 or self.black_fraction_variance < 0.0:
            raise ValueError("Brush belief variances must be non-negative.")
        if not all(
            math.isfinite(value)
            for value in (
                self.load_mean,
                self.load_variance,
                self.black_fraction_mean,
                self.black_fraction_variance,
            )
        ):
            raise ValueError("Brush belief moments must be finite.")
        if self.revision < 0:
            raise ValueError("revision must be non-negative.")
        if not self.inference_model_id or not self.calibration_status:
            raise ValueError(
                "inference_model_id and calibration_status must be non-empty."
            )


@dataclass(frozen=True, slots=True)
class BrushPreparationEFE:
    """EFE terms over conditional mark-material and pigment outcomes."""

    policy: BrushPreparationPolicy
    total: float
    material_risk: float
    pigment_risk: float
    ambiguity: float
    predicted_deposition_mean: float
    predicted_deposition_variance: float
    predicted_black_fraction_mean: float
    predicted_black_fraction_variance: float
    log_policy_prior: float
    approximation: str = (
        "diagonal Gaussian conditional mark-outcome EFE; desired deposition "
        "and pigment are inherited from the selected painting action; reload "
        "has an explicit policy prior and is not selected by a hard threshold"
    )


@dataclass(frozen=True, slots=True)
class BrushPreparationInference:
    selected: BrushPreparationPolicy
    posterior: tuple[tuple[BrushPreparationPolicy, float], ...]
    components: tuple[BrushPreparationEFE, ...]
    log_evidence: float


class BrushLoadingModel:
    """Transition, likelihood, and preparation-policy model for brush load."""

    def __init__(self, config: PainterConfig) -> None:
        self.config = config
        self.last_vfe: VFEComponents | None = None

    def unloaded_belief(self, selected_tone: float) -> BrushLoadBelief:
        tone = float(selected_tone >= 0.5)
        return BrushLoadBelief(
            load_mean=0.0,
            load_variance=self.config.brush_initial_load_std**2,
            black_fraction_mean=tone,
            black_fraction_variance=self.config.brush_reload_mixture_std**2,
            inference_model_id=BRUSH_LOADING_MODEL_VERSION,
            calibration_status=BRUSH_LOADING_CALIBRATION_STATUS,
        )

    def reload_transition(
        self,
        belief: BrushLoadBelief,
        selected_tone: float,
    ) -> BrushLoadBelief:
        """Reload to a full, uniformly selected-color brush distribution."""

        tone = float(selected_tone >= 0.5)
        return BrushLoadBelief(
            load_mean=1.0,
            load_variance=self.config.brush_reload_load_std**2,
            black_fraction_mean=tone,
            black_fraction_variance=self.config.brush_reload_mixture_std**2,
            revision=belief.revision + 1,
            inference_model_id=BRUSH_LOADING_MODEL_VERSION,
            calibration_status=BRUSH_LOADING_CALIBRATION_STATUS,
        )

    def stroke_transition(
        self,
        belief: BrushLoadBelief,
        action: StrokeAction,
    ) -> BrushLoadBelief:
        """Approximate load depletion when no camera patch is assimilated.

        Canvas pickup is represented as mixture uncertainty here. Its posterior
        mean must later be updated by the local camera/material likelihood.
        """

        depletion = (
            self.config.brush_belief_depletion_per_mark
            * float(np.clip(action.amount, 0.0, 1.0))
        )
        return BrushLoadBelief(
            load_mean=float(np.clip(belief.load_mean - depletion, 0.0, 1.0)),
            load_variance=float(
                min(
                    0.25,
                    belief.load_variance
                    + self.config.brush_load_process_std**2,
                )
            ),
            black_fraction_mean=belief.black_fraction_mean,
            black_fraction_variance=float(
                min(
                    0.25,
                    belief.black_fraction_variance
                    + self.config.brush_mixture_process_std**2,
                )
            ),
            revision=belief.revision + 1,
            inference_model_id=BRUSH_LOADING_MODEL_VERSION,
            calibration_status=BRUSH_LOADING_CALIBRATION_STATUS,
        )

    def infer_load_from_mark(
        self,
        prior: BrushLoadBelief,
        action: StrokeAction,
        observed_deposition: float,
        observation_variance: float,
    ) -> BrushLoadBelief:
        """Conjugate update for p(o_mark | load, selected mark amount).

        ``observed_deposition`` is a future camera-derived local-patch
        observation, not exact canvas thickness. The method exists now so the
        latent has an explicit likelihood rather than becoming controller
        bookkeeping.
        """

        if not math.isfinite(observed_deposition):
            raise ValueError("observed_deposition must be finite.")
        if not math.isfinite(observation_variance) or observation_variance <= 0.0:
            raise ValueError("observation_variance must be finite and positive.")
        coefficient = max(1e-4, float(np.clip(action.amount, 0.0, 1.0)))
        prior_variance = max(prior.load_variance, 1e-8)
        posterior_variance = 1.0 / (
            1.0 / prior_variance + coefficient * coefficient / observation_variance
        )
        posterior_mean = posterior_variance * (
            prior.load_mean / prior_variance
            + coefficient * observed_deposition / observation_variance
        )
        posterior_mean = float(np.clip(posterior_mean, 0.0, 1.0))
        complexity = 0.5 * (
            math.log(prior_variance / posterior_variance)
            + (
                posterior_variance
                + (posterior_mean - prior.load_mean) ** 2
            )
            / prior_variance
            - 1.0
        )
        negative_log_likelihood = 0.5 * (
            math.log(2.0 * math.pi * observation_variance)
            + (
                (observed_deposition - coefficient * posterior_mean) ** 2
                + coefficient * coefficient * posterior_variance
            )
            / observation_variance
        )
        self.last_vfe = VFEComponents(
            total=complexity + negative_log_likelihood,
            complexity=complexity,
            negative_log_likelihood=negative_log_likelihood,
            expected_log_likelihood=-negative_log_likelihood,
            units="nats",
            approximation=(
                "conjugate scalar Gaussian load update under a local "
                "camera-derived mark-deposition likelihood"
            ),
        )
        return BrushLoadBelief(
            load_mean=posterior_mean,
            load_variance=posterior_variance,
            black_fraction_mean=prior.black_fraction_mean,
            black_fraction_variance=prior.black_fraction_variance,
            revision=prior.revision + 1,
            inference_model_id=(
                f"{BRUSH_LOADING_MODEL_VERSION}:"
                "camera-derived-mark-deposition-likelihood-v0"
            ),
            calibration_status=BRUSH_LOADING_CALIBRATION_STATUS,
        )

    def infer_preparation(
        self,
        belief: BrushLoadBelief,
        action: StrokeAction,
    ) -> BrushPreparationInference:
        """Infer preserve versus reload under conditional outcome preferences."""

        selected_tone = float(action.tone >= 0.5)
        policies = (
            BrushPreparationPolicy("preserve", selected_tone),
            BrushPreparationPolicy("reload", selected_tone),
        )
        reload_probability = float(
            np.clip(self.config.brush_reload_policy_prior, 1e-6, 1.0 - 1e-6)
        )
        amount_preference_variance = max(
            self.config.brush_mark_amount_preference_std**2,
            1e-8,
        )
        pigment_preference_variance = max(
            self.config.brush_mark_pigment_preference_std**2,
            1e-8,
        )
        deposition_likelihood_variance = max(
            self.config.brush_deposition_likelihood_std**2,
            1e-8,
        )
        mixture_likelihood_variance = max(
            self.config.brush_mixture_likelihood_std**2,
            1e-8,
        )
        intended_deposition = float(np.clip(action.amount, 0.0, 1.0))
        components: list[BrushPreparationEFE] = []
        logits: list[float] = []
        for policy in policies:
            predicted = (
                belief
                if policy.kind == "preserve"
                else self.reload_transition(belief, selected_tone)
            )
            deposition_mean = intended_deposition * predicted.load_mean
            deposition_variance = (
                intended_deposition**2 * predicted.load_variance
                + deposition_likelihood_variance
            )
            pigment_variance = (
                predicted.black_fraction_variance
                + mixture_likelihood_variance
            )
            material_risk = 0.5 * (
                (deposition_mean - intended_deposition) ** 2
                + deposition_variance
            ) / amount_preference_variance
            pigment_risk = 0.5 * (
                (predicted.black_fraction_mean - selected_tone) ** 2
                + pigment_variance
            ) / pigment_preference_variance
            ambiguity = 0.5 * (
                math.log1p(
                    deposition_likelihood_variance
                    / amount_preference_variance
                )
                + math.log1p(
                    mixture_likelihood_variance
                    / pigment_preference_variance
                )
            )
            total = (
                self.config.brush_material_risk_precision * material_risk
                + self.config.brush_pigment_risk_precision * pigment_risk
                + self.config.brush_ambiguity_precision * ambiguity
            )
            log_prior = math.log(
                reload_probability
                if policy.kind == "reload"
                else 1.0 - reload_probability
            )
            components.append(
                BrushPreparationEFE(
                    policy=policy,
                    total=total,
                    material_risk=material_risk,
                    pigment_risk=pigment_risk,
                    ambiguity=ambiguity,
                    predicted_deposition_mean=deposition_mean,
                    predicted_deposition_variance=deposition_variance,
                    predicted_black_fraction_mean=predicted.black_fraction_mean,
                    predicted_black_fraction_variance=pigment_variance,
                    log_policy_prior=log_prior,
                )
            )
            logits.append(
                log_prior - self.config.brush_policy_precision * total
            )
        maximum = max(logits)
        weights = np.exp(np.asarray(logits) - maximum)
        posterior = weights / weights.sum()
        selected_index = int(np.argmax(posterior))
        return BrushPreparationInference(
            selected=policies[selected_index],
            posterior=tuple(
                (policy, float(probability))
                for policy, probability in zip(policies, posterior, strict=True)
            ),
            components=tuple(components),
            log_evidence=maximum + float(math.log(float(weights.sum()))),
        )
