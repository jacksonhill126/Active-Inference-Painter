from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .config import PainterConfig
from .policies import MotorPrimitiveLatent, Policy
from .precision_beliefs import ModalityWeights


# Execution-fidelity outcome channels: the ones the learned per-kind
# reliability belief is fitted on (and therefore the ones it inflates).
_FIDELITY_CHANNEL_PREFIXES = ("path_error", "pressure_error", "target_error_", "contact_loss")

# Scalar or batched-tensor EFE term, so the same identity serves the per-policy
# and batched evaluators without importing torch into this module.
_EFETerm = TypeVar("_EFETerm")


@dataclass(frozen=True, slots=True)
class MotorEFETerms:
    """Precision-weighted EFE terms for proprioceptive outcome modalities.

    The first four fields keep their order because they are serialized by
    ``dataclasses.asdict`` into the driver telemetry payload. ``risk`` is the
    pragmatic term, ``ambiguity`` is logged for telemetry only, and
    ``epistemic_value`` is the logged sum of the two quantities that are
    actually subtracted from expected free energy.
    """

    risk: float
    ambiguity: float
    epistemic_value: float
    approximation: str
    # Precision-weighted I(s; o) for the proprioceptive channel. Subtracted from
    # G exactly ONCE, via `epistemic_value`; because -I(s;o) already equals
    # E_q(s)[H[p(o|s)]] - H[q(o)], it carries the canonical ambiguity
    # contribution and `ambiguity` must not be added on top of it.
    mutual_information: float = 0.0
    # Precision-weighted PARAMETER novelty over the learned inverse-gamma
    # motion-reliability belief: the analogue of Dirichlet novelty in the
    # reference implementation, and a genuinely separate information gain.
    reliability_novelty: float = 0.0
    # Unscaled H[q(o)] in nats. Not a summand anywhere; logged so that the
    # canonical KL risk (= pragmatic - H[q(o)]) is derivable from telemetry
    # without re-deriving it from the forecast.
    forecast_outcome_entropy: float = 0.0
    # Declared unit normalization applied to the three summed terms above, so
    # the modality is commensurable with the material modalities. Recorded here
    # rather than inferred, so `raw * precision * normalizer == stored` is
    # checkable from telemetry alone.
    normalizer: float = 1.0
    normalizer_name: str = ""
    modality_precision: float = 1.0


def motor_efe_contribution(
    motor_risk: _EFETerm,
    motor_epistemic_value: _EFETerm,
) -> _EFETerm:
    """Motor modality contribution to G: pragmatic - I(s;o) - N_reliability.

    Deliberately omits the logged ``MotorEFETerms.ambiguity``: likelihood
    entropy in excess of the preference scale is a fourth, incommensurate
    quantity, and adding it alongside ``- mutual_information`` would double
    count ambiguity (see :func:`motor_efe_terms`). Single source of truth for
    every EFE total site, and accepts floats or tensors so the batched and
    scalar evaluators can share it.
    """

    return motor_risk - motor_epistemic_value


def motor_realization_policy_alternatives(policy: Policy, config: PainterConfig) -> list[Policy]:
    """Expand a canvas policy into first-stroke motor realization latents."""

    if (
        not config.motor_planning_enabled
        or policy.actions[0].stop
        or policy.motor_primitive is not None
    ):
        return [policy]
    kinds = list(config.motor_realization_kinds)[: max(1, config.motor_realization_candidate_limit)]
    if not kinds:
        kinds = ["cartesian_ik"]
    alternatives: list[Policy] = []
    for kind in dict.fromkeys(str(value) for value in kinds):
        alternatives.append(
            Policy(
                policy.actions,
                passage=policy.passage,
                passage_plan=policy.passage_plan,
                motor_primitive=_motor_primitive(kind, config),
                brush_preparation=policy.brush_preparation,
                passage_start_index=policy.passage_start_index,
            )
        )
    return alternatives


def motor_policy_log_prior(policy: Policy, config: PainterConfig) -> float:
    """Declared log p(pi_motor) term.

    All enabled primitive kinds are equiprobable for now. Keeping this explicit
    prevents motor outcome preferences from being hidden inside candidate
    generation or arbitrary rewards.
    """

    if not config.motor_planning_enabled or policy.motor_primitive is None:
        return 0.0
    kind_count = max(1, min(len(config.motor_realization_kinds), config.motor_realization_candidate_limit))
    return -float(np.log(kind_count))


def motor_realization_log_evidence(
    expected_free_energies: list[float],
    log_priors: list[float],
    policy_precision: float,
) -> tuple[float, np.ndarray]:
    """Marginal log evidence and q(motor realization | painting policy)."""

    if not expected_free_energies or len(expected_free_energies) != len(log_priors):
        raise ValueError("Motor EFE values and priors must be non-empty and aligned.")
    logits = np.asarray(log_priors, dtype=np.float64) - float(policy_precision) * np.asarray(
        expected_free_energies,
        dtype=np.float64,
    )
    maximum = float(np.max(logits))
    weights = np.exp(logits - maximum)
    normalizer = float(weights.sum())
    posterior = weights / max(normalizer, 1e-300)
    return maximum + float(np.log(max(normalizer, 1e-300))), posterior


def motor_efe_terms(
    forecast,
    config: PainterConfig,
    reliability_inflation: float = 1.0,
    reliability_epistemic_nats: float = 0.0,
    weights: ModalityWeights | None = None,
) -> MotorEFETerms:
    """EFE over a diagonal proprioceptive predictive density, in nats.

    ``risk`` is the PRAGMATIC term E_q[-log p*(o)] under declared zero-centered
    homeostatic outcome preferences, with policy-independent Gaussian
    normalizers omitted. It is deliberately *not* a KL: the -H[q(o)] term is
    absent. The motor contribution to expected free energy is therefore

        G_motor = risk - epistemic_value
                = pragmatic - I(s; o) - N_reliability

    and because -I(s; o) = E_q(s)[H[p(o|s)]] - H[q(o)], that expression is
    algebraically identical to the canonical `KL risk + ambiguity - novelty`.
    Subtracting the mutual information once therefore already accounts for the
    canonical ambiguity contribution.

    ``ambiguity`` is a different quantity: likelihood entropy in excess of each
    preference scale, measured against the preference variance rather than in
    absolute nats. It is logged for telemetry so the modality stays auditable,
    and it is never a summand in G -- adding it alongside -I(s; o) double counts
    ambiguity. ``forecast_outcome_entropy`` is logged for the same reason, so a
    reader can reconstruct the canonical KL risk as
    ``risk - forecast_outcome_entropy`` at unit risk precision.

    ``reliability_inflation`` is the learned per-motion-kind precision belief:
    the posterior mean of the squared realized-vs-forecast execution error
    ratio. A kind that is r-times jitterier than the body model realizes
    r-times the tracking error, so the belief scales the *expected squared
    error* (mean^2 plus variance) of the execution-fidelity channels -- path,
    pressure, target tracking, contact loss -- the same quantities the belief
    is fitted on. Effort channels (current, torque, velocity, acceleration,
    limit proximity) are not inflated. ``reliability_epistemic_nats`` is the
    belief's resolvable uncertainty, credited as parameter novelty for executing
    (and thereby measuring) the kind, under its own declared precision.

    ``weights`` carries the modality-level Gamma precision belief mean and the
    declared unit normalizer. The three summed terms are raw sums over all
    proprioceptive channels, so without normalization this modality is stated in
    a different unit from every material modality (measured: a raw 27-channel
    sum against per-cell-channel densities). When normalization is enabled the
    sums are divided by the ACTUAL channel count, reducing the modality to nats
    per proprioceptive channel. ``weights=None`` reproduces the historical
    arithmetic exactly.
    """

    labels = tuple(forecast.proprioceptive_labels)
    mean = np.asarray(forecast.proprioceptive_mean, dtype=np.float64)
    predictive_variance = np.asarray(forecast.proprioceptive_predictive_variance, dtype=np.float64)
    likelihood_variance = np.asarray(forecast.proprioceptive_likelihood_variance, dtype=np.float64)
    if not labels or not (len(labels) == mean.size == predictive_variance.size == likelihood_variance.size):
        raise ValueError("Execution forecast lacks a complete proprioceptive predictive density.")
    preference_std = np.asarray([_preference_std(label, config) for label in labels], dtype=np.float64)
    preference_variance = np.maximum(preference_std * preference_std, 1e-8)
    inflation = max(1e-2, float(reliability_inflation))
    is_fidelity = np.asarray(
        [1.0 if label.startswith(_FIDELITY_CHANNEL_PREFIXES) else 0.0 for label in labels],
        dtype=np.float64,
    )
    channel_inflation = 1.0 + (inflation - 1.0) * is_fidelity
    predictive_variance = np.maximum(predictive_variance, 0.0) * channel_inflation
    likelihood_variance = np.maximum(likelihood_variance, 1e-8) * channel_inflation
    outcome_variance = predictive_variance + likelihood_variance

    expected_negative_log_preference = 0.5 * np.sum(
        (mean * mean * channel_inflation + outcome_variance) / preference_variance
    )
    likelihood_excess_entropy = 0.5 * np.sum(
        np.log1p(likelihood_variance / preference_variance)
    )
    mutual_information = 0.5 * np.sum(
        np.log1p(predictive_variance / likelihood_variance)
    )
    # H[q(o)] in absolute nats. Logged only: it is the bridge between the
    # pragmatic term actually used and the canonical KL form of risk.
    outcome_entropy = 0.5 * np.sum(
        np.log(2.0 * np.pi * np.e * np.maximum(outcome_variance, 1e-12))
    )
    # Modality-level weight: gamma_motor (a Gamma precision belief mean, or the
    # declared constant when no ledger is injected) times the declared
    # per-proprioceptive-channel normalizer. Applied on top of, never instead
    # of, the three declared per-term precisions below, so both remain
    # separately attributable.
    channel_count = len(labels)
    if weights is None:
        modality_weight = 1.0
        normalizer = 1.0
        normalizer_name = ""
    else:
        modality_weight = weights.motor_weight(channel_count)
        normalizer = 1.0 / channel_count if weights.normalization_enabled else 1.0
        normalizer_name = weights.normalizer_name.get("motor_proprioceptive", "")
    risk = float(
        modality_weight
        * config.motor_proprioceptive_risk_precision
        * expected_negative_log_preference
    )
    ambiguity = float(
        modality_weight
        * config.motor_proprioceptive_ambiguity_precision
        * likelihood_excess_entropy
    )
    # The two subtracted terms carry separate declared precisions so they stay
    # separately attributable: state/observation information gain is not the
    # same quantity as parameter novelty over a learned belief.
    mutual_information_term = float(
        modality_weight * config.motor_proprioceptive_ambiguity_precision * mutual_information
    )
    reliability_novelty_term = float(
        modality_weight
        * config.motor_reliability_novelty_precision
        * max(0.0, float(reliability_epistemic_nats))
    )
    epistemic_value = mutual_information_term + reliability_novelty_term
    normalization_clause = (
        ""
        if weights is None
        else (
            "; terms reduced to nats per proprioceptive channel over "
            f"{channel_count} channels (normalizer {normalizer:.6g}, modality precision "
            f"{modality_weight * channel_count if weights.normalization_enabled else modality_weight:.4f})"
        )
    )
    return MotorEFETerms(
        risk=risk,
        ambiguity=ambiguity,
        epistemic_value=epistemic_value,
        approximation=(
            f"diagonal Gaussian motor EFE over {len(labels)} named normalized proprioceptive outcomes; "
            "risk is the pragmatic term E_q[-log p*(o)] with policy-independent Gaussian preference "
            "normalizers omitted, so the motor contribution to G is "
            "pragmatic - mutual_information - reliability_novelty; since "
            "-I(s;o) = E[H[p(o|s)]] - H[q(o)], that equals the canonical KL risk plus ambiguity, and "
            "likelihood excess entropy is therefore logged but never added again; likelihood excess "
            "entropy, forecast outcome entropy and process-observation mutual information are analytic "
            "in nats; reliability_novelty is parameter novelty over the learned inverse-gamma "
            "motion-reliability belief, the analogue of Dirichlet novelty in the reference; expected "
            "squared error of execution-fidelity channels scaled by the learned per-kind reliability "
            f"inflation {inflation:.3f}; hard safety limits remain external"
            f"{normalization_clause}"
        ),
        mutual_information=mutual_information_term,
        reliability_novelty=reliability_novelty_term,
        forecast_outcome_entropy=float(outcome_entropy),
        normalizer=float(normalizer),
        normalizer_name=normalizer_name,
        modality_precision=float(
            modality_weight * channel_count
            if weights is not None and weights.normalization_enabled
            else modality_weight
        ),
    )


def _preference_std(label: str, config: PainterConfig) -> float:
    if label.startswith("current_"):
        return float(config.motor_current_preference_std)
    if label.startswith("torque_"):
        return float(config.motor_torque_preference_std)
    if label.startswith("velocity_"):
        return float(config.motor_velocity_preference_std)
    if label.startswith("acceleration_"):
        return float(config.motor_acceleration_preference_std)
    if label.startswith("target_error_"):
        return float(config.motor_target_error_preference_std)
    if label.startswith("limit_proximity_"):
        return float(config.motor_limit_preference_std)
    if label == "contact_loss":
        return float(config.motor_contact_loss_preference_std)
    if label == "pressure_error":
        return float(config.motor_pressure_error_preference_std)
    if label == "path_error":
        return float(config.motor_path_error_preference_std)
    raise ValueError(f"No declared motor outcome preference for {label!r}.")


def _motor_primitive(kind: str, config: PainterConfig) -> MotorPrimitiveLatent:
    pivot = ""
    description = "Cartesian contact-aware IK realization"
    roll_start = 0.0
    roll_end = 0.0
    if kind == "joint_spline":
        description = "joint-space interpolation between contact poses"
    elif kind == "elbow_pivot":
        pivot = "elbow"
        description = "elbow-led joint-space arc between contact poses"
    elif kind == "shoulder_yaw_arc":
        pivot = "yaw"
        description = "shoulder-yaw-led joint-space arc between contact poses"
    elif kind in {"upper_arm_roll_positive", "upper_arm_roll_negative"}:
        pivot = "roll"
        sweep = abs(float(config.motor_roll_sweep_degrees))
        direction = 1.0 if kind == "upper_arm_roll_positive" else -1.0
        roll_start = -direction * sweep
        roll_end = direction * sweep
        description = "contact-aware upper-arm-axis roll sweep along the Cartesian mark path"
    elif kind in {"upper_arm_fixed_roll_positive", "upper_arm_fixed_roll_negative"}:
        pivot = "roll"
        magnitude = abs(float(config.motor_fixed_roll_degrees))
        direction = 1.0 if kind == "upper_arm_fixed_roll_positive" else -1.0
        roll_start = direction * magnitude
        roll_end = direction * magnitude
        description = (
            "contact-aware fixed upper-arm-axis roll posture along the declared mark path"
        )
    return MotorPrimitiveLatent(
        kind=kind,
        pivot_joint=pivot,
        description=description,
        roll_start_deg=roll_start,
        roll_end_deg=roll_end,
    )
