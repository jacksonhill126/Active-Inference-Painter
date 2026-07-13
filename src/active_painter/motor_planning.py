from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PainterConfig
from .policies import MotorPrimitiveLatent, Policy


@dataclass(frozen=True, slots=True)
class MotorEFETerms:
    """Precision-weighted EFE terms for proprioceptive outcome modalities."""

    risk: float
    ambiguity: float
    epistemic_value: float
    approximation: str


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
                motor_primitive=_motor_primitive(kind),
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


def motor_efe_terms(forecast, config: PainterConfig) -> MotorEFETerms:
    """EFE over a diagonal proprioceptive predictive density, in nats.

    Risk is expected negative log probability under declared zero-centered
    homeostatic outcome preferences, with policy-independent Gaussian
    normalizers omitted. Ambiguity is likelihood entropy in excess of each
    preference scale. Epistemic value is the diagonal Gaussian mutual
    information between process uncertainty and proprioceptive observations.
    """

    labels = tuple(forecast.proprioceptive_labels)
    mean = np.asarray(forecast.proprioceptive_mean, dtype=np.float64)
    predictive_variance = np.asarray(forecast.proprioceptive_predictive_variance, dtype=np.float64)
    likelihood_variance = np.asarray(forecast.proprioceptive_likelihood_variance, dtype=np.float64)
    if not labels or not (len(labels) == mean.size == predictive_variance.size == likelihood_variance.size):
        raise ValueError("Execution forecast lacks a complete proprioceptive predictive density.")
    preference_std = np.asarray([_preference_std(label, config) for label in labels], dtype=np.float64)
    preference_variance = np.maximum(preference_std * preference_std, 1e-8)
    predictive_variance = np.maximum(predictive_variance, 0.0)
    likelihood_variance = np.maximum(likelihood_variance, 1e-8)
    outcome_variance = predictive_variance + likelihood_variance

    expected_negative_log_preference = 0.5 * np.sum(
        (mean * mean + outcome_variance) / preference_variance
    )
    likelihood_excess_entropy = 0.5 * np.sum(
        np.log1p(likelihood_variance / preference_variance)
    )
    mutual_information = 0.5 * np.sum(
        np.log1p(predictive_variance / likelihood_variance)
    )
    risk = float(config.motor_proprioceptive_risk_precision * expected_negative_log_preference)
    ambiguity = float(config.motor_proprioceptive_ambiguity_precision * likelihood_excess_entropy)
    epistemic_value = float(config.motor_proprioceptive_ambiguity_precision * mutual_information)
    return MotorEFETerms(
        risk=risk,
        ambiguity=ambiguity,
        epistemic_value=epistemic_value,
        approximation=(
            f"diagonal Gaussian motor EFE over {len(labels)} named normalized proprioceptive outcomes; "
            "risk omits policy-independent preference normalizers; likelihood entropy and process-observation "
            "mutual information are analytic in nats; hard safety limits remain external"
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


def _motor_primitive(kind: str) -> MotorPrimitiveLatent:
    pivot = ""
    description = "Cartesian contact-aware IK realization"
    if kind == "joint_spline":
        description = "joint-space interpolation between contact poses"
    elif kind == "elbow_pivot":
        pivot = "elbow"
        description = "elbow-led joint-space arc between contact poses"
    elif kind == "shoulder_yaw_arc":
        pivot = "yaw"
        description = "shoulder-yaw-led joint-space arc between contact poses"
    return MotorPrimitiveLatent(kind=kind, pivot_joint=pivot, description=description)
