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


def motor_efe_terms(forecast, config: PainterConfig) -> MotorEFETerms:
    """Expected free-energy terms over predicted proprioceptive observations.

    Risk is a homeostatic prior preference over low current, torque, joint
    acceleration, limit proximity, and target-error observations. Ambiguity is
    a likelihood-entropy proxy from contact loss, pressure variance, target
    error, and execution covariance. These are separate outcome modalities and
    do not alter terminal coverage or composition preferences.
    """

    velocity_norm = float(forecast.joint_velocity_rms) / 5.0
    acceleration_norm = float(forecast.joint_acceleration_rms) / 70.0
    target_error_norm = float(forecast.joint_target_error_rms) / 45.0
    path_norm = float(forecast.joint_path_length_deg) / 360.0
    current_norm = float(forecast.joint_current_rms)
    torque_norm = float(forecast.joint_torque_rms)
    limit_proximity = float(forecast.joint_limit_proximity)

    unweighted_risk = (
        0.36 * current_norm * current_norm
        + 0.24 * torque_norm * torque_norm
        + 0.14 * acceleration_norm * acceleration_norm
        + 0.10 * velocity_norm * velocity_norm
        + 0.10 * limit_proximity * limit_proximity
        + 0.04 * target_error_norm * target_error_norm
        + 0.02 * path_norm * path_norm
    )
    path_covariance = float(np.sqrt(max(0.0, forecast.path_covariance[0] + forecast.path_covariance[1])))
    pressure_var = float(max(0.0, forecast.pressure_variance))
    unweighted_ambiguity = (
        0.35 * float(forecast.contact_loss_probability)
        + 0.25 * min(1.0, pressure_var)
        + 0.20 * min(1.0, path_covariance)
        + 0.20 * min(1.0, target_error_norm)
    )
    risk = float(config.motor_proprioceptive_risk_precision * unweighted_risk)
    ambiguity = float(config.motor_proprioceptive_ambiguity_precision * unweighted_ambiguity)
    return MotorEFETerms(
        risk=risk,
        ambiguity=ambiguity,
        epistemic_value=0.0,
        approximation=(
            "motor EFE uses normalized proprioceptive risk and likelihood-entropy proxies "
            "from the execution forecast; hard safety limits remain external"
        ),
    )


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
