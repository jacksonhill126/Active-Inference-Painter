from __future__ import annotations

import torch
from torch.distributions import Beta

from .preferences import TerminalCoveragePreference


def coverage_beta_approximation(
    mean: torch.Tensor,
    variance: torch.Tensor,
    concentration_floor: float = 0.0,
) -> Beta:
    # Approximation: the terminal forecast q(C_T | policy) is moment-matched
    # to a Beta distribution on material coverage C_T in [0, 1].
    #
    # Approximation (declared forecast-family restriction, not a clamp on risk):
    # `concentration_floor` restricts the moment match to the INTERIOR-UNIMODAL
    # Beta family, both concentrations >= the floor. A concentration below 1
    # puts a boundary spike in the forecast, and the resulting
    # digamma(alpha -> 0) makes the exact Beta-Beta KL diverge: measured 53248
    # nats on a near-blank low-variance forecast (mean 1e-4, std 2.3e-3), which
    # is a numerical artefact of the moment match rather than a belief the model
    # holds. The rescale is a COMMON factor on both concentrations, so the
    # forecast mean alpha/(alpha+beta) is preserved exactly and the term stays a
    # genuine KL between two Betas. Measured inert on every well-conditioned
    # forecast (mean 0.05 var 1e-4: 246.6508 -> 246.6508; 0.5/1e-6:
    # 37.2720 -> 37.2720; 0.87/1e-8: 4.4849 -> 4.4849; 0.87/1e-4:
    # 0.7181 -> 0.7181), and it bites only where alpha < floor or beta < floor.
    # Floor 0.0 restores the historical unrestricted family exactly.
    mean = torch.clamp(mean, 1e-4, 1.0 - 1e-4)
    max_variance = torch.clamp(mean * (1.0 - mean) - 1e-8, min=1e-8)
    variance = torch.minimum(torch.clamp(variance, min=1e-8), max_variance)
    concentration = torch.clamp(mean * (1.0 - mean) / variance - 1.0, min=2.0, max=1e6)
    alpha = torch.clamp(mean * concentration, min=1e-4)
    beta = torch.clamp((1.0 - mean) * concentration, min=1e-4)
    floor = float(concentration_floor)
    if floor > 0.0:
        scale = torch.clamp(
            torch.maximum(
                floor / torch.clamp(alpha, min=1e-12),
                floor / torch.clamp(beta, min=1e-12),
            ),
            min=1.0,
        )
        alpha = alpha * scale
        beta = beta * scale
    return Beta(alpha, beta)


def terminal_preference_terms(
    preference: TerminalCoveragePreference,
    coverage_mean: torch.Tensor,
    coverage_variance: torch.Tensor,
    precision: float = 1.0,
    concentration_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-policy terminal risk, entropy, and pragmatic value.

    Element-wise over a batch of terminal coverage forecasts, so both the
    single-policy and batched evaluators share one implementation. All three
    returned tensors are scaled by the SAME declared terminal-risk precision, so
    the identity `risk = -entropy - pragmatic_value` holds at any precision --
    including the posterior mean of a Gamma precision belief, and including a
    clamped boundary mean.

    `precision` is a plain float on purpose: it may be a hand-declared constant
    or a belief's posterior mean, and this function must not be able to tell the
    difference. The PREFERENCE itself (`preference`) is never touched here.
    """

    forecast = coverage_beta_approximation(
        coverage_mean,
        coverage_variance,
        concentration_floor=concentration_floor,
    )
    target = preference.distribution(coverage_mean.device)
    terminal_entropy = forecast.entropy()

    expected_log_coverage = torch.digamma(forecast.concentration1) - torch.digamma(
        forecast.concentration1 + forecast.concentration0
    )
    expected_log_uncovered = torch.digamma(forecast.concentration0) - torch.digamma(
        forecast.concentration1 + forecast.concentration0
    )
    log_beta_target = (
        torch.lgamma(target.concentration1)
        + torch.lgamma(target.concentration0)
        - torch.lgamma(target.concentration1 + target.concentration0)
    )
    pragmatic_value = (
        (target.concentration1 - 1.0) * expected_log_coverage
        + (target.concentration0 - 1.0) * expected_log_uncovered
        - log_beta_target
    )
    risk = -terminal_entropy - pragmatic_value
    return precision * risk, precision * terminal_entropy, precision * pragmatic_value


def project_summary_transition_support(current_mean: torch.Tensor, next_mean: torch.Tensor) -> torch.Tensor:
    # Structural transition support for material canvas states. Painting can
    # add material and wetness, but this model has no erasing or clearing
    # action inside a candidate painting policy.
    projected = next_mean.clone()
    projected[..., 0] = torch.maximum(projected[..., 0], current_mean[..., 0]).clamp(0.0, 1.0)
    projected[..., 1] = torch.clamp(projected[..., 1], min=0.0)
    projected[..., 2] = torch.maximum(projected[..., 2], projected[..., 1]).clamp(min=0.0)
    projected[..., 3] = torch.maximum(projected[..., 3], current_mean[..., 3]).clamp(min=0.0)
    projected[..., 4] = torch.clamp(projected[..., 4], 0.0, 1.0)
    projected[..., 5] = torch.clamp(projected[..., 5], 0.0, 1.0)
    return projected


def project_material_support(
    current: torch.Tensor,
    proposed: torch.Tensor,
    thickness_scale: float,
    ground_tone: float,
    paint_presence_threshold: float = 0.0001,
) -> torch.Tensor:
    # Structural support: material thickness and pigment mass have no
    # erasing/clearing action inside a candidate painting policy. Wetness is
    # persistent in the oil-paint process and therefore cannot spontaneously
    # decrease. Derived
    # ground-contrast and coverage fields are recomputed from surface tone,
    # thickness, and substrate tone instead of treated as free predictions.
    base = proposed.clamp(min=0.0)
    channels = [base[:, index : index + 1] for index in range(base.shape[1])]
    channels[0] = torch.maximum(channels[0], current[:, 0:1])
    channels[1] = torch.maximum(channels[1], current[:, 1:2])
    channels[2] = torch.maximum(channels[2], current[:, 2:3])
    scale = max(1e-8, float(thickness_scale))
    threshold = max(0.0, float(paint_presence_threshold))
    if len(channels) > 3:
        thickness = torch.clamp(channels[0], min=0.0)
        coverage = (thickness >= threshold).to(thickness.dtype)
        opacity = 1.0 - torch.exp(-thickness / scale)
        channels[3] = torch.clamp(channels[3], 0.0, 1.0)
    if len(channels) > 4:
        observed_tone = (1.0 - opacity) * float(ground_tone) + opacity * channels[3]
        channels[4] = torch.abs(observed_tone - float(ground_tone))
    if len(channels) > 5:
        channels[5] = coverage
    return torch.cat(channels, dim=1)
