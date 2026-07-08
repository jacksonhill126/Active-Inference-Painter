from __future__ import annotations

import torch
from torch.distributions import Beta

from .preferences import TerminalCoveragePreference


def coverage_beta_approximation(mean: torch.Tensor, variance: torch.Tensor) -> Beta:
    # Approximation: the terminal forecast q(C_T | policy) is moment-matched
    # to a Beta distribution on material coverage C_T in [0, 1].
    mean = torch.clamp(mean, 1e-4, 1.0 - 1e-4)
    max_variance = torch.clamp(mean * (1.0 - mean) - 1e-8, min=1e-8)
    variance = torch.minimum(torch.clamp(variance, min=1e-8), max_variance)
    concentration = torch.clamp(mean * (1.0 - mean) / variance - 1.0, min=2.0, max=1e6)
    alpha = torch.clamp(mean * concentration, min=1e-4)
    beta = torch.clamp((1.0 - mean) * concentration, min=1e-4)
    return Beta(alpha, beta)


def terminal_preference_terms(
    preference: TerminalCoveragePreference,
    coverage_mean: torch.Tensor,
    coverage_variance: torch.Tensor,
    precision: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-policy terminal risk, entropy, and pragmatic value.

    Element-wise over a batch of terminal coverage forecasts, so both the
    single-policy and batched evaluators share one implementation. All three
    returned tensors are scaled by the declared terminal-risk precision so the
    identity `risk = -entropy - pragmatic_value` holds at any precision.
    """

    forecast = coverage_beta_approximation(coverage_mean, coverage_variance)
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
    projected[..., 3] = torch.clamp(projected[..., 3], min=0.0)
    projected[..., 4] = torch.clamp(projected[..., 4], 0.0, 1.0)
    projected[..., 5] = torch.clamp(projected[..., 5], 0.0, 1.0)
    return projected


def project_material_support(
    current: torch.Tensor,
    proposed: torch.Tensor,
    thickness_scale: float,
    ground_tone: float,
) -> torch.Tensor:
    # Structural support: material thickness and pigment mass have no
    # erasing/clearing action inside a candidate painting policy; wetness
    # can decay, so it is only constrained to remain non-negative. Derived
    # observed-tone, ground-contrast, and coverage fields are recomputed
    # from the primary material channels and substrate tone instead of
    # treated as free predictions.
    base = proposed.clamp(min=0.0)
    channels = [base[:, index : index + 1] for index in range(base.shape[1])]
    channels[0] = torch.maximum(channels[0], current[:, 0:1])
    channels[2] = torch.maximum(channels[2], current[:, 2:3])
    scale = max(1e-8, float(thickness_scale))
    if len(channels) > 3:
        pigment_tone = torch.clamp(channels[2] / torch.clamp(channels[0], min=1e-6), 0.0, 1.0)
        coverage = 1.0 - torch.exp(-torch.clamp(channels[0], min=0.0) / scale)
        channels[3] = torch.clamp((1.0 - coverage) * float(ground_tone) + coverage * pigment_tone, 0.0, 1.0)
    if len(channels) > 4:
        channels[4] = torch.abs(channels[3] - float(ground_tone))
    if len(channels) > 5:
        channels[5] = 1.0 - torch.exp(-torch.clamp(channels[0], min=0.0) / scale)
    return torch.cat(channels, dim=1)
