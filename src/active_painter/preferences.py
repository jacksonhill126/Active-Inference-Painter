from __future__ import annotations

import torch
from torch.distributions import Beta

from .config import PainterConfig


class TerminalCoveragePreference:
    """Prior preference p*(coverage_T | stop).

    It is intentionally applied only at a terminal `stop` state, never at every
    intermediate step.
    """

    def __init__(self, config: PainterConfig) -> None:
        mean = config.target_coverage
        concentration = config.terminal_concentration
        self.alpha = torch.tensor(mean * concentration, dtype=torch.float32)
        self.beta = torch.tensor((1.0 - mean) * concentration, dtype=torch.float32)

    def distribution(self, device: torch.device | str = "cpu") -> Beta:
        return Beta(self.alpha.to(device), self.beta.to(device))

    def negative_log_prob(self, coverage: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(coverage, 1e-4, 1.0 - 1e-4)
        return -self.distribution(c.device).log_prob(c)
