import torch

from active_painter.config import PainterConfig
from active_painter.preferences import TerminalCoveragePreference


def test_terminal_preference_favors_target_band() -> None:
    pref = TerminalCoveragePreference(PainterConfig())
    target = pref.negative_log_prob(torch.tensor(0.87))
    low = pref.negative_log_prob(torch.tensor(0.55))
    high = pref.negative_log_prob(torch.tensor(0.98))
    assert target < low
    assert target < high
