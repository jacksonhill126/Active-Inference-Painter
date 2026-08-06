import numpy as np
import pytest
import torch

from active_painter.config import PainterConfig
from active_painter.precision_beliefs import MODALITY_NAMES, PrecisionLedger
from active_painter.preferences import TerminalCoveragePreference


def test_terminal_preference_favors_target_band() -> None:
    pref = TerminalCoveragePreference(PainterConfig())
    target = pref.negative_log_prob(torch.tensor(0.87))
    low = pref.negative_log_prob(torch.tensor(0.55))
    high = pref.negative_log_prob(torch.tensor(0.98))
    assert target < low
    assert target < high


def test_precision_learning_never_touches_the_declared_preference() -> None:
    """Precisions MAY be beliefs; preferences may NOT be learned from outcomes.

    This file is the tripwire for that rule. After enough precision updates to
    move every gamma, the terminal preference's Beta concentrations must still be
    bit-identical and still derive from the declared config constants.
    """

    cfg = PainterConfig()
    pref = TerminalCoveragePreference(cfg)
    alpha_before = float(pref.alpha.item())
    beta_before = float(pref.beta.item())

    ledger = PrecisionLedger(cfg)
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, size=16)
    G = (base - base.mean()) * 1.7
    for _ in range(5):
        ledger.observe_policy(G, 0.5 * G)
        for name in MODALITY_NAMES:
            ledger.observe(name, G, 0.5 * G)
    assert ledger.mean("terminal_coverage") != cfg.terminal_risk_precision

    assert float(pref.alpha.item()) == alpha_before
    assert float(pref.beta.item()) == beta_before
    # float32 storage, so approx against the float64 config product.
    assert alpha_before == pytest.approx(cfg.target_coverage * cfg.terminal_concentration, rel=1e-6)
    assert beta_before == pytest.approx(
        (1.0 - cfg.target_coverage) * cfg.terminal_concentration, rel=1e-6
    )
    # Reconstructing it from the live config must give the same numbers, i.e. no
    # learned state has crept into the preference.
    rebuilt = TerminalCoveragePreference(cfg)
    assert float(rebuilt.alpha.item()) == alpha_before
    assert float(rebuilt.beta.item()) == beta_before
