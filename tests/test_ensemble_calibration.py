"""Calibration checks for the ensemble-as-parameter-posterior approximation.

The audit notes flagged that nothing demonstrated ensemble variance behaves
like posterior uncertainty. These tests train the ensemble on a synthetic
transition rule and check (a) held-out predictive z-scores are in a sane band
rather than wildly over/under-confident, and (b) member disagreement is larger
off the training distribution than on it, which is the property the epistemic
term of expected free energy relies on.
"""

from __future__ import annotations

import numpy as np
import torch

from active_painter.config import PainterConfig
from active_painter.models import DynamicsEnsemble

NOISE_STD = 0.02


def synthetic_batch(rng: np.random.Generator, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.tensor(rng.uniform(0.0, 1.0, (size, 6)), dtype=torch.float32)
    action = torch.tensor(rng.uniform(0.0, 0.5, (size, 7)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(0.0, NOISE_STD, (size, 6)), dtype=torch.float32)
    next_state = state + 0.2 * action[:, 5:6] + noise
    return state, action, next_state


def trained_ensemble(train_steps: int = 600) -> tuple[DynamicsEnsemble, np.random.Generator]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = PainterConfig(hidden_dim=32, ensemble_size=5)
    model = DynamicsEnsemble(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(train_steps):
        state, action, next_state = synthetic_batch(rng, 256)
        loss = model.nll(state, action, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, rng


def test_ensemble_predictive_z_scores_are_roughly_calibrated() -> None:
    model, rng = trained_ensemble()
    state, action, next_state = synthetic_batch(rng, 512)

    with torch.no_grad():
        mean, aleatoric, epistemic = model.predictive_moments(state, action)
    squared_z = ((next_state - mean) ** 2 / (aleatoric + epistemic)).mean()

    assert 0.2 < float(squared_z) < 5.0


def test_bootstrap_trained_ensemble_disagrees_more_off_distribution() -> None:
    model, rng = trained_ensemble()
    state, action, _ = synthetic_batch(rng, 512)
    off_action = torch.zeros_like(action)
    off_action[:, 5] = 3.0  # far outside the training range of [0, 0.5]

    with torch.no_grad():
        _, _, epistemic_in = model.predictive_moments(state, action)
        _, _, epistemic_off = model.predictive_moments(state, off_action)

    assert float(epistemic_in.mean()) > 0.0
    assert float(epistemic_off.mean()) > 3.0 * float(epistemic_in.mean())
