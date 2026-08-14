from __future__ import annotations

import torch

from active_painter.mixture_transition import (
    LocalMixtureConfig,
    LocalMixtureDynamicsEnsemble,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    material = torch.zeros(2, 6, 8, 8)
    material[:, 3] = 0.34
    action = torch.rand(2, 12, 8, 8)
    target = material.clone()
    target[:, 0, 2:6, 2:6] = 0.01
    target[:, 2, 2:6, 2:6] = 0.005
    mask = torch.ones(2, 1, 8, 8)
    return material, action, target, mask


def test_local_mixture_is_normalized_and_has_explicit_identity_component() -> None:
    config = LocalMixtureConfig(hidden_channels=8, residual_blocks=0, ensemble_size=2)
    model = LocalMixtureDynamicsEnsemble(config)
    material, action, _, mask = _inputs()

    means, logvars, log_weights = model.forward_masked(material, action, mask)

    assert means.shape == (2, 2, 2, 6, 8, 8)
    assert logvars.shape == means.shape
    assert log_weights.shape == means.shape
    assert torch.allclose(log_weights.exp().sum(dim=1), torch.ones_like(material))
    assert torch.allclose(means[:, 0], material.unsqueeze(0), atol=1e-7)


def test_mixture_nll_and_uncertainty_decomposition_are_finite_and_trainable() -> None:
    config = LocalMixtureConfig(hidden_channels=8, residual_blocks=1, ensemble_size=3)
    model = LocalMixtureDynamicsEnsemble(config)
    material, action, target, mask = _inputs()

    loss = model.training_loss(material, action, target, mask)
    moments = model.predictive_moments(material, action, mask)
    exact_nll = model.exact_ensemble_mixture_nll(material, action, target, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(exact_nll).all()
    assert torch.isfinite(moments.total_variance).all()
    assert torch.all(moments.likelihood_variance >= 0.0)
    assert torch.all(moments.epistemic_variance >= 0.0)
    assert any(parameter.grad is not None for parameter in model.parameters())
