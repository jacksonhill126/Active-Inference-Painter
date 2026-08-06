from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from active_painter.arm_agent_driver import (
    ORACLE_OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
)
from active_painter.config import PainterConfig
from active_painter.policies import PolicySampler, policy_posterior_from_efe
from active_painter.policy_ranges import (
    PROPOSAL_DIRECTION,
    assert_nested_ranges,
    passage_stroke_count_range,
    proposal_support_for,
)
from active_painter.proposal import (
    BELIEF_SOURCE_POSTERIOR,
    BELIEF_SOURCE_UNINITIALIZED,
    PolicyProposalNetwork,
    ProposalTrainingBatch,
    _log_logit_normal_log_density,
    _logit_normal_log_density,
    _wrapped_normal_log_density,
    base_efe_policy_posterior,
    hand_written_kind_log_probabilities,
    hand_written_log_density,
    proposal_divergence,
)
from active_painter.spatial_agent import SpatialActiveInferencePainter


def _config(**overrides: object) -> PainterConfig:
    values: dict[str, object] = {
        "planner_state_kind": "spatial_material",
        "spatial_grid_size": 8,
        "spatial_hidden_channels": 4,
        "spatial_residual_blocks": 1,
        "spatial_ensemble_size": 2,
        "composition_hidden_channels": 4,
        "composition_latent_dim": 4,
        "canvas_latent_channels": 4,
        "relational_latent_dim": 6,
        "hierarchy_hidden_dim": 12,
        "learned_proposal_hidden_dim": 12,
        "candidate_policies": 17,
        "planning_horizon": 4,
        "passage_min_strokes": 2,
        "passage_max_strokes": 4,
        "passage_plan_proposal_mix": 0.0,
        "batch_size": 2,
    }
    values.update(overrides)
    return PainterConfig(**values)


def _zero_features(network: PolicyProposalNetwork) -> torch.Tensor:
    return torch.zeros(1, network.input_dim, dtype=torch.float32)


def test_declared_proposal_support_is_nested_inside_policy_representations() -> None:
    """A proposal draw must always decode into the declared mark/latent domains."""

    assert_nested_ranges()


def test_base_efe_target_is_the_production_policy_posterior_without_extra_terms() -> None:
    """The amortization target is exactly Q(pi), not a reward-like surrogate."""

    totals = np.asarray([3.2, -0.4, 1.1, 8.0], dtype=np.float64)
    stop_log_priors = np.asarray([-7.0, 0.0, 0.0, 0.0], dtype=np.float64)
    precision = 2.75

    actual = base_efe_policy_posterior(totals, stop_log_priors, precision)
    production = policy_posterior_from_efe(
        torch.tensor(totals, dtype=torch.float64),
        torch.tensor(stop_log_priors, dtype=torch.float64),
        precision,
    )

    assert actual == pytest.approx(production.numpy(), abs=1e-14)
    assert float(actual.sum()) == pytest.approx(1.0, abs=1e-14)
    with pytest.raises(ValueError, match="must align"):
        base_efe_policy_posterior([1.0], [0.0, 0.0], precision)


def test_elementary_learned_proposal_densities_are_normalized() -> None:
    """Every continuous factor integrates to one on its declared support."""

    dtype = torch.float64
    loc = torch.tensor(0.35, dtype=dtype)
    scale = torch.tensor(0.65, dtype=dtype)
    linear_support = proposal_support_for("mark", "mark")["amount"]
    log_support = proposal_support_for("mark", "mark")["width"]

    linear_grid = np.linspace(linear_support.low + 1e-8, linear_support.high - 1e-8, 4001)
    linear_density = np.asarray(
        [
            math.exp(
                float(_logit_normal_log_density(value, loc, scale, linear_support).item())
            )
            for value in linear_grid
        ]
    )
    assert np.trapezoid(linear_density, linear_grid) == pytest.approx(1.0, abs=2e-4)

    log_grid = np.geomspace(log_support.low * (1.0 + 1e-8), log_support.high / (1.0 + 1e-8), 4001)
    log_density = np.asarray(
        [
            math.exp(
                float(_log_logit_normal_log_density(value, loc, scale, log_support).item())
            )
            for value in log_grid
        ]
    )
    assert np.trapezoid(log_density, log_grid) == pytest.approx(1.0, abs=2e-4)

    angles = np.linspace(PROPOSAL_DIRECTION.low, PROPOSAL_DIRECTION.high, 4001)
    mean = torch.tensor(1.7, dtype=dtype)
    circular_scale = torch.tensor(0.9, dtype=dtype)
    circular_density = np.asarray(
        [
            math.exp(
                float(_wrapped_normal_log_density(value, mean, circular_scale).item())
            )
            for value in angles
        ]
    )
    assert np.trapezoid(circular_density, angles) == pytest.approx(1.0, abs=2e-4)


def test_learned_density_rejects_values_outside_declared_support() -> None:
    """Clamping a foreign value to an edge must not create proposal support."""

    config = _config(stroke_tone_prior=None)
    network = PolicyProposalNetwork(config)
    features = _zero_features(network)
    record = network.sample(features, family="mark", config=config)
    amount_support = proposal_support_for("mark", "mark")["amount"]
    foreign_amount = replace(record.latents[0], amount=amount_support.high + 0.01)
    foreign_tone = replace(record.latents[0], tone=0.25)

    amount_score = network.log_density(
        replace(record, latents=(foreign_amount,)), features, config
    )
    tone_score = network.log_density(
        replace(record, latents=(foreign_tone,)), features, config
    )

    assert amount_score.in_support is False
    assert math.isinf(amount_score.total) and amount_score.total < 0.0
    assert tone_score.in_support is False
    assert math.isinf(tone_score.total) and tone_score.total < 0.0


def test_learned_samples_are_seeded_and_obey_all_config_support_masks() -> None:
    """The declared kind, depth, tone, and continuous supports dominate learning."""

    config = _config(
        passage_polyline_mix=1.0,
        passage_min_strokes=3,
        passage_max_strokes=4,
        stroke_tone_prior=1.0,
    )
    first = PolicyProposalNetwork(config)
    second = PolicyProposalNetwork(config)
    second.load_state_dict(first.state_dict())
    first.seed_generator(91)
    second.seed_generator(91)
    features = _zero_features(first)

    first_mark = first.sample(features, family="mark", config=config, count=4)
    second_mark = second.sample(features, family="mark", config=config, count=4)
    first_passage = first.sample(features, family="passage", config=config, count=64)
    second_passage = second.sample(features, family="passage", config=config, count=64)

    assert first_mark == second_mark
    assert first_passage == second_passage
    stroke_range = passage_stroke_count_range(config)
    for latent in first_mark.latents + first_passage.latents:
        support = proposal_support_for(latent.family, latent.kind)
        assert support["center_x"].contains(latent.center_x)
        assert support["center_y"].contains(latent.center_y)
        assert support["length"].contains(latent.length)
        assert support["width"].contains(latent.width)
        assert support["amount"].contains(latent.amount)
        assert latent.tone == 1.0
        if latent.family == "passage":
            assert latent.kind == "polyline"
            assert int(stroke_range.low) <= latent.stroke_count <= int(stroke_range.high)
            assert support["spacing"].contains(abs(latent.spacing))

    intermediate_config = _config(stroke_tone_prior=0.37)
    intermediate = PolicyProposalNetwork(intermediate_config)
    intermediate_features = _zero_features(intermediate)
    intermediate_record = intermediate.sample(
        intermediate_features, family="mark", config=intermediate_config
    )
    assert intermediate_record.latents[0].tone == 0.37
    assert intermediate.log_density(
        intermediate_record, intermediate_features, intermediate_config
    ).in_support


def test_hand_written_density_matches_empirical_passage_categoricals() -> None:
    """The scored hand density describes draws from the real sampler."""

    config = _config(
        proposal_low_coverage_mix=0.0,
        passage_polyline_mix=0.30,
        stroke_tone_prior=None,
    )
    sampler = PolicySampler(config, seed=81)
    sample_count = 4000
    kinds = {"band": 0, "chain": 0, "polyline": 0}
    counts: dict[int, int] = {}
    tones = {0.0: 0, 1.0: 0}
    normalized_log_widths: list[float] = []
    width_support = proposal_support_for("passage", "band")["width"]

    for _ in range(sample_count):
        latent = sampler._passage_draw()
        record = sampler._passage_record(latent, "hand")
        score = hand_written_log_density(record, None, config)
        assert score.in_support
        assert math.isfinite(score.total)
        kinds[latent.kind] += 1
        counts[latent.stroke_count] = counts.get(latent.stroke_count, 0) + 1
        tones[latent.tone] += 1
        normalized_log_widths.append(
            (math.log(latent.width) - math.log(width_support.low)) / width_support.log_width
        )

    expected_kinds = {
        kind: math.exp(log_probability)
        for kind, log_probability in hand_written_kind_log_probabilities(config).items()
    }
    for kind, expected in expected_kinds.items():
        assert kinds[kind] / sample_count == pytest.approx(expected, abs=0.025)
    stroke_range = passage_stroke_count_range(config)
    expected_count_probability = 1.0 / (int(stroke_range.high) - int(stroke_range.low) + 1)
    for count in range(int(stroke_range.low), int(stroke_range.high) + 1):
        assert counts[count] / sample_count == pytest.approx(expected_count_probability, abs=0.025)
    assert tones[1.0] / sample_count == pytest.approx(0.5, abs=0.025)
    assert float(np.mean(normalized_log_widths)) == pytest.approx(0.5, abs=0.025)


def test_generic_proposal_divergence_is_exactly_zero_against_itself() -> None:
    """The diagnostic has a falsifiable zero control independent of quadrature."""

    config = _config()
    sampler = PolicySampler(config, seed=7)

    def draw():
        return sampler._passage_record(sampler._passage_draw(), "hand")

    def density(record):
        return hand_written_log_density(record, None, config)

    result = proposal_divergence(draw, density, density, samples=128)

    assert result.divergence_nats == 0.0
    assert result.sample_count == 128
    assert result.out_of_support_fraction == 0.0


def test_zero_learned_mix_preserves_candidate_and_rng_streams_exactly() -> None:
    """The default feature gate is a byte-for-byte behavioural compatibility path."""

    config = _config(
        candidate_policies=31,
        learned_proposal_mix=0.0,
        passage_proposal_mix=0.45,
        passage_polyline_mix=0.35,
        stroke_tone_prior=None,
    )
    network = PolicyProposalNetwork(config)
    network.seed_generator(1234)
    learned = PolicySampler(config, seed=19, learned_proposal=network)
    baseline = PolicySampler(config, seed=19)
    generator_before = network.generator.get_state().clone()

    learned_policies = learned.sample(belief_features=_zero_features(network))
    baseline_policies = baseline.sample()

    assert learned_policies == baseline_policies
    assert learned.last_proposal_records == baseline.last_proposal_records
    assert learned.last_learned_candidate_count == 0
    assert learned.rng.bit_generator.state == baseline.rng.bit_generator.state
    assert torch.equal(network.generator.get_state(), generator_before)


def test_nonzero_mix_keeps_stop_and_paired_hand_written_control() -> None:
    """Learned proposals augment the candidate set without displacing its control."""

    config = _config(
        candidate_policies=41,
        learned_proposal_mix=0.5,
        passage_proposal_mix=0.5,
        stroke_tone_prior=1.0,
    )
    network = PolicyProposalNetwork(config)
    sampler = PolicySampler(config, seed=23, learned_proposal=network)

    policies = sampler.sample(belief_features=_zero_features(network))
    records = sampler.last_proposal_records

    assert len(policies) == config.candidate_policies
    assert len(records) == len(policies)
    assert policies[0].actions[0].stop
    assert records[0] is None
    assert all(policy.actions[-1].stop for policy in policies)
    assert all(not action.stop for policy in policies[1:] for action in policy.actions[:-1])
    assert all(record is not None for record in records[1:])
    sources = [record.source for record in records[1:] if record is not None]
    assert set(sources) == {"hand", "learned"}
    assert sampler.last_learned_candidate_count == sources.count("learned")


def test_training_loss_is_posterior_weighted_log_proposal_density_only() -> None:
    """Refined motor weights and all outcome scores stay outside the objective."""

    config = _config(stroke_tone_prior=1.0)
    network = PolicyProposalNetwork(config)
    features = _zero_features(network)
    mark = network.sample(features, family="mark", config=config, count=2)
    passage = network.sample(features, family="passage", config=config)
    batch = ProposalTrainingBatch(
        features=features,
        records=(None, mark, passage),
        weights=(0.20, 0.30, 0.50),
        refined_weights=(0.99, 0.005, 0.005),
        belief_feature_source=BELIEF_SOURCE_POSTERIOR,
    )
    alternate_refinement = replace(batch, refined_weights=(0.0, 0.5, 0.5))

    heads = network.distribution(features)
    mark_log_q = torch.stack(list(network.log_density_terms(mark, heads, config).values())).sum()
    passage_log_q = torch.stack(
        list(network.log_density_terms(passage, heads, config).values())
    ).sum()
    expected = -(0.30 * mark_log_q + 0.50 * passage_log_q) / 0.80
    loss = network.training_loss(batch, config)
    alternate = network.training_loss(alternate_refinement, config)

    assert loss is not None
    assert alternate is not None
    assert float(loss.item()) == pytest.approx(float(expected.item()), rel=1e-6)
    assert float(alternate.item()) == pytest.approx(float(loss.item()), rel=1e-7)
    loss.backward()
    gradients = [parameter.grad for parameter in network.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_agent_refuses_fallback_features_then_trains_on_posterior_beliefs() -> None:
    """A zero placeholder may sample candidates but may never train recognition."""

    config = _config(stroke_tone_prior=1.0, learned_proposal_train_steps=1)
    agent = SpatialActiveInferencePainter(config, seed=31, device="cpu")
    assert agent.policy_proposal is not None
    fallback_features, fallback_source = agent.proposal_belief_features()
    assert fallback_features is not None
    assert fallback_source == BELIEF_SOURCE_UNINITIALIZED
    fallback_record = agent.policy_proposal.sample(
        fallback_features, family="mark", config=config
    )
    fallback_batch = ProposalTrainingBatch(
        features=fallback_features,
        records=(fallback_record,),
        weights=(1.0,),
        belief_feature_source=fallback_source,
    )

    assert agent.train_policy_proposal(fallback_batch) is None
    assert agent.proposal_no_belief_skips == 1
    assert agent.policy_proposal.update_count == 0

    agent.reset_hierarchy_beliefs(agent.belief)
    features, source = agent.proposal_belief_features()
    assert features is not None
    assert source == BELIEF_SOURCE_POSTERIOR
    record = agent.policy_proposal.sample(features, family="passage", config=config)
    batch = ProposalTrainingBatch(
        features=features,
        records=(record,),
        weights=(1.0,),
        target_support_fraction=1.0,
        belief_feature_source=source,
    )
    before = [parameter.detach().clone() for parameter in agent.policy_proposal.parameters()]

    loss = agent.train_policy_proposal(batch)

    assert loss is not None and math.isfinite(loss)
    assert agent.policy_proposal.update_count == 1
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy_proposal.parameters())
    )


def test_belief_feature_source_is_explicit_and_shape_checked() -> None:
    """Fallback zeros cannot masquerade as posterior canvas/relational means."""

    config = _config()
    network = PolicyProposalNetwork(config)
    device = torch.device("cpu")
    absent, absent_source = network.features_from_beliefs(None, None, config, device)
    no_hierarchy, hierarchy_source = network.features_from_beliefs(
        None, None, config, device, has_hierarchy=False
    )
    canvas_mean = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    relational_mean = torch.arange(6, dtype=torch.float32)
    posterior, posterior_source = network.features_from_beliefs(
        SimpleNamespace(mean=canvas_mean),
        SimpleNamespace(mean=relational_mean),
        config,
        device,
    )
    mismatched, mismatch_source = network.features_from_beliefs(
        SimpleNamespace(mean=torch.zeros(1)),
        SimpleNamespace(mean=torch.zeros(1)),
        config,
        device,
    )

    assert absent.shape == (1, network.input_dim)
    assert torch.count_nonzero(absent) == 0
    assert absent_source == BELIEF_SOURCE_UNINITIALIZED
    assert torch.equal(absent, no_hierarchy)
    assert hierarchy_source != BELIEF_SOURCE_POSTERIOR
    assert posterior_source == BELIEF_SOURCE_POSTERIOR
    assert torch.equal(posterior, torch.cat([canvas_mean.reshape(1, -1), relational_mean.reshape(1, -1)], dim=1))
    assert mismatch_source == BELIEF_SOURCE_UNINITIALIZED
    assert torch.count_nonzero(mismatched) == 0


def test_proposal_checkpoint_round_trips_parameters_optimizer_and_rng() -> None:
    """A resumed run continues both proposal learning and its private draw stream."""

    config = _config(stroke_tone_prior=1.0)
    root = Path("runs/test_driver_checkpoint_proposal")
    shutil.rmtree(root, ignore_errors=True)
    checkpoint = root / "proposal.pt"
    try:
        driver = ArmActiveInferenceDriver(
            config=config,
            bootstrap_transitions=0,
            bootstrap_train_steps=0,
            checkpoint_path=checkpoint,
            observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
            device="cpu",
        )
        assert isinstance(driver.agent, SpatialActiveInferencePainter)
        proposal = driver.agent.policy_proposal
        assert proposal is not None
        features = _zero_features(proposal)
        record = proposal.sample(features, family="mark", config=config)
        batch = ProposalTrainingBatch(
            features=features,
            records=(record,),
            weights=(1.0,),
            belief_feature_source=BELIEF_SOURCE_POSTERIOR,
        )
        assert driver.agent.train_policy_proposal(batch) is not None
        driver._save_checkpoint_if_due(force=True)
        assert driver.checkpoint_status == "saved"
        expected_state = {
            key: value.detach().clone() for key, value in proposal.state_dict().items()
        }
        expected_next = proposal.sample(features, family="passage", config=config)

        restored = ArmActiveInferenceDriver(
            config=config,
            bootstrap_transitions=0,
            bootstrap_train_steps=0,
            checkpoint_path=checkpoint,
            observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
            device="cpu",
        )

        assert restored.checkpoint_loaded
        assert isinstance(restored.agent, SpatialActiveInferencePainter)
        restored_proposal = restored.agent.policy_proposal
        assert restored_proposal is not None
        for key, expected in expected_state.items():
            assert torch.equal(restored_proposal.state_dict()[key], expected)
        assert restored.agent.policy_proposal_optimizer is not None
        assert restored.agent.policy_proposal_optimizer.state_dict()["state"]
        actual_next = restored_proposal.sample(
            _zero_features(restored_proposal), family="passage", config=config
        )
        assert actual_next == expected_next
        json.dumps(restored.diagnostics()["policyProposal"], allow_nan=False)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_proposal_construction_and_zero_mix_do_not_change_shared_models_or_efe() -> None:
    """The proposal has no path into dynamics, preferences, VFE, or EFE terms."""

    enabled_config = _config(
        learned_proposal_enabled=True,
        learned_proposal_mix=0.0,
        candidate_policies=9,
        stroke_tone_prior=1.0,
    )
    disabled_config = replace(enabled_config, learned_proposal_enabled=False)
    enabled = SpatialActiveInferencePainter(enabled_config, seed=47, device="cpu")
    disabled = SpatialActiveInferencePainter(disabled_config, seed=47, device="cpu")
    assert enabled.policy_proposal is not None
    assert disabled.policy_proposal is None

    for key, expected in disabled.dynamics.state_dict().items():
        assert torch.equal(enabled.dynamics.state_dict()[key], expected)
    assert enabled.composition is not None and disabled.composition is not None
    for key, expected in disabled.composition.state_dict().items():
        assert torch.equal(enabled.composition.state_dict()[key], expected)

    enabled.reset_hierarchy_beliefs(enabled.belief)
    disabled.reset_hierarchy_beliefs(disabled.belief)
    features, source = enabled.proposal_belief_features()
    assert features is not None and source == BELIEF_SOURCE_POSTERIOR
    enabled_policies = enabled.policy_sampler.sample(belief_features=features)
    disabled_policies = disabled.policy_sampler.sample()
    assert enabled_policies == disabled_policies

    enabled_efe = enabled.efe.evaluate_batch(enabled.belief, enabled_policies)
    disabled_efe = disabled.efe.evaluate_batch(disabled.belief, disabled_policies)
    assert [component.total for component in enabled_efe] == pytest.approx(
        [component.total for component in disabled_efe], rel=0.0, abs=1e-8
    )

    action = enabled_policies[1].actions[0]
    enabled.update_belief(action, enabled.belief)
    disabled.update_belief(action, disabled.belief)
    assert enabled.last_vfe is not None and disabled.last_vfe is not None
    assert enabled.last_vfe.total == pytest.approx(disabled.last_vfe.total, abs=1e-8)
    assert enabled.last_vfe.complexity == pytest.approx(disabled.last_vfe.complexity, abs=1e-8)
    assert enabled.last_vfe.negative_log_likelihood == pytest.approx(
        disabled.last_vfe.negative_log_likelihood, abs=1e-8
    )
