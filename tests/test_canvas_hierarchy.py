import numpy as np
import pytest
import torch

from active_painter.canvas_hierarchy import (
    PASSAGE_STEP_DESCRIPTOR_DIM,
    POLICY_DESCRIPTOR_DIM,
    HierarchicalCanvasModel,
    passage_step_descriptor,
    policy_descriptor,
    relational_observation_dim,
    relational_observation_vector,
)
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.policies import PassageLatent, Policy, PolicySampler
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_state import SpatialCanvasState


def _config() -> PainterConfig:
    return PainterConfig(
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_hidden_channels=6,
        canvas_latent_channels=4,
        relational_latent_dim=6,
        hierarchy_hidden_dim=12,
        mark_slot_count=4,
        batch_size=2,
        hierarchy_transition_batch_size=2,
        passage_trajectory_batch_size=2,
    )


def _field(config: PainterConfig, row: slice, col: slice) -> np.ndarray:
    material = np.zeros(
        (config.spatial_material_channels, config.spatial_grid_size, config.spatial_grid_size),
        dtype=np.float32,
    )
    material[0, row, col] = 0.01
    material[1, row, col] = 0.008
    material[2, row, col] = 0.006
    material[3, row, col] = 1.0
    material[4, row, col] = 0.66
    material[5, row, col] = 1.0
    return material


class _IdentitySpatialDynamics:
    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variance = torch.full_like(material, 1e-6)
        return material, variance, torch.zeros_like(material)


def test_policy_descriptor_distinguishes_stop_and_passage_geometry() -> None:
    config = _config()
    stop = policy_descriptor((StrokeAction.stop_action(),), config)
    passage = policy_descriptor(
        (
            StrokeAction(0.1, 0.2, 0.5, 0.2, 0.08, 0.4, 1.0),
            StrokeAction(0.2, 0.4, 0.7, 0.5, 0.10, 0.6, 0.0),
        ),
        config,
    )

    assert stop.shape == (POLICY_DESCRIPTOR_DIM,)
    assert stop[0] == 1.0
    assert passage[0] == 0.0
    assert passage[1] > 0.0
    assert passage[6] > 0.0


def test_passage_step_descriptor_tracks_phase_without_mutating_slow_latent() -> None:
    passage = PassageLatent("band", 0.4, 0.6, 0.3, 0.4, 0.08, 3, 0.1, 0.7, 1.0)

    first = passage_step_descriptor(passage, 0)
    last = passage_step_descriptor(passage, 2)

    assert first.shape == (PASSAGE_STEP_DESCRIPTOR_DIM,)
    assert np.allclose(first[1:12], last[1:12])
    assert first[12] < last[12]
    assert first[13] > last[13]


def test_passage_latent_decodes_one_coarse_material_prediction_per_mark() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    current = torch.tensor(_field(config, slice(1, 3), slice(1, 3))).unsqueeze(0)
    model.reset_persistent_beliefs(current)
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.4, 0.08, 3, 0.1, 0.6, 1.0)
    actions = (
        StrokeAction(0.2, 0.3, 0.6, 0.3, 0.1, 0.6, 1.0),
        StrokeAction(0.2, 0.5, 0.6, 0.5, 0.1, 0.6, 1.0),
        StrokeAction(0.2, 0.7, 0.6, 0.7, 0.1, 0.6, 1.0),
        StrokeAction.stop_action(),
    )

    prediction = model.predict_passage_trajectory(Policy(actions, passage=passage))

    assert prediction is not None
    assert prediction.canvas_mean.shape == (3, model.canvas_latent_dim)
    assert prediction.relational_mean.shape == (3, config.relational_latent_dim)
    assert prediction.coarse_material_mean.shape == (
        3,
        config.spatial_material_channels,
        config.spatial_grid_size,
        config.spatial_grid_size,
    )
    assert prediction.step_indices == (0, 1, 2)


def test_relational_observation_tracks_pairwise_spatial_relationships() -> None:
    config = _config()
    near = _field(config, slice(1, 3), slice(1, 3))
    near[:, 1:3, 4:6] = _field(config, slice(1, 3), slice(4, 6))[:, 1:3, 4:6]
    far = _field(config, slice(1, 3), slice(1, 3))
    far[:, 5:7, 5:7] = _field(config, slice(5, 7), slice(5, 7))[:, 5:7, 5:7]

    near_vector = relational_observation_vector(near, config)
    far_vector = relational_observation_vector(far, config)

    assert near_vector.shape == (relational_observation_dim(config),)
    assert far_vector.shape == near_vector.shape
    assert not np.allclose(near_vector, far_vector)


def test_relational_observation_subdivides_large_connected_material_mass() -> None:
    config = _config()
    connected = _field(config, slice(0, 8), slice(0, 8))

    vector = relational_observation_vector(connected, config)
    slots = vector[: config.mark_slot_count * 12].reshape(config.mark_slot_count, 12)

    assert np.count_nonzero(slots[:, 0] > 0.5) == config.mark_slot_count
    assert np.unique(np.round(slots[:, 1:3], 3), axis=0).shape[0] == config.mark_slot_count


def test_hierarchy_has_spatial_canvas_and_relational_likelihoods() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    fields = torch.tensor(
        np.stack(
            [
                _field(config, slice(1, 3), slice(1, 4)),
                _field(config, slice(4, 7), slice(3, 6)),
            ]
        )
    )

    canvas_mean, canvas_logvar = model.encode_canvas(fields)
    relational = model.relational_observations(fields)
    relation_mean, relation_logvar = model.encode_relations(relational)
    loss = model.training_loss(fields)

    assert canvas_mean.shape == (2, config.canvas_latent_channels * 2 * 2)
    assert canvas_logvar.shape == canvas_mean.shape
    assert relational.shape == (2, relational_observation_dim(config))
    assert relation_mean.shape == (2, config.relational_latent_dim)
    assert relation_logvar.shape == relation_mean.shape
    assert torch.isfinite(loss)


def test_transition_efe_is_gated_until_passage_likelihood_has_trained() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    current = torch.tensor(_field(config, slice(1, 3), slice(1, 3))).unsqueeze(0)
    terminal = torch.tensor(_field(config, slice(4, 7), slice(4, 7))).unsqueeze(0)
    descriptor = torch.tensor(
        policy_descriptor((StrokeAction(0.2, 0.2, 0.8, 0.8, 0.1, 0.5, 1.0),), config)
    ).unsqueeze(0)
    model.reset_persistent_beliefs(current)

    canvas_risk, relational_risk = model.transition_efe_terms(terminal, descriptor)
    assert canvas_risk.item() == 0.0
    assert relational_risk.item() == 0.0

    model.mark_transition_update()
    canvas_risk, relational_risk = model.transition_efe_terms(terminal, descriptor)
    assert torch.isfinite(canvas_risk).all()
    assert torch.isfinite(relational_risk).all()
    assert canvas_risk.item() >= 0.0
    assert relational_risk.item() >= 0.0


def test_canvas_and_relational_transition_risks_enter_efe_separately() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    material = _field(config, slice(1, 4), slice(1, 4))
    belief = SpatialCanvasState(material, np.full_like(material, -8.0))
    model.reset_persistent_beliefs(torch.tensor(material).unsqueeze(0))
    model.mark_transition_update()
    efe = SpatialExpectedFreeEnergy(
        config,
        _IdentitySpatialDynamics(),
        TerminalCoveragePreference(config),
        composition=model,
    )
    policy = Policy(
        (
            StrokeAction(0.2, 0.2, 0.7, 0.6, 0.1, 0.5, 1.0),
            StrokeAction.stop_action(),
        )
    )

    components = efe.evaluate(belief, policy)
    expected = (
        components.terminal_risk
        + components.ambiguity
        + components.transition_risk
        + components.transition_ambiguity
        + components.composition_risk
        + components.canvas_transition_risk
        + components.relational_transition_risk
        + components.motor_risk
        + components.motor_ambiguity
        - components.motor_epistemic_value
    )

    assert components.canvas_transition_risk >= 0.0
    assert components.relational_transition_risk >= 0.0
    assert components.total == pytest.approx(expected)


def test_high_level_posteriors_update_only_at_explicit_passage_boundary() -> None:
    config = _config()
    agent = SpatialActiveInferencePainter(config, seed=4, device="cpu")
    before_material = _field(config, slice(1, 3), slice(1, 3))
    after_material = _field(config, slice(1, 5), slice(1, 4))
    before = SpatialCanvasState(before_material, np.full_like(before_material, -8.0))
    after = SpatialCanvasState(after_material, np.full_like(after_material, -8.0))
    action = StrokeAction(0.2, 0.2, 0.7, 0.6, 0.1, 0.5, 1.0)

    agent.reset_hierarchy_beliefs(before)
    assert agent.composition is not None
    assert agent.composition.canvas_belief is not None
    assert agent.composition.canvas_belief.update_count == 0

    agent.reset_belief(after)
    assert agent.composition.canvas_belief.update_count == 0

    agent.add_passage_transition(before, (action,), after)
    agent.update_hierarchy_beliefs(after, (action,))
    assert agent.composition.canvas_belief.update_count == 1
    assert agent.composition.relational_belief is not None
    assert agent.composition.relational_belief.update_count == 1
    assert len(agent.passage_replay) == 1


def test_passage_transition_training_marks_likelihood_as_available() -> None:
    config = _config()
    agent = SpatialActiveInferencePainter(config, seed=5, device="cpu")
    action = StrokeAction(0.2, 0.2, 0.7, 0.6, 0.1, 0.5, 1.0)
    for offset in (0, 1):
        before_material = _field(config, slice(1 + offset, 3 + offset), slice(1, 3))
        after_material = _field(config, slice(1 + offset, 4 + offset), slice(1, 5))
        before = SpatialCanvasState(before_material, np.full_like(before_material, -8.0))
        after = SpatialCanvasState(after_material, np.full_like(after_material, -8.0))
        agent.add_passage_transition(before, (action,), after)

    assert agent.composition is not None
    agent._train_hierarchy_transitions((config.spatial_material_channels, 8, 8))

    assert agent.last_hierarchy_transition_loss is not None
    assert np.isfinite(agent.last_hierarchy_transition_loss)
    assert int(agent.composition.transition_update_count.item()) == 1


def test_per_mark_passage_likelihood_trains_without_fast_global_posterior_update() -> None:
    config = _config()
    agent = SpatialActiveInferencePainter(config, seed=6, device="cpu")
    passage = PassageLatent("chain", 0.5, 0.5, 0.0, 0.3, 0.07, 2, 0.08, 0.5, 1.0)
    for offset in (0, 1):
        before_material = _field(config, slice(1 + offset, 3 + offset), slice(1, 3))
        after_material = _field(config, slice(1 + offset, 4 + offset), slice(1, 5))
        before = SpatialCanvasState(before_material, np.full_like(before_material, -8.0))
        after = SpatialCanvasState(after_material, np.full_like(after_material, -8.0))
        if offset == 0:
            agent.reset_hierarchy_beliefs(before)
        agent.add_passage_step_transition(before, passage, offset, after)

    assert agent.composition is not None
    assert agent.composition.canvas_belief is not None
    assert agent.composition.canvas_belief.update_count == 0
    agent._train_hierarchy_transitions((config.spatial_material_channels, 8, 8))

    assert agent.last_passage_trajectory_loss is not None
    assert np.isfinite(agent.last_passage_trajectory_loss)
    assert int(agent.composition.passage_trajectory_update_count.item()) == 1
    assert agent.composition.canvas_belief.update_count == 0


def test_trained_passage_trajectory_likelihood_is_used_for_structured_policy_efe() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    material = _field(config, slice(1, 4), slice(1, 4))
    belief = SpatialCanvasState(material, np.full_like(material, -8.0))
    model.reset_persistent_beliefs(torch.tensor(material).unsqueeze(0))
    model.mark_passage_trajectory_update()
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.4, 0.08, 2, 0.1, 0.6, 1.0)
    policy = Policy(
        (
            StrokeAction(0.2, 0.3, 0.6, 0.3, 0.1, 0.6, 1.0),
            StrokeAction(0.2, 0.5, 0.6, 0.5, 0.1, 0.6, 1.0),
            StrokeAction.stop_action(),
        ),
        passage=passage,
    )
    efe = SpatialExpectedFreeEnergy(
        config,
        _IdentitySpatialDynamics(),
        TerminalCoveragePreference(config),
        composition=model,
    )

    components = efe.evaluate(belief, policy)

    assert components.hierarchy_transition_mode == "passage_trajectory"
    assert components.passage_trajectory_steps == 2
    assert components.canvas_transition_risk == 0.0
    assert components.relational_transition_risk == 0.0
    assert components.passage_canvas_trajectory_risk >= 0.0
    assert components.passage_relational_trajectory_risk >= 0.0
    assert components.passage_trajectory_observation_count == pytest.approx(2.0)


def test_passage_trajectory_efe_observes_every_intermediate_mark() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    material = _field(config, slice(1, 4), slice(1, 4))
    model.reset_persistent_beliefs(torch.tensor(material).unsqueeze(0))
    model.mark_passage_trajectory_update()
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.4, 0.08, 2, 0.1, 0.6, 1.0)
    policy = Policy(
        (
            StrokeAction(0.2, 0.3, 0.6, 0.3, 0.1, 0.6, 1.0),
            StrokeAction(0.2, 0.5, 0.6, 0.5, 0.1, 0.6, 1.0),
            StrokeAction.stop_action(),
        ),
        passage=passage,
    )
    fields = torch.tensor(np.stack([material, material]))

    canvas, relational, valid = model.passage_trajectory_efe_terms(
        fields,
        [policy, policy],
        [0, 1],
    )

    assert valid.tolist() == [True, True]
    assert canvas.shape == (2,)
    assert relational.shape == (2,)
    assert torch.isfinite(canvas).all()
    assert torch.isfinite(relational).all()


def test_polyline_passage_likelihood_waits_for_kind_specific_training_support() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    material = _field(config, slice(1, 4), slice(1, 4))
    model.reset_persistent_beliefs(torch.tensor(material).unsqueeze(0))
    model.mark_passage_trajectory_update()
    passage = PassageLatent("polyline", 0.5, 0.5, 0.0, 0.5, 0.4, 2, 0.1, 0.6, 1.0)
    actions = PolicySampler(config, seed=1).passage_actions(passage)
    policy = Policy(tuple(actions) + (StrokeAction.stop_action(),), passage=passage)
    fields = torch.tensor(np.stack([material, material]))

    _, _, unsupported = model.passage_trajectory_efe_terms(fields, [policy, policy], [0, 1])
    descriptor = torch.tensor(passage_step_descriptor(passage, 0)).reshape(1, -1)
    model.mark_passage_trajectory_update(descriptor)
    canvas, relational, supported = model.passage_trajectory_efe_terms(
        fields,
        [policy, policy],
        [0, 1],
    )

    assert unsupported.tolist() == [False, False]
    assert supported.tolist() == [True, True]
    assert torch.isfinite(canvas).all()
    assert torch.isfinite(relational).all()


def test_passage_trajectory_evaluation_reports_shuffled_conditioning_baseline() -> None:
    config = _config()
    model = HierarchicalCanvasModel(config)
    before = torch.tensor(
        np.stack(
            [
                _field(config, slice(1, 3), slice(1, 3)),
                _field(config, slice(4, 6), slice(4, 6)),
            ]
        )
    )
    after = torch.tensor(
        np.stack(
            [
                _field(config, slice(1, 4), slice(1, 4)),
                _field(config, slice(3, 7), slice(3, 7)),
            ]
        )
    )
    passages = (
        PassageLatent("band", 0.3, 0.3, 0.0, 0.3, 0.07, 2, 0.1, 0.5, 1.0),
        PassageLatent("radial", 0.7, 0.7, 1.0, 0.2, 0.1, 3, 0.2, 0.8, 0.0),
    )
    descriptors = torch.tensor(
        np.stack([passage_step_descriptor(passage, 0) for passage in passages])
    )

    metrics = model.passage_trajectory_evaluation(before, descriptors, after)

    assert set(metrics) == {
        "canvasKLNatsPerDim",
        "relationalKLNatsPerDim",
        "canvasCrossEntropyNatsPerDim",
        "relationalCrossEntropyNatsPerDim",
        "shuffledCrossEntropyNatsPerDim",
        "conditioningGainNatsPerDim",
    }
    assert all(np.isfinite(value) for value in metrics.values())
