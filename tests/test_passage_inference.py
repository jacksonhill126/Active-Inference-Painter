import numpy as np

from active_painter.arm_agent_driver import ArmActiveInferenceDriver
from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.passage_inference import PassageBelief, infer_passage_observation
from active_painter.policies import PassageLatent
from active_painter.spatial_state import spatial_canvas_state


def test_pixel_mark_likelihood_updates_slow_passage_posterior_without_copying_observation() -> None:
    cfg = PainterConfig(canvas_size=48, spatial_grid_size=16, planner_state_kind="spatial_material")
    sim = ArmPainterSim(cfg)
    passage = PassageLatent("band", 0.42, 0.48, 0.1, 0.32, 0.08, 3, 0.08, 0.55, 1.0)
    action = StrokeAction(0.25, 0.45, 0.57, 0.48, 0.08, 0.55, 1.0)
    before = spatial_canvas_state(sim, cfg)
    sim.canvas.paint_at(np.asarray([-1.0, sim.canvas.distance, 0.0]), pressure=0.8, tone=1.0, dt=0.2)
    after = spatial_canvas_state(sim, cfg)
    observation = infer_passage_observation(before, after, action, passage, 0, cfg)
    prior = PassageBelief.from_latent(passage, cfg)

    posterior = prior.update(observation, cfg)

    assert posterior.update_count == 1
    assert posterior.variance[0] < prior.variance[0]
    assert posterior.mean[0] != observation.mean[0]
    assert posterior.black_probability > prior.black_probability - 1e-8


def test_surface_tone_likelihood_updates_tone_factor_separately_from_geometry() -> None:
    cfg = PainterConfig(canvas_size=48, spatial_grid_size=16, planner_state_kind="spatial_material")
    sim = ArmPainterSim(cfg)
    passage = PassageLatent("chain", 0.5, 0.5, 0.0, 0.3, 0.07, 2, 0.08, 0.5, 0.0)
    action = StrokeAction(0.35, 0.5, 0.65, 0.5, 0.08, 0.5, 1.0)
    before = spatial_canvas_state(sim, cfg)
    sim.canvas.paint_at(np.asarray([0.0, sim.canvas.distance, 0.0]), pressure=0.8, tone=1.0, dt=0.2)
    after = spatial_canvas_state(sim, cfg)
    observation = infer_passage_observation(before, after, action, passage, 0, cfg)
    prior = PassageBelief.from_latent(passage, cfg)

    posterior = prior.update(observation, cfg)

    assert observation.black_probability > 0.5
    assert posterior.black_probability > prior.black_probability
    assert posterior.mean.shape == (7,)


def test_local_passage_candidates_include_immediate_stop_and_paired_tone_consequences() -> None:
    cfg = PainterConfig(
        candidate_policies=4,
        planning_horizon=3,
        passage_local_candidate_policies=5,
        stroke_tone_prior=None,
    )
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.3, 0.08, 3, 0.08, 0.5, 1.0)
    driver._active_passage = passage
    driver._active_passage_total_strokes = 3
    driver._active_passage_completed_strokes = 1
    driver._passage_belief = PassageBelief.from_latent(passage, cfg)

    policies, log_priors, _ = driver._local_passage_candidates()

    assert policies[0].actions == (StrokeAction.stop_action(),)
    assert len(policies) == len(log_priors) == cfg.passage_local_candidate_policies
    assert all(policy.actions[-1].stop for policy in policies)
    assert {policy.actions[0].tone for policy in policies[1:3]} == {0.0, 1.0}
    assert log_priors[0] == np.log1p(-cfg.passage_continuation_probability)


def test_local_passage_plan_preserves_slow_belief_and_completed_mark_count() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        candidate_policies=3,
        planning_horizon=2,
        passage_local_candidate_policies=3,
        motor_forecast_candidates=1,
        motor_forecast_samples=1,
    )
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    driver.reset(sim)
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.25, 0.07, 2, 0.08, 0.5, 1.0)
    driver._active_passage = passage
    driver._active_passage_total_strokes = 2
    driver._active_passage_completed_strokes = 1
    driver._passage_belief = PassageBelief.from_latent(passage, cfg)
    driver._passage_queue = [StrokeAction(0.4, 0.5, 0.6, 0.5, 0.08, 0.5, 1.0)]

    driver._background_local_passage_plan(sim)
    driver._consume_background_plan()

    assert driver._pending_error is None
    assert driver.current is not None
    assert driver._active_passage_completed_strokes == 1
    assert driver._passage_belief is not None
    assert driver.last_planning_profile["scope"] == "passage_local"
