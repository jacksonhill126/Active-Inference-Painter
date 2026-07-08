import numpy as np
import pytest

from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.motor_planning import motor_realization_policy_alternatives
from active_painter.policies import MotorPrimitiveLatent, PassageLatent, PassagePlanLatent, Policy, PolicySampler


def test_policy_sampler_includes_immediate_stop_and_only_terminal_stop() -> None:
    cfg = PainterConfig(candidate_policies=24, planning_horizon=4)
    policies = PolicySampler(cfg, seed=11).sample()

    assert len(policies) == cfg.candidate_policies
    assert len(policies[0].actions) == 1
    assert policies[0].actions[0].stop
    assert all(policy.actions[-1].stop for policy in policies)
    assert all(not action.stop for policy in policies[1:] for action in policy.actions[:-1])
    assert any(len(policy.actions) > 2 for policy in policies)


def test_policy_sampler_tone_prior_can_load_brush_or_leave_tone_random() -> None:
    black_cfg = PainterConfig(candidate_policies=8, planning_horizon=1, stroke_tone_prior=1.0)
    white_cfg = PainterConfig(candidate_policies=8, planning_horizon=1, stroke_tone_prior=0.0)
    random_cfg = PainterConfig(candidate_policies=24, planning_horizon=1, stroke_tone_prior=None)

    black_tones = [policy.actions[0].tone for policy in PolicySampler(black_cfg, seed=3).sample()[1:]]
    white_tones = [policy.actions[0].tone for policy in PolicySampler(white_cfg, seed=3).sample()[1:]]
    random_tones = [policy.actions[0].tone for policy in PolicySampler(random_cfg, seed=3).sample()[1:]]

    assert set(black_tones) == {1.0}
    assert set(white_tones) == {0.0}
    assert set(random_tones) == {0.0, 1.0}


def test_random_tone_support_pairs_same_mark_geometry_as_policy_alternatives() -> None:
    cfg = PainterConfig(
        candidate_policies=9,
        planning_horizon=1,
        passage_proposal_mix=0.0,
        stroke_tone_prior=None,
    )

    policies = PolicySampler(cfg, seed=4).sample()[1:]
    by_geometry: dict[tuple[float, ...], set[float]] = {}
    for policy in policies:
        action = policy.actions[0]
        key = (action.x0, action.y0, action.x1, action.y1, action.width, action.amount)
        by_geometry.setdefault(key, set()).add(action.tone)

    assert by_geometry
    assert all(tones == {0.0, 1.0} for tones in by_geometry.values())


def test_random_tone_support_pairs_same_passage_geometry_as_policy_alternatives() -> None:
    cfg = PainterConfig(
        candidate_policies=9,
        planning_horizon=3,
        passage_proposal_mix=1.0,
        stroke_tone_prior=None,
    )

    policies = PolicySampler(cfg, seed=12).sample()[1:]
    by_geometry: dict[tuple[tuple[float, ...], ...], set[float]] = {}
    for policy in policies:
        key = tuple(
            (action.x0, action.y0, action.x1, action.y1, action.width, action.amount)
            for action in policy.actions[:-1]
        )
        tones = {action.tone for action in policy.actions[:-1]}
        assert len(tones) == 1
        by_geometry.setdefault(key, set()).update(tones)

    assert by_geometry
    assert all(tones == {0.0, 1.0} for tones in by_geometry.values())


def test_policy_rejects_intermediate_stop_actions() -> None:
    stroke = StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0)
    with pytest.raises(ValueError, match="Stop may appear only as the final"):
        Policy((StrokeAction.stop_action(), stroke, StrokeAction.stop_action()))


def test_policy_carries_motor_realization_latent_for_first_non_stop_mark() -> None:
    stroke = StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0)
    primitive = MotorPrimitiveLatent("elbow_pivot", pivot_joint="elbow")

    policy = Policy((stroke, StrokeAction.stop_action()), motor_primitive=primitive)

    assert policy.motor_primitive is primitive
    with pytest.raises(ValueError, match="motor realization latent requires a non-stop"):
        Policy((StrokeAction.stop_action(),), motor_primitive=primitive)


def test_motor_realization_alternatives_are_policy_latents_not_extra_actions() -> None:
    cfg = PainterConfig(
        motor_realization_kinds=("cartesian_ik", "joint_spline", "elbow_pivot"),
        motor_realization_candidate_limit=3,
    )
    stroke = StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0)
    base = Policy((stroke, StrokeAction.stop_action()))

    alternatives = motor_realization_policy_alternatives(base, cfg)

    assert [policy.motor_primitive.kind for policy in alternatives if policy.motor_primitive is not None] == [
        "cartesian_ik",
        "joint_spline",
        "elbow_pivot",
    ]
    assert all(policy.actions == base.actions for policy in alternatives)


def test_policy_rejects_single_mark_passage_metadata() -> None:
    stroke = StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0)
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.4, 0.08, 1, 0.08, 0.5, 1.0)

    with pytest.raises(ValueError, match="passage policy must contain multiple marks"):
        Policy((stroke, StrokeAction.stop_action()), passage=passage)


def test_policy_rejects_single_passage_plan_metadata() -> None:
    stroke = StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0)
    passage = PassageLatent("band", 0.5, 0.5, 0.0, 0.4, 0.08, 1, 0.08, 0.5, 1.0)
    plan = PassagePlanLatent("progression", 0.5, 0.5, 0.0, 1, 1, 0.2, 0.0, 0.08, 0.5, 1.0, (passage,))

    with pytest.raises(ValueError, match="multiple passages"):
        Policy((stroke, StrokeAction.stop_action()), passage_plan=plan)


def test_policy_sampler_includes_higher_level_passage_candidates() -> None:
    cfg = PainterConfig(
        candidate_policies=40,
        planning_horizon=4,
        passage_proposal_mix=0.5,
        passage_min_strokes=3,
        passage_max_strokes=4,
    )

    policies = PolicySampler(cfg, seed=9).sample()
    passages = [policy for policy in policies if policy.passage is not None]

    assert passages
    assert all(policy.actions[-1].stop for policy in passages)
    assert all(3 <= len(policy.actions) - 1 <= 4 for policy in passages)
    assert all(policy.passage is not None and policy.passage.stroke_count == len(policy.actions) - 1 for policy in passages)
    assert all(
        len({action.tone for action in policy.actions[:-1]}) == 1
        for policy in passages
    )


def test_passage_candidates_are_disabled_by_zero_passage_mix() -> None:
    cfg = PainterConfig(
        candidate_policies=32,
        planning_horizon=4,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=0.0,
    )

    policies = PolicySampler(cfg, seed=9).sample()

    assert all(policy.passage is None for policy in policies)
    assert all(policy.passage_plan is None for policy in policies)
    assert any(len(policy.actions) > 2 for policy in policies)


def test_policy_sampler_includes_hierarchical_passage_plan_candidates() -> None:
    cfg = PainterConfig(
        candidate_policies=48,
        planning_horizon=6,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=1.0,
        passage_plan_min_passages=2,
        passage_plan_max_passages=3,
        passage_min_strokes=2,
        passage_max_strokes=3,
    )

    policies = PolicySampler(cfg, seed=17).sample()
    plans = [policy for policy in policies if policy.passage_plan is not None]

    assert plans
    assert all(policy.actions[-1].stop for policy in plans)
    assert all(policy.passage is None for policy in plans)
    assert all(policy.passage_plan is not None and policy.passage_plan.passage_count >= 2 for policy in plans)
    assert all(policy.passage_plan is not None and len(policy.passage_plan.passages) == policy.passage_plan.passage_count for policy in plans)
    assert all(policy.passage_plan is not None and policy.passage_plan.total_stroke_count == len(policy.actions) - 1 for policy in plans)
    assert all(
        len({action.tone for action in policy.actions[:-1]}) == 1
        for policy in plans
    )


def test_random_tone_support_pairs_same_passage_plan_geometry_as_policy_alternatives() -> None:
    cfg = PainterConfig(
        candidate_policies=9,
        planning_horizon=4,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=1.0,
        passage_plan_min_passages=2,
        passage_plan_max_passages=2,
        stroke_tone_prior=None,
    )

    policies = PolicySampler(cfg, seed=18).sample()[1:]
    by_geometry: dict[tuple[tuple[float, ...], ...], set[float]] = {}
    for policy in policies:
        assert policy.passage_plan is not None
        key = tuple(
            (action.x0, action.y0, action.x1, action.y1, action.width, action.amount)
            for action in policy.actions[:-1]
        )
        tones = {action.tone for action in policy.actions[:-1]}
        assert len(tones) == 1
        by_geometry.setdefault(key, set()).update(tones)

    assert by_geometry
    assert all(tones == {0.0, 1.0} for tones in by_geometry.values())


def test_passage_prior_can_target_low_coverage_regions_as_a_latent_policy_prior() -> None:
    cfg = PainterConfig(
        candidate_policies=36,
        planning_horizon=4,
        passage_proposal_mix=1.0,
        proposal_low_coverage_mix=1.0,
    )
    coverage = np.zeros((16, 16), dtype=np.float32)
    coverage[:, :8] = 1.0

    policies = PolicySampler(cfg, seed=5).sample(coverage)
    passage_centers_x = [policy.passage.center_x for policy in policies if policy.passage is not None]

    assert passage_centers_x
    assert float(np.mean(passage_centers_x)) > 0.6


def test_stroke_proposals_prefer_low_coverage_regions_when_field_given() -> None:
    cfg = PainterConfig(candidate_policies=64, planning_horizon=1, proposal_low_coverage_mix=1.0)
    coverage = np.zeros((16, 16), dtype=np.float32)
    coverage[:, :8] = 1.0  # left half fully covered

    policies = PolicySampler(cfg, seed=5).sample(coverage)
    starts_x = [policy.actions[0].x0 for policy in policies[1:]]

    assert float(np.mean(starts_x)) > 0.6


def test_stroke_proposals_ignore_field_at_zero_mix() -> None:
    cfg = PainterConfig(candidate_policies=64, planning_horizon=1, proposal_low_coverage_mix=0.0)
    coverage = np.zeros((16, 16), dtype=np.float32)
    coverage[:, :8] = 1.0

    policies = PolicySampler(cfg, seed=5).sample(coverage)
    starts_x = [policy.actions[0].x0 for policy in policies[1:]]

    assert 0.4 < float(np.mean(starts_x)) < 0.6
