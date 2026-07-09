import json
import time

import numpy as np

from active_painter.arm_control import ik_pose_for_canvas_point
from active_painter.arm_agent_driver import (
    ArmActiveInferenceDriver,
    StrokeExecution,
    canvas_summary_state,
    execute_stroke_action,
    pose_for_execution,
)
from active_painter.arm_sim import ArmPainterSim, ArmPose
from active_painter.config import PainterConfig
from active_painter.efe import EFEComponents
from active_painter.env import StrokeAction
from active_painter.policies import MotorPrimitiveLatent, PassageLatent, PassagePlanLatent, Policy
from active_painter.spatial_state import SpatialCanvasState
from active_painter.stroke_execution import ExecutionForecast, StrokeTiming, adaptive_stroke_timing


def make_driver() -> ArmActiveInferenceDriver:
    return ArmActiveInferenceDriver(bootstrap_transitions=72, bootstrap_train_steps=24)


def wait_for_driver(driver: ArmActiveInferenceDriver, sim: ArmPainterSim, timeout: float = 15.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)
        if driver.current is not None or driver.stopped:
            return
        time.sleep(0.01)
    raise AssertionError(f"driver did not finish background planning within {timeout:.1f}s")


def test_active_inference_driver_selects_stroke_at_low_coverage() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = make_driver()
    wait_for_driver(driver, sim)
    assert driver.current is not None
    assert not driver.current.action.stop
    assert driver.last_components is not None
    assert driver.trained_transitions >= 72


def test_active_inference_driver_reports_efe_decomposition() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = make_driver()
    wait_for_driver(driver, sim)
    diag = driver.diagnostics()
    assert diag["efe"] is not None
    assert "terminal_risk" in diag["efe"]
    assert "ambiguity" in diag["efe"]
    assert "epistemic_value" in diag["efe"]
    assert "terminal_entropy" in diag["efe"]
    assert "pragmatic_value" in diag["efe"]
    assert "transition_risk" in diag["efe"]
    assert "transition_ambiguity" in diag["efe"]
    assert diag["transitionModel"].startswith("learned DynamicsEnsemble")


def test_active_inference_driver_reports_policy_and_state_distributions() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = make_driver()
    wait_for_driver(driver, sim)
    diag = driver.diagnostics()
    assert diag["policyPrecision"] < 1.0
    assert diag["posteriorEntropy"] > 0.0
    assert len(diag["belief"]["mean"]) == 6
    assert len(diag["belief"]["std"]) == 6
    assert 0.0 < diag["topPolicies"][0]["posterior"] <= 1.0


def test_active_inference_driver_reports_passage_posterior_mass() -> None:
    cfg = PainterConfig(candidate_policies=4, passage_proposal_mix=0.0)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    stroke_a = StrokeAction(0.2, 0.3, 0.7, 0.3, 0.08, 0.5, 1.0)
    stroke_b = StrokeAction(0.2, 0.45, 0.7, 0.45, 0.08, 0.5, 1.0)
    passage = PassageLatent("band", 0.45, 0.37, 0.0, 0.5, 0.15, 2, 0.08, 0.5, 1.0)
    component = EFEComponents(1.0, 0.7, 0.1, 0.0, 0.5, 0.1)

    driver.last_ranked = [
        (Policy((stroke_a, stroke_b, StrokeAction.stop_action()), passage=passage), component, 0.65),
        (Policy((stroke_a, StrokeAction.stop_action())), component, 0.35),
    ]
    driver.last_components = component

    diag = driver.diagnostics()

    assert diag["passageCandidateCount"] == 1
    assert np.isclose(diag["passagePosteriorMass"], 0.65)
    assert diag["topPolicies"][0]["passage"]["kind"] == "band"
    assert diag["topPolicies"][0]["passage"]["stroke_count"] == 2


def test_active_inference_driver_reports_passage_plan_posterior_mass() -> None:
    cfg = PainterConfig(candidate_policies=2)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    stroke_a = StrokeAction(0.2, 0.3, 0.7, 0.3, 0.08, 0.5, 1.0)
    stroke_b = StrokeAction(0.2, 0.45, 0.7, 0.45, 0.08, 0.5, 1.0)
    first = PassageLatent("band", 0.45, 0.30, 0.0, 0.5, 0.08, 2, 0.08, 0.5, 1.0)
    second = PassageLatent("chain", 0.55, 0.45, 0.2, 0.4, 0.07, 2, 0.08, 0.5, 1.0)
    plan = PassagePlanLatent("progression", 0.5, 0.4, 0.0, 2, 4, 0.2, 0.1, 0.08, 0.5, 1.0, (first, second))
    component = EFEComponents(1.0, 0.7, 0.1, 0.0, 0.5, 0.1)
    driver.last_ranked = [
        (Policy((stroke_a, stroke_b, stroke_a, stroke_b, StrokeAction.stop_action()), passage_plan=plan), component, 0.55),
        (Policy((stroke_a, StrokeAction.stop_action())), component, 0.45),
    ]

    diag = driver.diagnostics()

    assert diag["passagePlanCandidateCount"] == 1
    assert np.isclose(diag["passagePlanPosteriorMass"], 0.55)
    assert diag["topPolicies"][0]["passagePlan"]["kind"] == "progression"
    assert diag["topPolicies"][0]["passagePlan"]["passage_count"] == 2


def test_spatial_material_driver_reports_spatial_planner_state() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        candidate_policies=6,
        planning_horizon=1,
        motor_forecast_candidates=2,
        policy_precision=0.35,
        batch_size=4,
    )
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)

    driver._background_plan(driver._planner_state(sim), None, sim)
    assert driver._pending_error is None
    driver._consume_background_plan()
    diagnostics = driver.diagnostics()

    json.dumps(diagnostics)
    assert driver.current is not None
    assert diagnostics["stateRepresentation"].startswith("Spatial Gaussian q(s) with pixel-local rollouts")
    assert diagnostics["transitionModel"].startswith("learned LocalSpatialDynamicsEnsemble")
    assert diagnostics["spatialTransitionMode"] == "local_patch"
    assert diagnostics["belief"]["names"] == [
        "thickness",
        "wetness",
        "black_mass",
        "observed_tone",
        "ground_contrast",
        "material_coverage",
    ]
    assert diagnostics["spatialBelief"]["gridSize"] == 8
    assert diagnostics["markEvents"]["activeCount"] >= 0
    assert "not a policy preference" in diagnostics["markEvents"]["approximation"]
    assert "terminal_risk" in diagnostics["efe"]
    assert diagnostics["efe"]["execution_forecast_used"] is True
    assert diagnostics["efe"]["rollout_mode"] == "local_patch"
    assert diagnostics["efe"]["rollout_grid_size"] == cfg.canvas_size
    assert diagnostics["topPolicies"][0]["motorFeasible"] is True


def test_driver_reports_motor_primitive_policy_latents_and_efe_terms() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        candidate_policies=4,
        planning_horizon=1,
        motor_forecast_candidates=1,
        motor_realization_kinds=("cartesian_ik", "joint_spline", "elbow_pivot"),
        motor_realization_candidate_limit=3,
        batch_size=4,
        policy_precision=0.35,
    )
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)

    driver._background_plan(canvas_summary_state(sim), None, sim)
    assert driver._pending_error is None
    driver._consume_background_plan()
    diagnostics = driver.diagnostics()

    json.dumps(diagnostics)
    assert diagnostics["motorPrimitiveCandidateCount"] >= 3
    assert diagnostics["topPolicies"][0]["motorPrimitive"] is not None
    assert "motorRisk" in diagnostics["topPolicies"][0]
    assert "motorAmbiguity" in diagnostics["topPolicies"][0]
    assert diagnostics["efe"]["motor_risk"] >= 0.0
    assert diagnostics["efe"]["motor_ambiguity"] >= 0.0


def test_summary_driver_replay_stores_selected_motor_realization_condition() -> None:
    cfg = PainterConfig(candidate_policies=2, motor_realization_candidate_limit=3)
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    before = canvas_summary_state(sim)
    after = before.copy()
    after[0] += 0.05
    action = StrokeAction(0.2, 0.3, 0.7, 0.4, 0.08, 0.6, 1.0)

    driver._add_transition_to_agent(before, action, after, MotorPrimitiveLatent("elbow_pivot"))

    stored_action = driver.agent.replay.data[-1][1]
    assert stored_action.shape == (cfg.action_dim,)
    assert np.allclose(stored_action[7:], [0.0, 0.0, 1.0])


def test_spatial_material_driver_can_use_dense_grid_transition_mode() -> None:
    cfg = PainterConfig(
        canvas_size=24,
        planner_state_kind="spatial_material",
        spatial_transition_mode="dense_grid",
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        candidate_policies=4,
        planning_horizon=1,
        motor_forecast_candidates=1,
        batch_size=4,
    )
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)

    driver._background_plan(driver._planner_state(sim), None, sim)
    assert driver._pending_error is None
    driver._consume_background_plan()
    diagnostics = driver.diagnostics()

    assert diagnostics["stateRepresentation"].startswith("Spatial Gaussian q(s_grid)")
    assert diagnostics["transitionModel"].startswith("learned SpatialDynamicsEnsemble")
    assert diagnostics["spatialTransitionMode"] == "dense_grid"
    assert diagnostics["efe"]["rollout_mode"] == "dense_grid"


def test_spatial_execution_forecast_covariance_propagates_to_material_variance() -> None:
    cfg = PainterConfig(planner_state_kind="spatial_material", spatial_grid_size=8)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    sim = ArmPainterSim(cfg)
    material = np.zeros((6, 8, 8), dtype=np.float32)
    next_material = material.copy()
    next_material[0] = np.linspace(0.0, 0.02, 8, dtype=np.float32)[None, :]
    next_material[1] = 0.5 * next_material[0]
    next_material[2] = 0.25 * next_material[0]
    coverage = 1.0 - np.exp(-np.clip(next_material[0], 0.0, None) / cfg.thickness_scale)
    pigment_tone = np.clip(next_material[2] / np.maximum(next_material[0], 1e-6), 0.0, 1.0)
    next_material[3] = np.clip((1.0 - coverage) * cfg.canvas_ground_tone + coverage * pigment_tone, 0.0, 1.0)
    next_material[4] = np.abs(next_material[3] - cfg.canvas_ground_tone)
    next_material[5] = coverage
    belief = SpatialCanvasState(material=material, logvar=np.full_like(material, -8.0))
    low = ExecutionForecast(
        next_state_mean=next_material.reshape(-1),
        next_state_variance=np.ones(next_material.size, dtype=np.float32) * 1e-6,
        canvas_delta_mean=next_material.reshape(-1),
        intended_start=(0.0, 0.0),
        intended_end=(1.0, 1.0),
        realized_start=(0.0, 0.0),
        realized_end=(1.0, 1.0),
        intended_path_length=1.414,
        realized_path_span=1.414,
        paint_motion_fraction=1.0,
        path_covariance=(0.0, 0.0),
        pressure_mean=0.5,
        pressure_variance=0.0,
        target_pressure_mean=0.5,
        contact_loss_probability=0.0,
        overshoot=0.0,
        execution_uncertainty=0.0,
        feasible=True,
    )
    high = ExecutionForecast(
        next_state_mean=low.next_state_mean,
        next_state_variance=low.next_state_variance,
        canvas_delta_mean=low.canvas_delta_mean,
        intended_start=low.intended_start,
        intended_end=low.intended_end,
        realized_start=low.realized_start,
        realized_end=low.realized_end,
        intended_path_length=low.intended_path_length,
        realized_path_span=low.realized_path_span,
        paint_motion_fraction=low.paint_motion_fraction,
        path_covariance=(0.35, 0.0),
        pressure_mean=low.pressure_mean,
        pressure_variance=0.08,
        target_pressure_mean=low.target_pressure_mean,
        contact_loss_probability=0.3,
        overshoot=low.overshoot,
        execution_uncertainty=0.9,
        feasible=True,
    )

    low_variance = driver._spatial_material_variance_from_forecast(belief, next_material, low, sim)
    high_variance = driver._spatial_material_variance_from_forecast(belief, next_material, high, sim)

    assert float(high_variance.mean()) > float(low_variance.mean())
    assert np.all(high_variance >= low_variance)


def test_active_inference_driver_diagnostics_with_execution_forecast_are_json_serializable() -> None:
    driver = ArmActiveInferenceDriver(bootstrap_transitions=0, bootstrap_train_steps=0)
    driver.last_execution_forecast = ExecutionForecast(
        next_state_mean=np.zeros(6, dtype=np.float32),
        next_state_variance=np.ones(6, dtype=np.float32),
        canvas_delta_mean=np.zeros(6, dtype=np.float32),
        intended_start=(0.0, 0.0),
        intended_end=(1.0, 1.0),
        realized_start=(0.0, 0.0),
        realized_end=(0.9, 0.9),
        intended_path_length=1.414,
        realized_path_span=1.273,
        paint_motion_fraction=0.9,
        path_covariance=(0.01, 0.02),
        pressure_mean=0.3,
        pressure_variance=0.04,
        target_pressure_mean=0.35,
        contact_loss_probability=0.1,
        overshoot=0.2,
        execution_uncertainty=0.4,
        feasible=True,
    )

    diagnostics = driver.diagnostics()

    json.dumps(diagnostics)
    assert isinstance(diagnostics["executionForecast"]["next_state_mean"], list)


def test_active_inference_driver_lifts_brush_while_waiting_for_background_plan() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(bootstrap_transitions=0, bootstrap_train_steps=0)
    contact_pose = ik_pose_for_canvas_point(0.0, 0.0, sim.canvas.distance)
    sim.actual_pose = contact_pose
    sim.target_pose = contact_pose
    sim.contact = sim.canvas.contact_from_tip(sim.kinematics.tip(sim.actual_pose), 1.0)
    sim.paint_enabled = True
    sim.intended_contact_pressure = 1.0
    driver.planning = True

    initial_tip_y = float(sim.kinematics.tip(sim.actual_pose)[1])
    for _ in range(180):
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)

    assert not sim.paint_enabled
    assert sim.intended_contact_pressure == 0.0
    assert float(sim.kinematics.tip(sim.actual_pose)[1]) < initial_tip_y - 0.05
    assert sim.contact.pressure == 0.0


def test_planning_hold_target_does_not_chase_actual_pose_drift() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(bootstrap_transitions=0, bootstrap_train_steps=0)
    driver.planning = True

    for _ in range(80):
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)
    held_target = sim.target_pose

    sim.actual_pose = ArmPose(
        yaw=held_target.yaw + 18.0,
        pitch=held_target.pitch - 12.0,
        roll=held_target.roll,
        elbow=held_target.elbow,
    ).clipped()
    driver.step(sim, 1.0 / 240.0)

    assert sim.target_pose == held_target


def test_global_hold_escapes_canvas_contact_before_center_translation() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(bootstrap_transitions=0, bootstrap_train_steps=0)
    sim.actual_pose = ArmPose(yaw=-16.37, pitch=-12.56, roll=0.0, elbow=77.38)
    sim.target_pose = sim.actual_pose
    sim.contact = sim.canvas.contact_from_tip(sim.kinematics.tip(sim.actual_pose), 0.0)

    initial_pressure = sim.contact.pressure
    for _ in range(80):
        driver._hold_retracted(sim, 1.0 / 240.0, scope="global")
        sim.step(1.0 / 240.0)

    assert initial_pressure > 0.9
    assert sim.contact.pressure == 0.0
    assert float(sim.kinematics.tip(sim.actual_pose)[1]) < sim.canvas.distance - 0.1


def test_driver_retracts_and_does_not_consume_pending_stroke_immediately_after_completion() -> None:
    cfg = PainterConfig(canvas_size=48, post_stroke_retract_seconds=0.2)
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    efe = EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ex = StrokeExecution(action=action, efe=efe, posterior=1.0, initial_state=canvas_summary_state(sim))
    ex.timing = adaptive_stroke_timing(sim, action)
    ex.controller.reset(sim, action, ex.timing)
    ex.initialized = True
    ex.t = ex.total - 0.001
    driver.current = ex

    driver.step(sim, 1.0 / 120.0)

    assert driver.current is None
    assert driver.phase_label() == "return_center"
    assert not sim.paint_enabled
    assert sim.intended_contact_pressure == 0.0

    pending = StrokeExecution(action=action, efe=efe, posterior=1.0)
    with driver._planner_lock:
        driver._pending_current = pending
        driver._pending_ranked = []
    driver.step(sim, 1.0 / 120.0)

    assert driver.current is None
    assert driver.phase_label() == "return_center"


def test_passage_queue_uses_local_hold_then_returns_center_after_final_mark() -> None:
    cfg = PainterConfig(
        canvas_size=48,
        passage_local_retract_seconds=0.02,
        passage_center_retract_seconds=0.2,
    )
    sim = ArmPainterSim(cfg)
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    first = StrokeAction(0.22, 0.30, 0.36, 0.34, 0.08, 0.55, 1.0)
    second = StrokeAction(0.25, 0.35, 0.39, 0.39, 0.08, 0.55, 1.0)
    passage = PassageLatent("chain", 0.30, 0.35, 0.0, 0.2, 0.08, 2, 0.08, 0.55, 1.0)
    efe = EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ex = StrokeExecution(action=first, efe=efe, posterior=1.0, initial_state=canvas_summary_state(sim))
    ex.timing = StrokeTiming(approach=0.01, press=0.01, paint=0.01, lift=0.01)
    ex.controller.reset(sim, ex.action, ex.timing)
    ex.initialized = True
    ex.t = ex.total
    driver.current = ex
    driver._passage_queue = [second]
    driver._active_passage = passage
    driver._active_passage_total_strokes = 2

    driver.step(sim, 1.0 / 120.0)

    assert driver.current is None
    assert driver.phase_label() == "local_passage_hold"
    assert driver.diagnostics()["passageQueueLength"] == 1
    local_target = sim.target_pose

    for _ in range(8):
        driver.step(sim, 1.0 / 120.0)
        sim.step(1.0 / 120.0)

    assert driver.current is not None
    assert driver.current.action == second
    assert sim.target_pose != local_target

    driver.current.timing = StrokeTiming(approach=0.01, press=0.01, paint=0.01, lift=0.01)
    driver.current.controller.reset(sim, driver.current.action, driver.current.timing)
    driver.current.initialized = True
    driver.current.t = driver.current.total
    driver.step(sim, 1.0 / 120.0)

    assert driver.current is None
    assert driver.phase_label() == "return_center"
    assert driver.diagnostics()["activePassage"] is None


def test_active_inference_planning_does_not_block_body_step() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = make_driver()
    started = time.perf_counter()
    driver.step(sim, 1.0 / 240.0)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    assert driver.diagnostics()["planning"]
    wait_for_driver(driver, sim)


def test_active_inference_stroke_realization_targets_canvas_face_not_behind_wall() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    efe = EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ex = StrokeExecution(action=action, efe=efe, posterior=1.0)
    ex.t = ex.approach + ex.press + 0.5 * ex.paint
    pose, brush_down, intended_pressure = pose_for_execution(sim, ex)
    tip = sim.kinematics.tip(pose)
    assert brush_down
    assert intended_pressure > 0.0
    # Painting targets slight bushing deflection, never beyond the hard
    # overtravel limit behind the canvas face.
    assert tip[1] <= sim.canvas.distance + sim.canvas.bushing_travel + 1e-6
    assert tip[1] >= sim.canvas.distance - 0.1


def test_executed_stroke_changes_material_coverage_at_planning_scale() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    before = canvas_summary_state(sim)[0]

    execute_stroke_action(sim, action, dt=1.0 / 90.0)
    after = canvas_summary_state(sim)[0]

    assert after - before > 0.005


def test_active_inference_driver_continues_after_first_material_mark() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(bootstrap_transitions=48, bootstrap_train_steps=16)
    deadline = time.perf_counter() + 30.0

    while time.perf_counter() < deadline and driver.current is None and not driver.stopped:
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)
        time.sleep(0.001)
    assert driver.current is not None

    while time.perf_counter() < deadline and (driver.current is not None or driver.planning):
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)
        time.sleep(0.001)

    while time.perf_counter() < deadline and driver.current is None and not driver.stopped:
        driver.step(sim, 1.0 / 240.0)
        sim.step(1.0 / 240.0)
        time.sleep(0.001)

    assert sim.canvas.material_coverage() > 0.005
    assert not driver.stopped
    assert driver.current is not None


def test_stop_demoted_by_declared_prior_is_reported_and_not_executed() -> None:
    # At low believed coverage the declared stop prior demotes an immediate
    # stop in the policy posterior even when stop has the lowest expected free
    # energy. The driver executes the posterior-ranked policy and reports the
    # demotion as a diagnostic.
    cfg = PainterConfig(minimum_stop_coverage=0.70)
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    stroke = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    stop_policy = Policy((StrokeAction.stop_action(),))
    continue_policy = Policy((stroke, StrokeAction.stop_action()))
    stop_components = EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    continue_components = EFEComponents(10.0, 10.0, 0.0, 0.0, 0.1, 0.0)

    def infer_prior_demoted_stop():
        return continue_policy, continue_components, [
            (continue_policy, continue_components, 0.99),
            (stop_policy, stop_components, 0.01),
        ]

    driver.agent.infer_policy = infer_prior_demoted_stop
    driver._background_plan(canvas_summary_state(sim), None)
    driver._consume_background_plan()

    assert not driver.stopped
    assert driver.current is not None
    assert driver.current.action == stroke
    assert driver.last_components is continue_components
    assert driver.diagnostics()["lastStopBlocked"] is True


def test_stop_selected_by_policy_inference_is_accepted_without_veto() -> None:
    cfg = PainterConfig(minimum_stop_coverage=0.70)
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    driver = ArmActiveInferenceDriver(config=cfg, bootstrap_transitions=0, bootstrap_train_steps=0)
    stop_policy = Policy((StrokeAction.stop_action(),))
    stop_components = EFEComponents(0.0, 0.0, 0.0, 0.0, 0.8, 0.0)

    def infer_stop_first():
        return stop_policy, stop_components, [(stop_policy, stop_components, 1.0)]

    driver.agent.infer_policy = infer_stop_first
    driver._background_plan(canvas_summary_state(sim), None)
    assert driver._consume_background_plan()

    assert driver.stopped
    assert driver.current is None
    assert driver.diagnostics()["lastStopBlocked"] is False
