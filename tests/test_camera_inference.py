from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from active_painter.arm_agent_driver import (
    ACTION_CONDITIONED_CAMERA_UPDATE_VERSION,
    ArmActiveInferenceDriver,
)
from active_painter.camera_geometry import CameraRigSpec, load_camera_rig
from active_painter.camera_inference import (
    CAMERA_SPATIAL_LIKELIHOOD_VERSION,
    CameraSpatialLikelihood,
)
from active_painter.camera_observation import (
    FOVEA_CANVAS_PRODUCT,
    GLOBAL_CANVAS_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
)
from active_painter.config import PainterConfig, SPATIAL_MATERIAL_PLANNER_STATE_KIND
from active_painter.env import StrokeAction
from active_painter.spatial_state import SpatialCanvasState
from active_painter.spatial_inference import SPATIAL_TRANSITION_PRIOR_VERSION


def _prior(
    cfg: PainterConfig,
    *,
    grid_size: int = 8,
    thickness: float = 0.005,
    surface_tone: float = 0.8,
    variance: float = 1e-4,
) -> SpatialCanvasState:
    material = np.zeros((cfg.spatial_material_channels, grid_size, grid_size), dtype=np.float32)
    material[0] = thickness
    material[1] = 0.7
    material[2] = 0.003
    material[3] = surface_tone
    logvar = np.full_like(material, math.log(variance), dtype=np.float32)
    return SpatialCanvasState(material=material, logvar=logvar)


def _frame(
    rig: CameraRigSpec,
    value: float | np.ndarray,
    *,
    camera_name: str = "canvas_right_oblique",
    sequence: int = 0,
    product_kind: str = GLOBAL_CANVAS_PRODUCT,
    product_id: str = "global",
    center: tuple[float, float] | None = None,
    span: tuple[float, float] | None = None,
    capture_time_s: float = 1.0,
    available_time_s: float = 1.04,
) -> CameraFrame:
    pixels = np.asarray(value, dtype=np.float32)
    if pixels.ndim == 0:
        pixels = np.full((32, 32), float(pixels), dtype=np.float32)
    foveal = product_kind == FOVEA_CANVAS_PRODUCT
    return CameraFrame(
        camera_name=camera_name,
        role=rig.camera(camera_name).role,
        sequence=sequence,
        product_kind=product_kind,
        product_id=product_id,
        capture_time_s=capture_time_s,
        available_time_s=available_time_s,
        calibration_revision=rig.version,
        observation_model=rig.observation_model,
        registration="canvas_plane_homography",
        sampling_kind=(
            "native_to_requested_canvas_uv"
            if foveal
            else "native_to_canvas_homography"
        ),
        source_resolution_px=(640, 360),
        declared_acquisition_resolution_px=(3840, 2160),
        grayscale=pixels,
        calibration_validity=np.ones_like(pixels, dtype=np.bool_),
        fovea_request_id="request-0" if foveal else None,
        center_canvas_uv=center if foveal else None,
        span_canvas_uv=span if foveal else None,
        selection_basis="sensor_posterior" if foveal else None,
        selection_revision="test-posterior-v0" if foveal else None,
    )


def _bundle(*frames: CameraFrame) -> CameraObservationBundle:
    monotonic_time_s = max(
        (frame.available_time_s for frame in frames),
        default=1.04,
    )
    return CameraObservationBundle(
        monotonic_time_s=monotonic_time_s,
        frames=frames,
    )


def test_camera_likelihood_updates_visible_factors_and_logs_vfe() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
    )
    rig = load_camera_rig()
    likelihood = CameraSpatialLikelihood(cfg, rig)
    prior = _prior(cfg)
    predicted_before = likelihood._predict_grayscale(prior.material)[0]

    posterior = likelihood.infer(prior, _bundle(_frame(rig, 0.12)))
    predicted_after = likelihood._predict_grayscale(posterior.material)[0]

    assert np.mean(np.abs(predicted_after - 0.12)) < np.mean(
        np.abs(predicted_before - 0.12)
    )
    np.testing.assert_allclose(posterior.material[1], prior.material[1])
    np.testing.assert_allclose(posterior.material[2], prior.material[2])
    assert likelihood.last_vfe is not None
    assert likelihood.last_vfe.version == CAMERA_SPATIAL_LIKELIHOOD_VERSION
    assert likelihood.last_vfe.total == pytest.approx(
        likelihood.last_vfe.complexity
        + likelihood.last_vfe.negative_log_likelihood
    )
    assert likelihood.last_vfe.expected_log_likelihood == pytest.approx(
        -likelihood.last_vfe.negative_log_likelihood
    )
    assert len(likelihood.last_vfe.factors) == 1
    assert likelihood.last_vfe.factors[0].observed_cell_count == 64
    assert likelihood.last_vfe.complexity == pytest.approx(
        sum(factor.complexity for factor in likelihood.last_vfe.factors)
    )
    assert posterior.posterior_revision == prior.posterior_revision + 1
    assert posterior.inference_model_id == (
        f"{CAMERA_SPATIAL_LIKELIHOOD_VERSION}:{rig.likelihood_model}"
    )
    assert posterior.calibration_status == rig.likelihood_status


def test_white_on_white_camera_evidence_does_not_observe_thickness() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
        canvas_ground_tone=0.0,
    )
    rig = load_camera_rig()
    prior = _prior(cfg, thickness=0.004, surface_tone=0.0, variance=1e-5)
    likelihood = CameraSpatialLikelihood(cfg, rig)

    posterior = likelihood.infer(prior, _bundle(_frame(rig, 1.0)))

    np.testing.assert_allclose(posterior.material[0], prior.material[0])
    np.testing.assert_allclose(posterior.logvar[0], prior.logvar[0], atol=1e-6)
    np.testing.assert_allclose(posterior.material[1:3], prior.material[1:3])


def test_correlated_global_and_foveal_products_form_one_exposure_factor() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
    )
    rig = load_camera_rig()
    likelihood = CameraSpatialLikelihood(cfg, rig)
    global_frame = _frame(rig, 0.45)
    fovea = _frame(
        rig,
        0.20,
        product_kind=FOVEA_CANVAS_PRODUCT,
        product_id="fovea:request-0",
        center=(0.5, 0.5),
        span=(0.5, 0.5),
    )

    likelihood.infer(_prior(cfg), _bundle(global_frame, fovea))

    assert likelihood.last_vfe is not None
    assert len(likelihood.last_vfe.factors) == 1
    assert likelihood.last_vfe.factors[0].product_ids == (
        "global",
        "fovea:request-0",
    )


def test_gross_image_mismatch_is_inferred_as_an_outlier_not_masked_by_truth() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
    )
    rig = load_camera_rig()
    prior = _prior(cfg, variance=1e-9)
    matched_likelihood = CameraSpatialLikelihood(cfg, rig)
    predicted = float(matched_likelihood._predict_grayscale(prior.material)[0][0, 0])
    matched_likelihood.infer(prior, _bundle(_frame(rig, predicted)))
    mismatched_likelihood = CameraSpatialLikelihood(cfg, rig)
    mismatched_likelihood.infer(prior, _bundle(_frame(rig, 1.0 - predicted)))

    assert matched_likelihood.last_vfe is not None
    assert mismatched_likelihood.last_vfe is not None
    assert (
        mismatched_likelihood.last_vfe.factors[0].mean_inlier_probability
        < matched_likelihood.last_vfe.factors[0].mean_inlier_probability
    )


def test_declared_camera_precision_controls_posterior_precision() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
    )
    rig = load_camera_rig()
    camera = rig.camera("canvas_right_oblique")
    precise_camera = replace(camera, likelihood_model_error_std=0.01)
    loose_camera = replace(camera, likelihood_model_error_std=0.20)
    precise_rig = replace(
        rig,
        cameras=tuple(
            precise_camera if item.name == camera.name else item
            for item in rig.cameras
        ),
    )
    loose_rig = replace(
        rig,
        cameras=tuple(
            loose_camera if item.name == camera.name else item
            for item in rig.cameras
        ),
    )
    prior = _prior(cfg, variance=1e-5)
    observation = _bundle(_frame(rig, 0.34))

    precise = CameraSpatialLikelihood(cfg, precise_rig).infer(prior, observation)
    loose = CameraSpatialLikelihood(cfg, loose_rig).infer(prior, observation)

    assert np.mean(precise.logvar[3]) < np.mean(loose.logvar[3])


def test_native_and_edge_products_are_not_material_likelihood_factors() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
    )
    rig = load_camera_rig()
    global_frame = _frame(rig, 0.4)
    native = replace(
        global_frame,
        product_kind="native_sensor",
        product_id="native",
        registration="native_sensor",
        sampling_kind="direct_acquisition",
        source_resolution_px=(32, 32),
    )
    likelihood = CameraSpatialLikelihood(cfg, rig)

    posterior = likelihood.infer(_prior(cfg), _bundle(native))

    assert posterior is not None
    assert likelihood.last_vfe is not None
    assert likelihood.last_vfe.factors == ()
    assert likelihood.last_vfe.ignored_products == (
        "canvas_right_oblique:0:native",
    )


def test_directly_constructed_fovea_cannot_claim_oracle_selection() -> None:
    rig = load_camera_rig()
    with pytest.raises(ValueError, match="selection_basis"):
        replace(
            _frame(
                rig,
                0.4,
                product_kind=FOVEA_CANVAS_PRODUCT,
                product_id="fovea:request-0",
                center=(0.5, 0.5),
                span=(0.5, 0.5),
            ),
            selection_basis="exact_simulator_pose",
        )


def test_sensor_driver_accepts_camera_bundle_without_process_access() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
        candidate_policies=2,
        planning_horizon=1,
    )
    driver = ArmActiveInferenceDriver(
        config=cfg,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
    )
    rig = load_camera_rig()

    posterior = driver.ingest_camera_observation(_bundle(_frame(rig, 0.4)))
    diagnostics = driver.diagnostics()

    assert isinstance(posterior, SpatialCanvasState)
    assert driver.observation_boundary_blocked is True
    assert diagnostics["observationBoundary"]["cameraPosteriorConnected"] is True
    assert diagnostics["observationBoundary"]["cameraExposureCount"] == 1
    assert diagnostics["cameraVfe"]["version"] == CAMERA_SPATIAL_LIKELIHOOD_VERSION
    assert "body posterior" in diagnostics["observationBoundary"]["blockedReason"]


def test_executed_action_prior_waits_for_causally_later_camera_exposure() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
        candidate_policies=2,
        planning_horizon=1,
    )
    driver = ArmActiveInferenceDriver(
        config=cfg,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
    )
    rig = load_camera_rig()
    previous = _prior(cfg, variance=1e-4)
    driver.agent.belief = previous
    driver.belief = previous
    action = StrokeAction(0.25, 0.5, 0.75, 0.5, 0.1, 0.6, tone=1.0)
    brush_revision = driver.brush_load_beliefs["black"].revision
    replay_size = len(driver.agent.replay)

    prior = driver.record_executed_action_transition(action)

    assert prior.posterior_revision == previous.posterior_revision + 1
    assert prior.inference_model_id == SPATIAL_TRANSITION_PRIOR_VERSION
    assert driver.action_camera_update_pending is True
    assert driver.action_camera_capture_boundary_required is True
    assert driver.brush_load_beliefs["black"].revision == brush_revision + 1
    assert len(driver.agent.replay) == replay_size

    # No frame is admissible until the runtime records the physical completion
    # boundary. A delivered pre-action exposure also remains inadmissible after
    # the boundary is known.
    before_boundary = _bundle(
        _frame(rig, 0.4, sequence=5, capture_time_s=1.9, available_time_s=2.0)
    )
    assert driver.ingest_camera_observation(before_boundary) is prior
    driver.register_action_camera_capture_boundary(2.0)
    stale = _bundle(
        _frame(rig, 0.4, sequence=6, capture_time_s=1.99, available_time_s=2.01)
    )
    assert driver.ingest_camera_observation(stale) is prior
    assert driver.action_camera_update_pending is True

    eligible = _bundle(
        _frame(rig, 0.2, sequence=7, capture_time_s=2.0, available_time_s=2.04)
    )
    posterior = driver.ingest_camera_observation(eligible)

    assert posterior.posterior_revision == prior.posterior_revision + 1
    assert posterior.inference_model_id.startswith(
        f"{ACTION_CONDITIONED_CAMERA_UPDATE_VERSION}:"
        f"{SPATIAL_TRANSITION_PRIOR_VERSION}:"
    )
    assert driver.action_camera_update_pending is False
    assert driver.action_transition_prior_count == 1
    assert driver.action_camera_update_count == 1
    assert driver.rejected_pre_action_camera_frames == 2
    assert len(driver.agent.replay) == replay_size + 1
    assert driver.trained_transitions == 1
    assert driver.agent.last_camera_vfe is not None
    assert driver.agent.last_camera_vfe.total == pytest.approx(
        driver.agent.last_camera_vfe.complexity
        + driver.agent.last_camera_vfe.negative_log_likelihood
    )
    diagnostics = driver.diagnostics()["observationBoundary"]["actionCameraLoop"]
    assert diagnostics["pending"] is False
    assert diagnostics["completedUpdateCount"] == 1
    assert diagnostics["lastUpdate"]["brushCameraLikelihoodApplied"] is False

    driver.ingest_camera_observation(
        _bundle(
            _frame(
                rig,
                0.2,
                sequence=8,
                capture_time_s=2.1,
                available_time_s=2.14,
            )
        )
    )
    assert driver.action_camera_update_count == 1
    assert len(driver.agent.replay) == replay_size + 1


def test_sensor_driver_emits_causally_paired_visual_transition_hook() -> None:
    cfg = PainterConfig(
        planner_state_kind=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        spatial_grid_size=8,
        candidate_policies=2,
        planning_horizon=1,
    )
    driver = ArmActiveInferenceDriver(
        config=cfg,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
    )
    rig = load_camera_rig()
    before = _bundle(
        _frame(rig, 0.8, sequence=1, capture_time_s=1.0, available_time_s=1.04)
    )
    driver.ingest_camera_observation(before)
    observed = []
    driver.on_visual_observed_transition = lambda *items: observed.append(items)
    action = StrokeAction(0.25, 0.4, 0.75, 0.6, 0.08, 0.5, tone=1.0)

    driver.record_executed_action_transition(action)
    driver.register_action_camera_capture_boundary(2.0)
    after = _bundle(
        _frame(rig, 0.2, sequence=2, capture_time_s=2.0, available_time_s=2.04)
    )
    driver.ingest_camera_observation(after)

    assert len(observed) == 1
    assert observed[0][0] is before
    assert observed[0][1] is action
    assert observed[0][3].frames[0].sequence == after.frames[0].sequence
    assert observed[0][3].frames[0].capture_time_s == after.frames[0].capture_time_s
