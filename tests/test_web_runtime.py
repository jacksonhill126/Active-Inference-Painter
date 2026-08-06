import errno
import io
import json
import numpy as np
from pathlib import Path
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from xml.etree import ElementTree

from PIL import Image
import pytest

from active_painter.camera_observation import (
    FOVEA_CANVAS_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
)

from active_painter.arm_agent_driver import (
    ORACLE_OBSERVATION_ACCESS_MODE,
    SENSOR_OBSERVATION_ACCESS_MODE,
)
from active_painter.version import code_build_info, code_version, package_version
from active_painter.web_server import (
    PainterRequestHandler,
    bind_server,
    build_parser,
    is_client_disconnect,
    render_index_html,
    resolved_bootstrap,
)
from active_painter.web_runtime import WebSimRuntime
from active_painter.web_robot_model import (
    legacy_tip_to_physical_target,
    load_robot_visual_model,
    physical_tip,
    retarget_legacy_robot_state,
)


def oracle_runtime(*args, **kwargs) -> WebSimRuntime:
    kwargs.setdefault("observation_access_mode", ORACLE_OBSERVATION_ACCESS_MODE)
    # Most runtime contract tests intentionally retain the obsolete fast
    # summary fixture. Tests of the research path opt into spatial_material
    # explicitly.
    kwargs.setdefault("planner_state_kind", "summary")
    if kwargs["planner_state_kind"] == "summary":
        kwargs.setdefault("driver_bootstrap_train_steps", 180)
    return WebSimRuntime(*args, **kwargs)


def test_web_runtime_defaults_to_fail_closed_sensor_boundary() -> None:
    runtime = WebSimRuntime(
        canvas_size=32,
        driver_bootstrap_transitions=8,
        driver_bootstrap_train_steps=1,
    )

    assert runtime.agent_enabled is False
    assert runtime.agent_driver.trained_transitions == 0
    assert runtime.planner_state_kind == "spatial_material"
    assert runtime.agent_driver.diagnostics()["stateRepresentationLifecycle"][
        "status"
    ] == "provisional_low_level_material_baseline"
    assert runtime.agent_driver.diagnostics()["observationBoundary"][
        "modelAccessBlocked"
    ] is True
    result = runtime.command({"type": "toggle_agent"})
    assert result["ok"] is False
    assert "fail-closed" in result["error"]


def test_web_runtime_state_contains_arm_canvas_and_contact() -> None:
    runtime = oracle_runtime(canvas_size=32)
    state = runtime.state()
    assert state["canvas"]["distance"] == 17.0
    assert len(state["points"]) == 3
    assert "pressure" in state["contact"]
    assert state["contact"]["touching"] is False
    assert state["contact"]["projectedOnCanvas"] == state["contact"]["onCanvas"]
    assert "yaw" in state["pose"]
    assert state["robot"]["mode"] == "legacy_canvas_cartesian_retarget"
    assert list(state["robot"]["jointPositionDeg"]) == ["yaw", "pitch", "roll", "elbow"]
    assert state["robot"]["alignmentErrorM"] < 1e-5
    assert len(state["robot"]["tipM"]) == 3
    assert state["telemetryLog"]["sampleCount"] >= 1
    assert state["telemetryLog"]["csvEndpoint"] == "/api/telemetry.csv"
    assert state["brushLoaded"] is False
    assert state["depositingPaint"] is False
    assert "velocityRadS" in state["motor"]["yaw"]
    assert "encoderAngleDeg" in state["motor"]["yaw"]
    assert "positionErrorDeg" in state["motor"]["yaw"]
    assert "elasticDeflectionDeg" in state["motor"]["yaw"]
    assert "encoderStdDeg" in state["motor"]["yaw"]
    assert state["cameraObservation"]["available"] is False
    assert state["cameraObservation"]["interfaceVersion"] == (
        "camera-observation-interface-v1"
    )
    assert state["cameraObservation"]["productContract"] == (
        "native_global_requested_fovea_v1"
    )
    assert state["cameraObservation"]["foveaDefault"] is None
    assert state["cameraObservation"]["consumedByInference"] is True
    assert state["cameraObservation"]["likelihoodModel"] == (
        "registered_grayscale_occlusion_mixture_linearized_v0"
    )
    foveation = state["cameraObservation"]["foveation"]
    assert foveation["interfaceVersion"] == "fovea-trace-v0"
    assert foveation["traceRetentionSeconds"] == pytest.approx(10.0)
    assert foveation["memoryHorizonSeconds"] is None
    assert foveation["trace"] == []


def test_fovea_trace_expires_and_can_follow_declared_agent_memory() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
        fovea_trace_retention_s=10.0,
    )
    rig = runtime.robot_model["cameraRig"]
    pixels = np.full((16, 16), 0.5, dtype=np.float32)
    frame = CameraFrame(
        camera_name="canvas_right_oblique",
        role="contact_tracking",
        sequence=4,
        product_kind=FOVEA_CANVAS_PRODUCT,
        product_id="fovea:operator-4",
        capture_time_s=0.96,
        available_time_s=1.0,
        calibration_revision=rig["version"],
        observation_model=rig["observationModel"],
        registration="canvas_plane_homography",
        sampling_kind="native_to_requested_canvas_uv",
        source_resolution_px=(640, 360),
        declared_acquisition_resolution_px=(3840, 2160),
        grayscale=pixels,
        calibration_validity=np.ones_like(pixels, dtype=np.bool_),
        fovea_request_id="operator-4",
        center_canvas_uv=(0.25, 0.75),
        span_canvas_uv=(0.2, 0.3),
        selection_basis="operator_diagnostic",
        selection_revision="test-pointer-v0",
    )
    runtime.sim_time = 1.1
    runtime._record_fovea_observations(
        CameraObservationBundle(monotonic_time_s=1.1, frames=(frame,))
    )

    foveation = runtime.state()["cameraObservation"]["foveation"]

    assert foveation["active"]["requestId"] == "operator-4"
    assert foveation["active"]["centerCanvasUv"] == [0.25, 0.75]
    assert foveation["active"]["ageSeconds"] == pytest.approx(0.1)
    assert foveation["retentionSource"] == (
        "visualization_default_no_foveation_memory_model_declared"
    )

    runtime.agent_driver.agent.foveation_memory_horizon_s = 2.0
    foveation = runtime.state()["cameraObservation"]["foveation"]
    assert foveation["traceRetentionSeconds"] == pytest.approx(2.0)
    assert foveation["memoryHorizonSeconds"] == pytest.approx(2.0)
    assert foveation["retentionSource"] == "agent_foveation_memory_horizon"

    runtime.sim_time = 3.1
    assert runtime.state()["cameraObservation"]["foveation"]["trace"] == []


def test_mujoco_runtime_exposes_lazy_model_facing_camera_png() -> None:
    pytest.importorskip("mujoco")
    from active_painter.camera_observation import CameraObservationProcess

    runtime = WebSimRuntime(
        canvas_size=32,
        plant_backend="mujoco",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    runtime._camera_process = CameraObservationProcess(
        native_resolution_overrides={
            "canvas_right_oblique": (640, 360),
            "canvas_left_oblique": (640, 360),
            "canvas_inspection_deployed": (640, 360),
            "brush_standoff_overhead": (320, 240),
        }
    )
    try:
        camera_state = runtime.state()["cameraObservation"]
        assert camera_state["available"] is True
        assert camera_state["nativeFrameRetained"] is True
        assert camera_state["foveaAddressing"] == "canvas_uv_center_and_span"
        image = Image.open(io.BytesIO(runtime.camera_png("canvas_right_oblique")))
        assert image.mode == "L"
        assert image.size == (512, 512)
        first_bundle = runtime.camera_observations()
        assert first_bundle.frames == ()
        backend = runtime.sim.plant.backend
        backend.data.time = 0.04
        delivered = runtime.camera_observations()
        assert any(
            frame.product_kind == "global_canvas"
            for frame in delivered.frames
        )
        camera_diagnostics = runtime.agent_driver.diagnostics()
        assert camera_diagnostics["observationBoundary"][
            "cameraExposureCount"
        ] == 2
        assert camera_diagnostics["cameraVfe"]["factors"]
        request_result = runtime.command(
            {
                "type": "request_fovea",
                "cameraName": "canvas_right_oblique",
                "centerCanvasUv": [0.3, 0.7],
                "spanCanvasUv": [0.2, 0.2],
            }
        )
        assert request_result["ok"] is True
        requested_foveation = runtime.state()["cameraObservation"]["foveation"]
        assert requested_foveation["trace"] == []
        assert requested_foveation["pendingRequestCount"] == 1
        backend.data.time = 0.08
        runtime.camera_observations()
        backend.data.time = 0.12
        foveal_delivery = runtime.camera_observations()
        assert any(
            frame.product_kind == FOVEA_CANVAS_PRODUCT
            for frame in foveal_delivery.frames
        )
        foveation = runtime.state()["cameraObservation"]["foveation"]
        assert foveation["active"]["centerCanvasUv"] == pytest.approx([0.3, 0.7])
        assert foveation["active"]["selectionBasis"] == "operator_diagnostic"
        with pytest.raises(RuntimeError, match="camera_clear_park"):
            runtime.camera_png("canvas_inspection_deployed")
        park_qpos = backend.keyframe_qpos("camera_clear_park")
        backend.set_state(park_qpos[:4], control_rad=park_qpos[:4])
        inspection = Image.open(
            io.BytesIO(runtime.camera_png("canvas_inspection_deployed"))
        )
        assert inspection.mode == "L"
        assert inspection.size == (512, 512)
    finally:
        runtime.stop()


def test_web_runtime_can_enable_spatial_material_planner() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    state = runtime.state()

    assert state["agent"]["stateRepresentation"].startswith("Spatial Gaussian q(s) with pixel-local rollouts")
    assert state["agent"]["transitionModel"].startswith("learned LocalSpatialDynamicsEnsemble")
    assert state["agent"]["spatialTransitionMode"] == "local_patch"
    assert state["agent"]["belief"]["names"] == [
        "thickness",
        "wetness",
        "black_mass",
        "surface_tone",
        "ground_contrast",
        "material_coverage",
    ]
    assert state["agent"]["spatialBelief"]["gridSize"] == 8
    assert [level["gridSize"] for level in state["agent"]["spatialBelief"]["materialPyramid"]] == [32, 16, 8]
    assert state["agent"]["spatialBelief"]["materialPyramid"][0]["name"] == "pixel"
    assert state["agent"]["spatialBelief"]["materialPyramid"][-1]["name"] == "planner"
    assert state["agent"]["markEvents"]["activeCount"] >= 0
    hierarchy = state["agent"]["composition"]["hierarchy"]
    assert hierarchy["canvas"]["dimensions"] == 32
    assert hierarchy["relational"]["dimensions"] == 24
    assert hierarchy["markSlots"] == 8
    assert hierarchy["passageTrajectory"]["enabled"] is True
    assert hierarchy["passageTrajectory"]["descriptorDimensions"] == 14
    assert state["agent"]["composition"]["passageReplaySize"] == 0
    assert state["agent"]["composition"]["passageStepReplaySize"] == 0
    # Precision beliefs must survive json.dumps even with a clamped belief:
    # unbounded, a disagreeing F drives gamma to ~1e-3 and 1/eps to ~1e300,
    # both of which break the JSON contract the viewer depends on.
    ledger = runtime.agent_driver.precision_ledger
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, size=24)
    G = (base - base.mean()) * 8.0
    for _ in range(6):
        ledger.observe_policy(G, -0.5 * G)
        ledger.observe("terminal_coverage", G, -0.5 * G)
    assert ledger.last_updates["terminal_coverage"].status == "clamped"
    runtime.agent_driver.gap_increment.observe(0.0, 0)
    runtime.agent_driver.gap_increment.observe(0.4, 2)
    refreshed = runtime.state()
    json.dumps(refreshed)
    beliefs = refreshed["agent"]["precisionBeliefs"]
    assert beliefs["policy"]["priorGamma"] == 0.35
    for entry in beliefs.values():
        assert np.isfinite(float(entry["gamma"]))
    assert refreshed["agent"]["gapProgress"]["hasObservations"] == 1.0


def test_web_runtime_uses_bounded_passage_planning_budget() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        planner_state_kind="spatial_material",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )

    assert runtime.agent_driver.config.planning_horizon == 4
    assert runtime.agent_driver.config.candidate_policies == 32
    assert runtime.agent_driver.config.motor_forecast_candidates == 2
    assert runtime.agent_driver.config.passage_proposal_mix == 0.45
    assert runtime.agent_driver.config.passage_plan_proposal_mix == 0.15


def test_web_server_uses_fast_spatial_bootstrap_defaults() -> None:
    parser = build_parser()

    spatial = parser.parse_args([])
    summary = parser.parse_args(["--planner-state-kind", "summary"])
    overridden = parser.parse_args(
        [
            "--planner-state-kind",
            "spatial_material",
            "--driver-bootstrap-transitions",
            "7",
            "--driver-bootstrap-train-steps",
            "3",
        ]
    )

    assert resolved_bootstrap(spatial) == (96, 24)
    assert resolved_bootstrap(summary) == (96, 180)
    assert resolved_bootstrap(overridden) == (7, 3)
    assert summary.port == 8017
    assert summary.telemetry_max_samples == 54_000
    assert summary.telemetry_sample_hz == 15.0
    assert summary.checkpoint_path is None
    assert summary.checkpoint_save_every_transitions == 10
    assert summary.plant_backend == "native"
    assert summary.observation_mode == SENSOR_OBSERVATION_ACCESS_MODE
    assert spatial.planner_state_kind == "spatial_material"
    assert parser.parse_args(["--plant-backend", "mujoco"]).plant_backend == "mujoco"


def test_web_server_exposes_checkpoint_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint-path",
            "runs/web/checkpoints/latest.pt",
            "--checkpoint-save-every-transitions",
            "5",
        ]
    )

    assert args.checkpoint_path == "runs/web/checkpoints/latest.pt"
    assert args.checkpoint_save_every_transitions == 5


def test_web_runtime_wires_checkpoint_path() -> None:
    root = Path("runs/test_web_runtime_checkpoint")
    shutil.rmtree(root, ignore_errors=True)
    path = root / "viewer_weights.pt"
    runtime = oracle_runtime(
        canvas_size=32,
        checkpoint_path=path,
        checkpoint_save_every_transitions=3,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    checkpoint = runtime.state()["agent"]["checkpoint"]

    assert checkpoint["path"] == str(path)
    assert checkpoint["status"] == "not_found"
    assert checkpoint["saveEveryTransitions"] == 3
    shutil.rmtree(root, ignore_errors=True)


def test_web_server_falls_back_when_requested_port_is_busy() -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy_port = blocker.getsockname()[1]
    try:
        server = bind_server("127.0.0.1", busy_port, PainterRequestHandler)
        try:
            assert server.server_address[1] != busy_port
        finally:
            server.server_close()
    finally:
        blocker.close()


def test_web_server_suppresses_expected_client_disconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    def disconnected(_handler: PainterRequestHandler) -> None:
        raise ConnectionResetError(errno.ECONNRESET, "client went away")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", disconnected)

    handler = object.__new__(PainterRequestHandler)
    handler.handle()


def test_web_server_reraises_unexpected_socket_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(_handler: PainterRequestHandler) -> None:
        raise OSError(errno.EINVAL, "unexpected socket failure")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", failed)

    handler = object.__new__(PainterRequestHandler)
    with pytest.raises(OSError, match="unexpected socket failure"):
        handler.handle()
    assert is_client_disconnect(ConnectionAbortedError(errno.ECONNABORTED, "aborted"))
    assert not is_client_disconnect(OSError(errno.EINVAL, "bad argument"))


def test_web_runtime_commands_update_modes_and_canvas_png() -> None:
    runtime = oracle_runtime(canvas_size=32)
    assert not runtime.state()["maxSpeed"]
    response = runtime.command({"type": "toggle_max_speed"})
    assert response["ok"]
    assert runtime.state()["maxSpeed"]
    png = runtime.canvas_png()
    assert png.startswith(b"\x89PNG")


def test_web_runtime_exposes_upper_arm_and_rolled_elbow_axes() -> None:
    runtime = oracle_runtime(canvas_size=32)
    runtime.sim.actual_pose.roll = 31.0

    state = runtime.state()

    assert np.linalg.norm(state["upperArmAxis"]) == pytest.approx(1.0)
    assert np.linalg.norm(state["elbowHingeAxis"]) == pytest.approx(1.0)
    assert np.dot(state["upperArmAxis"], state["elbowHingeAxis"]) == pytest.approx(0.0, abs=1e-12)


def test_web_canvas_png_renders_gray_ground_with_visible_white_and_black_paint() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    blank = np.asarray(Image.open(io.BytesIO(runtime.canvas_png())))
    expected_ground = (1.0 - runtime.sim.config.canvas_ground_tone) * 255.0
    assert abs(float(blank.mean()) - expected_ground) < 2.0

    runtime.sim.canvas.paint_at(
        np.asarray([-3.0, runtime.sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=0.0,
        dt=0.2,
    )
    white = np.asarray(Image.open(io.BytesIO(runtime.canvas_png())))
    assert float(white.max()) > float(blank.mean())

    runtime.sim.canvas.paint_at(
        np.asarray([3.0, runtime.sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=1.0,
        dt=0.2,
    )
    black = np.asarray(Image.open(io.BytesIO(runtime.canvas_png())))
    assert float(black.min()) < float(blank.mean())


def test_web_runtime_exports_arm_telemetry_csv() -> None:
    runtime = oracle_runtime(canvas_size=32, telemetry_sample_period=1.0 / 240.0)
    runtime.agent_enabled = False
    runtime.agent_driver.enabled = False
    for _ in range(4):
        runtime._advance_one_step(1.0 / 240.0)

    state = runtime.state()
    csv_text = runtime.telemetry_csv().decode("utf-8")

    assert state["telemetryLog"]["sampleCount"] >= 4
    assert "position_yaw_deg" in csv_text
    assert "velocity_yaw_rad_s" in csv_text
    assert "current_pitch_a" in csv_text
    assert "torque_elbow_nm" in csv_text
    assert "tip_x" in csv_text
    assert "target_tip_y" in csv_text
    assert "brush_loaded" in csv_text
    assert "depositing_paint" in csv_text


def test_web_runtime_manual_control_toggles_material_load_not_deposition_permission() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )

    loaded = runtime.command({"type": "toggle_brush_load"})["state"]
    unloaded = runtime.command({"type": "toggle_brush_load"})["state"]

    assert loaded["brushLoaded"] is True
    assert loaded["depositingPaint"] is False
    assert unloaded["brushLoaded"] is False


def test_web_runtime_default_telemetry_is_sparse_rolling_window() -> None:
    runtime = oracle_runtime(
        canvas_size=32,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    state = runtime.state()

    assert runtime.telemetry_sample_period == pytest.approx(1.0 / 15.0)
    assert state["telemetryLog"]["maxSamples"] == 54_000
    assert "rolling overwrite" in state["telemetryLog"]["retentionPolicy"]
    assert "estimatedSampleHz" in state["telemetryLog"]


def test_web_runtime_state_exposes_python_code_version() -> None:
    state = oracle_runtime(canvas_size=32).state()
    assert state["codeVersion"] == code_version()
    assert state["codeVersion"] != "unknown"
    assert "+code." in state["codeVersion"]


def test_web_runtime_state_consumes_stopped_episode_before_reporting() -> None:
    runtime = oracle_runtime(canvas_size=32)
    runtime.sim.canvas.paint_at(
        np.asarray([0.0, runtime.sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=1.0,
        dt=0.2,
    )
    runtime.agent_driver.stopped = True

    state = runtime.state()

    assert state["paintingCount"] == 1
    assert state["agent"]["stopped"] is False
    assert state["agent"]["phase"] == "global_planning"
    assert state["canvas"]["coverage"] == 0.0


def test_web_visualizer_has_no_scene_grid_and_uses_runtime_version_slot() -> None:
    main_js = Path("web/main.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    assert "GridHelper" not in main_js
    assert 'id="codeVersion"' in index_html
    assert "v..." not in index_html
    assert "__ACTIVE_PAINTER_VERSION__" not in index_html
    assert f"v{package_version()}" in index_html
    assert "/api/version" in index_html
    assert "state.codeVersion" in main_js
    assert "currentPlanningSeconds" in main_js
    assert "planningProfile" in main_js
    assert "Plan base EFE" in main_js
    assert "VFE F" in main_js
    assert "Checkpoint" in main_js
    assert "retentionPolicy" in main_js
    assert "Canvas transition risk" in main_js
    assert "Relational transition risk" in main_js
    assert "Relational observation" in main_js
    assert "Passage kind support" in main_js
    assert "/api/robot-model" in main_js
    assert "buildBody" in main_js
    assert "updateRobot(state.robot" in main_js
    assert 'applyJoint("brush_bend_x"' in main_js
    assert 'applyJoint("brush_bend_z"' in main_js
    assert "canvasTexture" in main_js
    assert "updateFoveaVisualization" in main_js
    assert "traceRetentionSeconds" in main_js
    assert "request_fovea" in main_js
    assert 'id="foveaCamera"' in index_html
    assert "mujoco-robstride-electromechanical-v4" not in main_js


def test_web_robot_model_is_derived_from_the_mjcf_geometry() -> None:
    model = load_robot_visual_model()

    assert model["version"] == "mujoco-robstride-electromechanical-v4"
    assert model["jointOrder"] == ["yaw", "pitch", "roll", "elbow"]
    assert model["kinematicConvention"] == "Rz_yaw_Rx_pitch_Ry_roll_Rx_elbow"
    assert (
        model["fidelity"]["powerElectronics"]
        == "voltage_limited_equivalent_not_phase_resolved"
    )
    assert model["fidelity"]["encoder"] == "ideal_joint_state_without_noise_or_delay"
    assert model["fidelity"]["thermal"] == "disabled_missing_vendor_constants"
    assert model["kinematics"]["yawOrigin"] == pytest.approx([0.0, 0.0, 0.285])
    assert model["kinematics"]["pitchAnchorAtZero"] == pytest.approx([0.075, 0.0, 0.391])
    assert model["kinematics"]["upperArmLength"] == pytest.approx(0.3302)
    assert model["kinematics"]["lowerArmLength"] == pytest.approx(0.3302)
    assert model["canvas"]["center"] == pytest.approx([0.075, 0.4826, 0.350])
    assert model["cameraRig"]["version"] == "provisional-multiview-v4"
    assert model["cameraRig"]["normalization"] == "role_dependent_v1"
    assert model["cameraRig"]["calibrationStatus"] == (
        "nominal_lens_geometry_pending_physical_calibration"
    )
    assert model["cameraRig"]["observationEncoding"] == (
        "linear_grayscale_float32_normalized_0_1"
    )
    assert model["cameraRig"]["shutterModel"] == (
        "heterogeneous_per_camera_v1"
    )
    assert model["cameraRig"]["observationModel"] == (
        "mujoco_native_global_foveal_composite_v1"
    )
    assert model["cameraRig"]["productContract"] == (
        "native_global_requested_fovea_v1"
    )
    assert model["cameraRig"]["foveaAddressing"] == (
        "canvas_uv_center_and_span"
    )
    assert model["cameraRig"]["foveaSelectionBoundary"] == (
        "external_observation_space_request_without_oracle_default"
    )
    assert model["cameraRig"]["noiseStatus"] == "provisional_not_calibrated"
    assert model["cameraRig"]["likelihoodModel"] == (
        "registered_grayscale_occlusion_mixture_linearized_v0"
    )
    assert model["cameraRig"]["likelihoodStatus"] == (
        "provisional_simulation_prior_pending_physical_calibration"
    )
    assert model["cameraRig"]["provisionalSpecularStrength"] == pytest.approx(
        0.08
    )
    cameras = {
        camera["name"]: camera for camera in model["cameraRig"]["cameras"]
    }
    assert cameras["canvas_right_oblique"]["likelihoodModelErrorStd"] == pytest.approx(
        0.035
    )
    assert cameras["canvas_right_oblique"]["likelihoodInlierProbability"] == pytest.approx(
        0.96
    )
    assert set(cameras) == {
        "canvas_right_oblique",
        "canvas_left_oblique",
        "canvas_inspection_deployed",
        "brush_standoff_overhead",
    }
    assert cameras["canvas_right_oblique"]["incidenceDeg"] == pytest.approx(
        31.7548481366
    )
    assert cameras["canvas_left_oblique"]["incidenceDeg"] == pytest.approx(
        32.5724697676
    )
    assert cameras["canvas_inspection_deployed"]["availability"] == "park_only"
    assert cameras["canvas_inspection_deployed"]["incidenceDeg"] == pytest.approx(
        0.0
    )
    assert cameras["canvas_right_oblique"]["channels"] == "grayscale"
    assert cameras["canvas_right_oblique"]["registration"] == (
        "canvas_plane_homography"
    )
    assert cameras["canvas_right_oblique"]["modelInputResolutionPx"] == [
        512,
        512,
    ]
    assert cameras["canvas_right_oblique"]["hardwareBaseline"] == (
        "OM_SYSTEM_OM-1"
    )
    assert cameras["canvas_right_oblique"]["hardwareStatus"] == "owned"
    assert cameras["canvas_right_oblique"]["lensStatus"] == (
        "confirmed_focal_length"
    )
    assert cameras["canvas_right_oblique"]["captureMode"] == (
        "MFT_full_width_16x9"
    )
    assert cameras["canvas_right_oblique"]["focalLengthMm"] == pytest.approx(
        25.0
    )
    assert cameras["canvas_right_oblique"]["activeSensorWidthMm"] == (
        pytest.approx(17.3)
    )
    assert cameras["canvas_right_oblique"][
        "fullFrameEquivalentFocalLengthMm"
    ] == pytest.approx(50.0)
    assert cameras["canvas_right_oblique"]["shutterModel"] == "rolling"
    assert cameras["canvas_right_oblique"]["acquisitionResolutionPx"] == [
        3840,
        2160,
    ]
    assert cameras["canvas_right_oblique"]["fovealResolutionPx"] == [256, 256]
    assert cameras["canvas_left_oblique"]["hardwareBaseline"] == (
        "Sony_ILCE-7RM2"
    )
    assert cameras["canvas_left_oblique"]["captureMode"] == (
        "Super35_full_width_16x9"
    )
    assert cameras["canvas_left_oblique"]["focalLengthMm"] == pytest.approx(
        35.0
    )
    assert cameras["canvas_left_oblique"][
        "fullFrameEquivalentFocalLengthMm"
    ] == pytest.approx(52.5)
    assert cameras["canvas_left_oblique"]["transport"] == (
        "clean_HDMI_capture"
    )
    assert cameras["canvas_right_oblique"]["sampleRateHz"] == pytest.approx(
        30.0
    )
    assert cameras["canvas_right_oblique"]["latencyS"] == pytest.approx(
        1.0 / 30.0
    )
    assert cameras["canvas_right_oblique"]["quantizationBits"] == 8
    assert cameras["brush_standoff_overhead"]["role"] == "brush_standoff"
    assert cameras["brush_standoff_overhead"]["registration"] == (
        "canvas_edge_profile"
    )
    assert cameras["brush_standoff_overhead"]["modelInputResolutionPx"] == [
        640,
        480,
    ]
    assert cameras["brush_standoff_overhead"]["fovealResolutionPx"] is None
    assert cameras["brush_standoff_overhead"]["sampleRateHz"] == pytest.approx(
        60.0
    )
    assert model["brush"]["diameter"] == pytest.approx(0.0127)
    assert model["brush"]["bendRangeRad"] == pytest.approx(
        [-0.349065850399, 0.349065850399]
    )
    assert model["brush"]["tangentialStiffnessNmPerRad"] == pytest.approx(
        [1.2, 1.2]
    )
    assert model["motors"]["yaw"]["model"] == "RobStride 03"
    assert model["motors"]["elbow"]["model"] == "RobStride 02"
    assert (
        model["motors"]["yaw"]["actuatorModel"]
        == "output_equivalent_dcmotor_position_v1"
    )
    assert model["motors"]["yaw"]["ratedVoltageV"] == pytest.approx(48.0)
    assert model["motors"]["yaw"]["voltageRangeV"] == pytest.approx([15.0, 60.0])
    assert model["motors"]["yaw"]["poleCount"] == 42
    assert model["motors"]["elbow"]["poleCount"] == 28
    assert model["motors"]["yaw"]["effectiveMotorConstantNmPerA"] == pytest.approx(
        2.327719630643
    )
    assert model["motors"]["yaw"]["equivalentViscousLossNmsPerRad"] == pytest.approx(
        0.068394108064
    )
    assert model["motors"]["elbow"]["electricalTimeConstantS"] == pytest.approx(
        0.000453448276
    )

    world_bodies = {body["name"]: body for body in model["world"]["bodies"]}
    assert set(world_bodies) == {
        "canvas",
        "canvas_right_camera_housing",
        "canvas_left_camera_housing",
        "canvas_inspection_camera_housing",
        "brush_standoff_camera_housing",
        "base",
    }
    assert world_bodies["canvas_right_camera_housing"]["xyAxes"] == pytest.approx(
        [
            0.880865989267,
            0.473365724311,
            0.0,
            -0.123600957574,
            0.230003724765,
            0.965308805451,
        ]
    )
    assert {
        geom["name"]
        for geom in world_bodies["canvas_inspection_camera_housing"]["geoms"]
    } == {
        "canvas_inspection_camera_body",
        "canvas_inspection_camera_barrel",
        "canvas_inspection_camera_glass",
    }


def test_web_robot_payload_tracks_physical_xml_edits() -> None:
    source = Path("models/active_inference_painter.xml")
    root_dir = Path("runs/test_web_robot_payload")
    edited = root_dir / "edited_robot.xml"
    shutil.rmtree(root_dir, ignore_errors=True)
    root_dir.mkdir(parents=True)
    try:
        tree = ElementTree.parse(source)
        root = tree.getroot()

        root.find(".//body[@name='pitch_output']").set("pos", "0.080 0 0.120")
        root.find(".//joint[@name='elbow']").set("range", "-1.0 2.0")
        root.find(".//body[@name='canvas']").set("pos", "0.090 0.50635 0.400")
        root.find(".//geom[@name='canvas_surface']").set(
            "size",
            "0.300 0.00635 0.200",
        )
        root.find(".//geom[@name='bristle_contact']").set(
            "fromto",
            "0 0 0 0 0.040 0",
        )
        root.find(".//geom[@name='bristle_contact']").set("size", "0.007")
        root.find(".//joint[@name='brush_compression']").set("range", "-0.020 0")
        tree.write(edited, encoding="utf-8", xml_declaration=True)

        model = load_robot_visual_model(edited)

        assert model["kinematics"]["yawToPitch"] == pytest.approx([0.080, 0.0, 0.120])
        assert model["kinematics"]["pitchAnchorAtZero"] == pytest.approx(
            [0.080, 0.0, 0.405]
        )
        assert model["jointRangeDeg"]["elbow"] == pytest.approx(
            np.rad2deg([-1.0, 2.0])
        )
        assert model["canvas"]["center"] == pytest.approx([0.090, 0.500, 0.400])
        assert model["canvas"]["width"] == pytest.approx(0.600)
        assert model["canvas"]["height"] == pytest.approx(0.400)
        assert model["brush"]["diameter"] == pytest.approx(0.014)
        assert model["brush"]["bristleLength"] == pytest.approx(0.040)
        assert model["brush"]["compressionTravel"] == pytest.approx(0.020)
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("legacy_tip", "expected_target"),
    [
        ([0.0, 17.0, 0.0], [0.075, 0.4826, 0.350]),
        ([-10.0, 17.0, -10.0], [-0.179, 0.4826, 0.096]),
        ([10.0, 17.0, 10.0], [0.329, 0.4826, 0.604]),
        ([0.0, 16.5, 0.0], [0.075, 0.4699, 0.350]),
    ],
)
def test_legacy_canvas_points_retarget_to_the_physical_robot(
    legacy_tip: list[float],
    expected_target: list[float],
) -> None:
    model = load_robot_visual_model()
    source_pose = {"yaw": 0.0, "pitch": -50.0, "roll": 0.0, "elbow": 100.0}

    target = legacy_tip_to_physical_target(model, np.asarray(legacy_tip))
    state = retarget_legacy_robot_state(model, source_pose, np.asarray(legacy_tip))

    assert target == pytest.approx(expected_target)
    assert state["mappedCartesianTargetM"] == pytest.approx(expected_target)
    assert state["alignmentErrorM"] < 1e-5
    assert physical_tip(model, state["jointPositionDeg"]) == pytest.approx(
        expected_target,
        abs=1e-5,
    )


def test_web_server_renders_literal_fallback_version_before_javascript_runs() -> None:
    html = render_index_html(Path("web/index.html")).decode("utf-8")
    assert f"v{code_version()}" in html
    assert "__ACTIVE_PAINTER_VERSION__" not in html


def test_code_build_info_increments_when_source_fingerprint_changes() -> None:
    root = Path("runs") / "test_code_build_info"
    shutil.rmtree(root, ignore_errors=True)
    try:
        package = root / "pyproject.toml"
        source = root / "src" / "active_painter"
        web = root / "web"
        models = root / "models"
        source.mkdir(parents=True)
        web.mkdir()
        models.mkdir()
        package.write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
        (source / "versioned.py").write_text("VALUE = 1\n", encoding="utf-8")
        (web / "main.js").write_text("console.log('one');\n", encoding="utf-8")
        (models / "active_inference_painter.xml").write_text(
            '<mujoco model="one"/>\n',
            encoding="utf-8",
        )
        stamp = root / ".stamp.json"

        first = code_build_info(root=root, metadata_path=stamp)
        second = code_build_info(root=root, metadata_path=stamp)
        (web / "main.js").write_text("console.log('two');\n", encoding="utf-8")
        third = code_build_info(root=root, metadata_path=stamp)
        (models / "active_inference_painter.xml").write_text(
            '<mujoco model="two"/>\n',
            encoding="utf-8",
        )
        fourth = code_build_info(root=root, metadata_path=stamp)

        assert first.build == second.build
        assert third.build == second.build + 1
        assert fourth.build == third.build + 1
        assert first.fingerprint != third.fingerprint
        assert third.fingerprint != fourth.fingerprint
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_web_runtime_driver_stop_callback_immediately_restarts_episode() -> None:
    runtime = oracle_runtime(canvas_size=32)
    runtime.sim.canvas.paint_at(
        np.asarray([0.0, runtime.sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=1.0,
        dt=0.2,
    )
    runtime.agent_driver._pending_stopped = True
    runtime.agent_driver._pending_ranked = []

    runtime.agent_driver.step(runtime.sim, 1.0 / 240.0)

    assert runtime.painting_count == 1
    assert runtime.agent_driver.stopped is False
    assert runtime.sim.canvas.material_coverage() == 0.0
    assert runtime.agent_driver.diagnostics()["phase"] == "global_planning"


def test_web_runtime_max_speed_releases_state_lock_between_physics_steps() -> None:
    original_advance = WebSimRuntime._advance_one_step
    entered = threading.Event()

    def slow_advance(self: WebSimRuntime, fixed_dt: float) -> None:
        _ = fixed_dt
        entered.set()
        time.sleep(0.02)

    WebSimRuntime._advance_one_step = slow_advance
    runtime = oracle_runtime(canvas_size=32)
    runtime.max_speed = True
    try:
        runtime.start()
        assert entered.wait(timeout=5.0)
        started = time.perf_counter()
        state = runtime.state()
        elapsed = time.perf_counter() - started
    finally:
        runtime.stop()
        WebSimRuntime._advance_one_step = original_advance

    assert state["maxSpeed"]
    assert elapsed < 0.5


def test_spatial_replay_capacity_is_bounded_for_long_runs() -> None:
    # Spatial transitions hold full-resolution material patches (~200 KB each),
    # so the 50k default would grow the three replays to ~15-20 GB over a long
    # run. The spatial driver must cap them well below that; summary mode (tiny
    # 6-float states) keeps the large default.
    spatial = oracle_runtime(
        canvas_size=64,
        planner_state_kind="spatial_material",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    agent = spatial.agent_driver.agent
    for replay in (
        agent.replay,
        agent.composition_replay,
        agent.passage_replay,
        agent.passage_step_replay,
    ):
        assert replay.data.maxlen is not None and replay.data.maxlen <= 8_000

    summary = oracle_runtime(
        canvas_size=64,
        planner_state_kind="summary",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    assert summary.agent_driver.agent.replay.data.maxlen == 50_000


def test_web_runtime_retains_learned_training_across_new_painting() -> None:
    runtime = oracle_runtime(canvas_size=32)
    agent = runtime.agent_driver.agent
    dynamics = runtime.agent_driver.agent.dynamics
    replay_size = len(runtime.agent_driver.agent.replay)
    trained_transitions = runtime.agent_driver.trained_transitions
    assert trained_transitions > 0
    assert replay_size > 0

    runtime.agent_driver.stopped = True
    assert runtime._restart_after_stop_if_needed()

    assert runtime.agent_driver.agent is agent
    assert runtime.agent_driver.agent.dynamics is dynamics
    assert len(runtime.agent_driver.agent.replay) == replay_size
    assert runtime.agent_driver.trained_transitions == trained_transitions


def test_web_runtime_restarts_after_stop_and_saves_every_fifth_canvas() -> None:
    archive_dir = Path("runs/test_web_runtime_archive")
    runtime = oracle_runtime(canvas_size=32, archive_dir=archive_dir)
    runtime.sim.canvas.paint_at(
        np.asarray([0.0, runtime.sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=1.0,
        dt=0.2,
    )

    for _ in range(4):
        runtime.agent_driver.stopped = True
        runtime._restart_after_stop_if_needed()
        assert runtime.last_saved_canvas is None

    runtime.agent_driver.stopped = True
    runtime._restart_after_stop_if_needed()

    assert runtime.painting_count == 5
    assert runtime.last_saved_canvas is not None
    saved = archive_dir / "painting_0005.png"
    assert saved.is_file()
    assert saved.read_bytes().startswith(b"\x89PNG")
    assert not runtime.agent_driver.stopped
    assert runtime.sim.canvas.material_coverage() == 0.0


def test_web_runtime_restart_lifts_brush_before_next_sim_step() -> None:
    runtime = oracle_runtime(canvas_size=32)
    runtime.sim.load_brush(amount=0.7, tone=1.0)
    runtime.sim.intended_contact_pressure = 1.0
    runtime.sim.contact = runtime.sim.canvas.contact_from_tip(
        np.asarray([0.0, runtime.sim.canvas.distance, 0.0]),
        runtime.sim.intended_contact_pressure,
    )
    runtime.sim.canvas.paint_at(runtime.sim.contact.brush_world, runtime.sim.contact.pressure, tone=1.0, dt=0.2)
    assert runtime.sim.canvas.material_coverage() > 0.0

    runtime.agent_driver.stopped = True
    assert runtime._restart_after_stop_if_needed()
    runtime.sim.step(1.0 / 240.0)

    assert not runtime.sim.brush.loaded
    assert runtime.sim.intended_contact_pressure == 0.0
    assert runtime.sim.contact.pressure == 0.0
    assert runtime.sim.canvas.material_coverage() == 0.0
