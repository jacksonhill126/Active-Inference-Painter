import errno
import io
import numpy as np
from pathlib import Path
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler

from PIL import Image
import pytest

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


def test_web_runtime_state_contains_arm_canvas_and_contact() -> None:
    runtime = WebSimRuntime(canvas_size=32)
    state = runtime.state()
    assert state["canvas"]["distance"] == 17.0
    assert len(state["points"]) == 3
    assert "pressure" in state["contact"]
    assert state["contact"]["touching"] is False
    assert state["contact"]["projectedOnCanvas"] == state["contact"]["onCanvas"]
    assert "yaw" in state["pose"]
    assert state["telemetryLog"]["sampleCount"] >= 1
    assert state["telemetryLog"]["csvEndpoint"] == "/api/telemetry.csv"
    assert "velocityRadS" in state["motor"]["yaw"]
    assert "encoderAngleDeg" in state["motor"]["yaw"]
    assert "positionErrorDeg" in state["motor"]["yaw"]
    assert "elasticDeflectionDeg" in state["motor"]["yaw"]
    assert "encoderStdDeg" in state["motor"]["yaw"]


def test_web_runtime_can_enable_spatial_material_planner() -> None:
    runtime = WebSimRuntime(
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


def test_web_runtime_uses_bounded_passage_planning_budget() -> None:
    runtime = WebSimRuntime(
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

    spatial = parser.parse_args(["--planner-state-kind", "spatial_material"])
    summary = parser.parse_args([])
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
    runtime = WebSimRuntime(
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
    runtime = WebSimRuntime(canvas_size=32)
    assert not runtime.state()["maxSpeed"]
    response = runtime.command({"type": "toggle_max_speed"})
    assert response["ok"]
    assert runtime.state()["maxSpeed"]
    png = runtime.canvas_png()
    assert png.startswith(b"\x89PNG")


def test_web_runtime_exposes_upper_arm_and_rolled_elbow_axes() -> None:
    runtime = WebSimRuntime(canvas_size=32)
    runtime.sim.actual_pose.roll = 31.0

    state = runtime.state()

    assert np.linalg.norm(state["upperArmAxis"]) == pytest.approx(1.0)
    assert np.linalg.norm(state["elbowHingeAxis"]) == pytest.approx(1.0)
    assert np.dot(state["upperArmAxis"], state["elbowHingeAxis"]) == pytest.approx(0.0, abs=1e-12)


def test_web_canvas_png_renders_gray_ground_with_visible_white_and_black_paint() -> None:
    runtime = WebSimRuntime(
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
    runtime = WebSimRuntime(canvas_size=32, telemetry_sample_period=1.0 / 240.0)
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


def test_web_runtime_default_telemetry_is_sparse_rolling_window() -> None:
    runtime = WebSimRuntime(
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
    state = WebSimRuntime(canvas_size=32).state()
    assert state["codeVersion"] == code_version()
    assert state["codeVersion"] != "unknown"
    assert "+code." in state["codeVersion"]


def test_web_runtime_state_consumes_stopped_episode_before_reporting() -> None:
    runtime = WebSimRuntime(canvas_size=32)
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
        source.mkdir(parents=True)
        web.mkdir()
        package.write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
        (source / "versioned.py").write_text("VALUE = 1\n", encoding="utf-8")
        (web / "main.js").write_text("console.log('one');\n", encoding="utf-8")
        stamp = root / ".stamp.json"

        first = code_build_info(root=root, metadata_path=stamp)
        second = code_build_info(root=root, metadata_path=stamp)
        (web / "main.js").write_text("console.log('two');\n", encoding="utf-8")
        third = code_build_info(root=root, metadata_path=stamp)

        assert first.build == second.build
        assert third.build == second.build + 1
        assert first.fingerprint != third.fingerprint
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_web_runtime_driver_stop_callback_immediately_restarts_episode() -> None:
    runtime = WebSimRuntime(canvas_size=32)
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
    runtime = WebSimRuntime(canvas_size=32)
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
    spatial = WebSimRuntime(
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

    summary = WebSimRuntime(
        canvas_size=64,
        planner_state_kind="summary",
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
    )
    assert summary.agent_driver.agent.replay.data.maxlen == 50_000


def test_web_runtime_retains_learned_training_across_new_painting() -> None:
    runtime = WebSimRuntime(canvas_size=32)
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
    runtime = WebSimRuntime(canvas_size=32, archive_dir=archive_dir)
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
    runtime = WebSimRuntime(canvas_size=32)
    runtime.sim.paint_enabled = True
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

    assert not runtime.sim.paint_enabled
    assert runtime.sim.intended_contact_pressure == 0.0
    assert runtime.sim.contact.pressure == 0.0
    assert runtime.sim.canvas.material_coverage() == 0.0
