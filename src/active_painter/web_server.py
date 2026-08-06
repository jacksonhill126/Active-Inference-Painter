from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .arm_agent_driver import (
    BOOTSTRAP_GENERATORS,
    ORACLE_OBSERVATION_ACCESS_MODE,
    SENSOR_OBSERVATION_ACCESS_MODE,
)
from .config import (
    SPATIAL_MATERIAL_PLANNER_STATE_KIND,
    SUMMARY_PLANNER_STATE_KIND,
)
from .version import code_version
from .web_robot_model import load_robot_visual_model
from .web_runtime import WebSimRuntime


WEB_ROOT = Path(__file__).resolve().parents[2] / "web"

CLIENT_DISCONNECT_ERRNOS = {
    errno.EPIPE,
    errno.ECONNABORTED,
    errno.ECONNRESET,
    10053,
    10054,
}


def render_index_html(path: Path, version: str | None = None) -> bytes:
    html = path.read_text(encoding="utf-8")
    version_text = version or code_version()
    return html.replace(
        '<span id="codeVersion" class="version-pill">v0.1.0</span>',
        f'<span id="codeVersion" class="version-pill">v{version_text}</span>',
    ).encode("utf-8")


def is_client_disconnect(exc: OSError) -> bool:
    return isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)) or (
        exc.errno in CLIENT_DISCONNECT_ERRNOS
    )


class PainterWebServer(ThreadingHTTPServer):
    runtime: WebSimRuntime
    web_root: Path


def bind_server(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> PainterWebServer:
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            return PainterWebServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if exc.errno not in {errno.EADDRINUSE, errno.EACCES, 10013, 10048, 98}:
                raise
    raise OSError(f"Could not bind {host}:{port}-{port + 19}") from last_error


class PainterRequestHandler(BaseHTTPRequestHandler):
    server: PainterWebServer

    def handle(self) -> None:
        try:
            super().handle()
        except OSError as exc:
            if not is_client_disconnect(exc):
                raise

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(data).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/version":
            build = self.server.runtime.code_build
            self._send_json(
                {
                    "version": build.version,
                    "packageVersion": build.package_version,
                    "codeBuild": build.build,
                    "codeFingerprint": build.short_fingerprint,
                }
            )
            return
        if parsed.path == "/api/state":
            self._send_json(self.server.runtime.state())
            return
        if parsed.path == "/api/robot-model":
            self._send_json(load_robot_visual_model())
            return
        if parsed.path == "/api/canvas.png":
            self._send_bytes(self.server.runtime.canvas_png(), "image/png")
            return
        if parsed.path.startswith("/api/camera/") and parsed.path.endswith(".png"):
            camera_name = unquote(parsed.path[len("/api/camera/") : -len(".png")])
            try:
                body = self.server.runtime.camera_png(camera_name)
            except (KeyError, RuntimeError, ValueError) as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    HTTPStatus.CONFLICT,
                )
                return
            self._send_bytes(body, "image/png")
            return
        if parsed.path == "/api/telemetry.csv":
            self._send_bytes(self.server.runtime.telemetry_csv(), "text/csv; charset=utf-8")
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/command":
            self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(self.server.runtime.command(data))

    def _serve_static(self, request_path: str) -> None:
        rel = "index.html" if request_path in ("", "/") else unquote(request_path.lstrip("/"))
        root = self.server.web_root.resolve()
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            self._send_json({"ok": False, "error": "forbidden"}, HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.name == "index.html":
            self._send_bytes(
                render_index_html(path, self.server.runtime.code_build.version),
                "text/html; charset=utf-8",
            )
            return
        self._send_bytes(path.read_bytes(), content_type)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--planner-state-kind",
        choices=(
            SPATIAL_MATERIAL_PLANNER_STATE_KIND,
            SUMMARY_PLANNER_STATE_KIND,
        ),
        default=SPATIAL_MATERIAL_PLANNER_STATE_KIND,
        help=(
            "planner representation; spatial_material is the provisional "
            "low-level research baseline, while summary is an obsolete "
            "compatibility fixture"
        ),
    )
    parser.add_argument("--spatial-grid-size", type=int, default=16)
    parser.add_argument("--stroke-tone-prior", choices=("black", "white", "random"), default="random")
    parser.add_argument("--save-every-paintings", type=int, default=5)
    parser.add_argument("--archive-dir", default="runs/web")
    parser.add_argument("--telemetry-max-samples", type=int, default=54_000)
    parser.add_argument("--telemetry-sample-hz", type=float, default=15.0)
    parser.add_argument("--driver-bootstrap-transitions", type=int, default=None)
    parser.add_argument("--driver-bootstrap-train-steps", type=int, default=None)
    parser.add_argument(
        "--driver-bootstrap-generator",
        choices=BOOTSTRAP_GENERATORS,
        default="motion_manifold",
        help=(
            "bootstrap mark source; motion_manifold seeds the likelihoods on the "
            "body's own reachable-motion manifold, random_stroke retains the "
            "previous iid source for attribution"
        ),
    )
    parser.add_argument(
        "--driver-bootstrap-composition-train-steps",
        type=int,
        default=0,
        help=(
            "canvas/relational gradient steps at each bootstrap episode boundary; "
            "a gradient budget, not an objective term. Measured: the compression "
            "gap does not discriminate structure below a few hundred steps"
        ),
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-save-every-transitions", type=int, default=10)
    parser.add_argument("--device", default=None, help="torch device for the planner, e.g. cuda, cuda:0, cpu (default: cuda if available)")
    parser.add_argument(
        "--plant-backend",
        choices=("native", "mujoco"),
        default="native",
        help="realized arm dynamics/contact backend; painting remains in VerticalCanvas",
    )
    parser.add_argument(
        "--observation-mode",
        choices=(
            SENSOR_OBSERVATION_ACCESS_MODE,
            ORACLE_OBSERVATION_ACCESS_MODE,
        ),
        default=SENSOR_OBSERVATION_ACCESS_MODE,
        help=(
            "sensor_equivalent fails closed by default; combine it with "
            "--enable-provisional-sensor-policy and --plant-backend mujoco for "
            "the simulation-only camera/body-posterior smoke loop. "
            "oracle_material_state is an explicit diagnostic-only legacy baseline"
        ),
    )
    parser.add_argument(
        "--enable-provisional-sensor-policy",
        action="store_true",
        help=(
            "run the MuJoCo painting-policy loop from registered camera/body "
            "posteriors and an independent forecast model; simulation-only, "
            "not hardware-calibrated"
        ),
    )
    return parser


def resolved_bootstrap(args: argparse.Namespace) -> tuple[int, int]:
    transitions = args.driver_bootstrap_transitions
    train_steps = args.driver_bootstrap_train_steps
    if transitions is None:
        transitions = 96
    if train_steps is None:
        train_steps = (
            24
            if args.planner_state_kind
            == SPATIAL_MATERIAL_PLANNER_STATE_KIND
            else 180
        )
    return transitions, train_steps


def main() -> None:
    # Cap torch intraop threads: the planner's small models lose more to
    # thread oversubscription against the sim/render threads than they gain
    # from extra cores.
    import os

    import torch

    torch.set_num_threads(max(2, min(8, (os.cpu_count() or 8) - 4)))
    args = build_parser().parse_args()
    bootstrap_transitions, bootstrap_train_steps = resolved_bootstrap(args)
    if args.enable_provisional_sensor_policy:
        # The existing bootstrap consumes exact synthetic material states and
        # is deliberately oracle-labelled. The runnable sensor smoke profile
        # starts without it until a sensor-posterior corpus/checkpoint exists.
        bootstrap_transitions = 0
        bootstrap_train_steps = 0
    stroke_tone_prior = {"black": 1.0, "white": 0.0, "random": None}[args.stroke_tone_prior]
    print(
        "Initializing Active-Inference Arm Painter "
        f"({args.planner_state_kind}, bootstrap={bootstrap_transitions}/{bootstrap_train_steps}, "
        f"generator={args.driver_bootstrap_generator}, "
        f"bootstrap-composition-steps={args.driver_bootstrap_composition_train_steps})...",
        flush=True,
    )
    runtime = WebSimRuntime(
        canvas_size=args.canvas_size,
        speed=args.speed,
        planner_state_kind=args.planner_state_kind,
        spatial_grid_size=args.spatial_grid_size,
        stroke_tone_prior=stroke_tone_prior,
        save_every_paintings=args.save_every_paintings,
        archive_dir=args.archive_dir,
        telemetry_max_samples=args.telemetry_max_samples,
        telemetry_sample_period=1.0 / args.telemetry_sample_hz if args.telemetry_sample_hz > 0 else 0.0,
        driver_bootstrap_transitions=bootstrap_transitions,
        driver_bootstrap_train_steps=bootstrap_train_steps,
        driver_bootstrap_generator=args.driver_bootstrap_generator,
        driver_bootstrap_composition_train_steps=args.driver_bootstrap_composition_train_steps,
        checkpoint_path=args.checkpoint_path,
        checkpoint_save_every_transitions=args.checkpoint_save_every_transitions,
        device=args.device,
        plant_backend=args.plant_backend,
        observation_access_mode=args.observation_mode,
        provisional_sensor_policy=args.enable_provisional_sensor_policy,
    )
    if runtime.agent_driver.observation_boundary_blocked:
        print(
            "Active inference disabled: the camera-conditioned painting posterior "
            "is connected, but sensor-conditioned body initialization and the "
            "action-conditioned observation loop are not; "
            "hidden simulator state remains denied.",
            flush=True,
        )
    elif args.observation_mode == ORACLE_OBSERVATION_ACCESS_MODE:
        print(
            "WARNING: oracle_material_state exposes hidden process truth to the "
            "model and is diagnostic-only.",
            flush=True,
        )
    elif args.enable_provisional_sensor_policy:
        print(
            "PROVISIONAL SENSOR SIMULATION: policy inference uses registered "
            "camera/body beliefs and an independent MuJoCo forecast model; "
            "no hardware-calibration claim is made.",
            flush=True,
        )
    print(f"Planner device: {runtime.agent_driver.agent.device}", flush=True)
    server = bind_server(args.host, args.port, PainterRequestHandler)
    server.runtime = runtime
    server.web_root = WEB_ROOT
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    print(f"Active-Inference Arm Painter web viewer: {url}", flush=True)
    if port != args.port:
        print(f"Port {args.port} was unavailable; using {port}.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    threading.Timer(1.0, runtime.start).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.stop()


if __name__ == "__main__":
    main()
