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

from .version import code_version
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
        if parsed.path == "/api/canvas.png":
            self._send_bytes(self.server.runtime.canvas_png(), "image/png")
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
    parser.add_argument("--planner-state-kind", choices=("summary", "spatial_material"), default="summary")
    parser.add_argument("--spatial-grid-size", type=int, default=16)
    parser.add_argument("--stroke-tone-prior", choices=("black", "white", "random"), default="random")
    parser.add_argument("--save-every-paintings", type=int, default=5)
    parser.add_argument("--archive-dir", default="runs/web")
    parser.add_argument("--telemetry-max-samples", type=int, default=18_000)
    parser.add_argument("--telemetry-sample-hz", type=float, default=60.0)
    parser.add_argument("--driver-bootstrap-transitions", type=int, default=None)
    parser.add_argument("--driver-bootstrap-train-steps", type=int, default=None)
    return parser


def resolved_bootstrap(args: argparse.Namespace) -> tuple[int, int]:
    transitions = args.driver_bootstrap_transitions
    train_steps = args.driver_bootstrap_train_steps
    if transitions is None:
        transitions = 96
    if train_steps is None:
        train_steps = 24 if args.planner_state_kind == "spatial_material" else 180
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
    stroke_tone_prior = {"black": 1.0, "white": 0.0, "random": None}[args.stroke_tone_prior]
    print(
        "Initializing Active-Inference Arm Painter "
        f"({args.planner_state_kind}, bootstrap={bootstrap_transitions}/{bootstrap_train_steps})...",
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
    )
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
