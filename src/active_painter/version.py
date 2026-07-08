from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import metadata
from pathlib import Path
import threading
import tomllib


DIST_NAME = "active-inference-painter"
CODE_VERSION_FILE = ".active_painter_code_version.json"
_CODE_VERSION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class CodeBuildInfo:
    package_version: str
    build: int
    fingerprint: str

    @property
    def short_fingerprint(self) -> str:
        return self.fingerprint[:10]

    @property
    def version(self) -> str:
        return f"{self.package_version}+code.{self.build}"


def package_version() -> str:
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        pass

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"
    return version if isinstance(version, str) else "unknown"


def code_build_info(
    root: Path | None = None,
    metadata_path: Path | None = None,
) -> CodeBuildInfo:
    """Return a monotonically bumped source-build identifier.

    The counter advances when the source fingerprint changes. This is a small
    runtime build stamp for the web viewer, not a package release version.
    """

    repo_root = root or Path(__file__).resolve().parents[2]
    stamp_path = metadata_path or repo_root / CODE_VERSION_FILE
    fingerprint = source_fingerprint(repo_root)
    with _CODE_VERSION_LOCK:
        previous = _read_code_version_file(stamp_path)
        build = int(previous.get("build", 0)) if isinstance(previous.get("build"), int) else 0
        if previous.get("fingerprint") != fingerprint:
            build += 1
            _write_code_version_file(stamp_path, {"build": build, "fingerprint": fingerprint})
    return CodeBuildInfo(package_version=package_version(), build=build, fingerprint=fingerprint)


def code_version() -> str:
    return code_build_info().version


def source_fingerprint(root: Path | None = None) -> str:
    repo_root = root or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in _source_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for rel in ("pyproject.toml",):
        path = root / rel
        if path.is_file():
            candidates.append(path)
    for folder in (root / "src" / "active_painter", root / "web"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if "__pycache__" in path.parts:
                continue
            candidates.append(path)
    return sorted(candidates)


def _read_code_version_file(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_code_version_file(path: Path, data: dict[str, object]) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return
