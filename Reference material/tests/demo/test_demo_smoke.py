"""Headless smoke tests for demo application orchestrators."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from active_inference.demo_topics import demo_topic_slugs
from active_inference.menu.runner import discover_demo_scripts

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "demo"
TOPICS = list(demo_topic_slugs())
SCRIPT_CASES = [
    (slug, script.name)
    for slug in TOPICS
    for script in discover_demo_scripts(slug)
]


def _fresh_data_pairs(
    data_root: Path,
    slug: str,
    stem: str,
    started_ns: int,
) -> list[tuple[Path, Path]]:
    """Return JSON/NPZ sidecars for ``slug`` written after ``started_ns``."""
    data_dir = data_root / "demo" / slug
    pairs: list[tuple[Path, Path]] = []
    for json_path in data_dir.glob(f"{stem}.json"):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        if json_path.stat().st_mtime_ns >= started_ns or npz_path.stat().st_mtime_ns >= started_ns:
            pairs.append((npz_path, json_path))
    return pairs


@pytest.mark.parametrize("slug,script_name", SCRIPT_CASES, ids=[f"{s}/{n}" for s, n in SCRIPT_CASES])
def test_demo_topic_script_runs_and_exports_raw_data(
    slug: str,
    script_name: str,
    tmp_path: Path,
) -> None:
    """Every demo script runs headlessly and writes fresh NPZ+JSON sidecars."""
    script = DEMO_ROOT / slug / script_name
    output_root = tmp_path / "output"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "error"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ACTIVE_INFERENCE_OUTPUT_ROOT"] = str(output_root)
    started_ns = time.time_ns()

    result = subprocess.run(
        [sys.executable, str(script), "--save"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert _fresh_data_pairs(output_root / "data", slug, script.stem, started_ns), (
        f"{slug}/{script_name} did not export fresh raw data"
    )


@pytest.mark.parametrize("slug", TOPICS, ids=TOPICS)
def test_declared_demo_topic_scripts_exist(slug: str) -> None:
    """Registry-declared demo topics have matching thin script wrappers."""
    assert (DEMO_ROOT / slug / f"visualize_{slug}.py").is_file()
