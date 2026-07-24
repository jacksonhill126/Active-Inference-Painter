"""Raw-data export helpers for chapter figures and animations.

The chapter orchestrators are intentionally thin: they compute numerical
traces, render educational figures, and delegate persistence to this module.
Every saved figure or GIF can therefore be paired with compressed numeric
arrays plus a JSON manifest that records provenance, shapes, dtypes, and
summary statistics.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from collections.abc import Mapping, Sequence
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np

from .io import default_data_dir

ArrayMap = Mapping[str, Any]

_CHAPTER_RE = re.compile(r"chapter_(\d{2})")
_EXTRAS_RE = re.compile(r"extras")
_DEMO_RE = re.compile(r"demo")
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
_SKIP_CLOSURE_NAMES = {
    "ax",
    "axes",
    "fig",
    "line",
    "lines",
    "txt",
    "text",
    "artists",
    "im",
    "scatter",
}


def _sanitize_name(name: str) -> str:
    """Return a stable NPZ key made from ``name``."""
    safe = _SAFE_NAME_RE.sub("_", str(name).strip()).strip("_")
    return safe or "array"


def _chapter_number(chapter: int | str) -> int:
    """Normalize integer or ``chapter_NN`` input to a chapter number."""
    if isinstance(chapter, str):
        match = _CHAPTER_RE.search(chapter)
        if match:
            number = int(match.group(1))
        else:
            number = int(chapter)
    else:
        number = int(chapter)
    if number <= 0:
        raise ValueError("chapter must be positive")
    return number


def _display_path(path: Path | str) -> str:
    """Return a repository-relative path when possible, otherwise ``str(path)``."""
    p = Path(path)
    try:
        repo_root = Path(__file__).resolve().parents[3]
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    """Convert NumPy/path values into JSON-serializable equivalents."""
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _coerce_numeric_array(name: str, value: Any) -> np.ndarray:
    """Validate and return a finite, non-empty numeric array."""
    arr = np.ma.asarray(value)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name}: arrays must be non-empty")
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name}: arrays must be numeric and non-object")
    arr = np.asarray(arr, dtype=np.result_type(arr.dtype, np.float64))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: arrays must contain only finite values")
    return arr


def _array_summary(arr: np.ndarray) -> dict[str, float | int]:
    """Return compact descriptive statistics for one numeric export array."""
    flat = np.asarray(arr, dtype=float).ravel()
    return {
        "size": int(flat.size),
        "finite_count": int(np.isfinite(flat).sum()),
        "finite_fraction": float(np.isfinite(flat).mean()),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def _prepare_arrays(arrays: ArrayMap) -> dict[str, np.ndarray]:
    """Validate, sanitize, and de-duplicate array names for NPZ storage."""
    prepared: dict[str, np.ndarray] = {}
    for raw_name, value in arrays.items():
        base = _sanitize_name(raw_name)
        name = base
        suffix = 1
        while name in prepared:
            suffix += 1
            name = f"{base}_{suffix}"
        prepared[name] = _coerce_numeric_array(name, value)
    if not prepared:
        raise ValueError("at least one numeric array is required")
    return prepared


def _seed_from_argv(argv: Sequence[str]) -> int | None:
    """Extract a ``--seed`` value from CLI argv when it is explicitly present."""
    for i, token in enumerate(argv):
        if token == "--seed" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
        if token.startswith("--seed="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def chapter_data_dir(chapter: int | str, *, root: Path | str | None = None) -> Path:
    """Return ``output/data/chapter_NN`` for a chapter number or label."""
    number = _chapter_number(chapter)
    base = Path(root) if root is not None else default_data_dir()
    return base / f"chapter_{number:02d}"


def _extra_topic(topic: str) -> str:
    """Return a stable extras topic directory name."""
    safe = _sanitize_name(topic)
    if not safe:
        raise ValueError("topic must not be empty")
    return safe


def extra_data_dir(topic: str, *, root: Path | str | None = None) -> Path:
    """Return the raw-data directory for one extras topic under ``output/data``."""
    base = Path(root) if root is not None else default_data_dir()
    return base / "extras" / _extra_topic(topic)


def _demo_slug(slug: str) -> str:
    """Return a stable demo topic directory name."""
    safe = _sanitize_name(slug)
    if not safe:
        raise ValueError("slug must not be empty")
    return safe


def demo_data_dir(slug: str, *, root: Path | str | None = None) -> Path:
    """Return the raw-data directory for one demo topic under ``output/data``."""
    base = Path(root) if root is not None else default_data_dir()
    return base / "demo" / _demo_slug(slug)


def infer_chapter_from_path(path: Path | str) -> int:
    """Infer ``NN`` from a path containing a ``chapter_NN`` component."""
    for part in Path(path).parts:
        match = _CHAPTER_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not infer chapter_NN from {path!s}")


def infer_extra_topic_from_path(path: Path | str) -> str:
    """Infer the extras topic slug from a path containing ``extras/<topic>``."""
    parts = Path(path).parts
    for index, part in enumerate(parts[:-1]):
        if _EXTRAS_RE.fullmatch(part):
            return _extra_topic(parts[index + 1])
    raise ValueError(f"could not infer extras/<topic> from {path!s}")


def infer_demo_slug_from_path(path: Path | str) -> str:
    """Infer the demo topic slug from a path containing ``demo/<slug>``."""
    parts = Path(path).parts
    for index, part in enumerate(parts[:-1]):
        if _DEMO_RE.fullmatch(part):
            return _demo_slug(parts[index + 1])
    raise ValueError(f"could not infer demo/<slug> from {path!s}")


def data_paths_for_figure(
    figure_path: Path | str,
    *,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return the NPZ/JSON sidecar paths corresponding to a figure artifact."""
    figure = Path(figure_path)
    chapter = infer_chapter_from_path(figure)
    stem = figure.stem
    data_dir = chapter_data_dir(chapter, root=root)
    return data_dir / f"{stem}.npz", data_dir / f"{stem}.json"


def data_paths_for_extra_figure(
    figure_path: Path | str,
    *,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return NPZ/JSON sidecars for a figure under ``output/figures/extras``."""
    figure = Path(figure_path)
    topic = infer_extra_topic_from_path(figure)
    stem = figure.stem
    data_dir = extra_data_dir(topic, root=root)
    return data_dir / f"{stem}.npz", data_dir / f"{stem}.json"


def data_paths_for_demo_figure(
    figure_path: Path | str,
    *,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return NPZ/JSON sidecars for a figure under ``output/figures/demo``."""
    figure = Path(figure_path)
    slug = infer_demo_slug_from_path(figure)
    stem = figure.stem
    data_dir = demo_data_dir(slug, root=root)
    return data_dir / f"{stem}.npz", data_dir / f"{stem}.json"


def _write_data_pair(
    data_dir: Path,
    stem: str,
    arrays: ArrayMap,
    metadata: Mapping[str, Any] | None,
    *,
    figures: Sequence[Path | str] | None = None,
    manifest_fields: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write validated arrays and a JSON manifest to ``data_dir``."""
    prepared = _prepare_arrays(arrays)
    metadata_dict = dict(metadata or {})
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _sanitize_name(stem)
    npz_path = data_dir / f"{safe_stem}.npz"
    json_path = data_dir / f"{safe_stem}.json"
    np.savez_compressed(npz_path, **prepared)

    argv = list(sys.argv[1:])
    seed = metadata_dict.get("seed")
    if seed is None:
        seed = _seed_from_argv(argv)
    summary = {name: _array_summary(arr) for name, arr in prepared.items()}
    manifest = {
        "schema_version": 1,
        **_jsonable(manifest_fields),
        "stem": safe_stem,
        "script": metadata_dict.get("script", Path(sys.argv[0]).name),
        "args": metadata_dict.get("args", argv),
        "seed": seed,
        "figures": [_display_path(path) for path in (figures or [])],
        "arrays": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "summary": summary[name],
            }
            for name, arr in prepared.items()
        },
        "summary": summary,
        "metadata": _jsonable(metadata_dict),
    }
    json_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def save_chapter_data(
    chapter: int | str,
    stem: str,
    arrays: ArrayMap,
    metadata: Mapping[str, Any] | None = None,
    *,
    figures: Sequence[Path | str] | None = None,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write validated chapter arrays to ``.npz`` plus a JSON manifest.

    ``arrays`` must contain finite numeric arrays only. The manifest records
    chapter, script, CLI arguments, optional seed, figure paths, array
    shape/dtype contracts, and per-array summary statistics so the analytical
    quantities visible in the figure are machine-readable.
    """
    number = _chapter_number(chapter)
    return _write_data_pair(
        chapter_data_dir(number, root=root),
        stem,
        arrays,
        metadata,
        figures=figures,
        manifest_fields={"chapter": number},
    )


def save_extra_data(
    topic: str,
    stem: str,
    arrays: ArrayMap,
    metadata: Mapping[str, Any] | None = None,
    *,
    figures: Sequence[Path | str] | None = None,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write validated extras-topic arrays to ``.npz`` plus a JSON manifest."""
    safe_topic = _extra_topic(topic)
    return _write_data_pair(
        extra_data_dir(safe_topic, root=root),
        stem,
        arrays,
        metadata,
        figures=figures,
        manifest_fields={"section": "extras", "topic": safe_topic},
    )


def save_demo_data(
    slug: str,
    stem: str,
    arrays: ArrayMap,
    metadata: Mapping[str, Any] | None = None,
    *,
    figures: Sequence[Path | str] | None = None,
    root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write validated demo-topic arrays to ``.npz`` plus a JSON manifest."""
    safe_slug = _demo_slug(slug)
    return _write_data_pair(
        demo_data_dir(safe_slug, root=root),
        stem,
        arrays,
        metadata,
        figures=figures,
        manifest_fields={"section": "demo", "slug": safe_slug},
    )


def _add_array_if_valid(arrays: dict[str, np.ndarray], name: str, value: Any) -> None:
    """Add ``value`` to ``arrays`` only if it is finite numeric data."""
    try:
        arr = _coerce_numeric_array(name, value)
    except (TypeError, ValueError):
        return
    arrays[_sanitize_name(name)] = arr


def _legend_labels(ax: Any) -> list[str]:
    """Return visible legend labels for an axes object."""
    handles, labels = ax.get_legend_handles_labels()
    out: list[str] = []
    for handle, label in zip(handles, labels):
        if not label:
            continue
        visible = getattr(handle, "get_visible", lambda: True)
        if visible():
            out.append(label)
    return out


def extract_figure_data(fig: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract numeric reconstruction arrays and semantic labels from a figure."""
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"axes": []}
    _add_array_if_valid(arrays, "figure_size_inches", fig.get_size_inches())
    _add_array_if_valid(arrays, "axes_count", [len(fig.axes)])

    for ax_i, ax in enumerate(fig.axes):
        ax_meta = {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "legend_labels": _legend_labels(ax),
            "texts": [text.get_text() for text in ax.texts],
        }
        metadata["axes"].append(ax_meta)
        for line_i, line in enumerate(ax.lines):
            _add_array_if_valid(arrays, f"axes_{ax_i}_line_{line_i}_x", line.get_xdata())
            _add_array_if_valid(arrays, f"axes_{ax_i}_line_{line_i}_y", line.get_ydata())
        for image_i, image in enumerate(ax.images):
            _add_array_if_valid(arrays, f"axes_{ax_i}_image_{image_i}", image.get_array())
        for collection_i, collection in enumerate(ax.collections):
            if hasattr(collection, "get_offsets"):
                offsets = collection.get_offsets()
                if np.asarray(offsets).size:
                    _add_array_if_valid(
                        arrays,
                        f"axes_{ax_i}_collection_{collection_i}_offsets",
                        offsets,
                    )
            if hasattr(collection, "get_array"):
                color_array = collection.get_array()
                if color_array is not None:
                    _add_array_if_valid(
                        arrays,
                        f"axes_{ax_i}_collection_{collection_i}_array",
                        color_array,
                    )
        for patch_i, patch in enumerate(ax.patches):
            if all(hasattr(patch, attr) for attr in ("get_x", "get_y", "get_width", "get_height")):
                _add_array_if_valid(
                    arrays,
                    f"axes_{ax_i}_patch_{patch_i}_bounds",
                    [
                        patch.get_x(),
                        patch.get_y(),
                        patch.get_width(),
                        patch.get_height(),
                    ],
                )
        for text_i, text in enumerate(ax.texts):
            _add_array_if_valid(arrays, f"axes_{ax_i}_text_{text_i}_xy", text.get_position())
    return arrays, metadata


def _collect_numeric_values(
    arrays: dict[str, np.ndarray],
    prefix: str,
    value: Any,
    *,
    depth: int = 0,
) -> None:
    """Collect numeric arrays from animation closure values without object data."""
    if depth > 2 or prefix in _SKIP_CLOSURE_NAMES:
        return
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.dtype != object and np.issubdtype(arr.dtype, np.number):
        _add_array_if_valid(arrays, prefix, arr)
        return
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:32]:
            _collect_numeric_values(arrays, f"{prefix}_{key}", item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, item in enumerate(list(value)[:32]):
            _collect_numeric_values(arrays, f"{prefix}_{i}", item, depth=depth + 1)
        return
    if isinstance(value, Number):
        _add_array_if_valid(arrays, prefix, [value])


def extract_animation_data(anim: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract animation time-series from explicit raw data, closures, and figure artists."""
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"animation_class": type(anim).__name__}

    explicit = getattr(anim, "_raw_data", None)
    if isinstance(explicit, Mapping):
        for name, value in explicit.items():
            _collect_numeric_values(arrays, f"raw_{name}", value)

    func = getattr(anim, "_func", None)
    if func is not None:
        closure_vars = inspect.getclosurevars(func)
        for name, value in closure_vars.nonlocals.items():
            _collect_numeric_values(arrays, f"closure_{name}", value)

    fig = getattr(anim, "_fig", None)
    if fig is not None:
        fig_arrays, fig_metadata = extract_figure_data(fig)
        arrays.update({f"figure_{name}": arr for name, arr in fig_arrays.items()})
        metadata["figure"] = fig_metadata
    _add_array_if_valid(arrays, "frame_count", [getattr(anim, "save_count", 0)])
    return arrays, metadata


def save_figure_data(
    fig: Any,
    figure_path: Path | str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path] | None:
    """Save figure-extracted raw data for chapter or extras figure paths."""
    try:
        chapter = infer_chapter_from_path(figure_path)
    except ValueError:
        chapter = None
    try:
        topic = infer_extra_topic_from_path(figure_path) if chapter is None else None
    except ValueError:
        topic = None
    if chapter is None and topic is None:
        return None
    arrays, figure_metadata = extract_figure_data(fig)
    export_metadata = {
        "kind": "figure",
        "figure_metadata": figure_metadata,
        **dict(metadata or {}),
    }
    if chapter is None and topic is not None:
        return save_extra_data(
            topic,
            Path(figure_path).stem,
            arrays,
            export_metadata,
            figures=[figure_path],
        )
    return save_chapter_data(
        chapter,
        Path(figure_path).stem,
        arrays,
        export_metadata,
        figures=[figure_path],
    )


def save_animation_data(
    anim: Any,
    animation_path: Path | str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path] | None:
    """Save animation-extracted raw data for chapter or extras animation paths."""
    try:
        chapter = infer_chapter_from_path(animation_path)
    except ValueError:
        chapter = None
    try:
        topic = infer_extra_topic_from_path(animation_path) if chapter is None else None
    except ValueError:
        topic = None
    if chapter is None and topic is None:
        return None
    arrays, animation_metadata = extract_animation_data(anim)
    export_metadata = {
        "kind": "animation",
        "animation_metadata": animation_metadata,
        **dict(metadata or {}),
    }
    if chapter is None and topic is not None:
        return save_extra_data(
            topic,
            Path(animation_path).stem,
            arrays,
            export_metadata,
            figures=[animation_path],
        )
    return save_chapter_data(
        chapter,
        Path(animation_path).stem,
        arrays,
        export_metadata,
        figures=[animation_path],
    )
