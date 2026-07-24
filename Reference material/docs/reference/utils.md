# `active_inference.utils` — module reference

Small helpers shared across the package. These modules provide grids, logging,
paths, and the raw NPZ+JSON export contract used by chapter and extras
`--save` runs.

## `utils.grids`

Constructs the discretized state domains used by `GridBayesianInference`
and the visualization helpers.

| Symbol | Role |
|---|---|
| `make_grid(low, high, n_points=500)` | Evenly-spaced 1-D `np.linspace` grid; validates bounds and finiteness. |
| `make_2d_grid(x_low, x_high, y_low, y_high, n_x=200, n_y=200)` | Two coordinate vectors for rectangular joint-density grids. |

```python
from active_inference import make_grid

x_grid = make_grid(0.0, 5.0, 500)
```

## `utils.logging`

Lightweight stdlib-logger factory so chapter scripts share a consistent
format and don't double-attach handlers when the same logger name is
requested multiple times.

| Symbol | Role |
|---|---|
| `get_logger(name="active_inference", level=logging.INFO)` | Returns an idempotent, formatted, non-propagating logger. |

```python
from active_inference import get_logger

LOG = get_logger("ch3.ex5")
LOG.info("Posterior mode = %.4f", 2.176)
```

## `utils.io`

Path conventions for figures and serialized data produced by chapter
scripts.

| Symbol | Role |
|---|---|
| `default_figure_dir()` | `<repo>/output/figures/`, or `ACTIVE_INFERENCE_FIGURE_DIR` / `ACTIVE_INFERENCE_OUTPUT_ROOT/figures` when overridden. |
| `default_data_dir()` | `<repo>/output/data/`, or `ACTIVE_INFERENCE_DATA_DIR` / `ACTIVE_INFERENCE_OUTPUT_ROOT/data` when overridden. |
| `default_demo_figure_dir(slug)` | `<repo>/output/figures/demo/<slug>/` for one application demo topic. |
| `ensure_dir(path)` | `mkdir -p` semantics; returns the resolved `Path`. |

```python
from active_inference.utils.io import default_figure_dir, ensure_dir

out = ensure_dir(default_figure_dir() / "chapter_03")
fig.savefig(out / "posterior.png")
```

## `utils.export`

Raw-data sidecars for figures and animations. `save_chapter_data`,
`save_extra_data`, and `save_demo_data` are the direct APIs; `save_or_show` and
`save_animation` call the extraction helpers automatically for paths under
`output/figures/chapter_NN/`, `output/figures/extras/<topic>/`, and
`output/figures/demo/<slug>/`.

| Symbol | Role |
|---|---|
| `save_chapter_data(chapter, stem, arrays, metadata, figures=...)` | Validate finite numeric arrays and write paired `NPZ` + `JSON` files under `output/data/chapter_NN/`. |
| `save_extra_data(topic, stem, arrays, metadata, figures=...)` | Validate finite numeric arrays and write paired `NPZ` + `JSON` files under `output/data/extras/<topic>/`. |
| `save_demo_data(slug, stem, arrays, metadata, figures=...)` | Validate finite numeric arrays and write paired `NPZ` + `JSON` files under `output/data/demo/<slug>/`. |
| `save_figure_data(fig, figure_path, metadata=...)` | Extract reconstructable Matplotlib figure data and write sidecars beside a saved figure. |
| `save_animation_data(anim, animation_path, metadata=...)` | Extract reconstructable animation frame/figure data and write sidecars beside a saved GIF. |
| `extract_figure_data(fig)` | Pull line, scatter, image, text-position, and axes metadata from a Matplotlib figure for reconstruction. |
| `extract_animation_data(anim)` | Pull explicit/closure time-series plus figure data from a Matplotlib animation. |
| `chapter_data_dir(chapter, root=...)` | Resolve `output/data/chapter_NN` for a chapter number or `chapter_NN` label. |
| `extra_data_dir(topic, root=...)` | Resolve `output/data/extras/<topic>` for an extras topic slug. |
| `demo_data_dir(slug, root=...)` | Resolve `output/data/demo/<slug>` for an application demo topic slug. |
| `infer_chapter_from_path(path)` | Infer a chapter number from a path containing a `chapter_NN` component. |
| `infer_extra_topic_from_path(path)` | Infer an extras topic slug from a path containing an `extras/<topic>` component. |
| `infer_demo_slug_from_path(path)` | Infer a demo topic slug from a path containing a `demo/<slug>` component. |
| `data_paths_for_figure(path)` | Map `output/figures/chapter_NN/name.png` to `output/data/chapter_NN/name.npz` and `.json`. |
| `data_paths_for_extra_figure(path)` | Map `output/figures/extras/<topic>/name.png` to `output/data/extras/<topic>/name.npz` and `.json`. |
| `data_paths_for_demo_figure(path)` | Map `output/figures/demo/<slug>/name.png` to `output/data/demo/<slug>/name.npz` and `.json`. |

Validate generated sidecars with:

```bash
uv run python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14
uv run python scripts/validate_raw_data_exports.py --root output/data
```

## `utils.notebooks`

Jupyter notebook export for chapter, extras, and demo orchestrators.
Requires `nbformat` (included in the `dev` dependency group and the
`notebooks` optional extra).

| Symbol | Role |
|---|---|
| `default_notebook_dir()` | `<repo>/output/notebooks/`, honoring `ACTIVE_INFERENCE_OUTPUT_ROOT`. |
| `chapter_notebook_path(chapter)` | Resolve `output/notebooks/chapter_NN.ipynb`. |
| `extra_notebook_path(topic)` | Resolve `output/notebooks/extras/<topic>.ipynb`. |
| `demo_notebook_path(slug)` | Resolve `output/notebooks/demo/<slug>.ipynb`. |
| `build_notebook(scripts, *, title, source_folder, ...)` | Assemble an nbformat v4 notebook from `ScriptEntry` descriptors. |
| `write_notebook(notebook, path)` | Validate and write a notebook file. |
| `export_chapter_notebook(chapter, ...)` | Export one chapter notebook. |
| `export_extra_notebook(topic, ...)` | Export one extras-topic notebook. |
| `export_demo_notebook(slug, ...)` | Export one demo notebook. |
| `export_all_notebooks(...)` | Batch export; returns `ExportResult`. |
| `ExportResult` | Dataclass summarizing written chapter, extras, and demo notebook paths. |
| `read_module_docstring(path)` | Parse a script docstring without executing it. |
| `relative_repo_path(path)` | Path relative to the repository root. |

```bash
uv sync --extra notebooks
uv run python scripts/export_notebooks.py
uv run python scripts/validate_notebook_exports.py
```

## Conventions

- All paths are returned as `pathlib.Path` objects.
- Logger names should mirror the chapter / module hierarchy
  (`"ch3.ex5"`, `"core.inference"`).
- Raw-data NPZ arrays must be finite, numeric, non-empty, and non-object; JSON
  manifests record script provenance, seed when present, figure paths,
  shapes/dtypes, and summary statistics.
