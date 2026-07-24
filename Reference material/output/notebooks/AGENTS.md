# `output/notebooks/` — agent guide

Generated Jupyter notebooks. **Do not hand-edit** `.ipynb` cells — regenerate
from orchestrators instead.

## Rules

1. Notebooks are produced by `scripts/export_notebooks.py`, which calls
   `active_inference.utils.notebooks`.
2. Discovery uses `active_inference.menu.runner` — the same inventory as
   smoke tests and `run_all_figures.py`, plus `interactive_*` scripts as
   markdown-only sections.
3. Generated `.ipynb` files are tracked in git for reading-group use; regenerate
   with `scripts/export_notebooks.py` when orchestrators change.
4. Interactive scripts are never executed inside exported notebook kernels.

## Regenerate

```bash
uv run python scripts/export_notebooks.py --clean
uv run python scripts/validate_notebook_exports.py
```

## Cell policy

| Script kind | Notebook behavior |
|---|---|
| `example_*`, `visualize_*`, `NN_*` | `runpy.run_path` → inline `plt.show()` |
| `animation_*` | subprocess `--save` + `Image(...)` embed |
| `interactive_*` | markdown instructions only |

## Path helpers

- `default_notebook_dir()` → `output/notebooks/`
- `chapter_notebook_path(n)` → `chapter_NN.ipynb`
- `extra_notebook_path(topic)` → `extras/<topic>.ipynb`
- `demo_notebook_path(slug)` → `demo/<slug>.ipynb`

Honor `ACTIVE_INFERENCE_OUTPUT_ROOT` like `utils.io`.
