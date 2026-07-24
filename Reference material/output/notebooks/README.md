# `output/notebooks/` — generated Jupyter notebooks

One notebook per chapter, extras topic, and application demo. Each notebook
sections mirror the runnable orchestrators in `chapters/`, `extras/`, and
`demo/`. Regenerate after orchestrator changes; committed copies are available
on GitHub for reading-group use.

## Layout

| Path | Source |
|---|---|
| `chapter_01.ipynb` … `chapter_14.ipynb` | `chapters/chapter_NN/*.py` |
| `extras/<topic>.ipynb` | `extras/<topic>/*.py` |
| `demo/<slug>.ipynb` | `demo/<slug>/*.py` |

## Regenerate

```bash
uv sync --extra notebooks
uv run python scripts/export_notebooks.py

# one chapter
uv run python scripts/export_notebooks.py --chapters 3

# chapters only (skip extras and demos)
uv run python scripts/export_notebooks.py --no-extras --no-demos

# validate section inventory
uv run python scripts/validate_notebook_exports.py
```

## Open locally

```bash
uv sync --extra notebooks
jupyter lab output/notebooks/chapter_01.ipynb
```

Static examples and animations run inline with `%matplotlib inline`.
Animation cells render GIFs via `--save` and embed them with
`IPython.display.Image`. Interactive slider scripts are documented in
markdown only — run them from a terminal or `./run.sh --web`.

## Reading group

Cohort-based textbook reading groups:
[textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/)
