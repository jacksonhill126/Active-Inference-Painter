# `tests/demo/` - agent guide

Smoke tests for application demo orchestrators under `demo/`.

## Rules

1. Discover slugs from `active_inference.demo_topics`; do not hard-code topic lists.
2. Require fresh NPZ+JSON sidecars under `output/data/demo/<slug>/`.
3. Use subprocess `--save` runs with `MPLBACKEND=Agg` and `PYTHONWARNINGS=error`.

## Running

```bash
pytest tests/demo -v
```
