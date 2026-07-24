# `tests/menu/` — tests for `src/active_inference/menu/`

Tests for the stdlib-only text menu that backs `run.sh`.

| Source file | Test file |
|---|---|
| `menu/runner.py`, `menu/tui.py` | [`test_runner.py`](test_runner.py) |

## Running

```bash
pytest tests/menu -v
```

## What's covered

`test_runner.py` (14 test functions) exercises:

- **Discovery** — `discover_chapters` / `discover_scripts` find the chapter dirs
  and their `example_*.py` / `animation_*.py` / `visualize_*.py` scripts.
- **Classification** — scripts are bucketed into examples / animations /
  visualizations by their filename convention.
- **Menu rendering** — the text menu lists chapters and scripts correctly.
- **Script resolution** — a user selection maps back to the right path.
