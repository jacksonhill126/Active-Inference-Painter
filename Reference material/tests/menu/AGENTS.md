# `tests/menu/` — agent guide

Tests for `src/active_inference/menu/` (the stdlib text menu behind `run.sh`).

```
src/active_inference/menu/{runner,tui}.py   ↔   tests/menu/test_runner.py
```

`test_runner.py` holds the discovery / classification / rendering / resolution
test classes (14 test functions). There is no display dependency — the menu is pure
stdlib, so tests call the functions directly with no subprocess or TTY.

## When adding a menu feature

1. Add a test class to `test_runner.py` (or a new `test_<module>.py` if you add a
   new menu module).
2. Cover at least: the new discovery/classification rule against a temp dir tree,
   and the rendered-output / resolution path it feeds.

## Tips

- Build fixture chapter trees with `tmp_path` rather than touching the real
  `chapters/` dir, so discovery tests stay hermetic.
- The classification rule keys off the `example_*.py` / `animation_*.py` /
  `visualize_*.py` filename convention — assert against it, not against counts
  that drift as chapters are added.
