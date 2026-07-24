# Application demos

Thin orchestrators under [`demo/`](../../demo/) compose existing library APIs
into application-themed figures. Workflows live in
[`src/active_inference/demo_workflows.py`](../../src/active_inference/demo_workflows.py);
registry metadata in
[`src/active_inference/demo_topics.py`](../../src/active_inference/demo_topics.py).

## Public API

| Symbol | Module | Role |
| --- | --- | --- |
| `build_eye_saccades_demo` | `demo_workflows` | POMDP fixation grid + first-action EFE bars |
| `build_bicycle_demo` | `demo_workflows` | Multivariate AIF balance trace + fault compensation |
| `build_drone_flight_demo` | `demo_workflows` | POMDP lattice + navigation path + LGS posterior |
| `build_demo` | `demo_topics` | Registry dispatch by slug |
| `demo_topic_slugs`, `demo_topic_spec` | `demo_topics` | Registry introspection |
| `main_visualize` | `demo_topics` | Shared `--save` CLI for `visualize_<slug>.py` wrappers |
| `save_demo_data` | `utils.export` | NPZ+JSON export to `output/data/demo/<slug>/` |

## Running

```bash
python demo/eye_saccades/visualize_eye_saccades.py --save
uv run pytest tests/demo tests/test_demo_workflows -q
```

See [`demo/README.md`](../../demo/README.md) for the full topic list.
