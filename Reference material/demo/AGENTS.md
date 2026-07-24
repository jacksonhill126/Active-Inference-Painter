# `demo/` — application demos

Cross-cutting **application demos** that compose existing library methods into
thin orchestrators with domain-themed plots. Unlike chapter scripts, demos are
not tied to a single book example number; unlike extras topics, they foreground
real-world metaphors (saccades, balance, aerial navigation).

## Hard rules

1. Import only from `active_inference` and the Python standard library.
2. Keep each script at or below ~120 lines; reusable logic lives in
   `src/active_inference/demo_workflows.py`.
3. Every non-interactive script accepts `--save` and writes PNG +
   NPZ+JSON sidecars via `save_demo_data`.
4. Stochastic demos also accept `--seed` and pass `rng` explicitly in library code.

## Layout

| Folder | Demo |
| --- | --- |
| [`eye_saccades/`](eye_saccades/) | Discrete EFE fixation planning on a retinotopic grid |
| [`bicycle/`](bicycle/) | Multivariate active inference balance metaphor + fault compensation |
| [`drone_flight/`](drone_flight/) | Waypoint lattice planning, kinematic path, LGS position fusion |

## Running

```bash
python demo/eye_saccades/visualize_eye_saccades.py --save
python demo/bicycle/visualize_bicycle.py --save
python demo/drone_flight/visualize_drone_flight.py --save
```

Or via the menu / web UI (`./run.sh` / `./run.sh --web`).

## Outputs

- Figures: `output/figures/demo/<slug>/`
- Raw data: `output/data/demo/<slug>/`

## Verification

```bash
uv run pytest tests/demo -v
uv run python scripts/validate_orchestrator_contracts.py
```
