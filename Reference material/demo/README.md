# Application demos

Pedagogical compositions of existing Active Inference algorithms with
application-themed labeling. v1 intentionally avoids bespoke physics plants;
see each demo README for the underlying book chapters and APIs.

| Demo | Script | Core methods |
| --- | --- | --- |
| Eye saccades | [`eye_saccades/visualize_eye_saccades.py`](eye_saccades/visualize_eye_saccades.py) | POMDP grid planning (Ch.9) |
| Riding a bicycle | [`bicycle/visualize_bicycle.py`](bicycle/visualize_bicycle.py) | Multivariate AIF (Ch.7) + fault control (Ch.13) |
| Drone flight | [`drone_flight/visualize_drone_flight.py`](drone_flight/visualize_drone_flight.py) | POMDP + navigation stub + LGS (Ch.3/9/13) |

```bash
uv run python demo/eye_saccades/visualize_eye_saccades.py --save
```

Agent guide: [`AGENTS.md`](AGENTS.md).
