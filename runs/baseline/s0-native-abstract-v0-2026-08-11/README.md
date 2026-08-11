# S0 Native Reference Web/Telemetry Capture

Run ID: `s0-native-abstract-v0-2026-08-11T180953Z`
Evidence level: provisional software/reference baseline
Backend: `native` / `native-abstract-v0`
Observation boundary: default `sensor_equivalent`, expected fail-closed

## Outcome

The current default web runtime started on an isolated capture port and served
all required surfaces:

- finite `state.json` from `/api/state`;
- valid 256 x 256 grayscale `canvas.png`;
- 56 telemetry samples with 96 named columns in `telemetry.csv`;
- runtime/version and XML-derived robot-model JSON; and
- the served HTML, JavaScript, and CSS viewer snapshot.

The default boundary correctly disabled painting-policy inference because the
native backend still lacks a sensor-conditioned body posterior and conforming
`PlantBackend` adapter. It did not fall back to exact simulator material state.
The scripted native arm continued moving and emitting telemetry, which is the
intended reproducible S0 plant/web reference. Canvas coverage stayed zero
because no painting policy or brush-load action was permitted.

`server-stderr.txt` is empty. `server-stdout.txt` records the expected boundary
message and capture URL.

## Reproduction

```powershell
python -m active_painter.web_server --host 127.0.0.1 --port 8026 --canvas-size 256 --device cpu --driver-bootstrap-transitions 0 --driver-bootstrap-train-steps 0 --telemetry-max-samples 512 --telemetry-sample-hz 30
```

After readiness, wait two seconds and capture `/api/state`,
`/api/canvas.png`, `/api/telemetry.csv`, `/api/version`,
`/api/robot-model`, `/`, `/main.js`, and `/style.css`.

Port 8026 is a capture convenience, not a plant or interface contract.

## Limitations

This bundle is not a painting-quality result, sensor-equivalent hardware run,
calibrated plant record, or evidence that the active-inference loop ran. The
complete classification is
`docs/SIMULATOR_SHORTCUT_CLASSIFICATION_2026-08-11.md`. The missing native
sensor adapter is tracked by T-109.
