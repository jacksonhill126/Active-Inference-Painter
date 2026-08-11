# Baseline Artifact Bundle Convention

Date: 2026-08-11
Closes definition portion of: T-107

## Location and identity

Baseline captures live under:

```text
runs/baseline/<run-id>/
```

The directory name is an immutable run identifier, never `latest`. A repeated
capture receives a new run ID rather than overwriting an accepted bundle. The
first conforming bundle is
`runs/baseline/s0-native-abstract-v0-2026-08-11/`.

## Required files

| File | Purpose |
| --- | --- |
| `README.md` | Human-readable scope, outcome, limitations, and reproduction command |
| `version-manifest.json` | Exact source, plant/backend/model, config, and artifact identities |
| `experiment-manifest.json` | Run purpose, boundary mode, seeds, timing, and termination |
| `resolved-config.json` | Fully resolved capture/runtime choices relevant to behavior |
| `test-summary.md` | Baseline and capture-focused validation results, including warnings/timeouts |
| `failure-log.jsonl` | Empty when no runtime failures occur; otherwise append-only structured failures |
| `state.json` | One finite `/api/state` response |
| `canvas.png` | One `/api/canvas.png` response with dimensions recorded in the manifest |
| `telemetry.csv` | Short `/api/telemetry.csv` capture with stable header |
| `version.json` | `/api/version` response |
| `robot-model.json` | `/api/robot-model` response used by the frontend |
| `frontend.html`, `main.js`, `style.css` | Served frontend snapshot sufficient to identify viewer behavior |
| `server-stdout.txt`, `server-stderr.txt` | Startup/boundary messages and runtime errors |

Additional traces, camera frames, checkpoints, and screenshots may be included,
but their absence must not be disguised by an empty path. Every evidence file
used in a claim receives a SHA-256 in `version-manifest.json`.

## Minimum manifest content

Every bundle records:

- run ID and UTC capture interval;
- package/viewer version, full Git commit, dirty flag, and source fingerprint;
- native and MuJoCo plant labels (`*-none` where not used);
- agent/checkpoint/calibration/hardware labels and exact absence where relevant;
- backend, observation-access mode, planner-state kind, compute device, canvas
  dimensions, bootstrap settings, and random seeds;
- whether painting-policy inference was enabled or fail-closed;
- row/column counts for telemetry and canvas dimensions/mode;
- test commands, exit status, passes, warnings, duration, and any wrapper
  timeout distinct from pytest results; and
- links to the consolidated shortcut/limitation classification.

## Interpretation rule

A baseline bundle proves reproducibility of the named software/reference
behavior. It does not upgrade an abstract plant, simulated sensor, or oracle
diagnostic into hardware evidence. The `evidence_level` and
`observation_access_mode` fields are mandatory specifically to prevent that
upgrade by implication.
