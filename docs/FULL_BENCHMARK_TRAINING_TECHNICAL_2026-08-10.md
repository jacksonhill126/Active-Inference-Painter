# Parallel benchmark and shared-pretraining run — technical record

Date: 2026-08-10 (America/New_York)

## Claim boundary

This is an **uncalibrated simulation-only integration baseline**. It is not a
painting-quality result, a sensor-equivalent result, a hardware-calibrated
result, or evidence sufficient for embodiment claims. Collection used the
selectable `mujoco-robstride-electromechanical-v4` backend and the provisional
camera-derived spatial-posterior path. Exact live process canvas material was
not stored as a training input.

## Scaling benchmark

Artifact:
`runs/benchmarks/parallel-20260810-170856/benchmark.json`

Configuration: 3 trajectories per worker, 16 transitions per trajectory,
256-pixel canvas, 16-by-16 spatial posterior, random black/white tone prior,
and two Torch CPU threads per worker. Each concurrency stage deliberately
reused the worker seeds, so the stages are throughput comparisons rather than
independent datasets that may be pooled.

| Workers | Trajectories | Transitions | Wall time (s) | Transitions/s | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 48 | 850.421 | 0.05644 | 1.000 | 1.000 |
| 4 | 12 | 192 | 723.442 | 0.26540 | 4.702 | 1.176 |
| 6 | 18 | 288 | 724.375 | 0.39758 | 7.044 | 1.174 |
| 8 | 24 | 384 | 810.204 | 0.47395 | 8.397 | 1.050 |

All 57 benchmark shards completed without a MuJoCo/OpenGL worker crash. The
slightly superlinear figures should not be treated as a stable scaling law;
startup, cache, and run-order effects are large at this sample size. Eight
workers had the highest measured raw throughput and used about 6.25 GB across
the collector processes.

## Fresh-seed corpus run

Directory: `runs/corpus-full-20260810-1802`

Intended configuration: 48 trajectories, 8 workers, at most 64 transitions
per trajectory, seed 30260810, random tone prior, and 256/16 canvas/spatial
resolution. The run was time-boxed to preserve a training and verification
window before an expected computer shutdown.

Actual retained corpus:

- 22 unique, independently loadable whole-trajectory shards;
- 1,408 transitions total;
- 19 train trajectories / 1,216 train transitions;
- 2 validation trajectories / 128 validation transitions;
- 1 test trajectory / 64 test transitions;
- 22 `fixed_horizon_truncation` terminations;
- 0 `policy_selected_stop` terminations;
- recovered manifest:
  `runs/corpus-full-20260810-1802/split_manifest_recovered.json`.

Correction recorded 2026-08-11: the retained v1 shard `config.canvas_size`
field is the 64-pixel inference-driver configuration, not durable evidence of
the process-canvas resolution. The 2026-08-10 command record says the intended
process canvas was 256, but v1 provenance did not store process and inference
resolution separately. This corpus must therefore be described as lacking
independent process-resolution provenance rather than asserted to prove either
64 or 256 process resolution. The v2 collector now records process canvas,
inference/acquisition approximation, and posterior grid separately.

The split was performed before local-patch extraction. No trajectory ID is in
more than one split.

### Recovery incident

During the first wave, worker 002 completed a valid compressed archive but
stalled at the temporary-file finalization boundary. The
`worker-002-...-trajectory-000000.tmp.npz` archive was loaded and all required
arrays, its 64-transition count, and its fixed-horizon metadata were verified.
It was copied non-destructively to the intended final filename; the forensic
temporary file was retained. The final copy passes
`load_trajectory_shard` and is the only worker-002 item in the recovered
manifest.

Seven other workers continued. Their first three trajectories completed, but
the fourth wave had not committed after approximately 42 minutes. Collection
was stopped at 19:45 to protect the training reserve. Seven partial in-memory
trajectories were intentionally discarded. The orphaned Windows worker
processes were terminated by exact PID, and the OS subsequently released all
of them.

Repeated-painting wall time grew materially even though process memory stayed
near 0.8 GB per active worker and CPU remained balanced. This indicates that
cross-painting runtime state makes later planning cycles progressively more
expensive. A future collector should recycle each runtime after one trajectory
or a small bounded trajectory batch and should record per-action planning,
forecast, camera, online-training, and diagnostic timings.

`discover_trajectory_shards` was corrected to exclude `.tmp.npz` archives so
an interrupted atomic write cannot enter a manifest.

## Centralized shared-parameter training

Artifacts:

- checkpoint:
  `runs/checkpoints/shared-pretraining-20260810-1408.pt`;
- report:
  `runs/checkpoints/shared-pretraining-20260810-1408.report.json`;
- checkpoint SHA-256:
  `C1AE1E9A768E18E62D23EF16302F887654F992A96F3564CD95748215A9958508`.

Configuration: CUDA on NVIDIA RTX A1000 Laptop GPU, batch size 64, 3,000
local-dynamics gradient steps, and 2,700 requested composition steps. GPU
utilization sampled at 77% with about 905 MiB allocated. Dynamics training
took 87.270 seconds; the entire command took about 97 seconds.

The trainer initially rejected the corpus because it compared complete
`PainterConfig` payloads and worker-specific `brush_seed` and
`canvas_grain_seed` differed by design. The compatibility guard now permits
only those two process-randomness fields to vary. All model-relevant plant,
likelihood, hierarchy, policy, camera, and learning fields must still match.
The checkpoint and report record this allowed variation.

Held-out conditional Gaussian negative log density, averaged per valid
material cell-channel and ensemble member, omits the constant
`0.5*log(2*pi)`. Negative values are therefore valid; lower is better.

| Split | Before | After | Reduction |
|---|---:|---:|---:|
| Validation | -1.17680 | -4.76471 | 3.58792 |
| Test | -1.54147 | -4.89061 | 3.34914 |

The last dynamics training-budget mean loss was -4.73362. Validation and test
were never used for gradient updates. The checkpoint independently reloads,
contains 1,216 trained transitions, and records `shared_pretraining`
provenance.

### Composition outcome

Composition training did **not** run. Although 2,700 steps were requested,
the corpus contained zero legitimately terminal canvases. The trainer returned
`terminal_training_canvases: 0` and `composition_loss_last: null`. This is the
intended fail-closed behavior: a 64-step truncation is not evidence that the
painting satisfied the terminal composition preference.

## Verification

- Full benchmark completed and wrote its JSON report.
- All 22 retained fresh-seed shards load; IDs are unique; transition total is
  1,408; termination total is 22 fixed-horizon / 0 policy-stop.
- Recovered split is 19/2/1 and whole-trajectory isolated.
- Real 3,000-step CUDA training completed and produced the held-out metrics
  above.
- Checkpoint independently loaded and its SHA-256 was recorded.
- Worker-randomness compatibility unit test passed.
- Real corpus verified that shard discovery ignores the retained temp file.
- Edited modules passed `py_compile`.
- The pytest test that uses `tmp_path` could not run after collection because
  Windows denied access to pytest-created temp directories, including a
  workspace-local `--basetemp`. Its worker-randomness helper assertion passed,
  and the actual recovered-manifest training run exercised the full trainer
  path successfully.

## Recommended next engineering work

1. Make collector runtime lifetime bounded—prefer one complete trajectory per
   spawned runtime initially—and finalize a manifest from every valid shard
   even if another worker fails.
2. Add explicit phase timings and archive-finalization diagnostics. Investigate
   the worker-002 finalization stall separately from planning slowdown.
3. Diagnose why no `stop` action was selected in 1,408 transitions. Do not
   manufacture terminal labels or weaken the terminal preference merely to
   obtain composition examples.
4. Collect genuine policy-stop canvases with the corrected bounded-lifetime
   collector, then train and evaluate the composition hierarchy on
   trajectory-isolated terminal canvases.
5. Run the saved checkpoint in a fresh sensor-simulation runtime and compare
   policy-cycle timing, camera VFE, EFE decomposition, mark diversity, and
   terminal behavior against an untrained seed.
