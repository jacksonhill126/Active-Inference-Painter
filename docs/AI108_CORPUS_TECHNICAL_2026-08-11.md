# AI-108 leakage-resistant corpus acceptance record

Date: 2026-08-11 (America/New_York)

## Decision

AI-108 is accepted for the uncalibrated simulation-only integration baseline.
The accepted manifest contains 16 whole painting trajectories and 256
camera-posterior transitions. It is not hardware-calibrated evidence, a
sensor-equivalent hardware corpus, a painting-quality result, or a terminal
composition corpus.

Canonical artifacts:

- manifest: `runs/corpus-ai108-combined-20260811/split_manifest.json`;
- audit: `runs/corpus-ai108-combined-20260811/corpus_audit.json`;
- bounded-fixed source: `runs/corpus-ai108-fixed-live-20260811`;
- full-roll source: `runs/corpus-ai108-full-roll-live-20260811`.

Every manifest shard has a SHA-256 digest. The manifest root is the common
`runs` directory, so the two collection profiles can be consumed as one
trajectory-isolated dataset without copying or rewriting evidence.

## Evidence boundary and resolutions

Every accepted shard declares:

- corpus schema `trajectory-posterior-corpus-v2`;
- process canvas size 256;
- reduced inference-driver canvas size 64;
- spatial posterior grid 16 by 16;
- plant `mujoco-robstride-electromechanical-v4`;
- observation access mode `sensor_equivalent`;
- camera likelihood `provisional_simulation_only_not_hardware_calibrated`;
- process truth not used as a training input;
- compact inferred pre-stroke brush context available.

The 64-pixel inference/acquisition configuration is a declared throughput
approximation in this provisional simulator path. It must not be described as
a sensor-equivalent camera realization. Process resolution, inference
resolution, and posterior-grid resolution are now separate provenance fields.

The stored transition fields are the pre/post camera-derived material
posterior means and log variances, selected eight-parameter mark, selected
conditional motor realization, and compact inferred pre-stroke brush belief.
Exact process canvas state, exact contact state, exact held paint, and exact
bristle microstructure are not training inputs.

## Collection profiles and throughput

Both batches used eight independent spawned runtimes, one trajectory per
runtime, 16 transitions per trajectory, random black/white stroke-tone prior,
and two Torch CPU threads per worker.

| Profile | Conditional motor support | Trajectories | Transitions | Wall time | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| `bounded_fixed_roll` | neutral IK, fixed +24 deg, fixed -24 deg | 8 | 128 | 260.4 s | 0.492 transitions/s |
| `research_full_roll` | neutral IK, fixed +/-24 deg, swept +/-32 deg | 8 | 128 | 293.2 s | 0.437 transitions/s |
| Sequential total | union of both profiles | 16 | 256 | 553.6 s | 0.462 transitions/s |

All 16 jobs completed and no worker failed. Individual bounded-fixed runtimes
took 207-251 seconds; full-roll runtimes took 243-281 seconds. This supports
the earlier diagnosis that recycling the complete runtime after one trajectory
avoids the progressive cross-painting slowdown. Five-way full-roll forecasting
cost about 13 percent more wall time than three-way fixed-roll forecasting in
this bounded comparison. That ratio is a simulation-throughput measurement,
not a general scaling law.

## Split and leakage evidence

The deterministic greedy multilabel strategy
`deterministic-greedy-multilabel-transition-balance-v1` assigned complete
trajectories before local-patch extraction:

| Split | Trajectories | Transitions |
| --- | ---: | ---: |
| Train | 10 | 160 |
| Validation | 3 | 48 |
| Test | 3 | 48 |

Trajectory IDs are unique across splits. No neighboring patches or successive
marks from one trajectory can enter different splits. Condition labels are
derived only from the stored pre-action posterior, selected action, motor
realization, brush-context availability, and trajectory metadata. They are
evaluation/stratification labels, not policy rewards, preferences, likelihood
terms, or aesthetic heuristics.

## Measured condition coverage

All required conditions occur in train, validation, and test separately. The
overall transition counts are:

- tone: 144 black, 112 white;
- surface/overlap: 113 dry blank/fresh, 143 dry existing/overlap;
- canvas edge: 195 edge, 61 interior;
- width: 94 narrow, 95 medium, 67 broad;
- length: 63 short, 127 medium, 66 long;
- curvature: 33 negative strong, 46 negative gentle, 81 straight, 40 positive
  gentle, 56 positive strong;
- direction: 72 vertical, 83 horizontal, 101 diagonal;
- reach: 30 center, 147 middle, 79 outer;
- motor realization: 39 neutral Cartesian IK, 94 fixed positive roll, 57
  fixed negative roll, 44 positive roll sweep, 22 negative roll sweep;
- brush context: 256 available, 0 unavailable.

The small held-out sweep counts are visible rather than hidden: test has two
positive-sweep and one negative-sweep transitions; validation has ten and
eight. Future larger learning-curve corpora should increase those counts, but
the acceptance condition that each held-out split actually contains every
declared stratum is met.

## Wetness and stopping boundaries

No accepted transition is labelled wet-over-wet. This is not a random coverage
failure: the current camera likelihood deliberately does not identify bulk
wetness, and the conservative warm-up transition prior does not invent a
wetness mean. Dry blank versus dry existing/overlap is therefore the legitimate
sensor-corpus material stratification. Wet-over-wet must remain a named
observability gap until a defensible likelihood or calibrated sensor cue is
added; exact simulator wetness must not be substituted.

All 16 trajectories end by fixed-horizon truncation. There are no genuine
policy-selected stops. Stop is not part of AI-108's transition-corpus
acceptance, but this means the corpus is not usable for terminal composition
training. Composition remains blocked until genuine stopped canvases exist;
terminal labels must not be manufactured.

## Consumer verification

The combined multi-root manifest initially exposed a real compatibility bug:
the fixed and full-roll collection profiles had different motor-candidate
configuration fields. Pooled trainers now treat worker seeds and declared
collection-policy support as allowed variations, construct one canonical
five-kind motor-conditioning vocabulary, and still require all model-relevant
configuration fields to match.

A two-gradient-step CPU cVAE consumer smoke loaded the accepted manifest,
materialized 160/48/48 train/validation/test patches, trained only on the train
split, evaluated both held-out splits, and wrote:

- `runs/corpus-ai108-combined-20260811/cvae_consumer_smoke.pt`;
- `runs/corpus-ai108-combined-20260811/cvae_consumer_smoke.report.json`.

This is a plumbing check, not a learning or calibration result. Two gradient
steps are insufficient for interpreting the held-out NLL or condition-ablation
signs.

A separate one-gradient-step conventional CNN consumer smoke materialized the
same 160/48/48 split, updated only the training replay, evaluated validation
and test, correctly found zero terminal composition canvases, and wrote
`cnn_consumer_smoke.pt` plus `cnn_consumer_smoke.report.json` beside the cVAE
artifacts. Its metrics are likewise plumbing evidence, not a learning result.

## Earlier corpora retained but not accepted

`runs/corpus-full-20260810-1802` remains useful legacy pretraining evidence:
22 v1 trajectories and 1,408 transitions with broad condition coverage. It
lacks compact brush context, contains no policy stops, and does not separately
attest process versus inference canvas resolution.

`runs/corpus-ai108-v2-live-20260811` contains 12 v2 trajectories and 384
transitions with complete brush context and broad fixed-roll/action coverage.
It predates the separate process-resolution provenance field, so it is retained
for explicitly provenance-qualified pretraining rather than used to satisfy
AI-108 acceptance. Neither corpus was rewritten after collection.

## Verification

- 12 focused trajectory-corpus and conditional-cVAE tests pass.
- Edited collection, audit, training, and runtime modules pass `py_compile`.
- All accepted shards independently load and their manifest hashes were
  generated after collection.
- The audit reports all eight AI-108 checks true.
- The downstream composition-readiness check remains false, as intended.

## Next work

1. Run AI-107 held-out calibration on this accepted corpus.
2. Expand the same provenance-complete profiles at several data sizes for
   AI-109 learning curves, especially held-out roll-sweep counts.
3. Diagnose the absence of policy-selected stops without weakening or
   manufacturing the terminal preference.
4. Design a defensible wetness observation/likelihood before requiring
   wet-over-wet sensor strata.
