# Action-Conditioned Material Transition And Camera Loop

Date: 2026-08-06

Status: implemented and locally verified

Scope: temporal ordering of executed actions, material/brush transition priors,
camera capture, camera likelihood assimilation, and the next planning gate

## Result

The sensor path now has an explicit repeated update clock:

```text
completed executed action
    -> material transition prior q-(s[t+1])
    -> compact brush depletion prior q-(b[t+1])
    -> post-physics camera capture boundary
    -> delivery through declared camera latency
    -> reject captures older than the boundary
    -> registered camera likelihood
    -> material posterior q(s[t+1]) and camera VFE
    -> one sensor-derived replay transition
    -> release the next-planning gate
```

The material update is `action-conditioned-camera-update-v0`. Its transition
factor is `spatial-action-transition-prior-v0`.

This closes the scheduling gap identified in the previous body, material, and
brush forecast reports. It does not make the complete painting loop
sensor-equivalent. The default remains fail-closed because substrate/model
forecast state, contact/compliance inference, persistent brush history, and the
native plant sensor adapter remain unresolved.

Later update, 2026-08-06: the opt-in
`provisional-sensor-simulation-v0` profile now exercises this clock repeatedly
using independent fixed context priors rather than live process copies. See
`docs/PROVISIONAL_SENSOR_SIMULATION_2026-08-06.md`. The default and the
calibration claims above are unchanged.

## Why Timing Is Part Of The Generative Model

A camera frame has both a capture time and a delivery time. Delivery after a
stroke does not prove that the image was captured after the stroke. A delayed
pre-action frame could otherwise be fused with the post-action transition
prior and incorrectly treated as evidence about the mark.

For each completed action the driver therefore stores one
`PendingActionCameraUpdate` containing:

- the executed action and motor realization;
- the previous spatial posterior;
- the predicted material prior;
- the compact brush transition revisions;
- an update identifier;
- the earliest admissible camera capture time; and
- the count of rejected older frames.

The MuJoCo runtime registers the earliest admissible time only after the
physics step that follows action completion. A frame is eligible only when

```text
frame.capture_time_s >= capture_not_before_s
```

within the declared floating-point tolerance. This uses observation metadata,
not simulator material, segmentation, contact, or visibility truth.

## Material Transition Prior

`SpatialVariationalStateEstimator.predict` now exposes the transition part that
was previously embedded inside the oracle observation-fusion method. In the
current mean-evaluated diagonal approximation,

```text
q-(s[t+1]) = Integral p_theta(s[t+1] | s[t], a[t], m[t])
                      q(s[t]) ds[t]
```

is represented by learned predictive means plus propagated posterior,
aleatoric, epistemic, and outside-support identity variance. Local-patch mode
evaluates the learned model only on action support and retains the declared
identity transition elsewhere.

The prior is versioned as `spatial-action-transition-prior-v0` and marked
`provisional_simulation_trained_not_hardware_calibrated`.

Prediction is not evidence. Calling `predict` does not create or overwrite a
VFE record. This prevents a model forecast from being counted as if it were a
sensor observation.

## Camera Posterior And VFE

After a causally eligible exposure arrives, `CameraSpatialLikelihood` performs
the existing nonlinear grayscale update. Its state complexity, occlusion
complexity, and expected negative log likelihood remain separately logged.
The completed posterior provenance includes:

- `action-conditioned-camera-update-v0`;
- `spatial-action-transition-prior-v0`; and
- the camera likelihood/model identifier.

The first eligible registered exposure completes the pending update. Later
continuous exposures may further refine the current posterior, but they cannot
add a second replay transition for the same action.

## Runtime Scheduling

The MuJoCo web runtime now polls automatically while either of these exists:

- a pending action-conditioned material update; or
- an outstanding foveal request.

Polling is bounded at 120 Hz. Camera-specific capture rates, dropout, and
latencies remain owned by `CameraObservationProcess`; the runtime does not
bypass them with an immediate render. A first poll may therefore create a
capture while returning no delivered frame. Later polls deliver it when its
declared latency has elapsed.

The camera process may return a mixture of older and eligible frames. The
driver filters by capture time before passing a new bundle to the likelihood.
An eligible native or edge-only product does not complete the update because
those products are not material likelihood factors.

## Brush Clock

The selected brush's `stroke_transition` now advances on the same recorded
executed-action event as the material transition prior. This keeps brush and
material revisions temporally aligned.

The brush posterior is not yet camera-corrected. `infer_load_from_mark`
requires a local deposition observation and variance, but the current camera
path has not declared a statistic that separates new mark evidence from:

- the material transition prediction;
- old paint already under the stroke;
- camera-unobservable white-on-white thickness;
- pickup and redeposition of held paint; and
- occlusion or saturated superficial tone.

Feeding the predicted material increment back into the brush likelihood would
double-count the model as evidence. The diagnostics therefore explicitly
report `brushCameraLikelihoodApplied=false`.

## Learning Boundary

When the eligible camera update completes, one transition is added to replay:

```text
(previous posterior, executed action and motor realization,
 camera-updated posterior)
```

No exact canvas field supplies the target. Additional camera exposures between
actions may refine the belief but do not create duplicate transition samples.

The existing `trained_transitions` counter records accepted transition pairs;
gradient training still occurs under the existing bounded training schedule.
This change does not add a reward, aesthetic score, or new EFE term.

## Planning Gate

Both global and local passage planning entry points refuse to start while a
`PendingActionCameraUpdate` exists. The driver phase is
`awaiting_post_action_camera` during that interval.

This establishes the required causal order for a sensor-only policy loop. The
current default still refuses painting-policy inference entirely. The later
bounded provisional profile opts into explicit independent model priors to run
the integration loop; it does not authorize fallback through the remaining
oracle dependencies.

## Verified Claims

Tests verify that:

- the transition prior increments revision and propagates uncertainty;
- prediction does not create or overwrite VFE;
- brush depletion advances once on the executed-action clock;
- no camera frame is accepted before a capture boundary exists;
- a frame captured before the boundary is rejected even if delivered later;
- the first eligible registered exposure completes the posterior and camera
  VFE decomposition;
- exactly one replay transition is added per executed action;
- later continuous exposures do not duplicate that transition;
- sensor-path execution completion does not call the process-material state
  constructor; and
- the MuJoCo runtime automatically polls through capture latency and completes
  the update.

Broader regression results are recorded in `docs/PROGRESS.md`.

## Remaining Work

The next narrow inference task is the local camera-derived brush-deposition
likelihood. It must declare its spatial support, physical units, uncertainty,
and observability gate so white-on-white or occluded regions do not become
false high-precision brush evidence.

After that, the higher-risk embodiment boundaries remain:

- infer a compliance/deformation latent before mapping contact beliefs into
  brush physics;
- replace copied substrate grain and model parameters with declared beliefs;
- provide the native plant with a `PlantBackend` sensor adapter;
- sample MuJoCo parameter uncertainty; and
- replace exact-contact controller and reliability inputs with calibrated
  sensor-conditioned estimates where required.
