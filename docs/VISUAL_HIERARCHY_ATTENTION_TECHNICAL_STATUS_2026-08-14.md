# Visual Hierarchy And Attention - Technical Status

Date: 2026-08-14

Status: owner-confirmed scope clarification; implementation remains pre-admission

## Executive decision

The center of the project is a hierarchical active-inference painting model
that can represent and predict:

- multiscale visual tone and oriented boundary evidence;
- continuous light and dark masses with uncertain boundaries;
- spatial relations such as adjacency, overlap, alignment, relative angle,
  scale, spacing, containment, and transformation;
- recurring visual motifs and slower structure across marks and passages;
- action-conditioned visual consequences of candidate marks.

Active attention remains an important part of the research program, but the
cognitive attention variable is **not defined as a rectangular crop being
steered around the image**. An attention trajectory is a temporal sequence of
spatial information-access and precision allocations. One requested crop is a
sensor-level realization of that allocation, not the painting-level primitive.

This distinction prevents the camera interface from dictating the internal
cognitive architecture and keeps foveation from displacing the visual
hierarchy as the main modeling objective.

## Current standing

| Component | Current evidence | Standing |
| --- | --- | --- |
| Detailed paint process | Thickness, wetness, pigment, brush history, grain, contact, pickup, blending, and anisotropic round-brush behavior exist in simulation | Retain as the generative process; do not copy into a canvas-wide agent latent |
| Coarse material posterior | Six-channel, usually 16x16 `SpatialCanvasState` is integrated with the provisional camera likelihood and runtime | Compatibility and simulation-integration baseline only; not the selected visual hierarchy |
| Material transition experiments | AI-107/AI-109 measured CNN, cVAE, ensemble, and identity/consequence mixture families over coarse material posterior patches | Useful negative and diagnostic evidence for that target; no composition or visual-hierarchy claim |
| Registered visual corpus | `registered-visual-trajectory-corpus-v1` and the causal pre/post collection path are implemented | Simulation-camera evidence; scaled collection is incomplete and not hardware calibrated |
| Local visual mark model | `action-conditioned-visual-mark-cvae-v0` completed a 3-trajectory, 10-example end-to-end smoke | Shadow-only; the smoke proves the path runs, not that it has learned a useful visual consequence model |
| Scaled visual run | 18 complete trajectories, 144 executed transitions, and 288 paired camera examples survived the 2026-08-12 Windows Update interruption | Resumable evidence retained locally; the planned 96-trajectory collection and 80-epoch run have not completed |
| Fixed-camera foveal interface | Explicit native-derived `FoveaRequest` products, global products, provenance, timing, and non-oracle selection boundaries exist | Useful sensor realization and future experimental seam |
| Active attention policy | No learned or inferred online attention-allocation policy exists | Deferred until the visual hierarchy and partial-observation likelihood can support a meaningful test |
| Target hierarchy | `z_visual`, `z_mass`, `z_relation`, and slower motif/composition states are specified semantically | Not implemented or admitted; this is the critical modeling gap |

The current visual cVAE has no policy influence. No current output is evidence
that the agent represents masses, spatial harmony, motifs, or composition.

## Core hierarchy

The minimum target factorization is:

```text
z_motif / z_composition
    slower recurring relational structure across marks and passages
                 <->
z_relation
    adjacency, overlap, alignment, angle, scale, spacing, containment,
    repetition, and transformation among visual regions
                 <->
z_mass
    continuous light/dark regions and uncertain boundaries
                 <->
z_visual
    absolute tone plus multiscale oriented boundary structure
                 <->
registered camera observations
```

The arrows are bidirectional inference messages. Local evidence updates larger
hypotheses, and larger hypotheses predict what local observations should look
like. A slow level is retained only when it improves held-out prediction,
calibration, temporal prediction, spatial intervention behavior, or later
policy inference. It must not be a fast downsampled copy of the current image.

"Harmonic relationship" is not a scalar harmony or beauty score. It refers to
learned probabilistic regularities among relative angles, intervals, scales,
repetitions, transformations, and mass relations that allow separated parts
of a painting to predict one another.

A visual motif is likewise not a semantic label or hand-authored template. It
is a recurring relational configuration whose latent representation improves
prediction across locations, marks, or passages.

## Attention state and attention trajectory

The cognitive attention action should be defined above the camera crop. A
minimal abstract state can contain:

```text
attention allocation at time t
    camera support              left, right, or both where justified
    spatial allocation          one region, several sparse regions, or a field
    scale allocation            which spatial-frequency detail is requested
    temporal allocation         when and for how long evidence is acquired
    precision allocation        expected reliability assigned to obtained evidence
```

Equivalently, the spatial part may be represented as a budgeted field
`rho_t(u, v, scale, camera)`. Its history across time is the attention
trajectory. That trajectory may jump, revisit a region, split across sparse
tiles, or remain broad. It need not imitate a smooth eye movement or a crop
center moving continuously through the image.

The distinction between two attention mechanisms must remain explicit:

1. **Sensory access allocation** changes which native-resolution evidence is
   made available to online inference.
2. **Inferential precision allocation** changes how strongly available
   likelihood evidence influences the posterior.

Precision weighting cannot substitute for restricted access if the full
high-resolution frame has already passed through the encoder. Otherwise the
unsampled information has leaked into the agent before attention is applied.

The current `FoveaRequest(camera, center, span, time)` is an acceptable first
realization because it enforces a clear access boundary. Future realizations
may use sparse tiles or another budgeted readout, provided they preserve the
same non-leakage contract. The cognition should select an allocation; the
sensor process should decide how that allocation is physically realized.

## Why attention remains relevant

Partial observation can make a shared painting-level latent predictively
necessary. If high-resolution region B is hidden, the model must predict it
from peripheral evidence, previous observations, and the inferred relations
among regions. Observing region A can then change the predicted distribution
for B.

This is a hypothesis, not an automatic consequence of foveation. A weak model
may instead memorize independent crops, repeatedly inspect an easy location,
or chase aleatoric noise. Therefore attention is introduced only after the
hierarchy can represent cross-region dependence under controlled full-view or
passively sampled conditions.

The attention policy must be derived from the declared generative model and
EFE. It should prefer an observation when plausible hidden-state hypotheses
predict distinguishable outcomes there. Raw edge density, novelty, visual
busyness, pixel error, or a hand-written messiness score are not admissible
attention objectives.

## Narrow critical path

### 1. Close the local visual baseline without expanding its scope

- Resume or deliberately rescope the registered visual collection.
- Train the existing visual cVAE and deterministic/no-interaction baselines.
- Measure normalized held-out likelihood, calibration, condition use, tone and
  oriented-boundary fidelity, and recursive rollout stability.
- Do not spend repeated cycles scaling the cVAE if its declared target or
  likelihood family is the limiting factor.

### 2. Establish an orientation-preserving full-canvas visual likelihood

- Compare the pixel-Beta baseline with a declared multiscale oriented
  coefficient likelihood and a modest learned encoder.
- Preserve absolute low-frequency tone separately from contrast-normalized
  boundary coefficients.
- Avoid double-counting the same image as independent pixel and edge factors.

### 3. Establish masses and cross-region predictive necessity

- Add a continuous mass/surface latent above local visual evidence.
- Test whether observing or intervening on region A changes calibrated
  predictions for related region B.
- Compare with an independent-patch model and spatially scrambled controls.

### 4. Establish relations and motifs

- Add relational and slower motif states only after the mass layer earns
  predictive admission.
- Test translation, rotation, scale, masking, substitution, freeze, reset, and
  shuffle interventions.
- Require causal influence on later prediction, not merely decodable labels.

### 5. Connect the accepted hierarchy to mark and passage policies

- Use the local visual consequence model to predict how a candidate mark
  changes local evidence and the higher-level posterior.
- Map likelihoods, transition priors, preferences, precision beliefs, VFE, and
  EFE explicitly; do not introduce a coherence or harmony reward.

### 6. Introduce attention trajectories as a controlled M3 manipulation

- Compare active allocation, random allocation, fixed target inspection, and
  uniform observation under matched pixel, latency, and compute budgets.
- Test reducible ambiguity, pure noise, model misspecification, occlusion, and
  repeated fixation.
- Ask whether restricted sampling increases the predictive and causal use of
  cross-region, motif, and slow states.

## Admission evidence

Before a claim that the agent represents composition or motifs, require:

1. tone and oriented-boundary fidelity across relevant scales;
2. calibrated partial- and full-observation likelihoods;
3. continuous-mass predictive advantage over local-only baselines;
4. cross-region prediction that survives appropriate controls;
5. motif or slow-state causal interventions;
6. stable action-conditioned visual rollouts;
7. explainable effects on mark or passage policy posteriors;
8. matched active/random/uniform attention evidence before attributing an
   effect to active sampling.

Attractive paintings, crop traces, latent visualizations, or high decoder
reconstruction quality alone do not satisfy these gates.

## Scope controls

Defer the following until a preceding capability gate requires them:

- literal cortical columns, spikes, or log-polar retinal simulation;
- explicit polygonal objects or named aesthetic motifs;
- a separate hand-built contour reward or harmony objective;
- smooth eye-like crop trajectories as a cognitive assumption;
- complex joint inference over long attention/mark/passage sequences;
- large pretrained semantic vision models;
- further material-latent hierarchy work that does not improve visual
  prediction.

## Immediate handoff

The next agent should treat the visual hierarchy as the main modeling target.
It should preserve the current foveal camera contract but should not expand
active attention merely because the interface exists. Attention-trajectory
work begins when the visual likelihood and persistent hierarchy can state what
an observation is expected to resolve.

The owner-facing explanation is
`docs/VISUAL_HIERARCHY_ATTENTION_OWNER_BRIEF_2026-08-14.md`. The canonical
process/model boundary remains `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

## Publish validation

The 2026-08-14 pre-push validation collected 545 tests.

- The complete Windows/Python 3.14 run reached 100 percent with seven
  `tmp_path` setup errors and no assertion-failure marker. Pytest could not
  print its normal terminal summary because cleanup of its own workspace-local
  base temporary directory raised `PermissionError: [WinError 5]`; the
  inaccessible pytest-only directory was subsequently removed.
- The exact deterministic CI file set passed: 168 tests in 56.35 seconds with
  the two expected obsolete-summary warnings.
- A focused visual VAE, learning-curve, mixture, uncertainty, camera, driver,
  and documentation run passed 92 tests. One legacy driver timing test failed
  only because background planning exceeded its fixed 15-second wall-clock
  deadline.
- An isolated behavioral probe with a 90-second outer bound published a valid
  non-stop stroke after 16.88 seconds, retained all 72 bootstrapped
  transitions, reported 15.38 seconds of planning, and had no pending planner
  error. The timing gate was not relaxed or reported as a passing test.

This record distinguishes the one measured performance-gate miss and the
Windows temporary-directory setup fault from probabilistic or behavioral
assertion failures.
