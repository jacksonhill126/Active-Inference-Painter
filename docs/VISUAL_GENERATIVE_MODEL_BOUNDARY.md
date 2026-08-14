# Visual Generative-Model Boundary

Status: owner-confirmed architectural direction, 2026-08-12

This is the canonical semantic boundary between the paint **generative
process** and the painter's intended visual **generative model**. Future
agents must read it before changing perception, canvas state, local transition
learning, composition, counterfactual mark prediction, or corpus schemas.

## Decision

The physical process may retain detailed paint variables and dynamics. The
painting generative model is observation-first: it must first represent and
predict the visual image with useful fidelity, especially tone, edges,
continuous masses, and their relationships.

Do not copy every process variable into a persistent agent latent merely
because the simulator exposes it.

In particular:

- process wetness and thickness remain legitimate physical causes;
- persistent canvas-wide wetness is not part of the target agent belief;
- wetness is not a composition, mass, or painting-level latent;
- if local material ambiguity becomes predictively necessary, infer an
  ephemeral action-conditioned interaction latent from a fresh image of the
  proposed target patch;
- that latent may compress wetness-, thickness-, viscosity-, pickup-, and
  texture-like effects without assigning them named coordinates;
- visually unresolved material causes are marginalized into a calibrated
  distribution over visual mark consequences.

## Current implementation versus target architecture

The distinction is mandatory in reports:

| Layer | Current implementation | Architectural standing |
| --- | --- | --- |
| Paint process | Native or MuJoCo-coupled process with thickness, wetness, pigment, surface tone, brush history, grain, and contact dynamics | Retain and improve as a source of credible observations and action consequences |
| `SpatialCanvasState` | Six-channel, usually 16x16 camera-conditioned material posterior | Provisional compatibility and integration baseline; not the target visual or composition representation |
| Local CNN/cVAE/mixture experiments | Predict coarse material-posterior patches | Valid negative/positive evidence about those declared targets only |
| Shadow visual local model | `action-conditioned-visual-mark-cvae-v0` predicts a Beta distribution over post-mark visual crops from a fresh pre-mark crop, action, brush belief, camera, motor realization, and ephemeral latent | Initial implementation; no policy influence and admission gates not yet passed |
| Target slow hierarchy | Tone/edge-preserving appearance, continuous masses/boundaries, and uncertain spatial relations | Required direction; must earn admission through held-out prediction and calibration |

The current six-channel state may remain readable for historical checkpoints,
tests, and the simulation-only integration baseline. It must not silently
define the next architecture.

## Target factorization

Let `o_t` be a registered, rectified camera image of the canvas and `R(a_t)` a
target region containing a proposed mark's swept support plus context. A
minimal local visual latent-variable model is:

```math
p_\psi(z^I_{t,R}\mid o_{t,R},a_t,b_t)
```

```math
p_\theta(o_{t+1,R}\mid o_{t,R},a_t,b_t,z^I_{t,R}).
```

During training, a recognition density may use the observed consequence:

```math
q_\phi(z^I_{t,R}\mid o_{t,R},o_{t+1,R},a_t,b_t).
```

At planning time the future image is unavailable, so counterfactual outcomes
must integrate or sample from the predictive prior `p_psi`, not the training
recognition density.

Here:

- `b_t` is the compact brush belief available before the mark;
- `z^I` is an ephemeral interaction/affordance latent, not a canvas-wide
  material map;
- the decoder returns a normalized distribution over visual consequences;
- ambiguity that cannot be inferred from appearance remains outcome
  uncertainty;
- after execution, a causally later camera observation supplies VFE and the
  next planning state.

The slow visual hierarchy may contain:

```text
z_visual       multiscale tone and oriented boundary structure
z_mass         continuous light/dark masses with uncertain boundaries
z_relation     adjacency, overlap, alignment, containment, and scale relations
z_motif        recurring relational configurations across space and time
z_composition  slower predictive structure across marks and passages
```

These are probabilistic causes retained only when they improve held-out
prediction, calibration, temporal prediction, or intervention behavior. They
are not hand-authored aesthetic rewards.

## Hierarchy and attention priority

The hierarchy above is the central modeling target. Active attention is an
important later mechanism and experimental manipulation, but it must not
displace tone/edge, mass, relation, motif, and visual-transition capability
work.

An attention trajectory is a temporal sequence of spatial sensory-access and
precision allocations. It need not be a smooth path of one rectangular crop.
It may allocate one region, several sparse regions, different scales, or a
broader low-resolution field over time. The current canvas-UV `FoveaRequest`
is one sensor-level realization of such an allocation, not the painting-level
cognitive primitive.

Keep sensory access and inferential precision distinct. A precision map over a
full high-resolution encoding is not foveated access if unsampled details have
already entered the encoder or persistent belief. Any crop, tile, or other
readout realization must preserve the rule that unavailable native-resolution
detail cannot affect online inference.

Develop the hierarchy first under controlled full-view and passive partial-
observation conditions. Admit an active attention policy only when the model
can declare which hidden visual hypotheses a candidate observation is
expected to distinguish. Compare active, random, fixed-target, and uniform
observation under matched information and compute budgets before attributing
cross-region organization to attention.

## Tone and edges

Observed tone and edge structure are the immediate perceptual priorities.
The model must preserve at least:

- tone and low-frequency mass;
- boundary position, orientation, strength, and uncertainty;
- continuity, curvature, and fragmentation across scale;
- the relationship between a local boundary and larger masses.

An edge is derived from appearance. Do not sum a pixel likelihood and an
independent edge likelihood over the same image unless their joint density or
dependence approximation is explicit. Acceptable approaches include a single
image likelihood with an edge-preserving latent or a declared joint density
over non-overlapping multiscale coefficients.

A 16x16 nearest-neighbour rendering is not evidence that the model preserves
angles or contours. Low bandwidth is acceptable; arbitrary square boundaries
as the composition representation are not.

## Material uncertainty

Painters need not predict microscopic paint behavior exactly. The process may
produce bristle variation, pickup, blending, sticking, chatter, opacity
changes, and broken edges. The agent may represent much of this as calibrated
aleatoric variation in visual outcomes.

Material predictability is subordinate to painting inference. Do not make
exact material realization the primary objective of mark selection. A mark
candidate is evaluated through the declared likelihoods, preferences, VFE,
and EFE over its predicted visual and bodily consequences.

## Clarification of the owner's VAE proposal

The owner's original VAE proposal meant a pretrained, stochastic,
action-conditioned **visual mark-consequence model**. It did not mean a VAE
over explicit coarse thickness/wetness/pigment channels.

`conditional-local-material-transition-cvae-v0` is a real, measured shadow
experiment over coarse material-posterior patches. AI-107/AI-109 found that it
did not materially improve that target and had unstable recursive rollouts.
That result must be retained, but it does **not** test or reject the visual VAE
proposal described here.

Use separate names:

- implemented historical experiment:
  `conditional-local-material-transition-cvae-v0`;
- implemented shadow visual family:
  `action-conditioned-visual-mark-cvae-v0`.

Never describe the former as the first implementation, rejection, or
validation of the latter.

## Corpus requirements

The existing trajectory-posterior corpora do not store raw registered camera
frames and are insufficient for end-to-end visual representation learning.
`registered-visual-trajectory-corpus-v1` now preserves, per transition:

- registered, rectified pre-action and causally later post-action canvas
  images at a resolution that preserves useful boundary orientation;
- camera product identifiers, timestamps, calibration/noise provenance,
  valid/occlusion masks, and rectification metadata;
- the selected action, compact brush belief, and conditional motor
  realization;
- action-aligned crop coordinates with enough surrounding visual context;
- whole-trajectory split identity and genuine-stop versus truncation
  provenance.

Process thickness, wetness, pigment, segmentation, and exact contact may be
retained separately for process validation or evaluator-only diagnostics, but
must not become agent training inputs unless a later decision explicitly
declares a supervised simulation-only auxiliary experiment.

The v1 visual corpus implementation is in
`src/active_painter/visual_trajectory_corpus.py`. The driver emits a record only
from the last accepted registered camera bundle before an action and the first
eligible bundle after the runtime's causal exposure boundary. Crop extraction
occurs after whole-trajectory splitting. The current files remain provisional
simulation-camera evidence, not hardware-calibrated or sensor-equivalent data.

## Admission gates

Before a visual model influences policy inference, require:

1. held-out normalized image or coefficient likelihood;
2. calibrated visual uncertainty, including ambiguous material outcomes;
3. tone and oriented-boundary fidelity at local and canvas scales;
4. one-, two-, four-, and passage-length visual rollout evidence;
5. action-, brush-, and context-ablation evidence showing that conditions are
   used;
6. denial of exact process material inputs;
7. comparison against deterministic and no-interaction-latent baselines;
8. explicit mapping of outputs into VFE and EFE without an aesthetic score or
   double-counted observation factor.

## Immediate implications

- Do not invest further hierarchy capacity in reconstructing coarse wetness.
- Do not train terminal composition claims from all six `SpatialCanvasState`
  channels as if they were the accepted visual state.
- Preserve the 2026-08-12 stop-pilot as valid termination evidence; its coarse
  final posteriors are not a visual-model corpus.
- Prioritize registered visual data retention, a visual baseline, and a
  shape/mass hierarchy before resuming composition-preference integration.
