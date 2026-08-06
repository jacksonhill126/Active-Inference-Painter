# Current Implementation Notes

This document preserves the detailed implementation-oriented README from the
early Python prototype. For the concise project entry point, see the repository
root `README.md`.

This is a clean Python/PyTorch starting point for the painter architecture.
It implements one feature rigorously: **policies are inferred so that the eventual `stop` state is expected to occur near a preferred material coverage of roughly 80–90%**.

> **Obsolete compatibility path:** `planner_state_kind="summary"` is retained
> only for regression tests, tractable reference fixtures, and old
> checkpoints. Its six hand-selected global aggregates are predictively
> insufficient for image-making and are not a candidate highest-level painting
> representation. Explicit construction emits a deprecation warning.

It deliberately separates:

- **Generative process**: stochastic wet-into-wet oil deposition with persistent
  wetness, conserved bulk pigment, an optically dominant surface tone, and
  paint-presence material coverage. A pixel counts once after its thickness
  crosses the declared presence threshold; additional layers change material
  and optics but not covered area. There is intentionally no wetness decay.
- **Generative model**: a learned ensemble transition density over latent canvas states plus an explicit observation likelihood.
- **Posterior inference**: variational state estimation by minimizing variational free energy.
- **Preferences**: a terminal Beta density over coverage, applied only when a policy terminates in `stop`.
- **Policy inference**: posterior over policies from expected free energy.
- **Execution**: arm realization stays below policy selection, but the planner can now compare declared first-stroke motor realization latents such as Cartesian IK, joint-space splines, elbow-led arcs, and positive/negative upper-arm roll sweeps through predicted canvas and proprioceptive outcomes.

No hand-written aesthetic reward is used. Coverage is not inferred from visible
color: white paint on white ground occupies canvas area, while repeated paint in
an already occupied pixel does not add coverage.

## Brush paint-handling

The arm's brush (`VerticalCanvas.paint_at` / `Brush` in `arm_sim.py`) is a round
contact patch with hard support, enriched with several physical behaviors. All of
this lives in the generative *process*, below the painting-policy boundary; the
learned transition model observes the resulting canvas transitions and adapts.
It is oil: paint does not dry within a session and the canvas keeps its wetness
(there is no wetness decay). Fresh paint carried by a brush is finite and can
decline within and across marks.

- **Brush loading.** Each dedicated white or black brush has a persistent
  normalized fresh-paint reservoir. The selected mark's `amount` controls its
  deposition setting; the latent brush load controls how fully that consequence
  can be realized. Contact consumes fresh load according to deposited film
  thickness (`brush_load_capacity_thickness`). Loaded/unloaded remains material
  state, not a controller paint gate: any positive-pressure canvas contact
  deposits, including press, lift, and unintended contact. The depletion
  calibration is provisional.
- **Directional shape.** Each deposition step paints the disc swept from the
  previous contact point, so travel elongates and connects the mark. The
  cross-stroke radius is unchanged (round brush); only the along-travel extent
  grows (`brush_directional_enabled`).
- **Bristle furrows.** A round brush is a bundle of hairs; a fraction run dry
  (`brush_bristle_gap_fraction`), carving lengthwise furrows that stay unpainted.
  Unlike a deposition-rate wobble these survive opacity build-up, so the mark
  reads as brushed (`brush_bristle_*`; set depth and gaps to 0 for smooth).
- **Canvas tooth/grain.** A fixed substrate texture (as in Krita/MyPaint): a
  lightly loaded brush deposits only on the raised tooth, while more pressure
  works paint into the valleys. This is the primary source of mark texture and
  survives opacity because unreached valleys stay bare
  (`canvas_grain_strength`, `canvas_grain_reach_*`).
- **Stroke-end taper.** Brush width ramps in/out over the ends of the paint
  phase (`brush_taper_fraction`, `brush_taper_min_width`), so marks come to
  points instead of round caps. Driven by the stroke controller's `flow`
  envelope, below the policy boundary.
- **Wet blending (dirty brush).** Per deposition step the head skims a
  pressure-scaled fraction of the wet surface layer into a small held reservoir
  (volume plus pigment mass, exactly conserved against the canvas ledger), and
  redeposits a share of it mixed with the fresh load, biased toward the leading
  edge so paint is pushed ahead of the stroke. This bidirectional-transfer loop
  is the same cheap core used by ArtRage and Krita's color-smudge engine
  (`brush_pickup_*`, `brush_capacity_thickness`, `brush_release_fraction`,
  `brush_push_forward`). Every knob is calibratable from a few real strokes.
- **Bristle-tip trailer dynamics.** The painting point is a damped follower of
  the commanded contact point (`brush_tip_lag_seconds`), so entries hook and
  curves flow like a pulled brush tip rather than a rigid stamp.

The policy sampler also biases proposed strokes toward longer sweeps (a declared
computational proposal bias), since a round brush over a short span reads as a
dab. This changes finite candidate support and is not a normalized policy prior.

The model maintains a compact `q(load, black_fraction)` for each dedicated
color brush. Before a selected mark it infers `preserve` versus instantaneous
`reload`; these are explicit preparation policies with a declared policy prior
and conditional expected-free-energy terms for the selected mark's material
amount and pigment outcome. There is no procedural low-load threshold. Reload
sets the selected brush to full and uniformly its nominal color, clearing
picked-up material. Preserve carries process and belief state across the mark.
The current load transition is an approximation and increases uncertainty
about canvas pickup. `infer_load_from_mark` defines the local camera-derived
deposition likelihood and VFE update. The registered grayscale camera
likelihood now updates the spatial material posterior, but it does not yet
extract the per-mark deposition statistic needed to drive this separate brush-
load update.

This makes incidental contact a real predicted paint consequence while keeping
paint preparation inside policy inference. Brush tilt relative to the canvas
normal is not yet modeled in the 2-D deposition footprint; the round footprint
is oriented only by travel direction.

## Core expected-free-energy terms

For a policy ending in `stop`:

- **Terminal risk**: divergence between the predicted terminal coverage distribution and the preferred terminal coverage density.
- **Ambiguity**: excess expected entropy of future observations above the dry-canvas likelihood baseline; wet/thick paint increases observation uncertainty.
- **Transition risk**: negative entropy of the predicted transition outcome under flat transition-outcome preferences, up to an omitted constant.
- **Transition ambiguity**: expected conditional entropy of transition outcomes under the learned ensemble members.
- **Epistemic value**: a moment-matched information-gain approximation
  `I(theta; s_next | s, a)` where the learned ensemble represents posterior
  uncertainty over transition parameters. It is logged as the identity implied
  by transition risk plus transition ambiguity, not added as a separate term.
- **Terminal entropy and pragmatic value**: logged so terminal risk can be
  checked as `KL[q(C_T|pi) || p*(C_T)] = -H[q(C_T|pi)] - E_q log p*(C_T)`.

Policy selection uses the risk-plus-ambiguity decomposition. The
information-gain/pragmatic-value form is a diagnostic identity, not an extra
score mixed into expected free energy.

State inference logs variational free energy separately from policy expected
free energy. In spatial mode, a local Gaussian transition prior and material
observation likelihood are fused at the native pixel level; the posterior is
then projected deterministically into the material pyramid. Coarse levels do
not independently guess material that disagrees with the pixel posterior.
The obsolete summary fixture reports its heteroscedastic expected likelihood
with the declared `summary_vfe_report_samples=4096` Monte Carlo budget. This is
a reporting-only approximation after optimization. Its RNG state is restored
after the draw, so it has no direct or random-stream path into later learning,
EFE, or policy selection; its accepted reference error is recorded in
`docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`.

The immediate `stop` policy is always available. Continuation policies are sampled as one or more strokes followed by `stop`, so the planner can anticipate coverage overshoot rather than merely checking a threshold afterward.

## Composition hierarchy (compression gap)

Spatial mode carries a hierarchical composition layer (`canvas_hierarchy.py`):
a spatial canvas latent with a learned decoder over the material fields. The
single declared structural preference over terminal canvases is

```
p*(s_T) ~ exp(composition_gap_precision * gap(s_T))
gap(s)  = ELBO_hierarchical(s) - max_m log p_m(s)   [nats per cell-channel]
```

where the baseline is the best member of a declared, hand-written,
parameter-free family of two context-free codes (`composition.py`): the
per-image per-channel iid-cell Gaussian, and a 3x3 hollow-neighbourhood local
Markov code that predicts each cell from the mean of its eight neighbours
excluding itself. Scoring against the better member means the gap cannot be
earned by local smoothness alone. All three codes share one quantization floor.
The hierarchy pays for its latent through the KL term, so a blank canvas scores
<= 0, iid noise scores <= 0, a locally smooth but globally unstructured canvas
(a soft blob, low-pass noise) scores <= 0, and only canvases whose distant parts
mutually predict each other score positive. `composition_local_baseline_enabled`
is the declared flag that restores the previous iid-only baseline exactly, so
the difference the family makes is separately measurable. No content term
(balance, contrast, subject) appears anywhere:
the preference references only the hierarchical model's explanatory advantage.
The hierarchy trains online alongside the dynamics ensemble; the per-policy
`composition_gap`/`composition_risk` components and the current belief gap are
logged in diagnostics.

The hierarchy's FIRST evidence now comes from the body. `config.bootstrap_generator`
selects the bootstrap mark source: `motion_manifold` (default) draws joint-space
sweeps of the arm's own reachable-motion manifold and projects them to canvas
geometry by forward kinematics (`motion_manifold.py`), while `random_stroke`
retains the previous iid source so the difference is measurable. Bootstrap runs
in episodes of `bootstrap_episode_marks` marks and clears the canvas only at an
episode boundary, so each finished canvas reaches the canvas/relational
likelihood whole through `spatial_agent.add_composition_canvas` rather than being
wiped mid-episode. `bootstrap_composition_train_steps` declares an explicit
gradient budget at each boundary; it is a gradient hyperparameter, not an
objective term, and defaults to 0 because the measured gap is flat below a few
hundred steps and the cost is real.

`diagnostics()["compositionBootstrap"]` reports that measurement: the gap on the
bootstrapped canvases, on a blank canvas, on a marginal-preserving cell shuffle
of the bootstrapped canvases, and on iid scatter, plus the generator that
actually ran, the painted path length, and the mean episode coverage. It is
EVIDENCE and NOT a decision quantity: no EFE term, VFE term, preference,
precision belief, or policy prior reads it, and the block carries a `declaredAs`
string saying so. The honest acceptance metric is
`gap(bootstrapped) - max(gap(blank), gap(shuffled))`, because the cell shuffle
preserves every per-channel marginal exactly and therefore isolates the
hierarchical code; the raw probe range is reported too but grows with model
confidence about an out-of-distribution probe rather than with discriminative
validity.

Measured at 900 gradient steps per episode boundary (96 marks, 4 episodes):
`motion_manifold` reaches a margin of +1.634 against `random_stroke`'s +1.088, a
1.50x win obtained while laying 41% less painted path (22.75 vs 38.62) and half
the coverage (0.169 vs 0.334), so the confound runs against the winning arm. Both
arms score a positive gap on their own bootstrapped canvases (+1.58 / +1.01),
which neither did before. At the default budget of 0 extra steps NEITHER arm
discriminates (margins -0.125 and -0.200): the generator change is not measurable
without a declared gradient budget. Full numbers, including the finding that the
binding null model is blank rather than the shuffle once the model is trained,
are in `docs/DEVELOPMENT_AUDIT.md`. Because the gap is evaluated on every candidate
terminal state including immediate stop, continue-vs-stop comparisons already
express compression progress: painting continues while strokes are expected to
increase the hierarchy's explanatory advantage near the coverage band.

The same model carries two slower transition levels that remain distinct from
the structural preference:

- a persistent `8 x 4 x 4` canvas posterior `q(z_canvas)` in the default
  16-cell planner configuration;
- a 24-dimensional relational posterior `q(z_relational)` inferred from eight
  uncertain region slots and every pairwise displacement, distance, overlap,
  tone difference, and material-mass relationship. The deterministic slot
  observation preserves disconnected components and subdivides large connected
  paint masses, so dense passages do not collapse into one relational object.

Both posteriors update only at executed passage boundaries. Two learned
Gaussian transition likelihoods operate over them. The aggregate likelihood is
conditioned on a deterministic descriptor of the whole proposed mark
trajectory. The passage likelihood is Markovian: it receives the persistent
`PassageLatent` plus a passage-relative phase for each subordinate mark, rolls
the canvas and relational latents forward one mark at a time, and decodes a
coarse material observation at every step. Real marks train this per-step
likelihood without directly updating the persistent canvas or relational
posterior mid-passage.

For structured passage candidates, the hierarchy therefore evaluates
`sum_t KL[q(z_t) || p(z_t | z_0, z_passage, phase_1:t)]` over every predicted
mark. Ensemble members are averaged at each step, while the temporal terms are
summed. Unstructured candidates use the aggregate policy transition. Canvas
and relational terms retain their separate declared precisions and remain zero
until the relevant likelihood has received training updates. Immediate `stop`
uses an identity latent transition prior. These are transition beliefs inside
EFE, not composition rewards.

## Rollouts, policy priors, and precisions

- **Member-wise trajectory rollouts.** With a learned ensemble, each member
  propagates its own state particle through a candidate policy, so parameter
  uncertainty compounds over the horizon instead of collapsing to a
  moment-matched mixture after every step. Terminal coverage variance is the
  across-member disagreement of aggregate coverage (which carries the spatial
  correlation a stroke induces) plus mean within-member predictive variance.
  Dense-grid policies are evaluated in one batched pass per rollout step;
  local-patch mode evaluates only the stroke-supported pixel patch.
- **Sparse pixel-local spatial rollouts.** Spatial mode defaults to
  `spatial_transition_mode="local_patch"`: brush transition likelihoods are
  evaluated on pixel-derived local patches around stroke support, while cells
  outside support use an explicit identity transition prior whose constant
  entropy is logged as an approximation and omitted from local EFE terms.
  Motor-conditioned first-transition rescoring uses the same sparse overlay
  path, with support expanded to include both the realized material delta and
  the action raster.
  Set `spatial_transition_mode="dense_grid"` to use the older dense planner
  grid rollout for debugging.
- **Declared stop prior.** The policy posterior is
  `softmax(-gamma * G + log p(pi))`, where `log p(stop-first)` follows a
  sigmoid in believed coverage centered at `minimum_stop_coverage` and
  continuation policies carry a flat prior. This replaces the previous
  procedural stop veto: premature stopping is a priori unlikely, never
  inadmissible, and demotions are logged as diagnostics.
- **Coverage-seeking stroke proposals.** In spatial mode, a declared fraction
  (`proposal_low_coverage_mix`) of candidate strokes start in low-coverage
  regions of the current belief. This is a computational proposal distribution:
  it changes which finite candidates exist but is not a normalized policy prior
  and is not added to expected free energy or the policy posterior.
- **Hierarchical passage proposals.** A declared fraction
  (`passage_proposal_mix`) of continuation candidates are generated from a
  slower `PassageLatent` transition prior over several related marks. Current
  passage kinds are parallel mark bands, chained mark phrases, and polylines.
  A polyline is represented by center, central direction, total length, signed
  turn, segment count, width, amount, and tone, then deterministically decoded
  into two to four endpoint-connected straight brush actions. Each segment is
  still a regular learned mark, with lift and local receding-horizon inference
  before the next segment; connected geometry does not imply uninterrupted
  brush contact. Every passage terminates in `stop`, and expected free energy
  scores the predicted consequences. The global mixture that samples passage
  candidates is a computational proposal. During local continuation inference,
  the persistent `PassageBelief.transition_log_prior` is the explicit
  transition prior over the multi-mark latent trajectory. Neither is an
  aesthetic reward.
- **Passage-plan proposals.** When the planning horizon is deep enough, a
  declared fraction (`passage_plan_proposal_mix`) of candidates are generated
  from a slower `PassagePlanLatent` over multiple passage latents. The plan
  carries a slowly evolving center, direction, turn, tone, and material amount;
  its child passages generate the actual marks. The plan is still only a
  computational proposal in the current global candidate set, and every
  candidate still terminates in immediate `stop`.
- **Amortized learned proposal (in progress).** Spatial mode can construct a
  factorized `PolicyProposalNetwork` conditioned on the current canvas and
  relational posterior means. It is trained after planning by
  self-normalized, posterior-weighted maximum likelihood toward the existing
  base-EFE painting-policy posterior. It proposes mark and passage latents only;
  immediate stop and passage-plan compounds remain outside its learned scope.
  `learned_proposal_mix=0.0` is the default, so the emitted candidate stream is
  the hand-written baseline unless an experiment explicitly raises the mix.
  The network is a proposal, not a policy prior: its log density is never added
  to EFE, VFE, a preference, or the normalized painting-policy posterior. It
  also does not correct finite-candidate bias. Sampling, density, training,
  checkpoint continuation, exact zero-mixture parity, and separation from EFE
  are covered by `tests/test_proposal.py`. The candidate-count, horizon, seed,
  mixture, posterior-mass, and top-action grid in
  `tests/test_proposal_convergence.py` and
  `docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md` produced a negative result:
  the current posterior is stable only as `Q(pi | sampled set S)`, not as a
  proposal-invariant continuous-policy posterior. The learned emission default
  therefore remains zero pending an M3 correction.
- **Receding-horizon passage inference.** A global plan predicts several
  passages but execution commits only to the first. Within that passage, each
  observed mark updates a slow diagonal-Gaussian posterior over center,
  direction, length, spacing, width, and amount, plus a beta-Bernoulli tone
  factor. A small local policy set is then inferred before the next mark. The
  arm performs the deeper global deliberation from a retracted pose only at a
  passage boundary. Local candidate policies retain the same passage latent and
  their passage-relative start index, so local correction cannot silently turn
  the remaining marks back into unrelated one-mark policies. Each passage kind
  has separate evidence support: a newly introduced polyline receives spatial
  rollout EFE but no passage-trajectory likelihood KL until that likelihood has
  trained on executed polyline steps.
- **Embodied motor realization priors.** During arm-driven planning, top
  canvas candidates are expanded into declared first-stroke
  `MotorPrimitiveLatent` alternatives (`cartesian_ik`, `joint_spline`,
  `elbow_pivot` by default). Each realization is forecast through the arm,
  contact, and canvas simulator before posterior policy selection. The chosen
  primitive contributes separate proprioceptive EFE terms over 27 named
  normalized outcomes: per-joint current, torque, velocity, acceleration,
  target error, and joint-limit proximity, plus contact loss, pressure error,
  and path error. Several stochastic coupled-arm rollouts estimate each
  predictive density. Motor risk is expected negative log probability under
  declared homeostatic outcome preferences; ambiguity is analytic likelihood
  excess entropy; epistemic value is analytic diagonal-Gaussian mutual
  information. Motor alternatives are marginalized under their declared
  priors before the conditional realization is selected. The selected
  primitive is also encoded into replay transitions
  and learned rollouts as motor-conditioned action channels, so the learned
  transition likelihood is `p(s_next | s, stroke, motor_realization)` rather
  than stroke-only. Hard joint/current/workspace limits remain external safety
  constraints, and no motor-ease reward is introduced.
- **Learned motion reliability.** Per motor realization kind, the driver
  maintains an inverse-gamma precision belief over the squared ratio of
  realized to forecast tracking error (`motor_reliability_*`), updated after
  every executed stroke from path and pressure residuals. The posterior mean
  scales the expected squared error of the execution-fidelity outcome channels
  inside motor EFE, so motions that prove jittery pay proportionally more risk
  and reliable ones win selection; the belief's remaining uncertainty is
  credited as information gain for trying a kind. The belief persists in the
  checkpoint and is reported in diagnostics (`motionReliability`). Forecast
  rollout particles beyond the first also perturb friction, backlash,
  transmission stiffness, and process noise (`body_param_jitter_fraction`), so
  motions that amplify body-parameter uncertainty forecast wider even before
  reliability evidence arrives. This is the sim-to-real seam: on hardware the
  same residuals calibrate the body model instead of a copied simulator.
- **Per-modality precision BELIEFS.** The seven modality precisions
  (`terminal_risk_precision`, `ambiguity_precision`, `transition_precision`,
  `composition_gap_precision`, `canvas_latent_transition_precision`,
  `relational_transition_precision`, and the modality-level
  `motor_modality_precision`) plus `policy_precision` are the posterior means of
  Gamma beliefs (`precision_beliefs.py`), updated by the reference textbook's
  Chapter 10 rule `dF/dgamma = (alpha/gamma - beta0) + (pi - pi0).(-G)` from the
  realized `(G, F)` candidate pair after each planning round. `F` is the
  brush-preparation negative log marginal evidence, which is free of the painting
  policy precision, so no learned gamma appears on both sides of its own update.
  Each belief's prior mean is exactly the previous declared constant and an
  unobserved belief returns it bit-identically, so
  `precision_beliefs_enabled=False` reproduces the old arithmetic exactly. The
  posterior mean is clamped to a declared `[0.1x, 10x]` support: measured
  unbounded, disagreeing evidence attenuates a modality 15x, which for terminal
  coverage would let outcome data switch a declared preference off. Measured
  outcome: the gammas drift only slightly (0.88 to 1.01 over six real planning
  rounds) and precision ordering does NOT track discriminativeness -- see spec
  register item 26 for the numbers and the reason.
- **Per-observation-channel modality units.** Every EFE modality is reduced to
  nats per observation channel and records the declared name of its normalizer in
  the component dataclass, so `raw * precision * normalizer == stored` is
  checkable from telemetry. Six modalities were already per-channel densities;
  the motor modality's raw 27-channel sum is the only new divisor, and that is a
  27x reweighting rather than a tidy-up. `modality_normalization_enabled=False`
  restores the historical mixed units.
- **Gap-progress stopping as a policy prior.** `log p(stop-first)` is a product
  of the coverage sigmoid and `logsigmoid(-s * E[dGap]/sd[dGap])` over a Gaussian
  random-walk belief on the per-mark compression-gap increment. Both factors are
  <= 0 and the progress factor is exactly 0 for continuations and for an
  unobserved belief, so it can only make stopping less unlikely and never
  manufactures value. The increment never enters expected free energy, and the
  terminal coverage preference itself remains declared and un-learned.
- **Bootstrap ensemble training.** Each ensemble member trains on its own
  Bernoulli-masked subset of every batch (`ensemble_bootstrap_probability`),
  keeping member disagreement usable as an approximate parameter posterior;
  calibration tests cover held-out z-scores and off-distribution disagreement.

## Provisional spatial material planner mode

The web entry point now defaults to `planner_state_kind="spatial_material"`.
This is an interim low-level material-transition baseline, not the final
painting-level representation. The obsolete summary fixture can still be
requested explicitly for compatibility:

```bash
python -m active_painter.web_server --planner-state-kind spatial_material
```

Spatial mode performs an initial dynamics bootstrap before the URL is printed.
For a quick no-bootstrap smoke test, add
`--driver-bootstrap-transitions 0 --driver-bootstrap-train-steps 0`.
`--driver-bootstrap-generator {motion_manifold,random_stroke}` selects the
bootstrap mark source and `--driver-bootstrap-composition-train-steps` declares
the episode-boundary canvas/relational gradient budget; both are named in the
startup line so an attribution A/B run is self-documenting.
The planner runs on CUDA automatically when available; pass `--device cpu`
(or `cuda:1`, etc.) to override. The resolved device is printed at startup.
The web renderer displays the canvas on a neutral gray ground so both white and
black paint are visible. Tone support is unconstrained by default; use
`--stroke-tone-prior black`, `--stroke-tone-prior white`, or
`--stroke-tone-prior random` to set the policy sampler's tone support. In
random mode, candidate geometries are proposed as matched black/white policy
alternatives where the candidate budget allows, so tone is selected by the EFE
posterior over predicted material consequences rather than by an unpaired
coin flip.

In that mode the driver evaluates policies over:

- `SpatialCanvasState`: explicit `thickness`, persistent `wetness`, conserved
  `black_mass`, surface-tone, ground-contrast, and material-coverage fields.
  Surface tone represents the optically dominant wet top layer separately from
  bulk pigment mass. Contrast and coverage are deterministic consequences of
  surface tone, thickness, and the canvas substrate, not reward variables.
- `MaterialPyramidLevel`: a coarse-grained material pyramid derived from the
  pixel canvas. The default live canvas exposes a native pixel level plus
  configured tile levels and the current planner grid. Coarse coverage fields
  are downsampled from pixel-derived material coverage rather than recomputed
  from already-averaged thickness, so material coverage mass is preserved
  across levels. Local patch rollouts use the native pixel level for brush
  transitions, then deterministically coarse-grain predicted terminal fields
  for planner-scale composition and mark-event summaries.
- `rasterize_stroke_action`: deterministic action-conditioning fields for the
  stroke footprint, start/end, width, amount, and tone.
- `LocalSpatialDynamicsEnsemble`: a masked, action-conditioned CNN ensemble
  for `p_theta(s_patch_next | s_patch, a_patch)` in the default sparse local
  mode. `SpatialDynamicsEnsemble` remains available for dense-grid rollouts.
- `SpatialExpectedFreeEnergy`: a risk-plus-ambiguity evaluator whose terminal
  coverage comes from the explicit pixel material-coverage field, including
  white paint on white ground.
- `MarkEventBelief`: a connected-component posterior summary over spatial
  material coverage, exposing mark centers, covariances, material mass,
  wetness, observed tone, ground contrast, and coverage for higher-level diagnostics. It is
  not currently a policy preference or reward term.

Execution forecasts are also observed as spatial material transitions before
policy selection, so motor feasibility affects admissibility and predicted
canvas outcomes rather than entering as a reward-like motor-ease term. The six
summaries remain diagnostics in this mode. No balance, flow, or composition
reward has been added.

The intended replacement architecture uses flexible multiscale latent layers
trained to explain and predict permitted camera observations and later
consequences. Spatial locality, temporal depth, and information bottlenecks are
structural assumptions; the feature contents are not a hand-written list of
"useful" aesthetics. Latents must demonstrate held-out predictive information,
calibration, and causal effects on later inference under freeze/shuffle/reset
tests. Thickness, wetness, and mobility remain local physical causes for
contact prediction rather than global image-making features. Coverage remains
a terminal material preference/readout, not the canvas representation.

## Install and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m active_painter.demo --episodes 3 --pretrain-steps 1200
pytest
```

The demo writes episode images and a JSON trace to `runs/`.

## Native arm visualizer

There is also a lightweight Python-native arm/canvas visualizer:

```bash
python -m active_painter.arm_visualizer
```

It shows a stochastic coupled 4-DOF arm plant with encoders, pose-dependent
inertia, Coriolis coupling, residual gravity, motor/link inertia, friction and
compliance; plus a vertical wet oil-paint canvas, soft wrist contact,
pressure-dependent brush width, motor telemetry, and material coverage.
The roll coordinate rotates the elbow hinge around the upper-arm axis. Two
fixed-endpoint roll-sweep policies are inferred alongside the existing motor
realizations; their start and end poses use exact fixed-roll IK, and their
proprioceptive and canvas consequences enter the motor EFE posterior.
The expensive embodied refinement is capped at the three best base-EFE canvas
policies, keeping the default stochastic forecast count below the previous
eight-policy, three-realization budget to account for richer fixed-roll IK,
while all canvas candidates still receive their base EFE evaluation.
This body simulation sits below the painting policy boundary; it does not select
painting policies.

For a smoother WebGL view, run the Python-backed Three.js viewer:

```bash
python -m active_painter.web_server
```

Then open `http://127.0.0.1:8017`. Python owns the simulation state and active
inference core. Three.js builds the direct-drive robot body hierarchy from
`models/active_inference_painter.xml` through `/api/robot-model`, and renders
the arm, controls, telemetry, and the Python material image from
`/api/canvas.png`. MuJoCo is not used as a paint renderer.

The same XML now declares a provisional four-camera rig: opposing continuous
oblique brush/contact views, a park-only deployed head-on inspection
reference, and a continuous overhead brush-standoff profile.
`/api/robot-model` exports their poses, roles, availability, grayscale
channels, model-input sizes, sample rates, registrations, ideal
FOV/resolution, incidence, axial canvas-depth range, and provisional
calibration status. `active_painter.camera_geometry` maps the three surface
views to a shared top-left canvas UV frame with an ideal planar homography.
The tangent standoff camera instead uses `canvas_edge_profile` and arbitrary
world-point projection; it is intentionally excluded from homographic
rectification.

The reproducible 243-pose sweep in
`active_painter.camera_pose_sweep` evaluates a 9 x 9 contact grid at -32, 0,
and +32 degrees of upper-arm roll. The two continuous oblique views have 100%
combined segmentation-based brush-tip visibility, and the overhead profile
retains 100% visibility in the sampled workspace. Publication-ready frames,
maps, CSVs, and interpretation are in
`docs/CAMERA_OBSERVABILITY_BRIEF.md`.

`CameraObservationProcess` now inserts superficial grayscale Python canvas
appearance into MuJoCo-rendered views while retaining rendered arm/brush
occlusion. It applies provisional XML-declared timing, noise, quantization,
and a weak lighting-residual specular approximation. Exact segmentation stays
inside the generative process and is absent from agent-facing frames. The
perfect `/api/canvas.png` remains diagnostic-only.

The `provisional-multiview-v4` XML assigns the owned OM System OM-1/25 mm and
Sony A7R II/35 mm to the opposing continuous oblique roles, with Super 35
selected provisionally for the Sony. Nominal 16:9 vertical fields of view and
active sensor widths are declared in the same XML pending physical
calibration. It separates 3840 x 2160 physical acquisition, 512 x 512 global
model input, 256 x 256 foveal output, and MuJoCo reference-render resolution.
The additional head-on inspection and overhead global-shutter cameras remain
planned.

The `camera-observation-interface-v1` simulator process now retains each
native grayscale exposure, derives the global view independently, and samples
explicitly requested foveae directly from native pixels. Requests use canvas
UV center and span and may cite only a sensor posterior, policy prediction, or
operator diagnostic as their selection basis. There is no default fovea and
no selector may read exact simulator pose, contact, visibility, segmentation,
or material state.

The web viewer exposes `fovea-trace-v0` as a diagnostic of delivered foveal
products. The current delivered fovea is drawn as a bright canvas-registered
box and crosshair; prior deliveries fade as a ghost trace. Its default
retention is 10 seconds because the agent does not yet declare a foveation
memory horizon. If a future agent declares `foveation_memory_horizon_s`, the
viewer uses that value and identifies it as the retention source. Clicking the
canvas preview creates an `operator_diagnostic` request, not an inferred gaze
policy, and the trace records the request only after the camera process has
actually delivered its foveal frame.

`camera-spatial-likelihood-v0` is now connected to the provisional spatial
active-inference model. It mosaics each global product with any native-derived
foveae from the same exposure, so correlated products enter only one
likelihood factor. Its nonlinear observation model predicts superficial
grayscale from latent thickness and surface tone. Wetness and bulk pigment
mass have zero direct image Jacobian and remain transition-prior beliefs;
white-on-white supplies no thickness evidence. A broad state-independent
outlier component gives an inferred occlusion responsibility without exposing
MuJoCo segmentation. Camera VFE logs state and occlusion complexity separately
from expected negative log likelihood. The per-camera uncertainty and inlier
priors are explicit in the XML and marked provisional.

Physical HDMI acquisition, physical lens/capture calibration, and a learned
camera encoder are not implemented. Sensor-equivalent control still fails
closed because the existing body posterior is not connected to motor-forecast
initialization and the action-conditioned transition-prior/camera-update loop
is not yet scheduled continuously; camera likelihood construction no longer
causes that block.

`active_painter.camera_calibration` now supplies the first hardware
calibration tool: a metric 11 x 8-inner-corner target generator and native-
frame Brown-Conrady intrinsic solver. It reports reprojection residuals,
coverage, tilt diversity, and acceptance gates. This is tooling rather than a
calibrated result; no measured OM-1 or A7R II frames have been supplied yet,
and accepted intrinsics have not replaced the nominal XML geometry.

The default `sensor_equivalent` observation mode currently fails closed:
the viewer and scripted execution remain available, but policy inference,
learning, oracle bootstrap, and process-derived planner-state construction are
disabled. That includes the motion-manifold bootstrap: the oracle gate is the
first statement of `bootstrap_dynamics`, so no bootstrap simulator is built and
no sweep sampler is constructed in sensor mode, and
`diagnostics()["compositionBootstrap"]` is `None` there.

The first compact sensor-conditioned bodily inference component now exists in
`active_painter.body_inference`. `BodyStateEstimator` constructs a diagonal
constant-velocity transition prior and assimilates encoder position, encoder
velocity, an optional contact switch, and optional contact-force samples
through separately named Gaussian/Bernoulli likelihood factors. The conjugate
mean-field updates report complexity and expected negative log likelihood
separately, in nats, both globally and per factor. Its precision values are
required through a versioned `BodyLikelihoodSpec`; the implementation has no
silent numerical defaults and is not called calibrated until hardware or
declared simulation measurements supply them.

This is not yet wired into painting policy inference. Motor current, bus
voltage, tool deflection, temperature, and fault flags are explicitly reported
as unassimilated by this posterior. Faults remain hard-safety inputs, while
current/deflection need declared conditional likelihoods before they may
provide contact or load evidence. The camera-conditioned material posterior is
present; body-to-motor-forecast initialization and continuous action-
conditioned observation scheduling remain fail-closed boundaries for live
policy inference.

To reproduce the legacy upper-bound comparator, opt in explicitly:

```bash
python -m active_painter.web_server --observation-mode oracle_material_state
```

This diagnostic mode exposes exact simulator material and body state and is
not a physically accessible observation condition.

The current controller still emits `native-abstract-v0` Cartesian motion. With
the native plant, a named `legacy_canvas_cartesian_retarget` adapter uses
conventional visualization-only IK to register that motion with the physical
MJCF canvas. With the MuJoCo plant, the same below-policy mapping generates
physical joint commands; realized joint telemetry, tip motion, compliance, and
contact then come directly from MuJoCo. Neither mapping selects or scores
painting policies.

The realized plant is selectable:

```bash
python -m active_painter.web_server --plant-backend native
python -m active_painter.web_server --plant-backend mujoco
```

`native` remains the default accepted reference. In `mujoco` mode, position
commands use the stable `yaw`, `pitch`, `roll`, `elbow` joint order in radians.
MuJoCo supplies encoder position/velocity, dynamic output-equivalent winding
current, actuator force, applied controller voltage, tip position, brush
compression, and exact brush/canvas contact. The `dcmotor` actuators include
back-EMF, finite electrical time constants, a 48 V controller limit, and peak
torque saturation. Realized contact drives deposition into the existing
`VerticalCanvas`; MuJoCo does not represent paint.

The brush has passive axial compression plus a two-axis lumped flexure at the
ferrule. Its rigid 35 mm bundle can rotate tangentially under isotropic canvas
friction and spring back after lift-off; individual bristles are not modeled.
The deflected MuJoCo tip is the point mapped into `VerticalCanvas`, and the web
robot state exposes both bend angles for the Three.js mirror. Bend limits,
stiffness, damping, and friction are provisional rather than measured.

The electrical model is not a transistor-, phase-, or thermal-level drive
simulation. Public RobStride torque, speed, voltage, resistance, inductance,
and current data ground an output-side integrated-drive equivalent. Back-EMF,
terminal resistance, and a one-point viscous-loss equivalent are derived to
preserve the published no-load speed/current and peak-stall operating points;
position-loop gains remain approximate. The viscous term does not identify the
real split among mechanical, magnetic, and electronics losses. Thermal
derating, cogging, detailed friction, gearbox compliance/backlash, and inverter
behavior await measured hardware data. These below-policy dynamics do not add
a painting preference.
The two physical encoders listed for each drive are metadata only at this
stage; MuJoCo joint sensors remain exact and have no quantization, noise,
sample/transport delay, dropout, or motor/output-side disagreement.

The current painting controllers still operate in the
`native-abstract-v0` canvas frame. A conventional execution adapter maps their
Cartesian targets into the hardware-oriented MJCF workspace and maps the
realized tip back to the material canvas. Counterfactual motor forecasts still
use a deep-copied native plant for tractability and are reported as
`native-abstract-v0 approximation`. Direct MuJoCo counterfactual forecasting
and sensor-only body-state inference remain future work.
When the active-inference driver selects `stop`, the web runtime automatically
starts a fresh painting. Every fifth completed painting is saved by default to
`runs/web/painting_####.png`; use `--save-every-paintings` and `--archive-dir`
to change that behavior.

The web runtime keeps a rolling arm telemetry log with joint positions, target
positions, velocities, currents, torques, voltages, contact pressure/force, and
the current driver phase. Download it from `http://127.0.0.1:8017/api/telemetry.csv`.
Use `--telemetry-sample-hz` and `--telemetry-max-samples` to control the log
rate and retention window.

## Architectural boundary

Painting cognition should remain active-inference based. A later robot backend may use conventional:

- forward kinematics;
- inverse kinematics;
- trajectory interpolation;
- low-level motor control;
- hard safety limits.

Those mechanisms realize an inferred Cartesian/contact policy; they do not select the painting policy.

## Next integration steps

1. Profile passage planning and batch motor realizations across candidate
   policies without changing posterior semantics.
2. Replace diagonal motor outcome covariance with structured joint/contact
   covariance and calibrate it against representative hardware data.
3. Learn a conditional brush/contact likelihood whose pressure trajectory
   depends on stroke phase, speed, curvature, brush loading, and local wet paint.
4. Stress-test long runs, checkpoint compatibility, and replay retention before
   raising policy depth or candidate count.
5. Add learned spatial/material latents only after pixel transition likelihoods
   are calibrated; retain deterministic decoding to material fields.
6. Replace the current deterministic composition ELBO approximation with an
   uncertainty-integrated higher-level latent model.

See `docs/history/CODEX_CONTINUATION_BRIEF.md` for the historical continuation
brief that accompanied this implementation snapshot.
