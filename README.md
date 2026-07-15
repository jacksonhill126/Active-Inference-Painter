# Active-Inference Painter — Python v0.1

This is a clean Python/PyTorch starting point for the painter architecture.
It implements one feature rigorously: **policies are inferred so that the eventual `stop` state is expected to occur near a preferred material coverage of roughly 80–90%**.

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
contact patch with hard support, enriched with four physical behaviors. All of
this lives in the generative *process*, below the painting-policy boundary; the
learned transition model observes the resulting canvas transitions and adapts.
It is oil: paint does not dry within a session and the brush does not run out
mid-stroke — deposition is consistent along a mark and the canvas keeps its
wetness (there is no wetness decay).

- **Brush loading.** The stroke's `amount` sets how heavily the brush is loaded,
  which scales deposited thickness/opacity uniformly along the mark (a fuller
  brush lays thicker paint). It never depletes or thins toward the end
  (`brush_load_min`/`brush_load_max`).
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
empirical policy prior), since a round brush over a short span reads as a dab.

Loading and carried tone are per-stroke state set on pen-down from the action,
so each stroke stays a deterministic function of the canvas and the action (no
cross-stroke brush memory the learned model cannot see). Brush tilt relative to
the canvas normal is not yet modeled; the round footprint is oriented only by
travel direction.

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

The immediate `stop` policy is always available. Continuation policies are sampled as one or more strokes followed by `stop`, so the planner can anticipate coverage overshoot rather than merely checking a threshold afterward.

## Composition hierarchy (compression gap)

Spatial mode carries a hierarchical composition layer (`canvas_hierarchy.py`):
a spatial canvas latent with a learned decoder over the material fields. The
single declared structural preference over terminal canvases is

```
p*(s_T) ~ exp(composition_gap_precision * gap(s_T))
gap(s)  = ELBO_hierarchical(s) - log p_flat(s)      [nats per cell-channel]
```

where the flat baseline is the best context-free iid-cell Gaussian for that
specific image, and both codes share one quantization floor. The hierarchy
pays for its latent through the KL term, so a blank canvas scores ~0, iid
noise scores <= 0, and only canvases whose parts mutually predict each other
score positive. No content term (balance, contrast, subject) appears anywhere:
the preference references only the hierarchical model's explanatory advantage.
The hierarchy trains online alongside the dynamics ensemble; the per-policy
`composition_gap`/`composition_risk` components and the current belief gap are
logged in diagnostics. Because the gap is evaluated on every candidate
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
  regions of the current belief. This is an empirical policy prior over the
  candidate set; scoring remains pure expected free energy.
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
  scores the predicted consequences. This is a policy prior over multi-mark
  latent trajectories, not fine-tuning and not an aesthetic reward.
- **Passage-plan proposals.** When the planning horizon is deep enough, a
  declared fraction (`passage_plan_proposal_mix`) of candidates are generated
  from a slower `PassagePlanLatent` over multiple passage latents. The plan
  carries a slowly evolving center, direction, turn, tone, and material amount;
  its child passages generate the actual marks. The plan is still only a
  policy prior, and every candidate still terminates in immediate `stop`.
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
- **Per-modality precisions.** `terminal_risk_precision`,
  `ambiguity_precision`, `transition_precision`,
  `motor_proprioceptive_risk_precision`, and
  `motor_proprioceptive_ambiguity_precision` weight the outcome modalities of
  expected free energy explicitly. Logged EFE components are
  precision-weighted contributions.
- **Bootstrap ensemble training.** Each ensemble member trains on its own
  Bernoulli-masked subset of every batch (`ensemble_bootstrap_probability`),
  keeping member disagreement usable as an approximate parameter posterior;
  calibration tests cover held-out z-scores and off-distribution disagreement.

## Spatial material planner mode

The default planner still runs on six global canvas summaries for compatibility
and speed. An explicit spatial material planner can be enabled with
`planner_state_kind="spatial_material"` or:

```bash
python -m active_painter.web_server --planner-state-kind spatial_material
```

Spatial mode performs an initial dynamics bootstrap before the URL is printed.
For a quick no-bootstrap smoke test, add
`--driver-bootstrap-transitions 0 --driver-bootstrap-train-steps 0`.
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
inference core; Three.js only renders the arm, canvas, controls, and telemetry.
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

See `CODEX_TASK.md` for a concrete continuation brief.
