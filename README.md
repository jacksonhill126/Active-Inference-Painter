# Active-Inference Painter — Python v0.1

This is a clean Python/PyTorch starting point for the painter architecture.
It implements one feature rigorously: **policies are inferred so that the eventual `stop` state is expected to occur near a preferred material coverage of roughly 80–90%**.

It deliberately separates:

- **Generative process**: stochastic paint deposition on a canvas with independent thickness, wetness, visible tone, and material coverage.
- **Generative model**: a learned ensemble transition density over latent canvas states plus an explicit observation likelihood.
- **Posterior inference**: variational state estimation by minimizing variational free energy.
- **Preferences**: a terminal Beta density over coverage, applied only when a policy terminates in `stop`.
- **Policy inference**: posterior over policies from expected free energy.
- **Execution**: arm realization stays below policy selection, but the planner can now compare declared first-stroke motor realization latents such as Cartesian IK, joint-space splines, and elbow-led arcs through predicted canvas and proprioceptive outcomes.

No hand-written aesthetic reward is used. Coverage is not inferred from visible color: white paint on white ground still adds thickness and coverage.

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

The immediate `stop` policy is always available. Continuation policies are sampled as one or more strokes followed by `stop`, so the planner can anticipate coverage overshoot rather than merely checking a threshold afterward.

## Composition hierarchy (compression gap)

Spatial mode carries a hierarchical composition layer (`composition.py`): a
latent code `z` with a learned decoder over the spatial material fields. The
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
  passage kinds are parallel mark bands and chained mark phrases; each passage
  still terminates in `stop`, and expected free energy scores the predicted
  terminal consequences. This is a policy prior over multi-mark latent
  trajectories, not fine-tuning and not an aesthetic reward.
- **Passage-plan proposals.** When the planning horizon is deep enough, a
  declared fraction (`passage_plan_proposal_mix`) of candidates are generated
  from a slower `PassagePlanLatent` over multiple passage latents. The plan
  carries a slowly evolving center, direction, turn, tone, and material amount;
  its child passages generate the actual marks. The plan is still only a
  policy prior, and every candidate still terminates in immediate `stop`.
- **Embodied motor realization priors.** During arm-driven planning, top
  canvas candidates are expanded into declared first-stroke
  `MotorPrimitiveLatent` alternatives (`cartesian_ik`, `joint_spline`,
  `elbow_pivot` by default). Each realization is forecast through the arm,
  contact, and canvas simulator before posterior policy selection. The chosen
  primitive contributes separate proprioceptive EFE terms: motor risk is a
  homeostatic prior over current, torque, acceleration, limit proximity, and
  target-error observations; motor ambiguity is a likelihood-entropy proxy
  from contact loss, pressure variance, path covariance, and tracking
  uncertainty. The selected primitive is also encoded into replay transitions
  and learned rollouts as motor-conditioned action channels, so the learned
  transition likelihood is `p(s_next | s, stroke, motor_realization)` rather
  than stroke-only. Hard joint/current/workspace limits remain external safety
  constraints, and no motor-ease reward is introduced.
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
The web renderer displays the canvas on a neutral gray ground so both white and
black paint are visible. Tone support is unconstrained by default; use
`--stroke-tone-prior black`, `--stroke-tone-prior white`, or
`--stroke-tone-prior random` to set the policy sampler's tone support. In
random mode, candidate geometries are proposed as matched black/white policy
alternatives where the candidate budget allows, so tone is selected by the EFE
posterior over predicted material consequences rather than by an unpaired
coin flip.

In that mode the driver evaluates policies over:

- `SpatialCanvasState`: explicit `thickness`, `wetness`, `black_mass`,
  observed-tone, ground-contrast, and material-coverage fields on a
  low-resolution grid. Observed tone, contrast, and coverage are deterministic
  fields derived from primary material quantities plus the canvas substrate
  tone, not independent reward variables.
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
  coverage is derived from the spatial thickness field.
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

It shows a conventional 4-DOF arm plant, a vertical paint canvas, soft wrist
contact, pressure-dependent brush width, motor telemetry, and material coverage.
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

1. Replace the 2-D brush executor with a MuJoCo arm/canvas generative process.
2. Preserve the same `StrokeAction` and `stop` semantics at the painting level.
3. Expand from explicit spatial material fields to learned spatial/material latents.
4. Add a conditional stroke/contact decoder whose pressure trajectory depends on stroke phase, speed, curvature, brush state, and local canvas state.
5. Increase planning depth with batched policy rollouts.
6. Add slower global composition latents and infer them without hand-written aesthetic scores.

See `CODEX_TASK.md` for a concrete continuation brief.
