# Active-Inference Painter Audit Notes

## 2026-07-06 third pass (reachability and stroke realization)

Problem: achievable material coverage was capped at ~65% of canvas area
(0.82 execution margin times the proposal margins), strictly below both the
0.87 terminal coverage preference and the 0.70 stop-prior midpoint, so
paintings could never legitimately terminate (observed: 329 strokes, zero
completed paintings, coverage asymptoting ~0.62).

Definite changes made, verified by forecast feasibility probes at all canvas
edges and a dense-tiling ceiling test (now 0.871, edge to edge):

- The normalized stroke space maps to 98% of the canvas
  (`CANVAS_REACH_FRACTION`); genuinely unreachable strokes are rejected per
  candidate by the motor feasibility forecast, not by a blanket margin. The
  whole plane is statically reachable by IK.
- Stroke timing is adaptive (`adaptive_stroke_timing`): approach time scales
  with the tip-to-start distance and paint time bounds the sweep speed.
  Fixed timing previously made distant starts unreachable within the approach
  phase and swept long strokes faster than the servo could track — the cause
  of marks splitting into a start dash and an end blob.
- The contact-aware controller travels with the brush pulled off the canvas
  and commands bounded Cartesian carrot steps while far from the reference:
  joint-space interpolation toward distant targets otherwise swings the tip
  through the canvas plane, where the overtravel safety rollback wedged the
  arm in place for the whole stroke (the mechanism behind edge strokes
  reporting total contact loss despite exact static IK).
- The press/paint references target slight bushing penetration
  (distance + 0.2, well under the 0.5 overtravel limit) so contact is robust
  to servo depth undershoot at extended reach, instead of relying on the
  0.08 near-surface gate at exactly zero deflection.
- The direct-vs-aware controller test now bounds the aware controller's
  deliberate paint gating rather than comparing against an ungated baseline
  that paints while off-track.

These are all conventional execution-layer changes beneath the painting
policy boundary; no EFE term changed in this pass.

## 2026-07-06 second pass (composition hierarchy, brush physics)

Definite changes made:

- Added `composition.py`: a latent composition hierarchy over spatial
  material fields with a declared structural terminal preference
  `p*(s_T) ~ exp(kappa * gap(s_T))`, `gap = ELBO_hier - log p_flat`, where the
  flat baseline is the best per-image iid-cell Gaussian and both codes share a
  quantization floor (`SIGMA_FLOOR`) to avoid differential-density artifacts.
  The hierarchy pays for its latent code via the KL term, so blank canvases
  score ~0 and iid noise scores <= 0; tests pin structured > shuffled > blank
  ordering. This is deliberately contentless: it is the single abstract prior
  over model structure, not an aesthetic score.
- The gap enters `SpatialExpectedFreeEnergy` as `composition_risk =
  -kappa * E[gap(s_T)]` over member particles, logged per policy and included
  in the total. With `composition_gap_precision = 0` (or no hierarchy) the
  term is exactly zero and prior behavior is unchanged.
- The hierarchy trains online (`composition_train_steps` per planning cycle)
  on replayed spatial states alongside the dynamics ensemble.
- Brush footprint in the arm simulator is now a compact contact patch defined
  in world units (`VerticalCanvas.brush_radius_world`): a hard-support disc
  with a thin smoothstep rim. This fixes two defects: the deposit sigma was
  previously capped at ~1.15 px regardless of canvas size (hairline marks on
  a 256 px canvas), and an intermediate Gaussian-footprint version let
  infinite tails accumulate past the coverage threshold wherever the brush
  dwelled, so mark size grew without bound in time and edges were fuzzy
  coverage gradients. With hard support, mark width is set by pressure alone
  and is dwell-invariant. Contact `brush_width_px` derives from the same
  world radius. Stroke width maps to pressure with gain 0.42 over a widened
  width range, and the policy sampler draws widths log-uniformly from
  [0.03, 0.30] for mark-scale variety.

Runtime/latency changes in this pass:

- Model learning (dynamics ensemble + composition hierarchy) now runs in the
  planner thread after the plan is published, overlapping the selected
  stroke's execution instead of extending the planning gap. Training
  previously sat on the planning critical path at 6-19s per cycle and was
  the dominant cause of slow stroke cadence. `_start_background_plan` will
  not launch a new planner thread while the previous one is still training,
  so training never races policy evaluation.
- Brush deposition computes only within the compact contact patch's bounding
  box instead of the full canvas (a ~250x cut for a 256 px canvas), which
  also makes execution forecasts ~6x cheaper.
- The web server caps torch intraop threads (small models lose more to
  oversubscription against the sim/render threads than they gain from extra
  cores), and the belief compression-gap diagnostic is cached per planning
  cycle rather than recomputed on every UI poll.
- Net effect measured in the live spatial web runtime: median stroke-to-stroke
  gap ~8.7s (from ~20s), planning 5-6s of which a large share is GIL
  contention with the 240 Hz sim loop. The remaining structural improvement
  is planning during stroke execution against the execution forecast.

Provisional notes for this pass:

- The compression gap is evaluated with a deterministic (mean-latent) ELBO
  and on member terminal means, not integrated over predictive state
  distributions.
- `composition_gap_precision` is a hand-set constant; deriving stopping from
  compression progress (replacing the coverage-band preference) remains the
  next architectural step once the gap signal proves informative in runs.
- The mark-event slot summary (`spatial_hierarchy.py`) remains diagnostic
  only and is unrelated to the composition hierarchy.
- Mid-stroke contact loss can split executed marks into a start dash and an
  end blob (visible at all widths); this predates the brush changes and lives
  below the painting-policy boundary in the stroke controller/timing.

## 2026-07-06 pass

Scope checked and reworked:

- Uncertainty propagation through policy rollouts.
- Terminal coverage variance aggregation in spatial mode.
- Ensemble dispersion as an approximate parameter posterior.
- The procedural minimum-stop-coverage veto in the arm driver.
- Candidate-policy proposal distribution.
- Summary/spatial EFE code duplication.

Definite changes made:

- Policy rollouts with a learned ensemble are now member-wise trajectory
  samples (each member propagates its own particle), so parameter uncertainty
  accumulates over the policy horizon. Previously rollouts re-evaluated the
  ensemble at the moment-matched mixture mean each step and discarded
  trajectory-level disagreement. Single-step numerics are unchanged; the
  epistemic identity `transition_risk + transition_ambiguity = -I(theta; s')`
  still holds and remains tested.
- Spatial terminal coverage variance previously assumed independent grid
  cells (sum of cell variances / N^2), underestimating variance because a
  stroke correlates neighboring cells. It is now across-member variance of
  aggregate coverage plus mean within-member cell-wise delta-method variance.
- The driver's procedural stop veto below `minimum_stop_coverage` was replaced
  by a declared policy prior: `log p(stop-first) = logsigmoid(sharpness *
  (coverage - midpoint))` added inside the policy softmax. Stop remains
  admissible at all coverages; prior demotions are surfaced via the existing
  `lastStopBlocked` diagnostic (now meaning "stop had lowest EFE but the prior
  demoted it").
- Ensemble NLL training now applies per-member Bernoulli bootstrap masks
  (`ensemble_bootstrap_probability`) so members do not train on identical
  batches and collapse together. New calibration tests check held-out
  z-scores and off-distribution disagreement.
- Candidate strokes in spatial mode can start preferentially in low-coverage
  belief regions (`proposal_low_coverage_mix`), declared as an empirical
  policy prior over the candidate set.
- Per-modality precision weights (`terminal_risk_precision`,
  `ambiguity_precision`, `transition_precision`) are declared in config and
  applied where each term enters expected free energy; defaults of 1.0
  preserve prior behavior. Logged components are precision-weighted.
- Candidate evaluation is batched across policies and ensemble members (one
  network pass per rollout step instead of one per policy per step), and
  `downsample_mean` is vectorized. Shared terminal-preference, Beta
  moment-matching, and support-projection code moved to `efe_common.py`.

Still provisional after this pass:

- Observation ambiguity is integrated over parameter uncertainty by averaging
  over member particles, but still evaluated at member means rather than
  integrating over each member's predictive state distribution.
- The initial belief variance still enters rollouts only through the terminal
  coverage variance of immediate-stop policies; member particles start from
  the belief mean rather than belief samples. Sampling initial particles per
  member is the natural next step but adds posterior-comparison noise.
- The stop prior's sharpness and midpoint are hand-chosen constants. The
  architecture goal is to derive stopping from a higher-level model
  (compression-progress-style), at which point this prior should be replaced.
- The moment-matched Beta terminal family and its concentration clamps are
  unchanged; a logit-normal family remains the candidate replacement.

## 2026-06-30 pass

Scope checked:

- Expected-free-energy term definitions and signs.
- Terminal coverage preference and `stop` policy availability.
- Observation ambiguity and differential-entropy unit effects.
- Learned-dynamics ensemble uncertainty fixture.
- Planner/control boundary for the arm runtime.

Definite fixes made:

- Replaced raw observation differential entropy in EFE with excess entropy above
  the dry-canvas likelihood baseline. This avoids rewarding extra policy steps
  merely because continuous entropy is negative in the chosen observation units.
- Added explicit `terminal_entropy` logging and made terminal risk satisfy
  `KL[q(C_T | pi) || p*(C_T)] = -H[q(C_T | pi)] - E_q log p*(C_T)` by
  construction.
- Added guard tests for policy sampler terminal-stop invariants, observation
  ambiguity baseline behavior, terminal-risk decomposition, and no base-entropy
  step reward.

Still provisional:

- The transition epistemic term treats learned next-state predictions as a
  transition-outcome modality with flat preferences. This is marked as an
  approximation, but it needs a written generative-model derivation before it
  should be considered rigorous.
- The ensemble is used as an approximate posterior over transition parameters;
  no calibration test currently shows that ensemble variance is a well-calibrated
  posterior uncertainty.
- Transition information gain is computed with a moment-matched diagonal
  Gaussian mixture approximation and remains sensitive to latent-state units and
  dimensionality.
- Terminal coverage forecasts are moment-matched to a Beta distribution. This is
  tested for broad target-band behavior, but the approximation needs a derivation
  and stress tests over variance regimes.
- The arm runtime observes exact simulator summary state in places where a real
  system would need an observation model; this is a simulator shortcut.

Primary references used for the decomposition audit:

- Friston et al., "Active inference and epistemic value" (2015).
- Sajid et al., "Active inference, Bayesian optimal design, and expected utility"
  (2021).

## 2026-06-30 second pass

Additional scope checked:

- Policy object invariants independent of `PolicySampler`.
- Terminal-risk behavior over different terminal forecast variances.
- UI/diagnostic wording around fixed policy precision.
- Contact realization boundary in the arm driver.

Definite fixes made:

- `Policy` now rejects any `stop` action before the final action. Previously a
  manually constructed policy could contain `stop`, then more strokes, then a
  final `stop`; EFE evaluation would silently ignore the suffix after the first
  stop.
- The web UI now labels `policy_precision` as fixed "Policy precision" rather
  than "Policy precision belief", because no precision posterior is inferred.

Additional provisional findings:

- Terminal forecast KL behaves sensibly near the target band for low/moderate
  variance, but broad forecast variance can shift the KL minimum away from the
  preference mode under the current moment-matched Beta approximation. This may
  be defensible, but it needs stress tests and a derivation.
- `_coverage_beta_approximation` clamps Beta concentration to `[2, 1e6]`. Those
  clamps are numerical stabilizers and should be justified or replaced with a
  more principled terminal forecast family.
- `ObservationModel.ambiguity` now avoids negative unit-dependent base entropy,
  but it evaluates ambiguity at the predicted mean state rather than integrating
  over the full predicted state distribution.
- `pose_for_execution` realizes a Cartesian stroke with conventional pressure
  shaping that depends on amount, phase, width, and speed. It does not yet
  condition contact predictions on a learned brush/contact state or model
  uncertainty, so it should remain classified as a provisional controller below
  the painting policy boundary.

## 2026-06-30 runtime behavior update

- The web runtime now treats `stop` as completion of the current painting rather
  than a permanent halt. It increments a painting counter, optionally archives
  the canvas image, clears the canvas, resets the arm pose, and restarts the
  active-inference driver.
- By default, every fifth completed web painting is saved to
  `runs/web/painting_####.png`. This is runtime bookkeeping outside the painting
  policy; it does not alter the EFE objective.
