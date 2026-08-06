# Active-Inference Painter Development Audit

This is the historical engineering audit and change record for the early
prototype. Current milestone status is maintained in `planning/`.

## 2026-08-04 in-progress pass (amortized policy proposal; not accepted)

The working tree contains an interrupted Feature-D implementation of an
amortized candidate-policy proposal conditioned on canvas and relational
posterior beliefs. `proposal.py` defines normalized factored densities over the
declared mark/passage support, posterior-weighted maximum-likelihood training,
and divergence diagnostics against the hand-written sampler. `policies.py`,
`spatial_agent.py`, and `arm_agent_driver.py` contain mixture sampling,
post-planning training, checkpoint, attribution, and web-diagnostic plumbing.

Charter placement: this is a COMPUTATIONAL PROPOSAL distribution. It changes
which finite hypotheses are available for EFE-based posterior selection, but it
enters no likelihood, VFE term, EFE term, prior preference, or normalized policy
prior. The training target is the existing base-EFE painting-policy posterior,
so it is amortized inference rather than an external reward. The emission mix
defaults to zero; the hand-written sampler remains the live candidate source
unless an experiment explicitly raises it.

This pass is NOT complete and records no acceptance result. The source refers
to `tests/test_proposal.py`, but that file is absent. No candidate-count,
horizon, seed, mixture, posterior-mass, or top-action convergence experiment has
been run, and no importance correction is applied. The feature therefore does
not resolve finite-proposal bias or close `AI-111`. Basic import, sampling,
density, gradient, zero-mixture-parity, and mandatory-stop smoke checks passed
during the 2026-08-04 handoff review, but those checks are not a replacement for
the missing probabilistic test suite.

## 2026-08-03 pass (motion-manifold bootstrap of the composition hierarchy)

Problem: the bootstrap drew its marks from `PolicySampler._stroke`, i.e. iid
uniform strokes, and fed those transitions to the composition replay. It also
cleared the canvas whenever coverage exceeded 0.94 OR on every 24th transition
regardless of position in the episode. So the hierarchy's first and only
training signal was the exact iid scatter previously measured to leave the
compression gap flat (0.16 nat range), on canvases that were destroyed before a
complete organized one ever existed. That is the cold-start trap.

Two changes. (1) A new `motion_manifold.py` samples sweeps in JOINT space:
draw a start configuration, draw a joint-velocity direction weighted by a
declared family (yaw/pitch/roll/elbow-dominant, coordinated), and integrate it
along the contact surface by projecting the velocity onto the null space of the
numerical tip-depth gradient at every step. Integration stops at the manifold's
real edge - a joint limit, the contact depth band, the canvas border, or the
requested joint-space arclength. The FK tip path is projected to normalized
stroke space (inverting `stroke_execution.stroke_world_endpoints`, NOT
`world_to_pixel`) and re-expressed as a polyline `PassageLatent`, which is then
RE-DECODED through `PolicySampler.passage_actions` so the latent still generates
its own actions. (2) The canvas is cleared only at episode boundaries
(`bootstrap_episode_marks`, default 24, matching the previous cadence), and each
finished canvas is handed to the canvas/relational likelihood whole through the
new `spatial_agent.add_composition_canvas`.

MEASURED ACCEPTANCE RESULT: POSITIVE, AND OBTAINED AGAINST A PAINT-BUDGET
HANDICAP. Two fresh drivers, `checkpoint_path=None`,
`planner_state_kind='spatial_material'`, 96 marks, 4 episodes of 24, identical in
every respect except `bootstrap_generator`. `bootstrap_composition_train_steps`
= 900 per episode boundary (2700 gradient steps per arm; the first boundary has
fewer than `batch_size` replay entries, so three boundaries plus the final close
train).

| generator | margin | gap(boot) | gap(blank) | gap(shuffled) | gap(iid) | probe range | painted path | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `motion_manifold` | **+1.6337** | +1.5809 | -0.0528 | -74.42 | -294.48 | 296.06 | 22.754 | 0.1692 |
| `random_stroke` | **+1.0876** | +1.0067 | -0.0809 | -20.05 | -145.98 | 146.99 | 38.620 | 0.3344 |

The margin is `gap(bootstrapped) - max(gap(blank), gap(cell-shuffled))`.
Motion-manifold wins by 1.50x while laying 41% LESS painted path and half the
coverage, so the result is not a coverage comparison in disguise - the confound
runs against the winning arm. Both arms now score a POSITIVE gap on their own
bootstrapped canvases (+1.58 / +1.01), which is the cold-start trap being
escaped: neither did before.

THE DECLARED NULL MODEL TURNED OUT NOT TO BE THE BINDING ONE, and that is worth
recording. The design named the marginal-preserving cell shuffle as the tight
null, because `flat_log_likelihood` is provably identical between a canvas and
its shuffle. Measured, after 2700 gradient steps the shuffle is scored at -74 /
-20, i.e. it has become strongly out-of-distribution and is a far LOOSER null
than blank at -0.05 / -0.08. The `max(...)` in the criterion therefore selects
blank, which is the principled tight null anyway: on a blank canvas both
baseline family members sit exactly on the shared quantization floor, and
`tests/test_composition_hierarchy.py::test_blank_canvas_never_scores_positive_gap`
already pins that a blank canvas cannot score positive. Had the criterion been
bootstrapped-minus-shuffled it would have reported +76.0 vs +21.1 - a bigger
apparent win, dominated by out-of-distribution overconfidence, which is exactly
the failure mode that disqualifies the raw probe range. The blank-bounded margin
is the number to trust.

MEASURED NEGATIVE RESULT AT THE DEFAULT BUDGET, reported because a reader who
saw only the default run would wrongly conclude the feature failed. With
`bootstrap_composition_train_steps = 0` (today's cost exactly) the online
`composition_train_steps` budget supplies only ~50 gradient steps across a whole
bootstrap, and NEITHER arm discriminates structure at all:

| generator | margin | gap(boot) | gap(blank) | gap(shuffled) | probe range |
| --- | --- | --- | --- | --- | --- |
| `motion_manifold` | -0.1250 | -1.2012 | -3.2390 | -1.0762 | 22.45 |
| `random_stroke` | -0.2002 | -1.2058 | -3.4137 | -1.0056 | 5.62 |

Both margins are negative. The generator change alone does nothing measurable
without a declared gradient budget; that is why
`bootstrap_composition_train_steps` exists as an explicit field rather than an
incidental side effect, and why it defaults to 0 rather than being silently
switched on.

COST CORRECTION. The design estimated ~43 ms per composition gradient step at
batch 16. Measured here: 330-450 ms per step at `batch_size=32` (64 concatenated
samples), so the 2700-step acceptance run cost 883 s and 1221 s wall clock per
arm. `HierarchicalCanvasModel.training_loss` calls `relational_observations`,
which is a per-sample numpy round-trip, and that dominates. The sweep sampler is
NOT the cost: 49 sweeps produced 96 marks inside a 96-mark bootstrap whose wall
clock was ~50 s either way, and 0 fallback marks were needed. Measured directly
on `tests/test_arm_agent_driver.py`'s own 72-mark bootstrap configuration, two
runs per arm: `random_stroke` 19.26 s / 19.02 s, `motion_manifold` 17.91 s /
20.05 s - identical within noise, so the generator change does not move the
suite's wall-clock deadlines. (Those deadlines are nevertheless sensitive to
machine contention: one heavily loaded full-suite run timed out
`test_active_inference_driver_selects_stroke_at_low_coverage` at its 15 s
`wait_for_driver` budget while the same test passed three times in a row and the
whole suite passed 415/415 in 527 s uncontended. The deadlines were not
relaxed.)

SWEEP GEOMETRY, measured over the 49 bootstrap sweeps: mean |total turn| 2.485
rad, mean path/chord ratio 1.523, mean normalized path length 0.485. Over 300
standalone sweeps the distribution is mean |turn| 1.87 rad with 73% above 0.3
rad, and strong per-family variation (elbow-dominant 3.41 rad vs pitch-dominant
0.79 rad). The path/chord ratio is deliberately NOT pinned by a test: its median
is only 1.018, so most sweeps are near-straight in gross shape and a ratio
threshold would be a false pin. The embodiment signature lives in curvature
MAGNITUDE and its family structure.

MEASURED HAZARD THAT SHAPED THE DESIGN. A strictly single-joint sweep cannot
paint. From the canvas-centre IK pose, holding three joints fixed and sweeping
the fourth keeps the tip inside the near-contact band for only +-5.55 deg
(yaw/pitch), +-7.35 deg (roll), or a 2.8 deg window (elbow); the entire
admissible extent is a 0.168-normalized bounding box (0.022 for the elbow),
below the 0.20 minimum mark length. A family therefore only UP-WEIGHTS its joint
inside an otherwise coordinated sweep (`bootstrap_manifold_dominance_ratio =
0.22`), and `test_single_joint_sweeps_cannot_paint` pins the measurement so a
future simplification back to isolated joints is caught by a test rather than by
a blank canvas.

MEASURED CONFOUND, NOT FIXED. The contact depth band caps a sweep's usable
arclength at ~0.49 normalized, which segments into about two marks of ~0.24,
whereas `PolicySampler._stroke` samples lengths uniformly on [0.20, 0.60] (mean
0.40). At equal mark counts the manifold arm therefore lays substantially less
paint (22.75 vs 38.62 path). `bootstrap_manifold_min_mark_length` plus the
bare-mark fallback for single-segment sweeps removes the dwell-dab failure mode
but does not close this gap, so `paintedPathLength` and `episodeCoverageMean` are
reported per arm in the evidence block. A path-matched control arm was not run.

WHAT THIS DOES NOT FIX. Approximation register item 8 is AMENDED, not deleted.
The circularity - the structural prior is evaluated by a code fitted on the
agent's own history - survives. What moves is the ORIGIN of that history, from an
arbitrary iid noise source to the agent's own embodiment. Claiming more would be
the "rename an ordinary controller as active inference" failure the charter
forbids.

Charter placement of every added quantity: `ManifoldSweep` is a generative-process
sample (conventional FK plus a numerical Jacobian null-space projection, the same
category as `arm_control.ik_pose_for_canvas_point`, below the painting-policy
boundary) and carries a `declared_as` string saying it is not a decision
quantity; the fitted `PassageLatent` is the already-declared transition prior;
the bootstrap triples and whole canvases are training data for GM-PIX-TRANS and
Q-CANVAS/Q-REL; `bootstrap_composition_train_steps` is a gradient budget that
appears in no objective; and `compositionBootstrap` is evidence that nothing in
the model reads. The terminal preference, `composition_gap_precision`, and the
gap's definition are untouched.

## 2026-08-03 pass (per-modality precision beliefs, gap-progress stopping)

Problem: seven hand-set constant precisions weighted seven EFE modalities that
were stated in incommensurable units, and the terminal coverage target 0.87 was
a hand-picked stopping criterion. Replaced the constants with Gamma precision
BELIEFS updated by the reference implementation's Chapter 10 rule, normalized
every modality to nats per observation channel, and added a gap-progress factor
to the stop POLICY PRIOR.

Attribution is exact rather than approximate: each belief's rate is seeded at
`beta0 = alpha0 / (declared constant)`, and an unobserved belief returns that
constant bit-identically. `precision_beliefs_enabled=False`,
`modality_precision_beliefs_enabled=False`, and
`modality_normalization_enabled=False` therefore reproduce the previous
arithmetic exactly, so any measured difference is attributable to the mechanism
rather than to a reparameterization.

MEASURED NEGATIVE RESULT, reported rather than tuned around. The stated hope was
that a composition modality whose contributions became discriminative across
candidates would gain precision while the arrangement-blind terminal-coverage
modality lost it, out of the same rule. It does not happen. Over six real
planning rounds (spatial driver, 16 candidates, `F` = the brush-preparation
VFE):

| modality | gamma | prior | contribution spread (std) |
| --- | --- | --- | --- |
| `terminal_coverage` | 1.0003 | 1.00 | 196.3 |
| `transition` | 0.8805 | 1.00 | 0.331 |
| `composition_gap` | 0.9859 | 1.00 | 0.190 |
| `observation_ambiguity` | 1.0106 | 1.00 | 0.0314 |
| `motor_proprioceptive` | 0.9973 | 1.00 | 0.0301 |
| `policy` | 0.3513 | 0.35 | -- |

The most discriminative modality gained the least. Cause, measured directly: the
rule responds to F/G agreement only inside a narrow intermediate band of a
modality's own absolute scale and saturates to exactly the prior mean outside it.
Sweeping one fixed 24-element shape across scale with fixed agreeing
`F = 0.5 G`, gamma is 1.0006, 1.0112, 1.0814, 1.4602, 1.4823, 1.0880, 1.00000,
1.00000 at contribution std 0.034, 0.144, 0.357, 0.721, 1.44, 3.56, 14.4, 144 --
non-monotone, peaking near std ~1. No `beta0` was tuned to manufacture the hoped
ordering; a mechanism that genuinely rewarded discriminativeness would be a
different inference problem and is left as its own feature. Recorded as spec
register item 26.

DEFECT FOUND AND FIXED while making the modalities commensurable. The
moment-matched terminal coverage forecast could produce a Beta concentration
below 1, and the resulting `digamma(alpha -> 0)` made the exact Beta-Beta KL
diverge: measured 53248.18 nats at coverage mean 1e-4 with std 2.29e-3.
Restricting the forecast to the interior-unimodal family (both concentrations
>= 1, mean preserved by a common rescale) caps that at 892.24 and leaves every
well-conditioned forecast BIT-UNCHANGED (246.6508 -> 246.6508 at mean 0.05 var
1e-4; 37.2720 -> 37.2720 at 0.5/1e-6; 4.4849 -> 4.4849 at 0.87/1e-8;
0.7181 -> 0.7181 at 0.87/1e-4). This is a declared forecast-family restriction,
not a clamp on risk, so `risk = -entropy - pragmatic` still holds exactly.

That fix is what makes the new unit tripwire a genuine test rather than a
tautology. On a 24-candidate blank-canvas set, the worst modality's mean
absolute contribution is 20.8x the median active modality with the floor and
1592x without it; the worst single candidate is 329x with and 3.8e4x without.
Both directions are asserted in `tests/test_modality_units.py`.

It also changes behaviour exactly in the low-coverage cold-start regime that the
composition features exist to attack, so `terminal_forecast_concentration_floor
= 0.0` restores the previous family and the difference needs A/B measurement
before being treated as settled. Note also that on a fully blank canvas every
floored forecast saturates at `alpha = 1`, making terminal risk scale-invariant
there (identical at belief logvar -4, -6, -8, -10).

MOTOR NORMALIZATION IS A 27x REWEIGHTING, not a tidy-up, and it has a measured
cost. The three motor EFE terms were raw sums over 27 named proprioceptive
outcomes. Measured on one representative stroke across the five declared
realization kinds, `G_motor` falls from `-4.4085, 1.4664, 1.0571, -4.4085,
-4.4085` to exactly one twenty-seventh of each, so the cross-kind spread drops
from 5.875 nats to 0.2176 nats -- now comparable to the material modalities'
0.19-0.33 nat per-candidate spreads instead of 20x larger. Inside
`motor_realization_log_evidence` at the web runtime's 0.35 policy precision the
modal realization is unchanged, but the conditional motor posterior flattens from
`[0.305, 0.039, 0.045, 0.305, 0.305]` (entropy 1.353) to
`[0.206, 0.191, 0.192, 0.206, 0.206]` (entropy 1.609, against a uniform maximum
of `log 5 = 1.609`). Per-channel normalization therefore makes motor realization
selection nearly indiscriminate at that precision. Gated by
`modality_normalization_enabled` so the difference stays attributable, and
deliberately NOT compensated by raising `motor_modality_precision`, which would
be exactly the hand-tuning this feature exists to remove.

Gap-progress stopping is a POLICY PRIOR, never a reward. `log p(stop)` is now a
product of the existing coverage sigmoid and
`logsigmoid(-s * E[dGap]/sd[dGap])` over a Gaussian random-walk belief on the
per-mark compression-gap increment. Both factors are bounded above by zero, the
progress factor is exactly `0.0` for continuations and for an unobserved belief
(so the `log(0.5)` midpoint identity is preserved verbatim), and containment is
enforced mechanically: a test asserts every field of every EFE component is
bit-identical after feeding the belief a large increment. `preferences.py` was
not touched, and `tests/test_preferences.py` now pins that the terminal Beta
concentrations are bit-identical after enough precision updates to move every
gamma.

Also landed: a de-duplication of six verbatim copies of the precision block and
total assembly into `SpatialExpectedFreeEnergy._modality_weights` /
`_assemble_total` (plus the `efe.py` pair), and one shared
`policies.policy_posterior_from_efe`.
`tests/test_epistemic_policy_selection.py` now calls that shared helper instead
of a hand-rolled softmax, so a test can no longer pass while exercising a
formula production does not use, and its batch-vs-single guard runs under all
four combinations of the precision-belief and normalization flags.

## 2026-08-03 pass (compression-gap baseline family)

Problem: the compression gap's opponent was a single per-image per-channel iid
Gaussian. On low-amplitude material channels it saturates against
`SIGMA_FLOOR`, and on structured canvases it is a weak opponent, so the gap
conflated "locally smooth" with "compositionally organized". Measured
demonstration of the defect, on a `CompositionHierarchy` trained 600 steps on a
50/50 mixture of band canvases and soft blobs (one model, two evaluation
inputs): the old gap scored a soft blob at +1.4706 and structured bands at
+1.4789, a separation of 0.0083 nats. The old gap was effectively blind to the
difference between long-range structure and a single smooth blob.

Definite changes made:

- The baseline is now a declared, hand-written, parameter-free BASELINE FAMILY
  and the gap is measured against the BEST member, so it only credits structure
  no member can explain: `gap(s) = ELBO_hier(s) - max_m log p_m(s)`. Member 1 is
  the existing iid-cell Gaussian, unchanged. Member 2 is a fixed 3x3
  hollow-neighbourhood local Markov code: each cell is predicted by the mean of
  its eight neighbours EXCLUDING itself (hollow, so it is a genuine predictive
  code and not the identity), replicate padding, `groups=channels` so channels
  never mix, and a per-image per-channel residual variance floored at the same
  `SIGMA_FLOOR`. Neither member has a learnable parameter and neither enters
  `training_loss`: the preference is not fit to outcomes.
- `flat_log_likelihood` was duplicated verbatim in `composition.py` and
  `canvas_hierarchy.py` — a latent divergence bug. The shared implementation now
  lives once in `composition.py`; `canvas_hierarchy` imports it, and both model
  classes delegate. A new test pins that the production `HierarchicalCanvasModel`
  and the module function cannot diverge, because every other invariant test
  exercises the vestigial `CompositionHierarchy` and would otherwise pass while
  the live agent kept the old opponent.
- Declared flag `composition_local_baseline_enabled: bool = True`. False
  restores the iid-only baseline exactly, so the difference the family makes is
  attributable and measurable; a test asserts both flag states against identical
  weights. The flag is deliberately NOT added to the checkpoint architecture
  metadata (it changes a preference readout, not tensor architecture, and that
  dict is compared by exact equality), and the donut kernel is built inline
  rather than registered as a buffer, so existing checkpoints keep loading.

Measured, 1500 Adam steps at `composition_lr` on `HierarchicalCanvasModel`,
16x16x6 fields, realistic per-channel scales `[0.006, 0.004, 0.003, 0.8, 0.6,
0.9]`, mean over 32 held-out images, nats per cell-channel (ceiling 2.9931):

| case | ELBO | iid | donut | max | OLD gap | NEW gap |
| --- | --- | --- | --- | --- | --- | --- |
| blank | 2.8019 | 2.9931 | 2.9931 | 2.9931 | -0.1912 | -0.1912 |
| bands | 2.5630 | 1.8597 | 2.4619 | 2.4619 | +0.7033 | +0.1011 |
| iid scatter | -0.6009 | 1.9436 | 1.9176 | 1.9436 | -2.5445 | -2.5445 |
| soft blob | 1.9503 | 1.9836 | 2.8352 | 2.8352 | -0.0333 | -0.8849 |
| lowpass noise | 2.5637 | 2.4370 | 2.8548 | 2.8548 | +0.1267 | -0.2911 |
| bands shuffled | -2.3963 | 1.8597 | 1.8396 | 1.8599 | -4.2559 | -4.2562 |

Low-pass noise was a FALSE POSITIVE under the iid-only baseline (+0.1267) and is
now negative (-0.2911). Blank is exactly unchanged (both members tie on the
shared ceiling), and iid scatter and the shuffle are unchanged to four decimals
(the iid member wins the family maximum there), so the three original invariants
are preserved rather than merely still passing. On this bands-only-trained model
the soft blob was already marginally negative under the old baseline (-0.0333);
the large false positive appears on the mixture-trained model quoted above,
which is the adversarial case and the basis of the new test.

On the mixture-trained `CompositionHierarchy` that the new tests use (600 steps,
50/50 bands and soft blobs, so the hierarchy had every opportunity to model the
blobs), the same comparison, mean over 32 held-out images:

| case | ELBO | iid | donut | OLD gap | NEW gap |
| --- | --- | --- | --- | --- | --- |
| bands | 2.4801 | 1.0012 | 2.1650 | +1.4789 | +0.3151 |
| soft blob | 2.7737 | 1.3031 | 2.8147 | +1.4706 | -0.0410 |
| lowpass noise | 1.6704 | 2.2194 | 2.8520 | -0.5490 | -1.1816 |

bands-minus-blob separation: OLD +0.0083, NEW +0.3561. That is the core result.
The old gap could not tell a soft blob from structured bands at all; the new one
separates them by 0.36 nats and puts the blob on the correct side of zero.

The ordering bands > blank > scatter > shuffled holds identically under both
baselines, and the discriminative range on structured input does NOT collapse:
bands-to-shuffled retains 88% (4.96 -> 4.36 nats). But the POSITIVE headroom
compresses ~86% (+0.7033 -> +0.1011), because the donut member takes 0.602 nats
on structured input while the ELBO stays capped at 2.9931 by `LOGVAR_FLOOR`.
This is inherent to raising the baseline: the ceiling is shared, so a stronger
opponent necessarily eats headroom.

WARNING, two illegal compensations. Do NOT raise `composition_gap_precision` to
recover the old effect size: that is tuning a preference weight to recover a lost
outcome, i.e. fitting a preference to data, which the charter forbids. Do NOT
lower `SIGMA_FLOOR`: it raises the ceiling for both members and the ELBO
simultaneously and invalidates every recorded measurement. If the composition
term's influence is judged too small against terminal coverage risk (which spans
289 nats), the legal responses are to state it as a finding, to make the
precision an explicit precision BELIEF with a declared update rule, or to
escalate to `AI-110`. Gap values in previously saved telemetry are NOT comparable
across this change.

Scope limits, stated so they cannot be misread. The gap still uses ALL SIX
material channels, unchanged; the derived-channel double-count (register item 3)
is untouched but is now amplified in relative terms, because the donut member
earns its entire advantage on the high-amplitude channels 3-5 that carry that
double count. Narrowing the gap to the four independent channels is the right
eventual fix but must be a separate change, since it moves both the ELBO and the
baseline and would make this before/after unattributable. Feature B also does
NOT address `AI-110`: it strengthens the opponent, which is orthogonal to the
self-referential closed loop in which the hierarchy trains online on the stream
it evaluates. A smaller measured gap is not evidence that loop is closed. The
recorded cold-start trap is likewise untouched.

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
- The former tracking-error paint gate has since been removed. A loaded brush
  deposits from physical pressure-bearing contact, and contact-loss forecasts
  now count only actual pressure loss.

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
