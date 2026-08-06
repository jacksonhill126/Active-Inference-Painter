from __future__ import annotations

from dataclasses import dataclass


SUMMARY_PLANNER_STATE_KIND = "summary"
SPATIAL_MATERIAL_PLANNER_STATE_KIND = "spatial_material"
SUMMARY_PLANNER_DEPRECATION = (
    "planner_state_kind='summary' is an obsolete six-aggregate compatibility "
    "fixture. It is non-spatial, predictively insufficient for image-making, "
    "and must not be treated as a highest-level painting belief. Use "
    "'spatial_material' only as the current provisional low-level material "
    "baseline while the learned perceptual hierarchy is implemented."
)


@dataclass(slots=True)
class PainterConfig:
    canvas_size: int = 48
    thickness_scale: float = 0.005
    # Material coverage is occupied canvas area, not accumulated thickness.
    # A pixel counts once when deposited paint exceeds this presence threshold.
    paint_presence_threshold: float = 0.0001
    canvas_ground_tone: float = 0.34
    paint_deposition_base_rate: float = 0.16
    paint_deposition_pressure_rate: float = 0.64
    oil_surface_opacity_thickness: float = 0.002
    oil_wet_pickup_fraction: float = 0.18
    base_observation_std: float = 0.008
    smear_observation_std: float = 0.02
    # Brush paint-handling model (generative process, below the policy
    # boundary). Oil paint does not dry within a session: the canvas keeps its
    # wetness (there is no wetness decay). Brush load is a
    # persistent normalized reservoir. The selected mark's `amount` scales its
    # requested deposition, while the current load scales how much of that
    # consequence can be realized. Contact consumes load; reloading resets it
    # to full and resets its mixture to the selected brush color.
    # `amount` 0 -> brush_load_min, 1 -> brush_load_max deposition scale.
    brush_load_min: float = 0.55
    brush_load_max: float = 1.45
    # Mean deposited-film thickness that consumes one full normalized fresh
    # load. Provisional until measured with repeated real brushmarks.
    brush_load_capacity_thickness: float = 1.6
    # Compact brush-loading generative-model assumptions. These values are
    # provisional simulation priors, not claimed hardware calibration.
    brush_initial_load_std: float = 0.03
    brush_reload_load_std: float = 0.025
    brush_reload_mixture_std: float = 0.02
    brush_load_process_std: float = 0.06
    brush_mixture_process_std: float = 0.10
    brush_belief_depletion_per_mark: float = 0.32
    brush_mark_amount_preference_std: float = 0.14
    brush_mark_pigment_preference_std: float = 0.10
    brush_deposition_likelihood_std: float = 0.08
    brush_mixture_likelihood_std: float = 0.08
    brush_reload_policy_prior: float = 0.08
    brush_policy_precision: float = 1.0
    brush_material_risk_precision: float = 1.0
    brush_pigment_risk_precision: float = 1.0
    brush_ambiguity_precision: float = 1.0
    # Directional (swept-capsule) footprint: each deposition step paints the
    # disc swept from the previous contact point, so travel elongates and
    # connects the mark. Round brush to start: the cross-stroke radius is
    # unchanged, only the along-travel extent.
    brush_directional_enabled: bool = True
    # Bristle furrows: a round brush is a bundle of hairs, so it leaves
    # lengthwise streaks. A fraction of the hairs run dry (`gap_fraction`),
    # carving furrows -- these survive the opacity saturation that washes out a
    # mere deposition-rate wobble, so the mark reads as brushed rather than
    # stamped. Dry gaps are INTERMITTENT: they open and close along the path
    # with segment scale `brush_streak_length` (world units) as hairs recharge
    # from the surrounding paint, so a furrow never splits a stroke end to end.
    # `gap_gain` is the residual bleed under a dry hair; `depth` sets variation
    # among the wet hairs. Set depth and gap_fraction to 0 for smooth.
    brush_bristle_count: int = 9
    brush_bristle_depth: float = 0.25
    brush_bristle_gap_fraction: float = 0.28
    brush_bristle_gap_gain: float = 0.12
    brush_streak_length: float = 1.8
    # Per-stroke angular wobble of the footprint boundary: a real brush cannot
    # stamp a perfect circle, so the contact-patch radius varies slightly with
    # direction (fixed low-order harmonics per stroke). 0 disables.
    brush_edge_wobble: float = 0.12
    brush_seed: int = 0
    # Canvas tooth/grain: a fixed substrate height texture. A light brush
    # deposits only on the raised tooth and leaves unreached valleys genuinely
    # bare (dry-brushing); pressing harder works paint into the valleys until
    # reach >= 1 fills everything. This is the primary source of brush texture
    # in real tools (Krita/MyPaint) and survives opacity build-up because bare
    # valleys stay bare. Strength 0 disables (smooth deposition).
    canvas_grain_strength: float = 1.0
    canvas_grain_period_px: float = 4.5
    canvas_grain_reach_base: float = 0.50
    canvas_grain_reach_pressure: float = 0.70
    canvas_grain_seed: int = 0
    # Stroke-end taper: brush width ramps in/out over this fraction of the paint
    # phase at each end, so marks come to points instead of round caps.
    brush_taper_fraction: float = 0.28
    brush_taper_min_width: float = 0.18
    # Bidirectional paint transfer (the "dirty brush" loop used by ArtRage /
    # Krita's color-smudge engine): per stamp the head skims a pressure-scaled
    # fraction of the wet surface layer into a small held reservoir, mixes it
    # with the fresh load, and redeposits the mixture with a leading-edge bias
    # so paint is pushed ahead of the stroke. Mass and pigment are exactly
    # conserved between canvas and brush, and every knob is calibratable from a
    # few real strokes (drag a clean brush through a wet patch: trail length
    # gives pickup/release; ridge geometry gives the push bias).
    brush_pickup_fraction: float = 0.14
    brush_pickup_depth: float = 0.02
    brush_capacity_thickness: float = 0.08
    brush_release_fraction: float = 0.10
    brush_push_forward: float = 0.6
    # Bristle-tip trailer dynamics: the painting point is a damped follower of
    # the contact point (it lags and cuts corners like a pulled brush tip).
    # Reset at each contact onset; 0 disables.
    brush_tip_lag_seconds: float = 0.06

    state_dim: int = 6
    action_dim: int = 12
    # Deprecated library default retained temporarily for checkpoint/test
    # compatibility. Runtime entry points default to spatial_material and
    # constructing an active driver with summary emits a FutureWarning.
    planner_state_kind: str = SUMMARY_PLANNER_STATE_KIND
    spatial_grid_size: int = 16
    material_pyramid_levels: tuple[int, ...] = (64, 32, 16)
    spatial_material_channels: int = 6
    spatial_action_channels: int = 11
    spatial_transition_mode: str = "local_patch"
    spatial_hidden_channels: int = 32
    spatial_residual_blocks: int = 3
    spatial_ensemble_size: int = 3
    local_patch_margin_cells: int = 8
    local_patch_min_cells: int = 16
    local_patch_batch_bucket_cells: int = 16
    local_patch_sequential_cell_limit: int = 8192
    local_identity_logvar: float = -12.0
    mark_slot_count: int = 8
    mark_activation_coverage: float = 0.08
    ensemble_size: int = 5
    hidden_dim: int = 96

    target_coverage: float = 0.87
    # Midpoint of the declared sigmoid stop prior p(stop-first policy):
    # log p(stop) = logsigmoid(stop_prior_sharpness * (coverage - midpoint)).
    # Continuation policies carry a flat prior; the policy softmax renormalizes.
    minimum_stop_coverage: float = 0.70
    stop_prior_sharpness: float = 40.0
    terminal_concentration: float = 110.0
    policy_precision: float = 3.0
    # Declared precisions over outcome modalities in expected free energy.
    # Logged EFE components are precision-weighted contributions.
    terminal_risk_precision: float = 1.0
    ambiguity_precision: float = 1.0
    transition_precision: float = 1.0
    # Mixture weight of the low-coverage-seeking stroke proposal (a declared
    # empirical policy prior); the remainder of proposals stay uniform.
    proposal_low_coverage_mix: float = 0.5
    # Mixture weight of the higher-level passage proposal. A passage is a
    # slower latent policy prior over several related marks; expected free
    # energy still scores the resulting terminal outcome.
    passage_proposal_mix: float = 0.35
    # Conditional mixture inside passage proposals. A polyline passage is a
    # low-dimensional latent over connected straight marks: its `spacing`
    # coordinate is interpreted as signed turn radians between segments.
    passage_polyline_mix: float = 0.35
    passage_min_strokes: int = 2
    passage_max_strokes: int = 4
    passage_lateral_jitter: float = 0.025
    passage_longitudinal_jitter: float = 0.035
    passage_plan_proposal_mix: float = 0.20
    passage_plan_min_passages: int = 2
    passage_plan_max_passages: int = 3
    passage_plan_spacing: float = 0.20
    passage_plan_center_jitter: float = 0.08
    passage_plan_turn_jitter: float = 0.45
    # --- Feature D: amortized candidate-policy proposal q_proposal(pi|belief) ---
    # This is a PROPOSAL distribution, not a policy prior: it changes which
    # hypotheses the posterior sees and enters no EFE term, no preference, and no
    # normalized p(pi) factor. Zero reproduces the hand-written sampler exactly
    # (no extra RNG draws), so the hand-written proposal stays a permanent
    # measurable baseline mixed in at (1 - mix) in the SAME planning round.
    # Training is independent of this weight: the proposal learns from the
    # hand-written support first, and the mix is ramped only once
    # diagnostics()['policyProposal'] shows the learned proposal assigning higher
    # likelihood to selected policies. Never set it to 1.0 -- at that point the
    # proposal would be trained on the posterior over candidates it alone
    # supplied, and the paired hand-written control would be gone. An extreme
    # config (mix 1.0 with both passage mixes 0.0 and planning_horizon 1) also
    # degenerates to depth-1 candidates only.
    learned_proposal_mix: float = 0.0
    # Spatial planner only: the summary planner has no canvas or relational
    # posterior to condition on, so it keeps the hand-written proposal.
    learned_proposal_enabled: bool = True
    learned_proposal_hidden_dim: int = 96
    learned_proposal_lr: float = 1e-3
    learned_proposal_train_steps: int = 1
    # Monte-Carlo sample count for the declared divergence diagnostic
    # D_KL(learned proposal || hand-written proposal), in nats per latent.
    learned_proposal_divergence_samples: int = 32
    # Floor on the learned factored proposal's per-parameter log scale, so a
    # collapsed factor cannot make log q unbounded above. A numerical support
    # bound on a proposal density, NOT a precision belief over any outcome.
    learned_proposal_min_log_scale: float = -4.0
    passage_local_candidate_policies: int = 6
    passage_continuation_probability: float = 0.92
    passage_belief_center_std: float = 0.08
    passage_belief_direction_std: float = 0.35
    passage_belief_geometry_std: float = 0.16
    passage_belief_transition_std: float = 0.015
    passage_belief_observation_std: float = 0.035
    # Declared structural prior over terminal canvases (spatial mode):
    # p*(s_T) ~ exp(precision * compression_gap(s_T)), where the gap is the
    # hierarchical code's explanatory advantage over the BEST member of a
    # declared, hand-written, parameter-free context-free baseline family.
    # Zero disables the composition hierarchy entirely.
    composition_gap_precision: float = 1.0
    # Membership of the context-free baseline family the compression gap is
    # measured against. True adds a parameter-free 3x3 hollow-neighbourhood
    # local Markov code beside the iid-cell Gaussian and scores against the
    # better of the two, so the gap cannot be earned by local smoothness
    # alone. False restores the iid-only baseline exactly. Both members are
    # hand-supplied codes, never fit to outcomes; this selects a reference
    # code, it does not learn a preference.
    composition_local_baseline_enabled: bool = True
    composition_latent_dim: int = 16
    composition_hidden_channels: int = 24
    composition_lr: float = 1e-3
    composition_train_steps: int = 2
    # Persistent high-level posteriors and passage-conditioned transition
    # likelihoods. These are EFE modalities, not aesthetic score terms.
    canvas_latent_channels: int = 8
    relational_latent_dim: int = 24
    hierarchy_hidden_dim: int = 96
    canvas_latent_process_std: float = 0.18
    relational_process_std: float = 0.14
    canvas_latent_transition_precision: float = 0.30
    relational_transition_precision: float = 0.30
    hierarchy_transition_batch_size: int = 8
    hierarchy_transition_train_steps: int = 1
    # A passage-conditioned Markov likelihood over the coarse canvas and
    # relational latents. It is trained from each observed mark inside a
    # passage while the persistent high-level posterior itself remains fixed
    # until the explicit passage boundary.
    passage_trajectory_enabled: bool = True
    passage_trajectory_batch_size: int = 8
    passage_trajectory_train_steps: int = 1
    # Per-member Bernoulli bootstrap keep-probability for ensemble training,
    # so members see different data and stay dispersed as a parameter posterior.
    ensemble_bootstrap_probability: float = 0.7

    planning_horizon: int = 3
    candidate_policies: int = 96
    # Expensive embodied refinement is applied to the best base-EFE painting
    # candidates. Three canvas candidates x five motor realizations stays below
    # the old eight x three budget because fixed-roll IK is richer per rollout.
    motor_forecast_candidates: int = 3
    motor_forecast_samples: int = 3
    # Independent motor-likelihood rollouts use deep-copied simulator states.
    # Batching changes only scheduling: equations, dt, and particle count stay
    # identical to sequential forecasts.
    # The default stays serial because CPython thread overhead is currently
    # larger than the overlap benefit in production-sized CPU benchmarks.
    # Raising this remains available for runtimes whose simulator releases
    # enough of the GIL to benefit.
    motor_forecast_workers: int = 1
    motor_planning_enabled: bool = True
    motor_realization_kinds: tuple[str, ...] = (
        "cartesian_ik",
        "joint_spline",
        "elbow_pivot",
        "upper_arm_roll_positive",
        "upper_arm_roll_negative",
    )
    motor_realization_candidate_limit: int = 5
    motor_roll_sweep_degrees: float = 32.0
    motor_proprioceptive_risk_precision: float = 0.35
    motor_proprioceptive_ambiguity_precision: float = 0.25
    # Declared precision on the reliability PARAMETER-novelty term. Split out of
    # motor_proprioceptive_ambiguity_precision because that one now scales only
    # the state/observation mutual information; parameter novelty over the
    # learned inverse-gamma reliability belief is a different quantity and must
    # be separately attributable. Default matches the old shared value so the
    # split is numerically inert.
    motor_reliability_novelty_precision: float = 0.25
    # Learned per-motion-family execution reliability: an inverse-gamma
    # precision belief over how much jitterier real execution is than the
    # body-model forecast (the squared ratio of realized to predicted tracking
    # error), maintained per motor realization kind and updated after every
    # executed stroke. Its posterior mean inflates that kind's forecast outcome
    # variance inside motor EFE -- a precision belief, not a reward -- so
    # reliable motions win selection and unproven ones carry both extra risk
    # and resolvable uncertainty. The prior starts mildly pessimistic.
    motor_reliability_enabled: bool = True
    motor_reliability_prior_mean: float = 1.6
    motor_reliability_prior_strength: float = 4.0
    # Body-parameter jitter for motor forecasts: rollout particles beyond the
    # first perturb friction, backlash, transmission stiffness, and process
    # noise by this log-normal fraction, so motions that amplify body
    # uncertainty (fast sweeps, backlash reversals, extended reach) forecast
    # wider even before any reliability evidence arrives. 0 disables.
    body_param_jitter_fraction: float = 0.12
    motor_current_preference_std: float = 0.35
    motor_torque_preference_std: float = 0.35
    motor_velocity_preference_std: float = 0.40
    motor_acceleration_preference_std: float = 0.45
    motor_target_error_preference_std: float = 0.20
    motor_limit_preference_std: float = 0.12
    motor_contact_loss_preference_std: float = 0.12
    motor_pressure_error_preference_std: float = 0.20
    motor_path_error_preference_std: float = 0.12
    motor_limit_margin_degrees: float = 12.0
    post_stroke_retract_seconds: float = 0.35
    passage_local_retract_seconds: float = 0.12
    passage_center_retract_seconds: float = 0.65
    global_planning_retract_depth: float = 3.0
    global_planning_clearance_fraction: float = 0.60
    global_planning_park_x_fraction: float = 0.0
    global_planning_park_z_fraction: float = -0.5
    local_passage_retract_depth: float = 1.0
    hold_damping_multiplier: float = 3.5
    hold_target_joint_speed_deg_s: float = 60.0
    hold_target_joint_accel_deg_s2: float = 140.0
    contact_release_pressure_threshold: float = 0.05
    contact_release_joint_speed_deg_s: float = 720.0
    background_planner_yield_seconds: float = 0.0005
    stroke_tone_prior: float | None = None
    inference_steps: int = 24
    inference_lr: float = 0.08
    # Posterior samples used only to report the summary-state VFE decomposition
    # after optimization. Sampling is RNG-isolated, so this does not change
    # q(s), later stochastic learning, EFE, or policy selection.
    # The previous implicit budget of 32 left the independently checked total
    # inside only a +-0.35 nat Monte Carlo band; 4096 makes that diagnostic
    # precise enough for the AI-104 reference-model acceptance gate while the
    # heteroscedastic summary likelihood remains explicitly approximate.
    summary_vfe_report_samples: int = 4096

    # Modality-level multiplier on the motor proprioceptive EFE contribution.
    # The three precisions above are applied INSIDE motor_efe_terms; this one is
    # the modality's own declared precision, and it seeds the prior mean of the
    # motor Gamma precision belief. The identity default makes today's
    # arithmetic exactly the unobserved-belief case.
    motor_modality_precision: float = 1.0

    # --- Feature C: Gamma precision beliefs over the EFE modalities -------
    # The seven declared precisions above become posterior means of Gamma
    # beliefs updated by the reference Ch.10 rule dF/dgamma =
    # (alpha/gamma - beta0) + (pi - pi0) . (-G). Each belief's rate is seeded
    # at beta0 = alpha0 / (declared constant), so an unobserved belief and a
    # disabled flag both reproduce the constant arithmetic exactly.
    precision_beliefs_enabled: bool = True
    # False keeps the seven modality precisions constant while still learning
    # the policy precision, so the two mechanisms are separately attributable.
    modality_precision_beliefs_enabled: bool = True
    # Gamma shape. alpha0 = 1 makes each belief exactly the reference's
    # single-parameter gamma = 1/beta. Declared, never updated from data.
    precision_belief_alpha0: float = 1.0
    precision_belief_kappa: float = 0.25
    precision_belief_iterations: int = 64
    # Declared bounded support, as a ratio of each belief's prior mean.
    # Measured: disagreeing evidence drives gamma to ~1e-3 unbounded, which
    # would let learned data switch a declared preference off. Also keeps
    # every diagnostics value finite and JSON-serializable.
    precision_belief_min_ratio: float = 0.1
    precision_belief_max_ratio: float = 10.0

    # --- Feature C: modality unit normalization ----------------------------
    # Reduce every EFE modality to nats per observation channel and record
    # each normalizer's name in the component dataclass. False keeps the
    # historical mixed units (scalar Beta / per-cell-channel / per-latent-dim
    # / raw 27-channel sum) for attribution.
    modality_normalization_enabled: bool = True
    # Restrict the moment-matched terminal coverage forecast to the interior
    # unimodal Beta family: both concentrations >= this floor, mean preserved
    # by a common rescale. Measured: removes a digamma(alpha -> 0) singularity
    # worth 53248 nats on a near-blank forecast, capping it at 892, and leaves
    # every well-conditioned forecast bit-unchanged. 0 disables.
    terminal_forecast_concentration_floor: float = 1.0
    # Structural enable flag for the composition hierarchy, kept separate from
    # composition_gap_precision so a learned precision can never construct or
    # destroy the model, and so the checkpoint architecture key stays a
    # declared constant rather than a learned quantity.
    composition_enabled: bool = True

    # --- Feature C: gap-progress stop policy prior -------------------------
    # Gaussian random-walk belief over the compression-gap increment per
    # completed mark. Its standardized posterior mean is a SECOND factor on
    # the stop policy prior: log p(stop) = logsigmoid(coverage term) +
    # logsigmoid(-gap_progress_stop_sharpness * mean/sqrt(var)). Both factors
    # are <= 0, so neither can manufacture value. Delta-gap never enters
    # expected free energy and the terminal coverage preference is untouched.
    gap_progress_stop_enabled: bool = True
    gap_increment_prior_mean: float = 0.05
    gap_increment_prior_std: float = 0.10
    gap_increment_process_std: float = 0.02
    gap_increment_observation_std: float = 0.05
    gap_progress_stop_sharpness: float = 2.0

    replay_capacity: int = 50_000
    batch_size: int = 128
    model_lr: float = 2e-3

    # --- Feature A: motion-manifold bootstrap of the composition hierarchy ---
    # Bootstrap mark source. 'motion_manifold' seeds the transition and
    # canvas/relational likelihoods on the body's own reachable-motion manifold
    # (joint-space sweeps projected by FK); 'random_stroke' retains the previous
    # iid PolicySampler._stroke source so hand-supplied structure stays
    # separately attributable. Generative-process exploration below the
    # painting-policy boundary: it supplies no preference and selects no policy.
    bootstrap_generator: str = "motion_manifold"
    # The sweep sampler's own RNG seed. Deliberately NOT agent.policy_sampler.rng:
    # sharing it would make switching generators also change the live planner's
    # candidate stream and confound the attribution comparison.
    bootstrap_manifold_seed: int = 4242
    # Marks per bootstrap episode. The canvas is cleared ONLY at episode
    # boundaries, so one episode accumulates a complete organized canvas for the
    # canvas/relational likelihood to encode. 24 preserves the previous clear
    # cadence exactly.
    bootstrap_episode_marks: int = 24
    # Joint-space arclength (summed degrees) of one sweep.
    bootstrap_manifold_amplitude_degrees: tuple[float, float] = (45.0, 130.0)
    # Sweep integration step. This is the exploration discretization, not a
    # commanded mark speed: emitted marks are re-timed by
    # stroke_execution.adaptive_stroke_timing, so a sweep's velocity profile
    # never reaches the canvas.
    bootstrap_manifold_step_degrees: tuple[float, float] = (0.25, 0.80)
    # A single-joint sweep cannot paint: from the canvas-centre IK pose the tip
    # holds the near-contact band for only a few degrees, a sub-0.1-normalized
    # dab. A family therefore only UP-WEIGHTS its joint inside a coordinated
    # sweep.
    bootstrap_manifold_families: tuple[str, ...] = (
        "yaw_dominant",
        "pitch_dominant",
        "roll_dominant",
        "elbow_dominant",
        "coordinated",
    )
    # Relative joint-velocity weight of the non-dominant joints within a family.
    bootstrap_manifold_dominance_ratio: float = 0.22
    # Chain each sweep from the pose the previous one ended at, so an episode
    # lays down a connected region instead of a scatter of independent arcs.
    bootstrap_manifold_chain_sweeps: bool = True
    # Per-joint perturbation applied to the chained start pose.
    bootstrap_manifold_chain_jitter_degrees: float = 3.0
    # Minimum emitted mark length, matching PolicySampler._stroke's floor.
    # Sweeps shorter than this are rejected; a sweep long enough for only one
    # mark is emitted without a passage latent, because a polyline PassageLatent
    # needs at least two marks (Policy.__post_init__).
    bootstrap_manifold_min_mark_length: float = 0.20
    # Canvas/relational gradient steps at each bootstrap episode boundary.
    # Measured: the compression gap does not discriminate structure at all below
    # a few hundred steps, while the online composition_train_steps budget
    # supplies only ~50 across a whole bootstrap. 0 keeps today's cost exactly;
    # the attribution A/B sets it explicitly. A gradient budget, not an
    # objective term.
    bootstrap_composition_train_steps: int = 0
    # Emit fitted polyline latents into the passage likelihood replays. Off by
    # default because it would open the kind-specific transition-EFE gates on
    # bootstrap-only evidence before any real painting.
    bootstrap_feeds_passage_likelihood: bool = False
