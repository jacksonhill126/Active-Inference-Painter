from __future__ import annotations

from dataclasses import dataclass


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
    # boundary). Oil paint does not dry within a session: paint is laid at a
    # consistent rate along a stroke and the canvas keeps its wetness (there is
    # no wetness decay). The stroke's `amount` sets how heavily the brush is
    # loaded, which scales deposited thickness/opacity uniformly along the mark
    # (a fuller brush lays thicker paint) -- it never runs out or thins toward
    # the end. `amount` 0 -> brush_load_min, 1 -> brush_load_max.
    brush_load_min: float = 0.55
    brush_load_max: float = 1.45
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
    # Reset at each pen-down; 0 disables.
    brush_tip_lag_seconds: float = 0.06

    state_dim: int = 6
    action_dim: int = 12
    planner_state_kind: str = "summary"
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
    passage_local_candidate_policies: int = 6
    passage_continuation_probability: float = 0.92
    passage_belief_center_std: float = 0.08
    passage_belief_direction_std: float = 0.35
    passage_belief_geometry_std: float = 0.16
    passage_belief_transition_std: float = 0.015
    passage_belief_observation_std: float = 0.035
    # Declared structural prior over terminal canvases (spatial mode):
    # p*(s_T) ~ exp(precision * compression_gap(s_T)), where the gap is the
    # hierarchical code's explanatory advantage over a context-free flat code.
    # Zero disables the composition hierarchy entirely.
    composition_gap_precision: float = 1.0
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
    global_planning_retract_depth: float = 4.0
    global_planning_clearance_fraction: float = 0.60
    global_planning_park_x_fraction: float = 0.0
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

    replay_capacity: int = 50_000
    batch_size: int = 128
    model_lr: float = 2e-3
