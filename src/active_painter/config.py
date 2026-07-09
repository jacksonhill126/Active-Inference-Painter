from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PainterConfig:
    canvas_size: int = 48
    thickness_scale: float = 0.005
    wetness_decay: float = 0.985
    canvas_ground_tone: float = 0.34
    paint_deposition_base_rate: float = 0.16
    paint_deposition_pressure_rate: float = 0.64
    oil_surface_opacity_thickness: float = 0.002
    oil_wet_pickup_fraction: float = 0.18
    base_observation_std: float = 0.008
    smear_observation_std: float = 0.02

    state_dim: int = 6
    action_dim: int = 10
    planner_state_kind: str = "summary"
    spatial_grid_size: int = 16
    material_pyramid_levels: tuple[int, ...] = (64, 32, 16)
    spatial_material_channels: int = 6
    spatial_action_channels: int = 9
    spatial_transition_mode: str = "local_patch"
    spatial_hidden_channels: int = 32
    spatial_residual_blocks: int = 3
    spatial_ensemble_size: int = 3
    local_patch_margin_cells: int = 8
    local_patch_min_cells: int = 16
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
    # Declared structural prior over terminal canvases (spatial mode):
    # p*(s_T) ~ exp(precision * compression_gap(s_T)), where the gap is the
    # hierarchical code's explanatory advantage over a context-free flat code.
    # Zero disables the composition hierarchy entirely.
    composition_gap_precision: float = 1.0
    composition_latent_dim: int = 16
    composition_hidden_channels: int = 24
    composition_lr: float = 1e-3
    composition_train_steps: int = 2
    # Per-member Bernoulli bootstrap keep-probability for ensemble training,
    # so members see different data and stay dispersed as a parameter posterior.
    ensemble_bootstrap_probability: float = 0.7

    planning_horizon: int = 3
    candidate_policies: int = 96
    motor_forecast_candidates: int = 8
    motor_planning_enabled: bool = True
    motor_realization_kinds: tuple[str, ...] = ("cartesian_ik", "joint_spline", "elbow_pivot")
    motor_realization_candidate_limit: int = 3
    motor_proprioceptive_risk_precision: float = 0.35
    motor_proprioceptive_ambiguity_precision: float = 0.25
    motor_limit_margin_degrees: float = 12.0
    post_stroke_retract_seconds: float = 0.35
    passage_local_retract_seconds: float = 0.12
    passage_center_retract_seconds: float = 0.65
    global_planning_retract_depth: float = 4.0
    global_planning_park_x_fraction: float = -0.62
    local_passage_retract_depth: float = 1.0
    hold_damping_multiplier: float = 2.5
    hold_target_joint_speed_deg_s: float = 180.0
    contact_release_pressure_threshold: float = 0.05
    contact_release_joint_speed_deg_s: float = 720.0
    background_planner_yield_seconds: float = 0.0005
    stroke_tone_prior: float | None = None
    inference_steps: int = 24
    inference_lr: float = 0.08

    replay_capacity: int = 50_000
    batch_size: int = 128
    model_lr: float = 2e-3
