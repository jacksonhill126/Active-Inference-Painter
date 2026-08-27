from __future__ import annotations

from typing import Mapping


LEARNING_INHERITANCE_VERSION = "online-learning-inheritance-v0"

RESUME_INDIVIDUAL_DEVELOPMENT = "resume_individual_development"
INITIALIZE_FROM_SHARED_PRETRAINING = "initialize_from_shared_pretraining"
CONTINUE_SHARED_PRETRAINING = "continue_shared_pretraining"
CHECKPOINT_LOAD_MODES = frozenset(
    {
        RESUME_INDIVIDUAL_DEVELOPMENT,
        INITIALIZE_FROM_SHARED_PRETRAINING,
        CONTINUE_SHARED_PRETRAINING,
    }
)

INDIVIDUAL_ONLINE_DEVELOPMENT = "individual_online_development"
SHARED_PRETRAINING = "shared_pretraining"
TRAINING_ROLES = frozenset({INDIVIDUAL_ONLINE_DEVELOPMENT, SHARED_PRETRAINING})

REGISTERED_CAMERA_OBSERVATION = "registered_camera_observation"
ORACLE_DIAGNOSTIC_EXECUTION = "oracle_diagnostic_execution"
PHYSICAL_SENSOR_OBSERVATION = "physical_sensor_observation"
MODEL_IMAGINED_ROLLOUT = "model_imagined_rollout"
OBSERVED_TRANSITION_SOURCES = frozenset(
    {
        REGISTERED_CAMERA_OBSERVATION,
        ORACLE_DIAGNOSTIC_EXECUTION,
        PHYSICAL_SENSOR_OBSERVATION,
    }
)


def validate_checkpoint_load_mode(mode: str) -> str:
    resolved = str(mode)
    if resolved not in CHECKPOINT_LOAD_MODES:
        raise ValueError(
            f"checkpoint_load_mode must be one of {sorted(CHECKPOINT_LOAD_MODES)}; "
            f"got {resolved!r}"
        )
    return resolved


def normalized_training_provenance(
    provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    result = dict(provenance or {})
    role = str(result.get("training_role", INDIVIDUAL_ONLINE_DEVELOPMENT))
    if role not in TRAINING_ROLES:
        raise ValueError(
            f"training_role must be one of {sorted(TRAINING_ROLES)}; got {role!r}"
        )
    result["training_role"] = role
    result["inheritance_contract"] = LEARNING_INHERITANCE_VERSION
    return result


def require_observed_transition_source(source: str) -> str:
    resolved = str(source)
    if resolved not in OBSERVED_TRANSITION_SOURCES:
        raise ValueError(
            "online replay accepts only realized observation evidence; "
            f"source={resolved!r} is not one of {sorted(OBSERVED_TRANSITION_SOURCES)}. "
            "Model-imagined rollouts are transition-prior samples, never observations."
        )
    return resolved


def checkpoint_component_manifest() -> dict[str, object]:
    """Machine-readable ownership and persistence contract for AI-112.

    This describes semantic component classes, not one checkpoint's values. It
    is embedded in every driver checkpoint and exposed in diagnostics so a
    resume cannot silently reinterpret episodic beliefs as inherited learning.
    """

    return {
        "version": LEARNING_INHERITANCE_VERSION,
        "components": {
            "shared_generative_parameters": {
                "checkpointKeys": [
                    "dynamics_state",
                    "composition_state",
                    "passage_kind_update_counts",
                ],
                "persistsAcrossPaintings": True,
                "resumeIndividual": "restore",
                "initializeFromSharedPretraining": "restore_parameters_only",
                "continueSharedPretraining": "restore",
                "architectureChange": "reject_unless_explicit_migration",
            },
            "individual_optimizer_state": {
                "checkpointKeys": [
                    "optimizer_state",
                    "composition_optimizer_state",
                    "proposal_optimizer_state",
                ],
                "persistsAcrossPaintings": True,
                "resumeIndividual": "restore",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "restore_generative_optimizers_only",
                "architectureChange": "reject",
            },
            "observational_replay": {
                "checkpointKeys": [
                    "replay",
                    "composition_replay",
                    "passage_replay",
                    "passage_step_replay",
                ],
                "persistsAcrossPaintings": True,
                "retention": "capacity_bounded_fifo",
                "resumeIndividual": "restore",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "reset_and_rebuild_from_training_split",
                "imaginedRolloutsAllowed": False,
            },
            "developmental_calibration": {
                "checkpointKeys": ["motion_reliability", "precision_ledger"],
                "persistsAcrossPaintings": True,
                "resumeIndividual": "restore",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "reset",
            },
            "episodic_posteriors": {
                "checkpointKeys": [],
                "examples": [
                    "canvas_material_belief",
                    "body_belief",
                    "brush_load_beliefs",
                    "gap_increment_belief",
                    "passage_belief",
                ],
                "persistsAcrossPaintings": False,
                "resumeIndividual": "reset",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "reset",
            },
            "canvas_and_passage_history": {
                "checkpointKeys": [],
                "examples": [
                    "current_canvas",
                    "active_passage",
                    "passage_queue",
                    "slow_canvas_and_relational_beliefs",
                ],
                "persistsAcrossPaintings": False,
                "resumeIndividual": "reset",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "reset",
            },
            "individual_policy_proposal": {
                "checkpointKeys": ["proposal_state", "proposal_generator_state"],
                "persistsAcrossPaintings": True,
                "resumeIndividual": "restore",
                "initializeFromSharedPretraining": "reset",
                "continueSharedPretraining": "reset",
                "policyInfluenceInM1": False,
            },
        },
        "compatibility": {
            "acrossPaintings": "apply the selected component ownership rule",
            "acrossCodeVersions": (
                "load only when schema, declared architecture, and inheritance "
                "manifest match exactly; run manifests retain the code revision"
            ),
            "acrossArchitectureChanges": (
                "reject by default; require an explicit versioned migration with tests"
            ),
        },
        "heldOutMonitoring": {
            "replayMayContainHeldOutTrajectories": False,
            "splitUnit": "whole_trajectory",
            "forgettingSignal": (
                "anchor held-out NLL degradation exceeding a predeclared seed or "
                "bootstrap uncertainty band, reported beside recent-data NLL"
            ),
            "policyInfluence": False,
            "responseToForgetting": (
                "fail the learning-admission report and diagnose replay coverage or "
                "update schedule; never turn the held-out score into a reward"
            ),
        },
    }
