from active_painter.config import M1_FORMAL_POLICY_BASELINE_ID, painting_policy_profile_id
from active_painter.runtime_profile import (
    RuntimeProfileSettings,
    measure_phase,
    representative_config,
    summarize_samples,
)


def test_representative_runtime_profile_uses_formal_m1_policy() -> None:
    config = representative_config(RuntimeProfileSettings())

    assert painting_policy_profile_id(config) == M1_FORMAL_POLICY_BASELINE_ID
    assert config.planner_state_kind == "spatial_material"
    assert config.composition_enabled is False
    assert config.composition_gap_precision == 0.0
    assert config.learned_proposal_mix == 0.0
    assert config.candidate_policies == 32
    assert config.planning_horizon == 4
    assert config.motor_forecast_workers == 1
    assert RuntimeProfileSettings().device == "cpu"


def test_quick_profile_changes_budget_without_enabling_legacy_policy() -> None:
    config = representative_config(RuntimeProfileSettings(quick=True))

    assert painting_policy_profile_id(config) == M1_FORMAL_POLICY_BASELINE_ID
    assert config.candidate_policies == 4
    assert config.planning_horizon == 1
    assert config.motor_realization_kinds == ("cartesian_ik",)


def test_phase_measurement_and_summary_report_resource_units() -> None:
    result, sample = measure_phase(lambda: 7, device="cpu")
    summary = summarize_samples([sample, dict(sample)])

    assert result == 7
    assert float(sample["wall_seconds"]) >= 0.0
    assert float(sample["process_cpu_seconds"]) >= 0.0
    assert float(sample["average_active_cpu_cores"]) >= 0.0
    assert sample["peak_cuda_memory_bytes"] is None
    assert summary["sample_count"] == 2
    assert summary["wall_seconds"]["minimum"] >= 0.0
