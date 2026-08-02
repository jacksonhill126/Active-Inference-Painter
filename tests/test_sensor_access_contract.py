import json
from pathlib import Path

import pytest

from active_painter.arm_agent_driver import (
    OBSERVATION_ACCESS_MODE,
    OBSERVATION_BASELINE_ID,
    ORACLE_OBSERVATION_ACCESS_MODE,
    ORACLE_OBSERVATION_BASELINE_ID,
    SENSOR_OBSERVATION_ACCESS_MODE,
    SENSOR_OBSERVATION_BASELINE_ID,
    ArmActiveInferenceDriver,
    PrivilegedStateAccessError,
)
from active_painter.arm_sim import (
    ArmKinematics,
    ArmPainterSim,
    Brush,
    ContactState,
    JointPlant,
    MotorTelemetry,
    VerticalCanvas,
)
from active_painter.config import PainterConfig
from active_painter.spatial_state import MATERIAL_CHANNELS
from active_painter.stroke_execution import ExecutionForecast


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "planning" / "variable-sensor-access-ledger.json"
REQUIRED_CLASSIFICATIONS = {
    "physical_sensor_observation",
    "derived_observation",
    "hidden_state",
    "simulator_only_evaluation_label",
    "execution_only_state",
}
REQUIRED_ENTRY_FIELDS = {
    "id",
    "source_type",
    "values",
    "classification",
    "consumers",
    "current_access",
    "likelihood_status",
    "physical_analogue",
    "research_status",
    "blockers",
}


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _entries_by_source(ledger: dict[str, object], source_type: str) -> set[str]:
    entries = ledger["entries"]
    assert isinstance(entries, list)
    return {
        str(value)
        for entry in entries
        if isinstance(entry, dict) and entry["source_type"] == source_type
        for value in entry["values"]
    }


def test_sensor_access_ledger_schema_and_privileged_blockers() -> None:
    ledger = _ledger()
    assert ledger["schema_version"] == "sensor-access-ledger-v0"
    classifications = ledger["classifications"]
    assert isinstance(classifications, dict)
    assert set(classifications) == REQUIRED_CLASSIFICATIONS

    entries = ledger["entries"]
    assert isinstance(entries, list)
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    for entry in entries:
        assert REQUIRED_ENTRY_FIELDS <= set(entry)
        assert entry["classification"] in REQUIRED_CLASSIFICATIONS
        assert entry["values"]
        assert entry["consumers"]
        assert entry["current_access"]
        access = " ".join(entry["current_access"])
        if "privileged" in access:
            assert entry["research_status"] == "oracle_only"
            assert entry["blockers"]


def test_ledger_covers_every_simulator_boundary_dataclass_field() -> None:
    ledger = _ledger()
    boundary_types = (
        ArmPainterSim,
        VerticalCanvas,
        JointPlant,
        MotorTelemetry,
        ContactState,
        Brush,
        ArmKinematics,
        ExecutionForecast,
    )
    for boundary_type in boundary_types:
        expected = set(boundary_type.__dataclass_fields__)
        recorded = _entries_by_source(ledger, boundary_type.__name__)
        assert recorded == expected, (
            f"{boundary_type.__name__} ledger mismatch: "
            f"missing={sorted(expected - recorded)}, extra={sorted(recorded - expected)}"
        )


def test_spatial_oracle_channels_are_explicitly_ledgered() -> None:
    ledger = _ledger()
    entries = ledger["entries"]
    assert isinstance(entries, list)
    spatial = next(entry for entry in entries if entry["id"] == "canvas.spatial_observation")
    recorded = set(spatial["values"])
    assert {f"pixel {channel}" for channel in MATERIAL_CHANNELS} <= recorded
    assert spatial["classification"] == "derived_observation"
    assert spatial["research_status"] == "oracle_only"


def test_default_model_boundary_fails_closed_without_reading_process_truth() -> None:
    class ForbiddenProcess:
        def __getattribute__(self, name: str):
            raise AssertionError(f"sensor-only model dereferenced process field {name!r}")

    driver = ArmActiveInferenceDriver(
        config=PainterConfig(candidate_policies=2, planning_horizon=1),
        bootstrap_transitions=8,
        bootstrap_train_steps=1,
    )
    process = ForbiddenProcess()

    driver.reset(process)  # type: ignore[arg-type]
    driver.step(process, 1.0 / 240.0)  # type: ignore[arg-type]
    boundary = driver.diagnostics()["observationBoundary"]

    assert OBSERVATION_BASELINE_ID == SENSOR_OBSERVATION_BASELINE_ID
    assert OBSERVATION_ACCESS_MODE == SENSOR_OBSERVATION_ACCESS_MODE
    assert driver.enabled is False
    assert driver.trained_transitions == 0
    assert boundary["baseline"] == SENSOR_OBSERVATION_BASELINE_ID
    assert boundary["mode"] == SENSOR_OBSERVATION_ACCESS_MODE
    assert boundary["modelAccessBlocked"] is True
    assert boundary["materialStateAccess"] == "denied"
    assert boundary["bodyForecastInitialization"] == "denied"
    assert boundary["cameraPosteriorConnected"] is False
    assert "spatial_material" in boundary["blockedReason"]

    with pytest.raises(PrivilegedStateAccessError, match="denied"):
        driver._planner_state(process)  # type: ignore[arg-type]


def test_explicit_oracle_diagnostic_mode_retains_legacy_baseline_label() -> None:
    ledger = _ledger()
    baseline = ledger["baseline"]
    assert isinstance(baseline, dict)
    driver = ArmActiveInferenceDriver(
        config=PainterConfig(candidate_policies=2, planning_horizon=1),
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )

    boundary = driver.diagnostics()["observationBoundary"]

    assert ORACLE_OBSERVATION_BASELINE_ID == baseline["id"]
    assert ORACLE_OBSERVATION_ACCESS_MODE == baseline["observation_mode"]
    assert boundary["baseline"] == baseline["id"]
    assert boundary["mode"] == baseline["observation_mode"]
    assert boundary["sensorEquivalent"] is False
    assert boundary["modelAccessBlocked"] is False
    assert "diagnostic-only" in boundary["materialStateAccess"]


def test_oracle_baseline_prohibits_sensor_only_claims() -> None:
    baseline = _ledger()["baseline"]
    assert isinstance(baseline, dict)
    prohibited = set(baseline["prohibited_claims"])
    assert "sensor-only state inference" in prohibited
    assert "embodied prediction without simulator-state leakage" in prohibited
    assert baseline["sensor_equivalent"] is False
