from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AGENT_ENTRY_DOCUMENTS = (
    "AGENTS.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CONTROL_PLANT_POLICY_BOUNDARY.md",
)
SUPPORT_DOCUMENTS = (
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "models").glob("*.md")),
    *sorted((ROOT / "planning").glob("*.md")),
)
VISUAL_BOUNDARY_ENTRY_DOCUMENTS = (
    "AGENTS.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CURRENT_IMPLEMENTATION.md",
    "docs/GENERATIVE_MODEL_SPEC.md",
    "docs/PROGRESS.md",
    "docs/RESEARCH_CHARTER.md",
    "docs/OWNER_CONTRIBUTION_BRIEF_2026-08-11.md",
    "docs/OWNER_STEERING_CATALOG_2026-08-11.md",
    "docs/PROJECT_DECISION_PROVENANCE_2026-08-11.md",
    "planning/GANTT.md",
    "planning/M1-formal-baseline-and-inference-audit.md",
    "planning/MILESTONE_INDEX.md",
    "planning/PROJECT_TRACKER.md",
    "planning/M2-calibrated-multiscale-generative-model.md",
)


@pytest.mark.parametrize("relative_path", AGENT_ENTRY_DOCUMENTS)
def test_agent_entry_documents_keep_plant_and_motor_terms_distinct(
    relative_path: str,
) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    normalized = text.lower().replace("-", " ")

    assert "native abstract v0" in normalized
    assert "mujoco robstride electromechanical v4" in normalized
    assert "motor realization" in normalized
    assert "robstride 03" in normalized
    assert "robstride 02" in normalized
    assert "baseline oracle v0" in normalized


def test_support_documents_do_not_restore_superseded_mujoco_status() -> None:
    stale_claims = (
        "a future mujoco backend",
        "| mujoco backend | not implemented |",
        "build mujoco as\nan abstract clone and execution backend",
        "policy forecasts still use `native-abstract-v0`",
        "counterfactual motor forecasts still deep-copy `native-abstract-v0`",
        "counterfactual motor forecasts currently use the native abstract plant",
        "counterfactual motor forecasts still use a deep-copied native plant",
        "body posterior is not connected to motor-forecast",
        "does not yet initialize motor forecasts",
        "sensor-conditioned body-state initialization remains future work",
        "both oracle paths start from exact process state",
        "forecast initialization remains `baseline-oracle-v0` until m2 connects",
        "native abstract body remains the declared counterfactual forecast approximation",
    )

    for path in SUPPORT_DOCUMENTS:
        text = path.read_text(encoding="utf-8").lower()
        for stale_claim in stale_claims:
            assert stale_claim not in text, f"stale claim in {path.relative_to(ROOT)}"


@pytest.mark.parametrize("relative_path", VISUAL_BOUNDARY_ENTRY_DOCUMENTS)
def test_agent_entry_documents_link_the_visual_model_boundary(
    relative_path: str,
) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "VISUAL_GENERATIVE_MODEL_BOUNDARY.md" in text


def test_visual_model_boundary_preserves_the_owner_confirmed_distinctions() -> None:
    text = (ROOT / "docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md").read_text(
        encoding="utf-8"
    )
    normalized = text.lower().replace("–", "-").replace("—", "-")

    assert "persistent canvas-wide wetness is not part of the target agent belief" in normalized
    assert "ephemeral interaction/affordance latent" in normalized
    assert "tone and oriented-boundary fidelity" in normalized
    assert "registered, rectified pre-action" in normalized
    assert "test or reject the visual vae" in normalized
    assert "at planning time the future image is unavailable" in normalized


def test_material_cvae_records_do_not_claim_to_implement_the_visual_vae() -> None:
    records = (
        ROOT / "docs/CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md",
        ROOT / "docs/CONDITIONAL_PATCH_VAE_OWNER_BRIEF_2026-08-11.md",
        ROOT / "docs/AI109_PREDICTIVE_LEARNING_CURVES_TECHNICAL_2026-08-12.md",
        ROOT / "docs/AI109_PREDICTIVE_LEARNING_CURVES_OWNER_BRIEF_2026-08-12.md",
    )
    stale_attributions = (
        "it is the\nfirst implementation of the owner's proposal",
        "we implemented the first careful version of the vae idea you brought back",
    )

    for path in records:
        text = path.read_text(encoding="utf-8").lower()
        for stale_attribution in stale_attributions:
            assert stale_attribution not in text, (
                f"stale VAE attribution in {path.relative_to(ROOT)}"
            )


def _tracker_task_block(task_id: str, next_task_id: str) -> str:
    text = (ROOT / "planning/PROJECT_TRACKER.md").read_text(encoding="utf-8")
    start = text.index(f"### {task_id} ")
    end = text.index(f"### {next_task_id} ", start)
    return text[start:end]


def test_m1_gate_repair_has_an_executable_non_circular_status_path() -> None:
    assert "Status: `Done`" in _tracker_task_block("AI-109", "AI-110")
    assert "AI-205, AI-206, AI-208, and AI-214" in _tracker_task_block(
        "AI-109", "AI-110"
    )
    assert "Status: `Done`" in _tracker_task_block("AI-110", "AI-111")
    assert "allowed disabled decision" in _tracker_task_block("AI-110", "AI-111")
    assert "Status: `Ready`" in _tracker_task_block("AI-112", "AI-113")
    assert "Status: `Ready`" in _tracker_task_block("AI-113", "AI-114")
    assert "AI-112, AI-113" in _tracker_task_block("AI-114", "AI-115")


def test_gate_repair_records_preserve_policy_and_visual_boundaries() -> None:
    technical = (ROOT / "docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md").read_text(
        encoding="utf-8"
    )
    owner = (ROOT / "docs/M1_GATE_REPAIR_OWNER_BRIEF_2026-08-26.md").read_text(
        encoding="utf-8"
    )

    for text in (technical, owner):
        assert "m1-formal-policy-baseline-v0" in text
        assert "VISUAL_GENERATIVE_MODEL_BOUNDARY.md" in text
    assert "legacy-material-composition-diagnostic-v0" in technical
    assert "M1 is not declared complete" in technical
    assert "Immediate `stop` remains" in owner
