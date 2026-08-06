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
    )

    for path in SUPPORT_DOCUMENTS:
        text = path.read_text(encoding="utf-8").lower()
        for stale_claim in stale_claims:
            assert stale_claim not in text, f"stale claim in {path.relative_to(ROOT)}"
