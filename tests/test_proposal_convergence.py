from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from active_painter.proposal_convergence import (
    TopAction,
    equal_efe_stop_mass_control,
    first_action_rms_distance,
    main,
    proposal_convergence_config,
    run_proposal_convergence,
)


def test_equal_efe_control_exposes_candidate_frequency_as_effective_prior_mass() -> None:
    """Duplicating equal-EFE continuations reduces stop mass without model evidence."""

    config = proposal_convergence_config()
    rows = equal_efe_stop_mass_control((2, 4, 8, 16), config)

    assert [row["stopMass"] for row in rows] == sorted(
        [row["stopMass"] for row in rows], reverse=True
    )
    for row in rows:
        continuation_count = int(row["continuationCount"])
        expected = 0.5 / (0.5 + continuation_count)
        assert row["stopMass"] == pytest.approx(expected, abs=1e-15)
        assert row["effectiveContinuationPriorMass"] == continuation_count


def test_first_action_distance_is_evaluation_only_normalized_and_stop_aware() -> None:
    stop = TopAction(stop=True)
    mark = TopAction(False, 0.1, 0.2, 0.7, 0.8, 0.08, 0.5, 1.0)
    moved = TopAction(False, 0.2, 0.3, 0.6, 0.7, 0.10, 0.7, 1.0)

    assert first_action_rms_distance(stop, stop) == 0.0
    assert first_action_rms_distance(stop, mark) == 1.0
    assert first_action_rms_distance(mark, mark) == 0.0
    assert first_action_rms_distance(mark, moved) == pytest.approx(
        first_action_rms_distance(moved, mark)
    )
    assert 0.0 < first_action_rms_distance(mark, moved) < 1.0


def test_convergence_grid_is_deterministic_normalized_and_json_safe() -> None:
    arguments = {
        "candidate_counts": (4, 8),
        "horizons": (1, 2),
        "seeds": (0, 1),
        "learned_mixtures": (0.0, 0.5),
        "model_seed": 73,
    }

    first = run_proposal_convergence(**arguments)
    second = run_proposal_convergence(**arguments)
    first_without_time = deepcopy(first)
    second_without_time = deepcopy(second)
    first_without_time.pop("generatedAtUtc")
    second_without_time.pop("generatedAtUtc")

    assert first_without_time == second_without_time
    assert first["grid"]["cellCount"] == 16
    assert first["fixture"]["compositionEnabled"] is False
    assert first["fixture"]["motorForecastingIncluded"] is False
    assert first["interpretation"]["importanceCorrectionApplied"] is False
    assert first["interpretation"]["decisionUse"] == "none; evaluation-only audit"
    assert len(first["fixture"]["modelAndProposalStateSha256"]) == 64
    for cell in first["cells"]:
        assert cell["posteriorSum"] == pytest.approx(1.0, abs=1e-12)
        assert sum(cell["familyMass"].values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(cell["sourceMass"].values()) == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= cell["stopMass"] <= 1.0
        assert 0.0 < cell["topMass"] <= 1.0
    json.dumps(first, allow_nan=False)


def test_summaries_use_largest_budget_horizon_and_zero_mix_as_references() -> None:
    report = run_proposal_convergence(
        candidate_counts=(4, 8),
        horizons=(1, 2),
        seeds=(0, 1),
        learned_mixtures=(0.0, 0.5),
        model_seed=79,
    )
    summaries = report["summaries"]

    largest_budget = [
        row for row in summaries["budgetConvergence"] if row["candidateCount"] == 8
    ]
    max_horizon = [
        row for row in summaries["horizonConvergence"] if row["horizon"] == 2
    ]
    zero_mix = [
        row for row in summaries["mixtureEffect"] if row["learnedMix"] == 0.0
    ]

    assert largest_budget and all(row["meanAbsoluteStopMassDelta"] == 0.0 for row in largest_budget)
    assert largest_budget and all(row["medianFirstActionRmsDistance"] == 0.0 for row in largest_budget)
    assert max_horizon and all(row["meanAbsoluteStopMassDelta"] == 0.0 for row in max_horizon)
    assert zero_mix and all(row["meanAbsoluteStopMassDelta"] == 0.0 for row in zero_mix)
    assert zero_mix and all(row["meanLearnedPosteriorMass"] == 0.0 for row in zero_mix)


def test_convergence_grid_rejects_mix_one_because_it_removes_the_paired_control() -> None:
    with pytest.raises(ValueError, match="removes the paired control"):
        run_proposal_convergence(
            candidate_counts=(4,),
            horizons=(1,),
            seeds=(0,),
            learned_mixtures=(1.0,),
        )


def test_cli_retains_manifest_versions_config_report_and_failure_history() -> None:
    root = Path("runs/test_proposal_convergence_bundle")
    shutil.rmtree(root, ignore_errors=True)
    output = root / "proposal-convergence.json"
    arguments = [
        "--candidate-counts",
        "4,8",
        "--horizons",
        "1",
        "--seeds",
        "0",
        "--learned-mixtures",
        "0",
        "--model-seed",
        "83",
        "--output",
        str(output),
    ]
    try:
        assert main(arguments) == 0
        manifest = json.loads((root / "experiment-manifest.json").read_text())
        version = json.loads((root / "version-manifest.json").read_text())
        config = json.loads((root / "resolved-config.json").read_text())
        failure_records = [
            json.loads(line)
            for line in (root / "failure-log.jsonl").read_text().splitlines()
        ]

        assert manifest["manifest_revision"] == 2
        assert manifest["identity"]["status"] == "completed"
        assert manifest["failure_ids"] == ["F-AI111-20260804-001"]
        assert {artifact["path"] for artifact in manifest["artifacts"]} == {
            "proposal-convergence.json",
            "resolved-config.json",
            "version-manifest.json",
            "failure-log.jsonl",
        }
        assert version["schema_version"] == "version-manifest-v1"
        assert version["code"]["sourceFiles"]
        assert config["learned_proposal_mix"] == 0.0
        assert [record["event"] for record in failure_records] == [
            "opened",
            "accepted_limitation",
        ]
        assert failure_records[-1]["status"] == "accepted_limitation"
        with pytest.raises(FileExistsError, match="already exist"):
            main(arguments)
    finally:
        shutil.rmtree(root, ignore_errors=True)
