"""Rebuild and audit a trajectory-level train/validation/test corpus manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .trajectory_corpus import (
    TERMINATION_POLICY_STOP,
    discover_trajectory_shards,
    split_trajectory_paths,
    write_split_manifest,
)


def audit_corpus(args: argparse.Namespace) -> dict[str, object]:
    corpus_dirs = [Path(value).resolve() for value in args.corpus_dir]
    paths = sorted(
        {
            path.resolve()
            for corpus_dir in corpus_dirs
            for path in discover_trajectory_shards(corpus_dir)
        }
    )
    if not paths:
        raise ValueError(
            "no completed trajectory shards found under "
            + ", ".join(str(path) for path in corpus_dirs)
        )

    ratios = tuple(float(value) for value in args.split_ratios)
    splits = split_trajectory_paths(paths, seed=int(args.split_seed), ratios=ratios)
    manifest_path = Path(args.output_manifest).resolve()
    write_split_manifest(
        manifest_path,
        splits,
        seed=int(args.split_seed),
        ratios=ratios,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = dict(manifest["condition_audit"])
    provenance = dict(manifest["provenance_summary"])
    split_counts = {
        name: len(manifest["trajectory_ids"][name])
        for name in ("train", "validation", "test")
    }
    termination_modes = set(provenance["termination_modes"])
    every_split_has_required_conditions = all(
        not bool(audit["splits"][name]["missing_required_conditions"])
        for name in ("train", "validation", "test")
    )
    checks = {
        "trajectory_ids_unique_across_splits": bool(
            audit["checks"]["trajectory_ids_unique_across_splits"]
        ),
        "all_shards_deny_process_truth_training_input": bool(
            audit["checks"]["all_shards_deny_process_truth_training_input"]
        ),
        "validation_split_nonempty": split_counts["validation"] > 0,
        "test_split_nonempty": split_counts["test"] > 0,
        "all_required_overall_conditions_observed": not bool(
            audit["missing_overall_conditions"]
        ),
        "all_required_conditions_observed_in_each_split": (
            every_split_has_required_conditions
        ),
        "all_trajectories_use_live_scale_256_canvas": bool(
            provenance["all_trajectories_use_live_scale_256_canvas"]
        ),
        "all_transitions_have_compact_brush_context": (
            provenance["brush_context_fraction"] == 1.0
        ),
        "contains_policy_selected_stop_trajectory": (
            TERMINATION_POLICY_STOP in termination_modes
        ),
    }
    acceptance_check_names = (
        "trajectory_ids_unique_across_splits",
        "all_shards_deny_process_truth_training_input",
        "validation_split_nonempty",
        "test_split_nonempty",
        "all_required_overall_conditions_observed",
        "all_required_conditions_observed_in_each_split",
        "all_trajectories_use_live_scale_256_canvas",
        "all_transitions_have_compact_brush_context",
    )
    report = {
        "schema": "trajectory-corpus-audit-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "uncalibrated simulation-only corpus evidence",
        "corpus_dirs": [str(path) for path in corpus_dirs],
        "manifest": str(manifest_path),
        "split_counts": split_counts,
        "total_bytes": sum(path.stat().st_size for path in paths),
        "condition_audit": audit,
        "provenance_summary": provenance,
        "checks": checks,
        "ai108_acceptance_check_names": list(acceptance_check_names),
        "acceptance_candidate": all(checks[name] for name in acceptance_check_names),
        "downstream_composition_ready": bool(
            checks["contains_policy_selected_stop_trajectory"]
        ),
        "interpretation": (
            "Condition bins are evaluation labels derived from stored camera-derived "
            "posterior beliefs and selected actions. They are not rewards, aesthetic "
            "heuristics, likelihood terms, or calibrated physical ground truth."
        ),
    }
    report_path = Path(args.output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(report_path)
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed posterior trajectory shards, rebuild a whole-trajectory "
            "multilabel split, and write measured condition/provenance evidence."
        )
    )
    parser.add_argument("--corpus-dir", required=True, nargs="+")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--split-seed", type=int, default=104729)
    return parser


def main() -> None:
    report = audit_corpus(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
