"""Reproducible finite-candidate convergence audit for painting-policy proposals.

This module is evaluation-only. It does not train a proposal, alter a policy
prior, add an importance weight, or participate in painting-policy selection.
It measures the posterior that the production equations assign *conditional on
the sampled candidate set* ``S``:

``Q(pi | S) = softmax(log p_stop(pi) - gamma * G(pi)), pi in S``.

Two fixtures are deliberately separated:

* an equal-EFE analytic control isolates the candidate-frequency effect; and
* a fixed, randomly initialized spatial model and material belief exercise the actual sampler,
  spatial EFE decomposition, stop prior, and policy-posterior implementation.

The spatial fixture is a mechanistic regression fixture, not evidence that an
untrained transition model or proposal paints well. Composition is disabled so
AI-110 cannot confound AI-111, motor forecasting is excluded so a finite motor
budget cannot hard-zero most candidates, and the proposal is never trained on
the observations it is evaluated against.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import PainterConfig
from .env import StrokeAction
from .policies import (
    Policy,
    PolicySampler,
    policy_posterior_from_efe,
    policy_stop_log_prior,
)
from .precision_beliefs import POLICY_PRECISION_KEY
from .policy_ranges import MARK_ACTION_RANGES
from .proposal import PolicyProposalNetwork, ProposalRecord
from .spatial_agent import SpatialActiveInferencePainter
from .spatial_state import SpatialCanvasState, project_material_fields


PROPOSAL_CONVERGENCE_SCHEMA = "proposal-convergence-v1"
PROPOSAL_CONVERGENCE_APPROXIMATION = (
    "fixed randomly initialized 8x8 spatial transition ensemble on a deterministic "
    "partially painted material belief; "
    "composition and motor forecasting excluded; learned proposal evaluated at "
    "initialization on fixed zero conditioning features; evaluation only"
)


@dataclass(frozen=True, slots=True)
class TopAction:
    """First painting action, the quantity receding-horizon control commits to."""

    stop: bool
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    width: float = 0.0
    amount: float = 0.0
    tone: float = 0.0


@dataclass(frozen=True, slots=True)
class ProposalConvergenceCell:
    candidate_count: int
    horizon: int
    seed: int
    learned_mix: float
    posterior_sum: float
    stop_mass: float
    top_mass: float
    posterior_entropy_nats: float
    effective_candidate_count: float
    top_index: int
    top_family: str
    top_source: str
    top_efe_nats: float
    top_action: TopAction
    family_mass: dict[str, float]
    source_mass: dict[str, float]


def proposal_convergence_config() -> PainterConfig:
    """Small declared fixture that isolates finite-candidate policy inference."""

    return PainterConfig(
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_transition_mode="dense_grid",
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_enabled=False,
        composition_gap_precision=0.0,
        canvas_latent_channels=4,
        relational_latent_dim=6,
        learned_proposal_hidden_dim=12,
        candidate_policies=128,
        planning_horizon=5,
        passage_proposal_mix=0.35,
        passage_plan_proposal_mix=0.0,
        proposal_low_coverage_mix=0.5,
        stroke_tone_prior=1.0,
        modality_normalization_enabled=True,
    )


def policy_family(policy: Policy) -> str:
    if policy.actions[0].stop:
        return "stop"
    if policy.passage_plan is not None:
        return "passage_plan"
    if policy.passage is not None:
        return "passage"
    if len(policy.actions) > 2:
        return "mark_sequence"
    return "mark"


def first_action(policy: Policy) -> TopAction:
    action = policy.actions[0]
    if action.stop:
        return TopAction(stop=True)
    return TopAction(
        stop=False,
        x0=float(action.x0),
        y0=float(action.y0),
        x1=float(action.x1),
        y1=float(action.y1),
        width=float(action.width),
        amount=float(action.amount),
        tone=float(action.tone),
    )


def first_action_rms_distance(left: TopAction, right: TopAction) -> float:
    """Evaluation-only normalized distance between committed first actions.

    This number is never read by the agent. Coordinates and tone already live
    on unit intervals; width and amount are normalized by their representational
    ranges. Stop/non-stop disagreement is assigned the maximum distance 1.
    """

    if left.stop or right.stop:
        return 0.0 if left.stop and right.stop else 1.0
    width_range = MARK_ACTION_RANGES["width"]
    amount_range = MARK_ACTION_RANGES["amount"]
    left_values = np.asarray(
        [
            left.x0,
            left.y0,
            left.x1,
            left.y1,
            (left.width - width_range.low) / width_range.width,
            (left.amount - amount_range.low) / amount_range.width,
            left.tone,
        ],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [
            right.x0,
            right.y0,
            right.x1,
            right.y1,
            (right.width - width_range.low) / width_range.width,
            (right.amount - amount_range.low) / amount_range.width,
            right.tone,
        ],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(np.square(left_values - right_values))))


def equal_efe_stop_mass_control(
    candidate_counts: Sequence[int],
    config: PainterConfig,
) -> list[dict[str, float | int]]:
    """Exact candidate-count control with equal EFE and flat continuation prior.

    Coverage is fixed at the declared stop-prior midpoint, so the immediate-stop
    log prior is ``log(0.5)`` while every continuation has log prior zero. Adding
    continuations must therefore reduce stop mass even though no likelihood,
    preference, precision, or EFE term changed.
    """

    stop = Policy((StrokeAction.stop_action(),))
    coverage = float(config.minimum_stop_coverage)
    stop_log_prior = policy_stop_log_prior(stop, coverage, config)
    rows: list[dict[str, float | int]] = []
    for candidate_count in _positive_unique(candidate_counts, "candidate counts"):
        priors = torch.zeros(candidate_count, dtype=torch.float64)
        priors[0] = stop_log_prior
        posterior = policy_posterior_from_efe(
            torch.zeros(candidate_count, dtype=torch.float64),
            priors,
            config.policy_precision,
        )
        rows.append(
            {
                "candidateCount": candidate_count,
                "continuationCount": candidate_count - 1,
                "stopLogPrior": float(stop_log_prior),
                "stopMass": float(posterior[0].item()),
                "oneContinuationMass": (
                    float(posterior[1].item()) if candidate_count > 1 else 0.0
                ),
                "effectiveContinuationPriorMass": float(
                    (candidate_count - 1) * math.exp(0.0)
                ),
            }
        )
    return rows


def _positive_unique(values: Iterable[int], name: str) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in values}))
    if not normalized or normalized[0] <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return normalized


def _mixtures(values: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(sorted({float(value) for value in values}))
    if not normalized or normalized[0] < 0.0 or normalized[-1] >= 1.0:
        raise ValueError("learned mixtures must lie in [0, 1); 1 removes the paired control")
    return normalized


def _source(record: ProposalRecord | None) -> str:
    return "unmodelled" if record is None else str(record.source)


def _state_hash(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode("utf-8"))
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _material_fixture_belief(
    config: PainterConfig,
    target_coverage: float = 0.84,
) -> SpatialCanvasState:
    """Deterministic partially painted state that keeps stop posterior measurable."""

    grid = int(config.spatial_grid_size)
    material = np.zeros(
        (config.spatial_material_channels, grid, grid), dtype=np.float32
    )
    if material.shape[0] > 3:
        material[3] = float(config.canvas_ground_tone)
    cells = grid * grid
    covered = min(cells, max(0, int(round(float(target_coverage) * cells))))
    flat = material.reshape(material.shape[0], -1)
    flat[0, :covered] = float(config.thickness_scale)
    if material.shape[0] > 1:
        flat[1, :covered] = 0.8 * float(config.thickness_scale)
    if material.shape[0] > 2:
        flat[2, :covered] = 0.5 * float(config.thickness_scale)
    if material.shape[0] > 3:
        flat[3, :covered] = 1.0
    projected = project_material_fields(material, config)
    return SpatialCanvasState(
        material=projected,
        logvar=np.full_like(projected, -20.0, dtype=np.float32),
    )


def _cell(
    *,
    condition: PainterConfig,
    seed: int,
    policies: Sequence[Policy],
    records: Sequence[ProposalRecord | None],
    totals: Sequence[float],
    posterior: np.ndarray,
) -> ProposalConvergenceCell:
    if len(policies) != len(records) or len(policies) != posterior.size:
        raise ValueError("policies, proposal records, and posterior must align")
    if not np.isfinite(posterior).all() or not np.isfinite(np.asarray(totals)).all():
        raise ValueError("convergence fixture produced a non-finite EFE or posterior")
    top_index = int(np.argmax(posterior))
    family_mass: dict[str, float] = {}
    source_mass: dict[str, float] = {}
    for policy, record, probability in zip(policies, records, posterior.tolist()):
        family = policy_family(policy)
        source = _source(record)
        family_mass[family] = family_mass.get(family, 0.0) + float(probability)
        source_mass[source] = source_mass.get(source, 0.0) + float(probability)
    positive = posterior[posterior > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    effective = float(1.0 / np.sum(np.square(posterior)))
    return ProposalConvergenceCell(
        candidate_count=len(policies),
        horizon=int(condition.planning_horizon),
        seed=int(seed),
        learned_mix=float(condition.learned_proposal_mix),
        posterior_sum=float(posterior.sum()),
        stop_mass=float(family_mass.get("stop", 0.0)),
        top_mass=float(posterior[top_index]),
        posterior_entropy_nats=entropy,
        effective_candidate_count=effective,
        top_index=top_index,
        top_family=policy_family(policies[top_index]),
        top_source=_source(records[top_index]),
        top_efe_nats=float(totals[top_index]),
        top_action=first_action(policies[top_index]),
        family_mass=family_mass,
        source_mass=source_mass,
    )


def run_proposal_convergence(
    *,
    candidate_counts: Sequence[int] = (8, 16, 32, 64, 128),
    horizons: Sequence[int] = (1, 3, 5),
    seeds: Sequence[int] = tuple(range(8)),
    learned_mixtures: Sequence[float] = (0.0, 0.25, 0.5),
    model_seed: int = 104729,
    config: PainterConfig | None = None,
    proposal_state: Mapping[str, torch.Tensor] | None = None,
    fixture_coverage: float = 0.84,
) -> dict[str, object]:
    """Run the fixed-model candidate convergence grid and return JSON-safe data."""

    counts = _positive_unique(candidate_counts, "candidate counts")
    depths = _positive_unique(horizons, "horizons")
    seed_values = tuple(sorted({int(seed) for seed in seeds}))
    if not seed_values:
        raise ValueError("seeds must not be empty")
    mixtures = _mixtures(learned_mixtures)
    base = config or proposal_convergence_config()
    base = replace(
        base,
        candidate_policies=max(counts),
        planning_horizon=max(depths),
        composition_enabled=False,
        composition_gap_precision=0.0,
        passage_plan_proposal_mix=0.0,
    )
    agent = SpatialActiveInferencePainter(base, seed=model_seed, device="cpu")
    agent.belief = _material_fixture_belief(base, fixture_coverage)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed + 5011)
        proposal = PolicyProposalNetwork(base)
    if proposal_state is not None:
        proposal.load_state_dict(dict(proposal_state))
    proposal.eval()
    features = torch.zeros(1, proposal.input_dim, dtype=torch.float32)
    coverage_field = agent.belief.coverage(base.paint_presence_threshold)
    believed_coverage = agent.belief.material_coverage_mean(base.paint_presence_threshold)
    policy_precision = float(agent.precision_ledger.mean(POLICY_PRECISION_KEY))
    cells: list[ProposalConvergenceCell] = []

    for horizon in depths:
        for learned_mix in mixtures:
            for candidate_count in counts:
                condition = replace(
                    base,
                    candidate_policies=candidate_count,
                    planning_horizon=horizon,
                    learned_proposal_mix=learned_mix,
                )
                for seed in seed_values:
                    proposal.seed_generator(model_seed + 5011 + seed)
                    sampler = PolicySampler(condition, seed=seed, learned_proposal=proposal)
                    policies = sampler.sample(
                        coverage_field,
                        belief_features=features,
                    )
                    records = sampler.last_proposal_records
                    components = agent.efe.evaluate_batch(agent.belief, policies)
                    totals = [float(component.total) for component in components]
                    log_prior = torch.tensor(
                        [
                            policy_stop_log_prior(
                                policy,
                                believed_coverage,
                                condition,
                                agent.gap_increment,
                            )
                            for policy in policies
                        ],
                        dtype=torch.float64,
                    )
                    posterior = policy_posterior_from_efe(
                        torch.tensor(totals, dtype=torch.float64),
                        log_prior,
                        policy_precision,
                    ).numpy()
                    cells.append(
                        _cell(
                            condition=condition,
                            seed=seed,
                            policies=policies,
                            records=records,
                            totals=totals,
                            posterior=posterior,
                        )
                    )

    summaries = summarize_convergence(cells)
    report: dict[str, object] = {
        "schemaVersion": PROPOSAL_CONVERGENCE_SCHEMA,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "posterior": "Q(pi | sampled candidate set S)",
            "proposalRole": "computational sampling distribution r(pi | belief)",
            "mixtureRole": "computational candidate-budget split, not a policy prior",
            "importanceCorrectionApplied": False,
            "topAction": (
                "posterior-argmax first painting action; evaluation metric only, "
                "while the bare agent may sample from the posterior"
            ),
            "decisionUse": "none; evaluation-only audit",
            "approximation": PROPOSAL_CONVERGENCE_APPROXIMATION,
        },
        "grid": {
            "candidateCounts": list(counts),
            "horizons": list(depths),
            "seeds": list(seed_values),
            "learnedMixtures": list(mixtures),
            "cellCount": len(cells),
        },
        "fixture": {
            "modelSeed": int(model_seed),
            "modelAndProposalStateSha256": _state_hash(agent.dynamics, proposal),
            "policyPrecision": policy_precision,
            "believedMaterialCoverage": float(believed_coverage),
            "requestedFixtureCoverage": float(fixture_coverage),
            "fixtureBeliefLogVariance": -20.0,
            "proposalConditioning": "fixed zero features; sampling allowed, training forbidden",
            "proposalState": "loaded checkpoint state" if proposal_state is not None else "initialization",
            "compositionEnabled": False,
            "motorForecastingIncluded": False,
            "strokeTonePrior": base.stroke_tone_prior,
            "resolvedConfig": asdict(base),
        },
        "equalEfeControl": equal_efe_stop_mass_control(counts, base),
        "cells": [_cell_dict(cell) for cell in cells],
        "summaries": summaries,
    }
    # Standard JSON rejects NaN/Infinity; validate before returning evidence.
    json.dumps(report, allow_nan=False)
    return report


def _cell_dict(cell: ProposalConvergenceCell) -> dict[str, object]:
    payload = asdict(cell)
    return {
        "candidateCount": payload["candidate_count"],
        "horizon": payload["horizon"],
        "seed": payload["seed"],
        "learnedMix": payload["learned_mix"],
        "posteriorSum": payload["posterior_sum"],
        "stopMass": payload["stop_mass"],
        "topMass": payload["top_mass"],
        "posteriorEntropyNats": payload["posterior_entropy_nats"],
        "effectiveCandidateCount": payload["effective_candidate_count"],
        "topIndex": payload["top_index"],
        "topFamily": payload["top_family"],
        "topSource": payload["top_source"],
        "topEfeNats": payload["top_efe_nats"],
        "topAction": payload["top_action"],
        "familyMass": payload["family_mass"],
        "sourceMass": payload["source_mass"],
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _pairwise_action_distances(cells: Sequence[ProposalConvergenceCell]) -> list[float]:
    return [
        first_action_rms_distance(left.top_action, right.top_action)
        for index, left in enumerate(cells)
        for right in cells[index + 1 :]
    ]


def summarize_convergence(cells: Sequence[ProposalConvergenceCell]) -> dict[str, object]:
    """Aggregate budget, horizon, mixture, and seed comparisons without thresholds."""

    if not cells:
        return {
            "budgetConvergence": [],
            "horizonConvergence": [],
            "mixtureEffect": [],
            "seedStability": [],
        }
    counts = sorted({cell.candidate_count for cell in cells})
    horizons = sorted({cell.horizon for cell in cells})
    mixtures = sorted({cell.learned_mix for cell in cells})
    seeds = sorted({cell.seed for cell in cells})
    by_key = {
        (cell.candidate_count, cell.horizon, cell.seed, cell.learned_mix): cell
        for cell in cells
    }
    max_count = counts[-1]
    max_horizon = horizons[-1]
    baseline_mix = mixtures[0]

    budget_rows: list[dict[str, object]] = []
    for horizon in horizons:
        for learned_mix in mixtures:
            for candidate_count in counts:
                pairs = [
                    (
                        by_key[(candidate_count, horizon, seed, learned_mix)],
                        by_key[(max_count, horizon, seed, learned_mix)],
                    )
                    for seed in seeds
                ]
                distances = [
                    first_action_rms_distance(current.top_action, reference.top_action)
                    for current, reference in pairs
                ]
                budget_rows.append(
                    {
                        "candidateCount": candidate_count,
                        "referenceCandidateCount": max_count,
                        "horizon": horizon,
                        "learnedMix": learned_mix,
                        "meanStopMass": _mean([current.stop_mass for current, _ in pairs]),
                        "referenceMeanStopMass": _mean(
                            [reference.stop_mass for _, reference in pairs]
                        ),
                        "meanAbsoluteStopMassDelta": _mean(
                            [abs(current.stop_mass - reference.stop_mass) for current, reference in pairs]
                        ),
                        "meanTopMass": _mean([current.top_mass for current, _ in pairs]),
                        "referenceMeanTopMass": _mean(
                            [reference.top_mass for _, reference in pairs]
                        ),
                        "topStopDecisionAgreementFraction": _mean(
                            [
                                float((current.top_family == "stop") == (reference.top_family == "stop"))
                                for current, reference in pairs
                            ]
                        ),
                        "topFamilyAgreementFraction": _mean(
                            [float(current.top_family == reference.top_family) for current, reference in pairs]
                        ),
                        "medianFirstActionRmsDistance": float(median(distances)),
                        "maxFirstActionRmsDistance": max(distances),
                    }
                )

    horizon_rows: list[dict[str, object]] = []
    for candidate_count in counts:
        for learned_mix in mixtures:
            for horizon in horizons:
                pairs = [
                    (
                        by_key[(candidate_count, horizon, seed, learned_mix)],
                        by_key[(candidate_count, max_horizon, seed, learned_mix)],
                    )
                    for seed in seeds
                ]
                distances = [
                    first_action_rms_distance(current.top_action, reference.top_action)
                    for current, reference in pairs
                ]
                horizon_rows.append(
                    {
                        "candidateCount": candidate_count,
                        "horizon": horizon,
                        "referenceHorizon": max_horizon,
                        "learnedMix": learned_mix,
                        "meanAbsoluteStopMassDelta": _mean(
                            [abs(current.stop_mass - reference.stop_mass) for current, reference in pairs]
                        ),
                        "topFamilyAgreementFraction": _mean(
                            [float(current.top_family == reference.top_family) for current, reference in pairs]
                        ),
                        "medianFirstActionRmsDistance": float(median(distances)),
                        "maxFirstActionRmsDistance": max(distances),
                    }
                )

    mixture_rows: list[dict[str, object]] = []
    for candidate_count in counts:
        for horizon in horizons:
            for learned_mix in mixtures:
                pairs = [
                    (
                        by_key[(candidate_count, horizon, seed, learned_mix)],
                        by_key[(candidate_count, horizon, seed, baseline_mix)],
                    )
                    for seed in seeds
                ]
                distances = [
                    first_action_rms_distance(current.top_action, reference.top_action)
                    for current, reference in pairs
                ]
                mixture_rows.append(
                    {
                        "candidateCount": candidate_count,
                        "horizon": horizon,
                        "learnedMix": learned_mix,
                        "referenceLearnedMix": baseline_mix,
                        "meanSignedStopMassDelta": _mean(
                            [current.stop_mass - reference.stop_mass for current, reference in pairs]
                        ),
                        "meanAbsoluteStopMassDelta": _mean(
                            [abs(current.stop_mass - reference.stop_mass) for current, reference in pairs]
                        ),
                        "topFamilyAgreementFraction": _mean(
                            [float(current.top_family == reference.top_family) for current, reference in pairs]
                        ),
                        "medianFirstActionRmsDistance": float(median(distances)),
                        "maxFirstActionRmsDistance": max(distances),
                        "meanLearnedPosteriorMass": _mean(
                            [current.source_mass.get("learned", 0.0) for current, _ in pairs]
                        ),
                    }
                )

    seed_rows: list[dict[str, object]] = []
    for candidate_count in counts:
        for horizon in horizons:
            for learned_mix in mixtures:
                group = [
                    by_key[(candidate_count, horizon, seed, learned_mix)] for seed in seeds
                ]
                family_counts: dict[str, int] = {}
                for cell in group:
                    family_counts[cell.top_family] = family_counts.get(cell.top_family, 0) + 1
                modal_family, modal_count = max(
                    family_counts.items(), key=lambda item: (item[1], item[0])
                )
                distances = _pairwise_action_distances(group)
                seed_rows.append(
                    {
                        "candidateCount": candidate_count,
                        "horizon": horizon,
                        "learnedMix": learned_mix,
                        "seedCount": len(group),
                        "modalTopFamily": modal_family,
                        "modalTopFamilyFraction": modal_count / len(group),
                        "meanStopMass": _mean([cell.stop_mass for cell in group]),
                        "stopMassStd": _std([cell.stop_mass for cell in group]),
                        "meanTopMass": _mean([cell.top_mass for cell in group]),
                        "topMassStd": _std([cell.top_mass for cell in group]),
                        "medianPairwiseFirstActionRmsDistance": (
                            float(median(distances)) if distances else 0.0
                        ),
                        "maxPairwiseFirstActionRmsDistance": max(distances, default=0.0),
                    }
                )

    return {
        "budgetConvergence": budget_rows,
        "horizonConvergence": horizon_rows,
        "mixtureEffect": mixture_rows,
        "seedStability": seed_rows,
    }


def _parse_csv(text: str, cast) -> tuple:
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


def _load_proposal_state(path: Path | None) -> Mapping[str, torch.Tensor] | None:
    if path is None:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only.
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("proposal_state"), dict):
        return payload["proposal_state"]
    if isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    raise ValueError("proposal checkpoint must be a driver payload or proposal state_dict")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = True
    source_paths = [
        Path(__file__).resolve(),
        root / "src" / "active_painter" / "policies.py",
        root / "src" / "active_painter" / "proposal.py",
        root / "src" / "active_painter" / "spatial_efe.py",
    ]
    return {
        "gitRevision": revision,
        "workingTreeDirty": dirty,
        "sourceFiles": {
            str(path.relative_to(root)).replace("\\", "/"): _file_sha256(path)
            for path in source_paths
        },
    }


def _failure_log_records(
    *,
    run_id: str,
    recorded_at: str,
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    control = list(report["equalEfeControl"])
    first_control = control[0]
    last_control = control[-1]
    budget_rows = list(report["summaries"]["budgetConvergence"])
    max_horizon = max(row["horizon"] for row in budget_rows)
    baseline_mix = min(row["learnedMix"] for row in budget_rows)
    smallest_budget_row = min(
        (
            row
            for row in budget_rows
            if row["horizon"] == max_horizon and row["learnedMix"] == baseline_mix
        ),
        key=lambda row: row["candidateCount"],
    )
    shared: dict[str, object] = {
        "schema_version": "failure-log-v1",
        "failure_id": "F-AI111-20260804-001",
        "recorded_at_utc": recorded_at,
        "run_id": run_id,
        "primary_category": "planner",
        "secondary_categories": ["uncertainty"],
        "failure_domain": "active_inference",
        "symptom": (
            "Immediate-stop posterior mass and deep-horizon winning geometry change "
            "as sampled candidate count changes while the model is fixed."
        ),
        "reproduction": {
            "status": "reproducible",
            "steps": [
                "Run active_painter.proposal_convergence with the recorded grid and model seed.",
                "Inspect equalEfeControl and budgetConvergence in proposal-convergence.json.",
            ],
            "expected": (
                "A proposal-invariant posterior remains stable after accounting for the "
                "declared policy prior and proposal distribution."
            ),
            "observed": (
                "No proposal correction is applied; equal-EFE stop mass changes from "
                f"{first_control['stopMass']:.8g} at {first_control['candidateCount']} candidates "
                f"to {last_control['stopMass']:.8g} at {last_control['candidateCount']}, and "
                f"the max-horizon smallest-budget median first-action distance to the "
                f"largest-budget reference is "
                f"{smallest_budget_row['medianFirstActionRmsDistance']:.8g}."
            ),
            "frequency": {"observed": 1, "attempted": 1},
            "minimal_case": "equal_efe_stop_mass_control in tests/test_proposal_convergence.py",
        },
        "suspected_causes": [
            {
                "cause": (
                    "The current posterior normalizes over enumerated candidate identities "
                    "without a log P(pi) - log r(pi | belief) correction."
                ),
                "confidence": "high",
                "evidence": [
                    "analytic equal-EFE multiplicity control",
                    "360-cell fixed-model convergence grid",
                ],
            }
        ],
        "severity": "major",
        "scientific_impact": "model_validity",
        "containment": (
            "Report Q(pi | sampled candidate set S), keep learned_proposal_mix at zero, "
            "and prohibit proposal-invariant posterior claims."
        ),
        "proposed_mitigation": (
            "M3 must declare the mixed discrete/continuous base measure, normalized "
            "P(pi) and r(pi | belief), then validate a proposal correction."
        ),
        "verification": (
            "Repeat nested and redrawn candidate-budget convergence on trained held-out "
            "checkpoints after the correction is implemented."
        ),
        "linked_tasks": ["AI-105", "AI-111", "AI-305"],
        "evidence": [
            {
                "path": "proposal-convergence.json",
                "state_revision": None,
                "policy_revision": None,
                "canvas_revision": None,
            },
            {
                "path": "docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md",
                "state_revision": None,
                "policy_revision": None,
                "canvas_revision": None,
            },
        ],
        "safety_response": {
            "external_stop_triggered": False,
            "motion_inhibited": False,
            "notes": "Evaluation-only run; no plant or controller was active.",
        },
    }
    opened = {
        **shared,
        "event": "opened",
        "event_revision": 1,
        "status": "open",
        "status_reason": "Deterministic reproduction recorded by the AI-111 audit.",
    }
    accepted = {
        **shared,
        "event": "accepted_limitation",
        "event_revision": 2,
        "status": "accepted_limitation",
        "status_reason": (
            "Bounded for M1 by the candidate-set-conditional interpretation and zero "
            "learned emission; correction remains mandatory for M3 invariance claims."
        ),
    }
    return [opened, accepted]


def _initial_manifest(args: argparse.Namespace, started_at: str, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "experiment-manifest-v1",
        "manifest_revision": 1,
        "supersedes_revision": None,
        "identity": {
            "run_id": run_id,
            "run_kind": "validation",
            "status": "running",
            "start_utc": started_at,
            "end_utc": None,
            "study_id": "AI-111",
            "replica_id": "fixed-model-grid",
            "parent_run_id": None,
        },
        "requested_grid": {
            "candidate_counts": args.candidate_counts,
            "horizons": args.horizons,
            "seeds": args.seeds,
            "learned_mixtures": args.learned_mixtures,
            "model_seed": args.model_seed,
        },
        "termination": {"status": "running", "reason": None},
    }


def _completed_manifest(
    *,
    initial: dict[str, object],
    report: dict[str, object],
    ended_at: str,
    elapsed_seconds: float,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    identity = dict(initial["identity"])
    identity.update({"status": "completed", "end_utc": ended_at})
    fixture = report["fixture"]
    return {
        "schema_version": "experiment-manifest-v1",
        "manifest_revision": 2,
        "supersedes_revision": 1,
        "identity": identity,
        "versions": {"path": "version-manifest.json"},
        "configuration": {
            "resolved_config_path": "resolved-config.json",
            "resolved_config_sha256": artifact_hashes["resolved-config.json"],
            "planner_mode": "spatial_material",
            "transition_mode": fixture["resolvedConfig"]["spatial_transition_mode"],
            "backend": "evaluation_only_no_plant",
            "compute_device": "cpu",
            "canvas_dimensions": [8, 8],
            "material_scale": fixture["resolvedConfig"]["thickness_scale"],
        },
        "randomness": {
            "model_seed": fixture["modelSeed"],
            "policy_sampler_seeds": report["grid"]["seeds"],
            "proposal_generator_seed_rule": "model_seed + 5011 + sampler_seed",
            "torch_model_initialization": fixture["modelSeed"],
            "plant_process_noise": "not_applicable",
        },
        "sensor_access": {
            "observation_mode": "deterministic_material_belief_fixture",
            "permitted_observations": ["fixed spatial material belief"],
            "derived_observations": ["material coverage field"],
            "evaluation_only_process_truth": [],
            "unavailable_real_platform_variables": "all; no plant was active",
        },
        "learning_state": {
            "learning_enabled": False,
            "proposal_checkpoint_loaded": fixture["proposalState"] == "loaded checkpoint state",
            "inherited_parameter_groups": [],
            "belief_reset": "fixed deterministic partial-coverage fixture",
            "replay_persisted": False,
            "optimizer_persisted": False,
            "training_data": "none",
        },
        "timing": {"wall_seconds": elapsed_seconds},
        "termination": {
            "status": "completed",
            "reason": "all declared convergence cells evaluated",
            "stop_source": "not_applicable",
            "safety_events": [],
        },
        "artifacts": [
            {
                "path": name,
                "format": Path(name).suffix.lstrip("."),
                "status": "present",
                "sha256": digest,
            }
            for name, digest in artifact_hashes.items()
        ],
        "active_inference": {
            "terms": [
                {
                    "name": "spatial_expected_free_energy",
                    "grounding": "efe_term",
                    "reported_in": "proposal-convergence.json",
                    "units": "nats",
                    "implementation_status": "fixed mechanistic fixture",
                },
                {
                    "name": "stop_policy_prior",
                    "grounding": "policy_prior",
                    "reported_in": "proposal-convergence.json",
                    "units": "log probability",
                    "implementation_status": "production equation",
                },
                {
                    "name": "candidate_set_policy_posterior",
                    "grounding": "policy_posterior",
                    "reported_in": "proposal-convergence.json",
                    "units": "probability conditional on sampled set",
                    "implementation_status": "accepted M1 approximation",
                },
            ],
            "proposal_decision_role": "none; computational candidate enumeration only",
        },
        "conventional_support": [
            {"name": "proposal_convergence_harness", "role": "evaluation_only analysis"}
        ],
        "approximations": [
            {
                "name": "finite_policy_proposal_set",
                "status": "accepted_limitation_for_M1",
                "effect": "posterior is Q(pi | sampled candidate set S)",
                "decision_task": "AI-111",
            },
            {
                "name": "randomly_initialized_fixed_model",
                "status": "mechanistic_fixture_only",
                "effect": "not evidence of learned painting quality",
                "decision_task": "AI-111",
            },
        ],
        "failure_ids": ["F-AI111-20260804-001"],
        "evidence_level": "provisional",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-counts", default="8,16,32,64,128")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--learned-mixtures", default="0,0.25,0.5")
    parser.add_argument("--model-seed", type=int, default=104729)
    parser.add_argument("--proposal-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.output.parent / "experiment-manifest.json"
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError("run artifacts already exist; choose a new run root or pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = args.output.parent.name
    started_at = datetime.now(timezone.utc).isoformat()
    initial = _initial_manifest(args, started_at, run_id)
    _write_json_atomic(manifest_path, initial)
    started_clock = time.perf_counter()
    try:
        report = run_proposal_convergence(
            candidate_counts=_parse_csv(args.candidate_counts, int),
            horizons=_parse_csv(args.horizons, int),
            seeds=_parse_csv(args.seeds, int),
            learned_mixtures=_parse_csv(args.learned_mixtures, float),
            model_seed=args.model_seed,
            proposal_state=_load_proposal_state(args.proposal_checkpoint),
        )
    except Exception as exc:
        failed = deepcopy(initial)
        failed["manifest_revision"] = 2
        failed["supersedes_revision"] = 1
        failed["identity"]["status"] = "failed"
        failed["identity"]["end_utc"] = datetime.now(timezone.utc).isoformat()
        failed["termination"] = {"status": "failed", "reason": repr(exc)}
        _write_json_atomic(manifest_path, failed)
        raise
    elapsed = time.perf_counter() - started_clock
    ended_at = datetime.now(timezone.utc).isoformat()
    resolved_path = args.output.parent / "resolved-config.json"
    version_path = args.output.parent / "version-manifest.json"
    failure_path = args.output.parent / "failure-log.jsonl"
    _write_json_atomic(args.output, report)
    _write_json_atomic(resolved_path, report["fixture"]["resolvedConfig"])
    repository = _repository_identity()
    _write_json_atomic(
        version_path,
        {
            "schema_version": "version-manifest-v1",
            "code": repository,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "modelAndProposalStateSha256": report["fixture"]["modelAndProposalStateSha256"],
        },
    )
    failure_records = _failure_log_records(
        run_id=run_id,
        recorded_at=ended_at,
        report=report,
    )
    _write_text_atomic(
        failure_path,
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in failure_records),
    )
    artifact_paths = {
        args.output.name: args.output,
        "resolved-config.json": resolved_path,
        "version-manifest.json": version_path,
        "failure-log.jsonl": failure_path,
    }
    artifact_hashes = {name: _file_sha256(path) for name, path in artifact_paths.items()}
    completed = _completed_manifest(
        initial=initial,
        report=report,
        ended_at=ended_at,
        elapsed_seconds=elapsed,
        artifact_hashes=artifact_hashes,
    )
    _write_json_atomic(manifest_path, completed)
    print(json.dumps({"output": str(args.output), "grid": report["grid"]}, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
