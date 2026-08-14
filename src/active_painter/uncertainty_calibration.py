"""Held-out uncertainty calibration for local material transition models.

This module is conventional evaluation around the painting generative model.
It does not contribute a reward, preference, policy score, or EFE term.  The
validation split may fit one scalar likelihood-variance temperature; the test
split is evaluation-only.

Four different quantities are deliberately kept separate:

* learned conditional (aleatoric) variance;
* ensemble disagreement (epistemic variance);
* the fixed camera likelihood declared by ``spatial_observation_variance``;
* EFE precision multipliers, which are inventoried but never inserted into a
  predictive interval or transition NLL.

The corpus target is a camera-derived posterior mean.  Metrics are therefore
reported both for the transition distribution itself and for an explicitly
named approximation that adds the target posterior variance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .conditional_patch_vae import (
    ConditionalPatchExample,
    ConditionalPatchVAEConfig,
    ConditionalPatchVAEEnsemble,
    conditional_patch_examples_from_shards,
    make_conditional_patch_batch,
)
from .config import PainterConfig
from .models import LocalSpatialDynamicsEnsemble
from .mixture_transition import (
    LocalMixtureConfig,
    LocalMixtureDynamicsEnsemble,
    MIXTURE_CHECKPOINT_SCHEMA,
)
from .offline_train import _config_from_json, _pooled_training_config_payloads
from .spatial_inference import spatial_observation_variance
from .spatial_state import INDEPENDENT_MATERIAL_CHANNELS
from .trajectory_corpus import (
    SPLIT_NAMES,
    TrajectoryShard,
    load_trajectory_shard,
    transition_condition_labels,
)


CALIBRATION_REPORT_SCHEMA = "local-transition-calibration-report-v2"
CALIBRATION_PROTOCOL = "ai107-heldout-calibration-v2-exact-mixture-intervals"
LOG_TWO_PI = math.log(2.0 * math.pi)
EPSILON = 1.0e-12
ACTION_SUPPORT_THRESHOLD = 1.0e-2

# Provisional M2 gates, declared before inspecting AI-107 test results.
M2_THRESHOLDS: dict[str, float | int] = {
    "maximum_absolute_signed_mean_z": 0.20,
    "minimum_mean_squared_z": 0.64,
    "maximum_mean_squared_z": 1.44,
    "minimum_90_interval_coverage": 0.82,
    "maximum_90_interval_coverage": 0.96,
    "minimum_50_interval_coverage": 0.40,
    "maximum_50_interval_coverage": 0.60,
    "minimum_stratum_transitions": 12,
    "minimum_ood_epistemic_ratio": 1.50,
}

_PRECISION_CONFIG_FIELDS = {
    "terminal_coverage": "terminal_risk_precision",
    "observation_ambiguity": "ambiguity_precision",
    "transition": "transition_precision",
    "composition_gap": "composition_gap_precision",
    "canvas_latent_transition": "canvas_latent_transition_precision",
    "relational_transition": "relational_transition_precision",
    "motor_proprioceptive": "motor_modality_precision",
    "policy": "policy_precision",
}


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    example: ConditionalPatchExample
    labels: Mapping[str, str]
    trajectory_id: str


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    labels: Mapping[str, str]
    trajectory_id: str
    residual: np.ndarray
    learned_likelihood_variance: np.ndarray
    latent_variance: np.ndarray
    ensemble_variance: np.ndarray
    target_posterior_variance: np.ndarray
    fixed_camera_likelihood_variance: np.ndarray
    mixture_nll: np.ndarray
    action_support: np.ndarray
    component_residual: np.ndarray
    component_variance: np.ndarray
    component_log_weight: np.ndarray


def calibration_metrics(
    residual: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float | int | bool]:
    """Return Gaussian moment calibration metrics for aligned flat arrays."""

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    variance = np.asarray(variance, dtype=np.float64).reshape(-1)
    if residual.shape != variance.shape or residual.size == 0:
        raise ValueError("residual and variance need the same non-empty shape")
    variance = np.clip(variance, EPSILON, 1.0e12)
    z = residual / np.sqrt(variance)
    nll = 0.5 * (z * z + np.log(variance) + LOG_TWO_PI)
    absolute_z = np.abs(z)
    return {
        "element_count": int(residual.size),
        "finite": bool(
            np.all(np.isfinite(residual))
            and np.all(np.isfinite(variance))
            and np.all(np.isfinite(nll))
        ),
        "mean_gaussian_nll_nats": float(np.mean(nll)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_variance": float(np.mean(variance)),
        "mean_std": float(np.mean(np.sqrt(variance))),
        "signed_mean_z": float(np.mean(z)),
        "mean_squared_z": float(np.mean(z * z)),
        "mean_absolute_z": float(np.mean(absolute_z)),
        "coverage_50": float(np.mean(absolute_z <= 0.6744897501960817)),
        "coverage_80": float(np.mean(absolute_z <= 1.2815515655446004)),
        "coverage_90": float(np.mean(absolute_z <= 1.6448536269514722)),
        "coverage_95": float(np.mean(absolute_z <= 1.959963984540054)),
    }


def validation_variance_scale(
    residual: np.ndarray,
    variance: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 100.0,
) -> float:
    """Maximum-likelihood scalar variance temperature on validation data."""

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    variance = np.asarray(variance, dtype=np.float64).reshape(-1)
    if residual.shape != variance.shape or residual.size == 0:
        raise ValueError("residual and variance need the same non-empty shape")
    raw = float(np.mean((residual * residual) / np.clip(variance, EPSILON, None)))
    if not math.isfinite(raw):
        raise ValueError("validation variance scale is non-finite")
    return float(np.clip(raw, minimum, maximum))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )


def _component_arrays(
    records: Sequence[PredictionRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("predictive mixture calibration needs records")
    return tuple(
        np.concatenate([getattr(record, field) for record in records], axis=-1).astype(
            np.float64
        )
        for field in (
            "component_residual",
            "component_variance",
            "component_log_weight",
        )
    )  # type: ignore[return-value]


def mixture_variance_scale(
    records: Sequence[PredictionRecord],
    *,
    minimum: float = 0.02,
    maximum: float = 50.0,
) -> float:
    """Fit one component-variance scale by validation mixture NLL."""

    residual, variance, log_weight = _component_arrays(records)
    variance = np.clip(variance, EPSILON, 1.0e12)

    def objective(log_scale: float) -> float:
        scale = math.exp(log_scale)
        component_log_density = log_weight - 0.5 * (
            residual * residual / (variance * scale)
            + np.log(variance * scale)
            + LOG_TWO_PI
        )
        return float(np.mean(-_logsumexp(component_log_density, axis=0)))

    left = math.log(minimum)
    right = math.log(maximum)
    for _ in range(48):
        first = left + (right - left) / 3.0
        second = right - (right - left) / 3.0
        if objective(first) <= objective(second):
            right = second
        else:
            left = first
    return float(math.exp(0.5 * (left + right)))


def predictive_mixture_calibration(
    records: Sequence[PredictionRecord],
    *,
    variance_scale: float = 1.0,
) -> dict[str, float | int | bool]:
    """Evaluate NLL and central coverage through the full mixture CDF.

    For a continuous predictive mixture, ``F(y)`` is uniform when calibrated.
    A central alpha interval contains the observation exactly when its PIT lies
    in ``[(1-alpha)/2, (1+alpha)/2]``. This avoids an invalid Gaussian moment
    collapse for multimodal predictions.
    """

    residual, variance, log_weight = _component_arrays(records)
    variance = np.clip(variance * float(variance_scale), EPSILON, 1.0e12)
    normalized_log_weight = log_weight - _logsumexp(log_weight, axis=0)[None, ...]
    z = residual / np.sqrt(variance)
    cdf = 0.5 * (1.0 + torch.erf(torch.from_numpy(z) / math.sqrt(2.0)).numpy())
    pit = (np.exp(normalized_log_weight) * cdf).sum(axis=0).reshape(-1)
    component_log_density = normalized_log_weight - 0.5 * (
        residual * residual / variance + np.log(variance) + LOG_TWO_PI
    )
    nll = -_logsumexp(component_log_density, axis=0).reshape(-1)
    ordered = np.sort(pit)
    count = ordered.size
    lower = np.arange(count, dtype=np.float64) / max(1, count)
    upper = np.arange(1, count + 1, dtype=np.float64) / max(1, count)
    ks = max(float(np.max(np.abs(ordered - lower))), float(np.max(np.abs(ordered - upper))))

    def coverage(alpha: float) -> float:
        tail = 0.5 * (1.0 - alpha)
        return float(np.mean((pit >= tail) & (pit <= 1.0 - tail)))

    return {
        "element_count": int(count),
        "finite": bool(np.all(np.isfinite(pit)) and np.all(np.isfinite(nll))),
        "variance_scale": float(variance_scale),
        "mean_mixture_nll_nats": float(np.mean(nll)),
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "uniform_reference_pit_mean": 0.5,
        "uniform_reference_pit_variance": 1.0 / 12.0,
        "pit_ks_distance_from_uniform": ks,
        "coverage_50": coverage(0.50),
        "coverage_80": coverage(0.80),
        "coverage_90": coverage(0.90),
        "coverage_95": coverage(0.95),
    }


def precision_inventory(
    config_payload: Mapping[str, object],
    ledger_payload: object,
) -> dict[str, object]:
    """Classify precision terms without relabeling unobserved priors as posteriors."""

    ledger = ledger_payload if isinstance(ledger_payload, dict) else {}
    terms: dict[str, object] = {}
    for name, field in _PRECISION_CONFIG_FIELDS.items():
        declared = float(config_payload.get(field, 0.0))
        state = ledger.get(name, {}) if isinstance(ledger, dict) else {}
        state = state if isinstance(state, dict) else {}
        observations = int(float(state.get("observations", 0.0)))
        alpha = float(state.get("alpha", 0.0))
        beta = float(state.get("beta", 0.0))
        posterior_mean = alpha / beta if alpha > 0.0 and beta > 0.0 else declared
        beliefs_enabled = bool(config_payload.get("precision_beliefs_enabled", True))
        if name != "policy":
            beliefs_enabled = beliefs_enabled and bool(
                config_payload.get("modality_precision_beliefs_enabled", True)
            )
        if declared <= 0.0:
            status = "structurally_off"
        elif not beliefs_enabled:
            status = "fixed_multiplier"
            posterior_mean = declared
        elif observations <= 0:
            status = "declared_prior_unobserved"
            posterior_mean = declared
        else:
            status = "inferred_gamma_posterior"
        terms[name] = {
            "config_field": field,
            "declared_prior_mean_or_fixed_value": declared,
            "observations": observations,
            "reported_mean": posterior_mean,
            "status": status,
            "used_in_predictive_intervals_or_transition_nll": False,
        }
    return {
        "terms": terms,
        "boundary": (
            "EFE inverse-temperature multipliers are not transition noise and "
            "are excluded from NLL, z-scores, and predictive intervals"
        ),
    }


def ood_disagreement_summary(
    in_distribution: Sequence[float] | np.ndarray,
    out_of_distribution: Sequence[float] | np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float | int | bool]:
    threshold = float(
        M2_THRESHOLDS["minimum_ood_epistemic_ratio"]
        if threshold is None
        else threshold
    )
    ind = np.asarray(in_distribution, dtype=np.float64).reshape(-1)
    ood = np.asarray(out_of_distribution, dtype=np.float64).reshape(-1)
    if ind.size == 0 or ood.size == 0:
        raise ValueError("OOD disagreement comparison needs two non-empty arrays")
    mean_ind = float(np.mean(ind))
    mean_ood = float(np.mean(ood))
    ratio = mean_ood / max(mean_ind, EPSILON)
    return {
        "in_distribution_element_count": int(ind.size),
        "out_of_distribution_element_count": int(ood.size),
        "in_distribution_mean_epistemic_variance": mean_ind,
        "out_of_distribution_mean_epistemic_variance": mean_ood,
        "ood_to_in_distribution_ratio": ratio,
        "required_ratio": threshold,
        "passes_provisional_m2_gate": bool(ratio >= threshold),
    }


def _manifest_paths(
    manifest_path: Path | str,
) -> tuple[dict[str, list[Path]], dict[str, object]]:
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "trajectory-split-manifest-v1":
        raise ValueError("unsupported trajectory split manifest")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("trajectory split manifest has no file mapping")
    root = Path(str(payload["root"]))
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    return (
        {
            split: [root / str(item) for item in files.get(split, [])]
            for split in SPLIT_NAMES
        },
        payload,
    )


def _loaded_splits(
    manifest_path: Path | str,
) -> tuple[dict[str, list[TrajectoryShard]], dict[str, object], PainterConfig]:
    paths, manifest = _manifest_paths(manifest_path)
    loaded = {
        split: [load_trajectory_shard(path) for path in paths[split]]
        for split in SPLIT_NAMES
    }
    payloads: list[dict[str, object]] = []
    for shard in (item for split in SPLIT_NAMES for item in loaded[split]):
        candidate = shard.metadata.get("config")
        if not isinstance(candidate, dict):
            raise ValueError(f"trajectory {shard.trajectory_id!r} has no config")
        payloads.append(candidate)
    return loaded, manifest, _config_from_json(_pooled_training_config_payloads(payloads))


def _pooled_config_for_shards(shards: Iterable[TrajectoryShard]) -> PainterConfig:
    payloads: list[dict[str, object]] = []
    for shard in shards:
        candidate = shard.metadata.get("config")
        if not isinstance(candidate, dict):
            raise ValueError(f"trajectory {shard.trajectory_id!r} has no config")
        payloads.append(candidate)
    return _config_from_json(_pooled_training_config_payloads(payloads))


def _records(
    shards: Iterable[TrajectoryShard],
    config: PainterConfig,
) -> list[CalibrationRecord]:
    records: list[CalibrationRecord] = []
    for shard in shards:
        examples = conditional_patch_examples_from_shards([shard], config)
        if len(examples) != shard.transition_count:
            raise ValueError(
                f"trajectory {shard.trajectory_id!r} produced {len(examples)} "
                f"patches for {shard.transition_count} transitions; labels cannot align"
            )
        for index, example in enumerate(examples):
            labels = dict(transition_condition_labels(shard, index))
            longest_side = max(example.bounds.height, example.bounds.width)
            labels["patch_size"] = (
                "small" if longest_side <= 16 else "large" if longest_side > 24 else "medium"
            )
            records.append(
                CalibrationRecord(
                    example=example,
                    labels=labels,
                    trajectory_id=shard.trajectory_id,
                )
            )
    return records


def _masked_numpy(tensor: torch.Tensor, mask: torch.Tensor) -> list[np.ndarray]:
    channel_count = len(INDEPENDENT_MATERIAL_CHANNELS)
    array = tensor[:, :channel_count].detach().cpu().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    return [
        array[index, :, valid[index, 0]].reshape(channel_count, -1)
        for index in range(array.shape[0])
    ]


def _masked_component_numpy(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> list[np.ndarray]:
    """Extract [component, channel, valid-cell] arrays from padded batches."""

    channel_count = len(INDEPENDENT_MATERIAL_CHANNELS)
    array = tensor[:, :, :channel_count].detach().cpu().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    return [
        # Slice the batch axis before applying the 2-D boolean mask.  Combining
        # both operations in one NumPy index moves the advanced-indexed cell
        # axis ahead of the component/channel axes and silently scrambles the
        # predictive mixture.
        array[:, index][:, :, valid[index, 0]].reshape(
            array.shape[0], channel_count, -1
        )
        for index in range(array.shape[1])
    ]


def _fixed_camera_variance(
    batch_material: torch.Tensor,
    mask: torch.Tensor,
    config: PainterConfig,
) -> list[np.ndarray]:
    material = batch_material.detach().cpu().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    rows: list[np.ndarray] = []
    for index in range(material.shape[0]):
        variance = spatial_observation_variance(material[index], config)
        rows.append(variance[: len(INDEPENDENT_MATERIAL_CHANNELS), valid[index, 0]])
    return rows


def _normal_mixture_nll(
    target: torch.Tensor,
    means: torch.Tensor,
    logvars: torch.Tensor,
) -> torch.Tensor:
    # Component dimensions precede [batch, channel, row, column].
    component_dims = means.ndim - target.ndim
    expanded = target
    for _ in range(component_dims):
        expanded = expanded.unsqueeze(0)
    log_density = -0.5 * (
        (expanded - means).square() / logvars.exp() + logvars + LOG_TWO_PI
    )
    for _ in range(component_dims):
        log_density = torch.logsumexp(log_density, dim=0) - math.log(means.shape[0])
        means = means[0]
    return -log_density


def _prediction_records_from_tensors(
    records: Sequence[CalibrationRecord],
    batch,
    mean: torch.Tensor,
    learned: torch.Tensor,
    latent: torch.Tensor,
    ensemble: torch.Tensor,
    mixture_nll: torch.Tensor,
    config: PainterConfig,
    component_means: torch.Tensor,
    component_logvars: torch.Tensor,
    component_log_weights: torch.Tensor,
) -> list[PredictionRecord]:
    target = batch.next_material[:, : len(INDEPENDENT_MATERIAL_CHANNELS)]
    target_logvar = batch.next_material_logvar[:, : len(INDEPENDENT_MATERIAL_CHANNELS)]
    mask = batch.mask
    residual = _masked_numpy(target - mean, mask)
    learned_np = _masked_numpy(learned, mask)
    latent_np = _masked_numpy(latent, mask)
    ensemble_np = _masked_numpy(ensemble, mask)
    target_variance_np = _masked_numpy(target_logvar.exp(), mask)
    mixture_np = _masked_numpy(mixture_nll, mask)
    fixed_np = _fixed_camera_variance(batch.material, mask, config)
    component_target = target.unsqueeze(0)
    component_residual_np = _masked_component_numpy(
        component_target - component_means, mask
    )
    component_variance_np = _masked_component_numpy(component_logvars.exp(), mask)
    component_log_weight_np = _masked_component_numpy(component_log_weights, mask)
    valid_masks = batch.mask.detach().cpu().numpy().astype(bool)
    action_footprint = batch.action[:, 0].detach().cpu().numpy()
    support_np = [
        action_footprint[index, valid_masks[index, 0]] >= ACTION_SUPPORT_THRESHOLD
        for index in range(action_footprint.shape[0])
    ]
    result: list[PredictionRecord] = []
    for index, record in enumerate(records):
        result.append(
            PredictionRecord(
                labels=record.labels,
                trajectory_id=record.trajectory_id,
                residual=residual[index],
                learned_likelihood_variance=learned_np[index],
                latent_variance=latent_np[index],
                ensemble_variance=ensemble_np[index],
                target_posterior_variance=target_variance_np[index],
                fixed_camera_likelihood_variance=fixed_np[index],
                mixture_nll=mixture_np[index],
                action_support=support_np[index],
                component_residual=component_residual_np[index],
                component_variance=component_variance_np[index],
                component_log_weight=component_log_weight_np[index],
            )
        )
    return result


@torch.no_grad()
def predict_cnn(
    model: LocalSpatialDynamicsEnsemble,
    records: Sequence[CalibrationRecord],
    *,
    config: PainterConfig,
    device: torch.device,
    batch_size: int,
) -> list[PredictionRecord]:
    model.eval()
    predictions: list[PredictionRecord] = []
    for start in range(0, len(records), max(1, int(batch_size))):
        chunk = records[start : start + max(1, int(batch_size))]
        batch = make_conditional_patch_batch([item.example for item in chunk], device)
        means, logvars = model.forward_masked(batch.material, batch.action, batch.mask)
        channels = len(INDEPENDENT_MATERIAL_CHANNELS)
        means = means[:, :, :channels]
        logvars = logvars[:, :, :channels]
        mean = means.mean(dim=0)
        learned = logvars.exp().mean(dim=0)
        ensemble = means.var(dim=0, unbiased=False)
        mixture = _normal_mixture_nll(batch.next_material[:, :channels], means, logvars)
        component_log_weights = torch.full_like(
            means, -math.log(means.shape[0])
        )
        predictions.extend(
            _prediction_records_from_tensors(
                chunk,
                batch,
                mean,
                learned,
                torch.zeros_like(learned),
                ensemble,
                mixture,
                config,
                means,
                logvars,
                component_log_weights,
            )
        )
    return predictions


@torch.no_grad()
def predict_cvae(
    model: ConditionalPatchVAEEnsemble,
    records: Sequence[CalibrationRecord],
    *,
    config: PainterConfig,
    device: torch.device,
    batch_size: int,
    latent_samples: int,
    seed: int,
) -> list[PredictionRecord]:
    model.eval()
    predictions: list[PredictionRecord] = []
    for batch_index, start in enumerate(range(0, len(records), max(1, int(batch_size)))):
        chunk = records[start : start + max(1, int(batch_size))]
        batch = make_conditional_patch_batch([item.example for item in chunk], device)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(seed) + 1009 * batch_index)
        member_means: list[torch.Tensor] = []
        member_logvars: list[torch.Tensor] = []
        for member in model.members:
            means, logvars = member.prior_predictions(
                batch,
                samples=int(latent_samples),
                generator=generator,
            )
            member_means.append(means)
            member_logvars.append(logvars)
        means = torch.stack(member_means, dim=0)
        logvars = torch.stack(member_logvars, dim=0)
        channels = len(INDEPENDENT_MATERIAL_CHANNELS)
        means = means[:, :, :, :channels]
        logvars = logvars[:, :, :, :channels]
        member_mean = means.mean(dim=1)
        mean = member_mean.mean(dim=0)
        learned = logvars.exp().mean(dim=(0, 1))
        latent = means.var(dim=1, unbiased=False).mean(dim=0)
        ensemble = member_mean.var(dim=0, unbiased=False)
        mixture = _normal_mixture_nll(
            batch.next_material[:, :channels],
            means.reshape(-1, *means.shape[2:]),
            logvars.reshape(-1, *logvars.shape[2:]),
        )
        component_means = means.reshape(-1, *means.shape[2:])
        component_logvars = logvars.reshape(-1, *logvars.shape[2:])
        component_log_weights = torch.full_like(
            component_means, -math.log(component_means.shape[0])
        )
        predictions.extend(
            _prediction_records_from_tensors(
                chunk,
                batch,
                mean,
                learned,
                latent,
                ensemble,
                mixture,
                config,
                component_means,
                component_logvars,
                component_log_weights,
            )
        )
    return predictions


@torch.no_grad()
def predict_mixture(
    model: LocalMixtureDynamicsEnsemble,
    records: Sequence[CalibrationRecord],
    *,
    config: PainterConfig,
    device: torch.device,
    batch_size: int,
) -> list[PredictionRecord]:
    model.eval()
    predictions: list[PredictionRecord] = []
    for start in range(0, len(records), max(1, int(batch_size))):
        chunk = records[start : start + max(1, int(batch_size))]
        batch = make_conditional_patch_batch([item.example for item in chunk], device)
        moments = model.predictive_moments(batch.material, batch.action, batch.mask)
        mixture = model.exact_ensemble_mixture_nll(
            batch.material, batch.action, batch.next_material, batch.mask
        )
        independent = len(INDEPENDENT_MATERIAL_CHANNELS)
        component_means, component_logvars, component_log_weights = model.forward_masked(
            batch.material, batch.action, batch.mask
        )
        component_means = component_means[:, :, :, :independent].flatten(0, 1)
        component_logvars = component_logvars[:, :, :, :independent].flatten(0, 1)
        component_log_weights = (
            component_log_weights[:, :, :, :independent] - math.log(len(model.members))
        ).flatten(0, 1)
        predictions.extend(
            _prediction_records_from_tensors(
                chunk,
                batch,
                moments.mean[:, :independent],
                moments.likelihood_variance[:, :independent],
                torch.zeros_like(moments.likelihood_variance[:, :independent]),
                moments.epistemic_variance[:, :independent],
                mixture,
                config,
                component_means,
                component_logvars,
                component_log_weights,
            )
        )
    return predictions


def _flatten(
    records: Sequence[PredictionRecord],
    field: str,
    channel: int | None = None,
) -> np.ndarray:
    values = [np.asarray(getattr(record, field)) for record in records]
    if not values:
        return np.asarray([], dtype=np.float64)
    if channel is not None:
        values = [value[channel] for value in values]
    return np.concatenate([value.reshape(-1) for value in values]).astype(np.float64)


def _flatten_action_support(
    records: Sequence[PredictionRecord],
    field: str,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for record in records:
        value = np.asarray(getattr(record, field))
        support = np.asarray(record.action_support, dtype=bool)
        if value.ndim != 2 or value.shape[1] != support.size:
            raise ValueError("action-support mask does not align with prediction field")
        selected.append(value[:, support].reshape(-1))
    if not selected or not any(value.size for value in selected):
        return np.asarray([], dtype=np.float64)
    return np.concatenate(selected).astype(np.float64)


def _variance(records: Sequence[PredictionRecord], kind: str) -> np.ndarray:
    learned = _flatten(records, "learned_likelihood_variance")
    if kind == "learned_likelihood_only":
        return learned
    if kind == "model_total":
        return learned + _flatten(records, "latent_variance") + _flatten(records, "ensemble_variance")
    if kind == "model_total_plus_target_posterior":
        return (
            learned
            + _flatten(records, "latent_variance")
            + _flatten(records, "ensemble_variance")
            + _flatten(records, "target_posterior_variance")
        )
    if kind == "fixed_camera_likelihood_only":
        return _flatten(records, "fixed_camera_likelihood_variance")
    raise ValueError(f"unknown variance kind: {kind}")


def _action_support_variance(
    records: Sequence[PredictionRecord],
    kind: str,
) -> np.ndarray:
    learned = _flatten_action_support(records, "learned_likelihood_variance")
    if kind == "learned_likelihood_only":
        return learned
    if kind == "model_total":
        return (
            learned
            + _flatten_action_support(records, "latent_variance")
            + _flatten_action_support(records, "ensemble_variance")
        )
    if kind == "model_total_plus_target_posterior":
        return (
            learned
            + _flatten_action_support(records, "latent_variance")
            + _flatten_action_support(records, "ensemble_variance")
            + _flatten_action_support(records, "target_posterior_variance")
        )
    if kind == "fixed_camera_likelihood_only":
        return _flatten_action_support(records, "fixed_camera_likelihood_variance")
    raise ValueError(f"unknown variance kind: {kind}")


def _summary(
    records: Sequence[PredictionRecord],
    *,
    variance_scale: float | None = None,
    mixture_scale: float | None = None,
) -> dict[str, object]:
    residual = _flatten(records, "residual")
    variants = {
        kind: calibration_metrics(residual, _variance(records, kind))
        for kind in (
            "learned_likelihood_only",
            "model_total",
            "model_total_plus_target_posterior",
            "fixed_camera_likelihood_only",
        )
    }
    if variance_scale is not None:
        variants["validation_scaled_model_total"] = calibration_metrics(
            residual,
            _variance(records, "model_total") * float(variance_scale),
        )
    support_residual = _flatten_action_support(records, "residual")
    support_variants = {
        kind: calibration_metrics(
            support_residual,
            _action_support_variance(records, kind),
        )
        for kind in (
            "learned_likelihood_only",
            "model_total",
            "model_total_plus_target_posterior",
            "fixed_camera_likelihood_only",
        )
    }
    if variance_scale is not None:
        support_variants["validation_scaled_model_total"] = calibration_metrics(
            support_residual,
            _action_support_variance(records, "model_total") * float(variance_scale),
        )
    mixture_nll = _flatten(records, "mixture_nll")
    components = {
        field: float(np.mean(_flatten(records, field)))
        for field in (
            "learned_likelihood_variance",
            "latent_variance",
            "ensemble_variance",
            "target_posterior_variance",
            "fixed_camera_likelihood_variance",
        )
    }
    channels: dict[str, object] = {}
    for index, name in enumerate(INDEPENDENT_MATERIAL_CHANNELS):
        channel_residual = _flatten(records, "residual", index)
        learned = _flatten(records, "learned_likelihood_variance", index)
        latent = _flatten(records, "latent_variance", index)
        ensemble = _flatten(records, "ensemble_variance", index)
        channels[name] = calibration_metrics(channel_residual, learned + latent + ensemble)
    return {
        "transition_count": len(records),
        "trajectory_count": len({record.trajectory_id for record in records}),
        "exact_or_sample_mixture_nll_nats_mean": float(np.mean(mixture_nll)),
        "mixture_nll_definition": (
            "exact ensemble Gaussian mixture for CNN; finite-sample ensemble-by-latent "
            "prior-predictive Gaussian mixture approximation for cVAE"
        ),
        "mean_variance_components": components,
        "moment_gaussian_variants": variants,
        "predictive_mixture_calibration": {
            "unscaled": predictive_mixture_calibration(records),
            "validation_scaled": (
                predictive_mixture_calibration(
                    records, variance_scale=float(mixture_scale)
                )
                if mixture_scale is not None
                else None
            ),
            "definition": (
                "central interval coverage evaluated through the exact CNN/mixture "
                "Gaussian-mixture CDF or finite-sample cVAE prior-predictive mixture CDF"
            ),
        },
        "action_support_diagnostic": {
            "footprint_channel_threshold": ACTION_SUPPORT_THRESHOLD,
            "threshold_selection": (
                "declared after the initial all-patch result as a non-gating diagnostic; "
                "it is not used to revise the provisional M2 pass/fail result"
            ),
            "moment_gaussian_variants": support_variants,
            "exact_or_sample_mixture_nll_nats_mean": float(
                np.mean(_flatten_action_support(records, "mixture_nll"))
            ),
        },
        "per_independent_material_channel_model_total": channels,
    }


def _strata(records: Sequence[PredictionRecord]) -> dict[str, object]:
    families = (
        "tone",
        "surface",
        "width",
        "length",
        "amount",
        "motor",
        "region",
        "reach",
        "patch_size",
    )
    output: dict[str, object] = {}
    minimum = int(M2_THRESHOLDS["minimum_stratum_transitions"])
    for family in families:
        labels = sorted({record.labels.get(family, "unknown") for record in records})
        family_output: dict[str, object] = {}
        for label in labels:
            selected = [record for record in records if record.labels.get(family, "unknown") == label]
            residual = _flatten(selected, "residual")
            metrics = calibration_metrics(residual, _variance(selected, "model_total"))
            family_output[label] = {
                "transition_count": len(selected),
                "trajectory_count": len({record.trajectory_id for record in selected}),
                "eligible_for_provisional_gate": len(selected) >= minimum,
                "metrics": metrics,
            }
        output[family] = family_output
    if "wet_over_wet" not in output["surface"]:
        output["surface"]["wet_over_wet"] = {
            "transition_count": 0,
            "eligible_for_provisional_gate": False,
            "status": "structurally_unavailable",
            "reason": (
                "the current camera likelihood does not identify bulk wetness; exact "
                "simulator wetness is forbidden as a sensor-corpus label"
            ),
        }
    return output


def _gate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    variants = summary["moment_gaussian_variants"]
    assert isinstance(variants, dict)
    chosen = variants.get("validation_scaled_model_total", variants["model_total"])
    assert isinstance(chosen, dict)
    mixture_payload = summary["predictive_mixture_calibration"]
    assert isinstance(mixture_payload, dict)
    mixture_chosen = mixture_payload.get("validation_scaled") or mixture_payload["unscaled"]
    assert isinstance(mixture_chosen, dict)
    checks = {
        "finite": bool(chosen["finite"]) and bool(mixture_chosen["finite"]),
        "absolute_signed_mean_z": abs(float(chosen["signed_mean_z"]))
        <= float(M2_THRESHOLDS["maximum_absolute_signed_mean_z"]),
        "mean_squared_z": float(M2_THRESHOLDS["minimum_mean_squared_z"])
        <= float(chosen["mean_squared_z"])
        <= float(M2_THRESHOLDS["maximum_mean_squared_z"]),
        "predictive_mixture_coverage_90": float(M2_THRESHOLDS["minimum_90_interval_coverage"])
        <= float(mixture_chosen["coverage_90"])
        <= float(M2_THRESHOLDS["maximum_90_interval_coverage"]),
        "predictive_mixture_coverage_50": float(M2_THRESHOLDS["minimum_50_interval_coverage"])
        <= float(mixture_chosen["coverage_50"])
        <= float(M2_THRESHOLDS["maximum_50_interval_coverage"]),
    }
    return {"checks": checks, "passes_all": bool(all(checks.values()))}


def _load_cnn(path: Path, device: torch.device) -> tuple[LocalSpatialDynamicsEnsemble, PainterConfig, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("CNN checkpoint has no PainterConfig payload")
    config = _config_from_json(config_payload)
    model = LocalSpatialDynamicsEnsemble(config).to(device)
    model.load_state_dict(payload["dynamics_state"])
    return model, config, payload


def _load_cvae(path: Path, device: torch.device) -> tuple[ConditionalPatchVAEEnsemble, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("cVAE checkpoint has no model config")
    model = ConditionalPatchVAEEnsemble(ConditionalPatchVAEConfig(**model_config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _load_mixture(
    path: Path,
    device: torch.device,
) -> tuple[LocalMixtureDynamicsEnsemble, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != MIXTURE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported local mixture checkpoint")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("mixture checkpoint has no model config")
    model = LocalMixtureDynamicsEnsemble(LocalMixtureConfig(**model_config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


@torch.no_grad()
def multi_step_rollout_metrics(
    model,
    shards: Sequence[TrajectoryShard],
    *,
    model_kind: str,
    config: PainterConfig,
    device: torch.device,
    horizons: Sequence[int],
    latent_samples: int,
    seed: int,
) -> dict[str, object]:
    """Roll a learned local likelihood mean forward without state resets.

    Each start point is initialized from its camera-derived posterior, then the
    learned mean is recursively inserted into later local patches. Errors are
    evaluated over the cumulative selected-action footprint, not untouched
    canvas background. This remains a posterior-trajectory simulation metric;
    it does not read exact process canvas state.
    """

    requested = sorted({int(value) for value in horizons if int(value) > 0})
    if not requested:
        raise ValueError("multi-step evaluation needs a positive horizon")
    accumulators = {
        horizon: {"squared": 0.0, "absolute": 0.0, "elements": 0, "windows": 0}
        for horizon in requested
    }
    model.eval()
    for shard_index, shard in enumerate(shards):
        examples = conditional_patch_examples_from_shards([shard], config)
        if len(examples) != shard.transition_count:
            raise ValueError("trajectory patch count does not align for rollout")
        for start in range(shard.transition_count):
            maximum = min(max(requested), shard.transition_count - start)
            rolling_material = np.asarray(shard.state_material[start], dtype=np.float32).copy()
            rolling_logvar = np.asarray(shard.state_logvar[start], dtype=np.float32).copy()
            cumulative_support = np.zeros(rolling_material.shape[-2:], dtype=bool)
            for offset in range(maximum):
                index = start + offset
                source = examples[index]
                rows, cols = source.bounds.slices()
                example = replace(
                    source,
                    material=rolling_material[:, rows, cols].copy(),
                    material_logvar=rolling_logvar[:, rows, cols].copy(),
                )
                batch = make_conditional_patch_batch([example], device)
                if model_kind == "cnn":
                    means, logvars = model.forward_masked(
                        batch.material, batch.action, batch.mask
                    )
                    prediction = means.mean(dim=0)
                    variance = logvars.exp().mean(dim=0) + means.var(
                        dim=0, unbiased=False
                    )
                elif model_kind == "cvae":
                    member_means: list[torch.Tensor] = []
                    member_logvars: list[torch.Tensor] = []
                    for member_index, member in enumerate(model.members):
                        generator = torch.Generator(device=device.type).manual_seed(
                            int(seed)
                            + 100_003 * shard_index
                            + 1_009 * start
                            + 97 * offset
                            + 17 * member_index
                        )
                        means, logvars = member.prior_predictions(
                            batch, samples=int(latent_samples), generator=generator
                        )
                        member_means.append(means)
                        member_logvars.append(logvars)
                    means = torch.stack(member_means, dim=0)
                    logvars = torch.stack(member_logvars, dim=0)
                    member_mean = means.mean(dim=1)
                    prediction = member_mean.mean(dim=0)
                    variance = (
                        logvars.exp().mean(dim=(0, 1))
                        + means.var(dim=1, unbiased=False).mean(dim=0)
                        + member_mean.var(dim=0, unbiased=False)
                    )
                elif model_kind == "mixture":
                    moments = model.predictive_moments(
                        batch.material, batch.action, batch.mask
                    )
                    prediction = moments.mean
                    variance = moments.total_variance
                else:
                    raise ValueError(f"unknown multi-step model kind: {model_kind}")
                height, width = source.bounds.height, source.bounds.width
                predicted_patch = prediction[0, :, :height, :width].detach().cpu().numpy()
                predicted_variance = variance[0, :, :height, :width].detach().cpu().numpy()
                rolling_material[:, rows, cols] = predicted_patch
                rolling_logvar[:, rows, cols] = np.log(
                    np.clip(predicted_variance, EPSILON, 1.0e6)
                )
                cumulative_support[rows, cols] |= source.action[0] >= ACTION_SUPPORT_THRESHOLD
                horizon = offset + 1
                if horizon not in accumulators:
                    continue
                target = np.asarray(shard.next_material[index, :4], dtype=np.float64)
                residual = rolling_material[:4].astype(np.float64) - target
                support = cumulative_support
                selected = residual[:, support]
                accumulator = accumulators[horizon]
                accumulator["squared"] += float(np.square(selected).sum())
                accumulator["absolute"] += float(np.abs(selected).sum())
                accumulator["elements"] += int(selected.size)
                accumulator["windows"] += 1
    result: dict[str, object] = {}
    for horizon, accumulator in accumulators.items():
        count = int(accumulator["elements"])
        result[str(horizon)] = {
            "horizon_marks": horizon,
            "window_count": int(accumulator["windows"]),
            "element_count": count,
            "rmse": (
                math.sqrt(float(accumulator["squared"]) / count) if count else None
            ),
            "mae": float(accumulator["absolute"]) / count if count else None,
        }
    return {
        "metric_region": (
            "cumulative action footprint at raster footprint >= "
            f"{ACTION_SUPPORT_THRESHOLD}"
        ),
        "state_initialization": "camera-derived posterior at every rollout window start",
        "process_truth_used": False,
        "horizons": result,
    }


@torch.no_grad()
def _amount_support_stress(
    model,
    records: Sequence[CalibrationRecord],
    *,
    model_kind: str,
    device: torch.device,
    latent_samples: int,
    seed: int,
) -> dict[str, object]:
    ind_values: list[np.ndarray] = []
    ood_values: list[np.ndarray] = []
    for index, record in enumerate(records):
        batch = make_conditional_patch_batch([record.example], device)
        shifted = batch.with_conditions(action=batch.action.clone())
        shifted.action[:, 4] = 1.5
        if model_kind == "cnn":
            ind_mean, _ = model.forward_masked(batch.material, batch.action, batch.mask)
            ood_mean, _ = model.forward_masked(batch.material, shifted.action, batch.mask)
            ind_epistemic = ind_mean[:, :, :4].var(dim=0, unbiased=False)
            ood_epistemic = ood_mean[:, :, :4].var(dim=0, unbiased=False)
        elif model_kind == "cvae":
            member_ind: list[torch.Tensor] = []
            member_ood: list[torch.Tensor] = []
            for member_index, member in enumerate(model.members):
                pair_seed = int(seed) + 1009 * index + 17 * member_index
                gen_ind = torch.Generator(device=device.type).manual_seed(pair_seed)
                gen_ood = torch.Generator(device=device.type).manual_seed(pair_seed)
                ind, _ = member.prior_predictions(batch, samples=latent_samples, generator=gen_ind)
                ood, _ = member.prior_predictions(shifted, samples=latent_samples, generator=gen_ood)
                member_ind.append(ind.mean(dim=0))
                member_ood.append(ood.mean(dim=0))
            ind_epistemic = torch.stack(member_ind)[:, :, :4].var(dim=0, unbiased=False)
            ood_epistemic = torch.stack(member_ood)[:, :, :4].var(dim=0, unbiased=False)
        elif model_kind == "mixture":
            ind_moments = model.predictive_moments(
                batch.material, batch.action, batch.mask
            )
            ood_moments = model.predictive_moments(
                batch.material, shifted.action, batch.mask
            )
            ind_epistemic = ind_moments.epistemic_variance[:, :4]
            ood_epistemic = ood_moments.epistemic_variance[:, :4]
        else:
            raise ValueError(f"unknown amount-support model kind: {model_kind}")
        ind_values.append(_masked_numpy(ind_epistemic, batch.mask)[0].reshape(-1))
        ood_values.append(_masked_numpy(ood_epistemic, batch.mask)[0].reshape(-1))
    result = ood_disagreement_summary(np.concatenate(ind_values), np.concatenate(ood_values))
    result.update(
        {
            "shift": "amount action channel forced from in-support [0,1] to 1.5",
            "interpretation": (
                "invalid-support sensitivity stress test; not a physical policy outcome "
                "and not sufficient as the meaningful OOD gate"
            ),
        }
    )
    return result


def _meaningful_motor_ood(
    checkpoint: Path,
    in_distribution_manifest: Path,
    shifted_manifest: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate a fixed-roll-trained CNN on held-out dynamic-roll conditions."""

    in_loaded, _, _ = _loaded_splits(in_distribution_manifest)
    shifted_loaded, _, _ = _loaded_splits(shifted_manifest)
    in_shards = in_loaded["validation"] + in_loaded["test"]
    shifted_shards = shifted_loaded["validation"] + shifted_loaded["test"]
    pooled_config = _pooled_config_for_shards(in_shards + shifted_shards)
    in_records = _records(in_shards, pooled_config)
    shifted_records = [
        record
        for record in _records(shifted_shards, pooled_config)
        if record.labels.get("motor")
        in {"upper_arm_roll_positive", "upper_arm_roll_negative"}
    ]
    if not shifted_records:
        raise ValueError("shift manifest has no held-out dynamic-roll transitions")
    model, model_config, payload = _load_cnn(checkpoint, device)
    in_predictions = predict_cnn(
        model,
        in_records,
        config=model_config,
        device=device,
        batch_size=batch_size,
    )
    shifted_predictions = predict_cnn(
        model,
        shifted_records,
        config=model_config,
        device=device,
        batch_size=batch_size,
    )
    result = ood_disagreement_summary(
        _flatten(in_predictions, "ensemble_variance"),
        _flatten(shifted_predictions, "ensemble_variance"),
    )
    provenance = payload.get("training_provenance", {})
    result.update(
        {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_training_provenance": provenance,
            "in_distribution_manifest": str(in_distribution_manifest.resolve()),
            "out_of_distribution_manifest": str(shifted_manifest.resolve()),
            "in_distribution_transition_count": len(in_records),
            "out_of_distribution_transition_count": len(shifted_records),
            "in_distribution_conditions": (
                "held-out neutral Cartesian IK and fixed +/-24-degree upper-arm roll"
            ),
            "out_of_distribution_conditions": (
                "held-out dynamic +/-32-degree upper-arm roll sweeps, absent from "
                "the checkpoint's fixed-condition training corpus"
            ),
            "status": (
                "positive_ood_sensitivity_evidence"
                if result["passes_provisional_m2_gate"]
                else "negative_ood_sensitivity_result"
            ),
        }
    )
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_calibration(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    loaded, manifest, corpus_config = _loaded_splits(args.manifest)
    records = {
        split: _records(loaded[split], corpus_config) for split in SPLIT_NAMES
    }
    if not records["validation"] or not records["test"]:
        raise ValueError("calibration requires non-empty validation and test splits")

    model_reports: dict[str, object] = {}
    if args.cnn_checkpoint:
        model, model_config, payload = _load_cnn(Path(args.cnn_checkpoint), device)
        predictions = {
            split: predict_cnn(
                model,
                records[split],
                config=model_config,
                device=device,
                batch_size=args.batch_size,
            )
            for split in SPLIT_NAMES
        }
        scale = validation_variance_scale(
            _flatten(predictions["validation"], "residual"),
            _variance(predictions["validation"], "model_total"),
        )
        mixture_scale = mixture_variance_scale(predictions["validation"])
        summaries = {
            split: _summary(
                predictions[split],
                variance_scale=scale,
                mixture_scale=mixture_scale,
            )
            for split in SPLIT_NAMES
        }
        model_reports["cnn"] = {
            "checkpoint": str(Path(args.cnn_checkpoint).resolve()),
            "validation_fitted_variance_scale": scale,
            "validation_fitted_mixture_component_variance_scale": mixture_scale,
            "validation": summaries["validation"],
            "test": summaries["test"],
            "train": summaries["train"],
            "test_strata": _strata(predictions["test"]),
            "test_provisional_m2_calibration_gate": _gate_summary(summaries["test"]),
            "amount_support_stress": _amount_support_stress(
                model,
                records["test"],
                model_kind="cnn",
                device=device,
                latent_samples=args.latent_samples,
                seed=args.seed,
            ),
            "precision_inventory": precision_inventory(
                payload.get("config", {}), payload.get("precision_ledger")
            ),
            "fixed_likelihood_inventory": {
                "camera_base_observation_std": float(model_config.base_observation_std),
                "camera_smear_observation_std": float(model_config.smear_observation_std),
                "camera_variance_status": (
                    "fixed functional likelihood assumption; reported separately from "
                    "learned transition aleatoric variance"
                ),
                "identity_transition_variance": float(
                    math.exp(model_config.local_identity_logvar)
                ),
                "identity_transition_variance_status": (
                    "fixed warm-up likelihood variance; not learned by this checkpoint"
                ),
            },
            "multi_step_rollout": {
                split: multi_step_rollout_metrics(
                    model,
                    loaded[split],
                    model_kind="cnn",
                    config=corpus_config,
                    device=device,
                    horizons=getattr(args, "rollout_horizons", (1, 2, 4, 8)),
                    latent_samples=getattr(args, "rollout_latent_samples", 4),
                    seed=int(args.seed) + index * 10_000,
                )
                for index, split in enumerate(SPLIT_NAMES)
            },
        }

    if args.cvae_checkpoint:
        model, payload = _load_cvae(Path(args.cvae_checkpoint), device)
        predictions = {
            split: predict_cvae(
                model,
                records[split],
                config=corpus_config,
                device=device,
                batch_size=args.batch_size,
                latent_samples=args.latent_samples,
                seed=args.seed + (0 if split == "validation" else 100_000),
            )
            for split in SPLIT_NAMES
        }
        scale = validation_variance_scale(
            _flatten(predictions["validation"], "residual"),
            _variance(predictions["validation"], "model_total"),
        )
        mixture_scale = mixture_variance_scale(predictions["validation"])
        summaries = {
            split: _summary(
                predictions[split],
                variance_scale=scale,
                mixture_scale=mixture_scale,
            )
            for split in SPLIT_NAMES
        }
        model_reports["cvae"] = {
            "checkpoint": str(Path(args.cvae_checkpoint).resolve()),
            "latent_prior_samples_per_member": int(args.latent_samples),
            "validation_fitted_variance_scale": scale,
            "validation_fitted_mixture_component_variance_scale": mixture_scale,
            "validation": summaries["validation"],
            "test": summaries["test"],
            "train": summaries["train"],
            "test_strata": _strata(predictions["test"]),
            "test_provisional_m2_calibration_gate": _gate_summary(summaries["test"]),
            "amount_support_stress": _amount_support_stress(
                model,
                records["test"],
                model_kind="cvae",
                device=device,
                latent_samples=args.latent_samples,
                seed=args.seed,
            ),
            "precision_inventory": {
                "status": "not_present_in_shadow_model_checkpoint",
                "boundary": "shadow cVAE predicts transitions and does not own EFE precisions",
            },
            "multi_step_rollout": {
                split: multi_step_rollout_metrics(
                    model,
                    loaded[split],
                    model_kind="cvae",
                    config=corpus_config,
                    device=device,
                    horizons=getattr(args, "rollout_horizons", (1, 2, 4, 8)),
                    latent_samples=getattr(args, "rollout_latent_samples", 4),
                    seed=int(args.seed) + 50_000 + index * 10_000,
                )
                for index, split in enumerate(SPLIT_NAMES)
            },
        }

    if args.mixture_checkpoint:
        model, payload = _load_mixture(Path(args.mixture_checkpoint), device)
        predictions = {
            split: predict_mixture(
                model,
                records[split],
                config=corpus_config,
                device=device,
                batch_size=args.batch_size,
            )
            for split in SPLIT_NAMES
        }
        scale = validation_variance_scale(
            _flatten(predictions["validation"], "residual"),
            _variance(predictions["validation"], "model_total"),
        )
        mixture_scale = mixture_variance_scale(predictions["validation"])
        summaries = {
            split: _summary(
                predictions[split],
                variance_scale=scale,
                mixture_scale=mixture_scale,
            )
            for split in SPLIT_NAMES
        }
        model_reports["mixture"] = {
            "checkpoint": str(Path(args.mixture_checkpoint).resolve()),
            "validation_fitted_variance_scale": scale,
            "validation_fitted_mixture_component_variance_scale": mixture_scale,
            "train": summaries["train"],
            "validation": summaries["validation"],
            "test": summaries["test"],
            "test_strata": _strata(predictions["test"]),
            "test_provisional_m2_calibration_gate": _gate_summary(summaries["test"]),
            "amount_support_stress": _amount_support_stress(
                model,
                records["test"],
                model_kind="mixture",
                device=device,
                latent_samples=args.latent_samples,
                seed=args.seed,
            ),
            "precision_inventory": {
                "status": "not_present_in_shadow_model_checkpoint",
                "boundary": "shadow mixture likelihood does not own EFE precisions",
            },
            "mixture_semantics": (
                "normalized identity-plus-consequence Gaussian mixture; learned "
                "mixing probabilities and component variances"
            ),
            "multi_step_rollout": {
                split: multi_step_rollout_metrics(
                    model,
                    loaded[split],
                    model_kind="mixture",
                    config=corpus_config,
                    device=device,
                    horizons=getattr(args, "rollout_horizons", (1, 2, 4, 8)),
                    latent_samples=getattr(args, "rollout_latent_samples", 4),
                    seed=int(args.seed) + 90_000 + index * 10_000,
                )
                for index, split in enumerate(SPLIT_NAMES)
            },
        }

    if not model_reports:
        raise ValueError("at least one checkpoint is required")
    report: dict[str, object] = {
        "schema": CALIBRATION_REPORT_SCHEMA,
        "protocol": CALIBRATION_PROTOCOL,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_schema": manifest.get("schema"),
        "device": str(device),
        "split_unit": "whole trajectory before patch extraction",
        "split_use": {
            "train": "model fitting only; not read by this evaluator",
            "validation": "fits one scalar model-total variance temperature",
            "test": "held-out metrics only; no parameter or scale fitting",
        },
        "test_independent_trajectory_warning": (
            "the test split has only three independent trajectories; cell counts are "
            "not independent sample counts and AI-109 must add data/capacity curves"
        ),
        "provisional_m2_thresholds_declared_before_test_inspection": M2_THRESHOLDS,
        "models": model_reports,
        "wet_over_wet_evidence": {
            "status": "structurally_unavailable",
            "reason": (
                "bulk wetness is not identified by the current camera likelihood; exact "
                "simulator wetness was not substituted as an oracle label"
            ),
        },
        "claim_boundary": (
            "uncalibrated simulation-only integration evidence; not hardware calibration, "
            "sensor equivalence, painting quality, or embodiment evidence"
        ),
    }
    motor_ood_args = (
        args.motor_ood_cnn_checkpoint,
        args.motor_ood_id_manifest,
        args.motor_ood_shift_manifest,
    )
    if any(motor_ood_args) and not all(motor_ood_args):
        raise ValueError(
            "meaningful motor OOD evaluation requires the checkpoint, ID manifest, "
            "and shifted manifest together"
        )
    if all(motor_ood_args):
        report["meaningful_motor_condition_ood"] = _meaningful_motor_ood(
            Path(args.motor_ood_cnn_checkpoint),
            Path(args.motor_ood_id_manifest),
            Path(args.motor_ood_shift_manifest),
            device=device,
            batch_size=args.batch_size,
        )
    _atomic_json(Path(args.output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AI-107 held-out transition calibration")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--cvae-checkpoint", default=None)
    parser.add_argument("--mixture-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--rollout-latent-samples", type=int, default=4)
    parser.add_argument("--motor-ood-cnn-checkpoint", default=None)
    parser.add_argument("--motor-ood-id-manifest", default=None)
    parser.add_argument("--motor-ood-shift-manifest", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_calibration(args)
    compact = {
        "schema": report["schema"],
        "output_models": list(report["models"]),
        "output": str(Path(args.output).resolve()),
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
