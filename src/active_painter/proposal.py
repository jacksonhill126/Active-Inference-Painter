"""Amortized candidate-policy proposal ``q_proposal(z_pi | belief)``.

WHAT THIS IS, IN CHARTER TERMS
-----------------------------
This module learns a PROPOSAL DISTRIBUTION over painting-policy latents. It is
the same category of object `policies.PolicySampler` already is (spec
factor-to-code row ``PROP-PAINT``, "Proposal only"): it decides which candidate
hypotheses exist to be scored, and it appears in NO expected-free-energy term, NO
variational-free-energy term, NO prior preference, and NO normalized ``p(pi)``
factor of the policy posterior. Selection still runs the identical
``softmax(log p_stop + painting_log_evidence + policy_log_priors)`` over whatever
candidate set it produced. Its only effect on a decision is which candidates the
posterior gets to see.

WHAT IT IS TRAINED TOWARD, AND WHY THAT IS NOT A REWARD
------------------------------------------------------
The target is the agent's own DECLARED BASE PAINTING-POLICY POSTERIOR over the
candidates it just enumerated (spec §10.5, first equation):

```math
w_i = softmax_i( log p_stop(pi_i) - gamma * G_base(pi_i) )
```

Every term is pre-existing and independently categorized: ``log p_stop`` is the
declared policy prior (``policies.policy_stop_log_prior``, row ``PRIOR-STOP``),
``G_base`` is the declared expected free energy (``spatial_efe``, row
``EFE-PIX``), and ``gamma`` is a declared precision (``config.policy_precision``,
row ``GAMMA-POLICY``). Nothing new enters it and none of its terms is rescaled.

The objective is self-normalized-importance-weighted maximum likelihood,

```math
L(theta) = - sum_i w_i log q_proposal(z_i | belief) / sum_i w_i
```

which equals ``KL( q_base(pi) || q_proposal(pi | belief) )`` up to the target's
own entropy, a constant in ``theta``. That makes it AMORTIZED VARIATIONAL
INFERENCE -- fitting a recognition density to a posterior the model already
computed -- and not a reward, a return, or a value function. A reward would have
to score outcome quality, accumulate into a return, or bootstrap a value
estimate; this does none of the three. The decisive asymmetry: **if the
EFE-induced posterior is wrong, the proposal faithfully learns the wrong thing.**
A reward pushes toward good outcomes regardless of the posterior; amortized
inference tracks the posterior regardless of outcomes. There is no scalar quality
signal anywhere in the gradient to rename (AGENTS.md, "Never rename an ordinary
reward or controller as active inference").

CORRESPONDENCE WITH THE REFERENCE'S HABIT PRIOR
-----------------------------------------------
This is the continuous, belief-conditioned generalization of the reference
implementation's discrete habit prior ``E``,
``active_inference.core.pomdp_extensions.habit_prior_from_counts``:

```python
return softmax(gamma * np.log(arr + 1.0))   # E = softmax(gamma log(counts + 1))
```

with two substitutions: self-normalized posterior weights ``w_i`` replace raw
visit counts, and a belief-conditioned network replaces the lookup table over a
finite action index. One difference is deliberate and is NOT a substitution: the
reference's ``E`` is summed into the policy posterior, whereas this proposal is
never summed into anything (see the AI-111 resolution in
``docs/GENERATIVE_MODEL_SPEC.md`` §10.2). Promoting it to a prior would
double-count, because it is trained toward a target that already contains
``log p_stop``.

DENSITIES ARE EXACTLY NORMALIZED, WHICH REQUIRED ONE DESIGN CHANGE
------------------------------------------------------------------
Every continuous factor is a LOGIT-NORMAL on its declared support interval
(``policy_ranges.PROPOSAL_SUPPORT``), not a clipped log-normal. Clipping a
density onto an interval places point masses on the boundary, which makes
``log q`` un-normalized and the divergence undefined -- a clipped log-normal
cannot legally be called a proposal density at all. The logit-normal has the
identical support, is absolutely continuous, is exactly normalized, and its
Jacobian is scored. Width keeps its scale-free character by being logit-normal in
``log(width)``, the absolutely-continuous analogue of the hand-written
log-uniform.

THE DIVERGENCE IS OVER THE LATENT, AND THE SHARED DECODER CANCELS EXACTLY
------------------------------------------------------------------------
``D_KL(learned || hand-written)`` is measured over the mark/passage LATENT, not
the emitted action tuple. Both proposals decode a latent through the identical
``policies.PolicySampler`` decoders, so every per-mark jitter, the polyline fit,
and every coordinate clip are factors of a common ``p(actions | z)`` that cancels
in the ratio. That is exact, not an approximation, and it is what makes the
metric tractable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn

from .config import PainterConfig
from .policy_ranges import (
    PROPOSAL_DIRECTION,
    PROPOSAL_FAMILIES,
    PROPOSAL_KINDS,
    PROPOSAL_SUPPORT,
    Interval,
    passage_stroke_count_range,
    proposal_support_for,
)


POLICY_PROPOSAL_VERSION = 1

# Declared out-of-support value of `log q_hand`. A learned sample cannot leave
# the hand-written support by construction (both are normalized over the same
# `PROPOSAL_SUPPORT` intervals and the categoricals are masked to the same
# declared config gates), so this floor exists to keep the divergence finite if a
# future config makes the two supports disagree, and the fraction of samples that
# hit it is reported beside the divergence rather than hidden.
PROPOSAL_LOG_DENSITY_FLOOR = -50.0

# Fixed categorical head width for `stroke_count`, masked at sample and score
# time to the config-derived range. Fixed rather than config-derived so the
# tensor shapes stay portable across `planning_horizon` changes and a checkpoint
# does not have to be discarded when the horizon moves.
PROPOSAL_STROKE_COUNT_SUPPORT = 8

# Number of wrap terms in the circular direction density (k in {-1, 0, +1}). The
# truncated sum is renormalized exactly over [-pi, pi], so the family is a
# genuine normalized density on the circle rather than an approximate one.
PROPOSAL_DIRECTION_WRAPS = 1

BELIEF_SOURCE_POSTERIOR = "canvas and relational posterior means"
BELIEF_SOURCE_UNINITIALIZED = "zero prior fallback (beliefs uninitialized)"
BELIEF_SOURCE_NO_HIERARCHY = "zero prior fallback (no hierarchy)"
BELIEF_SOURCE_SUMMARY = "zero prior fallback (summary planner)"
FALLBACK_BELIEF_SOURCES = frozenset(
    {
        BELIEF_SOURCE_UNINITIALIZED,
        BELIEF_SOURCE_NO_HIERARCHY,
        BELIEF_SOURCE_SUMMARY,
    }
)

BASE_TARGET_NAME = "base_efe_painting_posterior"

_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_TWO_PI = math.log(_SQRT_TWO_PI)

# Parameter names each family/kind factorizes over, in a fixed order so the
# per-factor diagnostics and the head dictionary stay aligned.
_MARK_PARAMETERS = ("center_x", "center_y", "direction", "length", "width", "amount")
_PASSAGE_SHARED_PARAMETERS = ("center_x", "center_y", "direction", "width", "amount")
_PASSAGE_KIND_PARAMETERS = ("length", "spacing")

# Parameters carried in log space, so the proposal stays scale-free where the
# hand-written proposal was log-uniform.
_LOG_SPACE_PARAMETERS = frozenset({"width"})


def _head_key(family: str, kind: str, parameter: str) -> str:
    if family == "mark":
        return f"mark.{parameter}"
    if parameter in _PASSAGE_KIND_PARAMETERS:
        return f"passage.{kind}.{parameter}"
    return f"passage.{parameter}"


def _parameters_for(family: str) -> tuple[str, ...]:
    if family == "mark":
        return _MARK_PARAMETERS
    return _PASSAGE_SHARED_PARAMETERS + _PASSAGE_KIND_PARAMETERS


# The two coordinate parameters share one reported factor, because the
# hand-written start-point proposal is a JOINT mixture over (x, y) that does not
# factorize -- reporting them separately would imply an independence the code
# does not have.
_FACTOR_KEY = {"center_x": "center", "center_y": "center"}


def _factor_key(parameter: str) -> str:
    return _FACTOR_KEY.get(parameter, parameter)


# --------------------------------------------------------------------------
# Dataclasses. Each carries an `approximation` string, per the register
# convention in docs/GENERATIVE_MODEL_SPEC.md §13.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProposalLatent:
    """One mark's or one passage's proposal coordinates.

    For ``family == "passage"`` these are the PassageLatent coordinates BEFORE
    `policies.fit_polyline_latent` runs. For ``family == "mark"``,
    ``center_x``/``center_y`` are the mark's START point and ``direction`` and
    ``length`` are the drawn heading and span -- i.e. exactly the coordinates
    `PolicySampler._stroke_draw` draws, so the two proposals are densities over
    the same random variable and the decoder cancels in their ratio.
    """

    family: str
    kind: str
    center_x: float
    center_y: float
    direction: float
    length: float
    spacing: float
    width: float
    amount: float
    stroke_count: int
    tone: float

    def value(self, parameter: str) -> float:
        return float(getattr(self, parameter))


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """Which proposal generated one candidate policy, and with what latents."""

    family: str
    source: str  # "hand" | "learned"
    latents: tuple[ProposalLatent, ...]
    approximation: str = (
        "latent-space record; the shared decoder p(actions | z) is a common factor "
        "of both proposals and is omitted from both densities"
    )


@dataclass(frozen=True, slots=True)
class ProposalFactorLogDensity:
    """Per-factor breakdown of one record's log proposal density, in nats."""

    total: float
    center: float
    direction: float
    length: float
    spacing: float
    width: float
    amount: float
    stroke_count: float
    kind: float
    tone: float
    latent_count: int
    in_support: bool
    approximation: str


@dataclass(slots=True)
class ProposalTrainingBatch:
    """One planning round's amortization target.

    `weights` are the self-normalized base-EFE policy posterior over the SAMPLED
    candidate set, aligned index-for-index with `records`. A `None` record marks
    a candidate the proposal is not responsible for (the immediate-stop policy,
    and passage-plan compounds, which stay hand-written).
    """

    features: torch.Tensor | None
    records: tuple[ProposalRecord | None, ...]
    weights: tuple[float, ...]
    target_name: str = BASE_TARGET_NAME
    target_support_fraction: float = 0.0
    belief_feature_source: str = BELIEF_SOURCE_UNINITIALIZED
    # The EMBODIED-REFINED posterior over the same candidate set. Carried for
    # measurement only and never trained on: it is supported on the stop
    # candidates plus at most `motor_forecast_candidates` forecast continuations,
    # so SNIS against it would collapse the proposal onto whatever the finite
    # motor budget happened to forecast. Its cross-entropy against the proposal is
    # reported so the gap between the two candidate targets is a number rather
    # than an assumption.
    refined_weights: tuple[float, ...] = ()
    approximation: str = (
        "self-normalized importance-weighted maximum likelihood; the target is the "
        "base-EFE policy posterior over the SAMPLED candidate support only, not over "
        "the full policy space, and not the embodied-refined posterior"
    )

    def modelled_indices(self) -> list[int]:
        return [
            index
            for index, record in enumerate(self.records)
            if record is not None and index < len(self.weights) and self.weights[index] > 0.0
        ]

    def modelled_mass(self) -> float:
        return float(sum(self.weights[index] for index in self.modelled_indices()))


@dataclass(frozen=True, slots=True)
class ProposalDivergence:
    """Monte-Carlo ``D_KL(learned || hand-written)`` for one family, in nats."""

    family: str
    divergence_nats: float
    sample_count: int
    out_of_support_fraction: float
    approximation: str = (
        "Monte-Carlo KL over the passage/mark LATENT with the shared decoder "
        "cancelled exactly; the hand-written density's coverage-cell boundary atoms "
        "are evaluated on the pre-clip parameterization"
    )


# --------------------------------------------------------------------------
# Declared target: the base-EFE painting policy posterior.
# --------------------------------------------------------------------------


def base_efe_policy_posterior(
    totals: list[float] | np.ndarray,
    stop_log_priors: list[float] | np.ndarray,
    policy_precision: float,
) -> np.ndarray:
    """``softmax(log p_stop - gamma * G)`` over the sampled candidate set.

    This is the declared base painting-policy posterior of spec §10.5, computed
    with the same shift-by-max stabilization `policy_posterior_from_efe` uses, so
    the normalized result is identical to the un-shifted expression.
    """

    g = np.asarray(totals, dtype=np.float64).reshape(-1)
    prior = np.asarray(stop_log_priors, dtype=np.float64).reshape(-1)
    if g.shape != prior.shape:
        raise ValueError("EFE totals and stop log priors must align.")
    if g.size == 0:
        return np.zeros(0, dtype=np.float64)
    logits = prior - float(policy_precision) * g
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(g.shape, 1.0 / g.size, dtype=np.float64)
    return weights / total


# --------------------------------------------------------------------------
# The hand-written proposal's density. This is the density of the code that
# actually runs, not an idealization of it.
# --------------------------------------------------------------------------


def _uniform_log_density(value: float, interval: Interval) -> float:
    if not interval.contains(value):
        return PROPOSAL_LOG_DENSITY_FLOOR
    return -math.log(interval.width)


def _log_uniform_log_density(value: float, interval: Interval) -> float:
    """Density of ``exp(U(log lo, log hi))`` with respect to Lebesgue in x."""

    if not interval.contains(value) or value <= 0.0:
        return PROPOSAL_LOG_DENSITY_FLOOR
    return -math.log(value) - math.log(interval.log_width)


def _start_point_log_density(
    x: float,
    y: float,
    coverage_field: np.ndarray | None,
    config: PainterConfig,
) -> float:
    """Density of `PolicySampler._start_point`, exactly as that method samples.

    Two declared components, matching the code branch for branch: with
    probability `proposal_low_coverage_mix` (and only when a usable coverage
    field exists) a grid cell is drawn with probability proportional to
    ``clip(1 - coverage, 0, 1) + 1e-3`` and the point is uniform inside that
    cell; otherwise the point is uniform on the declared start-point box.

    NAMED APPROXIMATION: the cell branch clips the sampled coordinate into the
    start-point box, which places point masses on the two edge values for edge
    cells. Those atoms are evaluated on the PRE-CLIP parameterization (uniform
    inside the cell) rather than as atoms, so the cell-branch density is exact
    for interior cells and approximate on the border ring.
    """

    x_support = PROPOSAL_SUPPORT["start_x"]
    y_support = PROPOSAL_SUPPORT["start_y"]
    usable = (
        coverage_field is not None
        and getattr(coverage_field, "ndim", 0) == 2
        and getattr(coverage_field, "size", 0) > 0
    )
    mix = float(np.clip(config.proposal_low_coverage_mix, 0.0, 1.0)) if usable else 0.0
    inside = x_support.contains(x) and y_support.contains(y)
    uniform_density = (1.0 / (x_support.width * y_support.width)) if inside else 0.0
    cell_density = 0.0
    if usable and inside:
        field = np.asarray(coverage_field, dtype=np.float64)
        weights = np.clip(1.0 - field, 0.0, 1.0) + 1e-3
        probabilities = weights / weights.sum()
        rows, cols = field.shape
        col = int(min(cols - 1, max(0, math.floor(float(x) * cols))))
        row = int(min(rows - 1, max(0, math.floor(float(y) * rows))))
        cell_density = float(probabilities[row, col]) * float(cols) * float(rows)
    density = mix * cell_density + (1.0 - mix) * uniform_density
    if density <= 0.0:
        return PROPOSAL_LOG_DENSITY_FLOOR
    return float(math.log(density))


def hand_written_kind_log_probabilities(config: PainterConfig) -> dict[str, float]:
    """Declared kind categorical of `PolicySampler._passage_kind`."""

    from .policies import PASSAGE_BAND_PROBABILITY

    polyline = float(np.clip(config.passage_polyline_mix, 0.0, 1.0))
    band = (1.0 - polyline) * PASSAGE_BAND_PROBABILITY
    chain = (1.0 - polyline) * (1.0 - PASSAGE_BAND_PROBABILITY)
    probabilities = {"band": band, "chain": chain, "polyline": polyline}
    return {
        kind: (math.log(value) if value > 0.0 else PROPOSAL_LOG_DENSITY_FLOOR)
        for kind, value in probabilities.items()
    }


def tone_support(config: PainterConfig) -> tuple[float, ...]:
    """Mirror of `PolicySampler._tone_support`, without importing the sampler."""

    if config.stroke_tone_prior is None:
        return (0.0, 1.0)
    return (float(config.stroke_tone_prior),)


def hand_written_log_density(
    record: ProposalRecord,
    coverage_field: np.ndarray | None,
    config: PainterConfig,
) -> ProposalFactorLogDensity:
    """Log density the HAND-WRITTEN proposal assigns to one record's latents.

    The repo has never had this function: `PolicySampler` was written as a
    sampler only, and spec §10.2 conceded its mixtures were "not mathematically
    realized as normalized p(pi) terms". Writing it down is what makes the
    declared divergence a falsifiable number instead of a plausible-looking one,
    which is why `tests/test_proposal.py` checks it against a large draw from the
    real sampler rather than only against its own quadrature.
    """

    terms = {
        "center": 0.0,
        "direction": 0.0,
        "length": 0.0,
        "spacing": 0.0,
        "width": 0.0,
        "amount": 0.0,
        "stroke_count": 0.0,
        "kind": 0.0,
        "tone": 0.0,
    }
    in_support = True
    kind_log_probabilities = hand_written_kind_log_probabilities(config)
    stroke_range = passage_stroke_count_range(config)
    tones = tone_support(config)
    direction_log_density = -math.log(PROPOSAL_DIRECTION.width)

    for latent in record.latents:
        support = proposal_support_for(latent.family, latent.kind)
        terms["center"] += _start_point_log_density(
            latent.center_x, latent.center_y, coverage_field, config
        )
        terms["direction"] += direction_log_density
        terms["length"] += _uniform_log_density(latent.length, support["length"])
        terms["width"] += _log_uniform_log_density(latent.width, support["width"])
        terms["amount"] += _uniform_log_density(latent.amount, support["amount"])
        if latent.tone in tones:
            terms["tone"] += -math.log(float(len(tones)))
        else:
            terms["tone"] += PROPOSAL_LOG_DENSITY_FLOOR
        if latent.family == "passage":
            terms["kind"] += kind_log_probabilities[latent.kind]
            if latent.kind == "polyline":
                # The random sign at `_passage_geometry` is a fair Bernoulli, and
                # the magnitude is uniform on the declared support, so the signed
                # turn's density is `0.5 * Uniform(|turn|)` on each sign branch.
                terms["spacing"] += math.log(0.5) + _uniform_log_density(
                    abs(latent.spacing), support["spacing"]
                )
            else:
                terms["spacing"] += _uniform_log_density(latent.spacing, support["spacing"])
            count = int(latent.stroke_count)
            if int(stroke_range.low) <= count <= int(stroke_range.high):
                terms["stroke_count"] += -math.log(
                    float(int(stroke_range.high) - int(stroke_range.low) + 1)
                )
            else:
                terms["stroke_count"] += PROPOSAL_LOG_DENSITY_FLOOR
        # The mark family's planning-DEPTH categorical stays hand-written on both
        # sides (`PolicySampler.sample` draws it for learned marks too), so it is
        # a common factor and contributes 0.0 to both densities.

    total = float(sum(terms.values()))
    if any(value <= PROPOSAL_LOG_DENSITY_FLOOR for value in terms.values()):
        in_support = False
    return ProposalFactorLogDensity(
        total=total,
        latent_count=len(record.latents),
        in_support=in_support,
        approximation=(
            "exact density of PolicySampler's declared draws, except that the "
            "low-coverage cell branch's boundary clip is evaluated on the pre-clip "
            "parameterization; the shared decoder and the mark-depth categorical "
            "cancel and are omitted"
        ),
        **terms,
    )


# --------------------------------------------------------------------------
# Generic divergence estimator.
# --------------------------------------------------------------------------


def proposal_divergence(
    sample_fn,
    log_p,
    log_q,
    *,
    samples: int,
    family: str = "passage",
) -> ProposalDivergence:
    """Monte-Carlo ``E_p[log p(z) - log q(z)]``.

    Deliberately generic in both densities so a test can pass the SAME callable
    twice and obtain exactly ``0.0`` -- the "zero at initialization against
    itself" property is then satisfied by construction rather than by two
    independently-implemented formulas happening to agree.
    """

    count = max(0, int(samples))
    if count == 0:
        return ProposalDivergence(family=family, divergence_nats=0.0, sample_count=0, out_of_support_fraction=0.0)
    total = 0.0
    out_of_support = 0
    for _ in range(count):
        latent = sample_fn()
        p_value = log_p(latent)
        q_value = log_q(latent)
        p_total = p_value.total if isinstance(p_value, ProposalFactorLogDensity) else float(p_value)
        q_total = q_value.total if isinstance(q_value, ProposalFactorLogDensity) else float(q_value)
        if isinstance(q_value, ProposalFactorLogDensity) and not q_value.in_support:
            out_of_support += 1
        total += float(p_total) - float(q_total)
    return ProposalDivergence(
        family=family,
        divergence_nats=float(total / count),
        sample_count=count,
        out_of_support_fraction=float(out_of_support / count),
    )


# --------------------------------------------------------------------------
# The learned proposal.
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ProposalHeads:
    """One forward pass of the proposal network, before masking is applied."""

    continuous: dict[str, tuple[torch.Tensor, torch.Tensor]]
    kind_logits: torch.Tensor
    stroke_count_logits: torch.Tensor
    tone_logit: torch.Tensor
    turn_sign_logit: torch.Tensor
    approximation: str = (
        "factored logit-normal / wrapped-Gaussian / masked-categorical proposal; "
        "no dependence between parameters beyond the shared belief features"
    )


def _masked_log_softmax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of a categorical restricted to the allowed support.

    The declared config gates (`passage_polyline_mix`, `passage_min_strokes`,
    `passage_max_strokes`, `planning_horizon`, `stroke_tone_prior`) DOMINATE the
    learned categoricals here. A learned proposal must not be able to re-enable a
    family the operator disabled: the hierarchy of authority runs declared config
    over learned proposal.
    """

    masked = logits.masked_fill(~allowed, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def _kind_mask(config: PainterConfig, device: torch.device) -> torch.Tensor:
    polyline = float(np.clip(config.passage_polyline_mix, 0.0, 1.0))
    allowed = [polyline < 1.0, polyline < 1.0, polyline > 0.0]
    return torch.tensor(allowed, dtype=torch.bool, device=device)


def _stroke_count_mask(config: PainterConfig, device: torch.device) -> torch.Tensor:
    stroke_range = passage_stroke_count_range(config)
    low = max(1, int(stroke_range.low))
    high = min(PROPOSAL_STROKE_COUNT_SUPPORT, max(low, int(stroke_range.high)))
    allowed = [low <= (index + 1) <= high for index in range(PROPOSAL_STROKE_COUNT_SUPPORT)]
    if not any(allowed):
        # The declared range lies entirely above the fixed head width. Declared
        # approximation: the learned proposal then supports only the largest
        # representable count, and the hand-written density scores it as
        # out-of-support so the divergence reports the disagreement instead of
        # hiding it.
        allowed[-1] = True
    return torch.tensor(allowed, dtype=torch.bool, device=device)


def _tone_mask(config: PainterConfig, device: torch.device) -> torch.Tensor:
    tones = tone_support(config)
    allowed = [0.0 in tones, 1.0 in tones]
    return torch.tensor(allowed, dtype=torch.bool, device=device)


class PolicyProposalNetwork(nn.Module):
    """``q_proposal(z_pi | q(z_canvas), q(z_relational))``, a learned proposal.

    Conditioning: the coarse canvas latent posterior mean and the relational
    latent posterior mean maintained by `canvas_hierarchy.HierarchicalCanvasModel`
    (`Q-CANVAS` / `Q-REL`). Those are the agent's own slow beliefs about canvas
    arrangement, which is exactly what a compositional mark vocabulary should be
    conditioned on.

    All parameters live in one small MLP trunk with one linear head per factor.
    Scales are clamped to a declared bounded range so ``log q`` is bounded above
    and the divergence estimator cannot diverge; that bound is a NUMERICAL
    SUPPORT BOUND on a proposal density, explicitly NOT a precision belief over
    any outcome.
    """

    def __init__(self, config: PainterConfig) -> None:
        super().__init__()
        self.cfg = config
        latent_grid = max(1, config.spatial_grid_size // 4)
        self._canvas_dim = int(config.canvas_latent_channels * latent_grid * latent_grid)
        self._relational_dim = int(config.relational_latent_dim)
        self._input_dim = self._canvas_dim + self._relational_dim
        hidden = int(config.learned_proposal_hidden_dim)
        # Parameter-free normalization: the canvas latent scale drifts as the
        # hierarchy trains, and an affine LayerNorm would give the proposal a
        # second, unattributable place to absorb that drift.
        self.normalizer = nn.LayerNorm(self._input_dim, elementwise_affine=False)
        self.trunk = nn.Sequential(
            nn.Linear(self._input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        head_keys: list[str] = []
        for parameter in _MARK_PARAMETERS:
            head_keys.append(_head_key("mark", "mark", parameter))
        for parameter in _PASSAGE_SHARED_PARAMETERS:
            head_keys.append(_head_key("passage", "band", parameter))
        for kind in PROPOSAL_KINDS:
            for parameter in _PASSAGE_KIND_PARAMETERS:
                head_keys.append(_head_key("passage", kind, parameter))
        self._head_keys = tuple(dict.fromkeys(head_keys))
        self.continuous_heads = nn.ModuleDict(
            {key.replace(".", "_"): nn.Linear(hidden, 2) for key in self._head_keys}
        )
        self.kind_head = nn.Linear(hidden, len(PROPOSAL_KINDS))
        self.stroke_count_head = nn.Linear(hidden, PROPOSAL_STROKE_COUNT_SUPPORT)
        self.tone_head = nn.Linear(hidden, 2)
        self.turn_sign_head = nn.Linear(hidden, 2)
        self.register_buffer("proposal_update_count", torch.zeros((), dtype=torch.int64))
        # A PRIVATE generator, never `PolicySampler.rng` and never the global
        # torch stream. Sampling from the learned proposal must not perturb the
        # hand-written candidate stream, or `learned_proposal_mix = 0.0` would
        # stop reproducing today's behaviour.
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    # -- shape metadata, for the checkpoint architecture dict ---------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return (
            2 * len(self._head_keys)
            + len(PROPOSAL_KINDS)
            + PROPOSAL_STROKE_COUNT_SUPPORT
            + 2
            + 2
        )

    def seed_generator(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    def mark_update(self) -> None:
        """Mirror of `HierarchicalCanvasModel.mark_transition_update`."""

        self.proposal_update_count += 1

    @property
    def update_count(self) -> int:
        return int(self.proposal_update_count.item())

    # -- belief features ----------------------------------------------------

    @staticmethod
    def features_from_beliefs(
        canvas_belief,
        relational_belief,
        config: PainterConfig,
        device: torch.device,
        *,
        has_hierarchy: bool = True,
    ) -> tuple[torch.Tensor, str]:
        """Concatenated canvas and relational posterior MEANS, plus their source.

        The source label is load-bearing, not decoration:
        `canvas_belief`/`relational_belief` are plain attributes initialized to
        `None`, absent in summary mode, absent when the composition hierarchy is
        disabled, and still `None` after a checkpoint load until
        `reset_hierarchy_beliefs` runs -- which `reset()` skips when the
        observation boundary is blocked, the live default. Sampling MAY fall back
        to zeros; training may NOT, or the proposal would quietly amortize a
        constant.
        """

        latent_grid = max(1, config.spatial_grid_size // 4)
        canvas_dim = int(config.canvas_latent_channels * latent_grid * latent_grid)
        input_dim = canvas_dim + int(config.relational_latent_dim)
        if not has_hierarchy:
            return torch.zeros(1, input_dim, device=device), BELIEF_SOURCE_NO_HIERARCHY
        if canvas_belief is None or relational_belief is None:
            return torch.zeros(1, input_dim, device=device), BELIEF_SOURCE_UNINITIALIZED
        canvas = canvas_belief.mean.detach().reshape(1, -1).to(device=device, dtype=torch.float32)
        relational = (
            relational_belief.mean.detach().reshape(1, -1).to(device=device, dtype=torch.float32)
        )
        features = torch.cat([canvas, relational], dim=1)
        if features.shape[1] != input_dim:
            # A config/checkpoint mismatch. Report it as a fallback rather than
            # crashing the planner thread, where the exception would surface only
            # as a diagnostics string.
            return torch.zeros(1, input_dim, device=device), BELIEF_SOURCE_UNINITIALIZED
        return features, BELIEF_SOURCE_POSTERIOR

    # -- distribution -------------------------------------------------------

    def distribution(self, features: torch.Tensor) -> ProposalHeads:
        hidden = self.trunk(self.normalizer(features.reshape(1, -1)))
        minimum = float(self.cfg.learned_proposal_min_log_scale)
        continuous: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for key in self._head_keys:
            raw = self.continuous_heads[key.replace(".", "_")](hidden).reshape(-1)
            loc = raw[0]
            log_scale = torch.clamp(raw[1], min=minimum, max=2.0)
            continuous[key] = (loc, torch.exp(log_scale))
        return ProposalHeads(
            continuous=continuous,
            kind_logits=self.kind_head(hidden).reshape(-1),
            stroke_count_logits=self.stroke_count_head(hidden).reshape(-1),
            tone_logit=self.tone_head(hidden).reshape(-1),
            turn_sign_logit=self.turn_sign_head(hidden).reshape(-1),
        )

    # -- sampling -----------------------------------------------------------

    def _normal_noise(self, device: torch.device) -> torch.Tensor:
        return torch.randn(1, generator=self.generator).to(device).reshape(())

    def _categorical_draw(self, log_probabilities: torch.Tensor) -> int:
        probabilities = torch.exp(log_probabilities.detach()).cpu()
        draw = torch.multinomial(probabilities, 1, generator=self.generator)
        return int(draw.item())

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        *,
        family: str,
        config: PainterConfig,
        count: int = 1,
    ) -> ProposalRecord:
        """Draw `count` latents of one family from the learned proposal."""

        if family not in PROPOSAL_FAMILIES:
            raise ValueError(f"Unknown proposal family {family!r}; expected one of {PROPOSAL_FAMILIES}.")
        heads = self.distribution(features)
        device = features.device
        latents: list[ProposalLatent] = []
        for _ in range(max(1, int(count))):
            latents.append(self._sample_one(heads, family=family, config=config, device=device))
        return ProposalRecord(family=family, source="learned", latents=tuple(latents))

    def _sample_one(
        self,
        heads: ProposalHeads,
        *,
        family: str,
        config: PainterConfig,
        device: torch.device,
    ) -> ProposalLatent:
        if family == "passage":
            kind_log_probabilities = _masked_log_softmax(
                heads.kind_logits, _kind_mask(config, heads.kind_logits.device)
            )
            kind = PROPOSAL_KINDS[self._categorical_draw(kind_log_probabilities)]
            count_log_probabilities = _masked_log_softmax(
                heads.stroke_count_logits, _stroke_count_mask(config, heads.stroke_count_logits.device)
            )
            stroke_count = self._categorical_draw(count_log_probabilities) + 1
        else:
            kind = "mark"
            stroke_count = 1
        if config.stroke_tone_prior is None:
            tone_log_probabilities = _masked_log_softmax(
                heads.tone_logit, _tone_mask(config, heads.tone_logit.device)
            )
            tone = float(self._categorical_draw(tone_log_probabilities))
        else:
            # A fixed configured tone is a one-point categorical, including for
            # legal intermediate tones. It consumes no proposal RNG and the
            # learned binary head has no authority to round it to black/white.
            tone = float(config.stroke_tone_prior)

        support = proposal_support_for(family, kind)
        values: dict[str, float] = {}
        for parameter in _parameters_for(family):
            if parameter == "direction":
                loc, scale = heads.continuous[_head_key(family, kind, parameter)]
                mean_angle = _direction_mean(loc)
                offset = scale * self._normal_noise(device)
                values["direction"] = float(
                    torch.remainder(mean_angle + offset, PROPOSAL_DIRECTION.high).item()
                )
                continue
            loc, scale = heads.continuous[_head_key(family, kind, parameter)]
            interval = support[parameter]
            unit = torch.sigmoid(loc + scale * self._normal_noise(device))
            if parameter in _LOG_SPACE_PARAMETERS:
                log_value = math.log(interval.low) + interval.log_width * float(unit.item())
                values[parameter] = float(math.exp(log_value))
            else:
                values[parameter] = float(interval.low + interval.width * float(unit.item()))
        spacing = values.get("spacing", 0.0)
        if family == "passage" and kind == "polyline":
            sign_log_probabilities = _masked_log_softmax(
                heads.turn_sign_logit,
                torch.ones(2, dtype=torch.bool, device=heads.turn_sign_logit.device),
            )
            sign = 1.0 if self._categorical_draw(sign_log_probabilities) == 1 else -1.0
            spacing = sign * spacing
        return ProposalLatent(
            family=family,
            kind=kind,
            center_x=values["center_x"],
            center_y=values["center_y"],
            direction=values["direction"],
            length=values["length"],
            spacing=float(spacing),
            width=values["width"],
            amount=values["amount"],
            stroke_count=int(stroke_count),
            tone=tone,
        )

    # -- scoring ------------------------------------------------------------

    def log_density_terms(
        self,
        record: ProposalRecord,
        heads: ProposalHeads,
        config: PainterConfig,
    ) -> dict[str, torch.Tensor]:
        """Per-factor log density of one record, differentiable in the heads."""

        device = heads.kind_logits.device
        zero = torch.zeros((), device=device)
        terms: dict[str, torch.Tensor] = {
            "center": zero.clone(),
            "direction": zero.clone(),
            "length": zero.clone(),
            "spacing": zero.clone(),
            "width": zero.clone(),
            "amount": zero.clone(),
            "stroke_count": zero.clone(),
            "kind": zero.clone(),
            "tone": zero.clone(),
        }
        kind_log_probabilities = _masked_log_softmax(heads.kind_logits, _kind_mask(config, device))
        count_log_probabilities = _masked_log_softmax(
            heads.stroke_count_logits, _stroke_count_mask(config, device)
        )
        tone_log_probabilities = (
            _masked_log_softmax(heads.tone_logit, _tone_mask(config, device))
            if config.stroke_tone_prior is None
            else None
        )
        sign_log_probabilities = _masked_log_softmax(
            heads.turn_sign_logit, torch.ones(2, dtype=torch.bool, device=device)
        )
        for latent in record.latents:
            family = latent.family
            kind = latent.kind
            support = proposal_support_for(family, kind)
            for parameter in _parameters_for(family):
                loc, scale = heads.continuous[_head_key(family, kind, parameter)]
                if parameter == "direction":
                    terms["direction"] = terms["direction"] + _wrapped_normal_log_density(
                        latent.direction, _direction_mean(loc), scale
                    )
                    continue
                value = latent.value(parameter)
                key = _factor_key(parameter)
                if parameter == "spacing" and kind == "polyline":
                    index = 1 if value >= 0.0 else 0
                    terms["spacing"] = terms["spacing"] + sign_log_probabilities[index]
                    value = abs(value)
                if parameter in _LOG_SPACE_PARAMETERS:
                    terms[key] = terms[key] + _log_logit_normal_log_density(
                        value, loc, scale, support[parameter]
                    )
                else:
                    terms[key] = terms[key] + _logit_normal_log_density(
                        value, loc, scale, support[parameter]
                    )
            if config.stroke_tone_prior is None:
                if latent.tone in (0.0, 1.0):
                    assert tone_log_probabilities is not None
                    terms["tone"] = terms["tone"] + tone_log_probabilities[int(latent.tone)]
                else:
                    terms["tone"] = terms["tone"] + torch.full_like(zero, float("-inf"))
            elif latent.tone != float(config.stroke_tone_prior):
                # A configured tone is a one-point categorical controlled by
                # the declared policy setting, not something the learned head
                # may round to a nearby binary value.
                terms["tone"] = terms["tone"] + torch.full_like(zero, float("-inf"))
            if family == "passage":
                terms["kind"] = terms["kind"] + kind_log_probabilities[PROPOSAL_KINDS.index(kind)]
                count = float(latent.stroke_count)
                if count.is_integer() and 1 <= int(count) <= PROPOSAL_STROKE_COUNT_SUPPORT:
                    terms["stroke_count"] = (
                        terms["stroke_count"] + count_log_probabilities[int(count) - 1]
                    )
                else:
                    terms["stroke_count"] = (
                        terms["stroke_count"] + torch.full_like(zero, float("-inf"))
                    )
        return terms

    def log_density(
        self,
        record: ProposalRecord,
        features: torch.Tensor,
        config: PainterConfig,
        heads: ProposalHeads | None = None,
    ) -> ProposalFactorLogDensity:
        heads = self.distribution(features) if heads is None else heads
        terms = self.log_density_terms(record, heads, config)
        values = {name: float(value.item()) for name, value in terms.items()}
        total = float(sum(values.values()))
        return ProposalFactorLogDensity(
            total=total,
            latent_count=len(record.latents),
            in_support=bool(math.isfinite(total)),
            approximation=(
                "factored logit-normal (log-space for width) on the declared proposal "
                "support, wrapped Gaussian on direction truncated to three wraps and "
                "exactly renormalized, masked categoricals for kind/stroke_count/tone/turn sign"
            ),
            **values,
        )

    # -- training -----------------------------------------------------------

    def training_loss(
        self,
        batch: ProposalTrainingBatch,
        config: PainterConfig,
    ) -> torch.Tensor | None:
        """``- sum_i w_i log q(z_i | belief) / sum_i w_i``.

        This is ``KL(q_base(pi) || q_proposal(pi | belief))`` up to the target's
        own entropy, which is constant in the parameters -- i.e. AMORTIZED
        VARIATIONAL INFERENCE, not a reward, a return, or a value function. See
        the module docstring for the full charter argument.
        """

        if batch.features is None:
            return None
        indices = batch.modelled_indices()
        if not indices:
            return None
        heads = self.distribution(batch.features)
        total_weight = float(sum(batch.weights[index] for index in indices))
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            return None
        loss = torch.zeros((), device=batch.features.device)
        for index in indices:
            record = batch.records[index]
            assert record is not None
            terms = self.log_density_terms(record, heads, config)
            log_density = torch.stack(list(terms.values())).sum()
            loss = loss - (float(batch.weights[index]) / total_weight) * log_density
        return loss

    # -- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def divergence_against_hand_written(
        self,
        features: torch.Tensor,
        coverage_field: np.ndarray | None,
        config: PainterConfig,
        *,
        family: str,
        samples: int | None = None,
    ) -> ProposalDivergence:
        heads = self.distribution(features)
        count = int(config.learned_proposal_divergence_samples if samples is None else samples)

        def draw() -> ProposalRecord:
            latent = self._sample_one(
                heads, family=family, config=config, device=features.device
            )
            return ProposalRecord(family=family, source="learned", latents=(latent,))

        return proposal_divergence(
            draw,
            lambda record: self.log_density(record, features, config, heads=heads),
            lambda record: hand_written_log_density(record, coverage_field, config),
            samples=count,
            family=family,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "version": POLICY_PROPOSAL_VERSION,
            "inputDimensions": self.input_dim,
            "outputDimensions": self.output_dim,
            "hiddenDimensions": int(self.cfg.learned_proposal_hidden_dim),
            "updateCount": self.update_count,
            "minLogScale": float(self.cfg.learned_proposal_min_log_scale),
            "approximation": (
                "factored proposal density; no dependence between parameters beyond "
                "the shared belief features"
            ),
        }


# --------------------------------------------------------------------------
# Elementary densities. Written out rather than taken from torch.distributions
# so the Jacobian of each declared transform is visible at the point of use.
# --------------------------------------------------------------------------


def _direction_mean(loc: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained head output onto the circle ``(0, 2*pi)``."""

    half = 0.5 * PROPOSAL_DIRECTION.high
    return half * torch.tanh(loc) + half


def _standard_normal_log_density(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return -0.5 * z * z - torch.log(scale) - _LOG_SQRT_TWO_PI


def _logit_normal_log_density(
    value: float,
    loc: torch.Tensor,
    scale: torch.Tensor,
    interval: Interval,
) -> torch.Tensor:
    """``x = lo + (hi - lo) * sigmoid(u)``, ``u ~ N(loc, scale)``.

    Exactly normalized on the OPEN interval ``(lo, hi)`` with the Jacobian
    ``-log((hi - lo) * s * (1 - s))`` scored, which is the whole reason this
    replaces the brief's "log-normal, clipped": a clipped density would put atoms
    on the boundary and leave the divergence undefined.
    """

    numeric = float(value)
    if not math.isfinite(numeric) or not interval.contains(numeric):
        return torch.full((), float("-inf"), device=loc.device, dtype=loc.dtype)
    span = interval.width
    unit = (numeric - interval.low) / span
    unit = min(max(unit, 1e-9), 1.0 - 1e-9)
    u = torch.as_tensor(math.log(unit / (1.0 - unit)), device=loc.device, dtype=loc.dtype)
    z = (u - loc) / scale
    jacobian = math.log(span) + math.log(unit) + math.log(1.0 - unit)
    return _standard_normal_log_density(z, scale) - jacobian


def _log_logit_normal_log_density(
    value: float,
    loc: torch.Tensor,
    scale: torch.Tensor,
    interval: Interval,
) -> torch.Tensor:
    """Logit-normal in ``log x`` over ``[log lo, log hi]``.

    The absolutely-continuous analogue of the hand-written log-uniform width
    draw: scale-free in the same sense, but a normalized density with a Jacobian
    rather than a clipped one.
    """

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0 or not interval.contains(numeric):
        return torch.full((), float("-inf"), device=loc.device, dtype=loc.dtype)
    positive = numeric
    span = interval.log_width
    unit = (math.log(positive) - math.log(interval.low)) / span
    unit = min(max(unit, 1e-9), 1.0 - 1e-9)
    u = torch.as_tensor(math.log(unit / (1.0 - unit)), device=loc.device, dtype=loc.dtype)
    z = (u - loc) / scale
    jacobian = math.log(positive) + math.log(span) + math.log(unit) + math.log(1.0 - unit)
    return _standard_normal_log_density(z, scale) - jacobian


def _wrapped_angle(value: float) -> float:
    """Same convention as `passage_inference._wrapped_angle`.

    Re-implemented rather than imported: `passage_inference` imports `policies`,
    which imports this module, so importing it back would close a cycle.
    """

    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _wrapped_normal_log_density(
    value: float,
    mean_angle: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Wrapped Gaussian on the circle, truncated to three wraps and renormalized.

    The truncated sum over ``k in {-1, 0, +1}`` integrates over ``[-3pi, 3pi]``,
    so dividing by ``erf(3 pi / (scale sqrt 2))`` makes this an EXACTLY normalized
    density on the circle for the truncated family. The declared approximation is
    the family itself (three wraps), not the normalization.
    """

    delta = torch.as_tensor(
        _wrapped_angle(float(value)), device=mean_angle.device, dtype=mean_angle.dtype
    ) - mean_angle
    delta = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
    offsets = torch.arange(
        -PROPOSAL_DIRECTION_WRAPS,
        PROPOSAL_DIRECTION_WRAPS + 1,
        device=mean_angle.device,
        dtype=mean_angle.dtype,
    ) * (2.0 * math.pi)
    z = (delta + offsets) / scale
    unnormalized = torch.logsumexp(_standard_normal_log_density(z, scale), dim=0)
    half_width = (2.0 * PROPOSAL_DIRECTION_WRAPS + 1.0) * math.pi
    mass = torch.erf(half_width / (scale * math.sqrt(2.0)))
    return unnormalized - torch.log(torch.clamp(mass, min=1e-12))
