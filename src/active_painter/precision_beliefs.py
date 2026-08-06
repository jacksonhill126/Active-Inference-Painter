"""Gamma precision beliefs over the expected-free-energy modalities.

Every quantity in this module is a PRECISION BELIEF or a TRANSITION PRIOR in
the sense of the project charter, never a reward:

* :class:`GammaPrecisionBelief` is an explicit Gamma(alpha, beta) density over
  an inverse temperature. Its posterior mean multiplies an already-identified
  EFE term; it never appears additively in any objective. It is updated by the
  reference implementation's Chapter 10 rule

      dF/dgamma = (alpha/gamma - beta0) + (pi - pi0) . (-G)
      beta <- max(beta - kappa * dF/dgamma, eps)

  which is driven purely by the mismatch between the EFE ordering ``G`` and a
  policy-dependent variational free energy ``F``. There is no step counter, no
  annealing schedule, and no outcome-quality signal anywhere in its input.
  :func:`precision_gradient` and :func:`learn_precision` mirror
  ``active_inference.core.pomdp.precision_gradient`` / ``learn_precision``
  arithmetic-for-arithmetic at ``alpha = 1`` (pinned in
  ``tests/test_precision_beliefs.py``).

* :class:`GapIncrementBelief` is a Gaussian random-walk belief over the
  per-mark compression-gap increment. It is a belief over a rate of change and
  is structurally barred from expected free energy: its only consumer is the
  STOP POLICY PRIOR in :func:`active_painter.policies.policy_stop_log_prior`.

Two properties are load-bearing for the charter and are pinned by tests:

1. Each belief's prior mean is seeded at exactly the existing declared config
   constant (``beta = alpha0 / constant``), and an unobserved belief returns
   that constant bit-identically. Turning the beliefs off therefore reproduces
   the hand-written arithmetic exactly, which makes attribution exact.
2. The posterior mean is clamped to a DECLARED BOUNDED SUPPORT. Measured
   unbounded, disagreeing evidence drives the terminal-coverage precision to
   ~1e-3, which would effectively delete the declared C matrix from G. A
   precision belief must not become a backdoor around "preferences are never
   learned from outcomes".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .config import PainterConfig


# Mirrors ``active_inference.core.pomdp._EPS`` so the gradient is the
# reference's arithmetic, not merely its algebra.
_EPS = 1e-16

POLICY_PRECISION_KEY = "policy"

MODALITY_NAMES: tuple[str, ...] = (
    "terminal_coverage",
    "observation_ambiguity",
    "transition",
    "composition_gap",
    "canvas_latent_transition",
    "relational_transition",
    "motor_proprioceptive",
)

LEDGER_KEYS: tuple[str, ...] = MODALITY_NAMES + (POLICY_PRECISION_KEY,)

# The declared config constant that seeds each belief's prior mean. Every one of
# these stays in the config untouched: the belief re-expresses the constant as a
# prior, it does not replace it.
_DECLARED_CONSTANT_FIELDS: dict[str, str] = {
    "terminal_coverage": "terminal_risk_precision",
    "observation_ambiguity": "ambiguity_precision",
    "transition": "transition_precision",
    "composition_gap": "composition_gap_precision",
    "canvas_latent_transition": "canvas_latent_transition_precision",
    "relational_transition": "relational_transition_precision",
    # The motor modality already carries three declared precisions INSIDE
    # motor_efe_terms (risk, ambiguity, reliability novelty). This gamma is a
    # modality-level multiplier on top of them, so its declared constant is the
    # identity and today's arithmetic is the unobserved-belief case.
    "motor_proprioceptive": "motor_modality_precision",
    POLICY_PRECISION_KEY: "policy_precision",
}

# Declared unit of each modality after normalization. These are DECLARATIONS
# about an existing quantity's denominator, not new decision content. Note that
# the denominators genuinely differ in kind between modality groups; a single
# "per observation channel" convention is defensible and declared, but it is
# not neutral (see the approximation register).
NORMALIZER_NAMES: dict[str, str] = {
    # A scalar Beta on one aggregate coverage channel. What bounds it is the
    # declared interior-unimodal forecast family (the concentration floor), not
    # a divisor.
    "terminal_coverage": "nats_per_aggregate_coverage_channel",
    # Already divided by independent_material_channel_count * full_area inside
    # _observation_ambiguity_scaled.
    "observation_ambiguity": "nats_per_independent_cell_channel",
    "transition": "nats_per_independent_cell_channel",
    # canvas_hierarchy averages over channels * grid * grid across ALL SIX
    # material channels, including the two deterministic ones. Declared here so
    # the pre-existing deviation is visible rather than hidden.
    "composition_gap": "nats_per_cell_channel_all_material_channels",
    "canvas_latent_transition": "nats_per_canvas_latent_dim",
    "relational_transition": "nats_per_relational_latent_dim",
    # The only genuinely new divisor in the feature.
    "motor_proprioceptive": "nats_per_proprioceptive_channel",
}

# Nominal proprioceptive channel count, used only for the recorded normalizer in
# the EFE component dataclasses. motor_efe_terms always divides by the ACTUAL
# label count it was handed.
DEFAULT_PROPRIOCEPTIVE_CHANNELS = 27

_UPDATE_STATUSES = frozenset(
    {
        "updated",
        "clamped",
        "structurally_off",
        "disabled",
        "no_free_energy",
        "degenerate_flat_F",
        "too_few_candidates",
    }
)


def _log_sum_exp(values: np.ndarray) -> float:
    """Reference-identical stable log-sum-exp for a 1-D vector."""

    maximum = float(np.max(values))
    safe = maximum if math.isfinite(maximum) else 0.0
    return float(np.log(np.sum(np.exp(values - safe))) + safe)


def _softmax(values: np.ndarray) -> np.ndarray:
    """Reference-identical stable softmax: ``exp(x - logsumexp(x))``."""

    values = np.asarray(values, dtype=float)
    return np.exp(values - _log_sum_exp(values))


def precision_gradient(
    G: np.ndarray,
    F: np.ndarray,
    gamma: float,
    *,
    log_prior: np.ndarray | None = None,
    alpha: float = 1.0,
    beta0: float = 1.0,
) -> float:
    r"""Free-energy gradient with respect to a precision (reference Eq. 23).

    .. math::
        \frac{\partial \mathcal F}{\partial \gamma}
          = \left(\frac{\alpha}{\gamma} - \beta_0\right) + (\pi - \pi_0)\cdot(-G)

    with :math:`\pi_0 = \sigma(\log p(\pi) - \gamma G)` the policy prior and
    :math:`\pi = \sigma(\log p(\pi) - F - \gamma G)` the policy posterior. The
    gradient measures whether the realized free energy ``F`` agrees with the
    EFE ordering ``G``: agreement raises the precision, disagreement lowers it.

    ``log_prior`` takes the reference's ``E`` role but is ALREADY a log, so it
    is never re-logged the way ``policy_posterior_full`` logs ``E``. At
    ``alpha = 1.0`` and ``log_prior=None`` this is the reference's arithmetic.
    """

    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if beta0 <= 0.0:
        raise ValueError(f"beta0 must be positive, got {beta0}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    G = np.asarray(G, dtype=float)
    F = np.asarray(F, dtype=float)
    if G.ndim != 1 or G.shape != F.shape:
        raise ValueError("G and F must be 1-D vectors with matching shape")
    beta = alpha / (gamma + _EPS)
    log_e = np.zeros(G.shape[0]) if log_prior is None else np.asarray(log_prior, dtype=float)
    if log_e.shape != G.shape:
        raise ValueError("log_prior must match the shape of G")
    pi0 = _softmax(log_e - gamma * G)
    pi = _softmax(log_e - F - gamma * G)
    return float((beta - beta0) + (pi - pi0) @ (-G))


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    """Outcome of a precision descent, mirroring the reference container."""

    gamma: float
    beta: float
    gamma_trace: tuple[float, ...]
    converged: bool
    grad_final: float


def learn_precision(
    G: np.ndarray,
    F: np.ndarray,
    *,
    log_prior: np.ndarray | None = None,
    alpha: float = 1.0,
    beta0: float = 1.0,
    beta_init: float | None = None,
    kappa: float = 0.25,
    n_iter: int = 64,
    tol: float = 1e-10,
) -> PrecisionResult:
    r"""Descend the Gamma rate on variational free energy (reference Eq. 23-25).

    Iterates ``beta <- max(beta - kappa * dF/dgamma, eps)`` and reads the
    precision off as the Gamma mean ``gamma = alpha / beta``. At ``alpha = 1``,
    ``beta_init=None`` and ``log_prior=None`` this reproduces
    ``active_inference.core.pomdp.learn_precision`` exactly.

    ``beta_init`` is the sequential-belief generalization: a persistent belief
    warm-starts at its own current rate rather than re-initializing at the
    prior rate every planning round, so evidence accumulates. ``beta0`` stays
    the PRIOR rate that anchors the gradient. Named approximation:
    "Gamma rate warm-started at the current posterior rate".
    """

    if beta0 <= 0.0:
        raise ValueError(f"beta0 must be positive, got {beta0}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    G = np.asarray(G, dtype=float)
    F = np.asarray(F, dtype=float)
    if G.shape != F.shape or G.ndim != 1:
        raise ValueError("G and F must be 1-D vectors with matching shape")
    beta = float(beta0 if beta_init is None else beta_init)
    if beta <= 0.0:
        raise ValueError(f"beta_init must be positive, got {beta_init}")
    trace = [alpha / beta]
    grad = float("inf")
    converged = False
    for _ in range(int(n_iter)):
        gamma = alpha / (beta + _EPS)
        grad = precision_gradient(G, F, gamma, log_prior=log_prior, alpha=alpha, beta0=beta0)
        beta = max(beta - kappa * grad, _EPS)
        trace.append(alpha / beta)
        if abs(grad) < tol:
            converged = True
            break
    return PrecisionResult(
        gamma=alpha / beta,
        beta=beta,
        gamma_trace=tuple(float(value) for value in trace),
        converged=converged,
        grad_final=abs(grad),
    )


@dataclass(frozen=True, slots=True)
class PrecisionUpdate:
    """Audit record of one attempted precision update.

    ``status`` is how this module refuses to report a degenerate no-op as
    convergence. Measured: a flat ``F`` makes the reference gradient exactly
    0.0 with ``converged=True``, which looks like a learned fixed point but is
    the trivial ``gamma = alpha / beta0``. Such rounds report
    ``degenerate_flat_F``, never ``updated``.
    """

    name: str
    gamma_before: float
    gamma_after: float
    gradient: float
    observations_used: int
    status: str
    approximation: str = ""

    def __post_init__(self) -> None:
        if self.status not in _UPDATE_STATUSES:
            raise ValueError(f"Unknown precision-update status {self.status!r}.")


@dataclass(slots=True)
class GammaPrecisionBelief:
    """Gamma(alpha, beta) belief over one inverse temperature.

    ``prior_mean`` is the declared config constant. ``mean()`` returns it
    bit-identically until the first accepted observation, which is what makes
    "beliefs enabled but unobserved" and "beliefs disabled" the same arithmetic.
    """

    alpha: float
    beta: float
    prior_beta: float
    prior_mean: float
    observations: int = 0

    @classmethod
    def from_prior(cls, prior_mean: float, alpha0: float = 1.0) -> "GammaPrecisionBelief":
        mean = float(prior_mean)
        alpha = max(1e-6, float(alpha0))
        if mean <= 0.0:
            raise ValueError("A Gamma precision belief needs a positive prior mean.")
        beta = alpha / mean
        return cls(alpha=alpha, beta=beta, prior_beta=beta, prior_mean=mean, observations=0)

    def mean(self) -> float:
        if self.observations <= 0:
            # Bit-identical to the declared constant: attribution must be exact,
            # not approximate.
            return float(self.prior_mean)
        return float(self.alpha / max(self.beta, _EPS))

    def resolvable_uncertainty(self) -> float:
        """Half-log of one plus the squared coefficient of variation.

        DIAGNOSTIC ONLY. The Gamma shape ``alpha`` is declared and never
        updated, so this is a constant; a constant inside expected free energy
        is either inert (shift-invariant) or an unowned offset. It is therefore
        forbidden by contract from entering any EFE term, and a test asserts
        that feeding it into nothing changes no component.
        """

        return float(0.5 * math.log1p(1.0 / max(self.alpha, 1e-12)))

    def update(
        self,
        G: np.ndarray,
        F: np.ndarray,
        *,
        log_prior: np.ndarray | None = None,
        kappa: float = 0.25,
        n_iter: int = 64,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
    ) -> tuple[float, float, bool]:
        """Descend the rate, then clamp the mean into the declared support.

        Returns ``(gamma_after, gradient, clamped)``.
        """

        result = learn_precision(
            G,
            F,
            log_prior=log_prior,
            alpha=self.alpha,
            beta0=self.prior_beta,
            beta_init=self.beta,
            kappa=float(kappa),
            n_iter=int(n_iter),
        )
        lower = max(1e-9, float(min_ratio) * self.prior_mean)
        upper = max(lower, float(max_ratio) * self.prior_mean)
        clamped_mean = min(max(result.gamma, lower), upper)
        clamped = clamped_mean != result.gamma
        self.beta = self.alpha / clamped_mean
        self.observations += 1
        return float(clamped_mean), float(result.grad_final if result.converged else result.grad_final), clamped

    def snapshot(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "prior_beta": float(self.prior_beta),
            "prior_mean": float(self.prior_mean),
            "observations": float(self.observations),
        }


@dataclass(slots=True)
class ModalityWeights:
    """Frozen-per-evaluation multiplier set for the EFE modalities.

    Built once per ``evaluate*`` call and read-only thereafter, so the four
    spatial evaluator paths and the two summary paths cannot silently drift
    apart the way six verbatim copies of the precision block could.
    """

    gamma: dict[str, float]
    normalizer: dict[str, float]
    normalizer_name: dict[str, str]
    normalization_enabled: bool = True
    concentration_floor: float = 1.0

    def weight(self, name: str) -> float:
        return float(self.gamma.get(name, 1.0) * self.normalizer.get(name, 1.0))

    def motor_weight(self, channel_count: int) -> float:
        """Motor multiplier at the ACTUAL proprioceptive channel count.

        The nominal normalizer recorded in the component dataclasses uses
        :data:`DEFAULT_PROPRIOCEPTIVE_CHANNELS`; this recomputes the genuine
        per-channel density from the labels actually forecast.
        """

        divisor = 1.0 / max(1, int(channel_count)) if self.normalization_enabled else 1.0
        return float(self.gamma.get("motor_proprioceptive", 1.0) * divisor)

    @property
    def terminal(self) -> float:
        return self.weight("terminal_coverage")

    @property
    def ambiguity(self) -> float:
        return self.weight("observation_ambiguity")

    @property
    def transition(self) -> float:
        return self.weight("transition")

    @property
    def composition(self) -> float:
        return self.weight("composition_gap")

    @property
    def canvas(self) -> float:
        return self.weight("canvas_latent_transition")

    @property
    def relational(self) -> float:
        return self.weight("relational_transition")

    @property
    def motor(self) -> float:
        return self.weight("motor_proprioceptive")


def constant_modality_weights(config: PainterConfig) -> ModalityWeights:
    """ModalityWeights reproducing the declared constants exactly.

    Used wherever no ledger is injected, so every evaluator keeps a legal
    hand-written fallback and the belief mechanism stays attributable.
    """

    enabled = bool(config.modality_normalization_enabled)
    return ModalityWeights(
        gamma={name: float(_declared_constant(config, name)) for name in MODALITY_NAMES},
        normalizer=_normalizers(enabled, DEFAULT_PROPRIOCEPTIVE_CHANNELS),
        normalizer_name=dict(NORMALIZER_NAMES),
        normalization_enabled=enabled,
        concentration_floor=(
            float(config.terminal_forecast_concentration_floor) if enabled else 0.0
        ),
    )


def _normalizers(enabled: bool, proprioceptive_channels: int) -> dict[str, float]:
    if not enabled:
        return {name: 1.0 for name in MODALITY_NAMES}
    normalizers = {name: 1.0 for name in MODALITY_NAMES}
    normalizers["motor_proprioceptive"] = 1.0 / max(1, int(proprioceptive_channels))
    return normalizers


def _declared_constant(config: PainterConfig, name: str) -> float:
    field_name = _DECLARED_CONSTANT_FIELDS.get(name)
    if field_name is None:
        raise KeyError(f"No declared precision constant for {name!r}.")
    return float(getattr(config, field_name))


@dataclass(slots=True)
class PrecisionLedger:
    """One Gamma precision belief per EFE modality plus the policy precision.

    All eight keys are created EAGERLY in ``__post_init__``. Precision updates
    run on the planner thread while ``summary()`` runs on the HTTP thread, so
    the lazy get-or-create pattern used by ``MotionReliabilityLedger`` would
    raise "dictionary changed size during iteration" here.
    """

    config: PainterConfig
    beliefs: dict[str, GammaPrecisionBelief] = field(default_factory=dict)
    last_updates: dict[str, PrecisionUpdate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in LEDGER_KEYS:
            constant = _declared_constant(self.config, name)
            if constant <= 0.0:
                # A declared constant of exactly 0.0 structurally switches the
                # modality off. No belief is created, so a learned precision can
                # never resurrect an isolation config.
                continue
            self.beliefs[name] = GammaPrecisionBelief.from_prior(
                constant,
                self.config.precision_belief_alpha0,
            )

    # -- declared constants -------------------------------------------------
    def declared_constant(self, name: str) -> float:
        return _declared_constant(self.config, name)

    def _beliefs_enabled(self, name: str) -> bool:
        if not self.config.precision_beliefs_enabled:
            return False
        if name == POLICY_PRECISION_KEY:
            return True
        return bool(self.config.modality_precision_beliefs_enabled)

    def mean(self, name: str) -> float:
        constant = _declared_constant(self.config, name)
        if constant <= 0.0:
            return 0.0
        belief = self.beliefs.get(name)
        if belief is None or not self._beliefs_enabled(name):
            return float(constant)
        return belief.mean()

    def prior_mean(self, name: str) -> float:
        return float(_declared_constant(self.config, name))

    # -- updates ------------------------------------------------------------
    def observe_policy(
        self,
        G: object,
        F: object,
        *,
        log_prior: object | None = None,
    ) -> PrecisionUpdate:
        return self.observe(POLICY_PRECISION_KEY, G, F, log_prior=log_prior)

    def observe(
        self,
        name: str,
        G: object,
        F: object,
        *,
        log_prior: object | None = None,
    ) -> PrecisionUpdate:
        """Update one precision belief from a realized (G, F) candidate pair.

        ``G`` is that modality's post-normalization, PRE-gamma per-candidate
        contribution vector; ``F`` is the policy-dependent variational free
        energy of the same candidates. Both must be defined on the same index
        set, and neither may depend on the precision being updated.
        """

        constant = _declared_constant(self.config, name)
        gamma_now = self.mean(name)
        if constant <= 0.0:
            return self._record(
                PrecisionUpdate(
                    name=name,
                    gamma_before=0.0,
                    gamma_after=0.0,
                    gradient=0.0,
                    observations_used=0,
                    status="structurally_off",
                    approximation="declared constant is exactly 0.0; the modality is switched off structurally",
                )
            )
        belief = self.beliefs.get(name)
        if belief is None or not self._beliefs_enabled(name):
            return self._record(
                PrecisionUpdate(
                    name=name,
                    gamma_before=gamma_now,
                    gamma_after=gamma_now,
                    gradient=0.0,
                    observations_used=0,
                    status="disabled",
                    approximation="precision beliefs disabled by declared config flag",
                )
            )
        g_vector = np.asarray(list(G), dtype=float) if not isinstance(G, np.ndarray) else G.astype(float)
        f_vector = np.asarray(list(F), dtype=float) if not isinstance(F, np.ndarray) else F.astype(float)
        prior_vector = (
            None
            if log_prior is None
            else (
                np.asarray(list(log_prior), dtype=float)
                if not isinstance(log_prior, np.ndarray)
                else log_prior.astype(float)
            )
        )
        if g_vector.ndim != 1 or g_vector.shape != f_vector.shape:
            raise ValueError("Precision observations need aligned 1-D G and F vectors.")
        if g_vector.size < 2:
            return self._record(
                PrecisionUpdate(
                    name=name,
                    gamma_before=gamma_now,
                    gamma_after=gamma_now,
                    gradient=0.0,
                    observations_used=int(g_vector.size),
                    status="too_few_candidates",
                    approximation="a precision gradient needs at least two candidates to compare",
                )
            )
        if not (np.all(np.isfinite(g_vector)) and np.all(np.isfinite(f_vector))):
            return self._record(
                PrecisionUpdate(
                    name=name,
                    gamma_before=gamma_now,
                    gamma_after=gamma_now,
                    gradient=0.0,
                    observations_used=int(g_vector.size),
                    status="no_free_energy",
                    approximation="non-finite G or F; no update attempted",
                )
            )
        if float(np.max(f_vector) - np.min(f_vector)) <= 1e-12:
            # Measured degeneracy: a flat F makes the reference gradient exactly
            # 0.0 and reports converged=True. Reported, never hidden.
            return self._record(
                PrecisionUpdate(
                    name=name,
                    gamma_before=gamma_now,
                    gamma_after=gamma_now,
                    gradient=0.0,
                    observations_used=int(g_vector.size),
                    status="degenerate_flat_F",
                    approximation=(
                        "policy-dependent free energy is flat across candidates, so dF/dgamma is "
                        "identically zero and the rule provably cannot learn"
                    ),
                )
            )
        gamma_after, gradient, clamped = belief.update(
            g_vector,
            f_vector,
            log_prior=prior_vector,
            kappa=self.config.precision_belief_kappa,
            n_iter=self.config.precision_belief_iterations,
            min_ratio=self.config.precision_belief_min_ratio,
            max_ratio=self.config.precision_belief_max_ratio,
        )
        return self._record(
            PrecisionUpdate(
                name=name,
                gamma_before=gamma_now,
                gamma_after=gamma_after,
                gradient=gradient,
                observations_used=int(g_vector.size),
                status="clamped" if clamped else "updated",
                approximation=(
                    "reference Ch.10 precision descent; Gamma rate warm-started at the current "
                    "posterior rate and the posterior mean clamped to the declared bounded support"
                ),
            )
        )

    def _record(self, update: PrecisionUpdate) -> PrecisionUpdate:
        self.last_updates[update.name] = update
        return update

    # -- weights ------------------------------------------------------------
    def weights(
        self,
        *,
        proprioceptive_channels: int = DEFAULT_PROPRIOCEPTIVE_CHANNELS,
    ) -> ModalityWeights:
        enabled = bool(self.config.modality_normalization_enabled)
        return ModalityWeights(
            gamma={name: self.mean(name) for name in MODALITY_NAMES},
            normalizer=_normalizers(enabled, proprioceptive_channels),
            normalizer_name=dict(NORMALIZER_NAMES),
            normalization_enabled=enabled,
            concentration_floor=(
                float(self.config.terminal_forecast_concentration_floor) if enabled else 0.0
            ),
        )

    # -- telemetry / persistence -------------------------------------------
    def summary(self) -> dict[str, dict[str, float | str]]:
        payload: dict[str, dict[str, float | str]] = {}
        for name in LEDGER_KEYS:
            belief = self.beliefs.get(name)
            update = self.last_updates.get(name)
            gamma = self.mean(name)
            payload[name] = {
                "gamma": float(gamma) if math.isfinite(gamma) else 0.0,
                "priorGamma": float(self.prior_mean(name)),
                "gradient": float(update.gradient) if update is not None and math.isfinite(update.gradient) else 0.0,
                "observations": float(belief.observations) if belief is not None else 0.0,
                "status": update.status if update is not None else ("structurally_off" if belief is None else "prior"),
                "resolvableUncertaintyNats": (
                    float(belief.resolvable_uncertainty()) if belief is not None else 0.0
                ),
                "normalizerName": NORMALIZER_NAMES.get(name, "policy_softmax_inverse_temperature"),
                "beliefsEnabled": float(bool(self._beliefs_enabled(name) and belief is not None)),
            }
        return payload

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {name: belief.snapshot() for name, belief in self.beliefs.items()}

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        for name, state in payload.items():
            key = str(name)
            if key not in self.beliefs or not isinstance(state, dict):
                continue
            if "alpha" not in state or "beta" not in state:
                continue
            existing = self.beliefs[key]
            self.beliefs[key] = GammaPrecisionBelief(
                alpha=float(state["alpha"]),
                beta=float(state["beta"]),
                prior_beta=float(state.get("prior_beta", existing.prior_beta)),
                # The prior mean is the DECLARED constant, so it is re-read from
                # the live config rather than trusted from disk.
                prior_mean=float(existing.prior_mean),
                observations=int(float(state.get("observations", 0.0))),
            )


@dataclass(slots=True)
class GapIncrementBelief:
    """Gaussian random-walk belief over the per-mark compression-gap increment.

    A TRANSITION PRIOR over a rate of change, updated by a scalar Kalman step
    from observed gaps. It is structurally barred from expected free energy:
    its only consumer is the stop POLICY PRIOR, and a test asserts every EFE
    component is bit-identical before and after feeding it a large increment.

    Named approximation: the increment is amortized over the exact number of
    marks elapsed between planning-cadence gap readings, because reading the
    gap at mark completion would run a model forward on the polling thread.
    """

    mean: float
    variance: float
    process_variance: float
    observation_variance: float
    last_gap: float | None = None
    last_mark_index: int = 0
    observations: int = 0

    @classmethod
    def from_config(cls, config: PainterConfig) -> "GapIncrementBelief":
        return cls(
            mean=float(config.gap_increment_prior_mean),
            variance=max(1e-12, float(config.gap_increment_prior_std) ** 2),
            process_variance=max(1e-12, float(config.gap_increment_process_std) ** 2),
            observation_variance=max(1e-12, float(config.gap_increment_observation_std) ** 2),
        )

    def has_observations(self) -> bool:
        return self.observations > 0

    def posterior_mean(self) -> float:
        return float(self.mean)

    def posterior_precision(self) -> float:
        return float(1.0 / max(self.variance, 1e-12))

    def standardized_progress(self) -> float:
        return float(self.mean / math.sqrt(max(self.variance, 1e-12) + 1e-12))

    def observe(self, gap: float | None, mark_index: int) -> bool:
        """Kalman-update from a gap reading. False on the first reading."""

        if gap is None or not math.isfinite(float(gap)):
            return False
        current = float(gap)
        index = int(mark_index)
        if self.last_gap is None:
            self.last_gap = current
            self.last_mark_index = index
            return False
        previous = self.last_gap
        elapsed = index - self.last_mark_index
        # Always advance the anchor, so consecutive readings are differenced.
        self.last_gap = current
        self.last_mark_index = index
        if elapsed <= 0:
            # No mark completed since the previous reading, so this gap change is
            # composition-model TRAINING drift, not a per-mark increment. Counting
            # it would attribute learning progress to mark-making. The reading
            # still advances the anchor, so the drift is discarded rather than
            # folded into the next genuine increment.
            return False
        increment = (current - previous) / float(elapsed)
        # Random walk: predict (variance grows by the process variance), then
        # fuse the observed increment under its declared observation variance.
        predicted_variance = self.variance + self.process_variance
        gain = predicted_variance / (predicted_variance + self.observation_variance)
        self.mean = float(self.mean + gain * (increment - self.mean))
        self.variance = float(max(1e-12, (1.0 - gain) * predicted_variance))
        self.observations += 1
        return True

    def stop_log_prior_term(self, config: PainterConfig) -> float:
        """Second declared factor on the stop policy prior, in log space.

        ``logsigmoid(-sharpness * mean / sqrt(variance))`` uses BOTH the
        posterior mean and its precision. It is <= 0 always, so it can only
        make stopping less unlikely and can never manufacture positive value
        for any candidate. Exactly 0.0 when the belief has no observations or
        the declared flag is off, which is what preserves the pre-existing
        coverage-sigmoid identity at the midpoint.
        """

        if not config.gap_progress_stop_enabled or not self.has_observations():
            return 0.0
        logit = -float(config.gap_progress_stop_sharpness) * self.standardized_progress()
        if logit >= 0.0:
            return -float(math.log1p(math.exp(-logit)))
        return float(logit - math.log1p(math.exp(logit)))

    def summary(self) -> dict[str, float]:
        return {
            "posteriorMean": float(self.mean),
            "posteriorStd": float(math.sqrt(max(self.variance, 1e-12))),
            "standardizedProgress": float(self.standardized_progress()),
            "observations": float(self.observations),
            "lastGap": float(self.last_gap) if self.last_gap is not None else 0.0,
            "lastMarkIndex": float(self.last_mark_index),
            "hasObservations": float(bool(self.has_observations())),
        }

    def snapshot(self) -> dict[str, float]:
        return {
            "mean": float(self.mean),
            "variance": float(self.variance),
            "last_gap": float(self.last_gap) if self.last_gap is not None else float("nan"),
            "last_mark_index": float(self.last_mark_index),
            "observations": float(self.observations),
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if "mean" not in payload or "variance" not in payload:
            return
        self.mean = float(payload["mean"])
        self.variance = max(1e-12, float(payload["variance"]))
        last_gap = float(payload.get("last_gap", float("nan")))
        self.last_gap = None if math.isnan(last_gap) else last_gap
        self.last_mark_index = int(float(payload.get("last_mark_index", 0.0)))
        self.observations = int(float(payload.get("observations", 0.0)))
