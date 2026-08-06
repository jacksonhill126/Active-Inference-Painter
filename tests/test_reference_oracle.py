"""Reference-oracle verification harness.

Every assertion here cross-checks a quantity this project computes against the
vendored textbook implementation (``Reference material/``) or against an
independent numerical integration. The harness is read-only measurement
apparatus: it constructs no decision quantity, and nothing under
``src/active_painter/`` imports it.

Each test's docstring names the identity under test. Where an oracle is only
valid inside a restricted regime, the restriction is stated in a comment
alongside the parametrization -- a tolerance that silently passes outside its
band of validity is worse than no oracle.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch
from torch.distributions import Beta, Normal

from _reference_loader import load_reference_module

from active_painter.agent import ActiveInferencePainter
from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.efe import EFEComponents, ExpectedFreeEnergy
from active_painter.efe_common import (
    coverage_beta_approximation,
    project_material_support,
    terminal_preference_terms,
)
from active_painter.env import StrokeAction
from active_painter.inference import VariationalStateEstimator
from active_painter.models import (
    DynamicsEnsemble,
    GaussianBelief,
    LocalSpatialDynamicsEnsemble,
    ObservationModel,
    SpatialDynamicsEnsemble,
)
from active_painter.motor_planning import motor_efe_contribution, motor_efe_terms
from active_painter.policies import (
    Policy,
    policy_posterior_from_efe,
    policy_stop_log_prior,
)
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_inference import (
    SpatialVariationalStateEstimator,
    spatial_observation_variance,
)
from active_painter.local_spatial import (
    local_patch_bounds_for_raster,
    pixel_logvar_from_state,
    pixel_material_from_state,
)
from active_painter.spatial_state import (
    SpatialCanvasState,
    independent_material_channel_count,
    rasterize_stroke_action,
    spatial_canvas_state,
)
from active_painter.stroke_execution import forecast_stroke_execution


@pytest.fixture(scope="module")
def pomdp():
    """The reference discrete-POMDP oracle module.

    Loaded through the stub-package loader; the whole file skips cleanly when
    the vendored reference tree is absent.
    """

    return load_reference_module("core.pomdp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discretize_beta(dist: Beta, bins: int) -> np.ndarray:
    """Midpoint-rule discretization of a continuous Beta onto uniform bins.

    APPROXIMATION: a continuous Beta density on material coverage in [0, 1] is
    represented by its density at the midpoint of each of ``bins`` uniform
    coverage bins, renormalized to a categorical. The reference oracle is
    discrete-only, so a discretization is unavoidable; it is declared here and
    in the approximation register rather than hidden inside the comparison.
    """

    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    centres = torch.clamp(0.5 * (edges[:-1] + edges[1:]), 1e-9, 1.0 - 1e-9)
    reference = Beta(
        dist.concentration1.to(torch.float64).reshape(()),
        dist.concentration0.to(torch.float64).reshape(()),
    )
    log_density = reference.log_prob(centres)
    weights = torch.exp(log_density - log_density.max())
    return (weights / weights.sum()).numpy()


class _FixedVarianceObservationStub:
    """Homoscedastic 1-D likelihood p(o|s) = N(o; s, variance).

    The production ``ObservationModel.std`` is state dependent (smear-driven),
    so the summary VFE problem it defines has no analytic posterior. This stub
    keeps the same likelihood *interface* while making the conjugate Gaussian
    posterior exact, which is what turns the estimator into a testable object.
    """

    def __init__(self, variance: float | tuple[float, ...]) -> None:
        self.variance = torch.as_tensor(variance, dtype=torch.float64)

    def distribution(self, state: torch.Tensor) -> Normal:
        variance = self.variance.to(device=state.device, dtype=state.dtype)
        return Normal(state, torch.sqrt(variance))


class _DeterministicSummaryDynamics:
    """Summary transition stub with hand-declared predictive moments.

    Deliberately NOT a ``DynamicsEnsemble`` subclass so ``ExpectedFreeEnergy``
    routes through the moment-matched mixture path, where the transition terms
    are exact functions of these constants rather than of trained weights.
    """

    coverage_step: float = 0.06
    aleatoric_variance: float = 2.5e-5
    epistemic_variance: float = 7.5e-5

    def predictive_moments(
        self,
        mean: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_mean = mean.clone()
        next_mean[..., 0] = torch.clamp(mean[..., 0] + self.coverage_step, 1e-4, 1.0 - 1e-4)
        aleatoric = torch.full_like(next_mean, self.aleatoric_variance)
        epistemic = torch.full_like(next_mean, self.epistemic_variance)
        return next_mean, aleatoric, epistemic


class _DenseSpatialStubDynamics:
    """Dense spatial transition stub with declared constant moments."""

    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = material + 0.01 * action_raster[:, :1]
        aleatoric = torch.full_like(material, 4e-4)
        epistemic = torch.full_like(material, 1e-4)
        return mean, aleatoric, epistemic


class _ControlledVarianceSpatialDynamics:
    """Identity-mean transition with a declared diagonal variance."""

    def __init__(self, variance: float) -> None:
        self.variance = float(variance)

    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del action_raster
        return (
            material.clone(),
            torch.full_like(material, self.variance),
            torch.zeros_like(material),
        )


def _fine_grid_vfe(
    q_mean: float,
    q_variance: float,
    prior_mean: float,
    prior_variance: float,
    observation: float,
    observation_variance: float,
    *,
    points: int = 400001,
    span: float = 12.0,
) -> tuple[float, float, float]:
    """Numerically integrate F = KL(q||p) - E_q[log p(o|s)] on a fine 1-D grid.

    Rectangle rule with explicit ``sum(...) * dx``: no scipy, no ``np.trapz``,
    so this integration shares no code path whatsoever with the estimator under
    test. Returns ``(F, KL, -E_q[log p(o|s)])``.
    """

    low = q_mean - span * math.sqrt(q_variance)
    high = q_mean + span * math.sqrt(q_variance)
    grid = np.linspace(low, high, points)
    dx = (high - low) / (points - 1)
    log_q = -0.5 * np.log(2.0 * np.pi * q_variance) - 0.5 * (grid - q_mean) ** 2 / q_variance
    log_p = -0.5 * np.log(2.0 * np.pi * prior_variance) - 0.5 * (grid - prior_mean) ** 2 / prior_variance
    log_likelihood = -0.5 * np.log(2.0 * np.pi * observation_variance) - 0.5 * (
        observation - grid
    ) ** 2 / observation_variance
    q = np.exp(log_q)
    kl = float(np.sum(q * (log_q - log_p)) * dx)
    negative_log_likelihood = float(-np.sum(q * log_likelihood) * dx)
    return kl + negative_log_likelihood, kl, negative_log_likelihood


def _analytic_diagonal_gaussian_vfe(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
) -> tuple[float, float, float]:
    """Independent closed form for a factorized linear-Gaussian VFE."""

    complexity = 0.5 * np.sum(
        np.log(prior_variance / posterior_variance)
        + (posterior_variance + (posterior_mean - prior_mean) ** 2) / prior_variance
        - 1.0
    )
    negative_log_likelihood = 0.5 * np.sum(
        np.log(2.0 * np.pi * observation_variance)
        + ((observation - posterior_mean) ** 2 + posterior_variance)
        / observation_variance
    )
    return (
        float(complexity + negative_log_likelihood),
        float(complexity),
        float(negative_log_likelihood),
    )


# The 1-D conjugate problem used by the summary-VFE tests. Prior N(0.30, 0.04),
# homoscedastic likelihood variance 0.01, observation 0.55; the exact posterior
# is therefore N(0.5, 0.008).
_PRIOR_MEAN = 0.30
_PRIOR_VARIANCE = 0.04
_LIKELIHOOD_VARIANCE = 0.01
_OBSERVATION = 0.55
_POSTERIOR_VARIANCE = 1.0 / (1.0 / _PRIOR_VARIANCE + 1.0 / _LIKELIHOOD_VARIANCE)
_POSTERIOR_MEAN = _POSTERIOR_VARIANCE * (
    _PRIOR_MEAN / _PRIOR_VARIANCE + _OBSERVATION / _LIKELIHOOD_VARIANCE
)


def _summary_conjugate_inference(seed: int) -> tuple[VariationalStateEstimator, GaussianBelief]:
    torch.manual_seed(seed)
    estimator = VariationalStateEstimator(
        PainterConfig(), _FixedVarianceObservationStub(_LIKELIHOOD_VARIANCE)
    )
    prior = GaussianBelief(
        torch.tensor([_PRIOR_MEAN], dtype=torch.float64),
        torch.tensor([math.log(_PRIOR_VARIANCE)], dtype=torch.float64),
    )
    posterior = estimator.infer(prior, torch.tensor([_OBSERVATION], dtype=torch.float64))
    return estimator, posterior


def _spatial_belief_pair(config: PainterConfig, seed: int = 0) -> tuple[SpatialCanvasState, SpatialCanvasState]:
    """A previous-belief / observation pair of dense material fields."""

    grid = config.spatial_grid_size
    channels = config.spatial_material_channels
    rng = np.random.default_rng(seed)
    material = np.zeros((channels, grid, grid), dtype=np.float32)
    material[0] = rng.uniform(0.0, 0.02, (grid, grid))
    material[1] = rng.uniform(0.0, 0.01, (grid, grid))
    material[2] = rng.uniform(0.0, 0.02, (grid, grid))
    if channels > 3:
        material[3] = rng.uniform(0.2, 0.8, (grid, grid))
    if channels > 4:
        material[4] = rng.uniform(0.0, 0.3, (grid, grid))
    if channels > 5:
        material[5] = rng.uniform(0.0, 1.0, (grid, grid))
    previous = SpatialCanvasState(material, np.full_like(material, -8.0))
    observed = np.clip(material + rng.normal(0.0, 0.005, material.shape), 0.0, None).astype(np.float32)
    observation = SpatialCanvasState(observed, np.full_like(material, -8.0))
    return previous, observation


# ---------------------------------------------------------------------------
# (a) terminal coverage risk == the reference full-KL risk
# ---------------------------------------------------------------------------


# The reference floors both logs at _EPS = 1e-16 (pomdp.py), so the discretized
# KL diverges from the continuous KL wherever the forecast puts mass where the
# Beta preference is numerically zero -- and that error GROWS with bin count.
# Measured divergence outside the band: coverage mean 0.60 / var 0.0020 is off
# by 7.7e-2 at 512 bins, and 0.30 / 0.0050 by 4.8e+1. The parametrization is
# therefore confined to the measured-valid band coverage mean in [0.70, 0.90].
# Do NOT sweep bin count expecting convergence, and do NOT read this test as
# certifying terminal risk in the low-coverage cold-start regime.
@pytest.mark.parametrize(
    ("coverage_mean", "coverage_variance", "bins"),
    [
        (0.87, 0.0010, 512),
        (0.85, 0.0008, 512),
        (0.80, 0.0015, 512),
        (0.75, 0.0015, 2048),
        (0.70, 0.0015, 512),
    ],
)
def test_terminal_coverage_risk_is_the_reference_full_kl_risk(
    pomdp,
    coverage_mean: float,
    coverage_variance: float,
    bins: int,
) -> None:
    """Identity: risk = q(o) . (log q(o) - log C), the FULL KL including -H[q(o)].

    The reference computes that dot product on a discrete outcome simplex
    (``pomdp.efe_components``). The project computes the same KL in closed form
    on the continuous Beta pair, as ``risk = -terminal_entropy -
    pragmatic_value``. Both must agree once the continuous pair is discretized
    onto a common coverage-bin grid.
    """

    config = PainterConfig()
    preference = TerminalCoveragePreference(config)
    mean = torch.tensor([coverage_mean], dtype=torch.float32)
    variance = torch.tensor([coverage_variance], dtype=torch.float32)

    project_risk, project_entropy, project_pragmatic = terminal_preference_terms(
        preference, mean, variance, precision=1.0
    )
    # Pin the closed form independently of the reference: risk is a genuine KL,
    # i.e. it carries the -H[q(o)] term, not just the pragmatic cross entropy.
    assert float(project_risk[0]) == pytest.approx(
        -float(project_entropy[0]) - float(project_pragmatic[0]), abs=1e-6
    )

    forecast = _discretize_beta(coverage_beta_approximation(mean, variance), bins)
    target = _discretize_beta(preference.distribution("cpu"), bins)
    # Normalized even though _nonnegative_vector would accept raw preferences:
    # only a normalized C makes the reference's risk a genuine KL rather than a
    # KL plus an arbitrary log constant.
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-12)

    model = pomdp.POMDPModel(A=np.eye(bins), D=forecast)
    reference = pomdp.efe_components(model, forecast, target)

    # Identity likelihood => zero ambiguity => reference.total IS reference.risk,
    # so this is a pure risk-to-risk comparison with no confounding term.
    assert reference.ambiguity == pytest.approx(0.0, abs=1e-12)
    assert reference.total == pytest.approx(reference.risk, abs=1e-12)
    assert abs(float(project_risk[0]) - reference.risk) < 1e-3


# ---------------------------------------------------------------------------
# (b) transition decomposition == negative parameter information gain
# ---------------------------------------------------------------------------


def test_transition_decomposition_is_the_negative_parameter_information_gain(pomdp) -> None:
    """Identity: transition_risk + transition_ambiguity == -I(theta; s') exactly.

    The project logs ``epistemic_value`` as that information-gain identity and
    does NOT subtract it into ``.total`` -- unlike the reference, whose
    ``EFEComponents.total = risk + ambiguity - novelty``. The two are different
    quantities: the project's ``epistemic_value`` is the state-transition
    information gain already *implied* by risk + ambiguity, while the
    reference's ``novelty`` is Dirichlet PARAMETER novelty over ``A``. Both
    facts are asserted here so that nobody "harmonizes" the sign difference and
    reintroduces a double count.
    """

    config = PainterConfig()
    efe = ExpectedFreeEnergy(
        config,
        _DeterministicSummaryDynamics(),
        ObservationModel(config),
        TerminalCoveragePreference(config),
    )
    belief = GaussianBelief(
        torch.tensor([0.70, 0.02, 0.02, 0.10, 0.05, 0.40], dtype=torch.float32),
        torch.full((6,), -12.0),
    )
    policy = Policy(
        (
            StrokeAction(0.2, 0.2, 0.8, 0.8, 0.08, 0.7, 1.0),
            StrokeAction(0.3, 0.3, 0.7, 0.7, 0.08, 0.7, 1.0),
            StrokeAction.stop_action(),
        )
    )
    components = efe.evaluate(belief, policy)

    transition_information = components.transition_risk + components.transition_ambiguity
    assert transition_information == pytest.approx(-components.epistemic_value, abs=1e-5)
    assert components.epistemic_value > 0.0

    reconstructed = (
        components.terminal_risk
        + components.ambiguity
        + components.transition_risk
        + components.transition_ambiguity
    )
    assert components.total == pytest.approx(reconstructed, abs=1e-5)
    # Positive regression pin for "no double counting": the reference-style
    # extra `- epistemic_value` subtraction would yield a materially different
    # total, so the project demonstrably does not perform it.
    assert components.total != pytest.approx(reconstructed - components.epistemic_value, abs=1e-3)

    # Reference contrast. `novelty` is zero unless Dirichlet counts are supplied,
    # and when they are it is parameter novelty over A -- an information gain
    # about MODEL PARAMETERS, which the reference subtracts. That quantity has
    # no counterpart among the project's canvas terms, so the differing sign
    # convention is correct and must not be reconciled.
    counts = np.array([[6.0, 1.0], [1.0, 6.0]])
    state = np.array([0.6, 0.4])
    preference = np.array([0.9, 0.1])
    model = pomdp.POMDPModel(A=pomdp.expected_A(counts), D=state)
    without_counts = pomdp.efe_components(model, state, preference)
    with_counts = pomdp.efe_components(model, state, preference, a=counts)
    assert without_counts.novelty == pytest.approx(0.0)
    assert with_counts.novelty == pytest.approx(pomdp.parameter_novelty(counts, state))
    assert with_counts.novelty > 0.0
    assert with_counts.total == pytest.approx(
        with_counts.risk + with_counts.ambiguity - with_counts.novelty
    )


# ---------------------------------------------------------------------------
# (c) policy posterior == the reference full policy posterior
# ---------------------------------------------------------------------------


_STUB_EFE_SCORES = (1.2, -0.4, 3.3, 0.0, 0.7, -1.1, 2.0, 0.3)


def _stub_summary_evaluate_batch(coverage: float):
    def evaluate_batch(belief, policies):
        return [
            EFEComponents(
                total=_STUB_EFE_SCORES[index % len(_STUB_EFE_SCORES)],
                terminal_risk=0.0,
                ambiguity=0.0,
                epistemic_value=0.0,
                terminal_coverage_mean=coverage,
                terminal_coverage_std=0.01,
            )
            for index, _ in enumerate(policies)
        ]

    return evaluate_batch


def test_policy_posterior_matches_reference_policy_posterior_full(pomdp) -> None:
    """Identity: Q(pi) = softmax(log E - F - gamma * G) with F = 0.

    Both painters shift G by ``- g.min()`` before the softmax, which is
    softmax-invariant, and both add the declared stop log prior as ``log E``
    without normalizing it -- also softmax-invariant. This test drives a
    controlled G through the real ``infer_policy`` code and compares the
    returned posterior against ``pomdp.policy_posterior_full``.

    Out of scope by design: ``arm_agent_driver`` also runs a third, marginal
    motor-evidence softmax over realization latents. That is a different
    distribution (q(motor realization | painting policy)) and is covered by
    ``tests/test_stroke_execution.py``.
    """

    coverage = 0.42
    config = PainterConfig(candidate_policies=8, planning_horizon=1)

    summary_agent = ActiveInferencePainter(config, seed=3, device="cpu")
    summary_agent.belief = GaussianBelief(
        torch.tensor([coverage, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -12.0),
    )
    summary_agent.efe = types.SimpleNamespace(
        evaluate_batch=_stub_summary_evaluate_batch(coverage)
    )

    spatial_config = PainterConfig(
        candidate_policies=8,
        planning_horizon=1,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_gap_precision=0.0,
    )
    spatial_agent = SpatialActiveInferencePainter(spatial_config, seed=5, device="cpu")
    spatial_coverage = spatial_agent.belief.material_coverage_mean(
        spatial_config.paint_presence_threshold
    )
    spatial_agent.efe = types.SimpleNamespace(
        evaluate_batch=_stub_summary_evaluate_batch(spatial_coverage)
    )

    for agent, cfg, believed in (
        (summary_agent, config, coverage),
        (spatial_agent, spatial_config, spatial_coverage),
    ):
        torch.manual_seed(0)
        _, _, ranked = agent.infer_policy()
        g = np.asarray([component.total for _, component, _ in ranked], dtype=float)
        log_prior = np.asarray(
            [policy_stop_log_prior(policy, believed, cfg) for policy, _, _ in ranked],
            dtype=float,
        )
        project = np.asarray([probability for _, _, probability in ranked], dtype=float)
        reference = pomdp.policy_posterior_full(
            g, F=None, E=np.exp(log_prior), gamma=cfg.policy_precision
        )
        # Production builds the softmax in float32; measured worst-case
        # deviation from the float64 reference is ~2e-8.
        np.testing.assert_allclose(project, reference, rtol=1e-6, atol=1e-6)


def test_discrete_efe_acceptance_matrix_and_policy_priors(pomdp) -> None:
    """Enumerated controls isolate deterministic, ambiguous, epistemic, and preferred cases.

    The informative and ambiguous controls have the same uniform predicted
    outcome and uniform preference. Their risk is therefore identical and the
    EFE difference is purely information-theoretic: deterministic observations
    carry one bit of state information, while the uniform likelihood carries
    none. The preference controls use deterministic observations, so their EFE
    difference is purely risk. No production EFE helper is reused to construct
    these expected values.
    """

    identity = np.eye(2, dtype=np.float64)
    uniform_likelihood = np.full((2, 2), 0.5, dtype=np.float64)
    uniform_state = np.asarray([0.5, 0.5], dtype=np.float64)
    uniform_preference = np.asarray([0.5, 0.5], dtype=np.float64)

    deterministic = pomdp.efe_components(
        pomdp.POMDPModel(A=identity, D=np.asarray([1.0, 0.0])),
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
    )
    informative = pomdp.efe_components(
        pomdp.POMDPModel(A=identity, D=uniform_state),
        uniform_state,
        uniform_preference,
    )
    ambiguous = pomdp.efe_components(
        pomdp.POMDPModel(A=uniform_likelihood, D=uniform_state),
        uniform_state,
        uniform_preference,
    )

    assert deterministic.risk == pytest.approx(0.0, abs=1e-12)
    assert deterministic.ambiguity == pytest.approx(0.0, abs=1e-12)
    assert deterministic.total == pytest.approx(0.0, abs=1e-12)
    assert informative.risk == pytest.approx(0.0, abs=1e-12)
    assert informative.ambiguity == pytest.approx(0.0, abs=1e-12)
    assert ambiguous.risk == pytest.approx(0.0, abs=1e-12)
    assert ambiguous.ambiguity == pytest.approx(math.log(2.0), abs=1e-12)
    assert ambiguous.total - informative.total == pytest.approx(math.log(2.0), abs=1e-12)

    # Manual mutual information, independent of the reference helper. Both
    # controls predict H[O] = log(2), but only the identity likelihood resolves
    # the hidden state.
    informative_information = math.log(2.0) - 0.0
    ambiguous_information = math.log(2.0) - math.log(2.0)
    assert informative_information == pytest.approx(math.log(2.0))
    assert ambiguous_information == pytest.approx(0.0)
    assert ambiguous.total - informative.total == pytest.approx(
        informative_information - ambiguous_information
    )

    preference = np.asarray([0.9, 0.1], dtype=np.float64)
    good = pomdp.efe_components(
        pomdp.POMDPModel(A=identity, D=np.asarray([1.0, 0.0])),
        np.asarray([1.0, 0.0]),
        preference,
    )
    bad = pomdp.efe_components(
        pomdp.POMDPModel(A=identity, D=np.asarray([0.0, 1.0])),
        np.asarray([0.0, 1.0]),
        preference,
    )
    assert good.ambiguity == pytest.approx(0.0, abs=1e-12)
    assert bad.ambiguity == pytest.approx(0.0, abs=1e-12)
    assert good.risk == pytest.approx(-math.log(0.9), abs=1e-12)
    assert bad.risk == pytest.approx(-math.log(0.1), abs=1e-12)
    assert good.total < bad.total

    g = torch.tensor([bad.total, good.total], dtype=torch.float64)
    no_prior = policy_posterior_from_efe(g, torch.zeros_like(g), gamma=2.0).numpy()
    no_prior_reference = pomdp.policy_posterior_full(g.numpy(), gamma=2.0)
    np.testing.assert_allclose(no_prior, no_prior_reference, atol=1e-12, rtol=0.0)
    assert no_prior[1] > no_prior[0]

    # A declared prior can oppose preferences without changing G. Production
    # and reference posteriors must still agree after normalization.
    policy_prior = np.asarray([0.95, 0.05], dtype=np.float64)
    with_prior = policy_posterior_from_efe(
        g,
        torch.tensor(np.log(policy_prior), dtype=torch.float64),
        gamma=2.0,
    ).numpy()
    with_prior_reference = pomdp.policy_posterior_full(
        g.numpy(), E=policy_prior, gamma=2.0
    )
    np.testing.assert_allclose(with_prior, with_prior_reference, atol=1e-12, rtol=0.0)
    assert float(no_prior.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(with_prior.sum()) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (d) Gaussian VFE against independent numerical integration
# ---------------------------------------------------------------------------


def test_summary_vfe_complexity_matches_the_analytic_gaussian_kl() -> None:
    """Identity: the complexity term of F = KL(q||p) - E_q[log p(o|s)].

    ``VariationalStateEstimator`` evaluates it with ``torch.kl_divergence``;
    this asserts the reported value against fine-grid numerical integration of
    the same KL at the estimator's own q, so a wrong prior or a transposed
    argument order cannot pass.
    """

    estimator, posterior = _summary_conjugate_inference(seed=0)
    q_mean = float(posterior.mean[0])
    q_variance = float(posterior.logvar.exp()[0])
    _, grid_kl, _ = _fine_grid_vfe(
        q_mean,
        q_variance,
        _PRIOR_MEAN,
        _PRIOR_VARIANCE,
        _OBSERVATION,
        _LIKELIHOOD_VARIANCE,
    )

    assert estimator.last_vfe is not None
    # Measured deviation across seeds 0-4: <= 3.4e-16.
    assert estimator.last_vfe.complexity == pytest.approx(grid_kl, abs=1e-4)
    assert estimator.last_vfe.units == "nats"


def test_summary_variational_posterior_approaches_the_conjugate_solution() -> None:
    """Identity: argmin_q F = the exact Bayesian posterior for a conjugate pair.

    With a Gaussian prior and a homoscedastic Gaussian likelihood the exact
    posterior is N(0.5, 0.008). The estimator is an approximate minimizer
    (config.inference_steps = 24 Adam steps at inference_lr = 0.08 on an
    8-sample reparameterized gradient), so the tolerances are the declared
    optimizer budget's accuracy, not the identity's.
    """

    _, posterior = _summary_conjugate_inference(seed=0)

    # Measured across seeds 0-4: mean error 0.006-0.036, variance error
    # 0.001-0.004.
    assert float(posterior.mean[0]) == pytest.approx(_POSTERIOR_MEAN, abs=0.05)
    assert float(posterior.logvar.exp()[0]) == pytest.approx(_POSTERIOR_VARIANCE, abs=0.01)


def test_summary_vfe_total_matches_fine_grid_integration_within_declared_monte_carlo_error() -> None:
    """Identity: F = KL(q||p) - E_q[log p(o|s)], evaluated at the reported q.

    The reported ``total`` estimates the expectation with the declared 4096
    posterior samples. Monte Carlo error does not shrink with more optimizer
    steps, so this remains a declared approximation rather than an analytic
    identity. The larger reporting-only budget closes the old +-0.35 nat band
    without changing the inferred posterior, EFE, or policy selection.
    """

    reported: list[float] = []
    integrated: list[float] = []
    for seed in range(5):
        estimator, posterior = _summary_conjugate_inference(seed)
        assert estimator.last_vfe is not None
        assert "4096 posterior state samples" in estimator.last_vfe.approximation
        grid_total, _, _ = _fine_grid_vfe(
            float(posterior.mean[0]),
            float(posterior.logvar.exp()[0]),
            _PRIOR_MEAN,
            _PRIOR_VARIANCE,
            _OBSERVATION,
            _LIKELIHOOD_VARIANCE,
        )
        assert estimator.last_vfe.total == pytest.approx(grid_total, abs=0.05)
        reported.append(estimator.last_vfe.total)
        integrated.append(grid_total)

    assert float(np.mean(reported)) == pytest.approx(float(np.mean(integrated)), abs=0.025)


def test_summary_vfe_reporting_does_not_advance_the_learning_rng_stream() -> None:
    """The reporting-only Monte Carlo estimate cannot perturb later inference."""

    config = PainterConfig(inference_steps=0, summary_vfe_report_samples=64)
    prior = GaussianBelief(
        torch.tensor([_PRIOR_MEAN], dtype=torch.float64),
        torch.tensor([math.log(_PRIOR_VARIANCE)], dtype=torch.float64),
    )
    observation = torch.tensor([_OBSERVATION], dtype=torch.float64)

    torch.manual_seed(2026)
    expected_next = torch.rand(8, dtype=torch.float64)
    torch.manual_seed(2026)
    estimator = VariationalStateEstimator(
        config, _FixedVarianceObservationStub(_LIKELIHOOD_VARIANCE)
    )
    estimator.infer(prior, observation)
    actual_next = torch.rand(8, dtype=torch.float64)

    assert torch.equal(actual_next, expected_next)
    assert estimator.last_vfe is not None
    assert "64 posterior state samples" in estimator.last_vfe.approximation


def test_multivariate_summary_vfe_selects_the_analytic_diagonal_gaussian_posterior() -> None:
    """A 3-D conjugate fixture verifies all reported VFE terms and the minimizer.

    The likelihood is diagonal and homoscedastic per dimension, so the exact
    posterior follows by adding prior and likelihood precisions. The production
    estimator still minimizes with reparameterized samples; a larger optimizer
    budget is declared for this validation fixture so optimization error does
    not obscure the identity being tested.
    """

    prior_mean = np.asarray([-0.2, 0.3, 1.1], dtype=np.float64)
    prior_variance = np.asarray([0.04, 0.25, 0.01], dtype=np.float64)
    observation = np.asarray([0.4, -0.1, 0.7], dtype=np.float64)
    observation_variance = np.asarray([0.09, 0.02, 0.16], dtype=np.float64)
    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / observation_variance)
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + observation / observation_variance
    )

    torch.manual_seed(0)
    estimator = VariationalStateEstimator(
        PainterConfig(inference_steps=512, inference_lr=0.02),
        _FixedVarianceObservationStub(tuple(observation_variance)),
    )
    inferred = estimator.infer(
        GaussianBelief(
            torch.tensor(prior_mean, dtype=torch.float64),
            torch.tensor(np.log(prior_variance), dtype=torch.float64),
        ),
        torch.tensor(observation, dtype=torch.float64),
    )
    inferred_mean = inferred.mean.numpy()
    inferred_variance = inferred.logvar.exp().numpy()

    np.testing.assert_allclose(inferred_mean, posterior_mean, atol=0.02, rtol=0.0)
    np.testing.assert_allclose(inferred_variance, posterior_variance, atol=0.005, rtol=0.0)

    inferred_total, inferred_complexity, inferred_nll = _analytic_diagonal_gaussian_vfe(
        inferred_mean,
        inferred_variance,
        prior_mean,
        prior_variance,
        observation,
        observation_variance,
    )
    optimum_total, _, _ = _analytic_diagonal_gaussian_vfe(
        posterior_mean,
        posterior_variance,
        prior_mean,
        prior_variance,
        observation,
        observation_variance,
    )
    assert estimator.last_vfe is not None
    assert estimator.last_vfe.complexity == pytest.approx(inferred_complexity, abs=1e-10)
    assert estimator.last_vfe.negative_log_likelihood == pytest.approx(inferred_nll, abs=0.06)
    assert estimator.last_vfe.total == pytest.approx(inferred_total, abs=0.06)
    assert inferred_total - optimum_total < 0.03

    # Independent perturbations of either sufficient statistic raise F. This
    # pins the analytic posterior as the VFE minimizer rather than merely a
    # plausible point approached by the optimizer.
    competitors = (
        (posterior_mean + 0.05, posterior_variance),
        (posterior_mean - 0.05, posterior_variance),
        (posterior_mean, 0.5 * posterior_variance),
        (posterior_mean, 2.0 * posterior_variance),
    )
    for candidate_mean, candidate_variance in competitors:
        candidate_total, _, _ = _analytic_diagonal_gaussian_vfe(
            candidate_mean,
            candidate_variance,
            prior_mean,
            prior_variance,
            observation,
            observation_variance,
        )
        assert candidate_total > optimum_total


@pytest.mark.parametrize(
    ("transition_variance", "observation_variance"),
    [
        (1e-2, 1e-6),
        (1e-4, 1e-4),
        (1e-6, 1e-2),
    ],
)
def test_spatial_posterior_tracks_precision_across_four_orders_of_magnitude(
    transition_variance: float,
    observation_variance: float,
) -> None:
    """The exact spatial posterior follows likelihood/transition precision."""

    config = PainterConfig(
        canvas_size=4,
        spatial_grid_size=4,
        material_pyramid_levels=(4,),
        base_observation_std=math.sqrt(observation_variance),
        smear_observation_std=0.0,
    )
    prior_values = np.asarray([0.20, 0.10, 0.08, 0.30, 0.0, 1.0], dtype=np.float32)
    observed_values = np.asarray([0.80, 0.60, 0.40, 0.70, 0.0, 1.0], dtype=np.float32)
    prior_material = np.broadcast_to(prior_values[:, None, None], (6, 4, 4)).copy()
    observed_material = np.broadcast_to(observed_values[:, None, None], (6, 4, 4)).copy()
    carried_variance = 1e-12
    previous = SpatialCanvasState(
        prior_material,
        np.full_like(prior_material, math.log(carried_variance)),
    )
    observed = SpatialCanvasState(
        observed_material,
        np.full_like(observed_material, math.log(observation_variance)),
    )

    estimator = SpatialVariationalStateEstimator(config, torch.device("cpu"))
    posterior = estimator.infer(
        previous,
        StrokeAction.stop_action(),
        observed,
        _ControlledVarianceSpatialDynamics(transition_variance),
    )
    actual_mean = pixel_material_from_state(posterior)[:4]
    actual_variance = np.exp(pixel_logvar_from_state(posterior, config)[:4])
    total_transition_variance = transition_variance + carried_variance
    expected_variance = 1.0 / (
        1.0 / total_transition_variance + 1.0 / observation_variance
    )
    expected_mean = expected_variance * (
        prior_material[:4] / total_transition_variance
        + observed_material[:4] / observation_variance
    )

    np.testing.assert_allclose(actual_mean, expected_mean, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(actual_variance, expected_variance, atol=1e-9, rtol=1e-5)


def test_local_transition_prior_is_identity_outside_the_stroke_patch() -> None:
    """Outside local support, Bayesian fusion uses the declared identity prior."""

    config = PainterConfig(
        canvas_size=16,
        spatial_grid_size=16,
        material_pyramid_levels=(16,),
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        local_identity_logvar=-10.0,
    )
    previous, observation = _spatial_belief_pair(config, seed=11)
    action = StrokeAction(0.45, 0.45, 0.55, 0.55, 0.03, 0.6, 1.0)
    raster = rasterize_stroke_action(action, 16, config=config)
    bounds = local_patch_bounds_for_raster(raster, 16, config)
    assert bounds is not None
    assert not (bounds.row0 <= 0 < bounds.row1 and bounds.col0 <= 0 < bounds.col1)

    estimator = SpatialVariationalStateEstimator(config, torch.device("cpu"))
    posterior = estimator.infer(
        previous,
        action,
        observation,
        LocalSpatialDynamicsEnsemble(config),
    )
    current_mean = pixel_material_from_state(previous)[:4, 0, 0]
    current_variance = np.exp(pixel_logvar_from_state(previous, config)[:4, 0, 0])
    observed_mean = pixel_material_from_state(observation)[:4, 0, 0]
    observed_variance = spatial_observation_variance(
        pixel_material_from_state(observation), config
    )[:4, 0, 0]
    identity_variance = math.exp(config.local_identity_logvar)
    transition_variance = current_variance + identity_variance
    expected_variance = 1.0 / (1.0 / transition_variance + 1.0 / observed_variance)
    expected_mean = expected_variance * (
        current_mean / transition_variance + observed_mean / observed_variance
    )

    np.testing.assert_allclose(
        pixel_material_from_state(posterior)[:4, 0, 0], expected_mean, atol=2e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        np.exp(pixel_logvar_from_state(posterior, config)[:4, 0, 0]),
        expected_variance,
        atol=1e-9,
        rtol=1e-5,
    )


def test_spatial_vfe_matches_independent_fine_grid_integration() -> None:
    """Identity: F = KL(q||p) - E_q[log p(o|s)] per independent cell-channel.

    ``SpatialVariationalStateEstimator`` fuses a Gaussian transition prior with
    a Gaussian observation likelihood in closed form, so its VFE is exact and
    can be checked to integration precision. Units are
    ``nats_per_independent_cell_channel`` -- MEANS over cells and channels, not
    sums -- and only the ``independent_material_channel_count`` primary factors
    contribute, because ground contrast and coverage are deterministic
    functions of the primary channels and must carry no separate likelihood
    evidence. This is the AI-104 gap.
    """

    config = PainterConfig(canvas_size=16, spatial_grid_size=8)
    previous, observation = _spatial_belief_pair(config)
    action = StrokeAction(0.2, 0.3, 0.7, 0.6, 0.08, 0.6, 1.0)
    dynamics = _DenseSpatialStubDynamics()

    estimator = SpatialVariationalStateEstimator(config, torch.device("cpu"))
    estimator.infer(previous, action, observation, dynamics)
    reported = estimator.last_vfe
    assert reported is not None
    assert reported.units == "nats_per_independent_cell_channel"

    # Independent replication of the dense (non-local) branch, then fine-grid
    # integration cell-channel by cell-channel. Nothing here calls back into
    # spatial_inference except the observation-variance likelihood definition.
    current_mean = pixel_material_from_state(previous)
    current_variance = np.exp(
        np.clip(pixel_logvar_from_state(previous, config), -30.0, 20.0)
    ).astype(np.float32)
    raster = rasterize_stroke_action(action, current_mean.shape[-1], config=config)
    mean_t, aleatoric_t, epistemic_t = dynamics.predictive_moments(
        torch.tensor(current_mean, dtype=torch.float32).unsqueeze(0),
        torch.tensor(raster, dtype=torch.float32).unsqueeze(0),
    )
    prior_mean = mean_t[0].numpy()
    prior_variance = np.clip(
        (aleatoric_t[0] + epistemic_t[0]).numpy() + current_variance, 1e-12, 1e6
    )
    observed = pixel_material_from_state(observation)
    observation_variance = spatial_observation_variance(observed, config)
    independent = slice(0, independent_material_channel_count(observed.shape[0]))

    posterior_variance = 1.0 / (
        1.0 / prior_variance[independent] + 1.0 / observation_variance[independent]
    )
    posterior_mean = posterior_variance * (
        prior_mean[independent] / prior_variance[independent]
        + observed[independent] / observation_variance[independent]
    )

    kl_values = np.zeros(posterior_mean.shape, dtype=np.float64)
    nll_values = np.zeros(posterior_mean.shape, dtype=np.float64)
    for index in np.ndindex(posterior_mean.shape):
        _, kl, negative_log_likelihood = _fine_grid_vfe(
            float(posterior_mean[index]),
            float(posterior_variance[index]),
            float(prior_mean[independent][index]),
            float(prior_variance[independent][index]),
            float(observed[independent][index]),
            float(observation_variance[independent][index]),
            points=20001,
            span=10.0,
        )
        kl_values[index] = kl
        nll_values[index] = negative_log_likelihood

    grid_complexity = float(np.mean(kl_values))
    grid_negative_log_likelihood = float(np.mean(nll_values))
    # Measured deviation: 1.7e-9 complexity, 3.0e-9 NLL, 4.6e-9 total.
    assert reported.complexity == pytest.approx(grid_complexity, abs=1e-6)
    assert reported.negative_log_likelihood == pytest.approx(
        grid_negative_log_likelihood, abs=1e-6
    )
    assert reported.total == pytest.approx(
        grid_complexity + grid_negative_log_likelihood, abs=1e-6
    )


# ---------------------------------------------------------------------------
# (e) motor EFE consistency
# ---------------------------------------------------------------------------


_MOTOR_ACTION = StrokeAction(0.3, 0.4, 0.7, 0.6, 0.08, 0.6, 1.0)


def _motor_forecast(config: PainterConfig):
    """One representative embodied execution forecast.

    The summary-state callable returns zeros: this test is about proprioceptive
    EFE arithmetic, so the canvas consequence of the stroke is irrelevant and is
    held at a constant (mirrors ``tests/test_motor_reliability.py``).
    """

    def summary(_sim: ArmPainterSim) -> np.ndarray:
        return np.zeros(config.state_dim, dtype=np.float32)

    return forecast_stroke_execution(
        ArmPainterSim(config), _MOTOR_ACTION, summary, rollout_samples=2
    )


def _proprioceptive_moments(forecast, *, likelihood_floor: float = 1e-8):
    """Predictive, likelihood, and outcome variances exactly as motor EFE floors them.

    Only public forecast arrays are used, and ``reliability_inflation`` is left
    at 1.0 everywhere it matters so the per-channel inflation mask does not have
    to be duplicated here. These are the raw ingredients of the information
    identities, independent of any preference scale.
    """

    predictive = np.maximum(np.asarray(forecast.proprioceptive_predictive_variance, dtype=np.float64), 0.0)
    likelihood = np.maximum(
        np.asarray(forecast.proprioceptive_likelihood_variance, dtype=np.float64), likelihood_floor
    )
    return predictive, likelihood, predictive + likelihood


def test_motor_efe_terms_split_the_conflated_epistemic_term() -> None:
    """Identity: -I(s;o) = E_q(s)[H[p(o|s)]] - H[q(o)], hence pragmatic - I == KL risk + ambiguity.

    ``MotorEFETerms.risk`` is the pragmatic term E_q[-log p*(o)] with
    policy-independent Gaussian preference normalizers omitted -- NOT a KL. The
    logged ``ambiguity`` is likelihood entropy measured against the *preference*
    scale, a fourth incommensurate quantity that must be observable but must
    never be summed into G. This test pins the split of the previously
    conflated ``epistemic_value`` into state/observation mutual information
    (already implied by pragmatic - H[q(o)] + E[H[p(o|s)]]) and a genuine
    parameter-novelty term over the learned motion-reliability belief.
    """

    config = PainterConfig(
        canvas_size=48,
        motor_proprioceptive_risk_precision=1.0,
        motor_proprioceptive_ambiguity_precision=1.0,
        motor_reliability_novelty_precision=1.0,
    )
    forecast = _motor_forecast(config)
    predictive, likelihood, outcome = _proprioceptive_moments(forecast)

    mutual_information = 0.5 * float(np.sum(np.log1p(predictive / likelihood)))
    outcome_entropy = 0.5 * float(np.sum(np.log(2.0 * np.pi * np.e * outcome)))
    conditional_entropy = 0.5 * float(np.sum(np.log(2.0 * np.pi * np.e * likelihood)))
    assert outcome_entropy - conditional_entropy == pytest.approx(mutual_information, abs=1e-9)

    terms = motor_efe_terms(forecast, config)
    assert terms.mutual_information == pytest.approx(mutual_information, rel=1e-9)
    assert terms.forecast_outcome_entropy == pytest.approx(outcome_entropy, rel=1e-12)
    # The default call supplies no reliability evidence, so parameter novelty is
    # exactly zero and the logged sum reduces to the mutual information alone.
    assert terms.reliability_novelty == pytest.approx(0.0)
    assert terms.epistemic_value == pytest.approx(terms.mutual_information + terms.reliability_novelty)

    curious = motor_efe_terms(forecast, config, reliability_epistemic_nats=0.3)
    assert curious.mutual_information == pytest.approx(terms.mutual_information)
    assert curious.reliability_novelty == pytest.approx(0.3)
    assert curious.epistemic_value == pytest.approx(
        curious.mutual_information + curious.reliability_novelty
    )

    # Bookkeeping identity at unit precisions: subtracting the mutual
    # information once is algebraically identical to the canonical KL risk plus
    # ambiguity, so adding a separate ambiguity term would double count.
    pragmatic = terms.risk
    assert pragmatic - mutual_information == pytest.approx(
        (pragmatic - outcome_entropy) + conditional_entropy, rel=1e-12
    )


def test_motor_contribution_to_g_is_pragmatic_minus_information_gain() -> None:
    """Identity: G_motor = pragmatic - I(s;o) - N_reliability, at every EFE total site.

    The logged ``MotorEFETerms.ambiguity`` is deliberately absent from that sum.
    All six EFE total sites (summary mixture, summary ensemble, spatial local
    mixture, spatial local ensemble single and batch, spatial dense mixture and
    ensemble) are exercised so no site can silently keep the old arithmetic.
    """

    unit_config = PainterConfig(
        canvas_size=48,
        motor_proprioceptive_risk_precision=1.0,
        motor_proprioceptive_ambiguity_precision=1.0,
        motor_reliability_novelty_precision=1.0,
    )
    unit_forecast = _motor_forecast(unit_config)
    predictive, likelihood, _ = _proprioceptive_moments(unit_forecast)
    mutual_information = 0.5 * float(np.sum(np.log1p(predictive / likelihood)))
    unit_terms = motor_efe_terms(unit_forecast, unit_config, reliability_epistemic_nats=0.28)
    assert motor_efe_contribution(unit_terms.risk, unit_terms.epistemic_value) == pytest.approx(
        unit_terms.risk - mutual_information - 0.28, rel=1e-12
    )

    # Production precisions: the declared per-term precisions differ, so the
    # identity is stated on the precision-weighted terms.
    config = PainterConfig(canvas_size=48)
    forecast = _motor_forecast(config)
    terms = motor_efe_terms(
        forecast, config, reliability_inflation=1.6, reliability_epistemic_nats=0.28
    )
    contribution = motor_efe_contribution(terms.risk, terms.epistemic_value)
    assert contribution == pytest.approx(
        terms.risk - terms.mutual_information - terms.reliability_novelty, rel=1e-12
    )
    assert terms.ambiguity > 0.0
    # The old arithmetic added `+ ambiguity` on top of `- mutual_information`,
    # which double counted ambiguity by exactly the scaled excess entropy.
    legacy_contribution = terms.risk + terms.ambiguity - terms.epistemic_value
    assert legacy_contribution - contribution == pytest.approx(terms.ambiguity, rel=1e-12)

    for site, (base, loaded) in _motor_total_sites(terms).items():
        assert base.execution_forecast_used, site
        assert loaded.execution_forecast_used, site
        assert loaded.motor_ambiguity == pytest.approx(terms.ambiguity), site
        assert loaded.total - base.total == pytest.approx(contribution, abs=2e-4), site
        assert loaded.total - base.total != pytest.approx(legacy_contribution, abs=1e-5), site


class _PatchStubDynamics:
    """Deterministic local-patch transition stub (not an ensemble)."""

    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deposited = 0.01 * action_raster[:, 4:5] * action_raster[:, 0:1]
        delta = torch.zeros_like(material)
        delta[:, 0:1] = deposited
        delta[:, 1:2] = 0.5 * deposited
        delta[:, 2:3] = deposited
        next_material = project_material_support(material, material + delta, 0.005, 0.34)
        return next_material, torch.full_like(next_material, 1e-6), torch.zeros_like(next_material)


def _summary_motor_pair(dynamics, terms):
    config = PainterConfig()
    efe = ExpectedFreeEnergy(
        config, dynamics, ObservationModel(config), TerminalCoveragePreference(config)
    )
    belief = GaussianBelief(
        torch.tensor([0.68, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -14.0),
    )
    policy = Policy((StrokeAction(0.2, 0.2, 0.8, 0.8, 0.08, 0.7, 1.0), StrokeAction.stop_action()))
    next_mean = torch.tensor([0.87, 0.1, 0.1, 0.0, 0.0, 0.0])
    next_variance = torch.full((6,), 1e-5)
    shared = dict(
        execution_uncertainty=0.01,
        contact_loss_probability=0.0,
        motor_overshoot=0.01,
        motor_feasible=True,
    )
    base = efe.evaluate_with_first_transition(belief, policy, next_mean, next_variance, **shared)
    loaded = efe.evaluate_with_first_transition(
        belief,
        policy,
        next_mean,
        next_variance,
        motor_risk=terms.risk,
        motor_ambiguity=terms.ambiguity,
        motor_epistemic_value=terms.epistemic_value,
        motor_efe_approximation=terms.approximation,
        **shared,
    )
    return base, loaded


def _local_spatial_fixture(dynamics):
    config = PainterConfig(
        canvas_size=32,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        local_patch_batch_bucket_cells=32,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        spatial_ensemble_size=2,
        composition_gap_precision=0.0,
    )
    belief = spatial_canvas_state(ArmPainterSim(config), config)
    material = torch.tensor(max(belief.pyramid, key=lambda level: level.grid_size).material)
    raster = torch.tensor(rasterize_stroke_action(_MOTOR_ACTION, config.canvas_size, config=config))
    support = raster[0] > 0.0
    next_material = material.clone()
    next_material[0, support] += 0.01
    next_material[1, support] += 0.005
    next_material[2, support] += 0.006
    next_material = project_material_support(
        material.unsqueeze(0),
        next_material.unsqueeze(0),
        config.thickness_scale,
        config.canvas_ground_tone,
    )[0]
    delta = torch.zeros_like(next_material)
    delta[:, support] = next_material[:, support] - material[:, support]
    policy = Policy((_MOTOR_ACTION, StrokeAction.stop_action()))
    efe = SpatialExpectedFreeEnergy(config, dynamics, TerminalCoveragePreference(config))
    return efe, belief, policy, next_material, torch.full_like(next_material, 1e-5), delta


def _dense_spatial_fixture(dynamics):
    config = PainterConfig(
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        composition_gap_precision=0.0,
    )
    previous, _ = _spatial_belief_pair(config)
    # A plain SpatialCanvasState carries no material pyramid, so the evaluator
    # takes the dense-grid rollout path regardless of spatial_transition_mode.
    assert not previous.pyramid
    material = torch.tensor(previous.material)
    next_material = material.clone()
    next_material[0] += 0.01
    next_material = project_material_support(
        material.unsqueeze(0),
        next_material.unsqueeze(0),
        config.thickness_scale,
        config.canvas_ground_tone,
    )[0]
    policy = Policy((_MOTOR_ACTION, StrokeAction.stop_action()))
    efe = SpatialExpectedFreeEnergy(config, dynamics, TerminalCoveragePreference(config))
    return efe, previous, policy, next_material, torch.full_like(next_material, 1e-5), None


def _spatial_motor_pair(fixture, terms):
    efe, belief, policy, next_material, next_variance, delta = fixture
    shared = dict(
        next_material_delta=delta,
        execution_uncertainty=0.2,
        contact_loss_probability=0.1,
        motor_overshoot=0.05,
        motor_feasible=True,
    )
    base = efe.evaluate_with_first_transition(belief, policy, next_material, next_variance, **shared)
    loaded = efe.evaluate_with_first_transition(
        belief,
        policy,
        next_material,
        next_variance,
        motor_risk=terms.risk,
        motor_ambiguity=terms.ambiguity,
        motor_epistemic_value=terms.epistemic_value,
        motor_efe_approximation=terms.approximation,
        **shared,
    )
    return base, loaded


def _spatial_batch_motor_pair(fixture, terms):
    efe, belief, policy, next_material, next_variance, delta = fixture
    transition = (
        (next_material, next_variance) if delta is None else (next_material, next_variance, delta)
    )

    def evaluate(risk: float, ambiguity: float, epistemic: float):
        return efe.evaluate_batch_with_first_transitions(
            belief,
            [policy],
            [transition],
            execution_uncertainties=[0.2],
            contact_loss_probabilities=[0.1],
            motor_overshoots=[0.05],
            motor_feasibilities=[True],
            motor_risks=[risk],
            motor_ambiguities=[ambiguity],
            motor_epistemic_values=[epistemic],
            motor_efe_approximations=[terms.approximation],
        )[0]

    return evaluate(0.0, 0.0, 0.0), evaluate(terms.risk, terms.ambiguity, terms.epistemic_value)


def _motor_total_sites(terms) -> dict[str, tuple[object, object]]:
    """One (base, motor-loaded) component pair per EFE total site."""

    summary_config = PainterConfig()
    torch.manual_seed(0)
    summary_ensemble = DynamicsEnsemble(summary_config)
    local_ensemble = LocalSpatialDynamicsEnsemble(
        PainterConfig(
            canvas_size=32,
            spatial_grid_size=8,
            spatial_transition_mode="local_patch",
            local_patch_batch_bucket_cells=32,
            local_patch_margin_cells=1,
            local_patch_min_cells=4,
            spatial_ensemble_size=2,
        )
    )
    dense_config = PainterConfig(
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
    )
    dense_ensemble = SpatialDynamicsEnsemble(dense_config)

    local_ensemble_fixture = _local_spatial_fixture(local_ensemble)
    return {
        "efe._evaluate_mixture": _summary_motor_pair(_DeterministicSummaryDynamics(), terms),
        "efe._evaluate_ensemble_batch": _summary_motor_pair(summary_ensemble, terms),
        "spatial_efe._evaluate_local_mixture": _spatial_motor_pair(
            _local_spatial_fixture(_PatchStubDynamics()), terms
        ),
        "spatial_efe._evaluate_local_ensemble_batch": _spatial_motor_pair(
            local_ensemble_fixture, terms
        ),
        "spatial_efe._evaluate_local_ensemble_batch (batched)": _spatial_batch_motor_pair(
            local_ensemble_fixture, terms
        ),
        "spatial_efe._evaluate_mixture": _spatial_motor_pair(
            _dense_spatial_fixture(_DenseSpatialStubDynamics()), terms
        ),
        "spatial_efe._evaluate_ensemble_batch": _spatial_motor_pair(
            _dense_spatial_fixture(dense_ensemble), terms
        ),
    }
