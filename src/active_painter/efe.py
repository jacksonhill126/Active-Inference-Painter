from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .action_encoding import encoded_action_vector
from .config import PainterConfig
from .efe_common import project_summary_transition_support, terminal_preference_terms
from .inference import normal_entropy_from_variance
from .models import DynamicsEnsemble, GaussianBelief, ObservationModel
from .motor_planning import motor_efe_contribution
from .policies import Policy
from .precision_beliefs import (
    MODALITY_NAMES,
    ModalityWeights,
    PrecisionLedger,
    constant_modality_weights,
)
from .preferences import TerminalCoveragePreference


@dataclass(slots=True)
class EFEComponents:
    total: float
    terminal_risk: float
    ambiguity: float
    epistemic_value: float
    terminal_coverage_mean: float
    terminal_coverage_std: float
    terminal_entropy: float = 0.0
    pragmatic_value: float = 0.0
    transition_risk: float = 0.0
    transition_ambiguity: float = 0.0
    execution_uncertainty: float = 0.0
    contact_loss_probability: float = 0.0
    motor_overshoot: float = 0.0
    motor_risk: float = 0.0
    motor_ambiguity: float = 0.0
    motor_epistemic_value: float = 0.0
    motor_efe_approximation: str = ""
    motor_feasible: bool = True
    execution_forecast_used: bool = False
    # Telemetry mirrors of the two components of motor_epistemic_value. Read by
    # the viewer only; no arithmetic anywhere consumes them.
    motor_mutual_information: float = 0.0
    motor_reliability_novelty: float = 0.0
    # --- declared units and per-modality precision weights ----------------
    # Modality fields stay POST-weighted, so `total` remains the exact sum of
    # its summands; the recorded precision and normalizer make the raw term
    # recoverable. `approximation` satisfies the charter's per-EFE-dataclass
    # approximation rule, which this container previously did not carry at all.
    modality_units: str = "nats_per_observation_channel"
    approximation: str = ""
    terminal_coverage_precision: float = 0.0
    terminal_coverage_normalizer: float = 1.0
    terminal_coverage_normalizer_name: str = ""
    observation_ambiguity_precision: float = 0.0
    observation_ambiguity_normalizer: float = 1.0
    observation_ambiguity_normalizer_name: str = ""
    transition_precision: float = 0.0
    transition_normalizer: float = 1.0
    transition_normalizer_name: str = ""
    motor_proprioceptive_precision: float = 0.0
    motor_proprioceptive_normalizer: float = 1.0
    motor_proprioceptive_normalizer_name: str = ""


class ExpectedFreeEnergy:
    """Expected-free-energy evaluator for terminal-coverage policies.

    This uses the risk-plus-ambiguity decomposition. Epistemic value is logged
    only as the information-gain identity implied by transition risk plus
    transition ambiguity, not as an extra term added to expected free energy.

    With a learned ensemble, policy rollouts are member-wise trajectory
    samples: each ensemble member propagates its own state particle, so
    parameter uncertainty accumulates over the policy horizon instead of being
    collapsed to a moment-matched mixture after every step. Terminal coverage
    variance combines across-member disagreement with mean within-member
    predictive variance.

    Logged components are BELIEF-weighted: `gamma_m * normalizer_m * raw term`,
    with `gamma_m` the posterior mean of a declared Gamma precision belief. With
    no injected ledger the weights are exactly the hand-declared config
    constants, so the hand-written mechanism stays available for attribution.
    """

    def __init__(
        self,
        config: PainterConfig,
        dynamics: DynamicsEnsemble,
        observation_model: ObservationModel,
        terminal_preference: TerminalCoveragePreference,
        precision_ledger: PrecisionLedger | None = None,
    ) -> None:
        self.cfg = config
        self.dynamics = dynamics
        self.observation_model = observation_model
        self.preference = terminal_preference
        self.precision_ledger = precision_ledger

    def _modality_weights(self) -> ModalityWeights:
        """Precision-belief means and declared normalizers for one evaluation.

        Built once per evaluate call and shared by both rollout paths, so the
        ensemble and mixture evaluators cannot drift apart.
        """

        if self.precision_ledger is None:
            return constant_modality_weights(self.cfg)
        return self.precision_ledger.weights()

    def _component_units(self, weights: ModalityWeights) -> dict[str, float | str]:
        """Declared units and per-modality weights for the component dataclass."""

        payload: dict[str, float | str] = {
            "modality_units": (
                "nats_per_observation_channel"
                if weights.normalization_enabled
                else "declared_mixed_units_pre_normalization"
            ),
            "approximation": (
                "modality contributions are gamma_m * normalizer_m * raw term, with gamma_m the "
                "posterior mean of a declared Gamma precision belief clamped to its declared "
                "bounded support; the terminal coverage forecast is a moment-matched Beta "
                "restricted to the interior-unimodal family (concentration floor "
                f"{weights.concentration_floor:g}); the six-aggregate summary state is a declared "
                "compatibility fixture, not a painting belief"
            ),
        }
        for name in MODALITY_NAMES:
            if name not in {
                "terminal_coverage",
                "observation_ambiguity",
                "transition",
                "motor_proprioceptive",
            }:
                continue
            payload[f"{name}_precision"] = float(weights.gamma.get(name, 0.0))
            payload[f"{name}_normalizer"] = float(weights.normalizer.get(name, 1.0))
            payload[f"{name}_normalizer_name"] = weights.normalizer_name.get(name, "")
        return payload

    @staticmethod
    def _assemble_total(
        *,
        terminal_risk,
        ambiguity,
        transition_risk,
        transition_ambiguity,
        motor_risk,
        motor_epistemic_value,
    ):
        """Single source of truth for the summary EFE total.

        `epistemic_value` is deliberately absent (transition_risk +
        transition_ambiguity already equals -I(theta; s_next)), and so is the
        logged motor ambiguity (-I(s; o) already carries it).
        """

        return (
            terminal_risk
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + motor_efe_contribution(motor_risk, motor_epistemic_value)
        )

    @torch.no_grad()
    def evaluate(self, belief: GaussianBelief, policy: Policy) -> EFEComponents:
        return self._evaluate(belief, policy)

    @torch.no_grad()
    def evaluate_batch(self, belief: GaussianBelief, policies: Sequence[Policy]) -> list[EFEComponents]:
        policies = list(policies)
        if not policies:
            return []
        if isinstance(self.dynamics, DynamicsEnsemble):
            return self._evaluate_ensemble_batch(belief, policies)
        return [self._evaluate_mixture(belief, policy) for policy in policies]

    @torch.no_grad()
    def evaluate_with_first_transition(
        self,
        belief: GaussianBelief,
        policy: Policy,
        next_state_mean: torch.Tensor,
        next_state_variance: torch.Tensor,
        *,
        execution_uncertainty: float,
        contact_loss_probability: float,
        motor_overshoot: float,
        motor_feasible: bool,
        motor_risk: float = 0.0,
        motor_ambiguity: float = 0.0,
        motor_epistemic_value: float = 0.0,
        motor_efe_approximation: str = "",
    ) -> EFEComponents:
        return self._evaluate(
            belief,
            policy,
            first_transition=(next_state_mean, next_state_variance),
            execution_uncertainty=execution_uncertainty,
            contact_loss_probability=contact_loss_probability,
            motor_overshoot=motor_overshoot,
            motor_feasible=motor_feasible,
            motor_risk=motor_risk,
            motor_ambiguity=motor_ambiguity,
            motor_epistemic_value=motor_epistemic_value,
            motor_efe_approximation=motor_efe_approximation,
        )

    def _evaluate(
        self,
        belief: GaussianBelief,
        policy: Policy,
        first_transition: tuple[torch.Tensor, torch.Tensor] | None = None,
        execution_uncertainty: float = 0.0,
        contact_loss_probability: float = 0.0,
        motor_overshoot: float = 0.0,
        motor_feasible: bool = True,
        motor_risk: float = 0.0,
        motor_ambiguity: float = 0.0,
        motor_epistemic_value: float = 0.0,
        motor_efe_approximation: str = "",
    ) -> EFEComponents:
        if isinstance(self.dynamics, DynamicsEnsemble):
            return self._evaluate_ensemble_batch(
                belief,
                [policy],
                first_transition=first_transition,
                execution_uncertainty=execution_uncertainty,
                contact_loss_probability=contact_loss_probability,
                motor_overshoot=motor_overshoot,
                motor_feasible=motor_feasible,
                motor_risk=motor_risk,
                motor_ambiguity=motor_ambiguity,
                motor_epistemic_value=motor_epistemic_value,
                motor_efe_approximation=motor_efe_approximation,
            )[0]
        return self._evaluate_mixture(
            belief,
            policy,
            first_transition=first_transition,
            execution_uncertainty=execution_uncertainty,
            contact_loss_probability=contact_loss_probability,
            motor_overshoot=motor_overshoot,
            motor_feasible=motor_feasible,
            motor_risk=motor_risk,
            motor_ambiguity=motor_ambiguity,
            motor_epistemic_value=motor_epistemic_value,
            motor_efe_approximation=motor_efe_approximation,
        )

    def _evaluate_ensemble_batch(
        self,
        belief: GaussianBelief,
        policies: list[Policy],
        first_transition: tuple[torch.Tensor, torch.Tensor] | None = None,
        execution_uncertainty: float = 0.0,
        contact_loss_probability: float = 0.0,
        motor_overshoot: float = 0.0,
        motor_feasible: bool = True,
        motor_risk: float = 0.0,
        motor_ambiguity: float = 0.0,
        motor_epistemic_value: float = 0.0,
        motor_efe_approximation: str = "",
    ) -> list[EFEComponents]:
        if first_transition is not None and len(policies) != 1:
            raise ValueError("An execution-forecast first transition applies to a single policy.")
        # One precision-belief/normalizer set for the whole candidate set.
        weights = self._modality_weights()
        device = belief.mean.device
        member_count = len(self.dynamics.members)
        policy_count = len(policies)
        state_dim = int(belief.mean.shape[-1])

        member_states = belief.mean.view(1, 1, state_dim).expand(policy_count, member_count, state_dim).clone()
        member_within = belief.logvar.exp().view(1, 1, state_dim).expand(policy_count, member_count, state_dim).clone()
        ambiguity = torch.zeros(policy_count, device=device)
        transition_risk = torch.zeros(policy_count, device=device)
        transition_ambiguity = torch.zeros(policy_count, device=device)
        epistemic_value = torch.zeros(policy_count, device=device)

        depths = [len(policy.actions) - 1 for policy in policies]
        first_transition_used = False
        for step in range(max(depths, default=0)):
            active = [index for index in range(policy_count) if depths[index] > step]
            if not active:
                break
            active_t = torch.tensor(active, dtype=torch.long, device=device)
            if first_transition is not None and step == 0:
                next_mean, next_variance = first_transition
                next_mean = next_mean.to(device, dtype=member_states.dtype).reshape(1, 1, state_dim)
                next_variance = next_variance.to(device, dtype=member_states.dtype).reshape(1, 1, state_dim)
                projected = project_summary_transition_support(
                    member_states[active_t],
                    next_mean.expand(len(active), member_count, state_dim),
                )
                member_states[active_t] = projected
                member_within[active_t] = torch.clamp(
                    next_variance.expand(len(active), member_count, state_dim),
                    min=1e-8,
                )
                first_transition_used = True
            else:
                actions = np.stack(
                    [
                        encoded_action_vector(
                            policies[index].actions[step],
                            self.cfg,
                            policies[index].motor_primitive if step == 0 else None,
                        )
                        for index in active
                    ]
                )
                action_batch = torch.from_numpy(actions).to(device=device, dtype=member_states.dtype)
                flat_states = member_states[active_t].reshape(len(active) * member_count, state_dim)
                flat_actions = action_batch.repeat_interleave(member_count, dim=0)
                all_means, all_logvars = self.dynamics(flat_states, flat_actions)
                rows = torch.arange(flat_states.shape[0], device=device)
                member_index = rows % member_count
                selected_means = all_means[member_index, rows].reshape(len(active), member_count, state_dim)
                selected_within = all_logvars[member_index, rows].exp().reshape(len(active), member_count, state_dim)

                aleatoric = selected_within.mean(dim=1)
                epistemic_variance = selected_means.var(dim=1, unbiased=False)

                # Approximation: for the transition-outcome modality, prior
                # preferences are flat up to an omitted constant. The transition
                # risk is therefore -H[q(s[t+1] | s[t], a[t])] with a
                # moment-matched diagonal Gaussian mixture marginal, and the
                # transition ambiguity is E_q(theta) H[p_theta(s[t+1] | s, a)].
                marginal_entropy = normal_entropy_from_variance(aleatoric + epistemic_variance)
                conditional_entropy = normal_entropy_from_variance(selected_within).mean(dim=1)
                transition_risk[active_t] = transition_risk[active_t] - marginal_entropy
                transition_ambiguity[active_t] = transition_ambiguity[active_t] + conditional_entropy
                epistemic_value[active_t] = epistemic_value[active_t] + torch.clamp(
                    marginal_entropy - conditional_entropy, min=0.0
                )

                projected = project_summary_transition_support(member_states[active_t], selected_means)
                member_states[active_t] = projected
                member_within[active_t] = torch.clamp(selected_within, min=1e-8)

            # Observation ambiguity is integrated over parameter uncertainty by
            # averaging across member particles.
            ambiguity[active_t] = ambiguity[active_t] + self.observation_model.ambiguity(
                member_states[active_t]
            ).mean(dim=1)

        coverage_member = member_states[..., 0]
        coverage_mean = torch.clamp(coverage_member.mean(dim=1), 1e-4, 1.0 - 1e-4)
        coverage_variance = torch.clamp(
            coverage_member.var(dim=1, unbiased=False) + member_within[..., 0].mean(dim=1),
            min=1e-8,
        )
        coverage_std = torch.sqrt(coverage_variance)
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=weights.terminal,
            concentration_floor=weights.concentration_floor,
        )

        # Belief-weighted modality contributions. `epistemic_value` carries the
        # SAME transition weight as transition_risk/transition_ambiguity because
        # R_trans + A_trans == -I(theta; s_next); it is logged, never summed.
        ambiguity = weights.ambiguity * ambiguity
        transition_risk = weights.transition * transition_risk
        transition_ambiguity = weights.transition * transition_ambiguity
        epistemic_value = weights.transition * epistemic_value
        motor_risk_t = torch.full((policy_count,), float(motor_risk if first_transition_used else 0.0), device=device)
        motor_ambiguity_t = torch.full((policy_count,), float(motor_ambiguity if first_transition_used else 0.0), device=device)
        motor_epistemic_t = torch.full(
            (policy_count,), float(motor_epistemic_value if first_transition_used else 0.0), device=device
        )
        # The motor modality enters G as pragmatic - I(s;o) - N_reliability.
        # motor_ambiguity_t stays logged but is deliberately not a summand:
        # -I(s;o) already carries the canonical ambiguity contribution.
        total = self._assemble_total(
            terminal_risk=terminal_risk,
            ambiguity=ambiguity,
            transition_risk=transition_risk,
            transition_ambiguity=transition_ambiguity,
            motor_risk=motor_risk_t,
            motor_epistemic_value=motor_epistemic_t,
        )

        return [
            EFEComponents(
                total=float(total[index].item()),
                terminal_risk=float(terminal_risk[index].item()),
                ambiguity=float(ambiguity[index].item()),
                epistemic_value=float(epistemic_value[index].item()),
                terminal_coverage_mean=float(coverage_mean[index].item()),
                terminal_coverage_std=float(coverage_std[index].item()),
                terminal_entropy=float(terminal_entropy[index].item()),
                pragmatic_value=float(pragmatic_value[index].item()),
                transition_risk=float(transition_risk[index].item()),
                transition_ambiguity=float(transition_ambiguity[index].item()),
                execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
                contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
                motor_overshoot=motor_overshoot if first_transition_used else 0.0,
                motor_risk=float(motor_risk_t[index].item()),
                motor_ambiguity=float(motor_ambiguity_t[index].item()),
                motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
                motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
                motor_feasible=motor_feasible,
                execution_forecast_used=first_transition_used,
                **self._component_units(weights),
            )
            for index in range(policy_count)
        ]

    def _evaluate_mixture(
        self,
        belief: GaussianBelief,
        policy: Policy,
        first_transition: tuple[torch.Tensor, torch.Tensor] | None = None,
        execution_uncertainty: float = 0.0,
        contact_loss_probability: float = 0.0,
        motor_overshoot: float = 0.0,
        motor_feasible: bool = True,
        motor_risk: float = 0.0,
        motor_ambiguity: float = 0.0,
        motor_epistemic_value: float = 0.0,
        motor_efe_approximation: str = "",
    ) -> EFEComponents:
        """Moment-matched mixture rollout for dynamics exposing only predictive moments."""

        # One precision-belief/normalizer set for the whole candidate set.
        weights = self._modality_weights()
        mean = belief.mean.unsqueeze(0)
        variance = belief.logvar.exp().unsqueeze(0)
        ambiguity = torch.tensor(0.0, device=mean.device)
        transition_risk = torch.tensor(0.0, device=mean.device)
        transition_ambiguity = torch.tensor(0.0, device=mean.device)
        epistemic_value = torch.tensor(0.0, device=mean.device)
        first_transition_used = False

        for step, action in enumerate(policy.actions):
            if action.stop:
                break
            if first_transition is not None and not first_transition_used:
                next_mean, next_variance = first_transition
                next_mean = next_mean.to(mean.device, dtype=mean.dtype)
                next_variance = next_variance.to(mean.device, dtype=mean.dtype)
                if next_mean.ndim == 1:
                    next_mean = next_mean.unsqueeze(0)
                if next_variance.ndim == 1:
                    next_variance = next_variance.unsqueeze(0)
                next_mean = project_summary_transition_support(mean, next_mean)
                next_variance = torch.clamp(next_variance, min=1e-8)
                first_transition_used = True
            else:
                a = torch.from_numpy(
                    encoded_action_vector(
                        action,
                        self.cfg,
                        policy.motor_primitive if step == 0 else None,
                    )
                ).to(mean.device).unsqueeze(0)
                next_mean, aleatoric, epistemic = self.dynamics.predictive_moments(mean, a)
                marginal_entropy = normal_entropy_from_variance(aleatoric + epistemic).mean()
                conditional_entropy = normal_entropy_from_variance(aleatoric).mean()
                next_mean = project_summary_transition_support(mean, next_mean)
                next_variance = torch.clamp(aleatoric + epistemic, min=1e-8)

                transition_risk = transition_risk - marginal_entropy
                transition_ambiguity = transition_ambiguity + conditional_entropy
                epistemic_value = epistemic_value + torch.clamp(marginal_entropy - conditional_entropy, min=0.0)

            ambiguity = ambiguity + self.observation_model.ambiguity(next_mean).mean()
            mean = next_mean
            variance = next_variance

        coverage_mean = torch.clamp(mean[..., 0], 1e-4, 1.0 - 1e-4)
        coverage_variance = torch.clamp(variance[..., 0], min=1e-8)
        coverage_std = torch.sqrt(coverage_variance)
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=weights.terminal,
            concentration_floor=weights.concentration_floor,
        )
        terminal_risk = terminal_risk.mean()
        terminal_entropy = terminal_entropy.mean()
        pragmatic_value = pragmatic_value.mean()

        # Belief-weighted modality contributions. `epistemic_value` carries the
        # SAME transition weight as transition_risk/transition_ambiguity because
        # R_trans + A_trans == -I(theta; s_next); it is logged, never summed.
        ambiguity = weights.ambiguity * ambiguity
        transition_risk = weights.transition * transition_risk
        transition_ambiguity = weights.transition * transition_ambiguity
        epistemic_value = weights.transition * epistemic_value
        motor_risk_value = float(motor_risk if first_transition_used else 0.0)
        motor_ambiguity_value = float(motor_ambiguity if first_transition_used else 0.0)
        motor_epistemic_value_used = float(motor_epistemic_value if first_transition_used else 0.0)
        # See _evaluate_ensemble_batch: motor_ambiguity_value is logged, not summed.
        total = self._assemble_total(
            terminal_risk=terminal_risk,
            ambiguity=ambiguity,
            transition_risk=transition_risk,
            transition_ambiguity=transition_ambiguity,
            motor_risk=motor_risk_value,
            motor_epistemic_value=motor_epistemic_value_used,
        )
        return EFEComponents(
            total=float(total.item()),
            terminal_risk=float(terminal_risk.item()),
            ambiguity=float(ambiguity.item()),
            epistemic_value=float(epistemic_value.item()),
            terminal_coverage_mean=float(coverage_mean.mean().item()),
            terminal_coverage_std=float(coverage_std.mean().item()),
            terminal_entropy=float(terminal_entropy.item()),
            pragmatic_value=float(pragmatic_value.item()),
            transition_risk=float(transition_risk.item()),
            transition_ambiguity=float(transition_ambiguity.item()),
            execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
            contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
            motor_overshoot=motor_overshoot if first_transition_used else 0.0,
            motor_risk=motor_risk_value,
            motor_ambiguity=motor_ambiguity_value,
            motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
            motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
            motor_feasible=motor_feasible,
            execution_forecast_used=first_transition_used,
            **self._component_units(weights),
        )
