from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .config import PainterConfig
from .efe_common import project_material_support, terminal_preference_terms
from .inference import normal_entropy_from_variance
from .local_spatial import local_patch_bounds_for_action, pixel_logvar_from_state, pixel_material_from_state
from .models import LocalSpatialDynamicsEnsemble, SpatialDynamicsEnsemble
from .policies import Policy
from .preferences import TerminalCoveragePreference
from .spatial_state import SpatialCanvasState, rasterize_stroke_action


class SpatialTransitionModel(Protocol):
    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class CompositionModel(Protocol):
    def compression_gap(self, fields: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(slots=True)
class SpatialEFEComponents:
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
    composition_gap: float = 0.0
    composition_risk: float = 0.0
    grid_size: int = 0
    material_channels: int = 0
    execution_uncertainty: float = 0.0
    contact_loss_probability: float = 0.0
    motor_overshoot: float = 0.0
    motor_risk: float = 0.0
    motor_ambiguity: float = 0.0
    motor_epistemic_value: float = 0.0
    motor_efe_approximation: str = ""
    motor_feasible: bool = True
    execution_forecast_used: bool = False
    rollout_mode: str = "dense_grid"
    rollout_grid_size: int = 0
    active_patch_area_fraction: float = 0.0
    local_transition_steps: int = 0
    sequential_patch_steps: int = 0
    identity_transition_approximation: str = ""


class SpatialExpectedFreeEnergy:
    """Risk-plus-ambiguity EFE for explicit spatial material fields.

    With a learned CNN ensemble, rollouts are member-wise trajectory samples:
    each ensemble member propagates its own material-field particle. Terminal
    coverage variance combines across-member disagreement of the aggregate
    coverage (which carries the spatial correlation induced by strokes) with
    the mean within-member cell-wise delta-method variance. Logged components
    are scaled by the declared per-modality precisions in the config.

    When a composition hierarchy is provided, terminal preferences include the
    declared structural prior p*(s_T) ~ exp(precision * compression_gap(s_T)):
    the composition risk is -precision * gap averaged over member particles.
    """

    def __init__(
        self,
        config: PainterConfig,
        dynamics: SpatialTransitionModel,
        terminal_preference: TerminalCoveragePreference,
        device: torch.device | str = "cpu",
        composition: CompositionModel | None = None,
    ) -> None:
        self.cfg = config
        self.dynamics = dynamics
        self.preference = terminal_preference
        self.device = torch.device(device)
        self.composition = composition

    def _composition_terms(self, terminal_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample compression gap and its risk contribution.

        `terminal_fields` is [N, C, H, W]; returns (gap, risk) with risk
        already scaled by the declared composition precision.
        """

        if self.composition is None or self.cfg.composition_gap_precision <= 0.0:
            zeros = torch.zeros(terminal_fields.shape[0], device=terminal_fields.device)
            return zeros, zeros
        gap = self.composition.compression_gap(terminal_fields)
        return gap, -self.cfg.composition_gap_precision * gap

    @torch.no_grad()
    def evaluate(self, belief: SpatialCanvasState, policy: Policy) -> SpatialEFEComponents:
        return self._evaluate(belief, policy)

    @torch.no_grad()
    def evaluate_batch(
        self,
        belief: SpatialCanvasState,
        policies: Sequence[Policy],
    ) -> list[SpatialEFEComponents]:
        policies = list(policies)
        if not policies:
            return []
        if self._uses_local_patch_rollout(belief):
            if isinstance(self.dynamics, SpatialDynamicsEnsemble):
                return self._evaluate_local_ensemble_batch(belief, policies)
            return [self._evaluate_local_mixture(belief, policy) for policy in policies]
        if isinstance(self.dynamics, SpatialDynamicsEnsemble):
            return self._evaluate_ensemble_batch(belief, policies)
        return [self._evaluate_mixture(belief, policy) for policy in policies]

    @torch.no_grad()
    def evaluate_with_first_transition(
        self,
        belief: SpatialCanvasState,
        policy: Policy,
        next_material_mean: torch.Tensor,
        next_material_variance: torch.Tensor,
        *,
        execution_uncertainty: float,
        contact_loss_probability: float,
        motor_overshoot: float,
        motor_feasible: bool,
        motor_risk: float = 0.0,
        motor_ambiguity: float = 0.0,
        motor_epistemic_value: float = 0.0,
        motor_efe_approximation: str = "",
    ) -> SpatialEFEComponents:
        return self._evaluate(
            belief,
            policy,
            first_transition=(next_material_mean, next_material_variance),
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
        belief: SpatialCanvasState,
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
    ) -> SpatialEFEComponents:
        if self._uses_local_patch_rollout(belief):
            if isinstance(self.dynamics, SpatialDynamicsEnsemble):
                return self._evaluate_local_ensemble_batch(
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
            return self._evaluate_local_mixture(
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
        if isinstance(self.dynamics, SpatialDynamicsEnsemble):
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

    def _uses_local_patch_rollout(self, belief: SpatialCanvasState) -> bool:
        return (
            self.cfg.spatial_transition_mode == "local_patch"
            and bool(belief.pyramid)
            and pixel_material_from_state(belief).shape[-1] >= belief.grid_size
        )

    def _evaluate_local_ensemble_batch(
        self,
        belief: SpatialCanvasState,
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
    ) -> list[SpatialEFEComponents]:
        if first_transition is not None and len(policies) != 1:
            raise ValueError("An execution-forecast first transition applies to a single policy.")
        if first_transition is not None:
            return [
                self._evaluate_local_ensemble_policy(
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
                for policy in policies
            ]

        device = self.device
        policy_count = len(policies)
        material = torch.tensor(pixel_material_from_state(belief), device=device, dtype=torch.float32)
        logvar = torch.tensor(pixel_logvar_from_state(belief, self.cfg), device=device, dtype=torch.float32)
        channels, height, width = material.shape
        field_shape = (channels, height, width)
        member_count = len(self.dynamics.members)
        member_states = material.view(1, 1, *field_shape).expand(policy_count, member_count, *field_shape).clone()
        member_within = logvar.exp().view(1, 1, *field_shape).expand_as(member_states).clone()
        full_area = float(height * width)
        touched = torch.zeros((policy_count, height, width), device=device, dtype=torch.bool)

        ambiguity = torch.zeros(policy_count, device=device)
        transition_risk = torch.zeros(policy_count, device=device)
        transition_ambiguity = torch.zeros(policy_count, device=device)
        epistemic_value = torch.zeros(policy_count, device=device)
        local_steps = [0 for _ in policies]
        sequential_steps = [0 for _ in policies]
        depths = [len(policy.actions) - 1 for policy in policies]

        for step in range(max(depths, default=0)):
            specs: list[tuple[int, object, torch.Tensor]] = []
            for policy_index, policy in enumerate(policies):
                if depths[policy_index] <= step:
                    continue
                action = policy.actions[step]
                bounds = local_patch_bounds_for_action(action, width, self.cfg)
                if bounds is None:
                    continue
                if bounds.area > max(1, self.cfg.local_patch_sequential_cell_limit):
                    sequential_steps[policy_index] += 1
                row_slice, col_slice = bounds.slices()
                action_raster = torch.tensor(
                    rasterize_stroke_action(
                        action,
                        width,
                        motor_primitive=policy.motor_primitive if step == 0 else None,
                        config=self.cfg,
                    ),
                    device=device,
                    dtype=torch.float32,
                )[:, row_slice, col_slice]
                specs.append((policy_index, bounds, action_raster))
            if not specs:
                continue

            buckets: dict[tuple[int, int], list[tuple[int, object, torch.Tensor]]] = {}
            for spec in specs:
                _, bounds, _ = spec
                buckets.setdefault((bounds.height, bounds.width), []).append(spec)

            for (bucket_h, bucket_w), bucket_specs in buckets.items():
                patch_count = len(bucket_specs)
                action_channels = bucket_specs[0][2].shape[0]
                state_batch = torch.zeros(
                    (member_count, patch_count, channels, bucket_h, bucket_w),
                    device=device,
                    dtype=member_states.dtype,
                )
                action_batch = torch.zeros(
                    (member_count, patch_count, action_channels, bucket_h, bucket_w),
                    device=device,
                    dtype=member_states.dtype,
                )
                for spec_index, (policy_index, bounds, action_raster) in enumerate(bucket_specs):
                    row_slice, col_slice = bounds.slices()
                    state_batch[:, spec_index] = member_states[policy_index, :, :, row_slice, col_slice]
                    action_batch[:, spec_index] = action_raster.unsqueeze(0).expand(member_count, -1, -1, -1)

                means_by_member: list[torch.Tensor] = []
                logvars_by_member: list[torch.Tensor] = []
                for member_index, member in enumerate(self.dynamics.members):
                    means, logvars = member(state_batch[member_index], action_batch[member_index])
                    means_by_member.append(means)
                    logvars_by_member.append(logvars)

                for spec_index, (policy_index, bounds, _) in enumerate(bucket_specs):
                    row_slice, col_slice = bounds.slices()
                    next_patch = torch.stack(
                        [means_by_member[member_index][spec_index] for member_index in range(member_count)],
                        dim=0,
                    )
                    selected_within = torch.stack(
                        [logvars_by_member[member_index][spec_index].exp() for member_index in range(member_count)],
                        dim=0,
                    ).clamp(min=1e-8)

                    aleatoric = selected_within.mean(dim=0)
                    epistemic_variance = next_patch.var(dim=0, unbiased=False)
                    marginal_entropy = self._scaled_normal_entropy(aleatoric + epistemic_variance, full_area)
                    conditional_entropy = self._scaled_normal_entropy(selected_within, full_area).mean()
                    transition_risk[policy_index] = transition_risk[policy_index] - marginal_entropy
                    transition_ambiguity[policy_index] = transition_ambiguity[policy_index] + conditional_entropy
                    epistemic_value[policy_index] = epistemic_value[policy_index] + torch.clamp(
                        marginal_entropy - conditional_entropy,
                        min=0.0,
                    )

                    member_states[policy_index, :, :, row_slice, col_slice] = next_patch
                    member_within[policy_index, :, :, row_slice, col_slice] = selected_within
                    touched[policy_index, row_slice, col_slice] = True
                    local_steps[policy_index] += 1
                    ambiguity[policy_index] = ambiguity[policy_index] + self._observation_ambiguity_scaled(
                        next_patch,
                        full_area,
                    ).mean()

        coverage_mean, coverage_variance = self._member_coverage_moments(member_states, member_within)
        coverage_std = torch.sqrt(torch.clamp(coverage_variance, min=1e-8))
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=self.cfg.terminal_risk_precision,
        )
        composition_fields = self._composition_fields_from_terminal(
            member_states.reshape(policy_count * member_count, *field_shape)
        )
        member_gap, member_composition_risk = self._composition_terms(composition_fields)
        composition_gap = member_gap.reshape(policy_count, member_count).mean(dim=1)
        composition_risk = member_composition_risk.reshape(policy_count, member_count).mean(dim=1)

        ambiguity = self.cfg.ambiguity_precision * ambiguity
        transition_risk = self.cfg.transition_precision * transition_risk
        transition_ambiguity = self.cfg.transition_precision * transition_ambiguity
        epistemic_value = self.cfg.transition_precision * epistemic_value
        motor_risk_t = torch.zeros(policy_count, device=device)
        motor_ambiguity_t = torch.zeros(policy_count, device=device)
        total = (
            terminal_risk
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + composition_risk
            + motor_risk_t
            + motor_ambiguity_t
        )
        patch_area = touched.float().mean(dim=(1, 2))

        return [
            SpatialEFEComponents(
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
                composition_gap=float(composition_gap[index].item()),
                composition_risk=float(composition_risk[index].item()),
                grid_size=belief.grid_size,
                material_channels=channels,
                execution_uncertainty=execution_uncertainty,
                contact_loss_probability=contact_loss_probability,
                motor_overshoot=motor_overshoot,
                motor_risk=float(motor_risk_t[index].item()),
                motor_ambiguity=float(motor_ambiguity_t[index].item()),
                motor_epistemic_value=0.0,
                motor_efe_approximation="",
                motor_feasible=motor_feasible,
                execution_forecast_used=False,
                rollout_mode="local_patch",
                rollout_grid_size=width,
                active_patch_area_fraction=float(patch_area[index].item()),
                local_transition_steps=local_steps[index],
                sequential_patch_steps=sequential_steps[index],
                identity_transition_approximation=self._local_identity_approximation(local_steps[index]),
            )
            for index in range(policy_count)
        ]

    def _evaluate_local_ensemble_policy(
        self,
        belief: SpatialCanvasState,
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
    ) -> SpatialEFEComponents:
        device = self.device
        material = torch.tensor(pixel_material_from_state(belief), device=device, dtype=torch.float32)
        logvar = torch.tensor(pixel_logvar_from_state(belief, self.cfg), device=device, dtype=torch.float32)
        channels, height, width = material.shape
        member_count = len(self.dynamics.members)
        member_states = material.unsqueeze(0).expand(member_count, channels, height, width).clone()
        member_within = logvar.exp().unsqueeze(0).expand_as(member_states).clone()
        full_area = float(height * width)
        touched = torch.zeros((height, width), device=device, dtype=torch.bool)

        ambiguity = torch.tensor(0.0, device=device)
        transition_risk = torch.tensor(0.0, device=device)
        transition_ambiguity = torch.tensor(0.0, device=device)
        epistemic_value = torch.tensor(0.0, device=device)
        local_steps = 0
        sequential_steps = 0
        first_transition_used = False

        for step, action in enumerate(policy.actions):
            if action.stop:
                break
            if first_transition is not None and not first_transition_used:
                next_mean, next_variance = self._transition_to_rollout_grid(first_transition, channels, height, width)
                current = member_states
                projected = project_material_support(
                    current,
                    next_mean.unsqueeze(0).expand_as(current),
                    self.cfg.thickness_scale,
                    self.cfg.canvas_ground_tone,
                )
                member_states = projected
                member_within = torch.clamp(next_variance.unsqueeze(0).expand_as(current), min=1e-8)
                changed = (projected - current).abs().sum(dim=1) > 1e-8
                touched = touched | changed.any(dim=0)
                ambiguity = ambiguity + self._observation_ambiguity(member_states).mean()
                first_transition_used = True
                continue

            bounds = local_patch_bounds_for_action(action, width, self.cfg)
            if bounds is None:
                continue
            if bounds.area > max(1, self.cfg.local_patch_sequential_cell_limit):
                sequential_steps += 1
            row_slice, col_slice = bounds.slices()
            action_raster = torch.tensor(
                rasterize_stroke_action(
                    action,
                    width,
                    motor_primitive=policy.motor_primitive if step == 0 else None,
                    config=self.cfg,
                ),
                device=device,
                dtype=torch.float32,
            )[:, row_slice, col_slice]
            patch_states = member_states[:, :, row_slice, col_slice]
            patch_actions = action_raster.unsqueeze(0).expand(member_count, -1, -1, -1)
            selected_means: list[torch.Tensor] = []
            selected_logvars: list[torch.Tensor] = []
            for member_index, member in enumerate(self.dynamics.members):
                mean, logvar_patch = member(
                    patch_states[member_index : member_index + 1],
                    patch_actions[member_index : member_index + 1],
                )
                selected_means.append(mean[0])
                selected_logvars.append(logvar_patch[0])
            next_patch = torch.stack(selected_means, dim=0)
            selected_within = torch.stack(selected_logvars, dim=0).exp().clamp(min=1e-8)

            aleatoric = selected_within.mean(dim=0)
            epistemic_variance = next_patch.var(dim=0, unbiased=False)
            marginal_entropy = self._scaled_normal_entropy(aleatoric + epistemic_variance, full_area)
            conditional_entropy = self._scaled_normal_entropy(selected_within, full_area).mean()
            transition_risk = transition_risk - marginal_entropy
            transition_ambiguity = transition_ambiguity + conditional_entropy
            epistemic_value = epistemic_value + torch.clamp(marginal_entropy - conditional_entropy, min=0.0)

            member_states[:, :, row_slice, col_slice] = next_patch
            member_within[:, :, row_slice, col_slice] = selected_within
            touched[row_slice, col_slice] = True
            local_steps += 1
            ambiguity = ambiguity + self._observation_ambiguity_scaled(next_patch, full_area).mean()

        coverage_mean, coverage_variance = self._member_coverage_moments(
            member_states.unsqueeze(0),
            member_within.unsqueeze(0),
        )
        coverage_std = torch.sqrt(torch.clamp(coverage_variance, min=1e-8))
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=self.cfg.terminal_risk_precision,
        )
        composition_fields = self._composition_fields_from_terminal(member_states)
        member_gap, member_composition_risk = self._composition_terms(composition_fields)
        composition_gap = member_gap.mean()
        composition_risk = member_composition_risk.mean()

        ambiguity = self.cfg.ambiguity_precision * ambiguity
        transition_risk = self.cfg.transition_precision * transition_risk
        transition_ambiguity = self.cfg.transition_precision * transition_ambiguity
        epistemic_value = self.cfg.transition_precision * epistemic_value
        motor_risk_value = float(motor_risk if first_transition_used else 0.0)
        motor_ambiguity_value = float(motor_ambiguity if first_transition_used else 0.0)
        total = (
            terminal_risk[0]
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + composition_risk
            + motor_risk_value
            + motor_ambiguity_value
        )
        return SpatialEFEComponents(
            total=float(total.item()),
            terminal_risk=float(terminal_risk[0].item()),
            ambiguity=float(ambiguity.item()),
            epistemic_value=float(epistemic_value.item()),
            terminal_coverage_mean=float(coverage_mean[0].item()),
            terminal_coverage_std=float(coverage_std[0].item()),
            terminal_entropy=float(terminal_entropy[0].item()),
            pragmatic_value=float(pragmatic_value[0].item()),
            transition_risk=float(transition_risk.item()),
            transition_ambiguity=float(transition_ambiguity.item()),
            composition_gap=float(composition_gap.item()),
            composition_risk=float(composition_risk.item()),
            grid_size=belief.grid_size,
            material_channels=channels,
            execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
            contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
            motor_overshoot=motor_overshoot if first_transition_used else 0.0,
            motor_risk=motor_risk_value,
            motor_ambiguity=motor_ambiguity_value,
            motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
            motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
            motor_feasible=motor_feasible,
            execution_forecast_used=first_transition_used,
            rollout_mode="local_patch",
            rollout_grid_size=width,
            active_patch_area_fraction=float(touched.float().mean().item()),
            local_transition_steps=local_steps,
            sequential_patch_steps=sequential_steps,
            identity_transition_approximation=self._local_identity_approximation(local_steps),
        )

    def _evaluate_local_mixture(
        self,
        belief: SpatialCanvasState,
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
    ) -> SpatialEFEComponents:
        device = self.device
        mean = torch.tensor(pixel_material_from_state(belief), device=device, dtype=torch.float32).unsqueeze(0)
        variance = torch.tensor(pixel_logvar_from_state(belief, self.cfg), device=device, dtype=torch.float32).exp().unsqueeze(0)
        _, channels, height, width = mean.shape
        full_area = float(height * width)
        touched = torch.zeros((height, width), device=device, dtype=torch.bool)
        ambiguity = torch.tensor(0.0, device=device)
        transition_risk = torch.tensor(0.0, device=device)
        transition_ambiguity = torch.tensor(0.0, device=device)
        epistemic_value = torch.tensor(0.0, device=device)
        local_steps = 0
        sequential_steps = 0
        first_transition_used = False

        for step, action in enumerate(policy.actions):
            if action.stop:
                break
            if first_transition is not None and not first_transition_used:
                next_mean, next_variance = self._transition_to_rollout_grid(first_transition, channels, height, width)
                proposed = next_mean.unsqueeze(0)
                projected = project_material_support(
                    mean,
                    proposed,
                    self.cfg.thickness_scale,
                    self.cfg.canvas_ground_tone,
                )
                changed = (projected - mean).abs().sum(dim=1) > 1e-8
                touched = touched | changed[0]
                mean = projected
                variance = torch.clamp(next_variance.unsqueeze(0), min=1e-8)
                ambiguity = ambiguity + self._observation_ambiguity(mean).mean()
                first_transition_used = True
                continue

            bounds = local_patch_bounds_for_action(action, width, self.cfg)
            if bounds is None:
                continue
            if bounds.area > max(1, self.cfg.local_patch_sequential_cell_limit):
                sequential_steps += 1
            row_slice, col_slice = bounds.slices()
            patch_mean = mean[:, :, row_slice, col_slice]
            action_raster = torch.tensor(
                rasterize_stroke_action(
                    action,
                    width,
                    motor_primitive=policy.motor_primitive if step == 0 else None,
                    config=self.cfg,
                ),
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)[:, :, row_slice, col_slice]
            next_patch, aleatoric, epistemic = self.dynamics.predictive_moments(patch_mean, action_raster)
            next_patch = project_material_support(
                patch_mean,
                next_patch,
                self.cfg.thickness_scale,
                self.cfg.canvas_ground_tone,
            )
            next_variance = torch.clamp(aleatoric + epistemic, min=1e-8)

            marginal_entropy = self._scaled_normal_entropy(next_variance, full_area).mean()
            conditional_entropy = self._scaled_normal_entropy(torch.clamp(aleatoric, min=1e-8), full_area).mean()
            transition_risk = transition_risk - marginal_entropy
            transition_ambiguity = transition_ambiguity + conditional_entropy
            epistemic_value = epistemic_value + torch.clamp(marginal_entropy - conditional_entropy, min=0.0)
            mean[:, :, row_slice, col_slice] = next_patch
            variance[:, :, row_slice, col_slice] = next_variance
            touched[row_slice, col_slice] = True
            local_steps += 1
            ambiguity = ambiguity + self._observation_ambiguity_scaled(next_patch, full_area).mean()

        coverage_mean, coverage_variance = self._coverage_moments(mean, variance)
        coverage_std = torch.sqrt(torch.clamp(coverage_variance, min=1e-8))
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=self.cfg.terminal_risk_precision,
        )
        terminal_risk = terminal_risk.mean()
        terminal_entropy = terminal_entropy.mean()
        pragmatic_value = pragmatic_value.mean()
        composition_fields = self._composition_fields_from_terminal(mean)
        composition_gap, composition_risk = self._composition_terms(composition_fields)
        composition_gap = composition_gap.mean()
        composition_risk = composition_risk.mean()

        ambiguity = self.cfg.ambiguity_precision * ambiguity
        transition_risk = self.cfg.transition_precision * transition_risk
        transition_ambiguity = self.cfg.transition_precision * transition_ambiguity
        epistemic_value = self.cfg.transition_precision * epistemic_value
        motor_risk_value = float(motor_risk if first_transition_used else 0.0)
        motor_ambiguity_value = float(motor_ambiguity if first_transition_used else 0.0)
        total = (
            terminal_risk
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + composition_risk
            + motor_risk_value
            + motor_ambiguity_value
        )
        return SpatialEFEComponents(
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
            composition_gap=float(composition_gap.item()),
            composition_risk=float(composition_risk.item()),
            grid_size=belief.grid_size,
            material_channels=channels,
            execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
            contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
            motor_overshoot=motor_overshoot if first_transition_used else 0.0,
            motor_risk=motor_risk_value,
            motor_ambiguity=motor_ambiguity_value,
            motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
            motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
            motor_feasible=motor_feasible,
            execution_forecast_used=first_transition_used,
            rollout_mode="local_patch",
            rollout_grid_size=width,
            active_patch_area_fraction=float(touched.float().mean().item()),
            local_transition_steps=local_steps,
            sequential_patch_steps=sequential_steps,
            identity_transition_approximation=self._local_identity_approximation(local_steps),
        )

    def _transition_to_rollout_grid(
        self,
        first_transition: tuple[torch.Tensor, torch.Tensor],
        channels: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, variance = first_transition
        mean = self._field_to_grid(mean, channels, height, width)
        variance = self._field_to_grid(variance, channels, height, width)
        return mean, torch.clamp(variance, min=1e-8)

    def _field_to_grid(self, field: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
        field = field.to(self.device, dtype=torch.float32)
        if field.ndim == 1:
            cells = int(field.numel() // max(1, channels))
            side = int(round(cells ** 0.5))
            field = field.reshape(channels, side, side)
        elif field.ndim == 4:
            field = field.squeeze(0)
        if field.shape[-2:] != (height, width):
            field = F.interpolate(field.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
        return field

    def _composition_fields_from_terminal(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim == 3:
            fields = fields.unsqueeze(0)
        target = int(self.cfg.spatial_grid_size)
        if fields.shape[-2:] == (target, target):
            return fields
        return F.interpolate(fields, size=(target, target), mode="area")

    def _scaled_normal_entropy(self, variance: torch.Tensor, full_area: float) -> torch.Tensor:
        entropy = 0.5 * torch.log(2.0 * math.pi * math.e * torch.clamp(variance, min=1e-9))
        channel_count = float(max(1, variance.shape[-3]))
        return entropy.sum(dim=(-3, -2, -1)) / max(1.0, channel_count * full_area)

    def _observation_ambiguity_scaled(self, material_mean: torch.Tensor, full_area: float) -> torch.Tensor:
        thickness = torch.clamp(material_mean[:, 0], min=0.0)
        wetness = torch.clamp(material_mean[:, 1], min=0.0)
        pigment = torch.clamp(material_mean[:, 2], min=0.0)
        smear = torch.clamp(0.5 * thickness + 0.75 * wetness + 0.35 * pigment, 0.0, 2.0)
        scale_values = [0.7, 0.9, 0.8, 0.6, 0.7, 0.5][: material_mean.shape[1]]
        scales = torch.tensor(scale_values, device=material_mean.device, dtype=material_mean.dtype).view(1, -1, 1, 1)
        std = self.cfg.base_observation_std + self.cfg.smear_observation_std * smear.unsqueeze(1) * scales
        entropy = Normal(material_mean, std).entropy()
        base_std = torch.full_like(material_mean, self.cfg.base_observation_std)
        base_entropy = Normal(material_mean, base_std).entropy()
        channel_count = float(max(1, material_mean.shape[1]))
        excess = torch.clamp(entropy - base_entropy, min=0.0)
        return excess.sum(dim=(1, 2, 3)) / max(1.0, channel_count * full_area)

    def _local_identity_approximation(self, local_steps: int) -> str:
        if local_steps <= 0:
            return ""
        return (
            "outside local stroke support uses an identity transition prior; "
            "constant outside-support entropy is omitted from local EFE terms"
        )

    def _evaluate_ensemble_batch(
        self,
        belief: SpatialCanvasState,
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
    ) -> list[SpatialEFEComponents]:
        if first_transition is not None and len(policies) != 1:
            raise ValueError("An execution-forecast first transition applies to a single policy.")
        device = self.device
        member_count = len(self.dynamics.members)
        policy_count = len(policies)
        material = torch.tensor(belief.material, device=device, dtype=torch.float32)
        channels, height, width = material.shape
        field_shape = (channels, height, width)

        member_states = material.view(1, 1, *field_shape).expand(policy_count, member_count, *field_shape).clone()
        member_within = (
            torch.tensor(belief.logvar, device=device, dtype=torch.float32)
            .exp()
            .view(1, 1, *field_shape)
            .expand(policy_count, member_count, *field_shape)
            .clone()
        )
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
                next_mean = next_mean.to(device, dtype=torch.float32).reshape(1, *field_shape)
                next_variance = next_variance.to(device, dtype=torch.float32).reshape(1, *field_shape)
                current_flat = member_states[active_t].reshape(len(active) * member_count, *field_shape)
                projected = project_material_support(
                    current_flat,
                    next_mean.expand(len(active) * member_count, *field_shape),
                    self.cfg.thickness_scale,
                    self.cfg.canvas_ground_tone,
                )
                member_states[active_t] = projected.reshape(len(active), member_count, *field_shape)
                member_within[active_t] = torch.clamp(
                    next_variance.view(1, 1, *field_shape).expand(len(active), member_count, *field_shape),
                    min=1e-8,
                )
                first_transition_used = True
            else:
                rasters = np.stack(
                    [
                        rasterize_stroke_action(
                            policies[index].actions[step],
                            belief.grid_size,
                            motor_primitive=policies[index].motor_primitive if step == 0 else None,
                            config=self.cfg,
                        )
                        for index in active
                    ]
                )
                action_batch = torch.from_numpy(rasters).to(device=device, dtype=torch.float32)
                flat_states = member_states[active_t].reshape(len(active) * member_count, *field_shape)
                flat_actions = action_batch.repeat_interleave(member_count, dim=0)
                all_means, all_logvars = self.dynamics(flat_states, flat_actions)
                rows = torch.arange(flat_states.shape[0], device=device)
                member_index = rows % member_count
                # Each spatial member projects its own prediction onto the
                # material support inside forward(), so no external projection
                # is applied on the ensemble path.
                selected_means = all_means[member_index, rows].reshape(len(active), member_count, *field_shape)
                selected_within = all_logvars[member_index, rows].exp().reshape(
                    len(active), member_count, *field_shape
                )

                aleatoric = selected_within.mean(dim=1)
                epistemic_variance = selected_means.var(dim=1, unbiased=False)
                marginal_entropy = normal_entropy_from_variance(
                    torch.clamp(aleatoric + epistemic_variance, min=1e-8)
                ).mean(dim=(1, 2))
                conditional_entropy = normal_entropy_from_variance(
                    torch.clamp(selected_within, min=1e-8)
                ).mean(dim=(1, 2, 3))
                transition_risk[active_t] = transition_risk[active_t] - marginal_entropy
                transition_ambiguity[active_t] = transition_ambiguity[active_t] + conditional_entropy
                epistemic_value[active_t] = epistemic_value[active_t] + torch.clamp(
                    marginal_entropy - conditional_entropy, min=0.0
                )

                member_states[active_t] = selected_means
                member_within[active_t] = torch.clamp(selected_within, min=1e-8)

            flat_members = member_states[active_t].reshape(len(active) * member_count, *field_shape)
            ambiguity[active_t] = ambiguity[active_t] + self._observation_ambiguity(flat_members).reshape(
                len(active), member_count
            ).mean(dim=1)

        coverage_mean, coverage_variance = self._member_coverage_moments(member_states, member_within)
        coverage_std = torch.sqrt(torch.clamp(coverage_variance, min=1e-8))
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=self.cfg.terminal_risk_precision,
        )
        member_gap, member_composition_risk = self._composition_terms(
            member_states.reshape(policy_count * member_count, *field_shape)
        )
        composition_gap = member_gap.reshape(policy_count, member_count).mean(dim=1)
        composition_risk = member_composition_risk.reshape(policy_count, member_count).mean(dim=1)

        ambiguity = self.cfg.ambiguity_precision * ambiguity
        transition_risk = self.cfg.transition_precision * transition_risk
        transition_ambiguity = self.cfg.transition_precision * transition_ambiguity
        epistemic_value = self.cfg.transition_precision * epistemic_value
        motor_risk_t = torch.full((policy_count,), float(motor_risk if first_transition_used else 0.0), device=device)
        motor_ambiguity_t = torch.full(
            (policy_count,), float(motor_ambiguity if first_transition_used else 0.0), device=device
        )
        total = (
            terminal_risk
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + composition_risk
            + motor_risk_t
            + motor_ambiguity_t
        )

        return [
            SpatialEFEComponents(
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
                composition_gap=float(composition_gap[index].item()),
                composition_risk=float(composition_risk[index].item()),
                grid_size=belief.grid_size,
                material_channels=channels,
                execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
                contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
                motor_overshoot=motor_overshoot if first_transition_used else 0.0,
                motor_risk=float(motor_risk_t[index].item()),
                motor_ambiguity=float(motor_ambiguity_t[index].item()),
                motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
                motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
                motor_feasible=motor_feasible,
                execution_forecast_used=first_transition_used,
                rollout_mode="dense_grid",
                rollout_grid_size=belief.grid_size,
            )
            for index in range(policy_count)
        ]

    def _member_coverage_moments(
        self,
        member_states: torch.Tensor,
        member_within: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminal coverage moments from member particles.

        Across-member variance of the aggregate coverage carries the spatial
        correlation a stroke induces between neighboring cells; the
        within-member term keeps the cell-independent delta-method variance of
        each member's own predictive density.
        """

        if member_states.shape[2] > 5:
            cell_coverage = torch.clamp(member_states[:, :, 5], 0.0, 1.0)
            cell_variance = torch.clamp(member_within[:, :, 5], min=1e-8)
        else:
            scale = max(1e-8, self.cfg.thickness_scale)
            thickness = torch.clamp(member_states[:, :, 0], min=0.0)
            cell_coverage = 1.0 - torch.exp(-thickness / scale)
            derivative = torch.exp(-thickness / scale) / scale
            cell_variance = derivative.square() * torch.clamp(member_within[:, :, 0], min=1e-8)
        member_coverage = cell_coverage.mean(dim=(-2, -1))
        cell_count = float(cell_coverage.shape[-2] * cell_coverage.shape[-1])
        within_aggregate = cell_variance.sum(dim=(-2, -1)) / (cell_count * cell_count)
        coverage_mean = torch.clamp(member_coverage.mean(dim=1), 1e-4, 1.0 - 1e-4)
        coverage_variance = torch.clamp(
            member_coverage.var(dim=1, unbiased=False) + within_aggregate.mean(dim=1),
            min=1e-8,
        )
        return coverage_mean, coverage_variance

    def _evaluate_mixture(
        self,
        belief: SpatialCanvasState,
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
    ) -> SpatialEFEComponents:
        """Moment-matched mixture rollout for dynamics exposing only predictive moments."""

        mean = torch.tensor(belief.material, device=self.device, dtype=torch.float32).unsqueeze(0)
        variance = torch.tensor(belief.logvar, device=self.device, dtype=torch.float32).exp().unsqueeze(0)
        ambiguity = torch.tensor(0.0, device=self.device)
        transition_risk = torch.tensor(0.0, device=self.device)
        transition_ambiguity = torch.tensor(0.0, device=self.device)
        epistemic_value = torch.tensor(0.0, device=self.device)
        first_transition_used = False

        for step, action in enumerate(policy.actions):
            if action.stop:
                break
            if first_transition is not None and not first_transition_used:
                next_mean, next_variance = first_transition
                next_mean = next_mean.to(self.device, dtype=mean.dtype)
                next_variance = next_variance.to(self.device, dtype=mean.dtype)
                if next_mean.ndim == 1:
                    next_mean = next_mean.reshape(1, *mean.shape[1:])
                elif next_mean.ndim == 3:
                    next_mean = next_mean.unsqueeze(0)
                if next_variance.ndim == 1:
                    next_variance = next_variance.reshape(1, *mean.shape[1:])
                elif next_variance.ndim == 3:
                    next_variance = next_variance.unsqueeze(0)
                next_mean = project_material_support(
                    mean, next_mean, self.cfg.thickness_scale, self.cfg.canvas_ground_tone
                )
                next_variance = torch.clamp(next_variance, min=1e-8)
                first_transition_used = True
            else:
                action_raster = torch.tensor(
                    rasterize_stroke_action(
                        action,
                        belief.grid_size,
                        motor_primitive=policy.motor_primitive if step == 0 else None,
                        config=self.cfg,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                next_mean, aleatoric, epistemic = self.dynamics.predictive_moments(mean, action_raster)
                next_mean = project_material_support(
                    mean, next_mean, self.cfg.thickness_scale, self.cfg.canvas_ground_tone
                )
                next_variance = torch.clamp(aleatoric + epistemic, min=1e-8)

                marginal_entropy = normal_entropy_from_variance(next_variance).mean()
                conditional_entropy = normal_entropy_from_variance(torch.clamp(aleatoric, min=1e-8)).mean()
                transition_risk = transition_risk - marginal_entropy
                transition_ambiguity = transition_ambiguity + conditional_entropy
                epistemic_value = epistemic_value + torch.clamp(marginal_entropy - conditional_entropy, min=0.0)
            ambiguity = ambiguity + self._observation_ambiguity(next_mean).mean()

            mean = next_mean
            variance = next_variance

        coverage_mean, coverage_variance = self._coverage_moments(mean, variance)
        coverage_std = torch.sqrt(torch.clamp(coverage_variance, min=1e-8))
        terminal_risk, terminal_entropy, pragmatic_value = terminal_preference_terms(
            self.preference,
            coverage_mean,
            coverage_variance,
            precision=self.cfg.terminal_risk_precision,
        )
        terminal_risk = terminal_risk.mean()
        terminal_entropy = terminal_entropy.mean()
        pragmatic_value = pragmatic_value.mean()
        composition_gap, composition_risk = self._composition_terms(mean)
        composition_gap = composition_gap.mean()
        composition_risk = composition_risk.mean()

        ambiguity = self.cfg.ambiguity_precision * ambiguity
        transition_risk = self.cfg.transition_precision * transition_risk
        transition_ambiguity = self.cfg.transition_precision * transition_ambiguity
        epistemic_value = self.cfg.transition_precision * epistemic_value
        motor_risk_value = float(motor_risk if first_transition_used else 0.0)
        motor_ambiguity_value = float(motor_ambiguity if first_transition_used else 0.0)
        total = (
            terminal_risk
            + ambiguity
            + transition_risk
            + transition_ambiguity
            + composition_risk
            + motor_risk_value
            + motor_ambiguity_value
        )
        return SpatialEFEComponents(
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
            composition_gap=float(composition_gap.item()),
            composition_risk=float(composition_risk.item()),
            grid_size=belief.grid_size,
            material_channels=int(mean.shape[1]),
            execution_uncertainty=execution_uncertainty if first_transition_used else 0.0,
            contact_loss_probability=contact_loss_probability if first_transition_used else 0.0,
            motor_overshoot=motor_overshoot if first_transition_used else 0.0,
            motor_risk=motor_risk_value,
            motor_ambiguity=motor_ambiguity_value,
            motor_epistemic_value=float(motor_epistemic_value if first_transition_used else 0.0),
            motor_efe_approximation=motor_efe_approximation if first_transition_used else "",
            motor_feasible=motor_feasible,
            execution_forecast_used=first_transition_used,
            rollout_mode="dense_grid",
            rollout_grid_size=belief.grid_size,
        )

    def _coverage_moments(self, material_mean: torch.Tensor, material_variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if material_mean.shape[1] > 5:
            cell_coverage = torch.clamp(material_mean[:, 5], 0.0, 1.0)
            cell_coverage_variance = torch.clamp(material_variance[:, 5], min=1e-8)
        else:
            thickness_mean = torch.clamp(material_mean[:, 0], min=0.0)
            thickness_variance = torch.clamp(material_variance[:, 0], min=1e-8)
            scale = max(1e-8, self.cfg.thickness_scale)
            cell_coverage = 1.0 - torch.exp(-thickness_mean / scale)
            derivative = torch.exp(-thickness_mean / scale) / scale
            cell_coverage_variance = derivative.square() * thickness_variance
        coverage_mean = torch.clamp(cell_coverage.mean(dim=(-2, -1)), 1e-4, 1.0 - 1e-4)
        cell_count = float(cell_coverage.shape[-2] * cell_coverage.shape[-1])
        coverage_variance = torch.clamp(cell_coverage_variance.sum(dim=(-2, -1)) / (cell_count * cell_count), min=1e-8)
        return coverage_mean.mean(), coverage_variance.mean()

    def _observation_ambiguity(self, material_mean: torch.Tensor) -> torch.Tensor:
        thickness = torch.clamp(material_mean[:, 0], min=0.0)
        wetness = torch.clamp(material_mean[:, 1], min=0.0)
        pigment = torch.clamp(material_mean[:, 2], min=0.0)
        smear = torch.clamp(0.5 * thickness + 0.75 * wetness + 0.35 * pigment, 0.0, 2.0)
        scale_values = [0.7, 0.9, 0.8, 0.6, 0.7, 0.5][: material_mean.shape[1]]
        scales = torch.tensor(scale_values, device=material_mean.device, dtype=material_mean.dtype).view(1, -1, 1, 1)
        std = self.cfg.base_observation_std + self.cfg.smear_observation_std * smear.unsqueeze(1) * scales
        entropy = Normal(material_mean, std).entropy().mean(dim=(1, 2, 3))
        base_std = torch.full_like(material_mean, self.cfg.base_observation_std)
        base_entropy = Normal(material_mean, base_std).entropy().mean(dim=(1, 2, 3))
        return torch.clamp(entropy - base_entropy, min=0.0)
