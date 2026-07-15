from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .canvas_hierarchy import HierarchicalCanvasModel, passage_step_descriptor, policy_descriptor
from .config import PainterConfig
from .env import StrokeAction
from .local_spatial import LocalPatchReplayBuffer
from .models import LocalSpatialDynamicsEnsemble, SpatialDynamicsEnsemble
from .policies import MotorPrimitiveLatent, PassageLatent, Policy, PolicySampler, policy_stop_log_prior
from .preferences import TerminalCoveragePreference
from .replay import ReplayBuffer
from .spatial_efe import SpatialEFEComponents, SpatialExpectedFreeEnergy
from .spatial_inference import SpatialVariationalStateEstimator
from .spatial_state import SpatialCanvasState, rasterize_stroke_action


class SpatialActiveInferencePainter:
    """Active-inference painter over explicit spatial material fields."""

    def __init__(self, config: PainterConfig, seed: int = 0, device: str | None = None) -> None:
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        np.random.seed(seed)

        if config.spatial_transition_mode == "local_patch":
            self.dynamics = LocalSpatialDynamicsEnsemble(config).to(self.device)
        else:
            self.dynamics = SpatialDynamicsEnsemble(config).to(self.device)
        self.preference = TerminalCoveragePreference(config)
        self.composition: HierarchicalCanvasModel | None = None
        self.composition_optimizer: torch.optim.Adam | None = None
        self.last_composition_loss: float | None = None
        self.last_hierarchy_transition_loss: float | None = None
        self.last_passage_trajectory_loss: float | None = None
        self.last_passage_trajectory_evaluation: dict[str, float] | None = None
        if config.composition_gap_precision > 0.0:
            self.composition = HierarchicalCanvasModel(config).to(self.device)
            self.composition_optimizer = torch.optim.Adam(
                self.composition.parameters(), lr=config.composition_lr
            )
        self.efe = SpatialExpectedFreeEnergy(
            config, self.dynamics, self.preference, self.device, composition=self.composition
        )
        self.estimator = SpatialVariationalStateEstimator(config, self.device)
        self.policy_sampler = PolicySampler(config, seed=seed)
        self.replay = (
            LocalPatchReplayBuffer(config.replay_capacity, seed=seed)
            if config.spatial_transition_mode == "local_patch"
            else ReplayBuffer(config.replay_capacity, seed=seed)
        )
        self.composition_replay = ReplayBuffer(config.replay_capacity, seed=seed + 101)
        self.passage_replay = ReplayBuffer(config.replay_capacity, seed=seed + 211)
        self.passage_step_replay = ReplayBuffer(config.replay_capacity, seed=seed + 307)
        self.optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=config.model_lr)
        material = np.zeros(
            (config.spatial_material_channels, config.spatial_grid_size, config.spatial_grid_size),
            dtype=np.float32,
        )
        logvar = np.full_like(material, -4.5, dtype=np.float32)
        self.belief = SpatialCanvasState(material=material, logvar=logvar)

    def reset_belief(self, observation: SpatialCanvasState) -> None:
        self.belief = self.estimator.initialize(observation)

    def reset_hierarchy_beliefs(self, observation: SpatialCanvasState) -> None:
        if self.composition is None:
            return
        fields = torch.tensor(observation.material, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.composition.reset_persistent_beliefs(fields)

    def update_hierarchy_beliefs(
        self,
        observation: SpatialCanvasState,
        actions: tuple[StrokeAction, ...],
    ) -> None:
        if self.composition is None:
            return
        fields = torch.tensor(observation.material, device=self.device, dtype=torch.float32).unsqueeze(0)
        descriptor = torch.tensor(
            policy_descriptor(actions, self.cfg),
            device=self.device,
            dtype=torch.float32,
        )
        self.composition.update_persistent_beliefs(fields, descriptor)

    def add_passage_transition(
        self,
        state: SpatialCanvasState,
        actions: tuple[StrokeAction, ...],
        next_state: SpatialCanvasState,
    ) -> None:
        self.passage_replay.add(
            state.flatten_mean(),
            policy_descriptor(actions, self.cfg),
            next_state.flatten_mean(),
        )

    def add_passage_step_transition(
        self,
        state: SpatialCanvasState,
        passage: PassageLatent,
        step_index: int,
        next_state: SpatialCanvasState,
    ) -> None:
        """Train the passage likelihood without updating the slow posterior."""

        if not self.cfg.passage_trajectory_enabled:
            return
        self.passage_step_replay.add(
            state.flatten_mean(),
            passage_step_descriptor(passage, step_index),
            next_state.flatten_mean(),
        )

    @property
    def last_vfe(self):
        return self.estimator.last_vfe

    def update_belief(
        self,
        previous_action: StrokeAction,
        observation: SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        self.belief = self.estimator.infer(
            self.belief,
            previous_action,
            observation,
            self.dynamics,
            motor_primitive,
        )

    def add_transition(
        self,
        state: SpatialCanvasState,
        action: StrokeAction,
        next_state: SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        if isinstance(self.replay, LocalPatchReplayBuffer):
            self.replay.add_from_states(state, action, next_state, self.cfg, motor_primitive)
        else:
            self.replay.add(
                state.flatten_mean(),
                rasterize_stroke_action(
                    action,
                    state.grid_size,
                    motor_primitive=motor_primitive,
                    config=self.cfg,
                ).reshape(-1),
                next_state.flatten_mean(),
            )
        self.composition_replay.add(
            state.flatten_mean(),
            rasterize_stroke_action(
                action,
                state.grid_size,
                motor_primitive=motor_primitive,
                config=self.cfg,
            ).reshape(-1),
            next_state.flatten_mean(),
        )

    def infer_policy(self) -> tuple[Policy, SpatialEFEComponents, list[tuple[Policy, SpatialEFEComponents, float]]]:
        coverage_field = self.belief.coverage(self.cfg.paint_presence_threshold)
        policies = self.policy_sampler.sample(coverage_field)
        components = self.efe.evaluate_batch(self.belief, policies)
        g = torch.tensor([component.total for component in components], device=self.device)
        believed_coverage = self.belief.material_coverage_mean(self.cfg.paint_presence_threshold)
        log_prior = torch.tensor(
            [policy_stop_log_prior(policy, believed_coverage, self.cfg) for policy in policies],
            device=self.device,
        )
        posterior = torch.softmax(-self.cfg.policy_precision * (g - g.min()) + log_prior, dim=0)
        index = int(torch.multinomial(posterior, 1).item())
        ranked = sorted(
            zip(policies, components, posterior.detach().cpu().tolist()),
            key=lambda item: item[2],
            reverse=True,
        )
        return policies[index], components[index], ranked

    def train_dynamics(self, gradient_steps: int = 1) -> float | None:
        if len(self.replay) < self.cfg.batch_size:
            return None
        total = 0.0
        for _ in range(gradient_steps):
            if isinstance(self.dynamics, LocalSpatialDynamicsEnsemble):
                assert isinstance(self.replay, LocalPatchReplayBuffer)
                batches = self.replay.sample_buckets(
                    self.cfg.batch_size,
                    self.device,
                    self.cfg.local_patch_batch_bucket_cells,
                    self.cfg.local_patch_sequential_cell_limit,
                )
                bootstrap_mask = self.dynamics.sample_bootstrap_mask(
                    self.cfg.batch_size,
                    self.device,
                    torch.float32,
                )
                normalizer = bootstrap_mask.sum().clamp(min=1.0)
                self.optimizer.zero_grad()
                step_loss = 0.0
                for batch in batches:
                    per_sample = self.dynamics.per_sample_nll(
                        batch.material,
                        batch.action,
                        batch.next_material,
                        batch.mask,
                    )
                    selected_mask = bootstrap_mask[:, list(batch.sample_indices)]
                    bucket_loss = (per_sample * selected_mask).sum() / normalizer
                    bucket_loss.backward()
                    step_loss += float(bucket_loss.item())
                loss_value = step_loss
            else:
                batch = self.replay.sample(self.cfg.batch_size, self.device)
                material = batch.state.reshape(
                    -1,
                    self.cfg.spatial_material_channels,
                    self.cfg.spatial_grid_size,
                    self.cfg.spatial_grid_size,
                )
                action = batch.action.reshape(
                    -1,
                    self.cfg.spatial_action_channels,
                    self.cfg.spatial_grid_size,
                    self.cfg.spatial_grid_size,
                )
                next_material = batch.next_state.reshape(
                    -1,
                    self.cfg.spatial_material_channels,
                    self.cfg.spatial_grid_size,
                    self.cfg.spatial_grid_size,
                )
                loss = self.dynamics.nll(material, action, next_material)
                self.optimizer.zero_grad()
                loss.backward()
                loss_value = float(loss.item())
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 5.0)
            self.optimizer.step()
            total += loss_value
        self._train_composition()
        return total / gradient_steps

    def _train_composition(self) -> None:
        if (
            self.composition is None
            or self.composition_optimizer is None
            or len(self.composition_replay) < self.cfg.batch_size
        ):
            return
        field_shape = (
            self.cfg.spatial_material_channels,
            self.cfg.spatial_grid_size,
            self.cfg.spatial_grid_size,
        )
        for _ in range(max(1, self.cfg.composition_train_steps)):
            batch = self.composition_replay.sample(self.cfg.batch_size, self.device)
            fields = torch.cat(
                [batch.state.reshape(-1, *field_shape), batch.next_state.reshape(-1, *field_shape)]
            )
            loss = self.composition.training_loss(fields)
            self.composition_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.composition.parameters(), 5.0)
            self.composition_optimizer.step()
            self.last_composition_loss = float(loss.item())
        self._train_hierarchy_transitions(field_shape)

    def _train_hierarchy_transitions(self, field_shape: tuple[int, int, int]) -> None:
        if self.composition is None or self.composition_optimizer is None:
            return
        aggregate_ready = len(self.passage_replay) >= self.cfg.hierarchy_transition_batch_size
        trajectory_ready = (
            self.cfg.passage_trajectory_enabled
            and len(self.passage_step_replay) >= self.cfg.passage_trajectory_batch_size
        )
        if not (aggregate_ready or trajectory_ready):
            return
        for _ in range(max(1, self.cfg.hierarchy_transition_train_steps) if aggregate_ready else 0):
            batch = self.passage_replay.sample(self.cfg.hierarchy_transition_batch_size, self.device)
            loss = self.composition.transition_training_loss(
                batch.state.reshape(-1, *field_shape),
                batch.action,
                batch.next_state.reshape(-1, *field_shape),
            )
            self.composition_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.composition.parameters(), 5.0)
            self.composition_optimizer.step()
            self.composition.mark_transition_update()
            self.last_hierarchy_transition_loss = float(loss.item())
        for _ in range(max(1, self.cfg.passage_trajectory_train_steps) if trajectory_ready else 0):
            batch = self.passage_step_replay.sample(self.cfg.passage_trajectory_batch_size, self.device)
            loss = self.composition.passage_trajectory_training_loss(
                batch.state.reshape(-1, *field_shape),
                batch.action,
                batch.next_state.reshape(-1, *field_shape),
            )
            self.composition_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.composition.parameters(), 5.0)
            self.composition_optimizer.step()
            self.composition.mark_passage_trajectory_update(batch.action)
            self.last_passage_trajectory_loss = float(loss.item())
            self.last_passage_trajectory_evaluation = self.composition.passage_trajectory_evaluation(
                batch.state.reshape(-1, *field_shape),
                batch.action,
                batch.next_state.reshape(-1, *field_shape),
            )

    def rebuild_passage_kind_support(self) -> None:
        if self.composition is None:
            return
        self.composition.rebuild_passage_kind_support(
            [transition[1] for transition in self.passage_step_replay.data]
        )

    @torch.no_grad()
    def belief_composition_gap(self) -> float | None:
        if self.composition is None:
            return None
        fields = torch.tensor(self.belief.material, device=self.device, dtype=torch.float32).unsqueeze(0)
        return float(self.composition.compression_gap(fields).item())

    @staticmethod
    def policy_dict(policy: Policy) -> list[dict[str, float | bool]]:
        return [asdict(action) for action in policy.actions]
