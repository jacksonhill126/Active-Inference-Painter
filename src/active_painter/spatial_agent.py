from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .canvas_hierarchy import HierarchicalCanvasModel, passage_step_descriptor, policy_descriptor
from .camera_inference import CameraSpatialLikelihood
from .camera_observation import CameraObservationBundle
from .config import PainterConfig
from .env import StrokeAction
from .local_spatial import LocalPatchReplayBuffer
from .models import LocalSpatialDynamicsEnsemble, SpatialDynamicsEnsemble
from .policies import (
    MotorPrimitiveLatent,
    PassageLatent,
    Policy,
    PolicySampler,
    policy_posterior_from_efe,
    policy_stop_log_prior,
)
from .precision_beliefs import (
    POLICY_PRECISION_KEY,
    GapIncrementBelief,
    PrecisionLedger,
)
from .preferences import TerminalCoveragePreference
from .proposal import (
    FALLBACK_BELIEF_SOURCES,
    BELIEF_SOURCE_NO_HIERARCHY,
    PolicyProposalNetwork,
    ProposalDivergence,
    ProposalTrainingBatch,
)
from .replay import ReplayBuffer
from .spatial_efe import SpatialEFEComponents, SpatialExpectedFreeEnergy
from .spatial_inference import SpatialVariationalStateEstimator
from .spatial_state import SpatialCanvasState, rasterize_stroke_action


class SpatialActiveInferencePainter:
    """Active-inference painter over explicit spatial material fields."""

    def __init__(
        self,
        config: PainterConfig,
        seed: int = 0,
        device: str | None = None,
        precision_ledger: PrecisionLedger | None = None,
        gap_increment: GapIncrementBelief | None = None,
    ) -> None:
        self.cfg = config
        self.precision_ledger = precision_ledger if precision_ledger is not None else PrecisionLedger(config)
        self.gap_increment = (
            gap_increment if gap_increment is not None else GapIncrementBelief.from_config(config)
        )
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
        # `composition_enabled` is the DECLARED STRUCTURE flag; the precision is
        # a belief. Splitting them keeps a learned precision from constructing or
        # destroying a model, and keeps the checkpoint architecture key a
        # declared constant. Every pre-existing config evaluates identically.
        if config.composition_enabled and config.composition_gap_precision > 0.0:
            self.composition = HierarchicalCanvasModel(config).to(self.device)
            self.composition_optimizer = torch.optim.Adam(
                self.composition.parameters(), lr=config.composition_lr
            )
        self.efe = SpatialExpectedFreeEnergy(
            config,
            self.dynamics,
            self.preference,
            self.device,
            composition=self.composition,
            precision_ledger=self.precision_ledger,
        )
        self.estimator = SpatialVariationalStateEstimator(config, self.device)
        self.camera_likelihood = CameraSpatialLikelihood(config)
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

        # --- Amortized candidate-policy proposal (Feature D) ----------------
        # Constructed LAST and inside `fork_rng`, and both facts are
        # load-bearing. Forking means this module's parameter initialization
        # consumes none of the GLOBAL torch stream, so not only is every earlier
        # module's initialization unchanged, every LATER `randn_like` in training
        # is unchanged too -- which is what keeps the torch-seeded tests
        # (test_epistemic_policy_selection, test_canvas_hierarchy) valid. Built on
        # CPU inside the fork and then moved: `.to(device)` consumes no
        # randomness, whereas constructing on CUDA inside a CPU-only fork would
        # shift the global CUDA stream. That ordering must not be reversed.
        self.policy_proposal: PolicyProposalNetwork | None = None
        self.policy_proposal_optimizer: torch.optim.Adam | None = None
        self.last_proposal_loss: float | None = None
        self.last_proposal_target_support_fraction: float | None = None
        self.last_proposal_belief_source: str = BELIEF_SOURCE_NO_HIERARCHY
        self.proposal_no_belief_skips = 0
        self.proposal_degenerate_target_skips = 0
        if config.learned_proposal_enabled and self.composition is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 5011)
                proposal = PolicyProposalNetwork(config)
            proposal.seed_generator(seed + 5011)
            self.policy_proposal = proposal.to(self.device)
            self.policy_proposal_optimizer = torch.optim.Adam(
                self.policy_proposal.parameters(), lr=config.learned_proposal_lr
            )
            # Plain attribute assignment, so `PolicySampler` stays constructed
            # where it was and the construction order above it is untouched.
            self.policy_sampler.learned_proposal = self.policy_proposal

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

    @property
    def last_camera_vfe(self):
        return self.camera_likelihood.last_vfe

    def assimilate_camera_observation(
        self,
        observation: CameraObservationBundle,
    ) -> SpatialCanvasState:
        """Update q(s) through the registered grayscale likelihood only."""

        self.belief = self.camera_likelihood.infer(self.belief, observation)
        return self.belief

    def predict_action_prior(
        self,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None = None,
        *,
        previous: SpatialCanvasState | None = None,
    ) -> SpatialCanvasState:
        """Advance the material belief without treating a prediction as data."""

        source = self.belief if previous is None else previous
        self.belief = self.estimator.predict(
            source,
            action,
            self.dynamics,
            motor_primitive,
        )
        return self.belief

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

    def proposal_belief_features(self) -> tuple[torch.Tensor | None, str]:
        """Belief features the learned proposal conditions on, plus their source.

        `None` means there is no learned proposal at all. A non-`None` tensor may
        still be a zero fallback -- the source label is the only way to tell, and
        `train_policy_proposal` refuses to train on a fallback.
        """

        if self.policy_proposal is None:
            return None, BELIEF_SOURCE_NO_HIERARCHY
        canvas_belief = self.composition.canvas_belief if self.composition is not None else None
        relational_belief = self.composition.relational_belief if self.composition is not None else None
        return PolicyProposalNetwork.features_from_beliefs(
            canvas_belief,
            relational_belief,
            self.cfg,
            self.device,
            has_hierarchy=self.composition is not None,
        )

    def train_policy_proposal(self, batch: ProposalTrainingBatch | None) -> float | None:
        """One SNIS amortization step toward the declared base-EFE posterior.

        Two refusals, both deliberate. A batch whose features came from the ZERO
        FALLBACK is skipped: `canvas_belief`/`relational_belief` are `None` until
        `reset_hierarchy_beliefs` runs, and `reset()` skips that when the
        observation boundary is blocked (the live default), so training there
        would silently amortize a constant. A batch whose target has no mass on any
        modelled candidate is skipped too, because SNIS against it is undefined.
        Both refusals are counted and reported rather than logged and forgotten.
        """

        if self.policy_proposal is None or self.policy_proposal_optimizer is None or batch is None:
            return None
        self.last_proposal_belief_source = batch.belief_feature_source
        if batch.belief_feature_source in FALLBACK_BELIEF_SOURCES or batch.features is None:
            self.proposal_no_belief_skips += 1
            return None
        if not batch.modelled_indices():
            self.proposal_degenerate_target_skips += 1
            return None
        loss_value: float | None = None
        for _ in range(max(1, int(self.cfg.learned_proposal_train_steps))):
            loss = self.policy_proposal.training_loss(batch, self.cfg)
            if loss is None:
                self.proposal_degenerate_target_skips += 1
                return None
            self.policy_proposal_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_proposal.parameters(), 5.0)
            self.policy_proposal_optimizer.step()
            self.policy_proposal.mark_update()
            loss_value = float(loss.item())
        self.last_proposal_loss = loss_value
        self.last_proposal_target_support_fraction = float(batch.target_support_fraction)
        return loss_value

    @torch.no_grad()
    def policy_proposal_divergence(
        self,
        coverage_field: np.ndarray | None,
        *,
        family: str = "passage",
    ) -> ProposalDivergence | None:
        """Declared `D_KL(learned proposal || hand-written proposal)`, in nats.

        Read-only diagnostics. No expected-free-energy term, policy, posterior, or
        control-flow branch reads this value; it is a quantity ABOUT the model, not
        in it.
        """

        if self.policy_proposal is None:
            return None
        features, _ = self.proposal_belief_features()
        if features is None:
            return None
        return self.policy_proposal.divergence_against_hand_written(
            features, coverage_field, self.cfg, family=family
        )

    def infer_policy(self) -> tuple[Policy, SpatialEFEComponents, list[tuple[Policy, SpatialEFEComponents, float]]]:
        coverage_field = self.belief.coverage(self.cfg.paint_presence_threshold)
        # No training call here: this bare-agent path is monkeypatched in tests
        # with hand-built EFE components, and a gradient step against those would
        # mistrain the proposal on a target the real planner never produced.
        features, _ = self.proposal_belief_features()
        policies = self.policy_sampler.sample(coverage_field, belief_features=features)
        components = self.efe.evaluate_batch(self.belief, policies)
        g = torch.tensor([component.total for component in components], device=self.device)
        believed_coverage = self.belief.material_coverage_mean(self.cfg.paint_presence_threshold)
        log_prior = torch.tensor(
            [
                policy_stop_log_prior(policy, believed_coverage, self.cfg, self.gap_increment)
                for policy in policies
            ],
            device=self.device,
        )
        # Policy precision is the ledger's Gamma posterior mean. No
        # `observe_policy` call here: this bare-agent path has no per-candidate
        # policy-dependent free energy, so gamma stays bit-identically at the
        # declared prior mean rather than pretending a flat F is evidence.
        posterior = policy_posterior_from_efe(
            g,
            log_prior,
            self.precision_ledger.mean(POLICY_PRECISION_KEY),
        )
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

    def add_composition_canvas(self, canvas: SpatialCanvasState) -> None:
        """Store one COMPLETED organized canvas as canvas/relational evidence.

        `add_transition` supplies per-mark local patches to the transition
        likelihood; this supplies a whole finished canvas to the compression
        code, which is the only object a long-range composition code can earn
        anything from. Both halves of the stored pair are the same canvas because
        `_composition_gradient_steps` concatenates state and next_state and never
        reads the action, so the structured canvas is guaranteed to reach the
        model whichever half is sampled.

        Deliberately a distinct method rather than a repurposing of
        `add_transition`: routing whole canvases through that path would cost the
        dynamics ensemble its per-stroke local patches, which it genuinely wants.
        """

        if self.composition is None:
            return
        fields = canvas.flatten_mean()
        # A stop action rasterizes to all zeros, so the stored conditioning
        # fields carry no phantom mark. The canvas/relational half of the
        # composition loss ignores the action entirely; this keeps the buffer's
        # tuple shape valid without inventing an action that never happened.
        action = rasterize_stroke_action(
            StrokeAction.stop_action(),
            canvas.grid_size,
            config=self.cfg,
        ).reshape(-1)
        self.composition_replay.add(fields, action, fields)

    def train_composition(self, gradient_steps: int = 1) -> float | None:
        """Explicit canvas/relational gradient budget.

        A GRADIENT BUDGET, not an objective term: it appears in no expected or
        variational free energy, no policy is ranked by it, and changing it
        cannot change which policy wins at fixed parameters. It exists because
        the compression gap does not discriminate structure at all below a few
        hundred gradient steps, so a bootstrap that wants a measurable gap has to
        declare its budget rather than inherit an incidental one.

        It must NOT call `_train_hierarchy_transitions`: the kind-specific
        transition-EFE gates stay shut until real painting supplies passage
        evidence.
        """

        if self.composition is None or self.composition_optimizer is None:
            return None
        steps = int(gradient_steps)
        if steps <= 0 or len(self.composition_replay) < self.cfg.batch_size:
            return None
        return self._composition_gradient_steps(self._composition_field_shape(), steps)

    @torch.no_grad()
    def composition_gap_for_fields(self, fields: np.ndarray) -> float | None:
        """Mean compression gap over an arbitrary (N, C, H, W) field batch.

        `belief_composition_gap` only ever evaluates the agent's own belief; the
        bootstrap evidence block needs the same declared quantity measured on
        reference probes it supplies itself.
        """

        if self.composition is None:
            return None
        batch = torch.tensor(np.asarray(fields, dtype=np.float32), device=self.device)
        if batch.ndim != 4 or batch.shape[0] == 0:
            return None
        return float(self.composition.compression_gap(batch).mean().item())

    def _composition_field_shape(self) -> tuple[int, int, int]:
        return (
            self.cfg.spatial_material_channels,
            self.cfg.spatial_grid_size,
            self.cfg.spatial_grid_size,
        )

    def _composition_gradient_steps(
        self,
        field_shape: tuple[int, int, int],
        steps: int,
    ) -> float | None:
        """Canvas VFE + relational ELBO descent on the composition replay."""

        if self.composition is None or self.composition_optimizer is None:
            return None
        for _ in range(max(1, int(steps))):
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
        return self.last_composition_loss

    def _train_composition(self) -> None:
        if (
            self.composition is None
            or self.composition_optimizer is None
            or len(self.composition_replay) < self.cfg.batch_size
        ):
            return
        field_shape = self._composition_field_shape()
        self._composition_gradient_steps(
            field_shape, max(1, self.cfg.composition_train_steps)
        )
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
