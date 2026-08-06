from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import threading
import time
from typing import Any, Callable
import warnings

import numpy as np
import torch

from .action_encoding import encoded_action_vector
from .agent import ActiveInferencePainter
from .arm_control import ik_pose_for_canvas_point
from .arm_sim import ArmPainterSim, ArmPose, JOINT_NAMES
from .brush_loading import (
    BrushLoadBelief,
    BrushLoadingModel,
    BrushPreparationInference,
)
from .config import (
    PainterConfig,
    SPATIAL_MATERIAL_PLANNER_STATE_KIND,
    SUMMARY_PLANNER_DEPRECATION,
    SUMMARY_PLANNER_STATE_KIND,
)
from .canvas_hierarchy import PASSAGE_STEP_DESCRIPTOR_DIM
from .camera_inference import CAMERA_SPATIAL_LIKELIHOOD_VERSION
from .camera_observation import CameraObservationBundle
from .efe import EFEComponents
from .env import StrokeAction
from .models import GaussianBelief
from .motor_planning import (
    motor_efe_contribution,
    motor_efe_terms,
    motor_policy_log_prior,
    motor_realization_log_evidence,
    motor_realization_policy_alternatives,
)
from .motion_manifold import (
    MotionManifoldSampler,
    blank_probe_fields,
    iid_scatter_probe_fields,
    shuffled_probe_fields,
)
from .motor_reliability import MotionReliabilityLedger, execution_error_ratio_sq
from .precision_beliefs import (
    POLICY_PRECISION_KEY,
    GapIncrementBelief,
    PrecisionLedger,
)
from .policies import (
    BrushPreparationPolicy,
    MotorPrimitiveLatent,
    PassageLatent,
    PassagePlanLatent,
    Policy,
    policy_stop_log_prior,
)
from .passage_inference import PassageBelief, infer_passage_observation
from .proposal import (
    BASE_TARGET_NAME,
    BELIEF_SOURCE_SUMMARY,
    FALLBACK_BELIEF_SOURCES,
    POLICY_PROPOSAL_VERSION,
    ProposalRecord,
    ProposalTrainingBatch,
    base_efe_policy_posterior,
    hand_written_log_density,
)
from .spatial_agent import SpatialActiveInferencePainter
from .spatial_efe import SpatialEFEComponents
from .spatial_hierarchy import infer_mark_event_belief
from .local_spatial import pixel_material_from_state
from .spatial_state import MATERIAL_CHANNELS, SpatialCanvasState, spatial_canvas_state, spatial_state_diagnostics
from .stroke_execution import (
    ContactAwareStrokeController,
    ExecutionForecast,
    StrokeTiming,
    adaptive_stroke_timing,
    controller_for_motor_primitive,
    forecast_stroke_execution,
    forecast_stroke_executions_batch,
    pose_for_reference,
    rate_limit_pose,
    stroke_world_endpoints,
    stroke_reference,
)

SENSOR_OBSERVATION_BASELINE_ID = "sensor-boundary-v0"
SENSOR_OBSERVATION_ACCESS_MODE = "sensor_equivalent"
ORACLE_OBSERVATION_BASELINE_ID = "baseline-oracle-v0"
ORACLE_OBSERVATION_ACCESS_MODE = "oracle_material_state"

# The unqualified names describe the safe live default.  The oracle constants
# remain available only for explicitly labelled diagnostic fixtures.
OBSERVATION_BASELINE_ID = SENSOR_OBSERVATION_BASELINE_ID
OBSERVATION_ACCESS_MODE = SENSOR_OBSERVATION_ACCESS_MODE


class PrivilegedStateAccessError(RuntimeError):
    """Raised when a sensor-only planner path attempts to read process truth."""


# Bootstrap mark sources. 'random_stroke' is the previous iid source, retained so
# the hand-supplied contribution of the motion-manifold generator stays
# separately measurable, as the charter's attribution clause requires.
BOOTSTRAP_GENERATORS: tuple[str, ...] = ("motion_manifold", "random_stroke")

# Bootstrapped canvases retained for the post-bootstrap compression-gap probe.
# 6 x 16 x 16 float32 is ~6 KB each and the list is dropped as soon as the
# evidence block is built, so the cap only bounds a transient.
MAX_RETAINED_BOOTSTRAP_CANVASES = 16


@dataclass(slots=True)
class StrokeExecution:
    action: StrokeAction
    efe: EFEComponents | SpatialEFEComponents
    posterior: float
    initial_state: np.ndarray | SpatialCanvasState | None = None
    forecast: ExecutionForecast | None = None
    motor_primitive: MotorPrimitiveLatent | None = None
    brush_preparation: BrushPreparationPolicy | None = None
    timing: StrokeTiming = field(default_factory=StrokeTiming)
    controller: ContactAwareStrokeController = field(default_factory=ContactAwareStrokeController)
    initialized: bool = False
    t: float = 0.0
    realized_path_error_sq_sum: float = 0.0
    realized_pressure_error_sq_sum: float = 0.0
    realized_contact_samples: int = 0

    @property
    def approach(self) -> float:
        return self.timing.approach

    @property
    def press(self) -> float:
        return self.timing.press

    @property
    def paint(self) -> float:
        return self.timing.paint

    @property
    def lift(self) -> float:
        return self.timing.lift

    @property
    def total(self) -> float:
        return self.timing.total


@dataclass(slots=True)
class ArmActiveInferenceDriver:
    config: PainterConfig = field(
        default_factory=lambda: PainterConfig(
            canvas_size=64,
            candidate_policies=80,
            planning_horizon=3,
            policy_precision=0.35,
            batch_size=32,
        )
    )
    bootstrap_transitions: int = 96
    bootstrap_train_steps: int = 180
    checkpoint_path: Path | str | None = None
    checkpoint_save_every_transitions: int = 10
    observation_access_mode: str = OBSERVATION_ACCESS_MODE
    enabled: bool = True
    on_stop: Callable[[], None] | None = None
    device: str | None = None
    agent: ActiveInferencePainter | SpatialActiveInferencePainter = field(init=False)
    belief: GaussianBelief | SpatialCanvasState = field(init=False)
    current: StrokeExecution | None = field(default=None, init=False)
    stopped: bool = field(default=False, init=False)
    last_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] = field(default_factory=list, init=False)
    last_components: EFEComponents | SpatialEFEComponents | None = field(default=None, init=False)
    stroke_count: int = field(default=0, init=False)
    trained_transitions: int = field(default=0, init=False)
    last_training_loss: float | None = field(default=None, init=False)
    last_training_seconds: float = field(default=0.0, init=False)
    checkpoint_status: str = field(default="disabled", init=False)
    checkpoint_loaded: bool = field(default=False, init=False)
    checkpoint_last_saved: str | None = field(default=None, init=False)
    checkpoint_last_error: str | None = field(default=None, init=False)
    checkpoint_architecture: dict[str, object] = field(default_factory=dict, init=False)
    last_stop_blocked: bool = field(default=False, init=False)
    last_execution_forecast: ExecutionForecast | None = field(default=None, init=False)
    last_motor_rejections: int = field(default=0, init=False)
    last_motor_primitive_candidates: int = field(default=0, init=False)
    planning: bool = field(default=False, init=False)
    last_planning_seconds: float = field(default=0.0, init=False)
    last_planning_profile: dict[str, object] = field(default_factory=dict, init=False)
    _planning_profile_current: dict[str, object] | None = field(default=None, init=False)
    _planning_forecast_cache: dict[tuple[object, ...], ExecutionForecast] = field(default_factory=dict, init=False)
    _planning_started_at: float | None = field(default=None, init=False)
    _planner_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _planner_thread: threading.Thread | None = field(default=None, init=False)
    _planner_generation: int = field(default=0, init=False)
    _pending_current: StrokeExecution | None = field(default=None, init=False)
    _pending_stopped: bool = field(default=False, init=False)
    _pending_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] | None = field(default=None, init=False)
    _pending_components: EFEComponents | SpatialEFEComponents | None = field(default=None, init=False)
    _pending_passage_queue: tuple[StrokeAction, ...] = field(default_factory=tuple, init=False)
    _pending_passage: PassageLatent | None = field(default=None, init=False)
    _pending_passage_plan: PassagePlanLatent | None = field(default=None, init=False)
    _pending_plan_scope: str = field(default="global", init=False)
    _pending_error: str | None = field(default=None, init=False)
    _transition_to_learn: tuple[
        np.ndarray | SpatialCanvasState,
        StrokeAction,
        MotorPrimitiveLatent | None,
        np.ndarray | SpatialCanvasState,
    ] | None = field(default=None, init=False)
    _post_stroke_retract_remaining: float = field(default=0.0, init=False)
    _hold_pose: ArmPose | None = field(default=None, init=False)
    _hold_command_pose: ArmPose | None = field(default=None, init=False)
    _hold_command_velocity: dict[str, float] = field(default_factory=dict, init=False)
    _hold_scope: str = field(default="global", init=False)
    _passage_queue: list[StrokeAction] = field(default_factory=list, init=False)
    _active_passage: PassageLatent | None = field(default=None, init=False)
    _active_passage_plan: PassagePlanLatent | None = field(default=None, init=False)
    _passage_belief: PassageBelief | None = field(default=None, init=False)
    _active_passage_total_strokes: int = field(default=0, init=False)
    _active_passage_completed_strokes: int = field(default=0, init=False)
    _hierarchy_passage_initial_state: SpatialCanvasState | None = field(default=None, init=False)
    _hierarchy_passage_actions: list[StrokeAction] = field(default_factory=list, init=False)
    _contact_release_count: int = field(default=0, init=False)
    _cached_belief_gap: float | None = field(default=None, init=False)
    _cached_passage_trajectory: dict[str, object] | None = field(default=None, init=False)
    # Amortized-proposal state (Feature D). The batch is built on the planner
    # thread during policy inference and consumed by the post-publish gradient
    # step, so proposal training keeps the pinned "training happens AFTER the plan
    # is published" ordering. The selected-record ring buffer is session-local
    # evidence for the H4 headline number and is deliberately NOT derived from
    # `last_ranked`, which tests hand-build with bare EFE components.
    _pending_proposal_batch: ProposalTrainingBatch | None = field(default=None, init=False)
    _pending_proposal_record: ProposalRecord | None = field(default=None, init=False)
    _pending_proposal_masses: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    _pending_refined_cross_entropy: float | None = field(default=None, init=False)
    _selected_proposal_records: deque = field(
        default_factory=lambda: deque(maxlen=64), init=False
    )
    _cached_policy_proposal: dict[str, object] | None = field(default=None, init=False)
    # Evidence block for the motion-manifold bootstrap. Written ONCE at the end
    # of bootstrap_dynamics and never recomputed: __post_init__'s
    # reset_hierarchy_beliefs and web_runtime's driver reset both re-initialize
    # the persistent hierarchy beliefs from the live blank canvas, so the probe
    # is unrecoverable afterwards.
    _bootstrap_composition: dict[str, object] | None = field(default=None, init=False)
    _bootstrap_episode_canvases: list[np.ndarray] = field(default_factory=list, init=False)
    _bootstrap_episode_coverage: list[float] = field(default_factory=list, init=False)
    _bootstrap_painted_path_length: float = field(default=0.0, init=False)
    _bootstrap_manifold_fallback_marks: int = field(default=0, init=False)
    motion_reliability: MotionReliabilityLedger = field(init=False)
    precision_ledger: PrecisionLedger = field(init=False)
    gap_increment: GapIncrementBelief = field(init=False)
    brush_loading_model: BrushLoadingModel = field(init=False)
    brush_load_beliefs: dict[str, BrushLoadBelief] = field(init=False)
    last_brush_preparation: BrushPreparationInference | None = field(default=None, init=False)
    _observation_boundary_blocked: bool = field(default=False, init=False)
    camera_observation_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.config.planner_state_kind == SUMMARY_PLANNER_STATE_KIND:
            warnings.warn(
                SUMMARY_PLANNER_DEPRECATION,
                FutureWarning,
                stacklevel=2,
            )
        elif self.config.planner_state_kind != SPATIAL_MATERIAL_PLANNER_STATE_KIND:
            raise ValueError(
                "planner_state_kind must be "
                f"{SUMMARY_PLANNER_STATE_KIND!r} or "
                f"{SPATIAL_MATERIAL_PLANNER_STATE_KIND!r}."
            )
        if self.observation_access_mode not in {
            SENSOR_OBSERVATION_ACCESS_MODE,
            ORACLE_OBSERVATION_ACCESS_MODE,
        }:
            raise ValueError(
                "observation_access_mode must be "
                f"{SENSOR_OBSERVATION_ACCESS_MODE!r} or "
                f"{ORACLE_OBSERVATION_ACCESS_MODE!r}."
            )
        self._observation_boundary_blocked = (
            self.observation_access_mode == SENSOR_OBSERVATION_ACCESS_MODE
        )
        if self._observation_boundary_blocked:
            # The camera-conditioned painting posterior now exists. Fail
            # closed until the sensor-conditioned body posterior initializes
            # motor forecasts; neither bootstrap nor live planning may obtain
            # an ArmPainterSim-derived material/body state in the meantime.
            self.enabled = False
        # Learned precision beliefs and the gap-increment belief are owned by the
        # driver and INJECTED into the agent, so the planner and the checkpoint
        # see one shared set of beliefs rather than two divergent copies.
        self.precision_ledger = PrecisionLedger(self.config)
        self.gap_increment = GapIncrementBelief.from_config(self.config)
        if self._uses_spatial_planner():
            self.agent = SpatialActiveInferencePainter(
                self.config,
                seed=17,
                device=self.device,
                precision_ledger=self.precision_ledger,
                gap_increment=self.gap_increment,
            )
        else:
            self.agent = ActiveInferencePainter(
                self.config,
                seed=17,
                device=self.device,
                precision_ledger=self.precision_ledger,
                gap_increment=self.gap_increment,
            )
        self.motion_reliability = MotionReliabilityLedger(self.config)
        self.brush_loading_model = BrushLoadingModel(self.config)
        self.brush_load_beliefs = {
            "white": self.brush_loading_model.unloaded_belief(0.0),
            "black": self.brush_loading_model.unloaded_belief(1.0),
        }
        self.belief = self.agent.belief
        self.checkpoint_architecture = self._checkpoint_architecture_metadata()
        loaded = self._load_checkpoint_if_available()
        if (
            self.bootstrap_transitions > 0
            and not loaded
            and not self._observation_boundary_blocked
        ):
            self.bootstrap_dynamics()
            self._save_checkpoint_if_due(force=True)
        if isinstance(self.agent, SpatialActiveInferencePainter):
            self.agent.reset_hierarchy_beliefs(self.agent.belief)

    def bootstrap_dynamics(self) -> None:
        """Seed the likelihoods from a declared bootstrap mark source.

        Two structural changes over the previous iid bootstrap, both required for
        the composition hierarchy to have anything to learn:

        1. The mark source is `config.bootstrap_generator`. 'motion_manifold'
           draws body-feasible joint-space sweeps (see `motion_manifold`), which
           carry correlated curvature, direction, and length that iid sampling
           cannot produce. 'random_stroke' keeps the previous
           `PolicySampler._stroke` source for attribution.
        2. The canvas is cleared ONLY at episode boundaries. The previous code
           cleared it whenever coverage exceeded 0.94 or on every 24th
           transition regardless of position in the episode, which destroyed each
           canvas before the hierarchy had ever encoded a complete organized one.

        The dynamics ensemble still trains on per-mark local patches exactly as
        before; what changes is that the COMPOSITION replay also receives whole
        finished canvases at each boundary.
        """

        self._require_oracle_diagnostic_mode("bootstrap dynamics")
        generator = str(self.config.bootstrap_generator)
        if generator not in BOOTSTRAP_GENERATORS:
            raise ValueError(
                f"bootstrap_generator must be one of {BOOTSTRAP_GENERATORS}; got {generator!r}."
            )
        sim = ArmPainterSim(replace(self.config))
        sampler = (
            MotionManifoldSampler(
                self.config,
                seed=self.config.bootstrap_manifold_seed,
                kinematics=sim.kinematics,
                canvas=sim.canvas,
            )
            if generator == "motion_manifold"
            else None
        )
        # Bootstrap evidence describes ONE bootstrap run, so the accumulators are
        # reset here rather than at construction: a second call must not report a
        # painted path or fallback count that mixes two runs.
        self._bootstrap_episode_canvases = []
        self._bootstrap_episode_coverage = []
        self._bootstrap_painted_path_length = 0.0
        self._bootstrap_manifold_fallback_marks = 0
        episode_marks = max(1, int(self.config.bootstrap_episode_marks))
        pending: list[tuple[PassageLatent | None, StrokeAction, int]] = []
        episode_start = self._planner_state(sim)
        episode_actions: list[StrokeAction] = []
        for i in range(self.bootstrap_transitions):
            if i > 0 and i % episode_marks == 0:
                self._close_bootstrap_episode(sim, episode_start, episode_actions)
                # An episode boundary invalidates the pending queue: its marks
                # were proposed against the canvas and pose that are about to be
                # discarded.
                pending.clear()
                if sampler is not None:
                    sampler.reset_chain()
                sim.reset_pose()
                sim.canvas.clear()
                episode_actions = []
                episode_start = self._planner_state(sim)
            passage, action, step_index = self._next_bootstrap_mark(sampler, pending)
            state = self._planner_state(sim)
            execute_stroke_action(sim, action, dt=1.0 / 90.0)
            next_state = self._planner_state(sim)
            self._add_transition_to_agent(state, action, next_state)
            episode_actions.append(action)
            self._bootstrap_painted_path_length += float(
                np.hypot(action.x1 - action.x0, action.y1 - action.y0)
            )
            if (
                passage is not None
                and self.config.bootstrap_feeds_passage_likelihood
                and isinstance(self.agent, SpatialActiveInferencePainter)
                and isinstance(state, SpatialCanvasState)
                and isinstance(next_state, SpatialCanvasState)
            ):
                self.agent.add_passage_step_transition(state, passage, step_index, next_state)
            self.trained_transitions += 1
            if len(self.agent.replay) >= self.config.batch_size and i % 4 == 0:
                self.last_training_loss = self.agent.train_dynamics(gradient_steps=2)
        self._close_bootstrap_episode(sim, episode_start, episode_actions)
        if len(self.agent.replay) >= self.config.batch_size:
            self.last_training_loss = self.agent.train_dynamics(gradient_steps=self.bootstrap_train_steps)
        self._record_bootstrap_composition_evidence(generator, sampler)

    def _next_bootstrap_mark(
        self,
        sampler: MotionManifoldSampler | None,
        pending: list[tuple[PassageLatent | None, StrokeAction, int]],
    ) -> tuple[PassageLatent | None, StrokeAction, int]:
        """Next bootstrap mark from the declared generator.

        The iid source stays genuinely reachable, not merely declared: with
        `bootstrap_generator='random_stroke'` this is exactly the previous call.
        """

        if sampler is None:
            return None, self.agent.policy_sampler._stroke(), 0
        if not pending:
            sampled = sampler.sample_marks()
            if sampled is not None:
                latent, actions = sampled
                pending.extend(
                    (latent, action, step_index)
                    for step_index, action in enumerate(actions)
                )
        if pending:
            return pending.pop(0)
        # The sweep sampler could not find a legal path from the current chain
        # pose. Fall back to the iid source so bootstrap never stalls, and count
        # it, so a run whose "manifold" evidence is really iid is visible.
        self._bootstrap_manifold_fallback_marks += 1
        return None, self.agent.policy_sampler._stroke(), 0

    def _close_bootstrap_episode(
        self,
        sim: ArmPainterSim,
        episode_start: np.ndarray | SpatialCanvasState,
        episode_actions: list[StrokeAction],
    ) -> None:
        """Hand one COMPLETED bootstrap canvas to the canvas/relational likelihood."""

        if (
            not isinstance(self.agent, SpatialActiveInferencePainter)
            or self.agent.composition is None
        ):
            return
        final = self._planner_state(sim)
        if not isinstance(final, SpatialCanvasState):
            return
        self.agent.add_composition_canvas(final)
        if len(self._bootstrap_episode_canvases) < MAX_RETAINED_BOOTSTRAP_CANVASES:
            self._bootstrap_episode_canvases.append(final.material.copy())
        # `_state_coverage` is retained here now that the coverage-triggered
        # mid-episode clear is gone: it records how far each episode filled the
        # canvas, which is what makes the A/B's paint budget auditable.
        self._bootstrap_episode_coverage.append(self._state_coverage(final))
        if (
            self.config.bootstrap_feeds_passage_likelihood
            and episode_actions
            and isinstance(episode_start, SpatialCanvasState)
        ):
            self.agent.add_passage_transition(episode_start, tuple(episode_actions), final)
        budget = int(self.config.bootstrap_composition_train_steps)
        if budget > 0:
            self.agent.train_composition(gradient_steps=budget)

    def _record_bootstrap_composition_evidence(
        self,
        generator: str,
        sampler: MotionManifoldSampler | None = None,
    ) -> None:
        """Measure the bootstrapped canvas/relational likelihood, once.

        EVIDENCE ONLY. No expected-free-energy term, variational free-energy
        term, preference, precision belief, policy prior, or policy posterior
        reads any value in this block; it is consumed only by the web UI and by a
        human reading an attribution A/B. Everything stored is a plain
        float/int/str/bool/None so the whole diagnostics dict stays JSON
        serializable, and the probe canvases themselves are never embedded.
        """

        canvases = self._bootstrap_episode_canvases
        self._bootstrap_episode_canvases = []
        if (
            not isinstance(self.agent, SpatialActiveInferencePainter)
            or self.agent.composition is None
        ):
            return
        coverage = list(self._bootstrap_episode_coverage)
        rng = np.random.default_rng(int(self.config.bootstrap_manifold_seed) + 1)
        gaps: dict[str, float | None] = {
            "bootstrapped": None,
            "blank": None,
            "shuffledBootstrapped": None,
            "iidScatter": None,
        }
        if canvases:
            bootstrapped = np.stack(canvases).astype(np.float32)
            gaps["bootstrapped"] = self.agent.composition_gap_for_fields(bootstrapped)
            gaps["shuffledBootstrapped"] = self.agent.composition_gap_for_fields(
                shuffled_probe_fields(bootstrapped, rng)
            )
        gaps["blank"] = self.agent.composition_gap_for_fields(
            blank_probe_fields(self.config)
        )
        gaps["iidScatter"] = self.agent.composition_gap_for_fields(
            iid_scatter_probe_fields(self.config, rng)
        )
        observed = [value for value in gaps.values() if value is not None]
        null_models = [
            gaps[key] for key in ("blank", "shuffledBootstrapped") if gaps[key] is not None
        ]
        margin: float | None = None
        if gaps["bootstrapped"] is not None and null_models:
            margin = float(gaps["bootstrapped"] - max(null_models))
        # Sweep geometry summary. Present only for the manifold arm, so a
        # random_stroke run cannot be mistaken for one that produced sweeps.
        sweep_statistics: dict[str, object] = {
            "manifoldSweepCount": None,
            "manifoldRejectedSweeps": None,
            "meanSweepTurnRadians": None,
            "meanSweepPathChordRatio": None,
            "meanSweepNormalizedPathLength": None,
        }
        if sampler is not None:
            statistics = sampler.statistics()
            sweep_statistics = {
                "manifoldSweepCount": int(statistics["sweepCount"]),
                "manifoldRejectedSweeps": int(statistics["rejectedSweeps"]),
                "meanSweepTurnRadians": float(statistics["meanSweepTurnRadians"]),
                "meanSweepPathChordRatio": float(statistics["meanSweepPathChordRatio"]),
                "meanSweepNormalizedPathLength": float(
                    statistics["meanSweepNormalizedPathLength"]
                ),
            }
        block: dict[str, object] = {
            "configuredGenerator": str(self.config.bootstrap_generator),
            "executedGenerator": generator,
            "episodes": len(coverage),
            "markCount": int(self.bootstrap_transitions),
            "episodeMarks": int(self.config.bootstrap_episode_marks),
            "canvasSize": int(self.config.canvas_size),
            "episodeCoverageMean": (
                float(np.mean(coverage)) if coverage else None
            ),
            "paintedPathLength": float(self._bootstrap_painted_path_length),
            "compositionTrainSteps": int(self.config.bootstrap_composition_train_steps),
            "passageLikelihoodFed": bool(self.config.bootstrap_feeds_passage_likelihood),
            "manifoldFallbackMarks": int(self._bootstrap_manifold_fallback_marks),
            "gap": gaps,
            # The DECLARED acceptance criterion, bounded by the TIGHTER of the two
            # null models. The cell shuffle preserves every per-channel marginal
            # exactly, so the flat baseline member is identical between a canvas
            # and its shuffle and the difference isolates the hierarchical code.
            # MEASURED: once the model is well trained the shuffle becomes
            # strongly out-of-distribution (-74 nats at 2700 gradient steps) and
            # blank (-0.05) is the tighter null, so the max() is load-bearing --
            # a shuffle-only criterion would report a much larger margin earned
            # by overconfidence rather than by discrimination.
            "discriminativeMargin": margin,
            # The raw probe spread, reported because it was the originally
            # requested metric. It is NOT the criterion: it grows when the model
            # becomes more confidently negative about an out-of-distribution
            # probe, i.e. with confidence rather than with validity.
            "probeRange": (
                float(max(observed) - min(observed)) if len(observed) > 1 else None
            ),
            "declaredAs": (
                "evidence only: a measurement of the canvas/relational likelihood after "
                "embodiment-driven bootstrap. NOT a decision quantity - no EFE term, VFE "
                "term, preference, precision belief, or policy prior reads it."
            ),
            "approximation": (
                "gap probed at bootstrap canvas_size="
                f"{int(self.config.canvas_size)} on "
                f"{int(self.config.spatial_material_channels)}x"
                f"{int(self.config.spatial_grid_size)}x"
                f"{int(self.config.spatial_grid_size)} fields; the live sim paints at a "
                "different resolution, so mark-to-cell scale differs. Earnable only on "
                "channels 3-5 (SIGMA_FLOOR 0.02 saturates the flat code on thickness-like "
                "channels), and channels 4-5 are deterministic functions of 0-3."
            ),
        }
        block.update(sweep_statistics)
        self._bootstrap_composition = block

    def _learned_proposal_architecture(self) -> dict[str, object]:
        """Shape metadata of the amortized proposal, for the architecture dict.

        `learned_proposal_mix` is DELIBERATELY absent. It is a runtime budget
        split, not a shape: putting it in a dict compared with exact equality
        would reject every checkpoint on disk the moment the mix was ramped, which
        is precisely the experiment the mixture weight exists to allow.
        """

        cfg = self.config
        enabled = bool(
            self._uses_spatial_planner()
            and cfg.learned_proposal_enabled
            and cfg.composition_enabled
            and cfg.composition_gap_precision > 0.0
        )
        latent_grid = max(1, cfg.spatial_grid_size // 4)
        input_dim = int(cfg.canvas_latent_channels * latent_grid * latent_grid) + int(
            cfg.relational_latent_dim
        )
        proposal = getattr(self.agent, "policy_proposal", None)
        return {
            "learned_proposal_enabled": enabled,
            "learned_proposal_version": POLICY_PROPOSAL_VERSION,
            "learned_proposal_hidden_dim": int(cfg.learned_proposal_hidden_dim),
            "learned_proposal_input_dim": input_dim,
            "learned_proposal_output_dim": (
                int(proposal.output_dim) if proposal is not None else 0
            ),
        }

    def _checkpoint_architecture_metadata(self) -> dict[str, object]:
        cfg = self.config
        return {
            "schema_version": 4,
            "agent_kind": "spatial_material" if self._uses_spatial_planner() else "summary",
            "observation_access_mode": self.observation_access_mode,
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "planner_state_kind": cfg.planner_state_kind,
            "spatial_grid_size": cfg.spatial_grid_size,
            "material_pyramid_levels": tuple(cfg.material_pyramid_levels),
            "spatial_material_channels": cfg.spatial_material_channels,
            "spatial_action_channels": cfg.spatial_action_channels,
            "spatial_transition_mode": cfg.spatial_transition_mode,
            "spatial_hidden_channels": cfg.spatial_hidden_channels,
            "spatial_residual_blocks": cfg.spatial_residual_blocks,
            "spatial_ensemble_size": cfg.spatial_ensemble_size,
            "ensemble_size": cfg.ensemble_size,
            "hidden_dim": cfg.hidden_dim,
            # DECLARED STRUCTURE only. `composition_enabled` is a constant and
            # `composition_gap_precision` is read as a declared constant here, never
            # as the learned belief mean -- a learned quantity in this dict would
            # invalidate every checkpoint on disk the moment it moved.
            "composition_enabled": bool(
                self._uses_spatial_planner()
                and cfg.composition_enabled
                and cfg.composition_gap_precision > 0.0
            ),
            "composition_latent_dim": cfg.composition_latent_dim,
            "composition_hidden_channels": cfg.composition_hidden_channels,
            "canvas_latent_channels": cfg.canvas_latent_channels,
            "relational_latent_dim": cfg.relational_latent_dim,
            "hierarchy_hidden_dim": cfg.hierarchy_hidden_dim,
            "passage_trajectory_enabled": cfg.passage_trajectory_enabled,
            "passage_step_descriptor_dim": PASSAGE_STEP_DESCRIPTOR_DIM,
            "motor_realization_kinds": tuple(cfg.motor_realization_kinds),
            "motor_roll_sweep_degrees": cfg.motor_roll_sweep_degrees,
            "thickness_scale": cfg.thickness_scale,
            "paint_presence_threshold": cfg.paint_presence_threshold,
            "canvas_ground_tone": cfg.canvas_ground_tone,
            "camera_spatial_likelihood_version": CAMERA_SPATIAL_LIKELIHOOD_VERSION,
            **self._learned_proposal_architecture(),
        }

    def _checkpoint_file(self) -> Path | None:
        if self.checkpoint_path is None:
            self.checkpoint_status = "disabled"
            return None
        return Path(self.checkpoint_path)

    def _load_checkpoint_if_available(self) -> bool:
        path = self._checkpoint_file()
        if path is None:
            return False
        if not path.is_file():
            self.checkpoint_status = "not_found"
            return False
        try:
            try:
                payload = torch.load(path, map_location=self.agent.device, weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location=self.agent.device)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload is not a dictionary")
            expected = self._checkpoint_architecture_metadata()
            found = payload.get("architecture")
            if found != expected:
                self.checkpoint_status = "incompatible"
                self.checkpoint_last_error = self._checkpoint_mismatch_summary(found, expected)
                return False
            self.agent.dynamics.load_state_dict(payload["dynamics_state"])
            if "optimizer_state" in payload:
                self.agent.optimizer.load_state_dict(payload["optimizer_state"])
            self._restore_replay(self.agent.replay, payload.get("replay"))
            if (
                isinstance(self.agent, SpatialActiveInferencePainter)
                and self.agent.composition is not None
                and payload.get("composition_state") is not None
            ):
                self.agent.composition.load_state_dict(payload["composition_state"])
                if (
                    self.agent.composition_optimizer is not None
                    and payload.get("composition_optimizer_state") is not None
                ):
                    self.agent.composition_optimizer.load_state_dict(payload["composition_optimizer_state"])
                self.agent.last_composition_loss = payload.get("last_composition_loss")
                self.agent.last_hierarchy_transition_loss = payload.get("last_hierarchy_transition_loss")
                self.agent.last_passage_trajectory_loss = payload.get("last_passage_trajectory_loss")
                self.agent.last_passage_trajectory_evaluation = payload.get(
                    "last_passage_trajectory_evaluation"
                )
            if (
                isinstance(self.agent, SpatialActiveInferencePainter)
                and self.agent.policy_proposal is not None
                and payload.get("proposal_state") is not None
            ):
                # `proposal_update_count` is a registered buffer, so it rides
                # inside `state_dict` and round-trips without a separate key.
                self.agent.policy_proposal.load_state_dict(payload["proposal_state"])
                proposal_generator_state = payload.get("proposal_generator_state")
                if isinstance(proposal_generator_state, torch.Tensor):
                    # The proposal owns a private CPU generator so its candidate
                    # stream cannot perturb PolicySampler or global torch RNG.
                    # It is learned-run state just as surely as optimizer moments:
                    # resume at the next draw, not back at the initial seed.
                    self.agent.policy_proposal.generator.set_state(
                        proposal_generator_state.detach().cpu()
                    )
                if (
                    self.agent.policy_proposal_optimizer is not None
                    and payload.get("proposal_optimizer_state") is not None
                ):
                    self.agent.policy_proposal_optimizer.load_state_dict(
                        payload["proposal_optimizer_state"]
                    )
                self.agent.last_proposal_loss = payload.get("last_proposal_loss")
            if isinstance(self.agent, SpatialActiveInferencePainter):
                self._restore_replay(self.agent.composition_replay, payload.get("composition_replay"))
                self._restore_replay(self.agent.passage_replay, payload.get("passage_replay"))
                self._restore_replay(self.agent.passage_step_replay, payload.get("passage_step_replay"))
                kind_support = payload.get("passage_kind_update_counts")
                if self.agent.composition is not None and isinstance(kind_support, dict):
                    self.agent.composition.passage_kind_update_counts = {
                        kind: max(0, int(kind_support.get(kind, 0)))
                        for kind in ("band", "chain", "polyline")
                    }
                else:
                    self.agent.rebuild_passage_kind_support()
            self.trained_transitions = int(payload.get("trained_transitions", 0))
            self.last_training_loss = payload.get("last_training_loss")
            self.motion_reliability.restore(payload.get("motion_reliability"))
            # Learned state, restored best effort: a pre-Feature-C checkpoint
            # lacks both keys and must still load with status 'loaded' and
            # fresh beliefs, so these are NOT architecture metadata.
            self.precision_ledger.restore(payload.get("precision_ledger"))
            self.gap_increment.restore(payload.get("gap_increment_belief"))
            self.checkpoint_loaded = True
            self.checkpoint_status = "loaded"
            self.checkpoint_last_error = None
            return True
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            self.checkpoint_status = "load_failed"
            self.checkpoint_last_error = repr(exc)
            return False

    def _save_checkpoint_if_due(self, *, force: bool = False) -> None:
        path = self._checkpoint_file()
        if path is None:
            return
        if self.checkpoint_status in {"incompatible", "load_failed"} and not self.checkpoint_loaded:
            return
        interval = max(1, int(self.checkpoint_save_every_transitions))
        if not force and self.trained_transitions % interval != 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {
                "schema_version": 4,
                "architecture": self._checkpoint_architecture_metadata(),
                "config": asdict(self.config),
                "trained_transitions": self.trained_transitions,
                "last_training_loss": self.last_training_loss,
                "dynamics_state": self.agent.dynamics.state_dict(),
                "optimizer_state": self.agent.optimizer.state_dict(),
                "replay": self._replay_snapshot(self.agent.replay),
                "motion_reliability": self.motion_reliability.snapshot(),
                "precision_ledger": self.precision_ledger.snapshot(),
                "gap_increment_belief": self.gap_increment.snapshot(),
            }
            if isinstance(self.agent, SpatialActiveInferencePainter):
                payload["composition_replay"] = self._replay_snapshot(self.agent.composition_replay)
                payload["passage_replay"] = self._replay_snapshot(self.agent.passage_replay)
                payload["passage_step_replay"] = self._replay_snapshot(self.agent.passage_step_replay)
                payload["last_composition_loss"] = self.agent.last_composition_loss
                payload["last_hierarchy_transition_loss"] = self.agent.last_hierarchy_transition_loss
                payload["last_passage_trajectory_loss"] = self.agent.last_passage_trajectory_loss
                payload["last_passage_trajectory_evaluation"] = (
                    self.agent.last_passage_trajectory_evaluation
                )
                if self.agent.composition is not None:
                    payload["composition_state"] = self.agent.composition.state_dict()
                    payload["passage_kind_update_counts"] = dict(
                        self.agent.composition.passage_kind_update_counts
                    )
                if self.agent.composition_optimizer is not None:
                    payload["composition_optimizer_state"] = self.agent.composition_optimizer.state_dict()
                if self.agent.policy_proposal is not None:
                    payload["proposal_state"] = self.agent.policy_proposal.state_dict()
                    payload["proposal_generator_state"] = (
                        self.agent.policy_proposal.generator.get_state()
                    )
                    payload["last_proposal_loss"] = self.agent.last_proposal_loss
                    # Reporting only, never compared: see
                    # `_learned_proposal_architecture`.
                    payload["learned_proposal_mix"] = float(self.config.learned_proposal_mix)
                if self.agent.policy_proposal_optimizer is not None:
                    payload["proposal_optimizer_state"] = (
                        self.agent.policy_proposal_optimizer.state_dict()
                    )
            temp_path = path.with_name(f"{path.name}.tmp")
            torch.save(payload, temp_path)
            temp_path.replace(path)
            self.checkpoint_status = "saved"
            self.checkpoint_last_saved = str(path)
            self.checkpoint_last_error = None
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            self.checkpoint_status = "save_failed"
            self.checkpoint_last_error = repr(exc)

    @staticmethod
    def _replay_snapshot(replay: object) -> dict[str, object] | None:
        data = getattr(replay, "data", None)
        if data is None:
            return None
        return {
            "data": list(data),
            "maxlen": getattr(data, "maxlen", None),
        }

    @staticmethod
    def _restore_replay(replay: object, snapshot: object) -> None:
        if not isinstance(snapshot, dict) or not hasattr(replay, "data"):
            return
        data = getattr(replay, "data")
        maxlen = getattr(data, "maxlen", None)
        restored = list(snapshot.get("data", []))
        data.clear()
        if maxlen is not None:
            restored = restored[-int(maxlen):]
        data.extend(restored)

    @staticmethod
    def _checkpoint_mismatch_summary(found: object, expected: dict[str, object]) -> str:
        if not isinstance(found, dict):
            return "checkpoint has no architecture metadata"
        changed = [
            f"{key}: checkpoint={found.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if found.get(key) != value
        ]
        extra = [f"{key}: checkpoint={value!r}, current=<missing>" for key, value in found.items() if key not in expected]
        details = changed + extra
        if not details:
            return "architecture metadata differs"
        # Raised from 6 so the newer architecture keys cannot crowd an older
        # mismatch line out of the message a reader (or a test) greps for.
        shown = details[:12]
        summary = "; ".join(shown)
        if len(details) > len(shown):
            summary = f"{summary}; (+{len(details) - len(shown)} more)"
        return summary

    def reset(self, sim: ArmPainterSim) -> None:
        with self._planner_lock:
            self._planner_generation += 1
            self.current = None
            self.stopped = False
            self.stroke_count = 0
            self.last_ranked = []
            self.last_components = None
            self.last_stop_blocked = False
            self.last_execution_forecast = None
            self.last_motor_rejections = 0
            self.last_motor_primitive_candidates = 0
            self.planning = False
            self.last_planning_profile = {}
            self._planning_profile_current = None
            self._planning_forecast_cache = {}
            self._planning_started_at = None
            self._pending_current = None
            self._pending_stopped = False
            self._pending_ranked = None
            self._pending_components = None
            self._pending_passage_queue = ()
            self._pending_passage = None
            self._pending_passage_plan = None
            self._pending_plan_scope = "global"
            self._pending_error = None
            self._transition_to_learn = None
            self._post_stroke_retract_remaining = 0.0
            self._hold_pose = None
            self._hold_command_pose = None
            self._hold_command_velocity = {}
            self._hold_scope = "global"
            self._passage_queue = []
            self._active_passage = None
            self._active_passage_plan = None
            self._passage_belief = None
            self._active_passage_total_strokes = 0
            self._active_passage_completed_strokes = 0
            self._hierarchy_passage_initial_state = None
            self._hierarchy_passage_actions = []
            self._contact_release_count = 0
            self._cached_belief_gap = None
            self._cached_passage_trajectory = None
            self.brush_load_beliefs = {
                "white": self.brush_loading_model.unloaded_belief(0.0),
                "black": self.brush_loading_model.unloaded_belief(1.0),
            }
            self.last_brush_preparation = None
            self.camera_observation_count = 0
        if self._observation_boundary_blocked:
            # Do not even touch the process object from the model-facing reset
            # path.  A separate execution/runtime reset owns plant mutations.
            return
        sim.unload_all_brushes()
        state = self._observe(sim)
        if isinstance(self.agent, SpatialActiveInferencePainter) and isinstance(state, SpatialCanvasState):
            self.agent.reset_hierarchy_beliefs(state)

    def _observe(self, sim: ArmPainterSim) -> GaussianBelief | SpatialCanvasState:
        self._require_oracle_diagnostic_mode("planner observation")
        state = self._planner_state(sim)
        self._reset_agent_belief(state)
        self.belief = self.agent.belief
        return self.belief

    def _uses_spatial_planner(self) -> bool:
        return (
            self.config.planner_state_kind
            == SPATIAL_MATERIAL_PLANNER_STATE_KIND
        )

    def _planner_state(self, sim: ArmPainterSim) -> np.ndarray | SpatialCanvasState:
        self._require_oracle_diagnostic_mode("planner state construction")
        if self._uses_spatial_planner():
            return spatial_canvas_state(sim, self.config)
        return canvas_summary_state(sim)

    @property
    def observation_boundary_blocked(self) -> bool:
        return self._observation_boundary_blocked

    def ingest_camera_observation(
        self,
        observation: CameraObservationBundle,
    ) -> SpatialCanvasState:
        """Assimilate permitted image products without dereferencing process truth.

        This completes the painting-state camera likelihood boundary. The
        full sensor-equivalent controller remains fail-closed until its body
        posterior replaces exact simulator initialization in motor forecasts.
        """

        if not isinstance(self.agent, SpatialActiveInferencePainter):
            raise RuntimeError(
                "camera likelihood requires planner_state_kind='spatial_material'"
            )
        posterior = self.agent.assimilate_camera_observation(observation)
        self.belief = posterior
        self.camera_observation_count += sum(
            factor.observed_cell_count > 0
            for factor in (self.agent.last_camera_vfe.factors if self.agent.last_camera_vfe else ())
        )
        return posterior

    def _require_oracle_diagnostic_mode(self, operation: str) -> None:
        if self._observation_boundary_blocked:
            raise PrivilegedStateAccessError(
                f"{operation} requires hidden simulator state and is denied in "
                f"{SENSOR_OBSERVATION_ACCESS_MODE!r} mode. Use the explicit "
                f"{ORACLE_OBSERVATION_ACCESS_MODE!r} diagnostic mode only for "
                "labelled oracle fixtures."
            )

    def _state_coverage(self, state: np.ndarray | SpatialCanvasState) -> float:
        if isinstance(state, SpatialCanvasState):
            return state.material_coverage_mean(self.config.paint_presence_threshold)
        return float(state[0])

    @staticmethod
    def _brush_key(tone: float) -> str:
        return "black" if tone >= 0.5 else "white"

    def _infer_brush_preparation(
        self,
        action: StrokeAction,
    ) -> BrushPreparationInference:
        """Infer an instantaneous preparation policy from q(brush load)."""

        key = self._brush_key(action.tone)
        return self.brush_loading_model.infer_preparation(
            self.brush_load_beliefs[key],
            action,
        )

    def _reset_agent_belief(self, state: np.ndarray | SpatialCanvasState) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            self.agent.reset_belief(state)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            self.agent.reset_belief(state)

    def _update_agent_belief(
        self,
        action: StrokeAction,
        state: np.ndarray | SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            self.agent.update_belief(action, state, motor_primitive)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            self.agent.update_belief(action, state, motor_primitive)

    def _add_transition_to_agent(
        self,
        state: np.ndarray | SpatialCanvasState,
        action: StrokeAction,
        next_state: np.ndarray | SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            assert isinstance(next_state, SpatialCanvasState)
            self.agent.add_transition(state, action, next_state, motor_primitive)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            assert isinstance(next_state, np.ndarray)
            self.agent.replay.add(state, encoded_action_vector(action, self.config, motor_primitive), next_state)

    def step(self, sim: ArmPainterSim, dt: float) -> None:
        if self._observation_boundary_blocked:
            # Sensor-equivalent perception is not implemented yet.  Returning
            # before dereferencing ``sim`` guarantees that policy inference,
            # learning, and planning cannot silently fall back to process
            # truth while that work is pending.
            return
        if not self.enabled or self.stopped:
            self._hold_retracted(sim, dt, scope="global")
            return
        if self._post_stroke_retract_remaining > 0.0:
            hold_scope = "passage" if self._passage_queue else "global"
            self._hold_retracted(sim, dt, scope=hold_scope)
            self._post_stroke_retract_remaining = max(0.0, self._post_stroke_retract_remaining - dt)
            if self._post_stroke_retract_remaining <= 0.0 and self._passage_queue:
                self._consume_background_plan()
            return
        if self._consume_background_plan():
            self._hold_retracted(sim, dt, scope="global")
            return
        if self.current is None:
            if self._passage_queue:
                if self.planning:
                    self._hold_retracted(sim, dt, scope="passage")
                    return
                if self._passage_belief is None:
                    self._start_next_passage_stroke(sim)
                elif self._pending_error is None:
                    self._start_local_passage_plan(sim)
                    self._hold_retracted(sim, dt, scope="passage")
                else:
                    self._start_next_passage_stroke(sim)
                return
            self._hold_retracted(sim, dt, scope="global")
            if self._global_retraction_ready(sim):
                self._start_background_plan(sim)
            return
        self._hold_pose = None
        self._execute_current(sim, dt)

    def _hold_retracted(self, sim: ArmPainterSim, dt: float, *, scope: str) -> None:
        sim.intended_contact_pressure = 0.0
        sim.control_damping_multiplier = max(1.0, float(self.config.hold_damping_multiplier))
        contact_escape = self._contact_escape_pose(sim, scope)
        if contact_escape is not None:
            self._hold_scope = scope
            desired = contact_escape
        else:
            if self._hold_pose is None or self._hold_scope != scope:
                self._hold_scope = scope
                self._hold_pose = self._passage_hold_pose(sim) if scope == "passage" else self._global_hold_pose(sim)
                self._hold_command_pose = sim.target_pose
                self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
            desired = self._hold_pose
        self._apply_contact_release(sim, desired, dt)
        sim.set_target(self._shaped_hold_target(sim, desired, dt))

    def _shaped_hold_target(self, sim: ArmPainterSim, desired: ArmPose, dt: float) -> ArmPose:
        if self._hold_command_pose is None:
            self._hold_command_pose = sim.target_pose
            self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        dt_eff = max(float(dt), 1.0 / 240.0)
        max_speed = max(1.0, float(self.config.hold_target_joint_speed_deg_s))
        max_accel = max(1.0, float(self.config.hold_target_joint_accel_deg_s2))
        values: dict[str, float] = {}
        for name in JOINT_NAMES:
            current = float(getattr(self._hold_command_pose, name))
            target = float(getattr(desired, name))
            error = target - current
            old_velocity = float(self._hold_command_velocity.get(name, 0.0))
            if abs(error) < 1e-6:
                velocity = 0.0
                value = target
            else:
                braking_speed = float(np.sqrt(max(0.0, 2.0 * max_accel * abs(error))))
                desired_velocity = float(np.sign(error) * min(max_speed, braking_speed))
                velocity_delta = float(np.clip(desired_velocity - old_velocity, -max_accel * dt_eff, max_accel * dt_eff))
                velocity = float(np.clip(old_velocity + velocity_delta, -max_speed, max_speed))
                step = velocity * dt_eff
                if abs(step) >= abs(error):
                    value = target
                    velocity = 0.0
                else:
                    value = current + step
            values[name] = value
            self._hold_command_velocity[name] = velocity
        self._hold_command_pose = ArmPose(**values).clipped()
        return self._hold_command_pose

    def _apply_contact_release(self, sim: ArmPainterSim, desired: ArmPose, dt: float) -> None:
        threshold = max(0.0, float(self.config.contact_release_pressure_threshold))
        if sim.contact.pressure <= threshold:
            return
        tip = sim.kinematics.tip(sim.actual_pose)
        current_overtravel = sim.canvas.overtravel_depth(tip)
        max_delta = max(82.0, float(self.config.contact_release_joint_speed_deg_s)) * max(float(dt), 1.0 / 240.0)
        release_pose = rate_limit_pose(desired, sim.actual_pose, max_delta=max_delta).clipped()
        release_tip = sim.kinematics.tip(release_pose)
        release_overtravel = sim.canvas.overtravel_depth(release_tip)
        if release_overtravel > current_overtravel and float(release_tip[1]) >= float(tip[1]):
            return
        sim.actual_pose = release_pose
        sim.target_pose = release_pose
        self._hold_command_pose = release_pose
        self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        sim.plant.reset_state(sim.actual_pose)
        sim.intended_contact_pressure = 0.0
        sim.refresh_contact()
        self._contact_release_count += 1

    def _contact_escape_pose(self, sim: ArmPainterSim, scope: str) -> ArmPose | None:
        tip = sim.kinematics.tip(sim.actual_pose)
        depth = (
            self.config.local_passage_retract_depth
            if scope == "passage"
            else self.config.global_planning_retract_depth
        )
        required_clearance = max(0.35, 0.5 * depth)
        if scope == "global":
            required_clearance = max(
                required_clearance,
                0.5,
                float(self.config.global_planning_clearance_fraction) * depth,
            )
        near_contact = sim.contact.on_canvas and (
            sim.contact.pressure > 0.01
            or float(tip[1]) > sim.canvas.distance - 0.08
        )
        escape_in_progress = self._hold_pose is None or self._hold_scope != scope
        if not near_contact and not escape_in_progress:
            return None
        # Continue the approximately Cartesian straight-back escape after
        # contact releases. Otherwise joint-space interpolation toward the low
        # camera-clear park can arc the tip back toward the canvas before
        # sufficient normal clearance has been established.
        if (
            sim.contact.pressure <= 0.01
            and float(tip[1]) <= sim.canvas.distance - required_clearance
        ):
            return None
        lateral_limit = 0.46 * min(sim.canvas.width, sim.canvas.height)
        x = float(np.clip(tip[0], -lateral_limit, lateral_limit))
        z = float(np.clip(tip[2], -lateral_limit, lateral_limit))
        if not np.isfinite(x) or not np.isfinite(z):
            x, z = self._active_passage_world_center(sim) if scope == "passage" else (0.0, 0.0)
        pose = ik_pose_for_canvas_point(x, z, sim.canvas.distance - depth)
        tip = sim.kinematics.tip(pose)
        if (
            not np.all(np.isfinite(tip))
            or float(tip[1]) > sim.canvas.distance - required_clearance
        ):
            return self._passage_hold_pose(sim) if scope == "passage" else self._global_hold_pose(sim)
        return pose

    def _global_hold_pose(self, sim: ArmPainterSim) -> ArmPose:
        x = float(self.config.global_planning_park_x_fraction) * sim.canvas.width
        z = float(self.config.global_planning_park_z_fraction) * sim.canvas.height
        return ik_pose_for_canvas_point(x, z, sim.canvas.distance - self.config.global_planning_retract_depth)

    def _global_retraction_ready(self, sim: ArmPainterSim) -> bool:
        tip = sim.kinematics.tip(sim.actual_pose)
        required_clearance = max(
            0.5,
            float(self.config.global_planning_clearance_fraction)
            * float(self.config.global_planning_retract_depth),
        )
        return bool(
            np.all(np.isfinite(tip))
            and float(tip[1]) <= sim.canvas.distance - required_clearance
            and sim.contact.pressure <= self.config.contact_release_pressure_threshold
        )

    def _passage_hold_pose(self, sim: ArmPainterSim) -> ArmPose:
        x, z = self._active_passage_world_center(sim)
        return ik_pose_for_canvas_point(x, z, sim.canvas.distance - self.config.local_passage_retract_depth)

    def _active_passage_world_center(self, sim: ArmPainterSim) -> tuple[float, float]:
        if self._active_passage is not None:
            x = (self._active_passage.center_x - 0.5) * sim.canvas.width * 0.98
            z = (0.5 - self._active_passage.center_y) * sim.canvas.height * 0.98
            return float(x), float(z)
        if self._active_passage_plan is not None:
            x = (self._active_passage_plan.center_x - 0.5) * sim.canvas.width * 0.98
            z = (0.5 - self._active_passage_plan.center_y) * sim.canvas.height * 0.98
            return float(x), float(z)
        actions = self._passage_queue
        if actions:
            centers: list[tuple[float, float]] = []
            for action in actions[: max(1, min(3, len(actions)))]:
                x0, z0, x1, z1 = stroke_world_endpoints(action, sim.canvas)
                centers.append((0.5 * (x0 + x1), 0.5 * (z0 + z1)))
            return tuple(float(v) for v in np.mean(np.asarray(centers, dtype=np.float64), axis=0))  # type: ignore[return-value]
        tip = sim.kinematics.tip(sim.actual_pose)
        lateral_limit = 0.46 * min(sim.canvas.width, sim.canvas.height)
        x = float(np.clip(tip[0], -lateral_limit, lateral_limit))
        z = float(np.clip(tip[2], -lateral_limit, lateral_limit))
        if not np.isfinite(x) or not np.isfinite(z):
            return 0.0, 0.0
        return x, z

    def _local_passage_candidates(self) -> tuple[list[Policy], list[float], dict[tuple[StrokeAction, ...], PassageLatent]]:
        if self._passage_belief is None or self._active_passage is None:
            raise RuntimeError("Local passage planning requires an active passage belief.")
        continuation_probability = float(np.clip(self.config.passage_continuation_probability, 1e-4, 1.0 - 1e-4))
        policies = [Policy((StrokeAction.stop_action(),))]
        log_priors = [float(np.log1p(-continuation_probability))]
        latent_by_actions: dict[tuple[StrokeAction, ...], PassageLatent] = {}
        candidate_limit = max(2, int(self.config.passage_local_candidate_policies))
        tones = (
            (0.0, 1.0)
            if self.config.stroke_tone_prior is None
            else (float(self.config.stroke_tone_prior),)
        )
        geometry_index = 0
        while len(policies) < candidate_limit:
            for tone in tones:
                if len(policies) >= candidate_limit:
                    break
                if geometry_index == 0:
                    latent = replace(self._passage_belief.mean_latent(), tone=float(tone))
                else:
                    latent = self._passage_belief.sample_latent(self.agent.policy_sampler.rng, tone=float(tone))
                remaining_actions = self.agent.policy_sampler.passage_actions(
                    latent,
                    start_index=self._active_passage_completed_strokes,
                )
                if not remaining_actions:
                    continue
                actions = tuple(remaining_actions) + (StrokeAction.stop_action(),)
                policy = Policy(
                    actions,
                    passage=latent,
                    passage_start_index=self._active_passage_completed_strokes,
                )
                policies.append(policy)
                log_priors.append(
                    float(np.log(continuation_probability))
                    + self._passage_belief.transition_log_prior(latent)
                )
                latent_by_actions[actions] = latent
            geometry_index += 1
        return policies, log_priors, latent_by_actions

    def _start_local_passage_plan(self, sim: ArmPainterSim) -> None:
        if not self._passage_queue or self._passage_belief is None or self.current is not None:
            return
        with self._planner_lock:
            if self.planning or self._pending_ranked is not None or self._pending_current is not None:
                return
            if self._planner_thread is not None and self._planner_thread.is_alive():
                return
            self.planning = True
            self._planning_started_at = time.perf_counter()
            self._pending_error = None
            body_snapshot = copy.deepcopy(sim)
            generation = self._planner_generation
        thread = threading.Thread(
            target=self._background_local_passage_plan,
            args=(body_snapshot, generation),
            name="active-painter-local-passage-plan",
            daemon=True,
        )
        with self._planner_lock:
            self._planner_thread = thread
        thread.start()

    def _background_local_passage_plan(
        self,
        body_snapshot: ArmPainterSim,
        generation: int | None = None,
    ) -> None:
        generation = self._planner_generation if generation is None else int(generation)
        with self._planner_lock:
            if generation != self._planner_generation:
                return
        started = time.perf_counter()
        profile: dict[str, object] = {
            "kind": "planning_profile",
            "scope": "passage_local",
            "totalSeconds": 0.0,
            "policyInferenceSeconds": 0.0,
            "policySampleSeconds": 0.0,
            "baseEFESeconds": 0.0,
            "motorForecastSeconds": 0.0,
            "motorEFERescoreSeconds": 0.0,
            "posteriorSeconds": 0.0,
            "selectedForecastSeconds": 0.0,
            "selectedForecastCacheHits": 0,
            "policyCount": 0,
            "motorForecastCount": 0,
            "motorForecastCacheHits": 0,
            "motorForecastBatchCount": 0,
            "motorForecastBatchJobs": 0,
            "motorForecastWorkers": 0,
            "candidateMotorRealizations": 0,
        }
        self._planning_profile_current = profile
        self._planning_forecast_cache = {}
        pending_current: StrokeExecution | None = None
        pending_stopped = False
        pending_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] | None = None
        pending_components: EFEComponents | SpatialEFEComponents | None = None
        pending_queue: tuple[StrokeAction, ...] = ()
        pending_passage = self._active_passage
        error: str | None = None
        try:
            policies, log_priors, latent_by_actions = self._local_passage_candidates()
            self._profile_set("policyCount", len(policies))
            phase_started = time.perf_counter()
            if self._uses_spatial_planner():
                ranked = self._infer_spatial_policy_with_execution_forecasts(
                    body_snapshot,
                    policies,
                    log_priors,
                )
            else:
                ranked = self._infer_policy_with_execution_forecasts(
                    body_snapshot,
                    policies,
                    log_priors,
                )
            self._profile_add_seconds("policyInferenceSeconds", time.perf_counter() - phase_started)
            pending_ranked = ranked
            policy, component, posterior = ranked[0]
            pending_components = component
            if policy.actions[0].stop:
                pending_stopped = True
            else:
                pending_passage = latent_by_actions.get(policy.actions, pending_passage)
                pending_queue = tuple(action for action in policy.actions[1:] if not action.stop)
                primitive = policy.motor_primitive
                pending_current = StrokeExecution(
                    action=policy.actions[0],
                    efe=component,
                    posterior=float(posterior),
                    initial_state=self._planner_state(body_snapshot),
                    forecast=self._profiled_forecast_action(
                        body_snapshot,
                        policy.actions[0],
                        primitive,
                        policy.brush_preparation,
                    ),
                    motor_primitive=primitive,
                    brush_preparation=policy.brush_preparation,
                    controller=controller_for_motor_primitive(primitive),
                )
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            error = repr(exc)
        if error is None:
            self._refresh_composition_diagnostics(
                pending_ranked[0][0] if pending_ranked else None
            )
        profile["totalSeconds"] = time.perf_counter() - started
        with self._planner_lock:
            if generation != self._planner_generation:
                return
            self._pending_current = pending_current
            self._pending_stopped = pending_stopped
            self._pending_ranked = pending_ranked
            self._pending_components = pending_components
            self._pending_passage_queue = pending_queue
            self._pending_passage = pending_passage
            self._pending_passage_plan = self._active_passage_plan
            self._pending_plan_scope = "passage_local"
            self._pending_error = error
            self.last_planning_seconds = float(profile["totalSeconds"])
            self.last_planning_profile = dict(profile)
            self.planning = False
            self._planning_started_at = None
        self._planning_profile_current = None
        self._planning_forecast_cache = {}

    def _start_background_plan(self, sim: ArmPainterSim) -> None:
        if self.current is not None or self._passage_queue:
            return
        with self._planner_lock:
            if (
                self.planning
                or self._pending_ranked is not None
                or self._pending_current is not None
                or self._pending_stopped
            ):
                return
            # The previous planner thread may still be training after its plan
            # was published; model updates must not race the next evaluation.
            if self._planner_thread is not None and self._planner_thread.is_alive():
                return
            transition = self._transition_to_learn
            self._transition_to_learn = None
            state = self._planner_state(sim)
            body_snapshot = copy.deepcopy(sim)
            self.planning = True
            self._planning_started_at = time.perf_counter()
            self._pending_error = None
            generation = self._planner_generation
        thread = threading.Thread(
            target=self._background_plan,
            args=(state, transition, body_snapshot, generation),
            name="active-painter-policy-plan",
            daemon=True,
        )
        with self._planner_lock:
            self._planner_thread = thread
        thread.start()

    def _background_plan(
        self,
        state: np.ndarray | SpatialCanvasState,
        transition: tuple[
            np.ndarray | SpatialCanvasState,
            StrokeAction,
            MotorPrimitiveLatent | None,
            np.ndarray | SpatialCanvasState,
        ] | None,
        body_snapshot: ArmPainterSim | None = None,
        generation: int | None = None,
    ) -> None:
        generation = self._planner_generation if generation is None else int(generation)
        with self._planner_lock:
            if generation != self._planner_generation:
                return
        started = time.perf_counter()
        pending_current: StrokeExecution | None = None
        pending_stopped = False
        pending_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] | None = None
        pending_components: EFEComponents | SpatialEFEComponents | None = None
        pending_passage_queue: tuple[StrokeAction, ...] = ()
        pending_passage: PassageLatent | None = None
        pending_passage_plan: PassagePlanLatent | None = None
        error: str | None = None
        profile: dict[str, object] = {
            "kind": "planning_profile",
            "scope": "global",
            "totalSeconds": 0.0,
            "beliefUpdateSeconds": 0.0,
            "policyInferenceSeconds": 0.0,
            "policySampleSeconds": 0.0,
            "baseEFESeconds": 0.0,
            "motorForecastSeconds": 0.0,
            "motorEFERescoreSeconds": 0.0,
            "posteriorSeconds": 0.0,
            "compositionDiagnosticSeconds": 0.0,
            "selectedForecastSeconds": 0.0,
            "selectedForecastCacheHits": 0,
            "trailingTrainingSeconds": 0.0,
            "proposalTrainingSeconds": 0.0,
            "proposalTargetSupportFraction": 0.0,
            "publishSeconds": 0.0,
            "policyCount": 0,
            "motorForecastCount": 0,
            "motorForecastCacheHits": 0,
            "motorForecastBatchCount": 0,
            "motorForecastBatchJobs": 0,
            "motorForecastWorkers": 0,
            "candidateMotorRealizations": 0,
            "trainingAfterPublish": False,
            "beliefUpdateRequiredBeforeInference": transition is not None,
        }
        self._planning_profile_current = profile
        self._planning_forecast_cache = {}
        try:
            phase_started = time.perf_counter()
            with self._planner_lock:
                if generation != self._planner_generation:
                    return
                if transition is not None:
                    before, action, motor_primitive, after = transition
                    self._add_transition_to_agent(before, action, after, motor_primitive)
                    self.trained_transitions += 1
                    self._update_agent_belief(action, after, motor_primitive)
                else:
                    self._reset_agent_belief(state)
                self.belief = self.agent.belief
            self._profile_add_seconds("beliefUpdateSeconds", time.perf_counter() - phase_started)
            phase_started = time.perf_counter()
            if body_snapshot is None:
                _, _, ranked = self.agent.infer_policy()
                self.last_motor_rejections = 0
                self.last_motor_primitive_candidates = 0
                self._profile_set("policyCount", len(ranked))
            elif self._uses_spatial_planner():
                ranked = self._infer_spatial_policy_with_execution_forecasts(body_snapshot)
            else:
                ranked = self._infer_policy_with_execution_forecasts(body_snapshot)
            self._profile_add_seconds("policyInferenceSeconds", time.perf_counter() - phase_started)
            pending_ranked = ranked
            policy, efe, prob = ranked[0]
            action = policy.actions[0]
            pending_components = efe
            # Premature termination is handled by the declared stop prior
            # inside policy inference, not by a procedural veto here.
            if action.stop:
                pending_stopped = True
            else:
                motor_primitive = policy.motor_primitive
                if policy.passage is not None:
                    passage_actions = policy.actions[:-1]
                    pending_passage_queue = tuple(passage_actions[1:])
                    pending_passage = policy.passage
                elif policy.passage_plan is not None:
                    # Receding-horizon hierarchy: the full plan supplies the
                    # global terminal rollout, but execution commits only to
                    # its first explicit passage boundary.
                    pending_passage_plan = policy.passage_plan
                    pending_passage = policy.passage_plan.passages[0]
                    first_passage_count = pending_passage.stroke_count
                    passage_actions = policy.actions[:first_passage_count]
                    pending_passage_queue = tuple(passage_actions[1:])
                pending_current = StrokeExecution(
                    action=action,
                    efe=efe,
                    posterior=float(prob),
                    initial_state=state,
                    forecast=self._profiled_forecast_action(
                        body_snapshot,
                        action,
                        motor_primitive,
                        policy.brush_preparation,
                    ),
                    motor_primitive=motor_primitive,
                    brush_preparation=policy.brush_preparation,
                    controller=controller_for_motor_primitive(motor_primitive),
                )
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            error = repr(exc)
        if error is None:
            phase_started = time.perf_counter()
            self._refresh_composition_diagnostics(
                pending_ranked[0][0] if pending_ranked else None
            )
            self._profile_add_seconds("compositionDiagnosticSeconds", time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        profile["totalSeconds"] = time.perf_counter() - started
        with self._planner_lock:
            if generation != self._planner_generation:
                return
            self._pending_current = pending_current
            self._pending_stopped = pending_stopped
            self._pending_ranked = pending_ranked
            self._pending_components = pending_components
            self._pending_passage_queue = pending_passage_queue
            self._pending_passage = pending_passage
            self._pending_passage_plan = pending_passage_plan
            self._pending_plan_scope = "global"
            self._pending_error = error
            self.last_planning_seconds = time.perf_counter() - started
            profile["totalSeconds"] = self.last_planning_seconds
            self.last_planning_profile = dict(profile)
            self.planning = False
            self._planning_started_at = None
        self._profile_add_seconds("publishSeconds", time.perf_counter() - phase_started)
        with self._planner_lock:
            self.last_planning_profile = dict(profile)
        # Model learning runs after the plan is published, so it overlaps the
        # selected stroke's execution instead of extending the planning gap.
        # _start_background_plan will not launch the next planner thread until
        # this one exits, so training never races policy evaluation.
        if error is None and transition is not None and generation == self._planner_generation:
            try:
                train_started = time.perf_counter()
                self.last_training_loss = self.agent.train_dynamics(gradient_steps=2)
                self.last_training_seconds = time.perf_counter() - train_started
                # One amortization step toward the base-EFE posterior this round
                # produced. Inside the same post-publish block, so the pinned
                # "training happens AFTER the plan is published" ordering holds for
                # the proposal too and the step overlaps stroke execution.
                proposal_started = time.perf_counter()
                proposal_support = 0.0
                if isinstance(self.agent, SpatialActiveInferencePainter):
                    batch = self._pending_proposal_batch
                    self.agent.train_policy_proposal(batch)
                    if batch is not None:
                        proposal_support = float(batch.target_support_fraction)
                self._pending_proposal_batch = None
                proposal_seconds = time.perf_counter() - proposal_started
                self._save_checkpoint_if_due()
                with self._planner_lock:
                    self.last_planning_profile = {
                        **self.last_planning_profile,
                        "trailingTrainingSeconds": self.last_training_seconds,
                        "proposalTrainingSeconds": proposal_seconds,
                        "proposalTargetSupportFraction": proposal_support,
                        "trainingAfterPublish": True,
                    }
            except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
                with self._planner_lock:
                    self._pending_error = repr(exc)
        self._planning_profile_current = None
        self._planning_forecast_cache = {}

    def _profile_add_seconds(self, key: str, seconds: float) -> None:
        profile = self._planning_profile_current
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + max(0.0, float(seconds))

    def _profile_increment(self, key: str, amount: int = 1) -> None:
        profile = self._planning_profile_current
        if profile is None:
            return
        profile[key] = int(profile.get(key, 0)) + int(amount)

    def _profile_set(self, key: str, value: object) -> None:
        profile = self._planning_profile_current
        if profile is not None:
            profile[key] = value

    def _profiled_forecast_action(
        self,
        sim: ArmPainterSim | None,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None = None,
        brush_preparation: BrushPreparationPolicy | None = None,
    ) -> ExecutionForecast | None:
        if sim is None or action.stop:
            return None
        belief = self.brush_load_beliefs[self._brush_key(action.tone)]
        key = self._forecast_cache_key(
            action,
            motor_primitive,
            brush_preparation,
            belief,
        )
        cached = self._planning_forecast_cache.get(key)
        if cached is not None:
            self._profile_increment("selectedForecastCacheHits")
            return cached
        started = time.perf_counter()
        forecast = self._forecast_action(
            sim,
            action,
            motor_primitive,
            brush_preparation,
            belief,
        )
        self._profile_add_seconds("selectedForecastSeconds", time.perf_counter() - started)
        return forecast

    def _forecast_cache_key(
        self,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None,
        brush_preparation: BrushPreparationPolicy | None = None,
        brush_belief: BrushLoadBelief | None = None,
    ) -> tuple[object, ...]:
        primitive_key = "" if motor_primitive is None else motor_primitive.kind
        preparation_key = (
            "reload"
            if brush_preparation is None
            else brush_preparation.kind
        )
        belief_key: tuple[object, ...] = ()
        if brush_belief is not None:
            belief_key = (
                brush_belief.revision,
                round(brush_belief.load_mean, 6),
                round(brush_belief.black_fraction_mean, 6),
            )
        return (
            tuple(float(x) for x in action.vector())
            + (primitive_key, preparation_key)
            + belief_key
        )

    def _refresh_composition_diagnostics(self, policy: Policy | None = None) -> None:
        # Cached so UI polling never runs a model forward concurrently with
        # background training.
        if isinstance(self.agent, SpatialActiveInferencePainter) and self.agent.composition is not None:
            self._cached_belief_gap = self.agent.belief_composition_gap()
            # Feed the Delta-gap belief here, at planner cadence, rather than at
            # mark completion: `belief_composition_gap` runs a model forward and
            # mark completion is on the polling thread. `observe` divides by the
            # exact number of marks elapsed since the previous reading, so the
            # per-mark denominator is exact even though the sampling is coarse.
            # Named approximation: per-mark gap increment amortized over the
            # marks between planning-cadence gap readings.
            self.gap_increment.observe(self._cached_belief_gap, self.stroke_count)
            self._cached_passage_trajectory = (
                self.agent.composition.passage_trajectory_diagnostics(policy)
                if policy is not None
                else None
            )
        # Also cached, and for the same reason: the divergence estimator runs the
        # proposal network forward, and diagnostics() is polled from the web
        # thread. Never runs there.
        self._refresh_policy_proposal_diagnostics()

    def _refresh_policy_proposal_diagnostics(self) -> None:
        """Measure the declared H4 numbers on the planner thread, once per plan.

        Everything here is EVIDENCE. No expected-free-energy term, variational
        free-energy term, preference, precision belief, policy prior, policy
        posterior, or control-flow branch reads any value in this block.
        """

        if not isinstance(self.agent, SpatialActiveInferencePainter):
            self._cached_policy_proposal = None
            return
        agent = self.agent
        if agent.policy_proposal is None:
            self._cached_policy_proposal = None
            return
        try:
            features, source = agent.proposal_belief_features()
            if features is None:
                self._cached_policy_proposal = None
                return
            coverage_field = (
                self.belief.coverage(self.config.paint_presence_threshold)
                if isinstance(self.belief, SpatialCanvasState)
                else None
            )
            block: dict[str, object] = {"beliefFeatureSource": str(source)}
            passage_divergence = agent.policy_proposal.divergence_against_hand_written(
                features, coverage_field, self.config, family="passage"
            )
            mark_divergence = agent.policy_proposal.divergence_against_hand_written(
                features, coverage_field, self.config, family="mark"
            )
            block["divergenceNats"] = float(passage_divergence.divergence_nats)
            block["markDivergenceNats"] = float(mark_divergence.divergence_nats)
            block["divergenceSamples"] = int(
                passage_divergence.sample_count + mark_divergence.sample_count
            )
            block["outOfHandSupportFraction"] = float(
                0.5
                * (
                    passage_divergence.out_of_support_fraction
                    + mark_divergence.out_of_support_fraction
                )
            )
            block["divergenceApproximation"] = str(passage_divergence.approximation)

            heads = agent.policy_proposal.distribution(features)
            selected = [record for record in list(self._selected_proposal_records) if record is not None]
            if selected:
                learned_values = [
                    agent.policy_proposal.log_density(
                        record, features, self.config, heads=heads
                    ).total
                    / max(1, len(record.latents))
                    for record in selected
                ]
                hand_values = [
                    hand_written_log_density(record, coverage_field, self.config).total
                    / max(1, len(record.latents))
                    for record in selected
                ]
                learned_mean = float(sum(learned_values) / len(learned_values))
                hand_mean = float(sum(hand_values) / len(hand_values))
                block["selectedMeanLogLikelihoodLearned"] = learned_mean
                block["selectedMeanLogLikelihoodHandWritten"] = hand_mean
                block["selectedLogLikelihoodAdvantage"] = learned_mean - hand_mean
                block["selectedSampleCount"] = len(selected)
            else:
                block["selectedMeanLogLikelihoodLearned"] = None
                block["selectedMeanLogLikelihoodHandWritten"] = None
                block["selectedLogLikelihoodAdvantage"] = None
                block["selectedSampleCount"] = 0

            batch = self._pending_proposal_batch
            cross_entropy: float | None = None
            if batch is not None and batch.refined_weights:
                total = 0.0
                mass = 0.0
                for index, weight in enumerate(batch.refined_weights):
                    if weight <= 0.0 or index >= len(batch.records):
                        continue
                    record = batch.records[index]
                    if record is None:
                        continue
                    log_density = agent.policy_proposal.log_density(
                        record, features, self.config, heads=heads
                    ).total
                    total -= float(weight) * float(log_density)
                    mass += float(weight)
                cross_entropy = float(total / mass) if mass > 0.0 else None
            block["refinedTargetCrossEntropy"] = cross_entropy
            self._cached_policy_proposal = block
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            self._cached_policy_proposal = {
                "status": "diagnostic_failed",
                "lastError": repr(exc),
            }

    def _current_planning_seconds(self) -> float:
        if not self.planning or self._planning_started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._planning_started_at)

    def _consume_background_plan(self) -> bool:
        with self._planner_lock:
            if self.planning:
                return False
            if self._pending_ranked is None and self._pending_current is None and not self._pending_stopped:
                return False
            pending_current = self._pending_current
            pending_stopped = self._pending_stopped
            pending_ranked = self._pending_ranked
            pending_components = self._pending_components
            pending_passage_queue = self._pending_passage_queue
            pending_passage = self._pending_passage
            pending_passage_plan = self._pending_passage_plan
            pending_plan_scope = self._pending_plan_scope
            # Diagnostic: stop had the lowest expected free energy, but the
            # declared stop prior demoted it below a continuation policy.
            stop_blocked = False
            if pending_ranked:
                lowest_efe = min(pending_ranked, key=lambda item: item[1].total)
                stop_blocked = bool(
                    lowest_efe[0].actions[0].stop and not pending_ranked[0][0].actions[0].stop
                )
            self._pending_current = None
            self._pending_stopped = False
            self._pending_ranked = None
            self._pending_components = None
            self._pending_passage_queue = ()
            self._pending_passage = None
            self._pending_passage_plan = None
            self._pending_plan_scope = "global"
        if pending_ranked is not None:
            self.last_ranked = pending_ranked
            # Session-local evidence for the H4 headline. Taken from the published
            # plan's own record rather than by walking `last_ranked`, which tests
            # hand-build from bare EFE components with no proposal provenance.
            if self._pending_proposal_record is not None:
                self._selected_proposal_records.append(self._pending_proposal_record)
                self._pending_proposal_record = None
        if pending_components is not None:
            self.last_components = pending_components
        self.last_stop_blocked = stop_blocked
        if pending_stopped:
            self.stopped = True
            if self.on_stop is not None:
                self.on_stop()
            return True
        self.current = pending_current
        if pending_current is not None:
            self._passage_queue = list(pending_passage_queue)
            self._active_passage = pending_passage
            self._active_passage_plan = pending_passage_plan
            if pending_plan_scope == "global":
                self._hierarchy_passage_initial_state = (
                    pending_current.initial_state
                    if isinstance(pending_current.initial_state, SpatialCanvasState)
                    else None
                )
                self._hierarchy_passage_actions = []
                self._active_passage_total_strokes = (
                    pending_passage.stroke_count if pending_passage is not None else 0
                )
                self._active_passage_completed_strokes = 0
                self._passage_belief = (
                    PassageBelief.from_latent(pending_passage, self.config)
                    if pending_passage is not None
                    else None
                )
            elif pending_passage is not None:
                self._active_passage_total_strokes = pending_passage.stroke_count
                if self._passage_belief is not None:
                    self._passage_belief = replace(self._passage_belief, template=pending_passage)
            self._hold_pose = None
            self._hold_command_pose = None
            self._hold_command_velocity = {}
        return False

    def _start_next_passage_stroke(self, sim: ArmPainterSim) -> None:
        if not self._passage_queue:
            return
        action = self._passage_queue.pop(0)
        preparation = self._infer_brush_preparation(action)
        self.current = StrokeExecution(
            action=action,
            efe=self.last_components if self.last_components is not None else EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            posterior=1.0,
            initial_state=self._planner_state(sim),
            brush_preparation=preparation.selected,
            controller=ContactAwareStrokeController(),
        )
        self._hold_pose = None
        self._hold_command_pose = None
        self._hold_command_velocity = {}

    def _execute_current(self, sim: ArmPainterSim, dt: float) -> None:
        assert self.current is not None
        ex = self.current
        self._hold_command_pose = None
        self._hold_command_velocity = {}
        if not ex.initialized:
            ex.timing = adaptive_stroke_timing(sim, ex.action)
            preparation = self._infer_brush_preparation(ex.action)
            if ex.brush_preparation is None:
                ex.brush_preparation = preparation.selected
            self.last_brush_preparation = preparation
            key = self._brush_key(ex.action.tone)
            sim.select_brush(ex.action.tone)
            sim.deposition_amount = ex.action.amount
            if ex.brush_preparation.kind == "reload":
                sim.load_brush(1.0, ex.action.tone)
                self.brush_load_beliefs[key] = (
                    self.brush_loading_model.reload_transition(
                        self.brush_load_beliefs[key],
                        ex.action.tone,
                    )
                )
            ex.controller.reset(sim, ex.action, ex.timing)
            ex.initialized = True
        ex.t += dt
        command = ex.controller.command(sim, ex.action, ex.t, dt, ex.timing)
        sim.control_damping_multiplier = 1.0
        sim.set_target(command.pose)
        sim.intended_contact_pressure = command.intended_pressure
        sim.brush_flow = command.reference.flow
        # Accumulate realized tracking residuals while painting (the contact
        # state reflects the last completed sim step). These become the
        # reliability observation for this stroke's motor realization kind.
        if command.reference.phase == "paint" and sim.contact.on_canvas and sim.contact.pressure > 0.001:
            x0, z0, x1, z1 = stroke_world_endpoints(ex.action, sim.canvas)
            px = float(sim.contact.brush_world[0])
            pz = float(sim.contact.brush_world[2])
            seg_x, seg_z = x1 - x0, z1 - z0
            seg_len_sq = seg_x * seg_x + seg_z * seg_z
            if seg_len_sq > 1e-12:
                proj = max(0.0, min(1.0, ((px - x0) * seg_x + (pz - z0) * seg_z) / seg_len_sq))
                dx, dz = px - (x0 + proj * seg_x), pz - (z0 + proj * seg_z)
            else:
                dx, dz = px - x0, pz - z0
            path_error_norm = float(np.hypot(dx, dz)) / max(1e-6, sim.canvas.width)
            pressure_error = float(sim.contact.pressure - command.intended_pressure)
            ex.realized_path_error_sq_sum += path_error_norm * path_error_norm
            ex.realized_pressure_error_sq_sum += pressure_error * pressure_error
            ex.realized_contact_samples += 1
        if ex.t >= ex.total:
            self._observe_motion_reliability(ex)
            key = self._brush_key(ex.action.tone)
            self.brush_load_beliefs[key] = self.brush_loading_model.stroke_transition(
                self.brush_load_beliefs[key],
                ex.action,
            )
            self.stroke_count += 1
            self.last_execution_forecast = ex.forecast
            self.current = None
            after = self._planner_state(sim)
            passage_continues = bool(self._passage_queue)
            if isinstance(after, SpatialCanvasState):
                self._hierarchy_passage_actions.append(ex.action)
            if self._active_passage_total_strokes > 0:
                self._active_passage_completed_strokes += 1
            passage_step_index = max(0, self._active_passage_completed_strokes - 1)
            if (
                isinstance(self.agent, SpatialActiveInferencePainter)
                and isinstance(ex.initial_state, SpatialCanvasState)
                and isinstance(after, SpatialCanvasState)
                and self._active_passage is not None
            ):
                self.agent.add_passage_step_transition(
                    ex.initial_state,
                    self._active_passage,
                    passage_step_index,
                    after,
                )
            if ex.initial_state is not None:
                if passage_continues:
                    self._add_transition_to_agent(ex.initial_state, ex.action, after, ex.motor_primitive)
                    self.trained_transitions += 1
                    self._update_agent_belief(ex.action, after, ex.motor_primitive)
                    self.belief = self.agent.belief
                else:
                    with self._planner_lock:
                        self._transition_to_learn = (ex.initial_state, ex.action, ex.motor_primitive, after)
                if self._passage_belief is not None and self._active_passage is not None:
                    observation = infer_passage_observation(
                        ex.initial_state,
                        after,
                        ex.action,
                        self._active_passage,
                        passage_step_index,
                        self.config,
                    )
                    self._passage_belief = self._passage_belief.update(observation, self.config)
                    self._active_passage = self._passage_belief.mean_latent()
            if (
                not passage_continues
                and isinstance(self.agent, SpatialActiveInferencePainter)
                and isinstance(after, SpatialCanvasState)
            ):
                self._complete_hierarchy_passage(after, ex.initial_state)
            if passage_continues:
                self._post_stroke_retract_remaining = max(0.0, self.config.passage_local_retract_seconds)
                self._start_local_passage_plan(sim)
                self._hold_retracted(sim, dt, scope="passage")
            else:
                had_active_passage = self._active_passage_total_strokes > 0
                self._active_passage = None
                self._active_passage_plan = None
                self._passage_belief = None
                self._active_passage_total_strokes = 0
                self._active_passage_completed_strokes = 0
                retract_seconds = (
                    self.config.passage_center_retract_seconds
                    if had_active_passage
                    else self.config.post_stroke_retract_seconds
                )
                self._post_stroke_retract_remaining = max(0.0, retract_seconds)
                self._hold_retracted(sim, dt, scope="global")

    def _observe_motion_reliability(self, ex: StrokeExecution) -> None:
        """Update the executed kind's reliability belief from realized-vs-forecast
        tracking residuals. Skipped when there is no forecast to compare against
        (e.g. queued passage strokes) or the stroke never made contact."""

        if ex.forecast is None or ex.realized_contact_samples < 4:
            return
        labels = tuple(ex.forecast.proprioceptive_labels)
        if "path_error" not in labels or "pressure_error" not in labels:
            return
        mean = np.asarray(ex.forecast.proprioceptive_mean, dtype=np.float64)
        predicted_path = float(mean[labels.index("path_error")])
        predicted_pressure = float(mean[labels.index("pressure_error")])
        realized_path = float(np.sqrt(ex.realized_path_error_sq_sum / ex.realized_contact_samples))
        realized_pressure = float(np.sqrt(ex.realized_pressure_error_sq_sum / ex.realized_contact_samples))
        kind = ex.motor_primitive.kind if ex.motor_primitive is not None else "cartesian_ik"
        path_floor = max(1e-4, 0.05 * float(self.config.motor_path_error_preference_std))
        pressure_floor = max(1e-4, 0.05 * float(self.config.motor_pressure_error_preference_std))
        # Two half-weight observations per stroke: path tracking and pressure
        # tracking each contribute evidence about this kind's execution jitter.
        self.motion_reliability.observe(
            kind, execution_error_ratio_sq(realized_path, predicted_path, path_floor), weight=0.5
        )
        self.motion_reliability.observe(
            kind, execution_error_ratio_sq(realized_pressure, predicted_pressure, pressure_floor), weight=0.5
        )

    def _complete_hierarchy_passage(
        self,
        after: SpatialCanvasState,
        fallback_initial_state: np.ndarray | SpatialCanvasState | None,
    ) -> None:
        if not isinstance(self.agent, SpatialActiveInferencePainter):
            return
        before = self._hierarchy_passage_initial_state
        if before is None and isinstance(fallback_initial_state, SpatialCanvasState):
            before = fallback_initial_state
        actions = tuple(self._hierarchy_passage_actions)
        if before is not None and actions:
            self.agent.add_passage_transition(before, actions, after)
            self.agent.update_hierarchy_beliefs(after, actions)
        self._hierarchy_passage_initial_state = None
        self._hierarchy_passage_actions = []

    def _yield_to_runtime(self) -> None:
        delay = max(0.0, float(self.config.background_planner_yield_seconds))
        if delay > 0.0:
            time.sleep(delay)
        else:
            time.sleep(0)

    def _forecast_motor_realizations(
        self,
        body_snapshot: ArmPainterSim,
        action: StrokeAction,
        motor_policies: list[Policy],
        summary_fn: Callable[[ArmPainterSim], np.ndarray],
        forecast_cache: dict[tuple[object, ...], ExecutionForecast],
    ) -> list[ExecutionForecast]:
        """Resolve cached forecasts and batch only the missing likelihoods."""

        resolved: list[ExecutionForecast | None] = [None] * len(motor_policies)
        missing_indices: list[int] = []
        missing_keys: list[tuple[object, ...]] = []
        missing_requests: list[
            tuple[
                StrokeAction,
                MotorPrimitiveLatent | None,
                bool,
                BrushLoadBelief | None,
            ]
        ] = []
        for index, motor_policy in enumerate(motor_policies):
            primitive = motor_policy.motor_primitive
            preparation = motor_policy.brush_preparation
            belief = self.brush_load_beliefs[self._brush_key(action.tone)]
            key = self._forecast_cache_key(
                action,
                primitive,
                preparation,
                belief,
            )
            cached = forecast_cache.get(key)
            if cached is not None:
                resolved[index] = cached
                self._profile_increment("motorForecastCacheHits")
                continue
            missing_indices.append(index)
            missing_keys.append(key)
            missing_requests.append(
                (
                    action,
                    primitive,
                    preparation is None or preparation.kind == "reload",
                    belief,
                )
            )

        if missing_requests:
            started = time.perf_counter()
            workers = min(len(missing_requests), max(1, int(self.config.motor_forecast_workers)))
            computed = forecast_stroke_executions_batch(
                body_snapshot,
                missing_requests,
                summary_fn,
                dt=1.0 / 45.0,
                max_workers=workers,
            )
            self._profile_add_seconds("motorForecastSeconds", time.perf_counter() - started)
            self._profile_increment("motorForecastCount", len(computed))
            self._profile_increment("motorForecastBatchCount")
            self._profile_increment("motorForecastBatchJobs", len(computed))
            self._profile_set("motorForecastWorkers", workers)
            for index, key, forecast in zip(missing_indices, missing_keys, computed):
                resolved[index] = forecast
                forecast_cache[key] = forecast
                self._planning_forecast_cache[key] = forecast
            self._yield_to_runtime()

        if any(forecast is None for forecast in resolved):
            raise RuntimeError("Motor forecast batch did not resolve every realization.")
        return [forecast for forecast in resolved if forecast is not None]

    def _modality_contribution_vectors(
        self,
        components,
        indices: list[int],
    ) -> dict[str, list[float]]:
        """Per-candidate, post-normalization, PRE-gamma modality contributions.

        The component dataclass stores POST-weighted terms, so each modality's
        own precision is divided back out here. That is what makes the update a
        function of the modality's own contribution vector rather than of its
        current weight -- the belief must not be estimated from a quantity that
        already contains it.

        Structurally-off modalities (a declared constant of exactly 0.0, or a
        term that is identically zero without an execution forecast) are omitted
        rather than fed a degenerate all-zero vector.
        """

        if not indices:
            return {}
        first = components[indices[0]]
        raw: dict[str, list[float]] = {}
        raw["terminal_coverage"] = [float(components[i].terminal_risk) for i in indices]
        raw["observation_ambiguity"] = [float(components[i].ambiguity) for i in indices]
        raw["transition"] = [
            float(components[i].transition_risk + components[i].transition_ambiguity)
            for i in indices
        ]
        raw["motor_proprioceptive"] = [
            float(
                motor_efe_contribution(
                    components[i].motor_risk, components[i].motor_epistemic_value
                )
            )
            for i in indices
        ]
        if hasattr(first, "composition_risk"):
            raw["composition_gap"] = [float(components[i].composition_risk) for i in indices]
            raw["canvas_latent_transition"] = [
                float(
                    components[i].canvas_transition_risk
                    + components[i].passage_canvas_trajectory_risk
                )
                for i in indices
            ]
            raw["relational_transition"] = [
                float(
                    components[i].relational_transition_risk
                    + components[i].passage_relational_trajectory_risk
                )
                for i in indices
            ]
        vectors: dict[str, list[float]] = {}
        for name, values in raw.items():
            gamma = self.precision_ledger.mean(name)
            if gamma == 0.0:
                continue
            unweighted = [value / gamma for value in values]
            if max(unweighted) - min(unweighted) <= 0.0 and max(abs(v) for v in unweighted) == 0.0:
                # Identically zero: no contribution, hence nothing to weight.
                continue
            vectors[name] = unweighted
        return vectors

    def _observe_precision_beliefs(
        self,
        components,
        brush_inferences: dict[int, BrushPreparationInference],
        policy_log_priors: list[float],
        non_stop_indices: list[int],
    ) -> None:
        """Update every precision belief from the realized (G, F) candidate pair.

        F_i = -brush_preparation_log_evidence_i. That quantity is the negative
        log marginal evidence of realizing candidate i's intended mark amount and
        pigment under the current brush-load posterior, with the preserve/reload
        preparation policy EXACTLY marginalized over its declared prior -- hence a
        variational free energy at its optimal variational posterior, not a score.
        Three properties make it admissible as the driving F:

        1. It is built from `brush_policy_precision`, NOT `policy_precision`, so
           it contains no painting policy precision and cannot bootstrap the
           gamma it is used to update.
        2. It is computed for EVERY non-stop candidate before the forecast-budget
           sort, so it is defined on the full candidate set and does not depend
           on any gamma-dependent pruning.
        3. It enters only the precision gradient. It is never added to any
           candidate's G or logit.

        STOP candidates are excluded: they realize no mark, so they have no
        realization free energy. That also keeps this mechanism disentangled
        from the gap-progress stop prior.
        """

        if not self.config.precision_beliefs_enabled:
            return
        indices = [index for index in non_stop_indices if index in brush_inferences]
        if not indices:
            return
        g_total = [float(components[index].total) for index in indices]
        free_energy = [-float(brush_inferences[index].log_evidence) for index in indices]
        log_prior = [float(policy_log_priors[index]) for index in indices]
        self.precision_ledger.observe_policy(g_total, free_energy, log_prior=log_prior)
        if not self.config.modality_precision_beliefs_enabled:
            return
        for name, vector in self._modality_contribution_vectors(components, indices).items():
            self.precision_ledger.observe(name, vector, free_energy, log_prior=log_prior)

    def _infer_policy_with_execution_forecasts(
        self,
        body_snapshot: ArmPainterSim,
        policies: list[Policy] | None = None,
        policy_log_priors: list[float] | None = None,
    ) -> list[tuple[Policy, EFEComponents, float]]:
        assert isinstance(self.agent, ActiveInferencePainter)
        assert isinstance(self.belief, GaussianBelief)
        agent = self.agent
        belief = self.belief
        phase_started = time.perf_counter()
        policies = agent.policy_sampler.sample() if policies is None else list(policies)
        policy_log_priors = [0.0] * len(policies) if policy_log_priors is None else list(policy_log_priors)
        if len(policy_log_priors) != len(policies):
            raise ValueError("Policy log priors must align with candidate policies.")
        self._profile_add_seconds("policySampleSeconds", time.perf_counter() - phase_started)
        self._profile_set("policyCount", len(policies))
        phase_started = time.perf_counter()
        base_components = agent.efe.evaluate_batch(belief, policies)
        self._profile_add_seconds("baseEFESeconds", time.perf_counter() - phase_started)
        believed_coverage = float(belief.mean[0].item())
        stop_indices = [i for i, policy in enumerate(policies) if policy.actions[0].stop]
        non_stop_indices = [i for i, policy in enumerate(policies) if not policy.actions[0].stop]
        brush_inferences: dict[int, BrushPreparationInference] = {}
        for index in non_stop_indices:
            inference = self._infer_brush_preparation(policies[index].actions[0])
            brush_inferences[index] = inference
            policies[index] = replace(
                policies[index],
                brush_preparation=inference.selected,
            )
        # FROZEN on the declared config constant, never the learned belief mean.
        # This is only the forecast-budget ordering, a declared fixed heuristic
        # BELOW the painting-policy boundary; a learned precision must not choose
        # which candidates get an expensive execution forecast, or gamma would
        # select the evidence set from which gamma is itself estimated.
        non_stop_indices = sorted(
            non_stop_indices,
            key=lambda i: (
                self.config.policy_precision * base_components[i].total
                - brush_inferences[i].log_evidence
            ),
        )
        forecast_budget = max(1, self.config.motor_forecast_candidates)
        components: list[EFEComponents] = list(base_components)
        rejections = 0
        motor_primitive_candidates = 0
        forecasted_indices: set[int] = set()
        active_indices: list[int] = list(stop_indices)
        policy_gamma = self.precision_ledger.mean(POLICY_PRECISION_KEY)
        # The modality weights are read ONCE per planning round, so every
        # candidate's motor EFE is scaled by the same belief mean and normalizer.
        modality_weights = self.precision_ledger.weights()
        painting_log_evidence = {
            index: -policy_gamma * base_components[index].total for index in stop_indices
        }
        forecast_cache: dict[tuple[object, ...], ExecutionForecast] = {}

        def forecast_index(index: int) -> bool:
            nonlocal rejections, motor_primitive_candidates
            policy = policies[index]
            first_action = policy.actions[0]
            if first_action.stop:
                return True
            brush_inference = brush_inferences[index]
            alternatives: list[tuple[Policy, EFEComponents, bool]] = []
            motor_policies = motor_realization_policy_alternatives(policy, self.config)
            motor_primitive_candidates += len(motor_policies)
            self._profile_increment("candidateMotorRealizations", len(motor_policies))
            forecasts = self._forecast_motor_realizations(
                body_snapshot,
                first_action,
                motor_policies,
                canvas_summary_state,
                forecast_cache,
            )
            for motor_policy, forecast in zip(motor_policies, forecasts):
                primitive = motor_policy.motor_primitive
                rescore_started = time.perf_counter()
                kind = primitive.kind if primitive is not None else "cartesian_ik"
                motor_terms = motor_efe_terms(
                    forecast,
                    self.config,
                    reliability_inflation=self.motion_reliability.expected_inflation(kind),
                    reliability_epistemic_nats=self.motion_reliability.epistemic_nats(kind),
                    weights=modality_weights,
                )
                mean = torch.tensor(forecast.next_state_mean, device=agent.device)
                variance = torch.tensor(forecast.next_state_variance, device=agent.device)
                comp = agent.efe.evaluate_with_first_transition(
                    belief,
                    motor_policy,
                    mean,
                    variance,
                    execution_uncertainty=forecast.execution_uncertainty,
                    contact_loss_probability=forecast.contact_loss_probability,
                    motor_overshoot=forecast.overshoot,
                    motor_feasible=forecast.feasible,
                    motor_risk=motor_terms.risk,
                    motor_ambiguity=motor_terms.ambiguity,
                    motor_epistemic_value=motor_terms.epistemic_value,
                    motor_efe_approximation=motor_terms.approximation,
                )
                self._profile_add_seconds("motorEFERescoreSeconds", time.perf_counter() - rescore_started)
                if comp.execution_forecast_used:
                    # Telemetry only: expose the two components of the single
                    # subtracted motor information gain. Gated on the forecast
                    # actually being used so a stop-first policy keeps logging
                    # zero motor terms.
                    comp = replace(
                        comp,
                        motor_mutual_information=motor_terms.mutual_information,
                        motor_reliability_novelty=motor_terms.reliability_novelty,
                    )
                if not forecast.feasible:
                    rejections += 1
                    comp = replace(comp, motor_feasible=False)
                alternatives.append((motor_policy, comp, bool(forecast.feasible)))
            feasible_alternatives = [entry for entry in alternatives if entry[2]]
            eligible = feasible_alternatives or alternatives
            log_evidence, motor_posterior = motor_realization_log_evidence(
                [entry[1].total for entry in eligible],
                [motor_policy_log_prior(entry[0], self.config) for entry in eligible],
                policy_gamma,
            )
            selected = int(np.argmax(motor_posterior))
            best_policy, best_component, best_feasible = eligible[selected]
            best_component = replace(
                best_component,
                motor_efe_approximation=(
                    f"{best_component.motor_efe_approximation}; motor realization posterior marginalized "
                    "within the painting policy"
                ).strip("; "),
            )
            policies[index] = best_policy
            components[index] = best_component
            # Approximation: preparation is marginalized under its explicit
            # mark-outcome EFE, while motor likelihoods are forecast only for
            # the modal preparation policy to keep rollout cost bounded.
            painting_log_evidence[index] = (
                log_evidence + brush_inference.log_evidence
            )
            forecasted_indices.add(index)
            if best_feasible:
                active_indices.append(index)
            return best_feasible

        has_feasible_non_stop = False
        for index in non_stop_indices[:forecast_budget]:
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop
        for index in non_stop_indices[forecast_budget:]:
            if has_feasible_non_stop:
                break
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop

        for index in non_stop_indices:
            if index not in forecasted_indices:
                components[index] = replace(base_components[index], motor_feasible=False)

        active_indices = list(dict.fromkeys(active_indices))
        if not active_indices:
            active_indices = [0]

        self.last_motor_rejections = rejections
        self.last_motor_primitive_candidates = motor_primitive_candidates
        phase_started = time.perf_counter()
        active_logits = torch.tensor(
            [
                policy_stop_log_prior(
                    policies[i], believed_coverage, self.config, self.gap_increment
                )
                + painting_log_evidence[i]
                + policy_log_priors[i]
                for i in active_indices
            ],
            device=agent.device,
        )
        active_posterior = torch.softmax(active_logits - active_logits.max(), dim=0)
        self._observe_precision_beliefs(
            components,
            brush_inferences,
            policy_log_priors,
            [i for i, policy in enumerate(policies) if not policy.actions[0].stop],
        )
        self._profile_set("motorRealizationMarginalization", "logsumexp over declared motor policy prior")
        self._profile_set(
            "brushPreparationMarginalization",
            "logsumexp over preserve/reload policy prior and conditional mark-outcome EFE; modal preparation used for motor rollout",
        )
        posterior_values = [0.0 for _ in policies]
        for index, prob in zip(active_indices, active_posterior.detach().cpu().tolist()):
            posterior_values[index] = prob
        ranked = sorted(
            zip(policies, components, posterior_values),
            key=lambda item: item[2],
            reverse=True,
        )
        self._profile_add_seconds("posteriorSeconds", time.perf_counter() - phase_started)
        return ranked

    def _infer_spatial_policy_with_execution_forecasts(
        self,
        body_snapshot: ArmPainterSim,
        policies: list[Policy] | None = None,
        policy_log_priors: list[float] | None = None,
    ) -> list[tuple[Policy, SpatialEFEComponents, float]]:
        assert isinstance(self.agent, SpatialActiveInferencePainter)
        assert isinstance(self.belief, SpatialCanvasState)
        agent = self.agent
        belief = self.belief
        phase_started = time.perf_counter()
        proposal_features, proposal_belief_source = agent.proposal_belief_features()
        supplied_policies = policies is not None
        policies = (
            agent.policy_sampler.sample(
                belief.coverage(self.config.paint_presence_threshold),
                belief_features=proposal_features,
            )
            if policies is None
            else list(policies)
        )
        # Receding-horizon local-passage candidates come from `PassageBelief`, a
        # transition prior with its own declared density, so they are outside the
        # amortized proposal's scope and carry no records.
        proposal_records: tuple[ProposalRecord | None, ...] = (
            tuple(agent.policy_sampler.last_proposal_records)
            if not supplied_policies
            else tuple(None for _ in policies)
        )
        if len(proposal_records) != len(policies):
            proposal_records = tuple(None for _ in policies)
        policy_log_priors = [0.0] * len(policies) if policy_log_priors is None else list(policy_log_priors)
        if len(policy_log_priors) != len(policies):
            raise ValueError("Policy log priors must align with candidate policies.")
        self._profile_add_seconds("policySampleSeconds", time.perf_counter() - phase_started)
        self._profile_set("policyCount", len(policies))
        phase_started = time.perf_counter()
        base_components = agent.efe.evaluate_batch(belief, policies)
        self._profile_add_seconds("baseEFESeconds", time.perf_counter() - phase_started)
        believed_coverage = belief.material_coverage_mean(self.config.paint_presence_threshold)
        # AMORTIZATION TARGET: the DECLARED base painting-policy posterior of spec
        # §10.5, `softmax(log p_stop - gamma G_base)`, over the full sampled
        # candidate set. Deliberately not the embodied-refined posterior below: that
        # one is hard-zeroed outside `active_indices` (stop plus at most
        # `motor_forecast_candidates` forecast continuations), so it has support on
        # a handful of candidates and would collapse the proposal onto the motor
        # budget rather than onto the agent's beliefs. Its cross-entropy is carried
        # for measurement so the gap is reported, not assumed.
        base_target = base_efe_policy_posterior(
            [component.total for component in base_components],
            [
                policy_stop_log_prior(policy, believed_coverage, self.config, self.gap_increment)
                for policy in policies
            ],
            self.config.policy_precision,
        )
        self._pending_proposal_batch = ProposalTrainingBatch(
            features=proposal_features,
            records=proposal_records,
            weights=tuple(float(value) for value in base_target),
            target_support_fraction=0.0,
            belief_feature_source=(
                proposal_belief_source
                if self._uses_spatial_planner()
                else BELIEF_SOURCE_SUMMARY
            ),
        )
        self._pending_proposal_batch.target_support_fraction = (
            self._pending_proposal_batch.modelled_mass()
        )
        stop_indices = [i for i, policy in enumerate(policies) if policy.actions[0].stop]
        non_stop_indices = [i for i, policy in enumerate(policies) if not policy.actions[0].stop]
        brush_inferences: dict[int, BrushPreparationInference] = {}
        for index in non_stop_indices:
            inference = self._infer_brush_preparation(policies[index].actions[0])
            brush_inferences[index] = inference
            policies[index] = replace(
                policies[index],
                brush_preparation=inference.selected,
            )
        # FROZEN on the declared config constant, never the learned belief mean.
        # This is only the forecast-budget ordering, a declared fixed heuristic
        # BELOW the painting-policy boundary; a learned precision must not choose
        # which candidates get an expensive execution forecast, or gamma would
        # select the evidence set from which gamma is itself estimated.
        non_stop_indices = sorted(
            non_stop_indices,
            key=lambda i: (
                self.config.policy_precision * base_components[i].total
                - brush_inferences[i].log_evidence
            ),
        )
        forecast_budget = max(1, self.config.motor_forecast_candidates)
        components: list[SpatialEFEComponents] = list(base_components)
        rejections = 0
        motor_primitive_candidates = 0
        forecasted_indices: set[int] = set()
        active_indices: list[int] = list(stop_indices)
        policy_gamma = self.precision_ledger.mean(POLICY_PRECISION_KEY)
        # The modality weights are read ONCE per planning round, so every
        # candidate's motor EFE is scaled by the same belief mean and normalizer.
        modality_weights = self.precision_ledger.weights()
        painting_log_evidence = {
            index: -policy_gamma * base_components[index].total for index in stop_indices
        }
        forecast_cache: dict[tuple[object, ...], ExecutionForecast] = {}
        rollout_grid_size = (
            pixel_material_from_state(belief).shape[-1]
            if self.config.spatial_transition_mode == "local_patch"
            else self.config.spatial_grid_size
        )
        material_shape = (
            self.config.spatial_material_channels,
            rollout_grid_size,
            rollout_grid_size,
        )

        def spatial_flat_state(working: ArmPainterSim) -> np.ndarray:
            state = spatial_canvas_state(working, self.config)
            if self.config.spatial_transition_mode == "local_patch":
                return pixel_material_from_state(state).reshape(-1)
            return state.flatten_mean()

        def forecast_index(index: int) -> bool:
            nonlocal rejections, motor_primitive_candidates
            policy = policies[index]
            first_action = policy.actions[0]
            if first_action.stop:
                return True
            brush_inference = brush_inferences[index]
            alternatives: list[tuple[Policy, SpatialEFEComponents, bool]] = []
            motor_policies = motor_realization_policy_alternatives(policy, self.config)
            motor_primitive_candidates += len(motor_policies)
            self._profile_increment("candidateMotorRealizations", len(motor_policies))
            forecasts = self._forecast_motor_realizations(
                body_snapshot,
                first_action,
                motor_policies,
                spatial_flat_state,
                forecast_cache,
            )

            rescore_started = time.perf_counter()
            first_transitions = []
            motor_terms_by_policy = []
            for motor_policy, forecast in zip(motor_policies, forecasts):
                next_material = forecast.next_state_mean.reshape(material_shape)
                mean = torch.tensor(next_material, device=agent.device, dtype=torch.float32)
                material_delta = torch.tensor(
                    forecast.canvas_delta_mean.reshape(material_shape),
                    device=agent.device,
                    dtype=torch.float32,
                )
                variance = torch.tensor(
                    self._spatial_material_variance_from_forecast(belief, next_material, forecast, body_snapshot),
                    device=agent.device,
                    dtype=torch.float32,
                )
                kind = (
                    motor_policy.motor_primitive.kind
                    if motor_policy.motor_primitive is not None
                    else "cartesian_ik"
                )
                motor_terms = motor_efe_terms(
                    forecast,
                    self.config,
                    reliability_inflation=self.motion_reliability.expected_inflation(kind),
                    reliability_epistemic_nats=self.motion_reliability.epistemic_nats(kind),
                    weights=modality_weights,
                )
                first_transitions.append((mean, variance, material_delta))
                motor_terms_by_policy.append(motor_terms)

            rescored = agent.efe.evaluate_batch_with_first_transitions(
                belief,
                motor_policies,
                first_transitions,
                execution_uncertainties=[forecast.execution_uncertainty for forecast in forecasts],
                contact_loss_probabilities=[forecast.contact_loss_probability for forecast in forecasts],
                motor_overshoots=[forecast.overshoot for forecast in forecasts],
                motor_feasibilities=[forecast.feasible for forecast in forecasts],
                motor_risks=[terms.risk for terms in motor_terms_by_policy],
                motor_ambiguities=[terms.ambiguity for terms in motor_terms_by_policy],
                motor_epistemic_values=[terms.epistemic_value for terms in motor_terms_by_policy],
                motor_efe_approximations=[terms.approximation for terms in motor_terms_by_policy],
            )
            self._profile_add_seconds("motorEFERescoreSeconds", time.perf_counter() - rescore_started)
            for motor_policy, forecast, motor_terms, comp in zip(
                motor_policies, forecasts, motor_terms_by_policy, rescored
            ):
                if comp.execution_forecast_used:
                    # Telemetry only: see the summary path above.
                    comp = replace(
                        comp,
                        motor_mutual_information=motor_terms.mutual_information,
                        motor_reliability_novelty=motor_terms.reliability_novelty,
                    )
                if not forecast.feasible:
                    rejections += 1
                    comp = replace(comp, motor_feasible=False)
                alternatives.append((motor_policy, comp, bool(forecast.feasible)))
            feasible_alternatives = [entry for entry in alternatives if entry[2]]
            eligible = feasible_alternatives or alternatives
            log_evidence, motor_posterior = motor_realization_log_evidence(
                [entry[1].total for entry in eligible],
                [motor_policy_log_prior(entry[0], self.config) for entry in eligible],
                policy_gamma,
            )
            selected = int(np.argmax(motor_posterior))
            best_policy, best_component, best_feasible = eligible[selected]
            best_component = replace(
                best_component,
                motor_efe_approximation=(
                    f"{best_component.motor_efe_approximation}; motor realization posterior marginalized "
                    "within the painting policy"
                ).strip("; "),
            )
            policies[index] = best_policy
            components[index] = best_component
            # Approximation: preparation is marginalized under its explicit
            # mark-outcome EFE, while motor likelihoods are forecast only for
            # the modal preparation policy to keep rollout cost bounded.
            painting_log_evidence[index] = (
                log_evidence + brush_inference.log_evidence
            )
            forecasted_indices.add(index)
            if best_feasible:
                active_indices.append(index)
            return best_feasible

        has_feasible_non_stop = False
        for index in non_stop_indices[:forecast_budget]:
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop
        for index in non_stop_indices[forecast_budget:]:
            if has_feasible_non_stop:
                break
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop

        for index in non_stop_indices:
            if index not in forecasted_indices:
                components[index] = replace(base_components[index], motor_feasible=False)

        active_indices = list(dict.fromkeys(active_indices))
        if not active_indices:
            active_indices = [0]

        self.last_motor_rejections = rejections
        self.last_motor_primitive_candidates = motor_primitive_candidates
        phase_started = time.perf_counter()
        active_logits = torch.tensor(
            [
                policy_stop_log_prior(
                    policies[i], believed_coverage, self.config, self.gap_increment
                )
                + painting_log_evidence[i]
                + policy_log_priors[i]
                for i in active_indices
            ],
            device=agent.device,
        )
        active_posterior = torch.softmax(active_logits - active_logits.max(), dim=0)
        self._observe_precision_beliefs(
            components,
            brush_inferences,
            policy_log_priors,
            [i for i, policy in enumerate(policies) if not policy.actions[0].stop],
        )
        self._profile_set("motorRealizationMarginalization", "logsumexp over declared motor policy prior")
        self._profile_set(
            "brushPreparationMarginalization",
            "logsumexp over preserve/reload policy prior and conditional mark-outcome EFE; modal preparation used for motor rollout",
        )
        posterior_values = [0.0 for _ in policies]
        for index, prob in zip(active_indices, active_posterior.detach().cpu().tolist()):
            posterior_values[index] = prob
        # PAIRED SAME-ROUND ATTRIBUTION. Both proposals generated candidates under
        # the same belief and were scored by the same expected free energy, so these
        # two masses read out which branch's candidates the posterior actually
        # selects. `policies[index] = best_policy` above replaces the object but
        # preserves its POSITION, so the index is a valid key into the records.
        learned_mass = 0.0
        hand_mass = 0.0
        for index, record in enumerate(proposal_records):
            if record is None:
                continue
            if record.source == "learned":
                learned_mass += float(posterior_values[index])
            elif record.source == "hand":
                hand_mass += float(posterior_values[index])
        self._pending_proposal_masses = (learned_mass, hand_mass)
        if self._pending_proposal_batch is not None:
            self._pending_proposal_batch.refined_weights = tuple(
                float(value) for value in posterior_values
            )
        selected_index = (
            int(active_indices[int(torch.argmax(active_posterior).item())])
            if active_indices
            else 0
        )
        self._pending_proposal_record = (
            proposal_records[selected_index] if selected_index < len(proposal_records) else None
        )
        ranked = sorted(
            zip(policies, components, posterior_values),
            key=lambda item: item[2],
            reverse=True,
        )
        self._profile_add_seconds("posteriorSeconds", time.perf_counter() - phase_started)
        return ranked

    def _spatial_material_variance_from_forecast(
        self,
        belief: SpatialCanvasState,
        next_material: np.ndarray,
        forecast: ExecutionForecast,
        body_snapshot: ArmPainterSim,
    ) -> np.ndarray:
        # Approximation: first-order propagation of execution dispersion into
        # material-field uncertainty. Spatial path covariance moves deposited
        # material across grid cells; pressure and contact-loss uncertainty
        # scale the deposited delta. This remains a predictive covariance, not a
        # scalar motor-ease objective.
        current_material = belief.material
        if current_material.shape != next_material.shape:
            current_material = pixel_material_from_state(belief)
        delta = next_material - current_material
        variance = np.full_like(next_material, 1e-6, dtype=np.float32)
        grid_size = max(1, int(next_material.shape[-1]))
        cell_width = max(1e-6, body_snapshot.canvas.width / grid_size)
        cell_height = max(1e-6, body_snapshot.canvas.height / grid_size)
        path_var_x = max(0.0, float(forecast.path_covariance[0])) / (cell_width * cell_width)
        path_var_z = max(0.0, float(forecast.path_covariance[1])) / (cell_height * cell_height)
        for channel in range(next_material.shape[0]):
            grad_z, grad_x = np.gradient(next_material[channel].astype(np.float64))
            variance[channel] += (path_var_x * grad_x * grad_x + path_var_z * grad_z * grad_z).astype(np.float32)
        contact_var = forecast.contact_loss_probability * (1.0 - forecast.contact_loss_probability)
        pressure_denominator = max(1e-6, forecast.target_pressure_mean * forecast.target_pressure_mean)
        pressure_var = max(0.0, forecast.pressure_variance) / pressure_denominator
        variance += np.asarray((contact_var + pressure_var) * delta * delta, dtype=np.float32)
        return np.clip(variance, 1e-8, 1.0).astype(np.float32)

    def _forecast_action(
        self,
        sim: ArmPainterSim | None,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None = None,
        brush_preparation: BrushPreparationPolicy | None = None,
        brush_belief: BrushLoadBelief | None = None,
    ) -> ExecutionForecast | None:
        if sim is None or action.stop:
            return None
        return forecast_stroke_execution(
            sim,
            action,
            canvas_summary_state,
            motor_primitive=motor_primitive,
            dt=1.0 / 45.0,
            brush_reload=(
                brush_preparation is None
                or brush_preparation.kind == "reload"
            ),
            brush_belief=brush_belief,
        )

    def diagnostics(self) -> dict[str, Any]:
        action = asdict(self.current.action) if self.current is not None else None
        efe = asdict(self.last_components) if self.last_components is not None else None
        vfe = asdict(self.agent.last_vfe) if self.agent.last_vfe is not None else None
        camera_vfe = (
            asdict(self.agent.last_camera_vfe)
            if isinstance(self.agent, SpatialActiveInferencePainter)
            and self.agent.last_camera_vfe is not None
            else None
        )
        posterior_values = [prob for _, _, prob in self.last_ranked]
        posterior_entropy = float(
            -sum(prob * np.log(max(prob, 1e-12)) for prob in posterior_values)
        ) if posterior_values else 0.0
        passage_values = [prob for policy, _, prob in self.last_ranked if policy.passage is not None]
        passage_posterior_mass = float(sum(passage_values)) if passage_values else 0.0
        passage_plan_values = [prob for policy, _, prob in self.last_ranked if policy.passage_plan is not None]
        passage_plan_posterior_mass = float(sum(passage_plan_values)) if passage_plan_values else 0.0
        motor_values = [prob for policy, _, prob in self.last_ranked if policy.motor_primitive is not None]
        motor_posterior_mass = float(sum(motor_values)) if motor_values else 0.0
        spatial_belief = (
            spatial_state_diagnostics(self.belief, self.config)
            if isinstance(self.belief, SpatialCanvasState)
            else None
        )
        mark_events = (
            infer_mark_event_belief(self.belief, self.config).diagnostics()
            if isinstance(self.belief, SpatialCanvasState)
            else None
        )
        composition = None
        if isinstance(self.agent, SpatialActiveInferencePainter) and self.agent.composition is not None:
            composition = {
                "currentBeliefGap": self._cached_belief_gap,
                "gapPrecision": self.precision_ledger.mean("composition_gap"),
                "gapPrecisionPrior": self.config.composition_gap_precision,
                "gapProgress": self.gap_increment.summary(),
                "localBaselineEnabled": self.config.composition_local_baseline_enabled,
                "lastTrainingLoss": self.agent.last_composition_loss,
                "lastTransitionTrainingLoss": self.agent.last_hierarchy_transition_loss,
                "lastPassageTrajectoryLoss": self.agent.last_passage_trajectory_loss,
                "passageTrajectoryEvaluation": self.agent.last_passage_trajectory_evaluation,
                "passageReplaySize": len(self.agent.passage_replay),
                "passageStepReplaySize": len(self.agent.passage_step_replay),
                "topPolicyPassageTrajectory": self._cached_passage_trajectory,
                "hierarchy": self.agent.composition.diagnostics(),
                "declaredAs": (
                    "structural prior p*(s_T) ~ exp(precision * compression_gap); gap = "
                    "hierarchical ELBO minus the best member of a declared parameter-free "
                    "context-free baseline family (iid-cell Gaussian; 3x3 hollow-neighbourhood "
                    "local Markov), nats/cell-channel over all material channels"
                ),
            }
        oracle_mode = self.observation_access_mode == ORACLE_OBSERVATION_ACCESS_MODE
        return {
            "enabled": self.enabled,
            "stopped": self.stopped,
            "planning": self.planning,
            "observationBoundary": {
                "baseline": (
                    ORACLE_OBSERVATION_BASELINE_ID
                    if oracle_mode
                    else SENSOR_OBSERVATION_BASELINE_ID
                ),
                "mode": self.observation_access_mode,
                "sensorEquivalent": False,
                "modelAccessBlocked": self._observation_boundary_blocked,
                "cameraLikelihood": CAMERA_SPATIAL_LIKELIHOOD_VERSION,
                "cameraPosteriorConnected": isinstance(
                    self.agent, SpatialActiveInferencePainter
                ),
                "cameraExposureCount": self.camera_observation_count,
                "materialStateAccess": (
                    "exact VerticalCanvas fields and deterministic transforms; "
                    "explicit diagnostic-only exception"
                    if oracle_mode
                    else "denied"
                ),
                "bodyForecastInitialization": (
                    "deep copy of ArmPainterSim process state; explicit "
                    "diagnostic-only exception"
                    if oracle_mode
                    else "denied"
                ),
                "blockedReason": (
                    (
                        "camera-conditioned painting posterior is implemented; "
                        "sensor-conditioned body posterior and motor-forecast "
                        "initialization, plus action-conditioned live observation "
                        "scheduling, are not connected"
                        if isinstance(self.agent, SpatialActiveInferencePainter)
                        else "camera likelihood requires the spatial_material planner; "
                        "summary mode is obsolete"
                    )
                    if self._observation_boundary_blocked
                    else None
                ),
            },
            "plannerError": self._pending_error,
            "lastPlanningSeconds": self.last_planning_seconds,
            "currentPlanningSeconds": self._current_planning_seconds(),
            "planningProfile": dict(self.last_planning_profile),
            "lastTrainingSeconds": self.last_training_seconds,
            "checkpoint": {
                "path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
                "status": self.checkpoint_status,
                "loaded": self.checkpoint_loaded,
                "lastSaved": self.checkpoint_last_saved,
                "lastError": self.checkpoint_last_error,
                "saveEveryTransitions": self.checkpoint_save_every_transitions,
                "architecture": dict(self.checkpoint_architecture),
            },
            "postStrokeRetractRemaining": self._post_stroke_retract_remaining,
            "planningScope": self._planning_scope(),
            "holdScope": self._hold_scope,
            "contactReleaseCount": self._contact_release_count,
            "passageQueueLength": len(self._passage_queue),
            "activePassage": asdict(self._active_passage) if self._active_passage is not None else None,
            "activePassagePlan": asdict(self._active_passage_plan) if self._active_passage_plan is not None else None,
            "activePassageBelief": self._passage_belief.diagnostics() if self._passage_belief is not None else None,
            "activePassageTotalStrokes": self._active_passage_total_strokes,
            "activePassageCompletedStrokes": self._active_passage_completed_strokes,
            "minimumStopCoverage": self.config.minimum_stop_coverage,
            "lastStopBlocked": self.last_stop_blocked,
            "motorRejections": self.last_motor_rejections,
            "motorPrimitiveCandidateCount": self.last_motor_primitive_candidates,
            "motorPrimitivePosteriorMass": motor_posterior_mass,
            "executionForecast": self._execution_forecast_diagnostics(),
            "stateRepresentationLifecycle": (
                {
                    "kind": SUMMARY_PLANNER_STATE_KIND,
                    "status": "obsolete_compatibility_fixture",
                    "architecturalRole": (
                        "regression, tractable-reference, and checkpoint compatibility only"
                    ),
                    "notValidAs": (
                        "a highest-level painting belief or image-making representation"
                    ),
                    "replacementDirection": (
                        "learned multiscale perceptual latents optimized and validated "
                        "for prediction; spatial_material is only an interim low-level baseline"
                    ),
                }
                if self.config.planner_state_kind
                == SUMMARY_PLANNER_STATE_KIND
                else {
                    "kind": SPATIAL_MATERIAL_PLANNER_STATE_KIND,
                    "status": "provisional_low_level_material_baseline",
                    "architecturalRole": (
                        "localized material transition prediction and simulator research"
                    ),
                    "notValidAs": (
                        "the final abstract or painting-level latent hierarchy"
                    ),
                    "replacementDirection": (
                        "learned camera-conditioned perceptual and slower predictive latents"
                    ),
                }
            ),
            "brushLoading": {
                "beliefs": {
                    key: asdict(belief)
                    for key, belief in self.brush_load_beliefs.items()
                },
                "currentPreparation": (
                    asdict(self.current.brush_preparation)
                    if (
                        self.current is not None
                        and self.current.brush_preparation is not None
                    )
                    else None
                ),
                "lastPreparationInference": (
                    asdict(self.last_brush_preparation)
                    if self.last_brush_preparation is not None
                    else None
                ),
                "lastLoadObservationVFE": (
                    asdict(self.brush_loading_model.last_vfe)
                    if self.brush_loading_model.last_vfe is not None
                    else None
                ),
                "modelAccess": (
                    "persistent q(load, black_fraction) per dedicated brush; "
                    "no exact process brush state"
                ),
            },
            "stateRepresentation": self._state_representation_diagnostics(),
            "transitionModel": self._transition_model_diagnostics(),
            "spatialTransitionMode": (
                self.config.spatial_transition_mode
                if isinstance(self.belief, SpatialCanvasState)
                else None
            ),
            # The posterior mean of the policy precision BELIEF; the declared
            # constant it was seeded from is reported beside it.
            "policyPrecision": self.precision_ledger.mean(POLICY_PRECISION_KEY),
            "policyPrecisionPrior": self.config.policy_precision,
            "precisionBeliefs": self.precision_ledger.summary(),
            "gapProgress": self.gap_increment.summary(),
            "posteriorEntropy": posterior_entropy,
            "passageCandidateCount": len(passage_values),
            "passagePosteriorMass": passage_posterior_mass,
            "passagePlanCandidateCount": len(passage_plan_values),
            "passagePlanPosteriorMass": passage_plan_posterior_mass,
            "trainedTransitions": self.trained_transitions,
            "lastTrainingLoss": self.last_training_loss,
            "belief": self._belief_diagnostics(),
            "spatialBelief": spatial_belief,
            "markEvents": mark_events,
            "composition": composition,
            "policyProposal": self._policy_proposal_diagnostics(),
            # Nothing is computed here. The block was measured once during
            # bootstrap; diagnostics() is polled from the web thread while
            # background training runs, so a compression_gap call here would
            # contend with it. Stays None in summary mode, in sensor mode, and on
            # checkpoint resume (where bootstrap never ran).
            "compositionBootstrap": self._bootstrap_composition,
            "strokeCount": self.stroke_count,
            "executing": action,
            "executingMotorPrimitive": (
                asdict(self.current.motor_primitive)
                if self.current is not None and self.current.motor_primitive is not None
                else None
            ),
            "executingBrushPreparation": (
                asdict(self.current.brush_preparation)
                if self.current is not None and self.current.brush_preparation is not None
                else None
            ),
            "motionReliability": self.motion_reliability.summary(),
            "efe": efe,
            "vfe": vfe,
            "cameraVfe": camera_vfe,
            "phase": self.phase_label(),
            "posterior": self.current.posterior if self.current is not None else None,
            "topPolicies": [
                {
                    "length": len(policy.actions),
                    "firstStop": policy.actions[0].stop,
                    "passage": asdict(policy.passage) if policy.passage is not None else None,
                    "passagePlan": asdict(policy.passage_plan) if policy.passage_plan is not None else None,
                    "passageStartIndex": policy.passage_start_index,
                    "passageBoundaries": [list(boundary) for boundary in policy.passage_boundaries],
                    "motorPrimitive": asdict(policy.motor_primitive) if policy.motor_primitive is not None else None,
                    "brushPreparation": (
                        asdict(policy.brush_preparation)
                        if policy.brush_preparation is not None
                        else None
                    ),
                    "posterior": prob,
                    "total": comp.total,
                    "terminalRisk": comp.terminal_risk,
                    "ambiguity": comp.ambiguity,
                    "epistemicValue": comp.epistemic_value,
                    "terminalEntropy": comp.terminal_entropy,
                    "pragmaticValue": comp.pragmatic_value,
                    "transitionRisk": comp.transition_risk,
                    "transitionAmbiguity": comp.transition_ambiguity,
                    "motorRisk": comp.motor_risk,
                    "motorAmbiguity": comp.motor_ambiguity,
                    "motorEpistemicValue": comp.motor_epistemic_value,
                    "motorMutualInformation": comp.motor_mutual_information,
                    "motorReliabilityNovelty": comp.motor_reliability_novelty,
                    "motorEFEApproximation": comp.motor_efe_approximation,
                    "compositionGap": getattr(comp, "composition_gap", 0.0),
                    "compositionRisk": getattr(comp, "composition_risk", 0.0),
                    "canvasTransitionRisk": getattr(comp, "canvas_transition_risk", 0.0),
                    "relationalTransitionRisk": getattr(comp, "relational_transition_risk", 0.0),
                    "passageCanvasTrajectoryRisk": getattr(comp, "passage_canvas_trajectory_risk", 0.0),
                    "passageRelationalTrajectoryRisk": getattr(
                        comp, "passage_relational_trajectory_risk", 0.0
                    ),
                    "passageTrajectoryObservationCount": getattr(
                        comp, "passage_trajectory_observation_count", 0.0
                    ),
                    "terminalCoverageMean": comp.terminal_coverage_mean,
                    "rolloutMode": getattr(comp, "rollout_mode", "dense_grid"),
                    "rolloutGridSize": getattr(comp, "rollout_grid_size", 0),
                    "activePatchAreaFraction": getattr(comp, "active_patch_area_fraction", 0.0),
                    "localTransitionSteps": getattr(comp, "local_transition_steps", 0),
                    "sequentialPatchSteps": getattr(comp, "sequential_patch_steps", 0),
                    "identityTransitionApproximation": getattr(comp, "identity_transition_approximation", ""),
                    "hierarchyTransitionMode": getattr(comp, "hierarchy_transition_mode", "unavailable"),
                    "passageTrajectorySteps": getattr(comp, "passage_trajectory_steps", 0),
                    "executionUncertainty": comp.execution_uncertainty,
                    "contactLossProbability": comp.contact_loss_probability,
                    "motorOvershoot": comp.motor_overshoot,
                    "motorFeasible": comp.motor_feasible,
                }
                for policy, comp, prob in self.last_ranked[:5]
            ],
        }

    def _policy_proposal_diagnostics(self) -> dict[str, Any]:
        """The falsifiable Feature-D payload, addressing RESEARCH_CHARTER H4.

        Wire format, not a debug dump: every value is a plain
        float/int/str/bool/None so `web_server`'s bare `json.dumps` cannot fail,
        and a diagnostic failure degrades to a status string rather than taking
        down the state poll. Read-only throughout -- nothing in the generative
        model, the EFE, or the policy posterior consumes any of it.
        """

        declared = (
            "amortized proposal q_proposal(z_pi | q(z_canvas), q(z_relational)) trained by "
            "self-normalized importance-weighted maximum likelihood toward the declared base "
            "painting-policy posterior softmax(log p_stop - gamma G_base); the continuous, "
            "belief-conditioned generalization of the reference habit prior "
            "E = softmax(gamma log(counts + 1)) "
            "(active_inference.core.pomdp_extensions.habit_prior_from_counts). A PROPOSAL, not a "
            "prior: never summed into the policy posterior, never in any EFE or VFE term, never a "
            "preference. The hand-written proposal is permanently mixed in at (1 - mix) in the same "
            "planning round as a paired control."
        )
        block: dict[str, Any] = {
            "status": "disabled",
            "declaredAs": declared,
            "researchHypothesis": "RESEARCH_CHARTER H4",
            "mixtureWeight": float(self.config.learned_proposal_mix),
            "effectiveMixtureWeight": 0.0,
            "updateCount": 0,
            "lastTrainingLoss": None,
            "lastTargetName": BASE_TARGET_NAME,
            "lastTargetSupportFraction": None,
            "refinedTargetCrossEntropy": None,
            "divergenceNats": None,
            "markDivergenceNats": None,
            "divergenceSamples": 0,
            "outOfHandSupportFraction": None,
            "selectedMeanLogLikelihoodLearned": None,
            "selectedMeanLogLikelihoodHandWritten": None,
            "selectedLogLikelihoodAdvantage": None,
            "selectedSampleCount": 0,
            "learnedPosteriorMass": 0.0,
            "handWrittenPosteriorMass": 0.0,
            "learnedCandidateCount": 0,
            "beliefFeatureSource": BELIEF_SOURCE_SUMMARY,
            "skippedNoBeliefUpdates": 0,
            "skippedDegenerateTargetUpdates": 0,
            "inputDimensions": 0,
            "approximation": (
                "target is the base-EFE posterior over the SAMPLED candidate support only; "
                "direction density truncated to three wraps; the hand-written density's "
                "coverage-cell boundary atoms evaluated pre-clip; plan-family compounds and "
                "the planning-depth categorical stay hand-written and cancel"
            ),
        }
        try:
            if not self._uses_spatial_planner():
                block["status"] = "unavailable_summary_planner"
                return block
            agent = self.agent
            if not isinstance(agent, SpatialActiveInferencePainter) or agent.policy_proposal is None:
                return block
            cached = self._cached_policy_proposal
            if isinstance(cached, dict) and cached.get("status") == "diagnostic_failed":
                block["status"] = "diagnostic_failed"
                block["lastError"] = str(cached.get("lastError"))
                return block
            update_count = agent.policy_proposal.update_count
            block["status"] = "learned" if update_count > 0 else "untrained"
            block["updateCount"] = int(update_count)
            block["lastTrainingLoss"] = (
                float(agent.last_proposal_loss) if agent.last_proposal_loss is not None else None
            )
            block["lastTargetSupportFraction"] = (
                float(agent.last_proposal_target_support_fraction)
                if agent.last_proposal_target_support_fraction is not None
                else None
            )
            block["skippedNoBeliefUpdates"] = int(agent.proposal_no_belief_skips)
            block["skippedDegenerateTargetUpdates"] = int(agent.proposal_degenerate_target_skips)
            block["inputDimensions"] = int(agent.policy_proposal.input_dim)
            block["beliefFeatureSource"] = str(agent.last_proposal_belief_source)
            block["learnedCandidateCount"] = int(
                agent.policy_sampler.last_learned_candidate_count
            )
            learned_mass, hand_mass = self._pending_proposal_masses
            block["learnedPosteriorMass"] = float(learned_mass)
            block["handWrittenPosteriorMass"] = float(hand_mass)
            # Effective, not declared: a nonzero mix with a fallback belief source
            # or a still-untrained network is not an emitting learned proposal.
            block["effectiveMixtureWeight"] = (
                float(self.config.learned_proposal_mix)
                if block["status"] == "learned"
                else 0.0
            )
            if isinstance(cached, dict):
                for key in (
                    "divergenceNats",
                    "markDivergenceNats",
                    "divergenceSamples",
                    "outOfHandSupportFraction",
                    "selectedMeanLogLikelihoodLearned",
                    "selectedMeanLogLikelihoodHandWritten",
                    "selectedLogLikelihoodAdvantage",
                    "selectedSampleCount",
                    "refinedTargetCrossEntropy",
                    "beliefFeatureSource",
                ):
                    if key not in cached:
                        continue
                    value = cached[key]
                    if value is None:
                        block[key] = None
                    elif key in {"divergenceSamples", "selectedSampleCount"}:
                        block[key] = int(value)
                    elif key == "beliefFeatureSource":
                        block[key] = str(value)
                    else:
                        block[key] = float(value)
            return block
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            return {
                "status": "diagnostic_failed",
                "declaredAs": declared,
                "researchHypothesis": "RESEARCH_CHARTER H4",
                "lastError": repr(exc),
            }

    def _belief_diagnostics(self) -> dict[str, object]:
        if isinstance(self.belief, SpatialCanvasState):
            std = np.sqrt(np.exp(np.clip(self.belief.logvar, -30.0, 20.0)))
            return {
                "names": list(MATERIAL_CHANNELS[: self.belief.material.shape[0]]),
                "mean": self.belief.material.mean(axis=(1, 2)).astype(float).tolist(),
                "std": std.mean(axis=(1, 2)).astype(float).tolist(),
            }
        assert isinstance(self.belief, GaussianBelief)
        belief_std = torch.sqrt(self.belief.logvar.exp())
        return {
            "names": [
                "coverage",
                "mean_thickness",
                "max_thickness",
                "mean_wetness",
                "overlap_fraction",
                "mean_ground_contrast",
            ],
            "mean": self.belief.mean.detach().cpu().tolist(),
            "std": belief_std.detach().cpu().tolist(),
        }

    def _state_representation_diagnostics(self) -> str:
        if isinstance(self.belief, SpatialCanvasState):
            if self.config.spatial_transition_mode == "local_patch":
                pixel_grid = pixel_material_from_state(self.belief).shape[-1]
                return (
                    f"Spatial Gaussian q(s) with pixel-local rollouts over {pixel_grid}x{pixel_grid} "
                    f"material fields and coarse {self.belief.grid_size}x{self.belief.grid_size} "
                    "planner observations, persistent spatial canvas and relational passage-level posteriors; "
                    "six canvas summaries are diagnostics only"
                )
            return (
                f"Spatial Gaussian q(s_grid) over {self.belief.grid_size}x{self.belief.grid_size} "
                "material fields: thickness, wetness, conserved black_mass, surface_tone, ground_contrast, material_coverage; "
                "six canvas summaries are diagnostics only"
            )
        return (
            "OBSOLETE compatibility fixture: Gaussian q(s) over six "
            "hand-selected canvas aggregates; non-spatial and not a valid "
            "highest-level painting representation"
        )

    def _transition_model_diagnostics(self) -> str:
        if isinstance(self.agent, SpatialActiveInferencePainter):
            if self.config.spatial_transition_mode == "local_patch":
                return (
                    "learned LocalSpatialDynamicsEnsemble p_theta(s_patch_next | s_patch, rasterized stroke patch) "
                    "with identity transition prior outside local support"
                )
            return (
                "learned SpatialDynamicsEnsemble p_theta(s_grid_next | s_grid, rasterized stroke) "
                "trained from this arm/canvas simulator"
            )
        return "learned DynamicsEnsemble p_theta(s_next | s, realized execution forecast) trained from this arm/canvas simulator"

    def _execution_forecast_diagnostics(self) -> dict[str, object] | None:
        forecast = self.current.forecast if self.current is not None else self.last_execution_forecast
        return forecast.diagnostics(include_state_fields=False) if forecast is not None else None

    def phase_label(self) -> str:
        if self._observation_boundary_blocked:
            return "sensor_boundary_blocked"
        if self.current is not None:
            return execution_phase(self.current)
        if self.stopped:
            return "stop"
        if self._post_stroke_retract_remaining > 0.0:
            return "local_passage_hold" if self._passage_queue else "return_center"
        if self._passage_queue:
            return "local_passage_hold"
        return "global_planning"

    def _planning_scope(self) -> str:
        if self.current is not None:
            return "stroke_execution"
        if self._passage_queue or self._active_passage_total_strokes > 0:
            return "passage_local"
        return "global"


def canvas_summary_state(sim: ArmPainterSim) -> np.ndarray:
    canvas = sim.canvas
    coverage = canvas.coverage_field()
    painted = canvas.thickness > 0.02
    overlap = canvas.thickness > sim.config.thickness_scale
    return np.asarray(
        [
            float(coverage.mean()),
            float(canvas.thickness.mean()),
            float(canvas.thickness.max(initial=0.0)),
            float(canvas.wetness.mean()),
            float(overlap.mean()),
            float((canvas.ground_contrast_field() * painted).mean()),
        ],
        dtype=np.float32,
    )


def execute_stroke_action(
    sim: ArmPainterSim,
    action: StrokeAction,
    dt: float = 1.0 / 120.0,
    *,
    reload: bool = True,
) -> None:
    ex = StrokeExecution(
        action=action,
        efe=EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        posterior=1.0,
        initial_state=canvas_summary_state(sim),
    )
    ex.timing = adaptive_stroke_timing(sim, action)
    sim.select_brush(action.tone)
    sim.deposition_amount = action.amount
    if reload:
        sim.load_brush(1.0, action.tone)
    ex.controller.reset(sim, ex.action, ex.timing)
    ex.initialized = True
    while ex.t < ex.total:
        ex.t += dt
        command = ex.controller.command(sim, ex.action, ex.t, dt, ex.timing)
        sim.control_damping_multiplier = 1.0
        sim.set_target(command.pose)
        sim.intended_contact_pressure = command.intended_pressure
        sim.brush_flow = command.reference.flow
        sim.step(dt)


def execution_phase(ex: StrokeExecution | None) -> str:
    if ex is None:
        return "planning"
    if ex.t < ex.approach:
        return "approach"
    if ex.t < ex.approach + ex.press:
        return "press"
    if ex.t < ex.approach + ex.press + ex.paint:
        return "paint"
    return "lift"


def pose_for_execution(sim: ArmPainterSim, ex: StrokeExecution) -> tuple[ArmPose, float]:
    reference = stroke_reference(ex.action, sim, ex.t, ex.timing)
    return pose_for_reference(reference), reference.pressure
