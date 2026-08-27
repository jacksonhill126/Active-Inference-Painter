"""Tests for the motion-manifold bootstrap of the composition hierarchy.

The declared claim: bootstrap marks are drawn from the agent's OWN reachable
motion manifold - joint-space sweeps integrated along the contact surface and
projected to canvas geometry by forward kinematics - so the first regularity the
canvas/relational likelihood sees is embodiment-induced rather than iid.

What is pinned here:

- sweeps stay inside the body's real joint limits and the contact depth band, so
  the manifold explored is the real one and not a clipped fiction;
- every emitted mark clears the minimum mark length, preserving the existing
  claim that a round brush over a short span paints a disc no real brush makes;
- the embodiment signature is CURVATURE MAGNITUDE (total turn), which is the
  property iid straight-mark sampling cannot produce. It is deliberately NOT
  pinned on the path/chord ratio: the measured median ratio is only ~1.02, so a
  ratio threshold would be a false pin;
- a single-joint sweep cannot paint, which is why a family only up-weights its
  joint inside an otherwise coordinated sweep;
- emitted marks come from the DECODED latent, so the passage machinery never
  trains on inconsistent latent/geometry pairs;
- both generators are genuinely reachable through config, and the bootstrap no
  longer clears the canvas mid-episode.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from active_painter import arm_agent_driver as driver_module
from active_painter.arm_agent_driver import (
    BOOTSTRAP_GENERATORS,
    ORACLE_OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
)
from active_painter.arm_control import ik_pose_for_canvas_point
from active_painter.arm_sim import (
    NEAR_SURFACE_TOLERANCE,
    ArmKinematics,
    ArmPainterSim,
    ArmPose,
    VerticalCanvas,
)
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.motion_manifold import (
    DEPTH_BAND_TRAVEL_FRACTION,
    JOINT_LIMITS,
    MANIFOLD_FAMILIES,
    MotionManifoldSampler,
    blank_probe_fields,
    iid_scatter_probe_fields,
    shuffled_probe_fields,
)
from active_painter.policies import PolicySampler, Policy, polyline_vertices
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.spatial_state import SpatialCanvasState, spatial_canvas_state


def manifold_config(**overrides) -> PainterConfig:
    """The bootstrap simulator's config, matching the driver's own defaults."""

    base = dict(
        canvas_size=64,
        candidate_policies=8,
        planning_horizon=4,
        policy_precision=0.35,
        batch_size=8,
        planner_state_kind="spatial_material",
        composition_enabled=True,
        composition_gap_precision=1.0,
        passage_trajectory_enabled=True,
    )
    base.update(overrides)
    return PainterConfig(**base)


def sample_sweeps(sampler: MotionManifoldSampler, count: int) -> list:
    sweeps = []
    attempts = 0
    while len(sweeps) < count and attempts < count * 12:
        attempts += 1
        sweep = sampler.sample_sweep()
        if sweep is not None:
            sweeps.append(sweep)
    return sweeps


def test_sweeps_respect_joint_limits() -> None:
    """Every integrated pose is legal, so the manifold explored is the real one.

    `ArmPose.clipped()` is the plant's own limit definition. If a sweep row
    needed clipping, the sampler would be exploring poses the body cannot hold
    and the "embodiment-induced structure" claim would be false.
    """

    cfg = manifold_config()
    lower = np.asarray([limit[0] for limit in JOINT_LIMITS])
    upper = np.asarray([limit[1] for limit in JOINT_LIMITS])
    total_rows = 0
    seen_families: set[str] = set()

    for family in MANIFOLD_FAMILIES:
        sampler = MotionManifoldSampler(cfg, seed=101)
        produced = 0
        for _ in range(40):
            sweep = sampler.sample_sweep(family=family)
            if sweep is None:
                continue
            produced += 1
            seen_families.add(sweep.family)
            for row in sweep.joint_path:
                pose = ArmPose(*(float(value) for value in row))
                clipped = pose.clipped()
                assert (pose.yaw, pose.pitch, pose.roll, pose.elbow) == (
                    clipped.yaw,
                    clipped.pitch,
                    clipped.roll,
                    clipped.elbow,
                )
                assert np.all(row >= lower) and np.all(row <= upper)
                total_rows += 1
            if produced >= 6:
                break
        assert produced >= 6, f"family {family} produced too few sweeps"

    assert seen_families == set(MANIFOLD_FAMILIES)
    assert total_rows >= 30

    # An undeclared family must fail loudly rather than degrade into an
    # unattributable coordinated sweep.
    with pytest.raises(ValueError, match="unknown motion-manifold famil"):
        MotionManifoldSampler(
            manifold_config(bootstrap_manifold_families=("wrist_dominant",)),
            seed=1,
        )


def test_sweeps_produce_non_degenerate_on_canvas_paths() -> None:
    """Sweeps stay in contact, stay on canvas, and emit real marks.

    The depth band's upper bound is strictly inside the bushing travel, so
    `VerticalCanvas.too_deep` -- an external hard limit -- can never fire on a
    sweep pose. The emitted-mark length floor preserves for this new mark source
    the claim `tests/test_arm_sim.py` makes about dwell dabs, which that file
    cannot reach because it only exercises `PolicySampler._stroke`.
    """

    cfg = manifold_config()
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    sampler = MotionManifoldSampler(cfg, seed=13, kinematics=sim.kinematics, canvas=sim.canvas)
    minimum = cfg.bootstrap_manifold_min_mark_length
    depth_lower = sim.canvas.distance - NEAR_SURFACE_TOLERANCE
    depth_upper = sim.canvas.distance + DEPTH_BAND_TRAVEL_FRACTION * sim.canvas.bushing_travel

    sweeps = sample_sweeps(sampler, 40)
    assert len(sweeps) >= 40
    for sweep in sweeps:
        assert sweep.normalized_path_length > 0.0
        assert len(sweep.joint_path) >= 3
        assert np.all(sweep.canvas_path >= -1e-9)
        assert np.all(sweep.canvas_path <= 1.0 + 1e-9)
        depths = sweep.tip_path[:, 1]
        assert np.all(depths >= depth_lower - 1e-9)
        assert np.all(depths <= depth_upper + 1e-9)
        assert not any(sim.canvas.too_deep(tip) for tip in sweep.tip_path)

    emitted = 0
    for _ in range(60):
        sampled = sampler.sample_marks()
        if sampled is None:
            continue
        _, actions = sampled
        for action in actions:
            length = float(np.hypot(action.x1 - action.x0, action.y1 - action.y0))
            assert length >= minimum - 1e-9, f"emitted a dwell dab of length {length:.4f}"
            emitted += 1
    assert emitted >= 40


def test_sweep_geometry_carries_curvature_the_iid_sampler_cannot() -> None:
    """The embodiment signature is total turn, and it is large.

    A revolute arm sweeping along the contact surface accumulates turn; the iid
    mark sampler emits straight segments whose total turn is identically zero, so
    this is the property that distinguishes the two sources. Thresholds sit well
    below the measured values (frac>0.3 rad ~ 0.73, mean |turn| ~ 0.94 rad).

    Deliberately NOT asserted on the path/chord ratio: the measured median is
    ~1.02, i.e. most sweeps are near-straight in gross shape, so a ratio
    threshold would pin a claim the mechanism does not support.
    """

    cfg = manifold_config()
    sampler = MotionManifoldSampler(cfg, seed=5)
    sweeps = sample_sweeps(sampler, 60)
    assert len(sweeps) >= 60

    turns = np.abs(np.asarray([sweep.total_turn_radians for sweep in sweeps]))
    assert float((turns > 0.3).mean()) >= 0.4
    assert float(turns.mean()) > 0.5

    # The iid baseline it is compared against: every mark is a straight segment,
    # so its within-mark turn is exactly zero by construction.
    iid = PolicySampler(cfg, seed=5)
    for _ in range(20):
        action = iid._stroke()
        assert (action.x1 - action.x0, action.y1 - action.y0) != (0.0, 0.0)


def test_sweep_sampling_is_deterministic_under_seed() -> None:
    """Same seed, same sweeps; and the agent's proposal stream is untouched.

    The second half is what keeps the attribution A/B clean: if the sweep sampler
    drew from `agent.policy_sampler.rng`, switching `bootstrap_generator` would
    also change the live planner's candidate stream, and any downstream
    behavioural difference between the two arms would be unattributable.
    """

    cfg = manifold_config()
    left = sample_sweeps(MotionManifoldSampler(cfg, seed=31), 6)
    right = sample_sweeps(MotionManifoldSampler(cfg, seed=31), 6)
    other = sample_sweeps(MotionManifoldSampler(cfg, seed=32), 6)

    assert len(left) == len(right) == 6
    for a, b in zip(left, right):
        assert a.family == b.family
        assert np.array_equal(a.joint_path, b.joint_path)
        assert np.array_equal(a.canvas_path, b.canvas_path)
    assert not all(
        a.joint_path.shape == b.joint_path.shape and np.array_equal(a.joint_path, b.joint_path)
        for a, b in zip(left, other)
    )

    left_marks = MotionManifoldSampler(cfg, seed=77).sample_marks()
    right_marks = MotionManifoldSampler(cfg, seed=77).sample_marks()
    assert left_marks is not None and right_marks is not None
    assert left_marks[0] == right_marks[0]
    assert left_marks[1] == right_marks[1]

    driver = ArmActiveInferenceDriver(
        config=manifold_config(bootstrap_generator="motion_manifold", bootstrap_episode_marks=3),
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )
    driver.bootstrap_transitions = 4
    before = driver.agent.policy_sampler.rng.bit_generator.state
    driver.bootstrap_dynamics()
    after = driver.agent.policy_sampler.rng.bit_generator.state
    block = driver.diagnostics()["compositionBootstrap"]
    assert block is not None
    if block["manifoldFallbackMarks"] == 0:
        assert after == before
    else:  # pragma: no cover - only when the sweep sampler cannot find a path
        assert after != before


@pytest.mark.parametrize("joint_index", range(4))
def test_single_joint_sweeps_cannot_paint(joint_index: int) -> None:
    """Holding three joints fixed cannot produce a mark-length span.

    Measured from the canvas-centre IK pose: sweeping one joint alone keeps the
    tip inside the near-contact band for at most ~15 degrees, whose whole
    admissible extent is a 0.17-normalized bounding box (0.02 for the elbow) -
    below the 0.20 minimum mark length. This is why `_family_weights` only
    up-weights a joint instead of isolating it, and this test exists so a future
    "simplification" back to strictly single-joint sweeps is caught here rather
    than by a blank canvas.
    """

    kinematics = ArmKinematics()
    canvas = VerticalCanvas(PainterConfig(canvas_size=48))
    pose = ik_pose_for_canvas_point(0.0, 0.0, canvas.distance)
    start = np.asarray([pose.yaw, pose.pitch, pose.roll, pose.elbow], dtype=np.float64)
    depth_lower = canvas.distance - NEAR_SURFACE_TOLERANCE
    depth_upper = canvas.distance + DEPTH_BAND_TRAVEL_FRACTION * canvas.bushing_travel
    half_width = 0.5 * canvas.width * 0.98
    half_height = 0.5 * canvas.height * 0.98

    admissible: list[tuple[float, float]] = []
    lower, upper = JOINT_LIMITS[joint_index]
    for delta in np.arange(-360.0, 360.001, 0.05):
        joints = start.copy()
        joints[joint_index] += delta
        if joints[joint_index] < lower or joints[joint_index] > upper:
            continue
        tip = kinematics.tip(ArmPose(*(float(value) for value in joints)))
        if not depth_lower <= float(tip[1]) <= depth_upper:
            continue
        if abs(float(tip[0])) > half_width or abs(float(tip[2])) > half_height:
            continue
        admissible.append(
            (
                float(tip[0]) / (canvas.width * 0.98) + 0.5,
                0.5 - float(tip[2]) / (canvas.height * 0.98),
            )
        )

    assert admissible, "the IK start pose itself must be in contact"
    points = np.asarray(admissible)
    span = float(np.hypot(*(points.max(axis=0) - points.min(axis=0))))
    assert span < PainterConfig().bootstrap_manifold_min_mark_length


def test_manifold_marks_come_from_the_decoded_polyline_latent() -> None:
    """A polyline latent must generate its own actions.

    `tests/test_policies.py` pins that a polyline passage decodes to connected
    constant-turn marks whose lengths sum to the latent's length. If the sampler
    emitted raw FK segments alongside a best-fit latent, the passage belief and
    the passage likelihood would train on latent/geometry pairs that do not
    correspond, and that invariant would silently be false for bootstrap data.
    """

    cfg = manifold_config()
    sampler = MotionManifoldSampler(cfg, seed=19)
    checked = 0
    for _ in range(60):
        sampled = sampler.sample_marks()
        if sampled is None:
            continue
        latent, actions = sampled
        if latent is None:
            assert len(actions) == 1
            continue
        checked += 1
        assert latent.kind == "polyline"
        assert latent.stroke_count == len(actions)
        decoded = PolicySampler(cfg, seed=0).passage_actions(latent)
        assert tuple(decoded) == actions
        policy = Policy(tuple(actions) + (StrokeAction.stop_action(),), passage=latent)
        assert policy.passage is latent
        vertices = polyline_vertices(latent)
        assert np.all(vertices >= 0.03 - 1e-9)
        assert np.all(vertices <= 0.97 + 1e-9)
    assert checked >= 10


@pytest.mark.parametrize("generator", BOOTSTRAP_GENERATORS)
def test_both_bootstrap_generators_are_reachable_through_config(
    generator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The iid baseline is genuinely still reachable, not merely declared.

    The charter requires the hand-written mechanism to stay available behind a
    declared flag so the two are measurable against each other. Counting real
    `PolicySampler._stroke` calls is the check that `random_stroke` still runs the
    previous code path, and that the manifold arm does not silently fall back
    onto it.
    """

    calls: list[int] = []
    original = PolicySampler._stroke

    def counted(self, coverage_field=None):
        calls.append(1)
        return original(self, coverage_field)

    monkeypatch.setattr(PolicySampler, "_stroke", counted)

    driver = ArmActiveInferenceDriver(
        config=manifold_config(bootstrap_generator=generator, bootstrap_episode_marks=3),
        bootstrap_transitions=6,
        bootstrap_train_steps=1,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )
    block = driver.diagnostics()["compositionBootstrap"]

    assert block is not None
    assert block["configuredGenerator"] == generator
    assert block["executedGenerator"] == generator
    if generator == "random_stroke":
        assert len(calls) >= 6
        assert block["manifoldSweepCount"] is None
    else:
        # Exact, not just "== 0": any iid mark in the manifold arm is a declared
        # fallback and must be accounted for rather than hidden.
        assert len(calls) == block["manifoldFallbackMarks"]
        assert block["manifoldSweepCount"] is not None
        assert block["manifoldSweepCount"] > 0


def test_bootstrap_does_not_clear_the_canvas_mid_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing happens only at episode boundaries, observed on the material.

    Asserted on observed coverage rather than on call counts, so it survives
    refactors: layering never reduces covered area
    (`tests/test_arm_sim.py::test_vertical_canvas_layering_does_not_increase_covered_area`
    guarantees the physics), so within an episode coverage can only be
    non-decreasing, and a mid-episode clear is the only way that can break.
    `sim.reset_pose()` calls `unload_all_brushes`, not `clear`, so `clear` is the
    correct hook for the call-index check.
    """

    episode_marks = 3
    mark_index = {"value": 0}
    clear_indices: list[int] = []
    coverage_trace: list[tuple[int, float]] = []

    original_execute = driver_module.execute_stroke_action
    original_clear = VerticalCanvas.clear

    def traced_execute(sim, action, dt=1.0 / 120.0, **kwargs):
        original_execute(sim, action, dt, **kwargs)
        coverage_trace.append((mark_index["value"], float(sim.canvas.material_coverage())))
        mark_index["value"] += 1

    def traced_clear(self):
        clear_indices.append(mark_index["value"])
        original_clear(self)

    monkeypatch.setattr(driver_module, "execute_stroke_action", traced_execute)
    monkeypatch.setattr(VerticalCanvas, "clear", traced_clear)

    ArmActiveInferenceDriver(
        config=manifold_config(bootstrap_episode_marks=episode_marks),
        bootstrap_transitions=2 * episode_marks,
        bootstrap_train_steps=1,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )

    assert mark_index["value"] == 2 * episode_marks
    assert clear_indices, "the episode boundary must still reset the canvas"
    assert all(index % episode_marks == 0 for index in clear_indices)
    # Coverage is non-decreasing inside every episode.
    for index, coverage in coverage_trace:
        if index % episode_marks == 0:
            previous = coverage
            continue
        assert coverage >= previous - 1e-9
        previous = coverage
    # And the whole first episode really did accumulate, i.e. the canvas the
    # composition replay received is a multi-mark canvas.
    first_episode = [value for index, value in coverage_trace if index < episode_marks]
    assert first_episode[-1] > 0.0


def test_composition_bootstrap_diagnostics_are_json_serializable() -> None:
    """The whole diagnostics dict stays JSON serializable with the new block.

    `tests/test_arm_agent_driver.py` already makes that claim for the dict as a
    whole; this extends it to `compositionBootstrap`, whose inputs are numpy
    arrays and torch tensors. Probe canvases must never be embedded.
    """

    driver = ArmActiveInferenceDriver(
        config=manifold_config(bootstrap_episode_marks=2),
        bootstrap_transitions=4,
        bootstrap_train_steps=1,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )
    diagnostics = driver.diagnostics()

    json.dumps(diagnostics)
    block = diagnostics["compositionBootstrap"]
    assert block is not None
    assert set(block["gap"]) == {
        "bootstrapped",
        "blank",
        "shuffledBootstrapped",
        "iidScatter",
    }
    for value in block["gap"].values():
        assert value is None or type(value) is float

    def assert_plain(value) -> None:
        assert not isinstance(value, (np.ndarray, np.generic, torch.Tensor))
        if isinstance(value, dict):
            for item in value.values():
                assert_plain(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                assert_plain(item)
        else:
            assert value is None or type(value) in (bool, int, float, str)

    assert_plain(block)
    assert "not a decision quantity" in str(block["declaredAs"]).lower()
    assert block["approximation"]


def test_whole_canvas_entry_point_does_not_open_passage_gates() -> None:
    """The episode-boundary budget trains the canvas/relational code only.

    `transition_update_count` and `passage_kind_update_counts` are the gates the
    kind-specific transition-EFE terms are conditioned on in
    `tests/test_canvas_hierarchy.py`. Bootstrap-only evidence must not open them,
    which is why `train_composition` deliberately does not call
    `_train_hierarchy_transitions`.
    """

    cfg = manifold_config(batch_size=2)
    agent = SpatialActiveInferencePainter(cfg, seed=3)
    assert agent.composition is not None
    sim = ArmPainterSim(PainterConfig(canvas_size=48, planner_state_kind="spatial_material"))
    state = spatial_canvas_state(sim, cfg)
    assert isinstance(state, SpatialCanvasState)

    for _ in range(cfg.batch_size):
        agent.add_composition_canvas(state)
    assert len(agent.composition_replay) == cfg.batch_size

    loss = agent.train_composition(gradient_steps=4)

    assert loss is not None and np.isfinite(loss)
    assert agent.last_composition_loss is not None
    assert np.isfinite(agent.last_composition_loss)
    assert int(agent.composition.transition_update_count.item()) == 0
    assert all(count == 0 for count in agent.composition.passage_kind_update_counts.values())
    assert agent.last_hierarchy_transition_loss is None
    assert agent.last_passage_trajectory_loss is None
    # A zero budget is an exact no-op, so the default configuration adds no cost.
    agent.last_composition_loss = None
    assert agent.train_composition(gradient_steps=0) is None
    assert agent.last_composition_loss is None


def test_probe_helpers_preserve_marginals_and_are_none_safe() -> None:
    """The shuffle is a valid null model, and the block is None-safe.

    `shuffled_probe_fields` must preserve every per-channel marginal exactly, or
    the acceptance margin would be measuring marginals rather than the
    hierarchical code: `flat_log_likelihood` depends on each channel's mean and
    variance alone. And wherever there is no composition hierarchy - summary
    mode, or the fail-closed sensor boundary - the evidence block must be None
    rather than a fabricated measurement.
    """

    cfg = manifold_config()
    rng = np.random.default_rng(0)
    fields = rng.uniform(0.0, 0.6, (5, cfg.spatial_material_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)).astype(np.float32)
    shuffled = shuffled_probe_fields(fields, np.random.default_rng(1))

    assert shuffled.shape == fields.shape
    assert np.allclose(shuffled.mean(axis=(2, 3)), fields.mean(axis=(2, 3)), atol=1e-6)
    assert np.allclose(shuffled.var(axis=(2, 3)), fields.var(axis=(2, 3)), atol=1e-6)
    for index in range(fields.shape[0]):
        for channel in range(fields.shape[1]):
            assert np.array_equal(
                np.sort(shuffled[index, channel].ravel()),
                np.sort(fields[index, channel].ravel()),
            )
    # Structure really is destroyed, so it is a null model and not a copy.
    assert not np.array_equal(shuffled, fields)

    blank = blank_probe_fields(cfg, count=3)
    assert blank.shape == (3, cfg.spatial_material_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    assert not blank.any()

    scatter = iid_scatter_probe_fields(cfg, np.random.default_rng(2), count=4)
    assert scatter.shape[0] == 4
    assert scatter.min() >= 0.0 and scatter.max() <= 1.0

    with pytest.warns(FutureWarning):
        summary_driver = ArmActiveInferenceDriver(
            config=PainterConfig(candidate_policies=2, planning_horizon=1, batch_size=4),
            bootstrap_transitions=2,
            bootstrap_train_steps=1,
            observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
        )
    assert summary_driver.diagnostics()["compositionBootstrap"] is None

    sensor_driver = ArmActiveInferenceDriver(
        config=manifold_config(),
        bootstrap_transitions=4,
        bootstrap_train_steps=1,
    )
    assert sensor_driver.observation_boundary_blocked is True
    assert sensor_driver.trained_transitions == 0
    assert sensor_driver.diagnostics()["compositionBootstrap"] is None
