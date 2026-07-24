"""Tests for the discrete active-inference Grid World agent (book §9.4–9.5, Alg. 9.5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.pomdp import evaluate_policy, one_hot
from active_inference.estimators.pomdp import (
    enumerate_policies,
    make_gridworld,
    simulate_pomdp_agent,
)


class TestGridWorld:
    def test_make_gridworld_shapes(self) -> None:
        m = make_gridworld(3, 3, goal=8)
        assert m.A.shape == (9, 9) and m.B.shape == (4, 9, 9)
        assert int(np.argmax(m.C)) == 8

    def test_transitions_clamp_at_walls(self) -> None:
        m = make_gridworld(3, 3, goal=8)
        # From top-left (cell 0), moving up/left stays in place; down→3, right→1.
        assert int(np.argmax(m.B[0][:, 0])) == 0   # up clamped
        assert int(np.argmax(m.B[2][:, 0])) == 0   # left clamped
        assert int(np.argmax(m.B[1][:, 0])) == 3   # down
        assert int(np.argmax(m.B[3][:, 0])) == 1   # right

    def test_agent_reaches_goal_optimally(self) -> None:
        # ISC: the EFE-planning agent navigates corner-to-corner in the minimum 4 steps.
        m = make_gridworld(3, 3, goal=8)
        res = simulate_pomdp_agent(m, start=0, horizon=4, gamma=8.0, max_steps=12)
        assert res.reached
        assert res.n_steps_to_goal == 4          # Manhattan distance 0→8 in a 3×3 grid
        assert res.states[-1] == 8

    def test_goal_reaching_policy_has_lower_efe(self) -> None:
        # The defining EFE property: a policy reaching the goal scores below one that doesn't.
        m = make_gridworld(3, 3, goal=8)
        belief = one_hot(0, 9)
        g_reach = evaluate_policy(m, belief, [1, 1, 3, 3], np.asarray(m.C))   # → cell 8
        g_stuck = evaluate_policy(m, belief, [0, 0, 2, 2], np.asarray(m.C))   # clamped at 0
        assert g_reach < g_stuck

    def test_enumerate_policies(self) -> None:
        pols = enumerate_policies(2, 3)
        assert len(pols) == 2 ** 3 and all(len(p) == 3 for p in pols)


# ---------------------------------------------------------------------------
# §10.1 — Trial-based parameter learning (Examples 10.1–10.3, Algorithm 10.1.1)
# ---------------------------------------------------------------------------
class TestParameterLearning:
    def test_learn_D_converges_to_example_10_1(self) -> None:
        # Book Example 10.1 / Eq. 8: d0=[1,1], s=[0.9,0.1] × 49 → d=[45.1,5.9], D≈[0.884,0.116].
        from active_inference.estimators.pomdp import learn_D_vector
        res = learn_D_vector([1.0, 1.0], np.tile([0.9, 0.1], (49, 1)))
        np.testing.assert_allclose(res.d_final, [45.1, 5.9], atol=1e-6)
        np.testing.assert_allclose(res.D_history[-1], [0.884, 0.116], atol=1e-3)
        # D moves monotonically from uniform toward the true proportion of state 0.
        assert res.D_history[0][0] == pytest.approx(0.5)
        assert np.all(np.diff(res.D_history[:, 0]) > 0)

    def test_learn_A_converges_to_true_likelihood(self) -> None:
        # Book Example 10.2 / Fig 10.1.3: learn a 2×2 A by counting state–observation pairs.
        from active_inference.estimators.pomdp import simulate_array_learning
        A_true = np.array([[0.7, 0.6], [0.3, 0.4]])
        B_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        res = simulate_array_learning(A_true=A_true, B_true=B_true,
                                      n_trials=5, steps_per_trial=1000, learn="A")
        assert res.final_A_error() < 0.05
        # Confidence (pseudocounts) grows monotonically over trials.
        conf_sums = res.a_confidence.sum(axis=(1, 2))
        assert np.all(np.diff(conf_sums) > 0)
        assert res.B_history is None

    def test_learn_B_converges_to_true_transition(self) -> None:
        # Book Example 10.3 / Fig 10.1.4: learn a 2×2 B by counting state→state transitions.
        from active_inference.estimators.pomdp import simulate_array_learning
        A_true = np.array([[0.7, 0.6], [0.3, 0.4]])
        B_true = np.array([[0.2, 0.6], [0.8, 0.4]])
        res = simulate_array_learning(A_true=A_true, B_true=B_true,
                                      n_trials=5, steps_per_trial=2000, learn="B")
        assert res.B_history is not None
        assert np.max(np.abs(res.B_history[-1] - B_true)) < 0.05
        # Learned B columns are valid categoricals.
        np.testing.assert_allclose(res.B_history[-1].sum(axis=0), [1.0, 1.0], atol=1e-9)

    def test_learn_rejects_bad_target(self) -> None:
        from active_inference.estimators.pomdp import simulate_array_learning
        with pytest.raises(ValueError):
            simulate_array_learning(A_true=np.eye(2), B_true=np.eye(2), learn="Z")

    def test_dirichlet_parameters_expected_arrays(self) -> None:
        from active_inference.estimators.pomdp import DirichletParameters
        p = DirichletParameters(a=np.array([[3.0, 1.0], [1.0, 3.0]]), d=np.array([2.0, 6.0]))
        np.testing.assert_allclose(p.A.sum(axis=0), [1.0, 1.0])
        np.testing.assert_allclose(p.D, [0.25, 0.75])
        assert p.B is None
        np.testing.assert_allclose(p.copy().a, p.a)

    def test_learning_agent_learns_likelihood_via_novelty(self) -> None:
        # Algorithm 10.1.1: a novelty-driven agent with uniform priors learns A by acting.
        from active_inference.estimators.pomdp import simulate_learning_agent
        A_true = np.array([[0.8, 0.2], [0.2, 0.8]])
        res = simulate_learning_agent(A_true=A_true, n_states=2,
                                      n_trials=40, steps_per_trial=20)
        assert res.final_A_error() < 0.1          # converges toward the truth
        assert res.novelty_first_trial > 0.0      # exploration is driven by positive novelty
        # The learned A is closer to truth than the uniform prior it started from.
        start_err = np.max(np.abs(res.A_history[0] - A_true))
        assert res.final_A_error() < start_err


class TestPrecisionSweep:
    def test_sweep_shapes_and_normalization(self) -> None:
        from active_inference.estimators.pomdp import precision_policy_sweep
        G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
        sweep = precision_policy_sweep(G, [0.0, 1.0, 2.0, 3.0])
        assert sweep.shape == (4, 5)
        np.testing.assert_allclose(sweep.sum(axis=1), np.ones(4))

    def test_sweep_concentrates_on_min_efe_as_gamma_grows(self) -> None:
        # As γ increases the posterior mass on the lowest-EFE policy (index 2) rises.
        from active_inference.estimators.pomdp import precision_policy_sweep
        G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
        sweep = precision_policy_sweep(G, [0.0, 1.0, 3.0])
        assert sweep[0, 2] < sweep[1, 2] < sweep[2, 2]
        assert np.isclose(sweep[0].max(), 0.2)        # γ=0 → uniform


# ---------------------------------------------------------------------------
# §10.3 — two-armed bandit (Example 10.7)
# ---------------------------------------------------------------------------
class TestTwoArmedBandit:
    def test_model_structure(self) -> None:
        from active_inference.estimators.pomdp import make_two_armed_bandit
        m = make_two_armed_bandit()
        assert m.n_factors == 2 and m.n_modalities == 3
        assert m.factor_sizes == [2, 4] and m.obs_sizes == [3, 3, 4]
        # Hint reveals context only when the hint action is chosen.
        np.testing.assert_allclose(m.A[0][:, 0, 1], [0, 1, 0])   # ctx0 → left-hint
        np.testing.assert_allclose(m.A[0][:, 1, 1], [0, 0, 1])   # ctx1 → right-hint
        np.testing.assert_allclose(m.A[0][:, 0, 0], [1, 0, 0])   # start → no hint
        # Reward depends on context × chosen machine.
        assert m.A[1][2, 0, 2] > m.A[1][2, 1, 2]                 # left machine wins more if ctx0
        # C[1] is softmax-normalized (valid distribution).
        assert np.isclose(m.C[1].sum(), 1.0)

    @pytest.mark.parametrize("ctx", [0, 1])
    def test_agent_learns_context_and_exploits(self, ctx: int) -> None:
        from active_inference.estimators.pomdp import make_two_armed_bandit, simulate_two_armed_bandit
        m = make_two_armed_bandit()
        res = simulate_two_armed_bandit(m, true_context=ctx, n_steps=15, gamma=4.0)
        # Context belief converges to the true better machine.
        assert int(np.argmax(res.context_belief[-1])) == ctx
        assert res.context_belief[-1][ctx] > 0.9
        # The agent exploits: a clear majority of pulls win.
        assert res.n_wins >= 9

    def test_hint_has_highest_context_info_gain(self) -> None:
        # The hint action resolves the binary context → ~1 bit (ln 2 nats) of info gain,
        # strictly more than not taking the hint.
        from active_inference.estimators.pomdp import make_two_armed_bandit, _TAB_HINT, _TAB_LEFT
        from active_inference.core.pomdp import (
            factorial_expected_observation, factorial_modality_ambiguity)
        m = make_two_armed_bandit()
        unc = np.array([0.5, 0.5])

        def info_gain(states, mod):
            o = factorial_expected_observation(m.A[mod], states)
            return float(-np.sum(o * np.log(o + 1e-16))) - factorial_modality_ambiguity(m.A[mod], states)

        hint = info_gain([unc, np.eye(4)[_TAB_HINT]], 0)
        nohint = info_gain([unc, np.eye(4)[_TAB_LEFT]], 0)
        assert hint == pytest.approx(np.log(2), abs=1e-6)
        assert hint > nohint + 0.5


# ---------------------------------------------------------------------------
# §10.4 — hierarchical depth (nested time scales)
# ---------------------------------------------------------------------------
class TestHierarchicalAgent:
    def _model(self):
        from active_inference.core.pomdp import HierarchicalPOMDP, POMDPModel
        top = POMDPModel(A=np.array([[0.9, 0.1], [0.1, 0.9]]), D=np.array([0.5, 0.5]))
        bot = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
        link = np.array([[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]])
        link = link / link.sum(0)
        return HierarchicalPOMDP(layers=[bot, top], link=[link])

    @pytest.mark.parametrize("top_state", [0, 1])
    def test_top_layer_contextualizes_bottom(self, top_state: int) -> None:
        from active_inference.estimators.pomdp import simulate_hierarchical_agent
        h = self._model()
        res = simulate_hierarchical_agent(h, true_top=top_state, n_macro=5, inner_steps=3,
                                          rng=np.random.default_rng(0))
        # Top layer infers its regime; that belief steers the bottom layer's prior.
        assert int(np.argmax(res.top_belief[-1])) == top_state
        bottom_focus = int(np.argmax(res.bottom_priors[-1]))
        assert bottom_focus == (0 if top_state == 0 else 2)      # link maps regime → bottom state
        assert res.bottom_belief.shape == (5, 3, 3)

    def test_rejects_non_two_layer(self) -> None:
        from active_inference.core.pomdp import HierarchicalPOMDP, POMDPModel
        from active_inference.estimators.pomdp import simulate_hierarchical_agent
        single = HierarchicalPOMDP(layers=[POMDPModel(A=np.eye(2), D=np.array([0.5, 0.5]))])
        with pytest.raises(ValueError):
            simulate_hierarchical_agent(single)
