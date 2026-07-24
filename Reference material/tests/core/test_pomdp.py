"""Tests for ``core.pomdp`` — Chapter 9 §9.1 discrete POMDP state inference.

Oracle discipline: the posterior is checked against the book's worked Eq. 15 result
(`[0.18, 0.40, 0.36, 0.06]` for observing "hot" under the weather model) and against an
independent direct (non-log-space) Bayes computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.pomdp import (
    POMDPModel,
    infer_states,
    is_stochastic_matrix,
    one_hot,
    softmax,
)

# Weather model (book §9.1): rows = [water, hot, bright], columns = [rainy, cloudy, sunny, snowy].
A_WEATHER = np.array([[0.80, 0.33, 0.05, 0.40],
                      [0.15, 0.33, 0.30, 0.05],
                      [0.05, 0.34, 0.65, 0.55]])
D_UNIFORM = np.array([0.25, 0.25, 0.25, 0.25])


class TestPrimitives:
    def test_softmax_normalizes(self) -> None:
        s = softmax(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(s.sum(), 1.0)
        # softmax is shift-invariant.
        np.testing.assert_allclose(s, softmax(np.array([1.0, 2.0, 3.0]) + 100.0))

    def test_softmax_normalizes_selected_axis(self) -> None:
        # Regression: logsumexp squeezes the reduced axis, so softmax must reinsert it
        # before broadcasting. Otherwise axis=-1 normalized the wrong dimension.
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        rowwise = softmax(x, axis=-1)
        colwise = softmax(x, axis=0)
        np.testing.assert_allclose(rowwise.sum(axis=-1), [1.0, 1.0])
        np.testing.assert_allclose(colwise.sum(axis=0), [1.0, 1.0])
        assert rowwise[0, 1] > rowwise[0, 0]
        assert colwise[1, 0] > colwise[0, 0]

    def test_one_hot(self) -> None:
        np.testing.assert_allclose(one_hot(1, 3), [0.0, 1.0, 0.0])

    def test_one_hot_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            one_hot(-1, 3)
        with pytest.raises(ValueError):
            one_hot(3, 3)

    def test_is_stochastic_matrix(self) -> None:
        assert is_stochastic_matrix(A_WEATHER)             # columns sum to 1
        assert not is_stochastic_matrix(np.array([[0.5, 0.6], [0.6, 0.4]]))
        assert not is_stochastic_matrix(np.array([1.0, 0.0]))


class TestModelValidation:
    def test_accepts_valid(self) -> None:
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        assert m.n_states == 4 and m.n_obs == 3

    def test_rejects_non_stochastic_A(self) -> None:
        with pytest.raises(ValueError):
            POMDPModel(A=np.array([[0.5, 0.5], [0.6, 0.4]]), D=np.array([0.5, 0.5]))

    def test_rejects_unnormalized_D(self) -> None:
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=np.array([0.5, 0.5, 0.5, 0.5]))

    def test_validates_B_transition_columns(self) -> None:
        bad_B = np.array([[0.5, 0.5, 0.5, 0.5]] * 4)  # columns don't sum to 1
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=D_UNIFORM, B=bad_B)

    def test_rejects_B_with_wrong_state_shape(self) -> None:
        bad_B = np.ones((1, 3, 3)) / 3.0
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=D_UNIFORM, B=bad_B)

    def test_rejects_negative_D_and_preferences(self) -> None:
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=np.array([0.5, 0.5, 0.5, -0.5]))
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=D_UNIFORM, C=np.array([1.0, -1.0, 0.0]))

    def test_rejects_bad_C_shape(self) -> None:
        with pytest.raises(ValueError):
            POMDPModel(A=A_WEATHER, D=D_UNIFORM, C=np.array([1.0, 0.0]))


class TestStateInference:
    def test_matches_book_eq15_hot(self) -> None:
        # Book Eq. 15: observing "hot" (index 1) under uniform prior.
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        np.testing.assert_allclose(infer_states(m, 1), [0.18, 0.40, 0.36, 0.06], atol=0.01)

    def test_water_observation(self) -> None:
        # Fig. 9.1.3 right panel: observing "water" (index 0).
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        post = infer_states(m, 0)
        assert post.sum() == pytest.approx(1.0)
        assert int(np.argmax(post)) == 0  # rainy is most likely after seeing water

    def test_softmax_equals_direct_bayes(self) -> None:
        # The log-space softmax (Eq. 13) equals direct likelihood×prior normalization (Eq. 12).
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        o = one_hot(1, 3)
        lik = A_WEATHER.T @ o
        direct = (lik * D_UNIFORM) / np.sum(lik * D_UNIFORM)
        np.testing.assert_allclose(infer_states(m, 1), direct, atol=1e-12)

    def test_one_hot_vector_and_index_agree(self) -> None:
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        np.testing.assert_allclose(infer_states(m, 1), infer_states(m, one_hot(1, 3)))

    def test_rejects_bad_observation_vector(self) -> None:
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        with pytest.raises(ValueError):
            infer_states(m, np.array([1.0, 0.0]))
        with pytest.raises(ValueError):
            infer_states(m, np.array([1.0, 1.0, 0.0]))

    def test_prior_override(self) -> None:
        # A non-uniform prior shifts the posterior toward the favored state.
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        biased = infer_states(m, 1, prior=np.array([0.0, 1.0, 0.0, 0.0]))
        assert biased[1] == pytest.approx(1.0, abs=1e-9)  # prior forces cloudy

    def test_rejects_bad_prior_override(self) -> None:
        m = POMDPModel(A=A_WEATHER, D=D_UNIFORM)
        with pytest.raises(ValueError):
            infer_states(m, 1, prior=np.ones(4))

    def test_informative_observation_beats_uniform(self) -> None:
        # A perfectly informative likelihood collapses the posterior onto one state.
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = POMDPModel(A=A, D=np.array([0.5, 0.5]))
        np.testing.assert_allclose(infer_states(m, 0), [1.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# §9.5 — expected free energy, policy and action selection
# ---------------------------------------------------------------------------


class TestExpectedFreeEnergy:
    def test_risk_matches_book_eq63(self) -> None:
        # Example 9.7: A=[[.85,.15],[.15,.85]], C=[1,0] (raw), states give o=[.815,.185]/[.5,.5].
        from active_inference.core.pomdp import efe_risk
        m = POMDPModel(A=np.array([[0.85, 0.15], [0.15, 0.85]]), D=np.array([0.5, 0.5]))
        C = np.array([1.0, 0.0])
        assert efe_risk(m, np.array([0.95, 0.05]), C) == pytest.approx(6.337, abs=0.01)
        assert efe_risk(m, np.array([0.5, 0.5]), C) == pytest.approx(17.728, abs=0.01)

    def test_risk_binds_A_orientation(self) -> None:
        # RedTeam (oracle-incompleteness): the Eq-63 oracle uses a SYMMETRIC A, so an
        # A-vs-Aᵀ transpose bug would survive it. This asymmetric case pins o = A·s
        # (correct = 10.8017; the transposed Aᵀ·s would give 13.3578).
        from active_inference.core.pomdp import efe_risk
        m = POMDPModel(A=np.array([[0.9, 0.2], [0.1, 0.8]]), D=np.array([0.5, 0.5]))
        assert efe_risk(m, np.array([0.7, 0.3]), np.array([1.0, 0.0])) == pytest.approx(10.8017, abs=1e-3)

    def test_ambiguity_matches_book_eq68(self) -> None:
        # Example 9.7 ambiguity: A=[[.6,.15],[.4,.85]], s_pi0=[.9,.1], s_pi1=[.1,.9].
        from active_inference.core.pomdp import efe_ambiguity
        m = POMDPModel(A=np.array([[0.6, 0.15], [0.4, 0.85]]), D=np.array([0.5, 0.5]))
        assert efe_ambiguity(m, np.array([0.9, 0.1])) == pytest.approx(0.648, abs=0.01)
        assert efe_ambiguity(m, np.array([0.1, 0.9])) == pytest.approx(0.448, abs=0.01)

    def test_ambiguity_zero_for_deterministic_likelihood(self) -> None:
        # An identity (deterministic) A has zero observation entropy → zero ambiguity.
        from active_inference.core.pomdp import efe_ambiguity
        m = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
        assert efe_ambiguity(m, np.array([0.3, 0.4, 0.3])) == pytest.approx(0.0, abs=1e-10)

    def test_efe_is_risk_plus_ambiguity(self) -> None:
        from active_inference.core.pomdp import (
            efe_components, efe_risk, efe_ambiguity, expected_free_energy)
        m = POMDPModel(A=np.array([[0.6, 0.15], [0.4, 0.85]]), D=np.array([0.5, 0.5]))
        s, C = np.array([0.7, 0.3]), np.array([1.0, 0.0])
        comp = efe_components(m, s, C)
        assert expected_free_energy(m, s, C) == pytest.approx(efe_risk(m, s, C) + efe_ambiguity(m, s))
        assert comp.total == pytest.approx(expected_free_energy(m, s, C))
        np.testing.assert_allclose(comp.expected_observation, m.A @ s)

    def test_policy_component_trace_sums_to_scalar_policy_efe(self) -> None:
        from active_inference.core.pomdp import (
            evaluate_policy, evaluate_policy_components, expected_free_energy)
        A = np.eye(2)
        D = np.array([0.5, 0.5])
        B = np.array([
            [[0.9, 0.9], [0.1, 0.1]],   # action 0 predicts state 0
            [[0.1, 0.1], [0.9, 0.9]],   # action 1 predicts state 1
        ])
        m = POMDPModel(A=A, D=D, B=B)
        C = np.array([0.9, 0.1])
        policy = [0, 1]
        tr = evaluate_policy_components(m, D, policy, C)
        assert tr.policy.tolist() == policy
        assert tr.states.shape == (2, 2)
        assert tr.expected_observations.shape == (2, 2)
        assert tr.total == pytest.approx(evaluate_policy(m, D, policy, C))
        assert tr.totals_per_step[0] == pytest.approx(expected_free_energy(m, tr.states[0], C))
        assert tr.risk_total + tr.ambiguity_total - tr.novelty_total == pytest.approx(tr.total)

    def test_policy_component_trace_supports_chapter10_novelty(self) -> None:
        from active_inference.core.pomdp import (
            efe_with_novelty, evaluate_policy_components, expected_A)
        a = np.array([[3.0, 1.0], [1.0, 3.0]])
        A = expected_A(a)
        B = np.array([[[0.7, 0.7], [0.3, 0.3]]])
        m = POMDPModel(A=A, D=np.array([0.5, 0.5]), B=B)
        C = np.array([1.0, 0.0])
        tr = evaluate_policy_components(m, np.array([0.5, 0.5]), [0], C, a=a)
        assert tr.novelty_total > 0.0
        assert tr.total == pytest.approx(efe_with_novelty(m, tr.states[0], C, a))

    def test_policy_posterior_favors_low_efe(self) -> None:
        from active_inference.core.pomdp import policy_posterior
        G = np.array([1.0, 5.0, 2.0])
        q = policy_posterior(G, gamma=1.0)
        assert q.sum() == pytest.approx(1.0)
        assert int(np.argmax(q)) == 0           # lowest EFE → highest probability
        np.testing.assert_allclose(q, np.exp(-G) / np.exp(-G).sum())

    def test_action_posterior_marginalizes_policies(self) -> None:
        from active_inference.core.pomdp import action_posterior
        policies = [[0, 1], [0, 3], [2, 1]]      # two start with action 0, one with 2
        q_pi = np.array([0.5, 0.3, 0.2])
        q_u = action_posterior(q_pi, policies, n_controls=4, tau=0)
        np.testing.assert_allclose(q_u, [0.8, 0.0, 0.2, 0.0])   # 0.5+0.3 on action 0


# ---------------------------------------------------------------------------
# §10.1 — Learning the POMDP parameters (Dirichlet) — book oracles
# ---------------------------------------------------------------------------
class TestDirichletLearning:
    """Verified against the worked examples of book §10.1 (Examples 10.1, 10.4)."""

    def test_expected_value_column_normalizes(self) -> None:
        from active_inference.core.pomdp import dirichlet_expected_value, expected_A
        a = np.array([[2.0, 1.0], [2.0, 3.0]])
        A = expected_A(a)
        np.testing.assert_allclose(A.sum(axis=0), [1.0, 1.0])     # columns are categoricals
        np.testing.assert_allclose(A, [[0.5, 0.25], [0.5, 0.75]])
        # generic helper agrees on axis=0
        np.testing.assert_allclose(dirichlet_expected_value(a, axis=0), A)

    def test_expected_D_normalizes_vector(self) -> None:
        from active_inference.core.pomdp import expected_D
        np.testing.assert_allclose(expected_D(np.array([45.1, 5.9])), [0.884, 0.116], atol=1e-3)

    def test_expected_B_normalizes_each_slice(self) -> None:
        from active_inference.core.pomdp import expected_B
        b = np.array([[[2.0, 1.0], [2.0, 3.0]], [[1.0, 1.0], [1.0, 1.0]]])  # (U=2, C, C)
        B = expected_B(b)
        for u in range(2):
            np.testing.assert_allclose(B[u].sum(axis=0), [1.0, 1.0])

    def test_a_update_is_outer_product(self) -> None:
        # Eq. 4a term: o ∘ s. State 0, observation 1 (one-hots) → single entry [1,0].
        from active_inference.core.pomdp import accumulate_a_counts
        m = accumulate_a_counts(one_hot(1, 2), one_hot(0, 2))
        np.testing.assert_allclose(m, [[0.0, 0.0], [1.0, 0.0]])

    def test_d_update_matches_example_10_1(self) -> None:
        # Eq. 7: d=[1,1] + s=[0.9,0.1] = [1.9,1.1].
        from active_inference.core.pomdp import accumulate_d_counts
        d = np.array([1.0, 1.0]) + accumulate_d_counts(np.array([0.9, 0.1]))
        np.testing.assert_allclose(d, [1.9, 1.1])

    def test_novelty_matrix_matches_example_10_4(self) -> None:
        # Eq. 12/18: W ≈ ½(1/a − 1/a₀), a₀ = column-sum broadcast.
        from active_inference.core.pomdp import novelty_matrix
        a = np.array([[2.5, 0.15], [0.8, 3.0]])
        W = novelty_matrix(a)
        np.testing.assert_allclose(W, [[0.048, 3.175], [0.473, 0.008]], atol=1e-3)

    def test_parameter_novelty_matches_example_10_4(self) -> None:
        # Eq. 17/19: A=E[Dir(a)], o=As, novelty = o·(Ws) = 0.483.
        from active_inference.core.pomdp import parameter_novelty, expected_A
        a = np.array([[2.5, 0.15], [0.8, 3.0]])
        s = np.array([0.15, 0.85])
        np.testing.assert_allclose(expected_A(a), [[0.758, 0.048], [0.242, 0.952]], atol=1e-3)
        assert parameter_novelty(a, s) == pytest.approx(0.483, abs=2e-3)

    def test_novelty_decreases_with_confidence(self) -> None:
        # As pseudocounts grow (more counts), information gain from one more shrinks.
        from active_inference.core.pomdp import parameter_novelty
        s = np.array([0.5, 0.5])
        a_low = np.array([[1.0, 1.0], [1.0, 1.0]])
        a_high = a_low * 100.0
        assert parameter_novelty(a_high, s) < parameter_novelty(a_low, s)

    def test_efe_with_novelty_subtracts_gain(self) -> None:
        # Eq. 15: G = risk + ambiguity − novelty (novelty makes a policy more attractive).
        from active_inference.core.pomdp import (
            efe_with_novelty, expected_free_energy, parameter_novelty, expected_A,
        )
        a = np.array([[3.0, 1.0], [1.0, 3.0]])
        m = POMDPModel(A=expected_A(a), D=np.array([0.5, 0.5]))
        s, C = np.array([0.6, 0.4]), np.array([1.0, 0.0])
        g = efe_with_novelty(m, s, C, a)
        assert g == pytest.approx(expected_free_energy(m, s, C) - parameter_novelty(a, s))
        assert g < expected_free_energy(m, s, C)   # novelty lowers EFE


# ---------------------------------------------------------------------------
# §10.2 — habit (baseline prior E) + policy precision γ
# ---------------------------------------------------------------------------
class TestHabitAndPrecision:
    """Verified against book Example 10.5 (Fig 10.2.3) and 10.6 (precision ordering)."""

    G5 = np.array([3.0, 2.0, 1.5, 2.2, 3.2])   # Example 10.5/10.6 EFE vector (Eq 27)

    def test_uniform_E_zero_gamma_is_uniform(self) -> None:
        from active_inference.core.pomdp import policy_posterior_full
        q = policy_posterior_full(self.G5, gamma=0.0)
        np.testing.assert_allclose(q, np.full(5, 0.2))         # Fig 10.2.3 top-left

    def test_uniform_E_matches_example_10_5_gamma_1_5(self) -> None:
        from active_inference.core.pomdp import policy_posterior_full
        q = policy_posterior_full(self.G5, gamma=1.5)
        # Fig 10.2.3 top-middle: mass concentrates on policy 2 (lowest EFE = 1.5).
        np.testing.assert_allclose(q, [0.053, 0.236, 0.499, 0.174, 0.039], atol=1e-3)
        assert int(np.argmax(q)) == 2

    def test_gamma_one_reduces_to_chapter9_posterior(self) -> None:
        from active_inference.core.pomdp import policy_posterior_full, policy_posterior
        # With uniform E and F=0, Eq 22 reduces to the Chapter 9 σ(−γG).
        np.testing.assert_allclose(policy_posterior_full(self.G5, gamma=1.0),
                                   policy_posterior(self.G5, gamma=1.0))

    def test_strong_habit_matches_example_10_5(self) -> None:
        from active_inference.core.pomdp import policy_posterior_full
        E = np.array([0.2, 0.4, 0.1, 0.4, 0.2])
        q = policy_posterior_full(self.G5, E=E, gamma=0.0)        # γ=0 ⇒ pure habit
        np.testing.assert_allclose(q, [0.154, 0.308, 0.077, 0.308, 0.154], atol=1e-3)

    def test_vfe_term_shifts_posterior(self) -> None:
        from active_inference.core.pomdp import policy_posterior_full
        # Adding F penalizes the policies with high VFE evidence.
        F = np.array([0.0, 0.0, 5.0, 0.0, 0.0])   # policy 2 looks bad after seeing data
        q = policy_posterior_full(self.G5, F=F, gamma=1.0)
        q_noF = policy_posterior_full(self.G5, gamma=1.0)
        assert q[2] < q_noF[2]                       # F pushed mass off policy 2

    def test_learn_precision_self_consistent_fixed_point(self) -> None:
        from active_inference.core.pomdp import learn_precision, precision_gradient
        res = learn_precision(self.G5, np.array([3.5, 2.3, 2.0, 2.5, 4.0]),
                              E=np.array([0.2] * 5))
        assert res.converged
        assert res.grad_final < 1e-6               # ∂F/∂γ ≈ 0 at the fixed point
        # the returned γ genuinely zeroes the gradient (independent recompute)
        assert abs(precision_gradient(self.G5, np.array([3.5, 2.3, 2.0, 2.5, 4.0]),
                                      res.gamma, E=np.array([0.2] * 5))) < 1e-6
        assert res.gamma > 0                        # γ = 1/β with β > 0

    def test_precision_higher_when_F_close_to_G(self) -> None:
        # Book Example 10.6: F close to G ⇒ high confidence (high γ); F far ⇒ low γ.
        from active_inference.core.pomdp import learn_precision
        E = np.array([0.2] * 5)
        g_close = learn_precision(self.G5, np.array([3.5, 2.3, 2.0, 2.5, 4.0]), E=E).gamma
        g_far = learn_precision(self.G5, np.array([0.05, 3.2, 5.0, 6.0, 0.003]), E=E).gamma
        assert g_close > g_far                      # matches book's 1.357 > 0.493 ordering


# ---------------------------------------------------------------------------
# §10.3 — factorial depth (state factors + observation modalities)
# ---------------------------------------------------------------------------
class TestFactorialDepth:
    def test_expected_observation_reduces_to_single_factor(self) -> None:
        from active_inference.core.pomdp import factorial_expected_observation
        A = np.array([[0.6, 0.15], [0.4, 0.85]])
        s = np.array([0.7, 0.3])
        np.testing.assert_allclose(factorial_expected_observation(A, [s]), A @ s)

    def test_infer_states_reduces_to_chapter9(self) -> None:
        # Single factor + single modality factorial inference == exact Eq.15 weather posterior.
        from active_inference.core.pomdp import FactorialPOMDP, factorial_infer_states
        A = np.array([[0.80, 0.33, 0.05, 0.40],
                      [0.15, 0.33, 0.30, 0.05],
                      [0.05, 0.34, 0.65, 0.55]])
        D = np.full(4, 0.25)
        post = factorial_infer_states(FactorialPOMDP(A=[A], D=[D]), [1])[0]
        np.testing.assert_allclose(post, [0.1807, 0.3976, 0.3614, 0.0602], atol=1e-3)

    def test_mean_field_independence(self) -> None:
        # A modality observing only factor 0 collapses factor 0; factor 1 stays at its prior.
        from active_inference.core.pomdp import FactorialPOMDP, factorial_infer_states
        A0 = np.zeros((2, 2, 3))
        for c0 in range(2):
            for c1 in range(3):
                A0[c0, c0, c1] = 1.0
        m = FactorialPOMDP(A=[A0], D=[np.array([0.5, 0.5]), np.full(3, 1 / 3)])
        post = factorial_infer_states(m, [0])
        np.testing.assert_allclose(post[0], [1.0, 0.0], atol=1e-6)   # factor 0 resolved
        np.testing.assert_allclose(post[1], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)  # factor 1 uninformed

    def test_efe_reduces_to_chapter9(self) -> None:
        from active_inference.core.pomdp import (
            FactorialPOMDP, factorial_efe, expected_free_energy, POMDPModel)
        A = np.array([[0.6, 0.15], [0.4, 0.85]])
        D, C, s = np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.7, 0.3])
        fac = factorial_efe(FactorialPOMDP(A=[A], D=[D], C=[C]), [s])
        flat = expected_free_energy(POMDPModel(A=A, D=D), s, C)
        assert fac == pytest.approx(flat)

    def test_efe_decomposes_to_pragmatic_minus_info_gain(self) -> None:
        # G = risk + ambiguity = (−o·log C) − I(o;s); a fully informative modality lowers G.
        from active_inference.core.pomdp import (
            FactorialPOMDP, factorial_efe, factorial_expected_observation,
            factorial_modality_ambiguity)
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        s, C = np.array([0.5, 0.5]), np.array([0.5, 0.5])
        m = FactorialPOMDP(A=[A], D=[np.array([0.5, 0.5])], C=[C])
        o = factorial_expected_observation(A, [s])
        risk = float(o @ (np.log(o + 1e-16) - np.log(C + 1e-16)))
        amb = factorial_modality_ambiguity(A, [s])
        assert factorial_efe(m, [s]) == pytest.approx(risk + amb)

    def test_predict_states_per_factor(self) -> None:
        from active_inference.core.pomdp import FactorialPOMDP, factorial_predict_states
        A = np.zeros((2, 2, 2))
        A[:, :, :] = 0.5
        B_ctx = np.eye(2)[:, :, None]                       # uncontrollable
        B_ch = np.zeros((2, 2, 2))
        for u in range(2):
            B_ch[u, :, u] = 1.0                             # action u → state u
        m = FactorialPOMDP(A=[A], D=[np.array([0.5, 0.5]), np.array([1.0, 0.0])], B=[B_ctx, B_ch])
        pred = factorial_predict_states(m, [np.array([0.3, 0.7]), np.array([1.0, 0.0])], [0, 1])
        np.testing.assert_allclose(pred[0], [0.3, 0.7])      # context unchanged (identity)
        np.testing.assert_allclose(pred[1], [0.0, 1.0])      # choice → action 1

    def test_validation_rejects_unnormalized_A(self) -> None:
        from active_inference.core.pomdp import FactorialPOMDP
        with pytest.raises(ValueError):
            FactorialPOMDP(A=[np.ones((2, 2))], D=[np.array([0.5, 0.5])])  # cols sum to 2


# ---------------------------------------------------------------------------
# §10.4 — hierarchical depth (nested POMDP layers)
# ---------------------------------------------------------------------------
class TestHierarchicalDepth:
    def test_layer_vfe_reduces_to_flat(self) -> None:
        from active_inference.core.pomdp import (
            hierarchical_layer_vfe, discrete_vfe, POMDPModel)
        A = np.array([[0.8, 0.2], [0.2, 0.8]])
        D, s = np.array([0.5, 0.5]), np.array([0.7, 0.3])
        assert hierarchical_layer_vfe(s, 0, A, prior=D) == pytest.approx(
            discrete_vfe(POMDPModel(A=A, D=D), s, 0, D))

    def test_layer_efe_reduces_to_flat(self) -> None:
        from active_inference.core.pomdp import (
            hierarchical_layer_efe, expected_free_energy, POMDPModel)
        A = np.array([[0.8, 0.2], [0.2, 0.8]])
        s, C = np.array([0.7, 0.3]), np.array([1.0, 0.0])
        assert hierarchical_layer_efe(A, s, C) == pytest.approx(
            expected_free_energy(POMDPModel(A=A, D=A @ s * 0 + 0.5), s, C))

    def test_top_down_prior_is_link_column(self) -> None:
        from active_inference.core.pomdp import HierarchicalPOMDP, POMDPModel
        top = POMDPModel(A=np.eye(2), D=np.array([0.5, 0.5]))
        bot = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
        link = np.array([[0.9, 0.1], [0.1, 0.8], [0.0, 0.1]])
        h = HierarchicalPOMDP(layers=[bot, top], link=[link])
        np.testing.assert_allclose(h.layer_prior(0, np.array([1.0, 0.0])), link[:, 0])
        np.testing.assert_allclose(h.layer_prior(0, None), bot.D)   # no higher belief → own D

    def test_link_validation(self) -> None:
        from active_inference.core.pomdp import HierarchicalPOMDP, POMDPModel
        top = POMDPModel(A=np.eye(2), D=np.array([0.5, 0.5]))
        bot = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
        with pytest.raises(ValueError):                            # not column-stochastic
            HierarchicalPOMDP(layers=[bot, top], link=[np.ones((3, 2))])

    def test_policy_posterior_matches_full(self) -> None:
        from active_inference.core.pomdp import hierarchical_policy_posterior, policy_posterior_full
        G = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(hierarchical_policy_posterior(G), policy_posterior_full(G))
