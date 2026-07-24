"""Tests for ``estimators.active_inference`` — the action-perception loop.

Oracle discipline: the *defining* active-inference property is verified — with action
the agent drives the true external state to its preferred set-point, and without action
it does not (it stays at the uncontrolled exogenous attractor).
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.predictive_coding import LinearFunction
from active_inference.core.generalized_filtering import DynamicStateSpaceModel
from active_inference.core.active_inference import ActiveInferenceAgent
from active_inference.estimators.active_inference import (
    ActiveEnvironment,
    simulate_active_inference,
)

V_STAR, V_PREF, THETA_Y = 10.0, 0.0, 3.0


def build(kappa_a=0.4):
    # Environment: drift = v* − x* (exogenous attractor at v*=10), obs g_E=x−θ_y.
    env = ActiveEnvironment(drift=LinearFunction(-1.0, V_STAR), g=LinearFunction(1.0, -THETA_Y),
                            omega_x=0.05, omega_y=0.05)
    # Agent: preference v=0 (attractor f_M=v−μ), matched obs g_M=μ−θ_y.
    model = DynamicStateSpaceModel(f=LinearFunction(-1.0, V_PREF), g=LinearFunction(1.0, -THETA_Y),
                                   s2_x=1.0, sigma2_y=0.05)
    agent = ActiveInferenceAgent(perception_model=model, forward_model=1.0,
                                 kappa_x=0.2, kappa_a=kappa_a)
    return agent, env


class TestActiveInferenceLoop:
    def test_action_drives_state_to_preference(self) -> None:
        # ISC (the headline AIF property): with action, x* → preference v=0.
        agent, env = build()
        res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=6000,
                                        dt=0.01, action_start=2000,
                                        rng=np.random.default_rng(0))
        assert res.settled_state() == pytest.approx(V_PREF, abs=0.3)
        # action settles near −v* (it must cancel the exogenous attractor force).
        assert res.settled_action() == pytest.approx(-V_STAR, abs=1.0)

    def test_without_action_stays_at_exogenous_attractor(self) -> None:
        # Perception-only (κ_a=0): x* converges to the uncontrolled attractor v*=10, NOT v.
        agent, env = build(kappa_a=0.0)
        res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=6000,
                                        dt=0.01, action_start=2000,
                                        rng=np.random.default_rng(0))
        assert res.settled_state() == pytest.approx(V_STAR, abs=0.3)

    def test_action_reduces_sensory_error_and_free_energy(self) -> None:
        agent, env = build()
        res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=6000,
                                        dt=0.01, action_start=2000,
                                        rng=np.random.default_rng(1))
        # mean |ε_y| after action is far below its pre-action level.
        pre = float(np.mean(np.abs(res.eps_y[1500:2000])))
        post = float(np.mean(np.abs(res.eps_y[-500:])))
        assert post < 0.5 * pre

    def test_result_shapes(self) -> None:
        agent, env = build()
        res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=500, dt=0.01)
        for arr in (res.xs, res.mus, res.actions, res.ys, res.free_energies, res.eps_y):
            assert arr.shape == (501,)


class TestMultivariateActiveInferenceLoop:
    def _build(self, *, kappa_a: float = 0.5):
        from active_inference.core.active_inference import MultivariateActiveInferenceAgent
        from active_inference.core.generalized_filtering import (
            GeneralizedVectorModel,
            LinearVectorFunction,
            correlated_embedding_precision,
        )
        from active_inference.estimators.active_inference import MultivariateActiveEnvironment

        M = 2
        preference = np.zeros(2)
        exogenous = np.array([6.0, -4.0])
        model = GeneralizedVectorModel(
            f=LinearVectorFunction(-np.eye(2), preference),
            g=LinearVectorFunction(np.eye(2), np.zeros(2)),
            precision_x=correlated_embedding_precision(np.eye(2), M, gamma=2.0),
            precision_y=correlated_embedding_precision(np.eye(2) * 20.0, M, gamma=2.0),
            embedding_dim=M,
            dim_x=2,
            dim_y=2,
        )
        forward = np.zeros((M * 2, 2))
        forward[[0, 2], :] = np.eye(2)
        agent = MultivariateActiveInferenceAgent(
            perception_model=model,
            forward_model=forward,
            kappa_x=0.4,
            kappa_a=kappa_a,
        )
        env = MultivariateActiveEnvironment(
            drift=LinearVectorFunction(-np.eye(2), exogenous),
            g=LinearVectorFunction(np.eye(2), np.zeros(2)),
            action_matrix=np.eye(2),
            omega_x=0.0,
            omega_y=0.0,
        )
        return agent, env, preference, exogenous

    def test_action_drives_vector_state_closer_to_preference_than_baseline(self) -> None:
        from active_inference.estimators.active_inference import simulate_multivariate_active_inference

        agent, env, preference, exogenous = self._build()
        active = simulate_multivariate_active_inference(
            agent,
            env,
            x0=np.array([4.0, -2.0]),
            mu0_tilde=np.zeros((2, 2)),
            n_steps=2500,
            dt=0.01,
            action_start=500,
            rng=np.random.default_rng(0),
        )
        passive_agent, _, _, _ = self._build(kappa_a=0.0)
        passive = simulate_multivariate_active_inference(
            passive_agent,
            env,
            x0=np.array([4.0, -2.0]),
            mu0_tilde=np.zeros((2, 2)),
            n_steps=2500,
            dt=0.01,
            action_start=500,
            rng=np.random.default_rng(0),
        )
        assert active.preference_error(preference) < 0.35
        assert active.preference_error(preference) < 0.25 * passive.preference_error(preference)
        np.testing.assert_allclose(passive.settled_state(), exogenous, atol=0.15)

    def test_multivariate_result_shapes(self) -> None:
        from active_inference.estimators.active_inference import simulate_multivariate_active_inference

        agent, env, _, _ = self._build()
        result = simulate_multivariate_active_inference(
            agent,
            env,
            x0=np.array([4.0, -2.0]),
            mu0_tilde=np.zeros((2, 2)),
            n_steps=100,
            dt=0.01,
            rng=np.random.default_rng(1),
        )
        assert result.xs.shape == (101, 2)
        assert result.mus.shape == (101, 2, 2)
        assert result.actions.shape == (101, 2)
        assert result.y_tildes.shape == (101, 2, 2)
        assert result.eps_y.shape == (101, 2, 2)
