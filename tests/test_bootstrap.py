# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.bootstrap_filter.

Cross-validates against:
1. Dynamax Kalman filter (exact solution for linear Gaussian SSMs)
2. Chopin's ``particles`` library (established NumPy reference)
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from smcjax.bootstrap import bootstrap_filter

# ---------------------------------------------------------------------------
# Helpers to define the LGSSM for smcjax
# ---------------------------------------------------------------------------


def _make_smcjax_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn) closures."""
    m0 = lgssm_params['initial_mean']
    P0 = lgssm_params['initial_cov']
    F = lgssm_params['dynamics_weights']
    Q = lgssm_params['dynamics_cov']
    H = lgssm_params['emissions_weights']
    R = lgssm_params['emissions_cov']

    def initial_sampler(key, n):
        return tfd.MultivariateNormalFullCovariance(m0, P0).sample(
            n, seed=key
        )

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return tfd.MultivariateNormalFullCovariance(mean, Q).sample(seed=key)

    def log_observation_fn(emission, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return tfd.MultivariateNormalFullCovariance(mean, R).log_prob(
            emission
        )

    return initial_sampler, transition_sampler, log_observation_fn


# ---------------------------------------------------------------------------
# Test: bootstrap filter vs. Kalman filter (exact)
# ---------------------------------------------------------------------------


class TestBootstrapVsKalman:
    """Bootstrap PF on a linear Gaussian SSM should approximate the Kalman filter."""

    def test_log_marginal_likelihood(self, lgssm_params, lgssm_data):
        """PF log-ML should be close to the Kalman exact log-ML."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)

        # Exact Kalman
        kalman_post = lgssm_filter(params, emissions)
        exact_ll = float(kalman_post.marginal_loglik)

        # Bootstrap PF with many particles
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf_post = bootstrap_filter(
            key=jr.PRNGKey(123),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=10_000,
        )
        pf_ll = float(pf_post.marginal_loglik)

        assert pf_ll == pytest.approx(exact_ll, rel=0.05), (
            f'PF log-ML {pf_ll:.2f} vs Kalman {exact_ll:.2f}'
        )

    def test_filtered_means(self, lgssm_params, lgssm_data):
        """PF weighted means should track the Kalman filtered means."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)

        kalman_post = lgssm_filter(params, emissions)
        kalman_means = kalman_post.filtered_means  # (T, 1)

        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf_post = bootstrap_filter(
            key=jr.PRNGKey(456),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=10_000,
        )

        # Compute weighted mean of particles at each time step
        from smcjax.weights import normalize

        weights = jnp.array(
            [normalize(pf_post.filtered_log_weights[t]) for t in range(50)]
        )  # (T, N)
        pf_means = jnp.sum(
            weights[:, :, None] * pf_post.filtered_particles, axis=1
        )  # (T, 1)

        assert jnp.allclose(pf_means, kalman_means, atol=0.15), (
            f'Max error: {jnp.max(jnp.abs(pf_means - kalman_means)):.4f}'
        )


# ---------------------------------------------------------------------------
# Test: bootstrap filter vs. particles library (Chopin)
# ---------------------------------------------------------------------------


class TestBootstrapVsParticles:
    """Cross-validate against Chopin's particles library."""

    def test_log_marginal_likelihood(self, lgssm_params, lgssm_data):
        """Log-ML estimates from smcjax and particles should agree."""
        import particles
        import particles.distributions as dists
        from particles.state_space_models import Bootstrap, StateSpaceModel

        _, emissions = lgssm_data
        emissions_np = np.array(emissions).squeeze(-1)  # (T,)

        # Define model for particles library
        rho = float(lgssm_params['dynamics_weights'][0, 0])
        sigma_x = float(jnp.sqrt(lgssm_params['dynamics_cov'][0, 0]))
        sigma_y = float(jnp.sqrt(lgssm_params['emissions_cov'][0, 0]))
        sigma_0 = float(jnp.sqrt(lgssm_params['initial_cov'][0, 0]))

        class LinearGauss1D(StateSpaceModel):
            default_params = {
                'rho': rho,
                'sigmaX': sigma_x,
                'sigmaY': sigma_y,
                'sigma0': sigma_0,
            }

            def PX0(self):
                return dists.Normal(scale=self.sigma0)

            def PX(self, t, xp):
                return dists.Normal(loc=self.rho * xp, scale=self.sigmaX)

            def PY(self, t, xp, x):
                return dists.Normal(loc=x, scale=self.sigmaY)

        ssm = LinearGauss1D()
        fk = Bootstrap(ssm=ssm, data=emissions_np)
        n_particles = 5_000
        pf_np = particles.SMC(fk=fk, N=n_particles, resampling='systematic')
        pf_np.run()
        particles_ll = pf_np.logLt

        # smcjax
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf_jax = bootstrap_filter(
            key=jr.PRNGKey(789),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=n_particles,
        )
        smcjax_ll = float(pf_jax.marginal_loglik)

        # Both are Monte Carlo estimates, so we allow generous tolerance.
        # With N=5000 and T=50, std of log-ML ≈ O(1), so atol=3 is ~3 sigma.
        assert smcjax_ll == pytest.approx(particles_ll, abs=3.0), (
            f'smcjax {smcjax_ll:.2f} vs particles {particles_ll:.2f}'
        )


# ---------------------------------------------------------------------------
# Test: convergence with increasing particles
# ---------------------------------------------------------------------------


class TestBootstrapConvergence:
    """PF estimates should improve with more particles."""

    def test_log_ml_converges(self, lgssm_params, lgssm_data):
        """Log-ML variance decreases with more particles."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)
        exact_ll = float(lgssm_filter(params, emissions).marginal_loglik)

        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        errors = []
        for n in [100, 1_000, 10_000]:
            pf = bootstrap_filter(
                key=jr.PRNGKey(999),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=n,
            )
            errors.append(abs(float(pf.marginal_loglik) - exact_ll))

        # Error should generally decrease with more particles
        assert errors[-1] < errors[0], (
            f'Error did not decrease: {errors}'
        )


class TestBootstrapESSTrace:
    """ESS trace should be reasonable."""

    def test_ess_bounded(self, lgssm_params, lgssm_data):
        """ESS should be between 1 and N at every time step."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        n = 1_000
        pf = bootstrap_filter(
            key=jr.PRNGKey(111),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=n,
        )
        assert jnp.all(pf.ess >= 0.9)  # ESS >= ~1
        assert jnp.all(pf.ess <= n + 0.1)
