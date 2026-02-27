# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.diagnostics.

Cross-validates against Dynamax Kalman filter and verifies
mathematical properties of diagnostic functions.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from smcjax.bootstrap import bootstrap_filter
from smcjax.diagnostics import (
    log_ml_increments,
    particle_diversity,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)


def _make_smcjax_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn)."""
    m0 = lgssm_params['initial_mean']
    P0 = lgssm_params['initial_cov']
    F = lgssm_params['dynamics_weights']
    Q = lgssm_params['dynamics_cov']
    H = lgssm_params['emissions_weights']
    R = lgssm_params['emissions_cov']

    def initial_sampler(key, n):
        return tfd.MultivariateNormalFullCovariance(
            m0, P0
        ).sample(n, seed=key)

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return tfd.MultivariateNormalFullCovariance(
            mean, Q
        ).sample(seed=key)

    def log_observation_fn(emission, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return tfd.MultivariateNormalFullCovariance(
            mean, R
        ).log_prob(emission)

    return initial_sampler, transition_sampler, log_observation_fn


def _run_bootstrap(lgssm_params, lgssm_data, n=10_000, seed=0):
    """Run bootstrap filter and return posterior."""
    _, emissions = lgssm_data
    init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
    return bootstrap_filter(
        key=jr.PRNGKey(seed),
        initial_sampler=init_fn,
        transition_sampler=trans_fn,
        log_observation_fn=obs_fn,
        emissions=emissions,
        num_particles=n,
    )


class TestWeightedMean:
    """Tests for weighted_mean."""

    def test_weighted_mean_matches_kalman(
        self, lgssm_params, lgssm_data
    ):
        """PF weighted means should track Kalman filtered means."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)
        kalman_post = lgssm_filter(params, emissions)
        kalman_means = kalman_post.filtered_means

        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        pf_means = weighted_mean(pf_post)

        assert jnp.allclose(pf_means, kalman_means, atol=0.15), (
            f'Max error: '
            f'{float(jnp.max(jnp.abs(pf_means - kalman_means))):.4f}'
        )


class TestWeightedVariance:
    """Tests for weighted_variance."""

    def test_weighted_variance_uniform_weights(
        self, lgssm_params, lgssm_data
    ):
        """With uniform weights, matches unweighted variance."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)

        # Create uniform-weight posterior for comparison
        n = pf_post.filtered_particles.shape[1]
        uniform_log_w = jnp.full_like(
            pf_post.filtered_log_weights, -jnp.log(n)
        )
        from smcjax.containers import ParticleFilterPosterior

        uniform_post = ParticleFilterPosterior(
            marginal_loglik=pf_post.marginal_loglik,
            filtered_particles=pf_post.filtered_particles,
            filtered_log_weights=uniform_log_w,
            ancestors=pf_post.ancestors,
            ess=pf_post.ess,
        )

        wvar = weighted_variance(uniform_post)
        # Unweighted variance
        uvar = jnp.var(
            pf_post.filtered_particles, axis=1
        )

        assert jnp.allclose(wvar, uvar, atol=1e-6)


class TestWeightedQuantile:
    """Tests for weighted_quantile."""

    def test_weighted_quantile_median_near_mean(
        self, lgssm_params, lgssm_data
    ):
        """For roughly symmetric posterior, median ≈ mean."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        means = weighted_mean(pf_post)
        medians = weighted_quantile(pf_post, jnp.array([0.5]))

        # medians shape: (ntime, 1, state_dim), squeeze quantile dim
        assert jnp.allclose(
            medians[:, 0, :], means, atol=0.2
        )

    def test_weighted_quantile_interval_contains_truth(
        self, lgssm_params, lgssm_data
    ):
        """95% credible interval should cover true state most of time."""
        states, _ = lgssm_data
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)

        quantiles = weighted_quantile(
            pf_post, jnp.array([0.025, 0.975])
        )  # (ntime, 2, state_dim)
        lower = quantiles[:, 0, :]
        upper = quantiles[:, 1, :]

        covered = jnp.all(
            (states >= lower) & (states <= upper), axis=-1
        )
        coverage = float(jnp.mean(covered))

        # With T=50, expect ~95% coverage but allow Monte Carlo
        # variation: anything above 70% is acceptable
        assert coverage > 0.70, f'Coverage {coverage:.2%} too low'


class TestLogMLIncrements:
    """Tests for log_ml_increments."""

    def test_log_ml_increments_sum_to_total(
        self, lgssm_params, lgssm_data
    ):
        """Increments should sum to total marginal log-likelihood."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        increments = log_ml_increments(pf_post)

        assert float(jnp.sum(increments)) == pytest.approx(
            float(pf_post.marginal_loglik), abs=1e-6
        )


class TestParticleDiversity:
    """Tests for particle_diversity."""

    def test_particle_diversity_bounded(
        self, lgssm_params, lgssm_data
    ):
        """Diversity should be in [0, 1] at every time step."""
        pf_post = _run_bootstrap(
            lgssm_params, lgssm_data, n=1_000
        )
        diversity = particle_diversity(pf_post)

        assert jnp.all(diversity >= 0.0)
        assert jnp.all(diversity <= 1.0)
        # With 1000 particles, first step should have high diversity
        assert float(diversity[0]) > 0.5


class TestDiagnosticsJIT:
    """All diagnostics should be JIT-compatible."""

    def test_diagnostics_jit_compatible(
        self, lgssm_params, lgssm_data
    ):
        """Diagnostics compile and run under jax.jit."""
        pf_post = _run_bootstrap(
            lgssm_params, lgssm_data, n=500
        )

        jax.jit(weighted_mean)(pf_post)
        jax.jit(weighted_variance)(pf_post)
        jax.jit(
            lambda p: weighted_quantile(p, jnp.array([0.5]))
        )(pf_post)
        jax.jit(log_ml_increments)(pf_post)
        jax.jit(particle_diversity)(pf_post)
