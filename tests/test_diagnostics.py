# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.diagnostics.

Cross-validates against Dynamax Kalman filter and verifies
mathematical properties of diagnostic functions.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest

from smcjax.bootstrap import bootstrap_filter
from smcjax.diagnostics import (
    log_bayes_factor,
    log_ml_increments,
    particle_diversity,
    replicated_log_ml,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)


def _mvn_sample(key, mean, cov, shape=()):
    """Sample from a multivariate normal using pure JAX."""
    chol = jnp.linalg.cholesky(cov)
    d = mean.shape[-1]
    z = jr.normal(key, (*shape, d))
    return mean + z @ chol.T


def _mvn_logpdf(x, mean, cov):
    """Log-pdf of a multivariate normal using jax.scipy."""
    return jstats.multivariate_normal.logpdf(x, mean, cov)


def _make_smcjax_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn)."""
    m0 = lgssm_params['initial_mean']
    P0 = lgssm_params['initial_cov']
    F = lgssm_params['dynamics_weights']
    Q = lgssm_params['dynamics_cov']
    H = lgssm_params['emissions_weights']
    R = lgssm_params['emissions_cov']

    def initial_sampler(key, n):
        return _mvn_sample(key, m0, P0, shape=(n,))

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, Q)

    def log_observation_fn(emission, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, mean, R)

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

    def test_weighted_mean_matches_kalman(self, lgssm_params, lgssm_data):
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

    def test_weighted_variance_uniform_weights(self, lgssm_params, lgssm_data):
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
            log_evidence_increments=pf_post.log_evidence_increments,
        )

        wvar = weighted_variance(uniform_post)
        # Unweighted variance
        uvar = jnp.var(pf_post.filtered_particles, axis=1)

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
        assert jnp.allclose(medians[:, 0, :], means, atol=0.2)

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

        covered = jnp.all((states >= lower) & (states <= upper), axis=-1)
        coverage = float(jnp.mean(covered))

        # With T=50, expect ~95% coverage but allow Monte Carlo
        # variation: anything above 70% is acceptable
        assert coverage > 0.70, f'Coverage {coverage:.2%} too low'


class TestLogMLIncrements:
    """Tests for log_ml_increments."""

    def test_log_ml_increments_sum_to_total(self, lgssm_params, lgssm_data):
        """Increments should sum to total marginal log-likelihood."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        increments = log_ml_increments(pf_post)

        assert float(jnp.sum(increments)) == pytest.approx(
            float(pf_post.marginal_loglik), abs=1e-6
        )


class TestParticleDiversity:
    """Tests for particle_diversity."""

    def test_particle_diversity_bounded(self, lgssm_params, lgssm_data):
        """Diversity should be in [0, 1] at every time step."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        diversity = particle_diversity(pf_post)

        assert jnp.all(diversity >= 0.0)
        assert jnp.all(diversity <= 1.0)
        # With 1000 particles, first step should have high diversity
        assert float(diversity[0]) > 0.5


class TestDiagnosticsJIT:
    """All diagnostics should be JIT-compatible."""

    def test_diagnostics_jit_compatible(self, lgssm_params, lgssm_data):
        """Diagnostics compile and run under jax.jit."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)

        jax.jit(weighted_mean)(pf_post)
        jax.jit(weighted_variance)(pf_post)
        jax.jit(lambda p: weighted_quantile(p, jnp.array([0.5])))(pf_post)
        jax.jit(log_ml_increments)(pf_post)
        jax.jit(particle_diversity)(pf_post)


class TestLogBayesFactor:
    """Tests for log_bayes_factor."""

    def test_log_bayes_factor_symmetric(self):
        """BF(M1, M2) = -BF(M2, M1)."""
        bf = log_bayes_factor(jnp.float64(-70.0), jnp.float64(-75.0))
        bf_rev = log_bayes_factor(jnp.float64(-75.0), jnp.float64(-70.0))
        assert float(bf) == pytest.approx(-float(bf_rev), abs=1e-10)

    def test_log_bayes_factor_value(self):
        """BF is difference of log-MLs."""
        bf = log_bayes_factor(jnp.float64(-70.0), jnp.float64(-75.0))
        assert float(bf) == pytest.approx(5.0, abs=1e-10)


class TestReplicatedLogML:
    """Tests for replicated_log_ml."""

    def test_replicated_log_ml_shape(self, lgssm_params, lgssm_data):
        """Should return array of shape (num_replicates,)."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)

        def filter_fn(key):
            return bootstrap_filter(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=500,
            ).marginal_loglik

        result = replicated_log_ml(jr.PRNGKey(0), filter_fn, num_replicates=10)
        assert result.shape == (10,)
        assert jnp.all(jnp.isfinite(result))

    def test_replicated_log_ml_variability(self, lgssm_params, lgssm_data):
        """Replicates should have non-zero variance."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)

        def filter_fn(key):
            return bootstrap_filter(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=200,
            ).marginal_loglik

        result = replicated_log_ml(jr.PRNGKey(1), filter_fn, num_replicates=20)
        assert float(jnp.var(result)) > 0.0


class TestParamWeightedMean:
    """Tests for param_weighted_mean."""

    def test_param_weighted_mean_shape(self, lgssm_params, lgssm_data):
        """Output shape should be (ntime, param_dim)."""
        from smcjax.diagnostics import param_weighted_mean
        from smcjax.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params['initial_mean']
        P0 = lgssm_params['initial_cov']
        F = lgssm_params['dynamics_weights']
        Q = lgssm_params['dynamics_cov']
        H = lgssm_params['emissions_weights']
        R = lgssm_params['emissions_cov']

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(42),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        result = param_weighted_mean(post)
        ntime = emissions.shape[0]
        assert result.shape == (ntime, 1)
        assert jnp.all(jnp.isfinite(result))

    def test_param_weighted_mean_finite_values(self, lgssm_params, lgssm_data):
        """All param mean values should be finite."""
        from smcjax.diagnostics import param_weighted_mean
        from smcjax.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params['initial_mean']
        P0 = lgssm_params['initial_cov']
        F = lgssm_params['dynamics_weights']
        Q = lgssm_params['dynamics_cov']
        H = lgssm_params['emissions_weights']
        R = lgssm_params['emissions_cov']

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(7),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        param_means = param_weighted_mean(post)
        ntime = emissions.shape[0]
        assert param_means.shape == (ntime, 1)
        assert jnp.all(jnp.isfinite(param_means))


class TestParamWeightedQuantile:
    """Tests for param_weighted_quantile."""

    def test_param_weighted_quantile_monotone(self, lgssm_params, lgssm_data):
        """Lower quantile <= upper quantile at every step."""
        from smcjax.diagnostics import param_weighted_quantile
        from smcjax.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params['initial_mean']
        P0 = lgssm_params['initial_cov']
        F = lgssm_params['dynamics_weights']
        Q = lgssm_params['dynamics_cov']
        H = lgssm_params['emissions_weights']
        R = lgssm_params['emissions_cov']

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(42),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        q = jnp.array([0.025, 0.5, 0.975])
        result = param_weighted_quantile(post, q)
        ntime = emissions.shape[0]
        assert result.shape == (ntime, 3, 1)
        # Monotonicity: q025 <= q50 <= q975
        assert jnp.all(result[:, 0, :] <= result[:, 1, :])
        assert jnp.all(result[:, 1, :] <= result[:, 2, :])


class TestCRPS:
    """Tests for crps."""

    def test_crps_nonnegative(self):
        """CRPS should always be non-negative."""
        from smcjax.diagnostics import crps

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = crps(predictions, jnp.float64(3.0))
        assert float(result) >= 0.0

    def test_crps_zero_for_perfect_prediction(self):
        """CRPS = 0 when all predictions equal observation."""
        from smcjax.diagnostics import crps

        obs = jnp.float64(5.0)
        predictions = jnp.full(100, 5.0)
        result = crps(predictions, obs)
        assert float(result) == pytest.approx(0.0, abs=1e-10)

    def test_crps_known_value(self):
        """CRPS for known distribution matches analytical result."""
        from smcjax.diagnostics import crps

        # For predictions = {0, 1} with equal weight, obs = 0.5:
        # E|Y - y| = 0.5*(|0-0.5| + |1-0.5|) = 0.5
        # E|Y - Y'| = 0.5*(|0-0| + |0-1| + |1-0| + |1-1|)/2
        #           = 0.5*(0 + 1 + 1 + 0)/2 but actually:
        # E|Y-Y'| = mean of all |yi-yj| = (0+1+1+0)/4 = 0.5
        # CRPS = 0.5 - 0.5*0.5 = 0.25
        predictions = jnp.array([0.0, 1.0])
        result = crps(predictions, jnp.float64(0.5))
        assert float(result) == pytest.approx(0.25, abs=1e-10)

    def test_crps_jit_compatible(self):
        """CRPS should work under jax.jit."""
        from smcjax.diagnostics import crps

        predictions = jnp.array([1.0, 2.0, 3.0])
        result = jax.jit(crps)(predictions, jnp.float64(2.0))
        assert jnp.isfinite(result)
