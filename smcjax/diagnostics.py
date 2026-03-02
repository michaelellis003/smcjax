# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
r"""Diagnostic utilities for particle filter posteriors.

Posterior summaries (Vehtari: *report posterior summaries with
uncertainty*; McElreath: *always report intervals, not just means*):

- :func:`weighted_mean` — weighted posterior mean at each time step
- :func:`weighted_variance` — weighted posterior variance
- :func:`weighted_quantile` — weighted quantiles for credible
  intervals
- :func:`param_weighted_mean` — weighted parameter mean (Liu-West)
- :func:`param_weighted_quantile` — weighted parameter quantiles

Computational faithfulness (Vehtari: *can we trust the computation?*):

- :func:`particle_diversity` — fraction of unique particles per step
- :func:`log_ml_increments` — per-step evidence contributions

Model comparison:

- :func:`log_bayes_factor` — log Bayes factor between two models
- :func:`replicated_log_ml` — Monte Carlo variability of log-ML

Scoring rules:

- :func:`crps` — Continuous Ranked Probability Score

All functions are pure, stateless, operate on arrays from
:class:`~smcjax.containers.ParticleFilterPosterior` or
:class:`~smcjax.containers.LiuWestPosterior`, and are
JIT-compatible.
"""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float, Int

from smcjax.containers import LiuWestPosterior, ParticleFilterPosterior
from smcjax.types import PRNGKeyT, Scalar
from smcjax.weights import normalize


def weighted_mean(
    posterior: ParticleFilterPosterior,
) -> Float[Array, 'ntime state_dim']:
    r"""Compute the weighted mean of particles at each time step.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted means, shape ``(ntime, state_dim)``.
    """
    # weights: (ntime, num_particles)
    weights = vmap(normalize)(posterior.filtered_log_weights)
    # particles: (ntime, num_particles, state_dim)
    return jnp.einsum('tn,tnd->td', weights, posterior.filtered_particles)


def weighted_variance(
    posterior: ParticleFilterPosterior,
) -> Float[Array, 'ntime state_dim']:
    r"""Compute the weighted variance of particles at each time step.

    Uses the formula :math:`V = \sum_i w_i (x_i - \mu)^2` where
    :math:`\mu` is the weighted mean.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted variances, shape ``(ntime, state_dim)``.
    """
    weights = vmap(normalize)(posterior.filtered_log_weights)
    means = weighted_mean(posterior)
    # (ntime, num_particles, state_dim) - (ntime, 1, state_dim)
    deviations = posterior.filtered_particles - means[:, None, :]
    return jnp.einsum('tn,tnd->td', weights, deviations**2)


def weighted_quantile(
    posterior: ParticleFilterPosterior,
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, 'ntime num_quantiles state_dim']:
    r"""Compute weighted quantiles of particles at each time step.

    Uses a sorted resampling approach for JIT compatibility:
    sorts particles, computes cumulative weights, and interpolates.

    Args:
        posterior: Particle filter posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, state_dim)``.
    """
    particles = posterior.filtered_particles
    log_weights = posterior.filtered_log_weights

    weights = vmap(normalize)(log_weights)  # (ntime, num_particles)

    def _quantile_one_time_dim(
        p: Float[Array, ' num_particles'],
        w: Float[Array, ' num_particles'],
    ) -> Float[Array, ' num_quantiles']:
        """Compute quantiles for one time step, one state dim."""
        sort_idx = jnp.argsort(p)
        p_sorted = p[sort_idx]
        w_sorted = w[sort_idx]
        # Cumulative weights, centered at midpoint
        cum_w = jnp.cumsum(w_sorted)
        # Interpolate to find quantile values
        return jnp.interp(q, cum_w, p_sorted)

    def _quantile_one_time(
        particles_t: Float[Array, 'num_particles state_dim'],
        weights_t: Float[Array, ' num_particles'],
    ) -> Float[Array, 'num_quantiles state_dim']:
        """Compute quantiles for one time step, all state dims."""
        return vmap(_quantile_one_time_dim, in_axes=(1, None))(
            particles_t, weights_t
        ).T

    return vmap(_quantile_one_time)(particles, weights)


def log_ml_increments(
    posterior: ParticleFilterPosterior,
) -> Float[Array, ' ntime']:
    r"""Extract per-step log marginal likelihood increments.

    The marginal log-likelihood can be decomposed as:

    .. math::

        \log p(y_{1:T}) = \sum_{t=1}^T
            \log p(y_t \mid y_{1:t-1})

    This function returns the individual increments, which diagnose
    which observations are hardest for the model.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Per-step evidence increments, shape ``(ntime,)``.  These sum
        to ``posterior.marginal_loglik``.
    """
    return posterior.log_evidence_increments


def particle_diversity(
    posterior: ParticleFilterPosterior,
) -> Float[Array, ' ntime']:
    r"""Compute the fraction of unique particles at each time step.

    Particle diversity measures path degeneracy: a value near 1 means
    most particles are distinct, while near 0 means heavy duplication
    after resampling.

    Uses an indicator-based method (not ``jnp.unique``) for JIT
    compatibility: counts the fraction of particles that differ from
    their predecessor in the sorted order.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Diversity fraction in [0, 1] at each time step,
        shape ``(ntime,)``.
    """
    ancestors = posterior.ancestors  # (ntime, num_particles)
    num_particles = ancestors.shape[1]

    def _diversity_one_step(
        anc: Int[Array, ' num_particles'],
    ) -> Float[Array, '']:
        """Count fraction of unique ancestors at one time step."""
        sorted_anc = jnp.sort(anc)
        # First element is always unique; subsequent are unique if
        # different from predecessor
        is_unique = jnp.concatenate(
            [
                jnp.array([True]),
                sorted_anc[1:] != sorted_anc[:-1],
            ]
        )
        return jnp.sum(is_unique) / num_particles

    return vmap(_diversity_one_step)(ancestors)


def log_bayes_factor(
    log_ml_1: Scalar,
    log_ml_2: Scalar,
) -> Scalar:
    r"""Compute the log Bayes factor between two models.

    .. math::

        \log BF_{12} = \log p(y_{1:T} \mid M_1)
                     - \log p(y_{1:T} \mid M_2)

    Positive values favour model 1; negative values favour model 2.

    Args:
        log_ml_1: Log marginal likelihood of model 1.
        log_ml_2: Log marginal likelihood of model 2.

    Returns:
        Scalar log Bayes factor.
    """
    return jnp.asarray(log_ml_1) - jnp.asarray(log_ml_2)


def replicated_log_ml(
    key: PRNGKeyT,
    filter_fn: Callable[[PRNGKeyT], Scalar],
    num_replicates: int,
) -> Float[Array, ' num_replicates']:
    r"""Run a particle filter multiple times to assess log-ML variability.

    Uses :func:`jax.vmap` over PRNG keys for efficient parallel
    evaluation.  The resulting distribution of log-ML estimates
    quantifies Monte Carlo uncertainty in the evidence.

    Args:
        key: JAX PRNG key.
        filter_fn: Function ``(key) -> scalar`` that runs a particle
            filter and returns the marginal log-likelihood.
        num_replicates: Number of independent filter runs.

    Returns:
        Array of log-ML estimates, shape ``(num_replicates,)``.
    """
    keys = jr.split(key, num_replicates)
    return jnp.asarray(vmap(filter_fn)(keys))


def param_weighted_mean(
    posterior: LiuWestPosterior,
) -> Float[Array, 'ntime param_dim']:
    r"""Compute the weighted mean of parameter particles at each step.

    Args:
        posterior: Liu-West filter posterior output.

    Returns:
        Weighted parameter means, shape ``(ntime, param_dim)``.
    """
    weights = vmap(normalize)(posterior.filtered_log_weights)
    return jnp.einsum('tn,tnd->td', weights, posterior.filtered_params)


def param_weighted_quantile(
    posterior: LiuWestPosterior,
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, 'ntime num_quantiles param_dim']:
    r"""Compute weighted quantiles of parameter particles at each step.

    Args:
        posterior: Liu-West filter posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, param_dim)``.
    """
    params = posterior.filtered_params
    weights = vmap(normalize)(posterior.filtered_log_weights)

    def _quantile_one_time_dim(
        p: Float[Array, ' num_particles'],
        w: Float[Array, ' num_particles'],
    ) -> Float[Array, ' num_quantiles']:
        """Compute quantiles for one time step, one param dim."""
        sort_idx = jnp.argsort(p)
        p_sorted = p[sort_idx]
        w_sorted = w[sort_idx]
        cum_w = jnp.cumsum(w_sorted)
        return jnp.interp(q, cum_w, p_sorted)

    def _quantile_one_time(
        params_t: Float[Array, 'num_particles param_dim'],
        weights_t: Float[Array, ' num_particles'],
    ) -> Float[Array, 'num_quantiles param_dim']:
        """Compute quantiles for one time step, all param dims."""
        return vmap(_quantile_one_time_dim, in_axes=(1, None))(
            params_t, weights_t
        ).T

    return vmap(_quantile_one_time)(params, weights)


def crps(
    predictions: Float[Array, ' num_samples'],
    observation: Scalar,
) -> Scalar:
    r"""Compute the Continuous Ranked Probability Score.

    CRPS is a proper scoring rule for probabilistic forecasts:

    .. math::

        \text{CRPS} = \mathbb{E}|Y - y|
                     - \tfrac{1}{2}\,\mathbb{E}|Y - Y'|

    where :math:`Y, Y'` are iid predictive samples and :math:`y`
    is the observation.

    Args:
        predictions: iid samples from the predictive distribution.
        observation: Observed scalar value.

    Returns:
        Scalar CRPS (lower is better, zero for perfect prediction).
    """
    obs = jnp.asarray(observation)
    abs_errors = jnp.abs(predictions - obs)
    # E|Y - y|
    term1 = jnp.mean(abs_errors)
    # E|Y - Y'| via all-pairs (efficient for moderate sample sizes)
    diffs = jnp.abs(predictions[:, None] - predictions[None, :])
    term2 = jnp.mean(diffs)
    return jnp.asarray(term1 - 0.5 * term2)
