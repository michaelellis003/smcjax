# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
r"""Bootstrap (SIR) particle filter.

The bootstrap filter [Gordon *et al.*, 1993] is the simplest Sequential
Monte Carlo algorithm.  At each time step it:

1. **Resamples** (conditionally on ESS) to focus particles on
   high-likelihood regions.
2. **Propagates** particles through the transition prior.
3. **Weights** particles by the observation likelihood.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float

from smcjax.containers import ParticleFilterPosterior, ParticleState
from smcjax.ess import ess as compute_ess
from smcjax.resampling import systematic
from smcjax.types import PRNGKeyT, Scalar
from smcjax.weights import log_normalize, normalize


def bootstrap_filter(
    key: PRNGKeyT,
    initial_sampler: Callable,
    transition_sampler: Callable,
    log_observation_fn: Callable,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: Callable = systematic,
    resampling_threshold: float = 0.5,
) -> ParticleFilterPosterior:
    r"""Run a bootstrap (SIR) particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles) -> particles``
            that draws from the initial state distribution
            :math:`p(z_1)`.
        transition_sampler: Function ``(key, state) -> state`` that draws
            from the transition distribution
            :math:`p(z_t \mid z_{t-1})`.  Will be ``vmap``-ped over the
            particle dimension internally.
        log_observation_fn: Function
            ``(emission, state) -> log_prob`` that evaluates the
            observation log-density :math:`\log p(y_t \mid z_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        resampling_fn: Resampling algorithm matching the Blackjax
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to :func:`~smcjax.resampling.systematic`.
        resampling_threshold: Fraction of ``num_particles`` below which
            resampling is triggered (e.g. 0.5 means resample when
            ``ESS < 0.5 * N``).

    Returns:
        :class:`~smcjax.containers.ParticleFilterPosterior` containing
        filtered particles, log weights, ancestor indices, the
        marginal log-likelihood estimate, and ESS trace.
    """
    key, init_key = jr.split(key)
    log_n = jnp.log(jnp.asarray(num_particles, dtype=jnp.float64))

    # --- Initialise at t=0 -------------------------------------------------
    # Sample from prior and weight by first observation.
    particles_0 = initial_sampler(init_key, num_particles)
    log_obs_0 = vmap(lambda z: log_observation_fn(emissions[0], z))(
        particles_0
    )
    # Evidence at t=0: (1/N) * sum_i p(y_0 | x_0^i)
    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0 = compute_ess(log_w_0)
    identity_ancestors = jnp.arange(num_particles, dtype=jnp.int32)

    init_state = ParticleState(
        particles=particles_0,
        log_weights=log_w_0,
        log_marginal_likelihood=log_ev_0,
    )

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(
        carry: ParticleState,
        args: tuple[PRNGKeyT, Float[Array, " emission_dim"]],
    ) -> tuple[ParticleState, tuple[Array, Array, Array, Array]]:
        state, (step_key, y_t) = carry, args
        k1, k2 = jr.split(step_key)

        # 1. Conditionally resample
        cur_ess = compute_ess(state.log_weights)
        threshold = resampling_threshold * num_particles
        w_norm = normalize(state.log_weights)

        do_resample = cur_ess < threshold
        ancestors = lax.cond(
            do_resample,
            lambda: resampling_fn(k1, w_norm, num_particles),
            lambda: identity_ancestors,
        )
        resampled_particles = state.particles[ancestors]

        # 2. Propagate through transition
        keys = jr.split(k2, num_particles)
        propagated = vmap(transition_sampler)(keys, resampled_particles)

        # 3. Weight by observation likelihood
        log_obs = vmap(lambda z: log_observation_fn(y_t, z))(propagated)

        # Compute evidence increment and normalize.
        # If resampled: weights were reset to uniform (1/N), so
        #   increment = logsumexp(log_obs) - log(N)
        # If not resampled: old normalized weights W_i sum to 1, so
        #   increment = logsumexp(log_W + log_obs)
        log_w_unnorm = jnp.where(
            do_resample,
            log_obs,
            state.log_weights + log_obs,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)
        log_ev_inc = jnp.where(
            do_resample,
            log_sum - log_n,
            log_sum,
        )

        new_state = ParticleState(
            particles=propagated,
            log_weights=log_w_norm,
            log_marginal_likelihood=(
                state.log_marginal_likelihood + log_ev_inc
            ),
        )
        ess_t: Array = jnp.asarray(compute_ess(log_w_norm))
        return new_state, (propagated, log_w_norm, ancestors, ess_t)

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, emissions.shape[0] - 1)
    final_state, (
        particles_rest,
        log_w_rest,
        ancestors_rest,
        ess_rest,
    ) = lax.scan(_step, init_state, (step_keys, emissions[1:]))

    # --- Combine t=0 with t=1..T-1 -----------------------------------------
    def _prepend(first: Array, rest: Array) -> Array:
        return jnp.concatenate(
            [jnp.expand_dims(first, 0), rest], axis=0
        )

    all_particles = _prepend(particles_0, particles_rest)
    all_log_w = _prepend(log_w_0, log_w_rest)
    all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    ess_0_arr: Array = jnp.asarray(ess_0)
    all_ess = _prepend(ess_0_arr, ess_rest)

    return ParticleFilterPosterior(
        marginal_loglik=final_state.log_marginal_likelihood,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
    )
