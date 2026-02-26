# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Containers for particle filter state and posteriors.

All containers are :class:`~typing.NamedTuple` subclasses so they are
registered as JAX PyTrees by default.
"""

from typing import NamedTuple

from jaxtyping import Array, Float, Int

from smcjax.types import Scalar


class ParticleState(NamedTuple):
    r"""State of a particle cloud at a single time step.

    Attributes:
        particles: Particle values, shape ``(num_particles, state_dim)``.
        log_weights: Unnormalized log importance weights,
            shape ``(num_particles,)``.
        log_marginal_likelihood: Running log marginal likelihood estimate.
    """

    particles: Float[Array, 'num_particles state_dim']
    log_weights: Float[Array, ' num_particles']
    log_marginal_likelihood: Scalar


class ParticleFilterPosterior(NamedTuple):
    r"""Full output of a particle filter run.

    Follows the Dynamax ``PosteriorGSSMFiltered`` convention of storing
    the marginal log-likelihood as a scalar summary alongside the
    time-indexed arrays.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Particle values at each time step,
            shape ``(ntime, num_particles, state_dim)``.
        filtered_log_weights: Unnormalized log weights at each time step,
            shape ``(ntime, num_particles)``.
        ancestors: Resampled ancestor indices at each time step,
            shape ``(ntime, num_particles)``.
        ess: Effective sample size at each time step,
            shape ``(ntime,)``.
    """

    marginal_loglik: Scalar
    filtered_particles: Float[Array, 'ntime num_particles state_dim']
    filtered_log_weights: Float[Array, 'ntime num_particles']
    ancestors: Int[Array, 'ntime num_particles']
    ess: Float[Array, ' ntime']
