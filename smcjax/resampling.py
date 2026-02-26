# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Particle resampling schemes.

All public functions share the signature
``(rng_key, weights, num_samples) -> indices`` where *weights* are
**normalized** (i.e. sum to one) — the same convention used by
Blackjax (``blackjax.smc.resampling``).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from smcjax.types import PRNGKeyT

# --- Public resampling functions -------------------------------------------


def systematic(
    rng_key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int[Array, " num_samples"]:
    """Systematic resampling.

    Args:
        rng_key: JAX PRNG key.
        weights: Normalized importance weights (sum to 1).
        num_samples: Number of indices to draw.

    Returns:
        Resampled ancestor indices.
    """
    return _systematic_or_stratified(
        rng_key, weights, num_samples, is_systematic=True
    )


def stratified(
    rng_key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int[Array, " num_samples"]:
    """Stratified resampling.

    Args:
        rng_key: JAX PRNG key.
        weights: Normalized importance weights (sum to 1).
        num_samples: Number of indices to draw.

    Returns:
        Resampled ancestor indices.
    """
    return _systematic_or_stratified(
        rng_key, weights, num_samples, is_systematic=False
    )


def multinomial(
    rng_key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int[Array, " num_samples"]:
    """Multinomial resampling.

    Higher variance than systematic/stratified; included for
    completeness and reference comparisons.

    Args:
        rng_key: JAX PRNG key.
        weights: Normalized importance weights (sum to 1).
        num_samples: Number of indices to draw.

    Returns:
        Resampled ancestor indices.
    """
    n = weights.shape[0]
    linspace = _sorted_uniforms(rng_key, num_samples)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


def residual(
    rng_key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int[Array, " num_samples"]:
    """Residual resampling.

    Deterministically copies the integer part of each expected count,
    then resamples the residuals via multinomial.

    Args:
        rng_key: JAX PRNG key.
        weights: Normalized importance weights (sum to 1).
        num_samples: Number of indices to draw.

    Returns:
        Resampled ancestor indices.
    """
    key1, key2 = jax.random.split(rng_key)
    n = weights.shape[0]
    n_sample_weights = num_samples * weights
    idx = jnp.arange(num_samples)

    integer_part = jnp.floor(n_sample_weights).astype(jnp.int32)
    sum_integer_part = jnp.sum(integer_part)

    residual_part = n_sample_weights - integer_part
    residual_sample = multinomial(
        key1,
        residual_part / (num_samples - sum_integer_part),
        num_samples,
    )
    residual_sample = jax.random.permutation(key2, residual_sample)

    integer_idx = jnp.repeat(
        jnp.arange(n + 1),
        jnp.concatenate(
            [integer_part, jnp.array([num_samples - sum_integer_part])],
            0,
        ),
        total_repeat_length=num_samples,
    )

    return jnp.where(idx >= sum_integer_part, residual_sample, integer_idx)


# --- Internal helpers -------------------------------------------------------


def _systematic_or_stratified(
    rng_key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
    is_systematic: bool,
) -> Int[Array, " num_samples"]:
    """Shared implementation for systematic and stratified resampling."""
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    cumsum = jnp.cumsum(weights)
    linspace = (
        jnp.arange(num_samples, dtype=weights.dtype) + u
    ) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


def _sorted_uniforms(
    rng_key: PRNGKeyT,
    n: int,
) -> Float[Array, " n"]:
    """Generate *n* sorted uniform random variates in [0, 1).

    Uses the exponential spacings trick (credit: Nicolas Chopin).
    """
    us = jax.random.uniform(rng_key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]
