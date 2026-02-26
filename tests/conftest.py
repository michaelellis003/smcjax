# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for smcjax."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcjax


@pytest.fixture
def package():
    """Return the top-level package module for introspection."""
    return smcjax


@pytest.fixture
def key():
    """Fixed JAX PRNG key for reproducibility."""
    return jr.PRNGKey(42)


@pytest.fixture
def lgssm_params():
    """Simple 1-D linear Gaussian SSM parameters.

    Model:
        z_0  ~ N(0, 1)
        z_t  = 0.9 * z_{t-1} + eps,  eps ~ N(0, 0.5^2)
        y_t  = z_t + eta,             eta ~ N(0, 1.0^2)

    Returns a dict with keys matching Dynamax ``make_lgssm_params``.
    """
    return dict(
        initial_mean=jnp.array([0.0]),
        initial_cov=jnp.array([[1.0]]),
        dynamics_weights=jnp.array([[0.9]]),
        dynamics_cov=jnp.array([[0.25]]),  # 0.5^2
        emissions_weights=jnp.array([[1.0]]),
        emissions_cov=jnp.array([[1.0]]),
    )


@pytest.fixture
def lgssm_data(key, lgssm_params):
    """Simulate T=50 observations from the 1-D LGSSM.

    Returns (states, emissions) each of shape (50, 1).
    """
    from dynamax.linear_gaussian_ssm.inference import (
        lgssm_joint_sample,
        make_lgssm_params,
    )

    params = make_lgssm_params(**lgssm_params)
    states, emissions = lgssm_joint_sample(params, key, num_timesteps=50)
    return states, emissions


# Configure JAX to use 64-bit floats for higher precision in tests.
jax.config.update('jax_enable_x64', True)
