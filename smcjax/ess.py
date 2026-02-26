# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Effective sample size (ESS) computation.

The ESS formula used here matches Blackjax
(``blackjax.smc.ess``) so that cross-validation tests can compare
outputs directly.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from smcjax.types import Scalar


def ess(log_weights: Float[Array, " num_particles"]) -> Scalar:
    r"""Compute the effective sample size from unnormalized log weights.

    .. math::

        \mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
            = \exp\!\bigl(2\,\mathrm{LSE}(\mathbf{lw})
                         - \mathrm{LSE}(2\,\mathbf{lw})\bigr)

    This is equivalent to :math:`1 / \sum_i \tilde{w}_i^2` where
    :math:`\tilde{w}_i` are the *normalized* weights.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        The effective sample size (scalar).
    """
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Float[Array, " num_particles"]) -> Scalar:
    """Compute the *log* effective sample size.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        Log of the effective sample size (scalar).
    """
    return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)
