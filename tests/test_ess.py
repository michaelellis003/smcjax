# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.ess — cross-validated against blackjax.smc.ess."""

import jax.numpy as jnp
import pytest
from blackjax.smc.ess import ess as blackjax_ess

from smcjax.ess import ess, log_ess


class TestESSMatchesBlackjax:
    """Cross-validate ESS computation against Blackjax."""

    def test_uniform_weights(self):
        """Uniform weights -> ESS = N."""
        lw = jnp.zeros(100)
        assert jnp.allclose(ess(lw), blackjax_ess(lw), atol=1e-5)
        assert jnp.allclose(ess(lw), 100.0, atol=1e-5)

    def test_degenerate_weights(self):
        """One particle has all weight -> ESS = 1."""
        lw = jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf])
        assert jnp.allclose(ess(lw), blackjax_ess(lw), atol=1e-5)
        assert jnp.allclose(ess(lw), 1.0, atol=1e-5)

    def test_partially_degenerate(self):
        """Two equal weights, rest zero -> ESS = 2."""
        lw = jnp.array([0.0, 0.0, -jnp.inf, -jnp.inf])
        assert jnp.allclose(ess(lw), blackjax_ess(lw), atol=1e-5)
        assert jnp.allclose(ess(lw), 2.0, atol=1e-5)

    @pytest.mark.parametrize(
        'log_weights',
        [
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
            jnp.array([100.0, 100.0, 100.0, 99.0, 99.0]),
        ],
    )
    def test_arbitrary_weights(self, log_weights):
        """ESS matches Blackjax for various weight vectors."""
        assert jnp.allclose(
            ess(log_weights), blackjax_ess(log_weights), atol=1e-5
        )


class TestLogESS:
    """Tests for the log_ess helper."""

    def test_log_ess_matches_ess(self):
        """log_ess should be log of ess."""
        lw = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(jnp.exp(log_ess(lw)), ess(lw), atol=1e-7)
