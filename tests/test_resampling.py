# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.resampling — cross-validated against Blackjax."""

import blackjax.smc.resampling as bj_resample
import jax.numpy as jnp
import jax.random as jr
import pytest

from smcjax.resampling import multinomial, residual, stratified, systematic


class TestSystematicMatchesBlackjax:
    """Systematic resampling should produce identical indices to Blackjax."""

    def test_identical_indices(self):
        key = jr.PRNGKey(0)
        w = jnp.array([0.1, 0.3, 0.4, 0.2])
        n = w.shape[0]
        ours = systematic(key, w, n)
        theirs = bj_resample.systematic(key, w, n)
        assert jnp.array_equal(ours, theirs)

    def test_uniform_weights(self):
        key = jr.PRNGKey(1)
        n = 100
        w = jnp.ones(n) / n
        ours = systematic(key, w, n)
        theirs = bj_resample.systematic(key, w, n)
        assert jnp.array_equal(ours, theirs)

    def test_degenerate_weight(self):
        """All weight on one particle -> all indices should equal that particle."""
        key = jr.PRNGKey(2)
        w = jnp.array([0.0, 0.0, 1.0, 0.0])
        idx = systematic(key, w, 4)
        assert jnp.all(idx == 2)


class TestStratifiedMatchesBlackjax:
    """Stratified resampling should produce identical indices to Blackjax."""

    def test_identical_indices(self):
        key = jr.PRNGKey(0)
        w = jnp.array([0.1, 0.3, 0.4, 0.2])
        n = w.shape[0]
        ours = stratified(key, w, n)
        theirs = bj_resample.stratified(key, w, n)
        assert jnp.array_equal(ours, theirs)


class TestMultinomialMatchesBlackjax:
    """Multinomial resampling should produce identical indices to Blackjax."""

    def test_identical_indices(self):
        key = jr.PRNGKey(0)
        w = jnp.array([0.1, 0.3, 0.4, 0.2])
        n = w.shape[0]
        ours = multinomial(key, w, n)
        theirs = bj_resample.multinomial(key, w, n)
        assert jnp.array_equal(ours, theirs)


class TestResidualMatchesBlackjax:
    """Residual resampling should produce identical indices to Blackjax."""

    def test_identical_indices(self):
        key = jr.PRNGKey(0)
        w = jnp.array([0.1, 0.3, 0.4, 0.2])
        n = w.shape[0]
        ours = residual(key, w, n)
        theirs = bj_resample.residual(key, w, n)
        assert jnp.array_equal(ours, theirs)


class TestResamplingProperties:
    """Statistical tests for resampling schemes."""

    @pytest.mark.parametrize(
        'resample_fn', [systematic, stratified, multinomial, residual]
    )
    def test_valid_indices(self, resample_fn):
        """All resampled indices must be in [0, N)."""
        key = jr.PRNGKey(7)
        n = 50
        w = jnp.ones(n) / n
        idx = resample_fn(key, w, n)
        assert jnp.all(idx >= 0)
        assert jnp.all(idx < n)

    @pytest.mark.parametrize(
        'resample_fn', [systematic, stratified, multinomial, residual]
    )
    def test_correct_shape(self, resample_fn):
        """Output should have num_samples elements."""
        key = jr.PRNGKey(8)
        w = jnp.array([0.25, 0.25, 0.25, 0.25])
        idx = resample_fn(key, w, 10)
        assert idx.shape == (10,)
