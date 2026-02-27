# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Sequential Monte Carlo and particle filtering in JAX."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from blackjax.smc.ess import ess, log_ess
from blackjax.smc.resampling import (
    multinomial,
    residual,
    stratified,
    systematic,
)

from smcjax.auxiliary import auxiliary_filter
from smcjax.bootstrap import bootstrap_filter
from smcjax.containers import ParticleFilterPosterior, ParticleState
from smcjax.diagnostics import (
    log_ml_increments,
    particle_diversity,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)
from smcjax.simulate import simulate
from smcjax.weights import log_normalize, normalize

try:
    __version__ = _version('smcjax')
except _PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'ParticleFilterPosterior',
    'ParticleState',
    '__version__',
    'auxiliary_filter',
    'bootstrap_filter',
    'ess',
    'log_ess',
    'log_ml_increments',
    'log_normalize',
    'multinomial',
    'normalize',
    'particle_diversity',
    'residual',
    'simulate',
    'stratified',
    'systematic',
    'weighted_mean',
    'weighted_quantile',
    'weighted_variance',
]
