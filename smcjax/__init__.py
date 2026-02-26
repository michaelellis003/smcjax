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

from smcjax.bootstrap import bootstrap_filter
from smcjax.containers import ParticleFilterPosterior, ParticleState
from smcjax.weights import log_normalize, normalize

try:
    __version__ = _version('smcjax')
except _PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'ParticleFilterPosterior',
    'ParticleState',
    '__version__',
    'bootstrap_filter',
    'ess',
    'log_ess',
    'log_normalize',
    'multinomial',
    'normalize',
    'residual',
    'stratified',
    'systematic',
]
