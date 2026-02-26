# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""This is the smcjax package."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .main import add, hello, multiply, subtract  # re-export

try:
    __version__ = _version('smcjax')
except _PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = ['__version__', 'add', 'hello', 'multiply', 'subtract']
