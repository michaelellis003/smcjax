# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures.

Add project-wide pytest fixtures here. They will be automatically
discovered by pytest and available to all test files.
"""

import pytest

import smcjax


@pytest.fixture
def package():
    """Return the top-level package module for introspection."""
    return smcjax
