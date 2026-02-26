# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Type aliases for smcjax.

Matches the conventions used by Dynamax (``dynamax.types``).
"""

from typing import Union

from jaxtyping import Array, Float, Int

PRNGKeyT = Array
"""JAX PRNG key."""

Scalar = Union[float, Float[Array, ""]]
"""Python float or scalar JAX array with float dtype."""

IntScalar = Union[int, Int[Array, ""]]
"""Python int or scalar JAX array with int dtype."""
