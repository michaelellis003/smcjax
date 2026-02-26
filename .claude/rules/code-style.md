---
paths:
  - "smcjax/**/*.py"
  - "tests/**/*.py"
---

# Code Style Rules

All code MUST conform to these standards enforced by Ruff and Pyright.

## Formatting (Ruff)

- **Line length:** 79 characters maximum
- **Indent:** 4 spaces (no tabs)
- **Quotes:** Single quotes everywhere
- **Target Python:** 3.10+

## Linting (Ruff)

Enabled rules (see `pyproject.toml [tool.ruff.lint]` for canonical list):

- **F** — pyflakes
- **E, W** — pycodestyle (errors and warnings)
- **N** — pep8-naming
- **D** — pydocstyle
- **UP** — pyupgrade
- **B** — flake8-bugbear
- **SIM** — flake8-simplify
- **I** — isort
- **C4** — flake8-comprehensions
- **C90** — mccabe complexity (max 10)
- **T20** — flake8-print (no print statements)
- **S** — flake8-bandit (security)
- **PT** — flake8-pytest-style
- **PLR** — Pylint refactoring rules
- **RUF** — Ruff-specific rules

Tests (`tests/*`) are exempt from docstrings (D), assert warnings
(S101), and magic value checks (PLR2004).

## Docstrings

- **Required** on all public modules, classes, and functions.
- Use **Google-style** convention.
- Include `Args` and `Returns` sections with types.

```python
def multiply(a: float, b: int) -> float:
    """Multiply a float by an integer.

    Args:
        a (float): The float number.
        b (int): The integer number.

    Returns:
        float: The product of the float and the integer.
    """
    return a * b
```

## Type Annotations

- All function parameters and return values MUST have type hints.
- Pyright enforces this (excludes `tests/` directory).

## Imports

- Public API re-exported from `smcjax/__init__.py`
  using `__all__`.
- Import ordering managed by isort via Ruff.
- Use `from .module import name` for intra-package imports.

## License Headers

- All `.py` files have SPDX license headers. The `insert-license`
  pre-commit hook enforces them on new files.
- Place after shebang (if present), before module docstring.
- Format:
  ```python
  # Copyright YYYY Author Name
  # SPDX-License-Identifier: LICENSE-ID
  ```
