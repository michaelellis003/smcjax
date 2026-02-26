---
paths:
  - "tests/**/*.py"
---

# Testing Rules

## Framework and Tools

- **pytest** for test execution
- **pytest-cov** for coverage reporting
- Tests live in `tests/` and import from the package directly

## Running Tests

```bash
# Full test suite with coverage
uv run pytest -v --durations=0 --cov --cov-report=xml

# Quick run (stop at first failure)
uv run pytest -x --tb=short

# Single test file
uv run pytest tests/test_main.py -v

# Single test function
uv run pytest tests/test_main.py::test_add_two_positive_integers_returns_sum -v
```

## Test File Organization

```
tests/
├── conftest.py               # Shared fixtures (when needed)
├── test_init.py              # Package-level export tests
├── test_main.py              # Unit tests for demo functions
├── test_main_module.py       # Tests for __main__.py entry point
└── test_<module>.py          # One test file per source module
```

Future growth pattern:
```
tests/
├── unit/                     # Fast, isolated, no I/O
│   ├── test_auth.py
│   └── test_payment.py
├── integration/              # Real deps, slower
│   └── test_auth_flow.py
└── conftest.py
```

## Test Naming Convention

```python
def test_<unit>_<scenario>_<expected_outcome>():
    """Test that <unit> <expected outcome> when <scenario>."""
```

Examples:
```python
def test_add_two_positive_integers_returns_sum():
    """Test that add returns the sum of two positive integers."""
    assert add(1, 2) == 3

def test_hello_default_name_returns_hello_world():
    """Test that hello returns 'Hello world' with default arg."""
    assert hello() == 'Hello world'

def test_multiply_by_zero_returns_zero():
    """Test that multiply returns zero when multiplied by zero."""
    assert multiply(5.0, 0) == 0.0
```

## Test Structure (Arrange-Act-Assert)

```python
def test_refresh_token_with_valid_token_returns_new_pair():
    """Test that refresh returns a new token pair for valid input."""
    # Arrange
    token = create_valid_token()

    # Act
    result = refresh(token)

    # Assert
    assert result.access_token != token
    assert result.refresh_token is not None
```

## What to Test (in order)

1. Happy path — normal expected usage
2. Boundary values — zero, one, max, empty
3. Edge cases — unicode, None, large inputs
4. Error handling — invalid input, exceptions

## CI Test Matrix

Tests run against Python 3.10, 3.11, 3.12, and 3.13 in CI.
macOS and Windows smoke tests run on the latest Python version.
Coverage reports are uploaded to Codecov.
