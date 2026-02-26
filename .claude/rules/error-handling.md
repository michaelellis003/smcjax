---
paths:
  - "smcjax/**/*.py"
  - "tests/**/*.py"
---

# Error Handling Patterns

## EAFP over LBYL

Python favors "Easier to Ask Forgiveness than Permission" over
"Look Before You Leap." Use try/except when the common case
succeeds and failure is exceptional.

```python
# Good — EAFP
try:
    value = mapping[key]
except KeyError:
    value = default

# Avoid — LBYL (duplicates the lookup)
if key in mapping:
    value = mapping[key]
else:
    value = default
```

Use LBYL only when the check is cheap and failure is common, or
when the operation has side effects that are hard to undo.

## Catch Specific Exceptions

Never catch more than you can handle. Broad catches hide bugs.

```python
# Good — specific
try:
    result = int(user_input)
except ValueError:
    result = 0

# Bad — swallows everything including KeyboardInterrupt
try:
    result = int(user_input)
except Exception:
    result = 0
```

Only use `except Exception` at true top-level boundaries (CLI
entry points, request handlers) where you intend to log and
continue.

## Chain Exceptions with `from`

When re-raising a different exception, always chain with `from`
so the original traceback is preserved.

```python
try:
    data = json.loads(raw)
except json.JSONDecodeError as exc:
    raise ConfigError(f'Invalid config: {path}') from exc
```

## Never Silently Swallow Exceptions

If you catch an exception and intentionally do nothing, explain
why with a comment. A bare `pass` in an except block is almost
always a bug.

```python
try:
    os.remove(tmp_path)
except FileNotFoundError:
    pass  # Already cleaned up — not an error
```

## Custom Exception Hierarchies

For packages, define a base exception that all custom exceptions
inherit from. This lets callers catch everything from your package
in one clause.

```python
class PackageError(Exception):
    """Base exception for this package."""

class ValidationError(PackageError):
    """Raised when input fails validation."""

class NotFoundError(PackageError):
    """Raised when a requested resource does not exist."""
```

## Fail Fast

Validate inputs at the boundary (public functions, CLI args, API
endpoints) and raise immediately. Don't let bad data travel deep
into the call stack.

```python
def process_order(order_id: int, quantity: int) -> Order:
    """Process an order.

    Args:
        order_id (int): The order identifier.
        quantity (int): Must be positive.

    Returns:
        Order: The processed order.

    Raises:
        ValueError: If quantity is not positive.
    """
    if quantity <= 0:
        raise ValueError(
            f'quantity must be positive, got {quantity}'
        )
    # ... rest of the logic
```

## Guard Clauses over Deep Nesting

Return or raise early to keep the happy path at the top
indentation level.

```python
# Good — guard clauses
def activate_user(user: User) -> None:
    if user.is_active:
        return
    if user.is_banned:
        raise PermissionError('Banned users cannot be activated')
    user.is_active = True
    user.save()

# Avoid — deep nesting
def activate_user(user: User) -> None:
    if not user.is_active:
        if not user.is_banned:
            user.is_active = True
            user.save()
        else:
            raise PermissionError('Banned users cannot be activated')
```

## Context Managers for Cleanup

Use `with` statements for anything that needs cleanup — files,
locks, database connections, temporary directories.

```python
from pathlib import Path

# Good
with Path('data.json').open() as f:
    data = json.load(f)

# Bad — risks leaking the file descriptor on exception
f = open('data.json')
data = json.load(f)
f.close()
```

## Document Raised Exceptions

Public functions that raise should document what they raise and
when, in the `Raises` section of the Google-style docstring.
