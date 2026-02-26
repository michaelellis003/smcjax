---
paths:
  - "smcjax/**/*.py"
  - "tests/**/*.py"
---

# Python Idioms

Prefer modern stdlib tools over hand-rolled solutions. These
patterns are not enforced by Ruff or Pyright but produce cleaner,
more maintainable code.

## Dataclasses for Structured Data

Use `dataclasses.dataclass` (or `NamedTuple` for immutable records)
instead of plain dicts when a value has a fixed set of fields.

```python
from dataclasses import dataclass

# Good
@dataclass
class Coordinate:
    x: float
    y: float

# Avoid — no autocomplete, no type safety, easy to misspell keys
point = {'x': 1.0, 'y': 2.0}
```

Use `@dataclass(frozen=True)` when instances should be immutable
and hashable.

## Enum for Fixed Value Sets

Use `enum.Enum` (or `enum.StrEnum` on 3.11+) instead of string
constants or magic values.

```python
from enum import Enum

class Status(Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    ARCHIVED = 'archived'
```

## Pathlib over os.path

Use `pathlib.Path` for all filesystem operations. It is clearer,
composable, and cross-platform.

```python
from pathlib import Path

config_path = Path('config') / 'settings.toml'
text = config_path.read_text()
```

## Generators for Lazy Iteration

When processing large sequences, use generators or generator
expressions to avoid loading everything into memory at once.

```python
# Good — lazy, O(1) memory
def read_records(path: Path):
    with path.open() as f:
        for line in f:
            yield parse(line)

# Avoid for large files — loads entire list into memory
records = [parse(line) for line in path.read_text().splitlines()]
```

Use list comprehensions when you need the full list in memory
(e.g., for random access or `len()`).

## Comprehensions over map/filter

Prefer list, dict, and set comprehensions over `map()` and
`filter()` with lambdas. They are more readable and Pythonic.

```python
# Good
squares = [x * x for x in numbers if x > 0]

# Avoid
squares = list(map(lambda x: x * x, filter(lambda x: x > 0, numbers)))
```

## Context Managers for Resource Cleanup

Use `with` for any resource that has setup and teardown — files,
locks, network connections, temporary directories.

For custom resources, implement `__enter__`/`__exit__` or use
`contextlib.contextmanager`.

```python
from contextlib import contextmanager

@contextmanager
def temporary_env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old
```

## functools for Caching and Wrapping

Use `functools.lru_cache` or `functools.cache` for pure-function
memoization instead of hand-rolled caches.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

Use `functools.partial` to fix arguments instead of writing
one-line wrapper functions.

## __slots__ for Memory-Critical Classes

When creating many instances of a class, define `__slots__` to
reduce per-instance memory overhead.

```python
class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
```

Only use this when you have measured a memory concern — it
restricts dynamic attribute assignment.

## Unpacking and Starred Expressions

Use tuple unpacking and starred expressions for clarity.

```python
first, *rest = items
head, *_, tail = sequence
```

## Use `any()` and `all()` for Boolean Checks

```python
# Good
if any(user.is_admin for user in users):
    grant_access()

# Avoid
found = False
for user in users:
    if user.is_admin:
        found = True
        break
```

## Logging over Print

Use the `logging` module instead of print statements for
diagnostic output. Ruff's T20 rule bans `print()`, but beyond
compliance, use structured log levels intentionally:

- `DEBUG` — detailed diagnostic info (disabled in production)
- `INFO` — confirmation that things work as expected
- `WARNING` — something unexpected but recoverable
- `ERROR` — a failure that prevents a specific operation
- `CRITICAL` — the program cannot continue
