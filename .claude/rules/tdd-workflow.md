# TDD Workflow Rules

You MUST follow Test-Driven Development for all code changes.

## The Red-Green-Refactor Cycle

Every feature or fix begins with a failing test. No exceptions.

1. **RED** — Write a single test describing one behavior from the
   acceptance criteria. Run it. It MUST fail. If it passes, the test
   is wrong or the behavior already exists.
2. **GREEN** — Write the MINIMUM code to make that test pass. Do not
   write "good" code yet. Just make it pass.
3. **REFACTOR** — Improve the code (extract functions, rename, remove
   duplication). Tests must still pass after refactoring.
4. **REPEAT** — Pick the next behavior. Write the next failing test.

## Test Ordering

Follow this progression for each unit of work:

1. Happy path — does it work for the normal case?
2. Boundary values — zero, one, max, empty string, empty list
3. Edge cases — unicode, None, very large inputs
4. Error handling — invalid input, expected exceptions

## When to Commit During TDD

- RED → do NOT commit (broken state)
- GREEN → candidate for commit
- REFACTOR → commit here (clean, passing state)

Each commit should represent one coherent, passing behavior.

## Documentation

After all TDD cycles for a unit of work are complete, update
documentation to reflect any user-facing changes:

- **Docstrings** — ensure new/changed functions have accurate
  Google-style docstrings with `Args` and `Returns` sections.
- **README.md** — update if the change affects setup, usage, or
  the public API.
- **CLAUDE.md** — update the repository structure or quick reference
  if files were added/removed or commands changed.

Documentation updates use the `docs` commit type:
```
docs(readme): add usage example for new endpoint
```

## Version Bump (Automated)

Version bumping is handled automatically by `python-semantic-release`
on merge to main. It reads conventional commit messages to determine
the bump type (`feat:` → minor, `fix:` → patch, `feat!:` → major).

Do **not** manually bump the version in `pyproject.toml`.

## Verification Before Moving On

After completing a TDD cycle, ALWAYS run:
```bash
uv run pytest -v --durations=0 --cov
```

Before pushing, ALWAYS run the full quality suite:
```bash
uv run pre-commit run --all-files
```

## Test Naming Convention

```python
def test_<unit>_<scenario>_<expected_outcome>():
    """Test that <unit> <expected outcome> when <scenario>."""

# Examples:
def test_add_two_positive_integers_returns_sum():
def test_hello_with_empty_string_returns_hello():
def test_multiply_by_zero_returns_zero():
```

## Acceptance Criteria Format

When planning work, write acceptance criteria in Given/When/Then:

```
Given <precondition>
When <action>
Then <expected outcome>
```

These translate directly into test cases.
