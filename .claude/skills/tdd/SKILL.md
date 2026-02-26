---
name: tdd
description: Run a TDD Red-Green-Refactor cycle for a feature or fix. Use when implementing new functionality or fixing bugs.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
argument-hint: "[description of the behavior to implement]"
---

# TDD Cycle

You are implementing a feature using strict Test-Driven Development.

**Behavior to implement:** $ARGUMENTS

## Protocol

Follow this exact sequence:

### Step 1: Understand
- Read the relevant source files and existing tests
- Identify the acceptance criteria from the description
- Plan the test cases: happy path first, then edge cases

### Step 2: RED — Write a Failing Test
- Write ONE test that describes the desired behavior
- Use the naming convention: `test_<unit>_<scenario>_<expected_outcome>`
- Include a Google-style docstring
- Run the test and confirm it FAILS:
  ```bash
  uv run pytest tests/ -x --tb=short -v
  ```
- If it passes, the behavior already exists — stop and inform the user

### Step 3: GREEN — Make It Pass
- Write the MINIMUM code to make the test pass
- Do not optimize or refactor yet
- Run the test and confirm it PASSES:
  ```bash
  uv run pytest tests/ -x --tb=short -v
  ```

### Step 4: REFACTOR — Clean Up
- Improve code quality (naming, duplication, structure)
- Ensure tests still pass after refactoring
- Run the full test suite:
  ```bash
  uv run pytest -v --durations=0 --cov
  ```

### Step 5: Verify Quality
- Run lint and format checks:
  ```bash
  uv run ruff check . --fix && uv run ruff format .
  ```
- Run type checking:
  ```bash
  uv run pyright
  ```

### Step 6: Report
Tell the user:
- What test was written (file and function name)
- What code was added/changed
- Current test count and pass/fail status
- Whether more TDD cycles are needed for remaining behaviors

If there are more behaviors to implement, ask if the user wants to
continue to the next RED-GREEN-REFACTOR cycle.
