# CLAUDE.md

This file provides guidance for AI assistants working with this repository.
Modular rules live in `.claude/rules/` — each covers a specific concern.

## Project Overview

**smcjax** — Sequential Monte Carlo and particle filtering in JAX. Uses uv, Ruff, Pyright, and pre-commit
hooks. Licensed Apache-2.0.

## Repository Structure

```
smcjax/         # Main package source
  __init__.py                    # Public API exports + __version__
  __main__.py                    # python -m entry point
  main.py                       # Core module with demo functions
  py.typed                      # PEP 561 type checking marker
tests/
  conftest.py                  # Shared test fixtures
  test_init.py                  # Package-level tests
  test_main.py                  # Unit tests for demo functions
  test_main_module.py           # Tests for __main__.py entry point
    test_init_flags.py          # Tests for flag validation and special chars
docs/
  index.md                     # Documentation landing page
  api.md                       # Auto-generated API reference
  publishing.md                # PyPI, TestPyPI, and conda-forge guide
.github/
  actions/setup-uv/            # Reusable CI composite action
  CODEOWNERS                   # Default code ownership for reviews
  dependabot.yml               # Automated dependency updates
  ISSUE_TEMPLATE/              # Bug report & feature request forms
  PULL_REQUEST_TEMPLATE.md     # PR checklist template
  workflows/
    ci.yml                      # CI: parallel lint, format, typecheck, test matrix
    dependabot-auto-merge.yml   # Auto-merge minor/patch Dependabot PRs
    docs.yml                    # Docs: build and deploy to GitHub Pages
    release.yml                 # Release: gated on CI, auto-version + PyPI publish
    test-publish.yml            # Manual TestPyPI publishing
.claude/
  settings.json                 # Claude Code project settings and hooks
  rules/                        # Modular instructions by topic
    tdd-workflow.md             # TDD Red-Green-Refactor lifecycle
    code-style.md               # Formatting, linting, docstrings
    design-principles.md        # KISS, YAGNI, SOLID, composition, SoC, least astonishment
    error-handling.md           # EAFP, custom exceptions, fail fast, guard clauses
    git-conventions.md          # Branching, commits, PR workflow
    python-idioms.md            # Dataclasses, pathlib, generators, enums, functools
    testing.md                  # Test structure and conventions
  skills/                       # Custom slash commands
    branch/SKILL.md             # /branch — create a feature branch
    tdd/SKILL.md                # /tdd — TDD cycle for a feature
    commit/SKILL.md             # /commit — conventional commit
    pr/SKILL.md                 # /pr — open a pull request
    lint/SKILL.md               # /lint — run all quality checks
    issue/SKILL.md              # /issue — scaffold an issue
  agents/
    code-reviewer.md            # Code review subagent
    test-writer.md              # Test-first subagent
  hooks/                          # Tool-use hooks
    protect-files.sh              # Block edits to protected files
    pre-push-check.sh            # Lint + typecheck + test before push
  pyproject.toml                  # pypkgkit package config, zero deps
  src/pypkgkit/
    __init__.py                   # Version via importlib.metadata
    __main__.py                   # python -m pypkgkit support
    cli.py                        # argparse entry point + main()
    scaffold.py                   # Download, extract, invoke logic
    py.typed                      # PEP 561 marker
  tests/
    conftest.py                   # Mock fixtures (release JSON, tarball)
    test_cli.py                   # Arg parsing + dispatch tests
    test_scaffold.py              # Download, extract, invoke tests
  README.md                       # PyPI readme for pypkgkit
recipe/
  meta.yaml                      # conda-forge recipe skeleton
scripts/
  setup-repo.sh                  # One-time repo setup (branch protection)
.editorconfig                   # Editor settings for non-Python files
.gitignore                      # Git ignore rules
.pre-commit-config.yaml         # Pre-commit hook definitions
mkdocs.yml                     # MkDocs documentation config
pyproject.toml                  # Project config, deps, tool settings
uv.lock                        # Locked dependency versions
README.md                      # User documentation
CHANGELOG.md                   # Release history
CONTRIBUTING.md                # Contribution guidelines
SECURITY.md                    # Security policy
LICENSE                        # Apache-2.0 license
```

## Quick Reference — Development Commands

```bash
uv sync                                      # Install deps
uv run pytest -v --durations=0 --cov         # Run tests
uv run pre-commit run --all-files            # Lint + format + type check
uv run ruff check . --fix                    # Ruff linter only
uv run ruff format .                         # Ruff formatter only
uv run pyright                               # Type checker only
uv run --group docs mkdocs serve             # Local docs preview
uv run --group docs mkdocs build --strict    # Build docs
```

## Development Lifecycle (TDD-First)

This project follows a strict TDD-first workflow. See
`.claude/rules/tdd-workflow.md` for the full Red-Green-Refactor protocol.

**The loop:**
1. Define work — `/issue` (Given/When/Then acceptance criteria)
2. Branch — `/branch <type>/<issue-id>-<short-description>`
3. RED — write one failing test
4. GREEN — minimal code to pass
5. REFACTOR — clean up, tests still pass
6. COMMIT — `<type>(<scope>): <description>` after each cycle
7. Repeat 3-6 until acceptance criteria are met
8. DOCS — update documentation to reflect user-facing changes
9. PUSH — run lint + tests before pushing
10. PR — self-review, open PR, request review
11. CI — parallel: ruff lint, ruff format, pyright, test matrix
12. MERGE — squash and merge to main
13. RELEASE — `python-semantic-release` auto-bumps version, tags,
   creates a GitHub Release, and publishes to PyPI (if enabled)

## Key Rules

- **Design principles** — KISS, YAGNI, SOLID, composition over
  inheritance, separation of concerns, least astonishment
- **Error handling** — EAFP over LBYL, specific exceptions, fail
  fast, guard clauses, chain with `from`
- **Python idioms** — dataclasses, enums, pathlib, generators,
  comprehensions, context managers, functools
- **Always write tests before implementation** (TDD)
- **Conventional Commits** — `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
- **79-char lines, 4-space indent, single quotes, Google docstrings**
- **All functions need type hints and docstrings**
- **Update docs** (README, docstrings, CLAUDE.md) when behavior changes
- **Version bumps are automated** — `python-semantic-release` reads
  conventional commits and bumps version on merge to main
- **Run `uv run pre-commit run --all-files` before every push**
- **PR target: < 400 lines changed**
