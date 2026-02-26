# Smcjax

[![CI](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/Python-3.10|3.11|3.12|3.13-blue)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/Pyright-enabled-brightgreen)](https://github.com/microsoft/pyright)
[![License](https://img.shields.io/github/license/michaelellis003/smcjax)](https://github.com/michaelellis003/smcjax/blob/main/LICENSE)

Sequential Monte Carlo and particle filtering in JAX

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

```bash
git clone https://github.com/michaelellis003/smcjax.git
cd smcjax
uv sync
```

### Running Tests

```bash
uv run pytest -v --cov
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## Write your code

The demo functions (`hello`, `add`, `subtract`, `multiply`) in `smcjax/main.py` show the TDD workflow in action. Replace them with your own code, update `__init__.py` exports, and rewrite the tests in `tests/test_main.py`.

### Adding dependencies

```bash
uv add requests                  # Runtime dependency
uv add --dev pytest-mock         # Dev dependency
uv add --group docs sphinx       # Named group (e.g. docs)
```

### Running tests

```bash
uv run pytest -v --durations=0 --cov
```

### Linting and formatting

Run everything at once (same checks as CI):

```bash
uv run pre-commit run --all-files
```

Or individually:

```bash
uv run ruff check . --fix        # Lint with auto-fix
uv run ruff format .             # Format
uv run pyright                   # Type check
```

### Docs

```bash
uv run --group docs mkdocs serve              # Preview at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict     # Build static site
```

Docs deploy to [GitHub Pages](https://michaelellis003.github.io/smcjax/) automatically on push to `main`. Enable it under **Settings > Pages > Source: GitHub Actions**.

## Claude Code

The template includes a full [Claude Code](https://docs.anthropic.com/en/docs/claude-code) configuration in `.claude/` so the AI assistant understands the project's conventions -- TDD workflow, code style, commit format, and more -- out of the box.

### Slash commands

| Command | What it does |
|---------|--------------|
| `/tdd` | Run a Red-Green-Refactor cycle for a feature or fix |
| `/commit` | Create a conventional commit with quality gates |
| `/pr` | Open a pull request with structured summary and test plan |
| `/branch` | Create a feature branch from the latest main |
| `/lint` | Run all quality checks (format, lint, type check, tests) |
| `/issue` | Scaffold a GitHub issue with Given/When/Then criteria |

### What's configured

- **Rules** -- TDD workflow, code style (79-char lines, Google docstrings), design principles (KISS, YAGNI, SOLID), error handling (EAFP, guard clauses), git conventions, Python idioms, and testing standards.
- **Hooks** -- auto-format and lint Python files after edits, pre-push quality gate before `git push`, file protection for `.env` and `uv.lock`, and a test summary on session stop.
- **Agents** -- a code-reviewer for PR-style feedback and a test-writer that follows TDD principles.
- **CLAUDE.md** -- the central instruction file that ties it all together.

## Push and review

### What CI checks

On every PR and push to `main`, CI runs these checks in parallel: Ruff lint, Ruff format, Pyright, pytest across Python 3.10--3.13, coverage enforcement, lockfile sync, macOS and Windows smoke tests, and build validation (sdist + wheel + twine check + install test). All must pass before merging.

### Conventional commits

Commit messages drive automatic versioning. Use the format `<type>(<scope>): <description>`:

- `feat:` -- minor version bump (0.3.0 -> 0.4.0)
- `fix:` / `perf:` -- patch bump (0.3.0 -> 0.3.1)
- `feat!:` or `BREAKING CHANGE` footer -- major bump (0.3.0 -> 1.0.0)
- `test:`, `refactor:`, `docs:`, `chore:`, `ci:` -- no version bump

### Branch naming

```
<type>/<issue-id>-<short-description>
```

For example: `feat/AUTH-42-jwt-refresh-rotation`, `fix/API-118-null-pointer`.

## Ship it

### Automatic releases

When you merge to `main`, [python-semantic-release](https://python-semantic-release.readthedocs.io/) reads your commit messages, bumps the version in `pyproject.toml`, creates a git tag, and publishes a GitHub Release with built artifacts. No manual steps needed.

### PyPI publishing

To publish to PyPI automatically on each release:

1. Uncomment the `PYPI-START`/`PYPI-END` block in `release.yml`.
2. Add a [trusted publisher](https://docs.pypi.org/trusted-publishers/) on pypi.org for your repo (workflow: `release.yml`).
3. Every merge to `main` with a `feat:` or `fix:` commit will auto-publish.

A manual **test-publish.yml** workflow is included for validating your pipeline against [TestPyPI](https://test.pypi.org) first. A conda-forge recipe skeleton lives in `recipe/meta.yaml`, ready to submit to [staged-recipes](https://github.com/conda-forge/staged-recipes) once your package is on PyPI. See the full [Publishing Guide](https://michaelellis003.github.io/smcjax/publishing/) for details.

### RELEASE_TOKEN

The release workflow works out of the box with the default `GITHUB_TOKEN`. However, commits made by `github-actions[bot]` don't trigger downstream workflows (like docs deploy). To fix that, create a fine-grained PAT with **Contents: read/write** and add it as a repo secret named `RELEASE_TOKEN`.

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

```bash
git clone https://github.com/michaelellis003/smcjax.git
cd smcjax
uv sync
```

### Running Tests

```bash
uv run pytest -v --cov
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

<details>
<summary>Project structure</summary>

```
smcjax/         # Package source
  __init__.py                    # Public API exports + __version__
  __main__.py                    # python -m entry point
  main.py                       # Core module with demo functions
  py.typed                       # PEP 561 type checking marker
tests/
  conftest.py                    # Shared test fixtures
  test_init.py                   # Package-level tests
  test_main.py                   # Unit tests for demo functions
  test_main_module.py            # Tests for __main__.py entry point
    test_init_flags.py           # Tests for flag validation and special chars
docs/
  index.md                       # Documentation landing page
  api.md                         # Auto-generated API reference
  publishing.md                  # PyPI, TestPyPI, and conda-forge guide
.github/
  actions/setup-uv/              # Reusable CI composite action
  workflows/
    ci.yml                       # Lint, format, typecheck, test matrix
    release.yml                  # Auto-version + GitHub Release
    docs.yml                     # Build and deploy to GitHub Pages
    test-publish.yml             # Manual TestPyPI publishing
    dependabot-auto-merge.yml    # Auto-merge minor/patch Dependabot PRs
  src/pypkgkit/                  # CLI source code
  tests/                         # CLI tests
scripts/
  setup-repo.sh                  # Branch protection setup
.pre-commit-config.yaml          # Pre-commit hook definitions
pyproject.toml                   # Project config, deps, tool settings
mkdocs.yml                       # MkDocs documentation config
```

</details>
