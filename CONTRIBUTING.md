# Contributing

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone and install dependencies:
    ```bash
    git clone <repo-url>
    cd <repo>
    uv sync
    ```
3. Enable pre-commit hooks:
    ```bash
    uv run pre-commit install
    ```
4. Run the tests to verify your setup:
    ```bash
    uv run pytest -v --cov
    ```

## Development Workflow

This project follows TDD (Test-Driven Development):

1. Create a branch: `<type>/<issue-id>-<description>`
2. **RED** — write one failing test
3. **GREEN** — write the minimum code to pass
4. **REFACTOR** — clean up while tests stay green
5. **COMMIT** — `<type>(<scope>): <description>`
6. Repeat steps 2-5 until done
7. Run quality checks before pushing:
    ```bash
    uv run pre-commit run --all-files
    ```
8. Open a PR (target < 400 lines changed)

See `.claude/rules/tdd-workflow.md` for the full protocol.

## Quality Checks

```bash
uv run pytest -v --cov               # Tests + coverage
uv run ruff check . --fix            # Lint (with auto-fix)
uv run ruff format .                 # Format
uv run pyright                       # Type check
uv run pre-commit run --all-files    # All checks at once
```

## Documentation

Preview documentation locally while developing:

```bash
uv run --group docs mkdocs serve     # Preview at http://127.0.0.1:8000
```

Docs are auto-deployed to GitHub Pages on merge to main.

## Keeping Tool Versions in Sync

Ruff and Pyright versions are pinned in two places:

- `pyproject.toml` `[dependency-groups]` — used by `uv run` and CI
- `.pre-commit-config.yaml` `rev` — used by pre-commit hooks

When updating either tool, update **both** files to keep versions
in sync. Run `uv run pre-commit autoupdate` to bump pre-commit
hook revisions, then verify the versions match `pyproject.toml`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Type       | When                                 |
|------------|--------------------------------------|
| `feat`     | New feature or behavior              |
| `fix`      | Bug fix                              |
| `test`     | Adding or correcting tests           |
| `refactor` | Code change with no behavior change  |
| `docs`     | Documentation only                   |
| `chore`    | Build, tooling, dependency updates   |

## Code Style

- 79-character line limit
- 4-space indentation
- Single quotes
- Google-style docstrings
- Type hints on all functions

See `.claude/rules/code-style.md` for full details.
