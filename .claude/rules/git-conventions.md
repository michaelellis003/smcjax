# Git Conventions

## Branch Naming

```
<type>/<issue-id>-<short-description>

# Examples:
feat/AUTH-42-jwt-refresh-rotation
fix/API-118-null-pointer-on-empty-cart
refactor/DB-77-normalize-user-table
test/CORE-55-add-payment-edge-cases
docs/README-update-setup-instructions
chore/CI-33-upgrade-node-to-22
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

Always branch from the latest `main`:
```bash
git checkout main
git pull origin main
git checkout -b <type>/<issue-id>-<short-description>
```

## Commit Messages (Conventional Commits)

Format:
```
<type>(<scope>): <short imperative description>

[optional body]

[optional footer(s)]
```

Examples:
```
feat(auth): add JWT refresh token rotation

Implements refresh token rotation with automatic revocation.
Tokens stored in Redis with configurable TTL.

Closes #42
```

```
fix(api): handle null cart in checkout endpoint

Returns 400 with clear error message instead of 500.

Fixes #118
```

```
test(payment): add edge cases for currency conversion

Covers zero-amount, negative-amount, and unsupported-currency.
```

### Commit Types

| Type       | When                                     |
|------------|------------------------------------------|
| `feat`     | New feature or behavior                  |
| `fix`      | Bug fix                                  |
| `test`     | Adding or correcting tests only          |
| `refactor` | Code change with no behavior change      |
| `docs`     | Documentation only                       |
| `chore`    | Build, tooling, dependency updates       |
| `perf`     | Performance improvement                  |
| `ci`       | CI/CD pipeline configuration changes     |
| `style`    | Formatting, whitespace (no logic change) |

Breaking changes: Add `!` after the type:
```
feat(api)!: change auth from API key to OAuth2

BREAKING CHANGE: X-API-Key header no longer accepted.
```

## Commit Frequency

- Commit after every Red-Green-Refactor cycle
- Each commit must compile and pass tests
- Stage intentionally — use specific files, not `git add .`

## Documentation Updates

Before pushing, review whether documentation needs updating:

- New public function/class → add Google-style docstring
- Changed behavior or API → update README usage section
- Added/removed files or commands → update CLAUDE.md structure
- Changed setup or dependencies → update README getting started

Use `docs(<scope>): <description>` commits for documentation-only
changes. If docs are part of a feature, include them in the
`feat` commit.

## Version Bump (Automated)

Version bumping is handled automatically by `python-semantic-release`
when a PR is merged to main. It reads conventional commit messages to
determine the bump type:

- `feat:` → minor bump (0.3.0 → 0.4.0)
- `fix:` / `perf:` → patch bump (0.3.0 → 0.3.1)
- `feat!:` / `BREAKING CHANGE` → major bump (0.3.0 → 1.0.0)

Do **not** manually bump the version in `pyproject.toml`. The
`release.yml` workflow handles version bumps, tagging, and GitHub
Release creation.

## Push Frequency

- Push every 3-5 commits or at end of session
- Run full quality suite before pushing:
  `uv run pre-commit run --all-files`
- Rebase on main before pushing:
  `git fetch origin && git rebase origin/main`

## Pull Request Guidelines

- Target: **< 400 lines** changed
- Title: short, imperative, under 70 characters
- Body: Summary, Related Issue, Changes list, Test plan
- Self-review the diff before requesting review
- Squash and merge to main (one commit per PR)
