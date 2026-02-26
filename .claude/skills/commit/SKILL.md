---
name: commit
description: Create a conventional commit with proper message format. Runs tests and lint before committing.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "[optional commit message override]"
---

# Conventional Commit

Create a well-formed conventional commit for the current staged or
unstaged changes.

## Protocol

### Step 1: Assess Changes
Run these commands to understand the current state:
```bash
git status
git diff --stat
git diff
git log --oneline -5
```

### Step 2: Verify Quality
Before committing, run the quality suite:
```bash
uv run pytest -x --tb=short -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
If any check fails, fix the issues first. Do NOT commit broken code.

### Step 3: Stage Files
Stage files intentionally — use specific file paths, never `git add .`:
```bash
git add <specific-files>
```

Do NOT stage:
- `.env` or credential files
- `uv.lock` (unless dependency changes are the point)
- Unrelated changes

### Step 4: Write Commit Message
Use Conventional Commits format:
```
<type>(<scope>): <short imperative description>

[optional body explaining why, not what]

[optional footer: Closes #123]
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `perf`, `ci`

If the user provided a message via $ARGUMENTS, use that as the basis
but ensure it follows conventional commit format.

### Step 5: Commit
Include a `Co-Authored-By` trailer when the commit was AI-assisted:
```bash
git commit -m "<message>

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Step 6: Verify
```bash
git log --oneline -1
git status
```

Report the commit hash and summary to the user.
