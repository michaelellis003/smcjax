---
name: pr
description: Open a pull request with structured summary, test plan, and conventional title. Use after pushing a feature branch.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "[optional base branch, defaults to main]"
---

# Open a Pull Request

Create a well-structured pull request for the current branch.

## Protocol

### Step 1: Gather Context
```bash
git branch --show-current
git log main..HEAD --oneline
git diff main...HEAD --stat
git diff main...HEAD
```

Identify:
- The branch type from the name (feat/, fix/, etc.)
- All commits since divergence from main
- Total lines changed
- Files affected

### Step 2: Verify Quality
Ensure all checks pass before opening the PR:
```bash
uv run pytest -v --durations=0 --cov
uv run pre-commit run --all-files
```

### Step 3: Check PR Size
- < 200 lines: ideal
- 200-400 lines: acceptable
- 400-800 lines: warn the user, suggest splitting
- > 800 lines: strongly recommend splitting

### Step 4: Rebase on Main
Rebase onto the latest main to ensure a clean merge:
```bash
git fetch origin
git rebase origin/main
```
If there are conflicts, resolve them and continue the rebase.
Do NOT use `--force` when pushing after rebase — ask the user first.

### Step 5: Push if Needed
```bash
git push -u origin $(git branch --show-current)
```

### Step 6: Create PR
Use `gh pr create` with this structure:

Title: Short imperative description (< 70 chars), matching the
conventional commit type.

Body:
```markdown
## Summary
<1-3 bullet points describing what and why>

## Related Issue
Closes #<issue-number> (if applicable)

## Changes
- <list of specific changes>

## Test Plan
- [ ] Unit tests pass locally
- [ ] Lint and format checks pass
- [ ] Type checking passes
- [ ] <any manual testing steps>
```

### Step 7: Report
Share the PR URL with the user.
