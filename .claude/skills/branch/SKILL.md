---
name: branch
description: Create a feature branch from the latest main following the naming convention. Use when starting new work.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "<type>/<issue-id>-<short-description>"
---

# Create a Feature Branch

Create a properly named branch from the latest main for new work.

## Protocol

### Step 1: Validate Branch Name
If $ARGUMENTS is provided, validate it matches the convention:
```
<type>/<issue-id>-<short-description>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

If no argument is provided, ask the user for:
- The type of work (feat, fix, refactor, etc.)
- The issue ID (or a short identifier)
- A brief description

### Step 2: Check Working Tree
```bash
git status
```
If there are uncommitted changes, warn the user and suggest:
- Committing the changes first (`/commit`)
- Stashing them (`git stash`)

Do NOT proceed with a dirty working tree unless the user confirms.

### Step 3: Update Main
```bash
git checkout main
git pull origin main
```

### Step 4: Create Branch
```bash
git checkout -b <type>/<issue-id>-<short-description>
```

### Step 5: Verify
```bash
git branch --show-current
git log --oneline -1
```

### Step 6: Report
Tell the user:
- The branch name created
- That they are ready to start a TDD cycle (`/tdd`)
