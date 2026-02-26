---
name: issue
description: Scaffold a well-structured GitHub issue with acceptance criteria in Given/When/Then format. Use when defining new work.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "[short description of the work]"
---

# Scaffold a GitHub Issue

Create a structured issue that drives TDD implementation.

## Protocol

### Step 1: Understand the Request
From the user's description ($ARGUMENTS), identify:
- Which component/module is affected
- What behavior is being added or changed
- What the expected inputs and outputs are

### Step 2: Research the Codebase
Read relevant source files and tests to understand:
- Current implementation
- Existing test coverage
- Related functionality

### Step 3: Draft the Issue
Use this template:

```markdown
## [<component>] <Short imperative description>

### Context
<Why this matters — link to user story, bug report, or decision>

### Acceptance Criteria
- Given <precondition>, when <action>, then <expected outcome>
- Given <edge case>, when <action>, then <expected outcome>
- Given <failure mode>, when <action>, then <graceful degradation>

### Inputs / Outputs
- **Input:** <shape, types, constraints>
- **Output:** <return value, side effects, status codes>
- **Errors:** <known failure modes and expected behavior>

### Technical Notes
- <implementation hints, relevant files, dependencies>

### Definition of Done
- [ ] Unit tests pass (TDD — written first)
- [ ] Integration test added (if applicable)
- [ ] Documentation updated
- [ ] No linting/formatting violations
- [ ] Type checking passes
```

### Step 4: Create the Issue
If `gh` CLI is available:
```bash
gh issue create --title "<title>" --body "<body>"
```

Otherwise, present the formatted issue to the user for manual creation.

### Step 5: Report
Share the issue URL (if created) and suggest a branch name:
`<type>/<ISSUE-ID>-<short-description>`
