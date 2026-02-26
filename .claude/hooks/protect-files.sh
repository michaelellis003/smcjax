#!/bin/bash
# Hook: Prevent edits to protected files (lock files, CI configs, etc.)
# Used by Claude Code PreToolUse hook on Edit|Write.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

BASENAME=$(basename "$FILE_PATH")

# Block .env files (exact match or .env.* like .env.local)
if [[ "$BASENAME" == ".env" || "$BASENAME" == .env.* ]]; then
    echo "BLOCKED: $FILE_PATH is a protected file ('.env'). Do not edit directly." >&2
    exit 2
fi

# Block uv.lock
if [[ "$BASENAME" == "uv.lock" ]]; then
    echo "BLOCKED: $FILE_PATH is a protected file ('uv.lock'). Do not edit directly." >&2
    exit 2
fi

# Block .git/ directory
if [[ "$FILE_PATH" == *"/.git/"* || "$FILE_PATH" == .git/* ]]; then
    echo "BLOCKED: $FILE_PATH is a protected file ('.git/'). Do not edit directly." >&2
    exit 2
fi

exit 0
