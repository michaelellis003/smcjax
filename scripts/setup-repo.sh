#!/usr/bin/env bash
# Setup script for repos created from smcjax.
#
# Configures a repository ruleset on main with required CI status checks
# and admin bypass (needed for automated semantic-release commits).
# GitHub does not copy rulesets from template repos, so this must be run
# once after creating a new repo.
#
# Prerequisites:
#   - gh CLI installed and authenticated (https://cli.github.com)
#   - jq installed (https://jqlang.github.io/jq/)
#   - Admin access to the target repository
#
# Usage:
#   ./scripts/setup-repo.sh
#   ./scripts/setup-repo.sh --require-reviews 1

set -euo pipefail

# --- Configuration ---------------------------------------------------------
# Number of required PR approvals before merging.
# Set to 0 to skip PR review requirements (default, suitable for solo devs).
# Use --require-reviews N to override.
REQUIRED_REVIEWS=0
RULESET_NAME="main branch protection"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-reviews)
            REQUIRED_REVIEWS="${2:-1}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./scripts/setup-repo.sh [--require-reviews N]"
            echo ""
            echo "Options:"
            echo "  --require-reviews N  Require N PR approvals (default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# Validate --require-reviews is an integer 0-10
if ! [[ "$REQUIRED_REVIEWS" =~ ^[0-9]+$ ]] || \
   [[ "$REQUIRED_REVIEWS" -gt 10 ]]; then
    echo "Error: --require-reviews must be an integer 0-10." >&2
    exit 1
fi

# Detect repo from git remote
REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null) || {
    echo "Error: could not detect repository." >&2
    echo "Run this from inside a GitHub-connected git repo." >&2
    exit 1
}

echo "Configuring repository ruleset for ${REPO} (main)..."

# Build the ruleset payload using jq for safe JSON construction
PAYLOAD=$(jq -n \
    --arg name "$RULESET_NAME" \
    --argjson reviews "$REQUIRED_REVIEWS" \
    '{
        name: $name,
        target: "branch",
        enforcement: "active",
        conditions: {
            ref_name: {
                include: ["refs/heads/main"],
                exclude: []
            }
        },
        bypass_actors: [
            {
                actor_id: 5,
                actor_type: "RepositoryRole",
                bypass_mode: "always"
            }
        ],
        rules: [
            {
                type: "pull_request",
                parameters: {
                    required_approving_review_count: $reviews,
                    dismiss_stale_reviews_on_push: false,
                    require_code_owner_review: true,
                    require_last_push_approval: false,
                    required_review_thread_resolution: false
                }
            },
            {
                type: "required_status_checks",
                parameters: {
                    strict_required_status_checks_policy: true,
                    required_status_checks: [
                        { context: "ci-pass" }
                    ]
                }
            },
            { type: "non_fast_forward" },
            { type: "deletion" }
        ]
    }')

# Check if a ruleset with this name already exists
EXISTING_ID=$(
    gh api "repos/${REPO}/rulesets" 2>/dev/null \
        | jq -r --arg name "$RULESET_NAME" \
            '.[] | select(.name == $name) | .id'
) || EXISTING_ID=""

if [[ -n "$EXISTING_ID" ]]; then
    # Update existing ruleset
    echo "$PAYLOAD" | gh api "repos/${REPO}/rulesets/${EXISTING_ID}" \
        --method PUT --input - > /dev/null
    echo "Done. Repository ruleset updated on main."
else
    # Create new ruleset
    echo "$PAYLOAD" | gh api "repos/${REPO}/rulesets" \
        --method POST --input - > /dev/null
    echo "Done. Repository ruleset enabled on main."
fi

echo ""
echo "Required status check:"
echo "  - ci-pass (gate job that requires all CI jobs to succeed)"
echo ""
echo "Bypass: repository admins (for automated releases)"
if [[ "$REQUIRED_REVIEWS" -gt 0 ]]; then
    echo ""
    echo "Required PR approvals: $REQUIRED_REVIEWS"
else
    echo ""
    echo "PR reviews: not required (solo developer mode)"
    echo "  To require reviews: ./scripts/setup-repo.sh --require-reviews 1"
fi
echo ""
echo "Note: ci-pass depends on all CI jobs, so adding or removing jobs"
echo "in ci.yml only requires updating the ci-pass 'needs' list."
