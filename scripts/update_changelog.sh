#!/usr/bin/env bash
# update_changelog.sh — Generate a versioned CHANGELOG.md entry from conventional commits.
#
# Usage:
#   ./scripts/update_changelog.sh <tag> [<prev_tag>]
#
# If <prev_tag> is omitted, the script uses the most recent tag before <tag>.
# If no previous tag exists, all commits up to <tag> are included.
#
# The script prepends a new section to CHANGELOG.md (creating the file if needed).

set -euo pipefail

CURRENT_TAG="${1:-}"
if [ -z "$CURRENT_TAG" ]; then
  echo "Usage: $0 <tag> [<prev_tag>]" >&2
  exit 1
fi

# Determine previous tag
if [ -n "${2:-}" ]; then
  PREV_TAG="$2"
else
  PREV_TAG=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | grep -v "^${CURRENT_TAG}$" | head -1 || true)
fi

# Determine commit range
if [ -n "$PREV_TAG" ] && git rev-parse -q --verify "$PREV_TAG" > /dev/null 2>&1; then
  LOG_RANGE="${PREV_TAG}..${CURRENT_TAG}"
else
  LOG_RANGE="${CURRENT_TAG}"
fi

echo "Building changelog for ${CURRENT_TAG} (range: ${LOG_RANGE})" >&2

# Collect commits by conventional-commit type
BREAKING=""
FEATURES=""
FIXES=""
CHORES=""
OTHER=""

while IFS= read -r line; do
  HASH=$(echo "$line" | awk '{print $1}')
  MSG=$(echo "$line" | cut -d' ' -f2-)
  SHORT_HASH="${HASH:0:7}"

  if echo "$MSG" | grep -qE "(BREAKING CHANGE|^[a-z]+(\(.+\))?!:)"; then
    BREAKING="${BREAKING}- ${MSG} (\`${SHORT_HASH}\`)\n"
  elif echo "$MSG" | grep -qE "^feat(\(.+\))?:"; then
    FEATURES="${FEATURES}- ${MSG} (\`${SHORT_HASH}\`)\n"
  elif echo "$MSG" | grep -qE "^fix(\(.+\))?:"; then
    FIXES="${FIXES}- ${MSG} (\`${SHORT_HASH}\`)\n"
  elif echo "$MSG" | grep -qE "^(chore|docs|test|refactor|ci|style|perf)(\(.+\))?:"; then
    CHORES="${CHORES}- ${MSG} (\`${SHORT_HASH}\`)\n"
  else
    OTHER="${OTHER}- ${MSG} (\`${SHORT_HASH}\`)\n"
  fi
done < <(git log "$LOG_RANGE" --pretty="%H %s" 2>/dev/null)

# Skip if nothing to record
if [ -z "$BREAKING$FEATURES$FIXES$CHORES$OTHER" ]; then
  echo "No commits found in range ${LOG_RANGE}; skipping changelog update." >&2
  exit 0
fi

# Build the new versioned section
DATE=$(date +%Y-%m-%d)
ENTRY="## [${CURRENT_TAG}] - ${DATE}\n\n"

if [ -n "$BREAKING" ]; then
  ENTRY="${ENTRY}### Breaking Changes\n\n${BREAKING}\n"
fi
if [ -n "$FEATURES" ]; then
  ENTRY="${ENTRY}### Features\n\n${FEATURES}\n"
fi
if [ -n "$FIXES" ]; then
  ENTRY="${ENTRY}### Bug Fixes\n\n${FIXES}\n"
fi
if [ -n "$CHORES" ]; then
  ENTRY="${ENTRY}### Chores\n\n${CHORES}\n"
fi
if [ -n "$OTHER" ]; then
  ENTRY="${ENTRY}### Other\n\n${OTHER}\n"
fi

CHANGELOG="CHANGELOG.md"

if [ -f "$CHANGELOG" ]; then
  # Prepend after the existing header block (lines 1-4)
  HEADER=$(head -4 "$CHANGELOG")
  BODY=$(tail -n +5 "$CHANGELOG")
  printf "%s\n\n%b\n%s" "$HEADER" "$ENTRY" "$BODY" > "$CHANGELOG"
else
  printf "# Changelog\n\nAll notable changes to this project will be documented in this file.\nThis project adheres to [Semantic Versioning](https://semver.org/).\n\n%b" "$ENTRY" > "$CHANGELOG"
fi

echo "CHANGELOG.md updated for ${CURRENT_TAG}." >&2
