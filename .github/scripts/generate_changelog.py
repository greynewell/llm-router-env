#!/usr/bin/env python3
"""Generate a changelog section from conventional commits between two git tags.

Environment variables:
    TAG      (required) The release tag being documented (e.g. v1.2.3)
    PREV_TAG (optional) The previous tag to diff against; defaults to full history

Output: updates CHANGELOG.md in the current working directory.
"""

import os
import re
import subprocess
import sys
from datetime import date


# Commit types that map to Keep-a-Changelog sections
_ADDED = {"feat"}
_FIXED = {"fix"}
_CHANGED = {"refactor", "chore", "docs", "test", "perf", "style", "build", "ci"}


def _run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def get_commits(log_range: str) -> list[dict]:
    """Return a list of dicts with 'subject' and 'is_breaking' for each commit."""
    commits = []
    for line in _run(["git", "log", log_range, "--pretty=%H %s"]).splitlines():
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        hash_, subject = parts[0], parts[1]

        full_body = _run(["git", "log", "--format=%B", "-n", "1", hash_])
        is_breaking = bool(re.match(r"^[a-z]+(\(.+\))?!:", subject)) or "BREAKING CHANGE" in full_body

        commits.append({"subject": subject, "is_breaking": is_breaking})
    return commits


def categorize(commits: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    breaking: list[str] = []
    added: list[str] = []
    fixed: list[str] = []
    changed: list[str] = []

    for c in commits:
        if "[skip ci]" in c["subject"]:
            continue
        m = re.match(r"^([a-z]+)(\(([^)]+)\))?!?:\s*(.*)", c["subject"])
        if not m:
            continue
        type_, scope, desc = m.group(1), m.group(3) or "", m.group(4)

        entry = f"- {desc} ({scope})" if scope else f"- {desc}"
        if c["is_breaking"]:
            breaking.append(f"- **BREAKING** {desc}{f' ({scope})' if scope else ''}")
            continue
        if type_ in _ADDED:
            added.append(entry)
        elif type_ in _FIXED:
            fixed.append(entry)
        elif type_ in _CHANGED:
            changed.append(entry)

    return breaking, added, fixed, changed


def build_section(tag: str, breaking: list[str], added: list[str], fixed: list[str], changed: list[str]) -> str:
    lines = [f"## [{tag}] - {date.today().isoformat()}", ""]
    if breaking:
        lines += ["### ⚠ Breaking Changes", ""] + breaking + [""]
    if added:
        lines += ["### Added", ""] + added + [""]
    if fixed:
        lines += ["### Fixed", ""] + fixed + [""]
    if changed:
        lines += ["### Changed", ""] + changed + [""]
    return "\n".join(lines)


_HEADER = (
    "# Changelog\n\n"
    "All notable changes to this project will be documented in this file.\n\n"
    "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),\n"
    "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
)


def update_changelog(tag: str, new_section: str) -> None:
    if os.path.exists("CHANGELOG.md"):
        with open("CHANGELOG.md") as f:
            existing = f.read()
        m = re.search(r"\n## ", existing)
        if m:
            content = existing[: m.start()] + "\n" + new_section + existing[m.start() :]
        else:
            content = existing.rstrip() + "\n\n" + new_section + "\n"
    else:
        content = _HEADER + new_section + "\n"

    with open("CHANGELOG.md", "w") as f:
        f.write(content)
    print(f"CHANGELOG.md updated for {tag}")


def main() -> None:
    tag = os.environ.get("TAG", "").strip()
    if not tag:
        print("ERROR: TAG environment variable is required", file=sys.stderr)
        sys.exit(1)

    prev_tag = os.environ.get("PREV_TAG", "").strip()
    log_range = f"{prev_tag}..{tag}" if prev_tag else tag
    print(f"Generating changelog: {log_range}")

    commits = get_commits(log_range)
    breaking, added, fixed, changed = categorize(commits)
    new_section = build_section(tag, breaking, added, fixed, changed)
    update_changelog(tag, new_section)


if __name__ == "__main__":
    main()
