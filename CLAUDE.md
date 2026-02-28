# LLM Router Env — Claude Instructions

## Stack

- Python 3.11+, Gymnasium 1.0+, Stable-Baselines3
- Install: `pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `pytest -x`

## Architecture

Single Gymnasium environment for RL-based LLM inference routing:
- **Action space:** `Discrete(n_models)` — index into available models
- **Observation space:** `Box` — prompt_length, prompt_complexity, queue_depths (per model), time_of_day, budget_remaining, quality_required
- **Reward:** `r = -cost_weight * cost + quality_weight * quality - latency_penalty * max(0, latency - sla_threshold) - quality_miss_penalty * max(0, quality_required - quality)`

Key modules:
- `llm_router_env/env.py` — core Gymnasium environment
- `llm_router_env/models.py` — model configs (cost, latency distributions, quality)
- `llm_router_env/traffic.py` — prompt traffic generator
- `llm_router_env/reward.py` — configurable reward function

## Development

Before committing:
```
ruff check .
pytest -x
find .github/workflows -name '*.yml' -exec python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" {} \;
```

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature (bumps MINOR)
- `fix:` — bug fix (bumps PATCH)
- `feat!:` or `BREAKING CHANGE` — breaking change (bumps MAJOR)
- `chore:`, `docs:`, `test:`, `refactor:` — no version bump

## Branch Naming

`claude/issue-{number}-{YYYYMMDD}-{HHMM}`

## Pull Requests

Always create PRs using `gh pr create`. Never substitute a compare link.

```
gh pr create \
  --repo OWNER/REPO \
  --title "..." \
  --body "..." \
  --base main \
  --head <branch>
```

## Issues

When creating a GitHub issue, always include `@claude` at the end of the body:

```
@claude please implement this
```

## Self-Improvement Loop

> **Note:** The loop is supervised, not fully autonomous. A human must approve each PR before it can be merged.

1. `claude-proactive.yml` (hourly) and `claude-self-improve.yml` (weekly) scan for issues
2. Issues trigger `claude-auto-assign.yml` → `claude.yml` implements them
3. `claude-code-review.yml` reviews the PR
4. `claude-pr-shepherd.yml` merges when CI passes **and at least one human has approved** (bot approvals are excluded)
5. `auto-tag.yml` tags the release
6. Repeat
