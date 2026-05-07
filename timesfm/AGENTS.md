# TimesFM — Agent Entry Point

Before touching code, follow the repo-wide map:

- `../docs/repo-map.md`
- `../AGENTS.md`
- `../CLAUDE.md`

The `../docs/repo-map.md` entry is the authoritative cross-area reference for:
- full-path discovery
- owner boundaries (`root` vs `timesfm`)
- high-risk coupling points before touching shared interfaces

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## TimesFM package boundaries

- `timesfm/src/timesfm/` contains the active package implementation (prefer edits here for model behavior).
- `timesfm/v1/` contains legacy and experimental content; avoid edits there unless explicitly requested.
- `timesfm/timesfm-forecasting/` contains agent skill packaging assets.
- `timesfm/.github/` and `timesfm/README.md` are packaging and contributor-facing references.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in `src/timesfm/`.
Archived v1/v2 code and notebooks are in `v1/`.

Run tests:

```bash
pytest v1/tests/
```

See `README.md` for full developer setup.
