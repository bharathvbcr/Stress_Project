# StressProject Agent Instructions

Before making edits in this repository, use this repo map as the primary navigation source:

- `docs/repo-map.md` (full file index + ownership matrix + dependency anchors)

Agent policy:

- Start with `docs/repo-map.md` to determine which area owns the change.
- For broad questions, trace from caller to callee using the map first, then narrow to file-level references.
- Prefer absolute path references in responses and change notes.
- After structural moves/refactors, update `docs/repo-map.md` in the same pass.
- For module-specific guidance, check nested agent files (for example `timesfm/AGENTS.md`), but treat `docs/repo-map.md` as the top-level contract.
- Use the Owner Matrix in `docs/repo-map.md` before deciding the file-level scope.

Repository map conventions:

- `timesfm/` is the packaged TimesFM project and contains its own AGENTS guidance.
- Root-level python scripts drive preprocessing, training, benchmarking, and utility utilities for stress prediction and model pipelines.
- Do not treat `outputs/` and `scratch/` as source-of-truth unless a task explicitly targets artifacts or experiments.
- After updating instructions in any area, ensure the corresponding map section remains the single source of ownership truth.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **StressProject** (3672 symbols, 5687 relationships, 194 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/StressProject/context` | Codebase overview, check index freshness |
| `gitnexus://repo/StressProject/clusters` | All functional areas |
| `gitnexus://repo/StressProject/processes` | All execution flows |
| `gitnexus://repo/StressProject/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
