---
description: "Plan and execute a structured refactor of a module or component following the project meta-spec"
agent: "agent"
argument-hint: "Target module or component to refactor (e.g., 'reward system in CustomEnv')"
---

You are refactoring a component of the tsp_rl_kg project. Follow the plan-first approach from [docs/meta-spec.md](../../docs/meta-spec.md).

## Process

1. **Explore** the target module in read-only mode. Identify all callers, dependencies, and tests.
2. **Generate a spec** in `docs/specs/` with:
   - Goals & Non-Goals
   - Target Files (restrict scope)
   - Sequence of changes (numbered steps)
   - Rejection Criteria
3. **Present the spec** for approval before writing any code.
4. **Implement** one sequence step at a time.
5. **Validate** after each step: run `uv run ruff check src/` and `uv run black --check src/`. Run `uv run pytest tests/ -v` if tests exist for the changed module.
6. **Update the spec** if edge cases or requirement changes are discovered during implementation.

## Constraints

- Do NOT change public APIs without listing the change in the spec and getting approval.
- Do NOT add new dependencies without asking first.
- Do NOT refactor code outside the target files listed in the spec.
- Preserve all existing behavior unless the spec explicitly says otherwise.

## Code Style

- Black, line-length 100
- isort with Black profile
- Type hints on all new/modified public functions
- Follow naming conventions from [.github/copilot-instructions.md](../copilot-instructions.md)
