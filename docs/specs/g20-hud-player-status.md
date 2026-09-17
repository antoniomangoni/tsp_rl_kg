# g20 - HUD Player Status & Route Scores

## Extended TOC

| Section | Summary | Tag |
|---|---|---|
| 1 | Goals & Non-Goals: Add inventory capacity and route score visibility without changing game mechanics | `goals` |
| 2 | Technical Decisions & Stack: Keep the status source in `GameManager`, keep `Renderer` presentational, use `uv` tooling | `decisions` |
| 3 | Project Structure & Git Workflow: Scope implementation to `src/`, tests to `tests/`, and keep commits focused | `workflow` |
| 4 | Testing & Conformance: Verify status data, route tracking, renderer compatibility, and style checks | `testing` |
| 5 | Sequence & Workstreams: Build the status model first, then render it, then verify route lifecycle behavior | `sequence` |
| 6 | Target Files: Concrete files expected to change during implementation | `files` |
| 7 | Commands & Code Style: Exact commands and local style rules | `commands` |
| 8 | Boundaries & Acceptance: Three-tier actions, rejection criteria, and self-audit checklist | `boundaries` |

---

## 1. Goals & Non-Goals `goals`

### Goals

- **G1 - Show resource inventory capacity in the HUD.** The player UI must show current wood and stone counts as `current/max`, using `Agent.resource_max` as the capacity source.
- **G2 - Show the game world's best route score.** The HUD must expose the world target route energy from `Target_Manager.target_route_energy`. This should be labeled clearly enough that it is not confused with the player's best completed route.
- **G3 - Show the current route score.** The HUD must show the energy spent on the current outpost route, calculated from total player energy minus the energy at the beginning of the current route attempt.
- **G4 - Track route completion during playable game loops.** When the player has visited all outposts, the completed route score must be recorded and the current route counter must reset for the next route attempt.
- **G5 - Keep rendering simple and readable.** Extend the existing pygame HUD with an additional row or equivalent compact layout. Avoid a larger UI redesign unless text overflow makes it necessary.

### Non-Goals

- Do not change terrain movement cost, route energy calculation, resource collection, path building, or outpost placement mechanics.
- Do not add a new UI framework or dependency.
- Do not change the RL observation pipeline, reward calculation, or training metrics as part of this work.
- Do not solve the separate "paths disappear after placement" issue in this workstream. That should have its own spec or test plan.
- Do not add a player-best-completed-route HUD field unless it is explicitly requested after this spec. This spec only requires world best and current route scores.

---

## 2. Technical Decisions & Stack `decisions`

### Architecture

`GameManager` owns the game-state semantics for the HUD. It should assemble a small status structure from:

- `self.agent.grid_x` and `self.agent.grid_y`
- `self.agent_controler.energy_spent`
- `self.agent_controler.wood`
- `self.agent_controler.stone`
- `self.agent_controler.resource_max`
- `self.target_manager.target_route_energy`
- `self.environment.outpost_locations`
- `self.visited_outposts`
- `self.route_start_energy`

`Renderer` owns only presentation. It should accept prepared status rows and draw them onto the HUD surface. It should not calculate route scores, inspect target managers, or mutate game state.

### HUD Data Shape

Use a row-based shape that keeps pygame rendering straightforward:

```python
list[dict[str, str | int | float]]
```

Recommended rows:

```python
[
    {
        "X": player_x,
        "Y": player_y,
        "Energy": total_energy,
        "Outposts": "visited/total",
    },
    {
        "Wood": "wood/resource_max",
        "Stone": "stone/resource_max",
        "World best": target_route_energy,
        "Current route": current_route_energy,
    },
]
```

Prefer the label `World best` or `Target route` over plain `Best route`, because the value is the game's algorithmic target route energy, not the player's best route history.

### Route Tracking Semantics

- `route_start_energy` stores `agent_controler.energy_spent` at the beginning of the current route attempt.
- `current_route_energy = agent_controler.energy_spent - route_start_energy`.
- `visited_outposts` stores outpost coordinates visited during the current route attempt.
- When `visited_outposts` includes every outpost:
  - append the completed current route energy to `route_energy_list`
  - clear `visited_outposts`
  - set `route_start_energy` to current total energy

### Stack

- Python: `>=3.14`, as declared in `pyproject.toml`.
- Dependency management: `uv`.
- Rendering: existing `pygame-ce` renderer.
- Tests: `pytest`.
- Lint/format: `ruff`, `black`, and existing project config in `pyproject.toml`.

---

## 3. Project Structure & Git Workflow `workflow`

### Project Structure

- Runtime implementation belongs under `src/tsp_rl_kg/`.
- Game-state orchestration belongs in `src/tsp_rl_kg/game_world/game_manager.py`.
- HUD drawing belongs in `src/tsp_rl_kg/renderer.py`.
- Tests belong in `tests/`, with focused coverage in `tests/test_game_manager.py` and optional renderer tests if a pure rendering seam exists.
- Specification documents belong in `docs/specs/`.

### Git Workflow

- Keep this work as a focused change set: HUD data, route tracking, renderer row support, and tests.
- Do not mix this implementation with the path persistence/rendering bug.
- Before committing, review `git diff --stat` and `git diff` to ensure only target files changed.
- Commit message recommendation: `Add HUD player status route metrics`.
- Open a PR only after tests and lint checks pass, or clearly document any blocked checks.

---

## 4. Testing & Conformance `testing`

### Automated Tests

Required:

- A `GameManager._build_status()` test that sets known wood and stone values and verifies:
  - `Wood == "{wood}/{resource_max}"`
  - `Stone == "{stone}/{resource_max}"`
  - world best route score equals `target_manager.target_route_energy`
  - current route score starts at `0`
- A route tracking test that simulates visiting all outposts and verifies:
  - completed route score is appended to `route_energy_list`
  - `visited_outposts` resets
  - `route_start_energy` resets to current total energy
- Existing game-manager tests must continue to pass.

Recommended:

- A renderer smoke test, if practical, that calls `render_ui()` with multiple rows under a dummy pygame display and verifies it does not raise.
- A label-focused test if the implementation chooses `World best` or `Target route`, so the UI wording remains intentional.

### Manual Verification

Run the non-headless game and confirm:

- Initial HUD displays position, energy, outpost progress, wood, stone, world best route score, and current route score.
- Wood/stone values update after collecting resources.
- Wood decreases after building a path.
- Stone decreases after placing a rock.
- Current route score increases as energy is spent.
- Current route score resets after a full outpost circuit.

---

## 5. Sequence & Workstreams `sequence`

### Phase A - Define HUD Status Semantics

1. Add or confirm `GameManager` fields for `visited_outposts` and `route_start_energy`.
2. Implement `_build_status()` as the only game-state-to-HUD adapter.
3. Include resource capacity, world best route score, and current route score in the returned status rows.
4. Choose the final label for the world best route score. Prefer `World best` or `Target route`.

### Phase B - Track Current Route Lifecycle

5. Implement `_update_route_tracking()` in `GameManager`.
6. Call `_update_route_tracking()` after successful game-loop actions.
7. Ensure route completion is based only on visiting all outposts, not on collecting resources or building paths.
8. Reset the route baseline after recording a completed route.

### Phase C - Render Multi-Row HUD Status

9. Change `Renderer.render_ui()` to accept `list[dict[str, str | int | float]]`.
10. Draw each row at a stable vertical offset using `font.get_linesize()`.
11. Keep `HUD_HEIGHT` large enough for the number of rendered rows.
12. Keep `Renderer` presentational: no access to target manager, route tracking fields, or route calculations.

### Phase D - Tests and Polish

13. Add focused tests for `_build_status()` and route completion tracking.
14. Update any existing tests affected by the status data shape.
15. Run targeted tests first, then broader checks.
16. Review labels and output readability in a live pygame window if possible.

---

## 6. Target Files `files`

| File | Expected Changes |
|---|---|
| `src/tsp_rl_kg/game_world/game_manager.py` | Build richer HUD status, track current route visits, reset completed route attempts |
| `src/tsp_rl_kg/renderer.py` | Render multi-row status dictionaries in the HUD |
| `tests/test_game_manager.py` | Add or update tests for resource capacity display and route tracking |
| `docs/specs/g20-hud-player-status.md` | This specification |

Do not modify files outside this list unless implementation reveals a direct dependency. If that happens, update this spec first.

---

## 7. Commands & Code Style `commands`

### Commands

Use targeted checks while implementing:

```bash
uv run pytest tests/test_game_manager.py -v
uv run ruff check src/tsp_rl_kg/game_world/game_manager.py src/tsp_rl_kg/renderer.py tests/test_game_manager.py
uv run black --check src/tsp_rl_kg/game_world/game_manager.py src/tsp_rl_kg/renderer.py tests/test_game_manager.py
```

Before considering the work complete:

```bash
uv run pytest tests/ -v
```

### Code Style

- Follow existing type hint style.
- Keep `_build_status()` small and deterministic.
- Prefer clear labels over terse labels when values could be ambiguous.
- Keep comments sparse; add them only around route lifecycle semantics if needed.
- Do not introduce new abstractions unless HUD status grows beyond this spec.

---

## 8. Boundaries & Acceptance `boundaries`

### Three-Tier Actions

#### Always Do

- Always keep `GameManager` as the source of HUD semantics.
- Always keep `Renderer` as a presentational component.
- Always update tests when changing the status data shape or label names.
- Always run the targeted game-manager tests before marking this work complete.
- Always self-audit against Goals, Non-Goals, and Rejection Criteria after each phase.

#### Ask First

- Ask before adding a new HUD field beyond resource capacity, world best route score, and current route score.
- Ask before adding dependencies or changing `pyproject.toml`.
- Ask before restructuring renderer architecture beyond multi-row text rendering.
- Ask before coupling HUD output to RL training metrics or reward internals.

#### Never Do

- Never modify `.venv/`, `__pycache__/`, generated recordings, or local artifacts.
- Never change movement costs or target route calculation for the sake of display.
- Never make the renderer mutate game state.
- Never combine this change with the path-disappearance fix.

### Acceptance Criteria

- HUD status includes wood and stone as `current/max`.
- HUD status includes a clearly labeled world target route energy.
- HUD status includes current route energy.
- Current route energy resets after all outposts have been visited and the completed route is recorded.
- The renderer can display all required status fields without raising.
- Targeted tests pass.

### Rejection Criteria

- Resource capacity is hard-coded instead of using `Agent.resource_max`.
- World best route score is calculated in the renderer.
- Current route score uses total lifetime energy without subtracting `route_start_energy`.
- Completed routes are not recorded in `route_energy_list`.
- Route tracking resets before every outpost has been visited.
- HUD label names make the world target score appear to be the player's personal best.
- Any existing test failure is left unexplained.

### Self-Audit Checklist

After each phase, verify:

- Does the change directly serve G1-G5?
- Did any non-goal slip into the diff?
- Are the target files still accurate?
- Do the tests prove the behavior rather than only checking implementation details?
- Are route score labels understandable from the UI alone?
