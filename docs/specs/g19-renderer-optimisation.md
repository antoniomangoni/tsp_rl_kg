# g19 — Renderer Dirty-Rect & Fog-of-War Optimisation

## Extended TOC

| § | Summary | Tag |
|---|---------|-----|
| 1 | Goals & Non-Goals: Scope entity/fog work to dirty tiles; don't touch headless path or CNN pipeline | `§goals` |
| 2 | Technical Decisions: Replace global entity draw with tile-scoped blits; deduplicate changed list; pre-allocate heatmap surface | `§decisions` |
| 3 | Sequence: Four phases: dirty-scoped entities, fog removal, heatmap alloc, dedup | `§sequence` |
| 4 | Target Files: `renderer.py`, `environment.py` | `§files` |
| 5 | Testing & Conformance: Visual + automated regression checks | `§testing` |

---

## 1. Goals & Non-Goals `§goals`

### Goals

- **G1 — Scope entity redraw to dirty tiles.** Replace `entity_group.draw(self.surface)` in `render_updated_tiles()` with targeted blits for only entities on changed tiles. Current cost: O(all_entities) per frame. Target: O(changed_tiles).
- **G2 — Eliminate `_overdraw_fog_on_entities()` full-group scan.** Once entity drawing is tile-scoped, fog overdraw on ALL entities is unnecessary — only changed undiscovered tiles need fog. Current cost: O(all_entities) per frame. Target: O(0) or O(changed_tiles).
- **G3 — Pre-allocate heatmap surface.** `render_heatmap()` allocates a new `pygame.Surface` per tile per call. Reuse a single surface created in `__init__`.
- **G4 — Deduplicate `changed_tiles_list`.** Convert from `list` to `set` to prevent duplicate work when a tile appears twice (e.g., entity move + discovery in same frame).

### Non-Goals

- Modifying the headless rendering path (`_get_vision_headless`).
- Changing the CNN observation pipeline or `get_clamped_surface()`.
- Altering `init_render()` — it runs once and is not a hot path.
- Refactoring `entity_group` away from `pygame.sprite.LayeredUpdates`.
- Changing discovery mechanics or `discovered_grid` semantics.

---

## 2. Technical Decisions & Stack `§decisions`

### Architecture

**Current hot path** (called every agent step in non-headless mode):

```
GameManager.rerender()
  → Renderer.render_updated_tiles()
      → for each tile in changed_tiles_list:
            update_tile(x, y)           # terrain + entity on terrain_surface
            blit terrain→surface
      → entity_group.draw(surface)      # ALL entities — O(N) ← PROBLEM
      → _overdraw_fog_on_entities()     # ALL entities — O(N) ← PROBLEM
      → pygame.display.update(dirty)
```

**Proposed hot path:**

```
GameManager.rerender()
  → Renderer.render_updated_tiles()
      → dirty = set(changed_tiles)             # deduplicated
      → for each (x, y) in dirty:
            update_tile(x, y)                   # terrain/fog on terrain_surface
            blit terrain→surface
            _draw_entity_on_tile(x, y)          # entity blit if discovered + entity present
      → pygame.display.update(dirty_rects)
```

Key changes:

1. **`render_updated_tiles()`**: Replace `entity_group.draw()` + `_overdraw_fog_on_entities()` with per-tile entity logic inside the existing dirty-tile loop. After blitting terrain, if discovered and entity present → blit entity. If undiscovered → tile is already fog from `update_tile()`.

2. **`_overdraw_fog_on_entities()`**: Remove entirely. Fog is already drawn by `update_tile()` for undiscovered tiles, and entity drawing is now scoped to discovered dirty tiles only.

3. **`changed_tiles_list` → `changed_tiles`**: Change from `list` to `set` in `Environment` to auto-deduplicate. Rename to `changed_tiles` (set semantics). Update `environment_changed()` and `single_environment_changed()` to use `.add()` instead of `.append()`. Update `changed_tiles_list.clear()` → `changed_tiles.clear()`.

4. **`render_heatmap()`**: Create `self._heat_surface` once in `__init__`, reuse with `.set_alpha()` and `.fill()` per tile.

### Why entity_group.draw() is safe to remove

`entity_group.draw()` (`LayeredUpdates.draw`) iterates every sprite and blits it. In the new design, `update_tile()` already handles the entity via `terrain_tile.entity_on_tile`. The only reason `entity_group.draw()` was needed was to catch entities that moved — but moved entities already appear in `changed_tiles_list` via `environment_changed()`. So scoping to dirty tiles is sound.

### Edge case: entity sprites larger than one tile

All entities use `BaseEntity` with `tile_size × tile_size` sprites (see `entities.py`). No multi-tile sprites exist, so per-tile entity blitting is correct.

---

## 3. Sequence `§sequence`

### Phase A — Scope entity draw to dirty tiles (G1 + G2)

**Depends on:** nothing
**Blocks:** Phase B (cleanup)

1. In `render_updated_tiles()`, remove the calls to `self.environment.entity_group.draw(self.surface)` and `self._overdraw_fog_on_entities()`.
2. In the dirty-tile loop (after blitting terrain→surface), add entity drawing logic: if `discovered_grid[x, y]` and `terrain_object_grid[x, y].entity_on_tile is not None`, blit the entity image onto `self.surface`.
3. The fog case is already handled: `update_tile()` blits `fog_tile` for undiscovered tiles, so no entity will appear on undiscovered dirty tiles.

### Phase B — Remove `_overdraw_fog_on_entities` (G2)

**Depends on:** Phase A

4. Delete the `_overdraw_fog_on_entities()` method from `Renderer`.
5. Remove its call from `init_render()`. Replace with inline fog-overdraw logic: after `entity_group.draw(self.surface)` in `init_render()`, loop over `entity_group` and blit fog for undiscovered sprites. Keep `entity_group.draw()` in `init_render()` since it runs once.

### Phase C — Deduplicate changed tiles (G4)

**Depends on:** nothing (parallel with A)

6. In `Environment.__init__()`, change `self.changed_tiles_list = []` → `self.changed_tiles: set[tuple[int, int]] = set()`.
7. In `Environment.environment_changed()`, replace `.append()` → `.add()`.
8. In `Environment.single_environment_changed()`, replace `.append()` → `.add()`.
9. In `Renderer.render_updated_tiles()`, update references from `changed_tiles_list` → `changed_tiles`.
10. In `GameManager` or any other consumer that references `changed_tiles_list`, update the name. Search for all usages.

### Phase D — Pre-allocate heatmap surface (G3)

**Depends on:** nothing (parallel with A, C)

11. In `Renderer.__init__()`, create `self._heat_surface = pygame.Surface((self.tile_size, self.tile_size))`.
12. In `render_heatmap()`, replace `heat_rect = pygame.Surface(...)` with `self._heat_surface.set_alpha(alpha)` + `self._heat_surface.fill(color[:3])` + blit `self._heat_surface`.

---

## 4. Target Files `§files`

| File | Changes |
|------|---------|
| `src/tsp_rl_kg/renderer.py` | Phases A, B, C (ref updates), D — scope entity draw, remove `_overdraw_fog_on_entities`, heatmap pre-alloc, rename `changed_tiles_list` refs |
| `src/tsp_rl_kg/game_world/environment.py` | Phase C — `changed_tiles_list` → `changed_tiles` set, `.append()` → `.add()` |

No other files reference `changed_tiles_list` (verified by grep).

---

## 5. Testing & Conformance `§testing`

### Automated

1. `uv run pytest tests/ -v` — all 210 existing tests must pass with zero regressions.
2. `uv run ruff check src/tsp_rl_kg/renderer.py src/tsp_rl_kg/game_world/environment.py` — no lint errors.
3. `uv run black --check src/tsp_rl_kg/renderer.py src/tsp_rl_kg/game_world/environment.py` — formatting.

### Visual verification (manual)

4. Run `uv run tsp-rl-kg` non-headless. Confirm:
   - Starting area rendered, rest is black fog.
   - Scouting reveals tiles with correct terrain + entities.
   - Entities on undiscovered tiles remain hidden (no flicker).
   - Agent movement renders correctly (old tile clears, new tile shows).
   - Heatmap overlay (if enabled) renders identically to before.

### Rejection criteria

- Any entity sprite visible on an undiscovered tile → **fail**.
- `entity_group.draw()` still called in `render_updated_tiles()` → **fail** (G1 not met).
- `_overdraw_fog_on_entities()` still exists as a method → **fail** (G2 not met).
- `pygame.Surface(...)` allocated inside `render_heatmap()` loop → **fail** (G3 not met).
- `changed_tiles_list` still a `list` → **fail** (G4 not met).
- Any existing test fails → **fail**.

---

## Three-Tier Actions

### Always Do

- Run `uv run pytest tests/ -v` before considering any phase complete.
- Run `uv run ruff check` and `uv run black --check` on changed files.
- Search for all references to renamed symbols (`changed_tiles_list`) before renaming.

### Ask First

- Adding new dependencies to `pyproject.toml`.
- Modifying `init_render()` beyond removing the `_overdraw_fog_on_entities()` call.

### Never Do

- Modify `_get_vision_headless()` or `get_clamped_surface()` (out of scope).
- Change `discovered_grid` semantics or discovery mechanics.
- Alter entity sprite classes or `BaseEntity`.
- Remove the fog-of-war feature introduced in the prior work.
