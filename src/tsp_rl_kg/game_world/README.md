# `tsp_rl_kg.game_world`

## Purpose / ownership
Owns the grid-world simulation: terrain generation, entities, action semantics, mutable environment state, and runtime orchestration for a single world instance.

## Main identifiers
- `GameManager` (`game_manager.py`): high-level world lifecycle and coordination.
- `Environment` (`environment.py`): source of truth for terrain/entity/player grid state.
- `Agent` (`agent.py`): executes actions against environment + KG updates.
- `ActionType` (`actions.py`): discrete action enum and movement/collect deltas.
- `HeightmapGenerator` (`heightmap_generator.py`): procedural terrain synthesis.
- `Terrain` subclasses (`terrains.py`) and entity classes (`entities.py`).

## Inputs / outputs and neighboring package interactions
- **Inputs:** `GameManagerConfig`, terrain thresholds, agent actions, and optional feature/projection settings.
- **Outputs:** updated world state (terrain/entity grids), player position/energy/path state, and rendered frames when not headless.
- **Neighbor interactions:**
  - Initializes and feeds `knowledge.KnowledgeGraph`.
  - Used by `rl.SimulationManager` and `rl.CustomEnv` as the simulation backend.
  - Provides state consumed by observation and reward systems.

## Extension points
- Add new `Terrain`/entity types and update encoding/schema mappings.
- Add new `ActionType` or movement rules in `agent.py`.
- Swap world generation logic by extending/replacing `HeightmapGenerator`.

## Cross-links
- [Top-level package](../README.md)
- [Knowledge graph module](../knowledge/README.md)
- [RL module](../rl/README.md)

## Tests
- `tests/test_environment.py`
- `tests/test_state_sync.py`
- `tests/test_custom_env.py`
