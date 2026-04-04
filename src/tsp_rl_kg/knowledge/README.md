# `tsp_rl_kg.knowledge`

## Purpose / ownership
Owns the runtime knowledge graph view over environment state. It translates mutable world arrays into graph tensors and keeps player-centric graph state synchronized with world evolution.

## Main identifiers
- `KnowledgeGraph` (`knowledge_graph.py`): central class for graph ownership, player-edge rewiring, node feature refresh, discovery updates, and optional visualization.
- `Graph_Manager` re-export shim (`graph_idx_manager.py`) for compatibility with graph index utilities.

## Inputs / outputs and neighboring package interactions
- **Inputs:** `Environment` references (terrain/entity/discovery arrays), `vision_range`, projection policy, and feature encoder.
- **Outputs:** PyG graph tensors (`graph.x`, `edge_index`, `edge_attr`) and index bookkeeping exposed through `graph_manager`.
- **Neighbor interactions:**
  - Consumes graph construction/projection/encoding logic from `graph` package.
  - Is initialized from `game_world.GameManager` and updated indirectly during agent actions.
  - Supplies graph data consumed by `observation` encoders and `rl` model stack.

## Extension points
- Plug in custom `GraphConstitution` builders.
- Plug in custom `ProjectionPolicy` for partial observability.
- Plug in custom feature encoders implementing graph encoding protocol.

## Cross-links
- [Graph module](../graph/README.md)
- [Game world module](../game_world/README.md)
- [Observation module](../observation/README.md)

## Tests
- `tests/test_knowledge_graph.py`
- `tests/test_state_sync.py`
- `tests/test_projection.py`
