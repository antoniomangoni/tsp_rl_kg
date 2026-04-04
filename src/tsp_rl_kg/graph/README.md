# `tsp_rl_kg.graph`

## Purpose / ownership
Owns graph construction, indexing, projection policies, and node/edge feature encoding used to materialize world state as PyTorch Geometric data.

## Main identifiers
- `DefaultGridConstitution`, `GraphConstitution` (`constitution.py`): graph build strategy contract + default implementation.
- `ProjectionPolicy`, `KHopProjection`, `FullGraphProjection`, `CompletenessProjection` (`projection.py`).
- `FeatureEncoder` protocol and concrete encoders `RawIntEncoder`, `OneHotEncoder`, `EmbeddingLookupEncoder` (`feature_encoder.py`).
- `Graph_Manager` (`graph_idx_manager.py`): coordinate/z-level to node/edge index bookkeeping.
- `build_feature_encoder(...)`: config-driven encoder factory.

## Inputs / outputs and neighboring package interactions
- **Inputs:** environment grid sizes/state, semantic schema files, feature encoding config, and projection parameters.
- **Outputs:** graph topology tensors, encoded node/edge features, semantic descriptor metadata, and index lookup helpers.
- **Neighbor interactions:**
  - Consumed by `knowledge.KnowledgeGraph` for graph instantiation and updates.
  - Consumed by `rl.training.Trainer` to instantiate feature encoders from config.
  - Supports `observation`/`rl` by defining graph tensor shapes and semantics.

## Extension points
- Add projection strategies implementing `ProjectionPolicy`.
- Add encoders implementing `FeatureEncoder` and integrate into `build_feature_encoder`.
- Replace constitution logic for non-grid or multi-layer graph layouts.

## Cross-links
- [Knowledge module](../knowledge/README.md)
- [Observation module](../observation/README.md)
- [RL training module](../rl/training/README.md)

## Tests
- `tests/test_constitution.py`
- `tests/test_projection.py`
- `tests/test_feature_encoder.py`
- `tests/test_semantic_schema.py`
