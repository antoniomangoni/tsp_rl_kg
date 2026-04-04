# `tsp_rl_kg` package

## Purpose / ownership
This top-level package owns the end-to-end integration of:
- world simulation (`game_world`),
- graph state construction and projection (`graph`, `knowledge`),
- RL environment/model/training orchestration (`rl`), and
- shared utilities/config wiring (`utils`, `config`).

It also re-exports backend protocol identifiers so downstream code can depend on stable interfaces without importing deep module paths.

## Main identifiers
- `TrainingBackend`, `Evaluator`, `CurriculumController`, `MetricsSink` (protocol contracts re-exported in `__init__.py`).
- `Transition`, `TransitionCollectionStats`, `SequenceBatch`, `TrajectoryStore` (typed training data contracts).

## Inputs / outputs and neighboring package interactions
- **Inputs:** typed config objects (`tsp_rl_kg.config`), world state from `game_world`, and graph tensors from `knowledge`/`graph`.
- **Outputs:** Gymnasium-compatible observations/actions through `rl.custom_env.CustomEnv`, training artifacts via `rl.training`, and protocol-level abstractions for backend implementations.
- **Neighbor interactions:** this package is the integration boundary; subpackages call each other through explicit interfaces (e.g., `GameManager` -> `KnowledgeGraph` -> observation encoder -> RL model/trainer).

## Extension points
- Implement a custom training backend by following `TrainingBackend` protocol and integrating under `rl/training/backends`.
- Add new feature encoders/projections in `graph` and wire them through trainer/environment setup.
- Add new curriculum/metrics behavior by implementing protocol contracts consumed by callbacks/services.

## Cross-links
- [Game world module](./game_world/README.md)
- [Knowledge module](./knowledge/README.md)
- [Graph module](./graph/README.md)
- [Observation module](./observation/README.md)
- [RL module](./rl/README.md)
- [Utilities module](./utils/README.md)

## Tests
Relevant tests validating package-level integration and contracts:
- `tests/test_training_entrypoints.py`
- `tests/test_training_backends.py`
- `tests/test_state_sync.py`
- `tests/test_example_configs.py`
