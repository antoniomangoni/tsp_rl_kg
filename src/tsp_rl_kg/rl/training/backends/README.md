# `tsp_rl_kg.rl.training.backends`

## Purpose / ownership
Owns backend abstraction contracts and concrete algorithm backend adapters used by the training stack.

## Main identifiers
- `TrainingBackend` protocol (`base.py`): core backend contract (`build`, `train`, `predict`, `save`, `collect_metrics`).
- Data/contract types in `base.py`: `CurriculumDecision`, `Transition`, `SequenceBatch`, `TransitionCollectionStats`, plus evaluator/curriculum/metrics/trajectory-related protocols.
- `SB3TrainingBackend` (`sb3.py`): Stable-Baselines3 implementation (PPO/DQN) with monitoring, callbacks, evaluation, and metric extraction.
- Re-export surface in `__init__.py` for protocol-oriented imports.

## Inputs / outputs and neighboring package interactions
- **Inputs:** configured train/eval environments, algorithm/model/evaluation configs, metric sinks, and callback controllers.
- **Outputs:** trained backend model state, predictions, saved artifacts, and normalized metrics dictionaries.
- **Neighbor interactions:**
  - Consumed by `rl.training.ModelTrainer` and `rl.training.Trainer`.
  - Uses `rl.agent_model.AgentModel` in SB3 policy kwargs.
  - Emits curriculum signals to `rl.training.curriculum` through callback wiring.

## Extension points
- Add new backend classes (e.g., custom torch trainer or other RL libraries) that satisfy `TrainingBackend`.
- Extend protocol types for richer transition/evaluation data while preserving compatibility.

## Cross-links
- [Training module](../README.md)
- [Top-level package interface](../../../README.md)

## Tests
- `tests/test_training_backends.py`
- `tests/test_training_entrypoints.py`
- `tests/test_evaluation.py`
