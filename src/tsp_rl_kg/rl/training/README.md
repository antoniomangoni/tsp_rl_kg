# `tsp_rl_kg.rl.training`

## Purpose / ownership
Owns experiment orchestration and training lifecycle: environment creation, backend selection/invocation, curriculum callbacks/services, evaluation, metrics, trajectory utilities, and CLI entrypoints.

## Main identifiers
- `Trainer` (`trainer.py`): high-level experiment setup + run orchestration.
- `ModelTrainer` (`model_trainer.py`): backend-agnostic train/eval/save workflow.
- `EnvironmentManager` (`environment_manager.py`): train/eval environment factories.
- `EpisodeEvaluator` (`evaluation.py`), `CurriculumService` (`curriculum.py`), `CurriculumCallback` (`callbacks.py`).
- `TrainingMetrics` (`metrics.py`).
- `InMemoryTrajectoryStore`, `OnlineTrajectoryCollector` (`trajectory_store.py`).
- `RandomSequenceSampler`, `PeriodicModelUpdateScheduler` (`sequence_sampler.py`).
- CLI/entrypoints in `run.py`; ablation workflow in `ablation_study.py`.

## Inputs / outputs and neighboring package interactions
- **Inputs:** `TrainingConfig` trees, backend configs, env/model dependencies from `rl`, and optional ablation study specs.
- **Outputs:** trained model artifacts, evaluation summaries, metrics CSV/logs, and profiling outputs.
- **Neighbor interactions:**
  - Instantiates `rl.CustomEnv` via `EnvironmentManager`.
  - Uses backend contracts from `rl.training.backends`.
  - Uses utility helpers (`utils.config_files`, `utils.logger`) for config loading and runtime logging.

## Extension points
- Add new trainers/services while keeping protocol boundaries.
- Add new entrypoint modes in `run.py`.
- Add alternative samplers, schedulers, or trajectory stores for offline/online RL workflows.

## Cross-links
- [Backends module](./backends/README.md)
- [RL module](../README.md)
- [Utilities module](../../utils/README.md)

## Tests
- `tests/test_training_entrypoints.py`
- `tests/test_evaluation.py`
- `tests/test_trajectory_store.py`
- `tests/test_curriculum_controller.py`
