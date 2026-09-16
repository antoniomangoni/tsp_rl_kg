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

## Reliability contract

World generation uses separate reproducible training/evaluation seed streams and verifies
that the pools are disjoint. Periodic and final evaluation restart a fixed world schedule.
Curriculum transitions happen at episode reset; callbacks never reset an environment while
SB3 holds its previous observation. Trainer setup and execution both clean up on failure.

Each study/run has a unique directory. Every seed attempt has a status, artifact directory,
and error details when it fails. Successful seeds alone contribute to aggregates, which
are marked incomplete after any failure. Partial results are written before a study-level
exception; MLflow retains failed child runs and marks the parent failed.

`trajectory_store.py`, `sequence_sampler.py`, their protocol types, and the standalone
replay/sequence/world-model config classes are experimental utilities. Active TrainingConfig
rejects these unused sections. Supported algorithm parameters are forwarded to SB3;
DQN replay settings belong in `algorithm.hyperparameters`.

Regression coverage: `tests/test_experiment_reliability.py`,
`tests/test_training_integration.py`, and `tests/test_packaging.py`. Integration tests
use temporary local MLflow storage and bounded CPU runs.
