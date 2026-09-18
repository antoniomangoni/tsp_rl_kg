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

## Outputs and tracking

Paths are relative to the current working directory. The CLIs log the generated
results directory when a run starts.

| Workflow | Output location | Contents |
| --- | --- | --- |
| `tsp train` | `results/manual_<timestamp>_<id>/manual_<algorithm>/` | Final model ZIP, metrics CSV, profiler report, train/eval simulation CSVs |
| `tsp train --benchmark` | Same training layout, plus `results/benchmark_<timestamp>.json` | Vision-only run summary with evaluation metrics and artifact paths |
| `tsp-study` | `results/<timestamp>_<id>/` | Base config, study summary, combined and per-experiment results |
| Each study seed | `<study-dir>/<experiment>_seed_<seed>/` | Model, metrics, profiling, simulation exports, or failure diagnostics |

Training writes `<experiment>_metrics.csv`, `profile_stats.txt`, and
`<backend>_custom_env_<experiment>.zip`. Simulation exports are scoped to
`train/` and `eval/`, each containing `static_data.csv` and `game_data.csv`.
Periodic evaluation can also create `best_model.zip` and `evaluations.npz` when
its configured interval is reached.

For studies, start with `study_summary.json` and `ablation_study_results.json`.
Per-experiment `<experiment>_results.json` files include every seed attempt,
error details, success counts, and the `incomplete` flag. The study also writes
`base_config.json`; successful seed results contain the resolved configuration.
A nonzero exit after seed failures can still leave useful successful artifacts.

Studies create an MLflow parent run and nested seed runs. Set
`mlflow_experiment_name` and optionally `mlflow_tracking_uri` in the selected
study mapping. For example, local SQLite tracking can be configured with
`mlflow_tracking_uri = "sqlite:///mlflow.db"` in `[study]`. When omitted, the
study uses MLflow's configured tracking URI. Plain `tsp train` does not create
an MLflow run; trainer logging occurs only if a run is already active.

Use `configs/ablation.toml` from the repository root for a bounded pipeline
check. The full sweep in `configs/ablation_full.toml` is intended for research
runs. Integration coverage uses temporary local tracking and does not require
an external MLflow service.
