# Training orchestration (`trainer.py`, `environment_manager.py`, `curriculum.py`, `evaluation.py`, `trajectory_store.py`)

```mermaid
flowchart TD
    TR["Trainer.setup/run"] --> EM["EnvironmentManager\nmake_env / make_eval_env"]
    EM --> CENV["CustomEnv(train)"]
    EM --> EENV["CustomEnv(eval)"]

    TR --> MT["ModelTrainer"]
    MT --> BACKEND["TrainingBackend impl\n(SB3TrainingBackend)"]
    BACKEND --> CURR["CurriculumService.on_step"]
    CURR --> MSINK["TrainingMetrics.record"]

    BACKEND -->|post-train| EVAL["EpisodeEvaluator.evaluate"]
    EVAL --> RES["mean/std reward + episode length"]

    BACKEND -. optional rollout collection .-> COL["OnlineTrajectoryCollector.collect"]
    COL --> STORE["InMemoryTrajectoryStore"]
    STORE --> EP["EpisodeTrajectory\n(completed episodes)"]

    TR --> ART["save model + metrics CSV + profiler stats"]
```

Related diagrams:
- [Training backend protocols](training_backend_protocols.md)
- [SB3 backend internals](training_sb3_backend.md)
