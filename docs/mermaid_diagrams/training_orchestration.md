# Training orchestration

```mermaid
flowchart TD
    Study[AblationStudy: isolated experiment and seed directories] --> Trainer
    Trainer --> Factory[EnvironmentManager: isolated seeded world generation]
    Factory --> Train[Training world pool]
    Factory --> Eval[Disjoint held-out world pool]
    Train --> Step[SB3 PPO or DQN transitions]
    Step --> Boundary[Episode boundary: restore template and advance pending curriculum]
    Step --> Metrics[CurriculumCallback: record metrics without resetting env]
    Step --> Periodic[FixedWorldEvalCallback: fixed schedule and episode count]
    Eval --> Periodic
    Step --> Final[EpisodeEvaluator: configured determinism and fixed schedule]
    Eval --> Final
    Trainer --> Artifacts[Model, metrics, config, profiler and scoped simulation exports]
    Trainer --> Cleanup[Finally: close initialized environments]
    Artifacts --> Results[Record every attempted seed; aggregate successes only]
    Results --> Failure[Any failed seed: partial results, failed MLflow parent, nonzero exit]
```

Trajectory collectors, stores, sequence samplers and model-update schedulers remain
standalone experimental utilities; active SB3 training does not invoke them.

- [Training backend protocols](training_backend_protocols.md)
- [SB3 backend internals](training_sb3_backend.md)
