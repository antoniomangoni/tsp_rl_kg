# SB3 backend internals (`rl/training/backends/sb3.py`)

```mermaid
flowchart TD
    SB3B["SB3TrainingBackend"]
    ALG["SB3_ALGORITHMS\n{PPO, DQN}"]
    MON["_ensure_monitored_env()\nMonitor(env)"]
    BUILD["build()"]
    LEARN["model.learn(...)"]
    CB["CurriculumCallback"]
    CURR["CurriculumService"]
    EVCB["EvalCallback\n(optional when output_dir set)"]
    MET["collect_metrics()"]

    SB3B --> MON
    SB3B --> BUILD --> ALG
    BUILD --> MODEL["stable_baselines3 model\n(policy + AgentModel extractor)"]

    SB3B -->|train()| CURR
    CURR --> CB
    SB3B -->|train()| EVCB
    MODEL --> LEARN
    CB --> LEARN
    EVCB --> LEARN

    SB3B -->|predict() / save()| MODEL
    SB3B --> MET --> OUT["MetricsDict\nmean_reward, losses, exploration_rate"]
```

Related diagrams:
- [Training backend protocols](training_backend_protocols.md)
- [Training orchestration](training_orchestration.md)
