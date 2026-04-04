# System overview (`main.py` and package boundaries)

```mermaid
flowchart LR
    CLI["main.py Typer CLI\nplay / train / simulate"]

    subgraph ConfigPkg["tsp_rl_kg.config + utils"]
      CFG["TrainingConfig\nGameManagerConfig\nSimulationManagerConfig"]
      CFGL["load_config_file\nfind_mapping_section"]
    end

    subgraph GameplayPkg["game_world + rl runtime"]
      GM["game_world.game_manager.GameManager"]
      SIM["rl.simulation_manager.SimulationManager"]
      ENV["rl.custom_env.CustomEnv"]
    end

    subgraph TrainPkg["rl.training"]
      TR["trainer.Trainer"]
      MT["model_trainer.ModelTrainer"]
      BE["backends (TrainingBackend)"]
      EV["evaluation.EpisodeEvaluator"]
      TS["trajectory_store"]
    end

    subgraph GraphObsPkg["knowledge + graph + observation"]
      KG["knowledge.KnowledgeGraph"]
      CON["graph.constitution"]
      PROJ["graph.projection"]
      FE["graph.feature_encoder"]
      OE["observation.encoder"]
    end

    CLI --> CFGL --> CFG
    CLI -->|play| GM
    CLI -->|simulate| SIM
    CLI -->|train| TR

    TR --> ENV
    TR --> MT --> BE --> EV
    BE -. optional world-model data .-> TS

    ENV --> KG
    KG --> CON
    KG --> PROJ
    KG --> FE
    KG --> OE
```

Related diagrams:
- [Training backend protocols](training_backend_protocols.md)
- [SB3 backend internals](training_sb3_backend.md)
- [Training orchestration](training_orchestration.md)
- [KG and observation flow](kg_observation_flow.md)
