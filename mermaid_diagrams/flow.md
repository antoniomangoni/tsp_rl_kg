```mermaid
flowchart TD
 subgraph subGraph0["Environment Components"]
        E["Environment"]
        I["Terrains"]
        J["Entities"]
        P["Heightmap Generator"]
  end
 subgraph AgentModel["AgentModel"]
        M["Vision Processor"]
        N["Graph Processor"]
  end
 subgraph subGraph2["Model Components"]
        K["PPO Model"]
        L["AgentModel"]
        AgentModel
  end
 subgraph subGraph3["PPO Model"]
        R["Policy Network"]
        S["Value Network"]
        T["Experience Buffer"]
        U["Advantage Computation"]
        V["Optimization Process"]
        F["Agent"]
  end
    A["Training"] --> B["CustomEnv"]
    B --> C["SimulationManager"] & T
    C --> D["GameManager"]
    D -- creates --> E
    H["Target Manager"] --> D
    E --> I & J & O["Game World"]
    J -- are in --> I
    A -- starts --> K
    B <-- experiences/actions --> K
    K --> L & R & S
    G["KnowledgeGraph"] -- creates --> Q["Observations"]
    Q --> L
    L -- controls --> F
    F -- creates --> Q
    O -- has --> F & G
    D -- has --> O
    P --> E
    R --> F
    S --> U
    T --> U
    U --> V
    V --> R & S