```mermaid
flowchart TD
    A[Start Reward Calculation] --> B{Agent Action}
    B --> |Movement| C[Apply Step Penalty]
    C --> D[Apply Time Penalty]
    D --> E{Reached New Outpost?}
    E --> |Yes| F[Add Outpost Reward]
    E --> |No| G{Closer to Unvisited Outpost?}
    G --> |Yes| H[Add Closer Reward]
    G --> |No| I[Add Farther Penalty]
    F --> J{All Outposts Visited?}
    H --> K{Circular Behavior?}
    I --> K
    J --> |Yes| L[Add Completion Reward]
    J --> |No| K
    K --> |Yes| M[Add Circular Behavior Penalty]
    K --> |No| N[Final Reward Calculation]
    M --> N
    L --> O{Better Than Previous Best?}
    O --> |Yes| P[Add Improvement Reward]
    O --> |No| Q[Check Max Non-Improvement]
    P --> R[Update Best Route]
    Q --> S{Better Than Algorithm?}
    R --> S
    S --> |Yes| T[Add Better Than Algo Reward]
    S --> |No| U[Normalize Final Reward]
    T --> U
    N --> U

    %% subgraph "Base Rewards/Penalties"
    %%     direction LR
    %%     BA[New Outpost: 30]
    %%     BB[Completion: 100]
    %%     BC[Route Improvement: 200]
    %%     BD[Better Than Algo: 200]
    %%     BE[Step Penalty: -0.5]
    %%     BF[Farther Penalty: -1.0]
    %%     BG[Circular Penalty: -2.0]
    %%     BH[Closer Reward: 0.55]
    %% end