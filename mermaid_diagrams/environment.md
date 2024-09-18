```mermaid
graph TD
    A[HeightmapGenerator] --> |"generates terrain"| B[Environment]
    B --> |"contains"| C[Terrains]
    B --> |"contains"| D[Entities]
    C --> |"may contain"|D
