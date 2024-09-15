```mermaid
---
config:
  layout: elk
---
graph TD
    GM[Game Manager] --> |creates/manages| ENV[Environment]
    GM --> |creates/manages| AGENT[Agent]
    GM --> |initializes| KG[Knowledge Graph]
    KGM[Knowledge Graph Indicies] --> |manages indices| KG
    GM --> |creates| TM[Target Manager]
    GM --> |creates| REND[Renderer]
    ENV --> |generates terrain| HG[Heightmap Generator]
    ENV --> |contains| ENT[Entities]
    ENV --> |contains| TER[Terrains]
    TER --> |contains| ENT
    ENV --> |contains| PLAYER[Player]
    PLAYER --> |is the| AGENT
    AGENT --> |interacts with| ENV
    AGENT --> |updates| KG
    KG --> |represents| ENV
    TM --> |analyzes| ENV
    REND --> |visualizes| ENV
    REND --> |visualizes| AGENT