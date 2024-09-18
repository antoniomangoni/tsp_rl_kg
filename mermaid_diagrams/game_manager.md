```mermaid
graph TD
    GM[Game Manager] --> |creates/manages| ENV[Environment]
    GM --> |creates/manages| AGENT[Agent]
    GM --> |initializes| KG[Knowledge Graph]
    GM --> |creates| TM[Target Manager]
    GM --> |creates| REND[Renderer]
    
    ENV --> |generates terrain| HG[Heightmap Generator]
    ENV --> |contains| ENT[Entities]
    ENV --> |contains| TER[Terrains]
    TER--> |contains| ENT
    ENV --> |contains| PLAYER[Player]
    
    AGENT --> |interacts with| ENV
    AGENT <--> |updates/uses| KG
    
    KG --> |represents| ENV
    
    TM --> |analyzes| ENV
    
    REND --> |visualizes| ENV
    REND --> |visualizes| AGENT
