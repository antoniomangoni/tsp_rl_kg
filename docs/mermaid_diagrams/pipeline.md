```mermaid
graph TD
    A[Environment] --> B[Agent]
    B --> C[KnowledgeGraph]
    C --> D[GraphProcessor]
    A --> E[VisionProcessor]
    D --> F[AgentModel]
    E --> F
    F --> G[Action Selection]
    G --> B