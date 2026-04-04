# KG and observation flow (`knowledge_graph.py`, `graph/*`, `observation/encoder.py`)

```mermaid
flowchart LR
    ENV["Environment\nterrain_index_grid\nentity_index_grid\ndiscovered_grid"]
    KG["KnowledgeGraph"]

    CON["DefaultGridConstitution\n(GraphConstitution)"]
    FE["FeatureEncoder\nRawInt / OneHot / EmbeddingLookup"]
    PROJ["ProjectionPolicy\nCompleteness / KHop / FullGraph"]
    SUB["Subgraph Data\n(x, edge_index, edge_attr)"]
    OBS["PaddedPyGObservationEncoder"]
    OUT["Gym Dict obs\nvision + node_features + edge_attr + edge_index"]

    ENV -->|shared refs (read-only)| KG
    KG -->|build graph| CON
    CON -->|encode nodes/edges| FE
    KG -->|get_subgraph(player terrain idx)| PROJ
    PROJ --> SUB
    ENV -->|vision window| OBS
    SUB --> OBS
    OBS --> OUT
```

Related diagrams:
- [System overview](system_overview.md)
