# Knowledge and observations (schema v2)

```mermaid
flowchart TD
    Template[Immutable world template] --> Reset[Episode reset]
    Seed[Run seed + template fingerprint] --> Prior[Nested initial tile prior]
    Reset --> World[Live environment]
    Prior --> Memory[KnowledgeState remembered tiles]
    World --> Sense[Local sensing / scout / direct actions]
    Sense --> Memory
    Memory --> Graph[Known induced graph with local node IDs]
    World --> Vision[Canonical semantic RGB window]
    Graph --> Pad[Padded observations + node/edge counts]
    Vision --> Pad
    Pad --> Compact[Compact per-sample graph batch]
    Compact --> GAT[GAT and real-node pooling]
```

Completeness affects initial knowledge only. Sensing controls visual discovery independently.
Unseen changes never refresh memory. Episode reset restores the pristine world and prior.
The player is separate from underlying entities and encoded exclusively by its feature encoder.
