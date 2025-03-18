```mermaid
flowchart TD
    A["Input / Observation"] --> B1["Convolutional Layers"]
    B1 --> B2["Fully Connected Layers"]
    A --> C1["GAT Layers"]
    C1 --> C2["Fully Connected Layers"]
    C2 --> D["Combined Features"]
    B2 --> D["Combined Features"]
    D --> E1
    E1 --> E2
    
    subgraph VisionProcessor["VisionProcessor"]
        direction TB
        B1["Convolutional Layers"]
        B2["Fully Connected Layers"]
    end
    
    subgraph GraphProcessor["GraphProcessor"]
        direction TB
        C1["GAT Layers"]
        C2["Fully Connected Layers"]
    end

    subgraph AgentModel["Agent Model"]
        direction TB
        E1["Fully Connected Layers"]
        E2["Output"]
    end