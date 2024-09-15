```mermaid
graph TD
    A[HeightmapGenerator] -->|"generate() / save_heightmap()"| B[Environment]
    B -->|initialize_environment()| C[Terrain Types]
    C -->|create_image() / add_entity()| D[Entity]
    D -->|move() / update()| E[Specific Entities]
    E -->|handle_entity()| F[Entity Manager]
    F -->|manage entities| G[User Actions]
    G -->|interact with entities| H[Environment]

    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    class A,B,C,D,E,F,G,H main;
