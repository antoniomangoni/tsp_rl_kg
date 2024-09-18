```mermaid
graph TD
    A[Heightmap Generator] --> B[Environment]
    B --> C[Terrain]
    B --> D[Entities]
    E[Pygame Renderer] --> B
    F[Gymnasium Interface] --> B
    B --> G[Game World]
    H[Target Manager] --> G