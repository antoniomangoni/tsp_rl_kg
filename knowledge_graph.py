import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, environment, agent):
        super(KnowledgeGraph, self).__init__()
        self.environment = environment
        self.agent = agent
        self.terrain_features, self.entity_features, self.nodes, self.edges = self.construct_graph()
        self.print_kg()

    def construct_graph(self):
        height, width = self.environment.heightmap.shape
        terrain_features = []
        entity_features = []
        edge_indices = []

        node_index = lambda x, y: x * width + y

        for x in range(height):
            for y in range(width):
                terrain_object = self.environment.terrain_object_grid[x][y]
                terrain_feature = terrain_object.elevation
                terrain_node_idx = node_index(x, y)
                terrain_features.append([terrain_feature])

                if terrain_object.entity_on_tile:  # Check if there is an entity on the tile
                    entity_feature = terrain_object.entity_on_tile.id
                    entity_node_idx = len(terrain_features) - 1 + len(entity_features)  # Index based on counts
                    entity_features.append([entity_feature])
                    edge_indices.append([terrain_node_idx, entity_node_idx])  # "Contains" relationship

                # Connect nodes in four directions
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        edge_indices.append([terrain_node_idx, node_index(nx, ny)])

        # Convert to tensors
        terrain_features = torch.tensor(terrain_features, dtype=torch.float)
        entity_features = torch.tensor(entity_features, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return terrain_features, entity_features, torch.cat((terrain_features, entity_features), dim=0), edge_indices

    def to_torch_geometric(self):
        data = pyg.data.Data(x=self.nodes, edge_index=self.edges)
        return data

    def print_kg(self):
        G = nx.Graph()
        num_terrain_nodes = len(self.terrain_features)
        total_nodes = num_terrain_nodes + len(self.entity_features)
        for i, feature in enumerate(self.nodes):
            if i < num_terrain_nodes:
                G.add_node(i, terrain=feature.item(), entity=None)
            else:
                G.add_node(i, terrain=None, entity=feature.item())
        for edge in self.edges.t().tolist():
            G.add_edge(edge[0], edge[1])

        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
