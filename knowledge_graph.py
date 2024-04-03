import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, terrain_indicies_grid, entity_indicies_grid, agent):
        super(KnowledgeGraph, self).__init__()
        self.entity_indicies_grid = entity_indicies_grid
        self.terrain_indicies_grid = terrain_indicies_grid
        self.agent = agent
        self.nodes, self.edges = self.construct_graph()
        self.print_kg()

    def construct_graph(self):
        # Assume entity_indicies_grid and terrain_indicies_grid have the same shape
        height, width = self.terrain_indicies_grid.shape
        node_features = []
        edge_indices = []

        # Mapping each (x, y) to a node index
        node_index = lambda x, y: x * width + y

        # Iterate over each tile to construct the graph
        for x in range(height):
            for y in range(width):
                # Combine terrain and entity indices for node features
                terrain_feature = self.terrain_indicies_grid[x, y]
                entity_feature = self.entity_indicies_grid[x, y]
                node_features.append([terrain_feature, entity_feature])

                # Connect nodes in all eight directions (or four if preferred)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip the node itself
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width:
                            edge_indices.append([node_index(x, y), node_index(nx, ny)])

        # Convert to tensors
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return node_features, edge_indices

    def to_torch_geometric(self):
        # Use the nodes and edges to create a torch geometric Data object
        data = pyg.data.Data(x=self.nodes, edge_index=self.edges)
        return data

    def print_kg(self):
        # Convert the nodes and edges to a networkx graph for visualization
        G = nx.Graph()
        for i, (terrain, entity) in enumerate(self.nodes):
            G.add_node(i, terrain=terrain, entity=entity)
        for edge in self.edges.t().tolist():
            G.add_edge(edge[0], edge[1])
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
