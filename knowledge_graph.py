import torch
import torch_geometric as pyg
from torch_geometric.utils import subgraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, environment, vision_range, completion=1.0):
        super(KnowledgeGraph, self).__init__()
        self.environment = environment
        self.vision_range = vision_range
        self.completion = completion
        self.terrain_features, self.full_nodes, self.full_edges, self.terrain_graph = self.construct_terrain_graph()
        self.subgraph_nodes, self.subgraph_edges = self.get_terrain_subgraph(completion)
        self.graph = self.add_entities_to_graph()
        self.visualize()

    def construct_terrain_graph(self):
        """Constructs the graph for the entire terrain."""
        height, width = self.environment.heightmap.shape
        terrain_features = []
        edge_indices = []

        for x in range(height):
            for y in range(width):
                terrain_object = self.environment.terrain_object_grid[x][y]
                terrain_feature = terrain_object.elevation
                node_idx = self.node_index(x, y, width)
                terrain_features.append([terrain_feature])

                # Connect nodes in four directions
                edge_indices.extend([[node_idx, self.node_index(nx, ny, width)]
                                     for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                                     if 0 <= (nx := x + dx) < height and 0 <= (ny := y + dy) < width])

        terrain_features = torch.tensor(terrain_features, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        terrain_nodes = torch.cat((terrain_features), dim=0)
        return terrain_features, terrain_nodes, edge_indices, pyg.data.Data(x=terrain_nodes, edge_index=edge_indices)

    def get_terrain_subgraph(self, completion):
        """Extracts a subgraph around the player based on the completion parameter."""
        player_idx = self.node_index(self.environment.player.x, self.environment.player.y, self.environment.heightmap.shape[1])
        # Assume completion influences the number of nodes in the player's vicinity
        radius = int(completion * self.vision_range)  # Adjust radius calculation as needed
        nodes_to_include = set([player_idx])  # Start with the player node
        
        # Add nearby nodes based on radius - simplistic breadth-first search approach
        for _ in range(radius):
            new_nodes = set()
            for node in nodes_to_include:
                # Consider nodes directly connected to current node set
                edges = self.full_edges.t().numpy()
                connected = np.unique(edges[np.isin(edges[:, 0], list(nodes_to_include))][:, 1])
                new_nodes.update(connected)
            nodes_to_include.update(new_nodes)
        
        subgraph_nodes, subgraph_edge_indices = subgraph(torch.tensor(list(nodes_to_include), dtype=torch.long), self.full_edges, relabel_nodes=True)
        return self.terrain_graph.x[subgraph_nodes], subgraph_edge_indices

    def add_entities_to_graph(self):
        """Adds entities to the subgraph."""
        entity_features = []
        edge_indices = list(self.subgraph_edges)
        player_node_idx = self.node_index(self.environment.player.x, self.environment.player.y, self.environment.heightmap.shape[1])
        start_entity_idx = len(self.subgraph_nodes)

        for entity in self.environment.entities:
            entity_feature = [entity.id]
            entity_node_idx = start_entity_idx + len(entity_features)
            entity_features.append(entity_feature)
            terrain_node_idx = self.node_index(entity.x, entity.y, self.environment.heightmap.shape[1])

            edge_indices.append([entity_node_idx, terrain_node_idx])
            edge_indices.append([entity_node_idx, player_node_idx])

        entity_features = torch.tensor(entity_features, dtype=torch.float)
        all_nodes = torch.cat((self.subgraph_nodes, entity_features), dim=0)
        all_edges = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return pyg.data.Data(x=all_nodes, edge_index=all_edges)

    def visualize(self):
        """Visualizes the current graph (subgraph with entities)."""
        G = nx.Graph()
        for i, feature in enumerate(self.graph.x):
            G.add_node(i, value=feature.item())
        G.add_edges_from(self.graph.edge_index.t().tolist())
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, font_weight='bold', node_size=10, node_color='skyblue', font_size=8, font_color='black')
        plt.show()

    def node_index(self, x, y, width):
        """Calculates a node's index in a flat array from 2D coordinates."""
        return x * width + y
