import networkx as nx
import numpy as np
from numpy.linalg import norm
import torch_geometric as pyg
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self, environment, vision_range, completion=1.0):
        self.environment = environment
        self.vision_range = vision_range
        self.completion = completion
        self.terrain_graph = self.construct_terrain_graph()
        self.visualize(self.terrain_graph)
        
        distance = self.get_distance(completion)
        print(f"Distance: {distance}")
        self.subgraph = self.get_subgraph_bfs(self.terrain_graph, distance)
        self.visualize(self.subgraph)

        self.graph = self.add_entities_to_graph(self.subgraph)
        self.visualize(self.graph)

        self.pyg_graph = self.convert_to_pyg(self.graph)  # Convert to PyTorch Geometric format

    def get_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        max_dimension = max(self.environment.heightmap.shape)
        if completion >= 1.0:
            return float('inf')  # Represents the entire graph
        return max(completion * max_dimension, self.vision_range)
    
    def construct_terrain_graph(self):
        G = nx.Graph()
        height, width = self.environment.heightmap.shape
        for y in range(height):
            for x in range(width):
                idx = x + y * width
                G.add_node(idx, pos=(x, y), feature=self.environment.terrain_index_grid[y, x])
                if x > 0:
                    G.add_edge(idx, idx - 1)
                if x < width - 1:
                    G.add_edge(idx, idx + 1)
                if y > 0:
                    G.add_edge(idx, idx - width)
                if y < height - 1:
                    G.add_edge(idx, idx + width)
        return G

    def get_subgraph_bfs(self, graph, max_depth):
        """Generate a subgraph using BFS from the player's current position."""
        start_node = self.node_index(self.environment.player.grid_x, self.environment.player.grid_y)
        bfs_tree = nx.bfs_tree(graph, source=start_node, depth_limit=max_depth)
        return graph.subgraph(bfs_tree.nodes()).copy()

    def add_entities_to_graph(self, graph):
        # Add entities logic
        return graph

    def convert_to_pyg(self, graph):
        # Convert NetworkX graph to PyTorch Geometric data object
        return from_networkx(graph)

    def visualize(self, graph):
        # self.print_graph(graph)
        pos = nx.get_node_attributes(graph, 'pos')
        plt.figure(figsize=(8, 6))
        nx.draw(graph, pos, with_labels=True, node_size=12, node_color='blue', cmap='viridis', edge_color='gray', width=0.5)
        plt.show()

    def print_graph(self, g):
        print(g.nodes(data=True))
        print(g.edges(data=True))

    def node_index(self, x, y):
        return x * self.environment.heightmap.shape[1] + y