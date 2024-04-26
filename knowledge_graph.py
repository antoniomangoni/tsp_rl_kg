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
        self.player_position = (self.environment.player.grid_x, self.environment.player.grid_y)

        print(self.environment.terrain_colour_map)
        self.terrain_graph = self.construct_terrain_graph()
        self.visualize(self.terrain_graph)

        self.subgraph = self.get_subgraph_bfs(self.terrain_graph, self.get_distance(completion))
        self.visualize(self.subgraph)

        self.graph = self.add_entities_to_graph(self.subgraph)
        self.visualize(self.graph)

        self.pyg_graph = self.convert_to_pyg(self.graph)  # Convert to PyTorch Geometric format

    def get_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        max_dimension = max(self.environment.heightmap.shape)
        if completion >= 1.0:
            return float('inf')  # Represents the entire graph
        return max(int(completion * max_dimension), self.vision_range)
    
    def construct_terrain_graph(self):
        G = nx.Graph()
        height, width = self.environment.heightmap.shape
        for y in range(height):
            for x in range(width):
                idx = x + y * width
                G.add_node(idx, type=0, pos=(x, y), feature=self.environment.terrain_index_grid[y, x])
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
        G = graph.copy()
        height, width = self.environment.heightmap.shape
        idx_offset = 10000  # Offset to distinguish entity nodes from terrain nodes
        player_terrain_idx = self.node_index(self.environment.player.grid_x, self.environment.player.grid_y) # Index of the terrain the player node is connected to
        player_idx = player_terrain_idx + idx_offset
        G.add_node(player_idx, type=2, pos=self.player_position, feature=0)  # Add the player node
        G.add_edge(player_idx, player_terrain_idx)  # Connect player node to the corresponding terrain node
        for y in range(height):
            for x in range(width):
                entity = self.environment.entity_index_grid[y, x]
                if entity != 0:  # Check if there is an entity
                    idx = self.node_index(x, y)  # Index of the current terrain node
                    entity_idx = idx + idx_offset
                    # Check and add entities only if they haven't been added to the graph yet
                    if entity_idx not in G.nodes():
                        # Add the entity node
                        G.add_node(entity_idx, type=1, pos=(x, y), feature=entity)
                        # Connect entity to the corresponding terrain node
                        G.add_edge(entity_idx, idx)
                        # Connect entity to the player node
                        G.add_edge(entity_idx, player_idx)
        return G

    def convert_to_pyg(self, graph):
        # Convert NetworkX graph to PyTorch Geometric data object
        return from_networkx(graph)

    def visualize0(self, graph):
        pos = nx.get_node_attributes(graph, 'pos')
        terrain_types = nx.get_node_attributes(graph, 'feature')
        node_types = nx.get_node_attributes(graph, 'type')

        node_colors = []
        for node in graph.nodes():
            node_type = node_types.get(node, 1)  # Default to type 2 if not found
            if node_type == 0:
                terrain_type = terrain_types.get(node, 0)  # Default terrain type if not found
                color = self.environment.terrain_colour_map.get(terrain_type, (255, 0, 0))  # Red if not found
                node_colors.append((color[0] / 255, color[1] / 255, color[2] / 255))
            elif node_type == 1:
                node_colors.append((0, 0, 0))  # Black for entities
            else:
                node_colors.append((1, 0, 0))  # Red for the player

        plt.figure(figsize=(8, 6))
        nx.draw(graph, pos, with_labels=False, node_size=12, node_color=node_colors, edge_color='gray', width=0.5)
        plt.show()

    def print_graph(self, g):
        print(g.nodes(data=True))
        print(g.edges(data=True))

    def node_index(self, x, y):
        return x * self.environment.heightmap.shape[1] + y
    
    def visualize_(self, graph):
        # Ensure every node has a position; provide a default if missing
        default_pos = (0, 0)  # Default to (0, 0) or any logical default
        pos = {node: (graph.nodes[node]['pos'] if 'pos' in graph.nodes[node] else default_pos) for node in graph.nodes()}
        
        plt.figure(figsize=(8, 6))
        nx.draw(graph, pos, with_labels=False, node_size=12, node_color='black', edge_color='gray', width=0.5)
        plt.show()