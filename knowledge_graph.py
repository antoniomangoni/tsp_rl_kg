import networkx as nx
import numpy as np
from numpy.linalg import norm
import torch_geometric as pyg
import torch
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self, environment, vision_range, completion=1.0):
        self.environment = environment
        print(self.environment.terrain_index_grid)
        print(self.environment.entity_index_grid)
        self.vision_range = vision_range
        self.completion = completion
        self.player_position = (self.environment.player.grid_x, self.environment.player.grid_y)
        
        # Construct the initial terrain graph and convert to PyG format
        self.terrain_graph = self.construct_terrain_graph()
        # Extract the subgraph based on the player's position and vision range / completion
        self.subgraph = self.get_subgraph_bfs(self.terrain_graph, self.get_distance(completion))
        self.graph = self.convert_to_pyg(self.subgraph)  # Convert to PyTorch Geometric format
        
        # Add player to the graph
        self.player_idx = self.add_player_to_graph()
        # Add entities to the graph
        self.add_entities_to_graph(self.graph)
        
        # Visualize the graph with player
        self.visualise(self.graph)

    def add_player_to_graph(self):
        player_idx = self.node_index(self.environment.player.grid_x, self.environment.player.grid_y)
        # Ensure the graph.x tensor exists and has a size method
        if self.graph.x is not None and player_idx >= self.graph.x.size(0):
            player_feature = torch.tensor([0], dtype=torch.float)  # Example feature for player
            self.graph.x = torch.cat([self.graph.x, player_feature.view(1, 1)], dim=0)
            player_edge = torch.tensor([[player_idx], [player_idx]], dtype=torch.long)
            self.graph.edge_index = torch.cat([self.graph.edge_index, player_edge], dim=1)
        return player_idx

    def move_player_in_graph(self, new_x, new_y):
        new_player_idx = self.node_index(new_x, new_y)
        if new_player_idx not in self.graph.x.size(0):
            new_player_feature = torch.tensor([0], dtype=torch.float)

    def add_entities_to_graph(self):
        # Assuming `self.graph` is already initialized as a PyTorch Geometric `Data` object
        node_features = self.graph.x.tolist()  # Convert existing node features to list
        edges = self.graph.edge_index.t().tolist()  # Convert edge indices to list of pairs

        height, width = self.environment.height, self.environment.width
        idx_offset = max(self.environment.height * self.environment.width, self.graph.num_nodes)  # Offset to distinguish entity nodes from terrain nodes

        for y in range(height):
            for x in range(width):
                entity = self.environment.entity_index_grid[y, x]
                if entity != 0:  # Check if there is an entity
                    idx = self.node_index(x, y)  # Index of the current terrain node
                    entity_idx = idx + idx_offset
                    # Add the entity node
                    node_features.append([entity])  # Assume entity value is used as the node feature
                    # Connect entity to the corresponding terrain node
                    edges.append([entity_idx, idx])
                    edges.append([idx, entity_idx])  # For undirected graph, add both directions

        # Convert lists back to tensor
        self.graph.x = torch.tensor(node_features, dtype=torch.float)
        self.graph.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    def add_entities_to_graph(self, graph):
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                entity_type = self.environment.entity_index_grid[y, x]
                if entity_type != 0:
                    self.add_entity_to_graph(x, y, entity_type)

    def get_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        max_dimension = max(self.environment.heightmap.shape)
        if completion >= 1.0:
            return float('inf')  # Represents the entire graph
        return max(int(completion * max_dimension), self.vision_range)
    
    def construct_terrain_graph(self):
        G = nx.Graph()
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                idx = x + y * self.environment.width
                G.add_node(idx, type=0, pos=(x, y), feature=self.environment.terrain_index_grid[y, x])
                if x > 0:
                    G.add_edge(idx, idx - 1)
                if x < self.environment.width - 1:
                    G.add_edge(idx, idx + 1)
                if y > 0:
                    G.add_edge(idx, idx - self.environment.width)
                if y < self.environment.height - 1:
                    G.add_edge(idx, idx + self.environment.width)
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
        print(graph.nodes(data=True))
        print(graph.edges(data=True))

        features = [data.get('feature', [0]) for _, data in graph.nodes(data=True)]
        node_features = torch.tensor(features, dtype=torch.float)

        edge_index = []
        for start_node, end_node in graph.edges():
            edge_index.append([start_node, end_node])
            edge_index.append([end_node, start_node])  # Because it's undirected

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return pyg.data.Data(x=node_features, edge_index=edge_index_tensor)

    def visualize(self, graph):
        graph = graph.to_networkx()
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