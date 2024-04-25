import torch
import torch_geometric as pyg
from torch_geometric.utils import subgraph, to_networkx, from_networkx
import networkx as nx
import matplotlib.pyplot as plt
from terrains import Terrain

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, environment, vision_range, completion=1.0):
        super(KnowledgeGraph, self).__init__()
        self.environment = environment
        self.vision_range = vision_range
        self.completion = completion

        self.terrain_graph = self.construct_terrain_graph()
        if not isinstance(self.terrain_graph, pyg.data.Data):
            raise TypeError("Terrain graph is not a PyG data object")

        self.visualize(self.terrain_graph)
        
        radius = self.get_radius(completion)
        self.subgraph = self.get_terrain_subgraph(self.terrain_graph, radius)
        if self.subgraph is None:
            print("No valid subgraph was created; subgraph is None")
            return
        elif not isinstance(self.subgraph, pyg.data.Data):
            print(f"Error: Subgraph is not a PyG data object, it is of type: {type(self.subgraph)}")
            raise TypeError("Subgraph is not a PyG data object")

        self.visualize(self.subgraph)
        
        self.graph = self.add_entities_to_graph()
        if not isinstance(self.graph, pyg.data.Data):
            raise TypeError("Graph after adding entities is not a PyG data object")
        
        self.visualize(self.graph)

    def get_radius(self, completion):
        """Calculates the effective radius for subgraph extraction."""
        max_dimension = max(self.environment.heightmap.shape)
        print(f"Max dimension: {max_dimension}")
        if completion >= 1.0:
            return float('inf')  # Represents the entire graph
        return max(completion * max_dimension, self.vision_range)

    def construct_terrain_graph(self):
        """Constructs a graph for the entire terrain indices with Manhattan neighborhood connections."""
        height, width = self.environment.heightmap.shape
        terrain_features = self.environment.terrain_index_grid.flatten()
        node_features = torch.tensor(terrain_features, dtype=torch.float).view(-1, 1)

        # Prepare to collect edges
        edges = []

        # Initialize positional data for the nodes
        pos = []

        # Connect each node to its Manhattan neighbors (up, down, left, right)
        for y in range(height):
            for x in range(width):
                idx = x + y * width
                pos.append([x, y])
                if x > 0:  # left
                    edges.append((idx, idx - 1))
                if x < width - 1:  # right
                    edges.append((idx, idx + 1))
                if y > 0:  # up
                    edges.append((idx, idx - width))
                if y < height - 1:  # down
                    edges.append((idx, idx + width))

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        pos = torch.tensor(pos, dtype=torch.float)

        return pyg.data.Data(x=node_features, edge_index=edge_index, pos=pos)

    def get_terrain_subgraph(self, data, radius):
        if radius == float('inf'):
            return data.copy()

        center_idx = self.node_index(self.environment.player.grid_x, self.environment.player.grid_y)
        center_position = data.pos[center_idx]
        distances = torch.norm(data.pos - center_position, dim=1)
        mask = distances <= radius
        subgraph_nodes = mask.nonzero(as_tuple=True)[0]

        if subgraph_nodes.size(0) == 0:
            print("No nodes found within the specified radius.")
            return None

        subgraph_nodes, sub_edge_index = subgraph(subgraph_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        sub_node_features = data.x[subgraph_nodes]
        return pyg.data.Data(x=sub_node_features, edge_index=sub_edge_index)

    def add_entities_to_graph(self):
        """Adds entities to the subgraph, connecting each to their nearest terrain node and the player."""
        features = self.environment.entity_index_grid.flatten()
        entity_features = torch.tensor(features, dtype=torch.float).view(-1, 1)
        num_existing_nodes = self.subgraph.x.shape[0]
        self.subgraph.x = torch.cat([self.subgraph.x, entity_features], dim=0)
        # Define additional logic to create edges
        return self.subgraph

    def visualize(self, graph):
        """Visualises the graph using NetworkX."""
        G = to_networkx(graph)
        # use terrain.get_colour() to get the colour of the terrain
        # node_color = [self.environment.terrain_colour_map(int(node)) for node in graph.x.flatten()]
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=12, node_color='blue', cmap='viridis', edge_color='gray', width=0.5)
        plt.show()

    def node_index(self, x, y):
        """Converts grid coordinates to node index."""
        return x * self.environment.heightmap.shape[1] + y
