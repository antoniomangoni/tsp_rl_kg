import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class KnowledgeGraph:
    def __init__(self, environment, vision_range, completion=1.0):
        self.environment = environment
        self.terrain_array = environment.terrain_index_grid
        self.entity_array = environment.entity_index_grid
        self.vision_range = vision_range
        self.player_pos = (self.environment.player.grid_x, self.environment.player.grid_y)
        self.terrain_pos_idx_dict = {} # Dictionary to map terrain positions to their node indices
        self.player_idx = None
        self.max_terrain_nodes = self.terrain_array.size
        self.distance = self.get_distance(completion) # Graph distance
        self.terrain_node_type = 0
        self.entity_node_type = 1

        self.graph = Data()
        self.graph.x = torch.empty((0, 4), dtype=torch.float)  # Node features: node_type, type_id, x, y
        self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)  # Edge connections
        self.graph.edge_attr = torch.empty((0, 1), dtype=torch.float)  # Edge attributes: distance
        self.create_player_node()

        # print(self.environment.terrain_colour_map)
        self.add_terrain_nodes_to_graph()
        # self.print_graph()        
        self.visualise_graph()

    def get_node_features(self, coor, node_type):
        x, y = coor
        if node_type == self.terrain_node_type:
            return torch.tensor([[node_type, self.terrain_array[x, y], x, y]], dtype=torch.float)
        elif node_type == self.entity_node_type:
            assert self.entity_array[x, y] > 0, f"Invalid entity type at position ({x}, {y}), type: {self.entity_array[x, y]}, \n {self.entity_array}, \n {self.graph.x}"
            return torch.tensor([[node_type, self.entity_array[x, y], x, y]], dtype=torch.float)
        else:
            raise ValueError(f"Invalid node type: {node_type}")
    
    def create_player_node(self):
        player_features = self.get_node_features(self.player_pos, self.entity_node_type)
        self.graph.x = torch.cat([self.graph.x, player_features], dim=0)
        self.player_idx = len(self.graph.x) - 1

    def create_edge(self, node1, node2, distance):
        new_edge = torch.tensor([[node1, node2], [node2, node1]], dtype=torch.long).view(2, -1)
        new_attr = torch.tensor([[distance], [distance]], dtype=torch.float)
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge], dim=1)
        self.graph.edge_attr = torch.cat([self.graph.edge_attr, new_attr], dim=0)

    def create_entity_edge(self, node_idx, coor):
        terrain_idx = self.terrain_pos_idx_dict[coor]
        self.create_edge(node_idx, terrain_idx, distance=0)

        cart_distance = self.get_cartesian_distance(coor, self.player_pos)
        self.create_edge(node_idx, self.player_idx, cart_distance)

    def add_terrain_nodes_to_graph(self):
        player_x, player_y = self.player_pos
        terrain_nodes_to_calculate_edges = []

        for y in range(player_y - self.distance, player_y + self.distance + 1):
            for x in range(player_x - self.distance, player_x + self.distance + 1):
                # Skip if the node is out of bounds or already added
                if not self.environment.within_bounds(x, y):
                    continue
                # Add the terrain node if it does not exist
                if (x, y) not in self.terrain_pos_idx_dict:
                    node_index = self.add_terrain_node((x, y), connect=False)
                    # Add node index to list of terrain nodes to calculate edges for
                    terrain_nodes_to_calculate_edges.append((node_index, (x, y)))

                if self.entity_array[x, y] > 0:
                    self.add_entity_node((x, y))

        # Calculate edges for all deferred nodes
        self.finalize_graph_edges(terrain_nodes_to_calculate_edges)

        # Print connections for verification
        # self.print_graph_connections()

    def add_terrain_node(self, position, connect=True):
        if position not in self.terrain_pos_idx_dict:
            new_node_index = self.create_node(self.terrain_node_type, position)
            self.terrain_pos_idx_dict[position] = new_node_index

            if connect == True:
                self.create_terrain_edges(new_node_index, position)
            return new_node_index
    
    def finalize_graph_edges(self, t_nodes_to_calculate_edges):
        for node_index, position in t_nodes_to_calculate_edges:
            self.create_terrain_edges(node_index, position)

    def create_terrain_edges(self, terrain_idx, coor):
        x, y = coor
        neighbours = self.get_manhattan_neighbours(x, y)
        for neighbour in neighbours:
            if neighbour in self.terrain_pos_idx_dict:
                neighbour_idx = self.terrain_pos_idx_dict[neighbour]
                self.create_edge(terrain_idx, neighbour_idx, distance=1)

    def add_entity_node(self, position):
        entity_idx = self.create_node(self.entity_node_type, position)
        self.create_entity_edge(entity_idx, position)

    def create_node(self, node_type, position):
        features = self.get_node_features(position, node_type)
        self.graph.x = torch.cat([self.graph.x, features], dim=0)
        return len(self.graph.x) - 1
    
    def add_entity_nodes(self):
        for y in range(self.entity_array.shape[0]):
            for x in range(self.entity_array.shape[1]):
                # Skip if the node is out of bounds or already added
                if not self.environment.within_bounds(x, y):
                    continue
                if (x, y) == self.player_pos:
                    continue
                # Add entity if one exists in the array (non-zero value, 1 is a fish which is also not included)
                if self.entity_array[x, y] > 0:
                    self.add_entity_node((x, y))
    
    def get_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        completion = min(completion, 1)
        return max(int(completion * self.terrain_array.shape[0]), self.vision_range)

    def get_cartesian_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])	
    
    def get_manhattan_neighbours(self, x, y):
        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if self.environment.within_bounds(new_x, new_y):
                neighbours.append((new_x, new_y))
        return neighbours
    
    def visualise_graph(self, node_size=100, edge_color="tab:gray", show_ticks=True):
        # These colors are in RGB format, normalized to [0, 1] --> green, grey twice, red, brown, black
        entity_colour_map = {2: (0.13, 0.33, 0.16), 3: (0.61, 0.65, 0.62),4: (0.61, 0.65, 0.62),
                             5: (0.78, 0.16, 0.12), 6: (0.46, 0.31, 0.04), 7: (0, 0, 0)}
        
        # Convert to undirected graph for visualization
        G = to_networkx(self.graph, to_undirected=True)
        
        # Use a 2D spring layout, as z-coordinates are manually assigned
        pos = nx.spring_layout(G, seed=42)  # 2D layout
        node_colors = []

        for node in G.nodes():
            node_type = self.graph.x[node][0].item()
            # The node features are [Node Type, Entity/Terrain Type, X Pos, Y Pos] and the z-coordinate is the node type
            x, y, z = self.graph.x[node][2].item(), self.graph.x[node][3].item(), self.graph.x[node][0].item()

            # Assign z-coordinate based on node type
            z = 0 if node_type == self.terrain_node_type else 1
            pos[node] = (x, y, z)  # Update position to include z-coordinate
            
            # Set node color based on node type
            if node_type == self.terrain_node_type:
                terrain_type = int(self.graph.x[node][1].item())
                color = self.environment.terrain_colour_map.get(terrain_type, (255, 0, 0))
                node_colors.append([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0])
            elif node_type == self.entity_node_type:
                entity_type = int(self.graph.x[node][1].item())
                color = entity_colour_map[int(self.graph.x[node][1])]
                if color is None:
                    color = entity_colour_map[3]
                    print(f"Entity type {entity_type} not found in colour map. now it is {color}")
                    
                node_colors.append(color)  # Directly use the color name

        # assert (pos[0][0], pos[0][1]) == self.player_pos and pos[0][2] == 1, "Player position does not match the graph position"
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        try:
            node_colors = np.array(node_colors)
        except ValueError:
            print("Printing node colors")
            print(node_colors)
        # Scatter plot for nodes
        ax.scatter(*node_xyz.T, s=node_size, color=node_colors, edgecolor='w', depthshade=True)
        # Draw edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color=edge_color)

        # Configure axis visibility and ticks
        if show_ticks:
            # Set tick labels based on the data range
            ax.set_xticks(np.linspace(min(pos[n][0] for n in G.nodes()), max(pos[n][0] for n in G.nodes()), num=5))
            ax.set_yticks(np.linspace(min(pos[n][1] for n in G.nodes()), max(pos[n][1] for n in G.nodes()), num=5))
            ax.set_zticks([0, 1])  # Only two levels: 0 for terrain, 1 for entities
        else:
            ax.grid(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.zaxis.set_ticks([])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Terrain/Entity")
        plt.title("Game World")
        plt.show()

    def print_graph_connections(self):
        edge_index_np = self.graph.edge_index.numpy()  # Convert edge_index to a numpy array for easier handling
        for i in range(0, edge_index_np.shape[1], 2):  # Step by 2 to handle bi-directional edges
            print(f"Node {edge_index_np[0, i]} -> Node {edge_index_np[1, i]}")

        # get the node features
        print(self.graph.x)