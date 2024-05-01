import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class KG:
    def __init__(self, environment, vision_range, completion=1.0):
        self.environment = environment
        self.terrain_array = environment.terrain_index_grid
        self.entity_array = environment.entity_index_grid
        self.vision_range = vision_range
        self.player_pos = (self.environment.player.grid_x, self.environment.player.grid_y)
        print(f"Player position: {self.player_pos}")
        self.terrain_pos_list = [self.player_pos]  # Start by keeping track of the initial terrain position
        self.max_terrain_nodes = self.terrain_array.size
        self.distance = self.get_distance(completion) # Graph distance
        self.terrain_node_type = 0
        self.entity_node_type = 1
        # Initialize nodes with initial features including positions
        # Assuming: [Node Type, Entity/Terrain Type, X Pos, Y Pos, Additional Feature 1 (e.g., Zero-padding), Additional Feature 2 (e.g., Zero-padding)]
        player_node_features = self.get_node_features(self.player_pos, self.entity_node_type)  # Player node
        terrain_node_features = self.get_node_features(self.player_pos, self.terrain_node_type)  # Initial terrain node
        # Initialize edges between the player and the initial terrain
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Create the graph with initial nodes and edges
        self.graph = Data(x=torch.cat([player_node_features, terrain_node_features], dim=0),
                          edge_index=edge_index)
        
        # print(self.environment.terrain_colour_map)
        self.add_terrain_to_graph()
        # self.print_graph()        
        self.visualise_graph()

    def get_node_features(self, coor, node_type):
        x, y = coor
        if node_type == self.terrain_node_type:
            return torch.tensor([[node_type, self.terrain_array[y, x], x, y]], dtype=torch.float)
        elif node_type == self.entity_node_type:
            return torch.tensor([[node_type, self.entity_array[y, x], x, y]], dtype=torch.float)
        else:
            raise ValueError(f"Invalid node type: {node_type}")
        
    def add_terrain_node(self, position, wait=False):
        if position not in self.terrain_pos_list:
            new_terrain_features = self.get_node_features(position, self.terrain_node_type)
            self.graph.x = torch.cat([self.graph.x, new_terrain_features], dim=0)
            new_node_index = len(self.graph.x) - 1

            self.terrain_pos_list.append(position)

            if self.environment.entity_index_grid[position[1], position[0]] != 0:
                self.add_entity_node(new_node_index, position)

            if wait == False:
                self.create_terrain_edges(new_node_index, position)
            else:
                return new_node_index, position
    
    def create_edge(self, node1, node2):
        new_edge = torch.tensor([[node1, node2], [node2, node1]], dtype=torch.long).view(2, -1)
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge], dim=1)
    
    def create_terrain_edges(self, terrain_idx, coor):
        x, y = coor
        neighbours = self.environment.get_neighbours(x, y)
        for neighbour in neighbours:
            if neighbour in self.terrain_pos_list:
                neighbour_idx = self.terrain_pos_list.index(neighbour)
                self.create_edge(terrain_idx, neighbour_idx)

    def add_entity_node(self, terrain_idx, position):
        entity_features = self.get_node_features(position, self.entity_node_type)
        self.graph.x = torch.cat([self.graph.x, entity_features], dim=0)
        entity_idx = len(self.graph.x) - 1

        # Format the new edges and concatenate to the graph
        new_edges = torch.tensor([[entity_idx, terrain_idx], [terrain_idx, entity_idx], [entity_idx, 0], [0, entity_idx]], dtype=torch.long).view(2, -1)
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)

        # Debugging: Print connections of the entity node
        print(f"Entity Node {entity_idx} connections: Terrain {terrain_idx}, Player 0")


    def get_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        completion = min(completion, 1)
        return max(int(completion * self.terrain_array.shape[0]), self.vision_range)
        
    def finalize_graph_edges(self, t_nodes_to_calculate_edges):
        for node_index, position in t_nodes_to_calculate_edges:
            self.create_terrain_edges(node_index, position)

    def add_terrain_to_graph(self):
        player_x, player_y = self.player_pos
        terrain_nodes_to_calculate_edges = []

        for y in range(player_y - self.distance, player_y + self.distance + 1):
            for x in range(player_x - self.distance, player_x + self.distance + 1):
                if not self.environment.within_bounds(x, y):
                    continue
                if (x, y) not in self.terrain_pos_list:
                    node_index, _ = self.add_terrain_node((x, y), wait=True)
                    terrain_nodes_to_calculate_edges.append((node_index, (x, y)))

        # Calculate edges for all deferred nodes
        self.finalize_graph_edges(terrain_nodes_to_calculate_edges)

        # Print connections for verification
        self.print_graph_connections()

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
                color = entity_colour_map.get(entity_type)  # Default to grey if type not found
                if color is None:
                    color = entity_colour_map[3]
                    print(f"Entity type {entity_type} not found in colour map. now it is {color}")
                    
                node_colors.append(color)  # Directly use the color name

            
        print(f"Player node at position ({pos[0][0]}, {pos[0][1]}, {pos[0][2]})")
        assert (pos[0][0], pos[0][1]) == self.player_pos and pos[0][2] == 1, "Player position does not match the graph position"
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
        ax.set_zlabel("Z")
        plt.title("3D Graph Visualization")
        plt.show()

    def print_graph_connections(self):
        edge_index_np = self.graph.edge_index.numpy()  # Convert edge_index to a numpy array for easier handling
        for i in range(0, edge_index_np.shape[1], 2):  # Step by 2 to handle bi-directional edges
            print(f"Node {edge_index_np[0, i]} -> Node {edge_index_np[1, i]}")

        # get the node features
        print(self.graph.x)