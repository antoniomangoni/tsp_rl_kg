from graph_idx_manager import IDX_Manager

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class KnowledgeGraph():
    def __init__(self, environment, vision_range, completion=1.0):
        self.environment = environment
        self.terrain_array = environment.terrain_index_grid
        self.entity_array = environment.entity_index_grid

        self.player_pos = (self.environment.player.grid_x, self.environment.player.grid_y)
        self.entity_array[self.player_pos] = 0  # Remove the player from the entity array

        assert max(self.entity_array.flatten()) < 7, "Entity type exceeds the maximum value of 6"

        self.idx_manager = IDX_Manager()

        self.vision_range = vision_range
        self.discovered_terrain = np.zeros_like(self.terrain_array)
        self.distance = self.get_graph_distance(completion) # Graph distance, in terms of edges, from the player node
        self.terrain_z_level = 0
        self.entity_z_level = 1
        self.player_z_level = 2

        self.init_full_graph()
        self.add_terrain_nodes_to_graph()
        self.create_player_node()
        self.create_player_edges()
        self.verify_graph()
        
    def get_node_features(self, coor, z_level):
        x, y = coor
        if z_level == self.terrain_z_level:
            return torch.tensor([[x, y, z_level, self.terrain_array[x, y], self.discovered_terrain[x, y]]], dtype=torch.float)
        elif z_level == self.entity_z_level:
            assert self.entity_array[x, y] > 0, f"Invalid entity type at position ({x}, {y}), type: {self.entity_array[x, y]}, \n {self.entity_array}, \n {self.graph.x}"
            return torch.tensor([[x, y, z_level, self.entity_array[x, y], self.discovered_terrain[x, y]]], dtype=torch.float)
        else:
            raise ValueError(f"Invalid z level: {z_level}")
        
    def init_full_graph(self):
        num_possible_nodes = self.environment.width * self.environment.height * 2 + 1  # 2 z-levels and a player node
        num_possible_edges = self.compute_total_possible_edges()
        feature_size = 4 # x, y, z_level, type_id, mask
        edge_attr_size = 1 # distance
        self.graph = Data(
            x=torch.zeros((num_possible_nodes, feature_size)),  # Initialize with zero
            edge_index=torch.zeros((2, num_possible_edges), dtype=torch.long),
            edge_attr=torch.zeros((num_possible_edges, edge_attr_size))
        )
        self.populate_possible_edges()  # Predefine all possible edges

    def activate_node(self, coordinates, z_level):
        """ Activates a node by setting its features. """
        features = self.get_node_features(coordinates, z_level)
        node_idx = self.idx_manager.get_idx(coordinates, z_level)
        self.graph.x[node_idx] = torch.tensor(features, dtype=torch.float)

    def activate_edge(self, edge_idx, attributes):
        """ Activates an edge by setting its attributes. """
        self.graph.edge_attr[edge_idx] = torch.tensor(attributes, dtype=torch.float)

    def populate_possible_edges(self):
        """ This method would calculate and set all possible edges based on your environment's topology. """
        # Example logic here

    def compute_total_possible_edges(self):
        """ This method would calculate the total number of possible edges based on the environment's topology. 
        This only works for rectangles.
        All terrain edges are connected in a manhattan neighbourhood up to 4 other nodes (up, down, left, right)
        except for the edges of the map
        """
        edges = (self.environment.width * (self.environment.height - 1)) + (self.environment.height * (self.environment.width - 1))
        for x in range(self.environment.width - 1):
            for y in range(self.environment.height):
                if self.environment.entity_index_grid[x, y] > 0:
                    # One edge to the terrain node and one to the player node
                    edges += 2

    def create_player_node(self):
        player_features = torch.tensor([[self.player_z_level, self.environment.player.id, self.player_pos[0], self.player_pos[1]]], dtype=torch.float)
        self.graph.x = torch.cat([self.graph.x, player_features], dim=0)
        self.idx_manager.player_idx = self.idx_manager.create_idx(self.player_pos, self.player_z_level)
        print(f"Player node created at position {self.player_pos} with index {self.idx_manager.player_idx}")

    def create_player_edges(self):
        print("Creating player edges")
        for id, idx in self.idx_manager.id_idx_dict.items():
            if id == None or idx == None:
                print(f"Invalid id or idx: {id}, {idx}")
                continue
            if idx == self.idx_manager.player_idx or id[1] != self.entity_z_level:
                continue
            distance = self.get_cartesian_distance(self.player_pos, id[0])
            self.create_edge(self.idx_manager.player_idx, idx, distance)

    def recaluclate_player_edges(self):
        player_idx = self.idx_manager.player_idx
        for id, idx in self.idx_manager.id_idx_dict.items():
            if idx == player_idx:
                continue
            distance = self.get_cartesian_distance(self.player_pos, id[0])
            mask = ((self.graph.edge_index[0] == player_idx) & (self.graph.edge_index[1] == idx)) | \
                ((self.graph.edge_index[1] == player_idx) & (self.graph.edge_index[0] == idx))
            self.graph.edge_attr[mask] = distance
        
    def move_player_node(self, new_x, new_y):
        self.player_pos = (new_x, new_y)
        print(f'There are {self.graph.num_nodes} nodes in the graph')
        print(f'Player node is at position {self.player_pos} with index {self.idx_manager.player_idx}')
        self.idx_manager.idx_id_dict[self.idx_manager.player_idx] = (self.player_pos, self.player_z_level)
        new_features = torch.tensor([[self.player_z_level, self.environment.player.id, new_x, new_y]], dtype=torch.float)
        self.graph.x[self.idx_manager.player_idx] = new_features
        self.recaluclate_player_edges()

    def create_node(self, z_level, position):
        features = self.get_node_features(position, z_level)
        self.graph.x = torch.cat([self.graph.x, features], dim=0)
        idx = self.idx_manager.create_idx(position, z_level)
        return idx

    def create_edge(self, node1_idx, node2_idx, distance):
        if self.idx_manager.has_edge(node1_idx, node2_idx):
            # print(f"Edge between {node1_idx} and {node2_idx} already exists.")
            return
        print(f"Creating edge between {node1_idx} and {node2_idx} with distance {distance}")
        new_edge = torch.tensor([[node1_idx, node2_idx], [node2_idx, node1_idx]], dtype=torch.long)
        new_attr = torch.tensor([[distance], [distance]], dtype=torch.float)
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge], dim=1)
        self.graph.edge_attr = torch.cat([self.graph.edge_attr, new_attr], dim=0)
        self.idx_manager.add_edge(node1_idx, node2_idx, attr={'distance': distance})

    def remove_entity_node(self, position):
        entity_idx = self.idx_manager.get_idx(position, self.entity_z_level)
        if entity_idx is None:
            print(f"No entity node at position {position} with z_level {self.entity_z_level}")
            return False

        # Remove edges from PyTorch Geometric graph
        edges_to_remove = []
        for i, (start, end) in enumerate(self.graph.edge_index.t()):
            if entity_idx in [start.item(), end.item()]:
                edges_to_remove.append(i)

        if edges_to_remove:
            keep_edges = [i for i in range(self.graph.edge_index.size(1)) if i not in edges_to_remove]
            self.graph.edge_index = self.graph.edge_index[:, keep_edges]
            self.graph.edge_attr = self.graph.edge_attr[keep_edges]

        # Adjust graph data structures for node removal
        node_mask = torch.arange(self.graph.num_nodes) != entity_idx
        self.graph.x = self.graph.x[node_mask]

        # Adjust edge indices to account for removed node
        adjustment_mask = self.graph.edge_index >= entity_idx
        self.graph.edge_index[adjustment_mask] -= 1

        # Also remove the node and edges from IDX_Manager
        self.idx_manager.remove_idx(position, self.entity_z_level)
        return True

    def add_terrain_node(self, position, connect=True):
        new_node_index = self.create_node(self.terrain_z_level, position)
        if connect == True:
            self.create_terrain_edges(new_node_index, position)
        return new_node_index

    def add_terrain_nodes_to_graph(self):
        player_x, player_y = self.player_pos
        terrain_nodes_to_calculate_edges = []

        for y in range(player_y - self.distance, player_y + self.distance + 1):
            for x in range(player_x - self.distance, player_x + self.distance + 1):
                # Skip if the node is out of bounds
                if not self.environment.within_bounds(x, y):
                    continue
                self.discovered_terrain[x, y] = 1
                # Add the terrain node if it does not exist
                if not (self.idx_manager.get_idx((x, y), self.terrain_z_level)):    
                    # print(f"[add_terrain_nodes_to_graph()] Adding terrain node at position ({x}, {y})")
                    terrain_node_index = self.add_terrain_node((x, y), connect=False)
                    # Add node index to list of terrain nodes to calculate edges for
                    terrain_nodes_to_calculate_edges.append((terrain_node_index, (x, y)))

                if self.entity_array[x, y] > 0:
                    self.add_entity_node((x, y), terrain_node_index)

        # Calculate edges for all deferred nodes
        self.connect_all_terrain_edges(terrain_nodes_to_calculate_edges)

    def add_entity_node(self, position):
        terrain_idx = self.idx_manager.get_idx(position, self.terrain_z_level)
        entity_idx = self.create_node(self.entity_z_level, position)
        self.create_edge(entity_idx, terrain_idx, distance=0)

    def add_entity_nodes(self):
        for y in range(self.entity_array.shape[0]):
            for x in range(self.entity_array.shape[1]):
                # Skip if the node is out of bounds or already added
                if not self.environment.within_bounds(x, y):
                    continue
                if self.entity_array[x, y] > 0:
                    terrain_idx = self.idx_manager.get_idx((x, y), self.terrain_z_level)
                    if terrain_idx is not None:
                        self.add_entity_node((x, y), terrain_idx)

    def add_path_node(self, position):
        assert self.entity_array[position] == 6, f"Path id is not 6 at position {position}, but {self.entity_array[position]}"
        path_idx = self.create_node(self.entity_z_level, position)
        self.create_edge(path_idx, self.idx_manager.get_idx(position, self.terrain_z_level), distance=0)

    def create_terrain_edges(self, terrain_idx, coor):
        neighbours = self.get_manhattan_neighbours(coor)
        for neighbour in neighbours:
            neighbour_idx =  self.idx_manager.get_idx(neighbour, self.terrain_z_level)
            if neighbour_idx is not None:
                self.create_edge(terrain_idx, neighbour_idx, distance=1)
    
    def connect_all_terrain_edges(self, t_nodes_to_calculate_edges):
        for node_index, position in t_nodes_to_calculate_edges:
            self.create_terrain_edges(node_index, position)
    
    def get_graph_distance(self, completion):
        """Calculates the effective distance for subgraph extraction."""
        completion = min(completion, 1)
        return max(int(completion * self.terrain_array.shape[0]), self.vision_range)

    def get_cartesian_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])	

    def get_manhattan_neighbours(self, coor):
        x, y = coor
        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if self.environment.within_bounds(new_x, new_y):
                neighbours.append((new_x, new_y))
        return neighbours
    
    def landfill_node(self, x, y):
        node_idx = self.idx_manager.get_idx((x, y), self.terrain_z_level)
        # get Node features: z_level, type_id, x, y
        node_features = self.graph.x[node_idx]
        # Add 1 to the terrain type to indicate land fill
        node_features[1] += 1
        self.graph.x[node_idx] = node_features
        print(f"Landfill node at position ({x}, {y}) with index {node_idx}")
        self.visualise_graph()

    def verify_player_node_connections(self):
        player_edges = [e for e in self.graph.edge_index.t() if self.idx_manager.player_idx in e]
        unique_connected_nodes = set(e[1] if e[0] == self.idx_manager.player_idx else e[0] for e in player_edges)

        num_of_entities = sum(1 for features in self.graph.x if features[0] == self.entity_z_level)
        if len(unique_connected_nodes) != num_of_entities * 2:
            print(f"Player node is missing edges to entities. Expected {num_of_entities}, found {len(unique_connected_nodes)}.")
            return False
        return True

    def verify_terrain_node_connections(self, position):
        terrain_idx = self.idx_manager.get_idx(position, self.terrain_z_level)
        if terrain_idx is None:
            print(f"No terrain node at position {position}, verification skipped.")
            return

        neighbours = self.get_manhattan_neighbours(position)
        count = 0
        for neighbour in neighbours:
            neighbour_idx = self.idx_manager.get_idx(neighbour, self.terrain_z_level)
            if neighbour_idx is not None and self.idx_manager.has_edge(terrain_idx, neighbour_idx):
                count += 1

        if count != len(neighbours):
            print(f"Node at position {position} is missing edges to neighbours. Expected {len(neighbours)}, found {count}.")
            return False
        return True
        
    def verify_entities_node_connections(self):
        for i in range(self.graph.num_nodes):
            if self.graph.x[i][0] == 1:
                position = (int(self.graph.x[i][2].item()), int(self.graph.x[i][3].item()))
                # Check if the entity node has an edge to the terrain node of the same position
                terrain_idx = self.idx_manager.get_idx(position, self.terrain_z_level)
                if terrain_idx is None:
                    print(f"No terrain node at position {position}, verification skipped.")
                    return
                if not self.idx_manager.has_edge(i, terrain_idx):
                    print(f"Entity node at position {position} is missing an edge to the terrain node.")
                    return False
                # Check if entity node has an edge to the player node
                if not self.idx_manager.has_edge(i, self.idx_manager.player_idx):
                    print(f"Entity node at position {position} is missing an edge to the player node.")
                    return False
                # Check that the entity node has 2 edges
                edges = [self.idx_manager.get_edge_attr(i, idx) for idx in range(self.graph.num_nodes) if self.idx_manager.has_edge(i, idx)]
                if len(edges) != 2:
                    print(f"Entity node at position {position} has {len(edges)} edges, expected 2.")
                    return False
        return True    
            
    def verify_graph(self):
        break_flag = False
        if not self.verify_player_node_connections():
            print("Player node verification failed.")
            break_flag = True
        if not self.verify_entities_node_connections():
            print("Entity node verification failed.")
            break_flag = True
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                if self.discovered_terrain[x, y] == 1:
                    if not self.verify_terrain_node_connections((x, y)):
                        print(f"Terrain node verification failed at position ({x}, {y}).")
                        break_flag = True
                        continue
        if break_flag:
            print("Graph verification failed.")
            self.visualise_graph()
            exit(1)
        print("Graph verification passed.")
        return
    
    def print_graph(self):
        print(self.graph)
        print(self.idx_manager.idx_id_dict)
        print(self.idx_manager.id_idx_dict)
        print(self.idx_manager.edges)
        print(self.idx_manager.edge_attrs)

    def visualise_graph(self, node_size=100, edge_color="tab:gray", show_ticks=True):
        # These colors are in RGB format, normalized to [0, 1] --> green, grey twice, red, brown, black
        entity_colour_map = {2: (0.13, 0.33, 0.16), 3: (0.61, 0.65, 0.62),4: (0.61, 0.65, 0.62),
                             5: (0.78, 0.16, 0.12), 6: (0.46, 0.31, 0.04), 7: (0, 0, 0)}
        
        # Convert to undirected graph for visualization
        G = to_networkx(self.graph, to_undirected=True)
        print("[visualise_graph()] The size of the graph is: ", self.graph.num_nodes)
        print("[visualise_graph()] The size of the networkx graph is: ", len(G.nodes))
        # Use a 2D spring layout, as z-coordinates are manually assigned
        pos = {} # nx.spring_layout(G, seed=42)  # 2D layout
        node_colors = []

        for node in G.nodes():
            if node < len(self.graph.x):  # Ensure the node index is within the current graph size
                x, y, z = self.graph.x[node][2].item(), self.graph.x[node][3].item(), self.graph.x[node][0].item()
                pos[node] = (x, y, z)
                if z == self.terrain_z_level:
                    terrain_type = int(self.graph.x[node][1].item())
                    color = self.environment.terrain_colour_map.get(terrain_type, (255, 0, 0))
                    node_colors.append([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0])
                elif z == self.entity_z_level:
                    color = entity_colour_map[int(self.graph.x[node][1])]
                    node_colors.append(color)
                elif z == self.player_z_level:
                    color = (0, 0, 0)  # Black for player
                    # color = entity_colour_map[self.environment.player.id]   
                    node_colors.append(color)
            else:
                print(f"Invalid node index: {node}")

        # Visualization continues if positions and node colors are valid
        if pos and node_colors:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # invert x-axis to match the game world
            ax.invert_xaxis()
            node_xyz = np.array([pos[v] for v in sorted(G) if v in pos])  # Only include valid nodes
            edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges() if u in pos and v in pos])

            ax.scatter(*node_xyz.T, s=node_size, color=node_colors, edgecolor='w', depthshade=True)
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color=edge_color)

            if show_ticks:
                ax.set_xticks(np.linspace(min(pos[n][0] for n in pos), max(pos[n][0] for n in pos), num=5))
                ax.set_yticks(np.linspace(min(pos[n][1] for n in pos), max(pos[n][1] for n in pos), num=5))
                ax.set_zticks([0, 1])  # Assuming z-levels are either 0 or 1
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