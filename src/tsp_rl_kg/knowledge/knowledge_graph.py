import torch
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
from tsp_rl_kg.knowledge.graph_idx_manager import Graph_Manager


class KnowledgeGraph():
    def __init__(self, environment, vision_range, completion=1.0, plot=False):
        self.environment = environment
        self.terrain_array = environment.terrain_index_grid
        self.entity_array = environment.entity_index_grid

        self.player_pos = (self.environment.player.grid_x, self.environment.player.grid_y)
        self.entity_array[self.player_pos] = 0  # Remove the player from the entity array

        assert max(self.entity_array.flatten()) < 7, "Entity type exceeds the maximum value of 6"

        self.graph_manager = Graph_Manager()

        self.vision_range = vision_range
        self.distance = self.get_graph_distance(completion) # Graph distance, in terms of edges, from the player node
        self.discovered_coordinates = self.calculate_discovered_coordinates()

        self.terrain_z_level = 0
        self.entity_z_level = 1
        self.player_z_level = 2

        self.init_graph_tensors()
        self.complete_graph()
        # if plot:
        #     self.visualise_graph()

    def create_node(self, coordinates, z_level, mask=0):
        x, y, z_level, type_id, mask = self.create_node_features(coordinates, z_level, mask)
        node_idx = self.graph_manager.create_idx(coordinates, z_level)
        self.graph.x[node_idx] = [x, y, z_level, type_id, mask]
        return node_idx
        
    def create_node_features(self, coor, z_level, mask):
        x, y = coor
        if z_level == self.terrain_z_level:
            type_id = self.terrain_array[x, y]
        elif z_level == self.entity_z_level:
            type_id = self.entity_array[x, y]
            if type_id == 0:
                mask = 0
        elif z_level == self.player_z_level:
            type_id = 0  # Assuming player type_id is 0 and always visible
            mask = 1
        else:
            exit(f"Invalid z-level: {z_level}")
        return x, y, z_level, type_id, mask

    def init_graph_tensors(self):
        self.num_possible_nodes = self.environment.width * self.environment.height * 2 + 1  # 2 z-levels and a player node
        self.graph_manager.set_max_nodes(self.num_possible_nodes)
        self.num_possible_edges, _, _ = self.compute_total_possible_edges()
        self.graph_manager.set_max_edges(self.num_possible_edges)
        feature_size = 5 # x, y, z_level, type_id, mask
        edge_attr_size = 2 # distance, mask
        self.graph = Data(
            x=torch.full((self.num_possible_nodes, feature_size), -1, dtype=torch.int),
            edge_index=torch.full((2, self.num_possible_edges), -1, dtype=torch.int),
            edge_attr=torch.full((self.num_possible_edges, edge_attr_size), -1, dtype=torch.int)
        )
        # Preallocated tensors for updates
        self.single_node_feature = torch.empty((feature_size), dtype=torch.int)
        self.single_edge_feature = torch.empty((edge_attr_size), dtype=torch.int)

    def complete_graph(self):
        self.add_nodes()
        self.create_terrain_edges()
        self.add_entity_edges()  
        self.verify_graph_integrity()

    def count_entity_nodes(self):
        activated_entities = 0
        deactivated_entities = 0
        for node in self.graph.x:
            if node[2] == self.entity_z_level:
                if node[4] == 1:
                    activated_entities += 1
                else:
                    deactivated_entities += 1
        print(f"Activated entities: {activated_entities}, Deactivated entities: {deactivated_entities}")
        assert activated_entities + deactivated_entities == self.entity_array.size, "Entity nodes do not match the entity array"
        return activated_entities, deactivated_entities

    def verify_graph_integrity(self):
        # Verify all nodes are initialized
        assert torch.all(self.graph.x[:, 4] >= 0), "Some nodes are uninitialized."
        
        # Verify all edges are initialized
        assert torch.all(self.graph.edge_index >= 0), "Some edges are uninitialized."
        assert torch.all(self.graph.edge_attr[:, 1] >= 0), "Some edge attributes are uninitialized."
        
    def is_node_active(self, idx):
        if self.graph.x[idx][4] == 1:
            return True
        else:
            return False
    
    def should_edge_be_active(self, node_idx1, node_idx2):
        if self.is_node_active(node_idx1) and self.is_node_active(node_idx2):
            return True
    
    def discover_this_coordinate(self, x, y):
        if self.discovered_coordinates[x, y]:
            return False
        self.discovered_coordinates[x, y] = 1
        terrain_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
        self.activate_node_and_maybe_its_edges(terrain_idx)
        if self.entity_array[x, y] > 1:
            self.activate_node_and_maybe_its_edges(self.graph_manager.get_node_idx((x, y), self.entity_z_level))
        return True

    def activate_node_and_maybe_its_edges(self, idx):
        self.set_node_mask_1(idx)
        # activate the nodes edges if the corresponding node is activated
        node_pairs = self.graph_manager.retrieve_edge_node_pairs_from_node(idx)
        for node_pair in node_pairs:
            node_idx1, node_idx2 = node_pair
            if self.should_edge_be_active(node_idx1, node_idx2):
                self.activate_edge(node_idx1, node_idx2)

    def activate_edge(self, node_idx1, node_idx2):
        direct_edge_idx, reverse_edge_idx = self.graph_manager.retrieve_edge_indices(node_idx1, node_idx2)
        self.set_edge_mask_1(direct_edge_idx)
        self.set_edge_mask_1(reverse_edge_idx)

    def check_edges_active_of_node(self, idx):
        print(f"Checking edges of node {idx}")
        for edge in self.graph_manager.nodeTuples_edgeIdx_dict:
            if idx in edge:
                print(f"Edge {edge} is connected to node {idx}")
                edge_idx_1, edge_idx_2 = self.graph_manager.retrieve_edge_indices(edge[0], edge[1])
                if self.graph.edge_attr[edge_idx_1][1] == 0:
                    print(f"Edge {edge_idx_1} is not active")
                if self.graph.edge_attr[edge_idx_2][1] == 0:
                    print(f"Edge {edge_idx_2} is not active")

    def deactivate_node_and_its_edges(self, node_idx):
        self.set_node_mask_0(node_idx)
        edge_indices = self.graph_manager.retrieve_edge_indicies_from_node(node_idx)
        for edge_idx in edge_indices:
            self.set_edge_mask_0(edge_idx)

    def set_new_node_type(self, idx, new_type):
        self.graph.x[idx][3] = new_type

    def set_node_mask_0(self, idx):
        self.graph.x[idx][4] = 0

    def set_node_mask_1(self, idx):
        # x, y, z_level, type_id, mask
        self.graph.x[idx][4] = 1

    def set_edge_mask_0(self, idx):
        self.graph.edge_attr[idx][1] = 0

    def set_edge_mask_1(self, idx):
        self.graph.edge_attr[idx][1] = 1

    def build_path_node(self, x, y):
        assert self.entity_array[x, y] == 6, "Entity type is not 6"
        node_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
        self.set_new_node_type(node_idx, self.entity_array[x, y])
        self.activate_node_and_maybe_its_edges(node_idx)
        self.check_entites_active()

    def elevate_terrain_node(self, x, y):
        self.terrain_array[x, y] += 1
        node_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
        self.set_new_node_type(node_idx, self.terrain_array[x, y])

    def remove_entity_node(self, x, y):
        self.entity_array[x, y] = 0
        node_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
        self.set_new_node_type(node_idx, 0)
        self.deactivate_node_and_its_edges(node_idx)

    def move_player_node(self, x, y):
        self.player_pos = (x, y)
        self.discover_this_coordinate(x, y)
        # change player node features
        self.graph.x[self.graph_manager.player_idx][0] = x
        self.graph.x[self.graph_manager.player_idx][1] = y
        # recalculate edge distances to player
        self.recalculate_edge_distances_to_player()

    def recalculate_edge_distances_to_player(self):
        height, width = self.entity_array.shape
        for x in range(width):
            for y in range(height):
                if not self.discovered_coordinates[x, y]:
                    continue
                if self.entity_array[x, y] > 1:
                    entity_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
                    if entity_idx is not None and self.is_node_active(entity_idx):
                        player_idx = self.graph_manager.player_idx
                        entity_pos = (x, y)
                        player_pos = self.player_pos
                        distance = self.get_cartesian_distance(entity_pos, player_pos)
                        edge_indices = self.graph_manager.retrieve_edge_indices(entity_idx, player_idx)
                        if edge_indices:
                            edge_idx_1, edge_idx_2 = edge_indices
                            self.graph.edge_attr[edge_idx_1][0] = distance
                            self.graph.edge_attr[edge_idx_2][0] = distance
                        else:
                            print(f"No edge indices found for nodes {entity_idx} and {player_idx}")
                    else:
                        print(f"Entity node {entity_idx} at position {(x, y)} is not active or not found")

    def create_node(self, coordinates, z_level, mask=0):
        features = self.create_node_features(coordinates, z_level, mask)
        if features is None:
            return None
        node_idx = self.graph_manager.create_idx(coordinates, z_level)
        self.graph.x[node_idx] = torch.tensor(features, dtype=torch.int)
        return node_idx
    
    def add_nodes(self):
        # Adding player node
        self.graph_manager.player_idx = self.create_node(self.player_pos, self.player_z_level, mask=1)
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                # Add terrain nodes
                self.create_node((x, y), self.terrain_z_level, mask=self.discovered_coordinates[x, y])
                # If an entity is present at the location check if it is discovered
                self.create_node((x, y), self.entity_z_level, mask=self.discovered_coordinates[x, y])
    
    def create_edge(self, node_idx1, coor_1, node_idx2, coor_2, distance=None, active=None):
        """Create an edge ensuring undirected consistency."""
        if active == None:
            if self.is_node_active(node_idx1) and self.is_node_active(node_idx2):
                active = 1
            else:
                active = 0

        if distance is None:
            distance = self.get_cartesian_distance(coor_1, coor_2)
        
        direct_edge_idx, reverse_edge_idx = self.graph_manager.create_edge_idx(node_idx1, node_idx2)
        self.add_edge_to_graph(node_idx1, node_idx2, distance, active, direct_edge_idx, reverse_edge_idx)

    def add_edge_to_graph(self, idx1, idx2, distance, active, direct_edge_idx, reverse_edge_idx):
        self.graph.edge_index[:, direct_edge_idx] = torch.tensor([idx1, idx2], dtype=torch.int)
        self.graph.edge_index[:, reverse_edge_idx] = torch.tensor([idx2, idx1], dtype=torch.int)
        self.graph.edge_attr[direct_edge_idx] = torch.tensor([distance, active], dtype=torch.float)
        self.graph.edge_attr[reverse_edge_idx] = torch.tensor([distance, active], dtype=torch.float)

    def create_terrain_edges(self):
        height, width = self.environment.height, self.environment.width
        for x in range(width):
            for y in range(height):
                current_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
                # Check and connect the right and bottom neighbors to create undirected edges
                if x < width - 1:  # Right neighbour
                    right_idx = self.graph_manager.get_node_idx((x + 1, y), self.terrain_z_level)
                    self.create_edge(current_idx, (x, y), right_idx, (x + 1, y), distance=1, active=1)
                if y < height - 1:  # Bottom neighbour
                    bottom_idx = self.graph_manager.get_node_idx((x, y + 1), self.terrain_z_level)
                    self.create_edge(current_idx, (x, y), bottom_idx, (x, y + 1), distance=1, active=1)

    def add_entity_edges(self):
        for x in range(self.environment.width):
            for y in range(self.environment.height):
                entity_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
                terrain_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
                # Connect to the terrain node
                self.create_edge(entity_idx, (x, y), terrain_idx, (x, y), 0)
                # Connect to the player node
                self.create_edge(entity_idx, (x, y), self.graph_manager.player_idx, self.player_pos)

    def calculate_discovered_coordinates(self):
        # discovered = np.full_like(self.terrain_array, False, dtype=bool)
        discovered = np.zeros_like(self.terrain_array, dtype=int)
        for x in range(self.player_pos[0] - self.distance, self.player_pos[0] + self.distance + 1):
            for y in range(self.player_pos[1] - self.distance, self.player_pos[1] + self.distance + 1):
                if self.environment.within_bounds(x, y):
                    discovered[x, y] = 1
        return discovered
    
    def set_current_completness(self):
        # set all discovered coordinates to 0
        self.discovered_coordinates = np.zeros_like(self.terrain_array, dtype=int)
        for x in range(self.player_pos[0] - self.distance, self.player_pos[0] + self.distance + 1):
            for y in range(self.player_pos[1] - self.distance, self.player_pos[1] + self.distance + 1):
                if self.environment.within_bounds(x, y):
                    self.discovered_coordinates[x, y] = 1

    def compute_total_possible_edges(self):
        # Intra-terrain edges   
        terrain_edges = 2 * ((self.environment.width * (self.environment.height - 1)) + (self.environment.height * (self.environment.width - 1)))
        # Entity edges to terrain nodes and player node
        entity_edges = 0
        for _ in range(self.environment.width):
            for _ in range(self.environment.height):
                entity_edges += 4 # 2 edges to terrain nodes and 2 to the player node
        
        return terrain_edges + entity_edges, terrain_edges, entity_edges

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
    
    def check_entites_active(self):
        flag = False
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                if not self.discovered_coordinates[x, y]:
                    continue
                if self.entity_array[x, y] > 1:
                    entity_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
                    if not self.is_node_active(entity_idx):
                        print(f"Entity node {entity_idx} at position {(x, y)} is not active")
                        flag = True

        if flag:
            print(self.entity_array)

    def check_path_nodes(self):
        for node_idx in range(self.graph.x.shape[0]):
            if self.graph.x[node_idx][3] == 6:
                if self.is_node_active(node_idx):
                    print(f"Path node {node_idx} is active")
                    self.check_edges_active_of_node(node_idx)
                else:
                    print(f"Path node {node_idx} is not active")
    
    def visualise_graph(self, node_size=100, edge_color="tab:gray", show_ticks=True):
        # self.check_path_nodes()
        # Convert to undirected graph for visualization
        G = to_networkx(self.graph, to_undirected=True)

        # Use a 2D spring layout, as z-coordinates are manually assigned
        pos = {} # nx.spring_layout(G, seed=42)  # 2D layout
        node_xyz = []
        node_colors = []
        for node in sorted(G):
            node_data = self.graph.x[node]
            x, y, z, type_id, mask = node_data
            if mask:
                pos[node] = (x.item(), y.item(), z.item())  # Ensure all items are floats
                color = self.resolve_color(type_id.item(), z.item(), mask.item())
                if color:  # Only append if color is resolved
                    node_xyz.append(pos[node])
                    node_colors.append(color)
        
        if node_xyz:  # Ensure there are nodes to plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            node_xyz = np.array(node_xyz)
            # invert x-axis to match the game world
            ax.invert_xaxis()
            ax.scatter(*node_xyz.T, s=node_size, color=node_colors, edgecolor='w', depthshade=True)
            # Plot edges
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    ax.plot(*np.array([pos[edge[0]], pos[edge[1]]]).T, color=edge_color)

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
            ax.set_zlabel("Terrain --- Entity --- Agent")
            plt.title("Game World")
            plt.show()

    def resolve_color(self, type_id, z, mask):
        # These colors are in RGB format, normalized to [0, 1] --> green, grey twice, red, brown, black
        entity_colour_map = {2: (0.13, 0.33, 0.16), 3: (0.61, 0.65, 0.62),4: (0.61, 0.65, 0.62),
                             5: (0.78, 0.16, 0.12), 6: (0.46, 0.31, 0.04)}
        if mask == 0:
            return [0.5, 0.5, 0.5, 0.0]  # transparent grey
        elif z == self.terrain_z_level:
            return [c / 255.0 for c in self.environment.terrain_colour_map.get(type_id, (255, 0, 0))]
        elif z == self.entity_z_level:
            return entity_colour_map.get(type_id)
        elif z == self.player_z_level:
            return (0, 0, 0)  # black for player
        return None

    def get_subgraph(self):
        node_idx = self.graph_manager.get_node_idx(self.player_pos, self.terrain_z_level)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.distance,
            edge_index=self.graph.edge_index
        )

        subgraph_data = Data(
            x=self.graph.x[subset],  # Node features of the subgraph
            edge_index=edge_index,   # Edges of the subgraph
            edge_attr=self.graph.edge_attr[edge_mask]  # Edge attributes of the subgraph
        )

        return subgraph_data
