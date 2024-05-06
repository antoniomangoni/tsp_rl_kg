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

        self.entity_terrain_edge_array = np.zeros_like(self.terrain_array)
        self.player_entity_edge_array = np.zeros_like(self.terrain_array)

        self.player_pos = (self.environment.player.grid_x, self.environment.player.grid_y)
        self.entity_array[self.player_pos] = 0  # Remove the player from the entity array

        assert max(self.entity_array.flatten()) < 7, "Entity type exceeds the maximum value of 6"

        self.idx_manager = IDX_Manager()

        self.vision_range = vision_range

        self.max_terrain_nodes = self.terrain_array.size
        self.distance = self.get_distance(completion) # Graph distance, in terms of edges, from the player node

        self.terrain_z_level = 0
        self.entity_z_level = 1
        self.player_z_level = 2

        self.init_graph()
        self.create_player_node()
        self.add_terrain_nodes_to_graph()
        self.create_player_edges()

        self.visualise_graph()

    def get_node_features(self, coor, z_level):
        x, y = coor
        if z_level == self.terrain_z_level:
            return torch.tensor([[z_level, self.terrain_array[x, y], x, y]], dtype=torch.float)
        elif z_level == self.entity_z_level:
            assert self.entity_array[x, y] > 0, f"Invalid entity type at position ({x}, {y}), type: {self.entity_array[x, y]}, \n {self.entity_array}, \n {self.graph.x}"
            return torch.tensor([[z_level, self.entity_array[x, y], x, y]], dtype=torch.float)
        else:
            raise ValueError(f"Invalid z level: {z_level}")
    
    def init_graph(self):
        self.graph = Data()
        self.graph.x = torch.empty((0, 4), dtype=torch.float)  # Node features: z_level, type_id, x, y
        self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)  # Edge connections
        self.graph.edge_attr = torch.empty((0, 1), dtype=torch.float)  # Edge attributes: distance

    def create_player_node(self):
        player_features = torch.tensor([[self.player_z_level, self.environment.player.id, self.player_pos[0], self.player_pos[1]]], dtype=torch.float)
        self.graph.x = torch.cat([self.graph.x, player_features], dim=0)

        self.idx_manager.player_idx = self.idx_manager.create_idx(self.player_pos, self.player_z_level)

    def create_player_edges(self):
        player_idx = self.idx_manager.player_idx
        for id, idx in self.idx_manager.id_idx_dict.items():
            if idx == player_idx:
                continue
            if id[1] == self.entity_z_level:
                distance = self.get_cartesian_distance(self.player_pos, id[0])
                self.create_edge(player_idx, idx, distance)

    def recaluclate_player_edges(self, new_x, new_y):
        self.player_pos = (new_x, new_y)
        player_idx = self.idx_manager.player_idx
        for id, idx in self.idx_manager.id_idx_dict.items():
            if idx == player_idx:
                continue
            distance = self.get_cartesian_distance(self.player_pos, id[0])
            mask = ((self.graph.edge_index[0] == player_idx) & (self.graph.edge_index[1] == idx)) | \
                ((self.graph.edge_index[1] == player_idx) & (self.graph.edge_index[0] == idx))
            self.graph.edge_attr[mask] = distance

    def create_node(self, z_level, position):
        """ Every node has the following features: [Node Type, Entity/Terrain Type, X Pos, Y Pos]
            Since every node has an idx, we set it here and return it."""
        # print(f"[create_node()] Creating node z_level {z_level} at position {position}")
        features = self.get_node_features(position, z_level)
        self.graph.x = torch.cat([self.graph.x, features], dim=0)
        idx = self.idx_manager.create_idx(position, z_level)
        return idx

    def add_terrain_node(self, position, connect=True):
        new_node_index = self.create_node(self.terrain_z_level, position)
        if connect == True:
            self.create_terrain_edges(new_node_index, position)
        # print(f"[add_terrain_node()] Added terrain node at position {position} with index {new_node_index}")
        return new_node_index

    def add_terrain_nodes_to_graph(self):
        player_x, player_y = self.player_pos
        terrain_nodes_to_calculate_edges = []

        for y in range(player_y - self.distance, player_y + self.distance + 1):
            for x in range(player_x - self.distance, player_x + self.distance + 1):
                # Skip if the node is out of bounds
                if not self.environment.within_bounds(x, y):
                    continue
                # Add the terrain node if it does not exist
                if not self.idx_manager.verify_node_exists((x, y), self.terrain_z_level):    
                    # print(f"[add_terrain_nodes_to_graph()] Adding terrain node at position ({x}, {y})")
                    node_index = self.add_terrain_node((x, y), connect=False)
                    # print(f'[add_terrain_nodes_to_graph] The graph has {self.graph.num_nodes} nodes and {self.graph.num_edges} edges')
                    # Add node index to list of terrain nodes to calculate edges for
                    terrain_nodes_to_calculate_edges.append((node_index, (x, y)))

                if self.entity_array[x, y] > 0:
                    self.add_entity_node((x, y))

        # Calculate edges for all deferred nodes
        self.finalise_terrain_edges(terrain_nodes_to_calculate_edges)

    def add_entity_node(self, position):
        entity_idx = self.create_node(self.entity_z_level, position)
        self.create_entity_edge(entity_idx, position)

    def add_entity_nodes(self):
        for y in range(self.entity_array.shape[0]):
            for x in range(self.entity_array.shape[1]):
                # Skip if the node is out of bounds or already added
                if not self.environment.within_bounds(x, y):
                    continue
                # Add entity if one exists in the array (non-zero value, 1 is a fish which is also not included)
                if self.entity_array[x, y] > 0:
                    self.add_entity_node((x, y))

    def add_path_node(self, position):
        print(f"[add_path_node()] Adding path node at position {position}")
        path_idx = self.create_node(self.entity_z_level, position)
        features = self.get_node_features(position, self.entity_z_level)
        print(f"[add_path_node()] Added path node at position {position} with index {path_idx}")
        self.visualise_graph
        assert features[1] == 6, "Entity not a wood path"
        self.create_entity_edge(path_idx, position)
        print(f"[add_path_node()] Created edge from path to terrain node")
        self.visualise_graph
        distance = self.get_cartesian_distance(position, self.player_pos)
        self.create_edge(path_idx, self.idx_manager.player_idx, distance=distance)
        print(f"[add_path_node()] Created edge from path to player node")
        self.visualise_graph

    def create_edge(self, node1_idx, node2_idx, distance):
        new_edge = torch.tensor([[node1_idx, node2_idx], [node2_idx, node1_idx]], dtype=torch.long).view(2, -1)
        new_attr = torch.tensor([[distance], [distance]], dtype=torch.float)
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge], dim=1)
        self.graph.edge_attr = torch.cat([self.graph.edge_attr, new_attr], dim=0)
    
    def create_entity_edge(self, new_entity_idx, coor):
        terrain_idx = self.idx_manager.get_idx(coor, self.terrain_z_level)
        self.create_edge(new_entity_idx, terrain_idx, distance=0)

    def create_terrain_edges(self, terrain_idx, coor):
        neighbours = self.get_manhattan_neighbours(coor)
        for neighbour in neighbours:
            if self.idx_manager.verify_node_exists(neighbour, self.terrain_z_level):
                neighbour_idx = self.idx_manager.get_idx(neighbour, self.terrain_z_level)
                self.create_edge(terrain_idx, neighbour_idx, distance=1)
    
    def finalise_terrain_edges(self, t_nodes_to_calculate_edges):
        for node_index, position in t_nodes_to_calculate_edges:
            self.create_terrain_edges(node_index, position)

    def remove_entity_node(self, position):
        if not self.idx_manager.verify_node_exists(position, self.entity_z_level):
            print(f"No entity node at position {position} with z_level {self.entity_z_level}")
            return

        entity_idx = self.idx_manager.get_idx(position, self.entity_z_level)

        # Remove the node from the graph's node features and adjust indices
        node_mask = torch.arange(self.graph.num_nodes) != entity_idx
        self.graph.x = self.graph.x[node_mask]

        # Adjust the indices in the edge index tensor
        adjustment_mask = self.graph.edge_index >= entity_idx
        self.graph.edge_index[adjustment_mask] -= 1

        # Remove the edges connected to the node
        edge_mask = (self.graph.edge_index == entity_idx).any(dim=0)
        self.graph.edge_index = self.graph.edge_index[:, ~edge_mask]
        self.graph.edge_attr = self.graph.edge_attr[~edge_mask]

        # Update IDX Manager
        self.idx_manager.remove_idx(position, self.entity_z_level)
        # self.verify_terrain_node_connections(position)
    
    def get_distance(self, completion):
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

    def verify_terrain_node_connections(self, position):
        x, y = position
        break_flag = False
        node_idx = self.idx_manager.get_idx(position, self.terrain_z_level)
        neighbours = self.get_manhattan_neighbours(position)
        # count number of edges to neighbours
        count = 0
        for neighbour in neighbours:
            if self.idx_manager.verify_node_exists(neighbour, self.terrain_z_level):
                neighbour_idx = self.idx_manager.get_idx(neighbour, self.terrain_z_level)
                if (node_idx, neighbour_idx) in self.graph.edge_index.T.tolist() or (neighbour_idx, node_idx) in self.graph.edge_index.T.tolist():
                    count += 1
        if count != len(neighbours):
            print(f"Node at position {position} with type {self.idx_manager.get_type(node_idx)} is missing edges to neighbours")
            print(f"Node has {count} edges, but should have {len(neighbours)}")
            break_flag = True
        # verify that the node is connected to the player node if the self.player_pos == position
        if (node_idx, self.idx_manager.player_idx) not in self.graph.edge_index.T.tolist() and (self.idx_manager.player_idx, node_idx) not in self.graph.edge_index.T.tolist():
            print(f"Player node is not connected to node at position {position} with index {node_idx}")
            break_flag = True
        if break_flag:
            self.visualise_graph()
            raise ValueError("Invalid terrain node connections")
