import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import permutations

class RL_Manager:
    def __init__(self, environment):
        self.environment = environment
        self.width, self.height = environment.width, environment.height
        self.terrain_index_grid = environment.terrain_index_grid
        self.entity_index_grid = environment.entity_index_grid
        self.outpost_locations = environment.outpost_locations
        self.shortest_path, self.min_path_length = self.get_target_trade_route()
        print(f"Shortest path: {self.shortest_path}, length: {self.min_path_length}")

    def get_target_trade_route(self):
        graph = self.create_grid_graph(self.outpost_locations)
        shortest_path_indices, min_path_length = self.find_shortest_tsp_path(graph)
        shortest_path_coords = [self.outpost_locations[index] for index in shortest_path_indices]
        return shortest_path_coords, min_path_length

    @staticmethod
    def calculate_distance(coord1, coord2):
        """Calculate Manhattan distance between two points."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def create_grid_graph(self, outpost_coords):
        """Create a grid graph from outpost coordinates, assuming outposts are in a grid."""
        G = nx.Graph()
        # Convert outpost coordinates to indices in a sorted list to handle arbitrary coordinates
        sorted_coords = sorted(outpost_coords)
        coord_to_index = {coord: index for index, coord in enumerate(sorted_coords)}

        # Add nodes
        for coord in outpost_coords:
            G.add_node(coord_to_index[coord], pos=coord)

        # Add edges - only for direct neighbors in a grid
        for coord in outpost_coords:
            x, y = coord
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Possible direct neighbors
            for nx, ny in neighbors:
                if (nx, ny) in coord_to_index:  # Check if neighbor exists
                    G.add_edge(coord_to_index[coord], coord_to_index[(nx, ny)], 
                               weight=self.calculate_distance(coord, (nx, ny)))
        return G


