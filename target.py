import networkx as nx
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import numpy as np
from queue import PriorityQueue


class Target_Manager:
    def __init__(self, environment):
        self.environment = environment
        self.width, self.height = environment.width, environment.height
        self.terrain_index_grid = environment.terrain_index_grid
        self.entity_index_grid = environment.entity_index_grid
        self.outpost_locations = environment.outpost_locations
        
        self.energy_req_grid = self.get_energy_grid()
        
        self.G = nx.Graph()
        self.shortest_path, self.min_path_length = self.get_target_trade_route()
        self.target_route_energy = self.get_route_energy(self.shortest_path)

    def get_energy_grid(self):
        grid = np.zeros_like(self.terrain_index_grid)
        for x in range(self.width):
            for y in range(self.height):
                grid[x, y] = self.environment.terrain_object_grid[x, y].energy_requirement
        return grid
    
    def get_energy_required(self, path):
        """Calculate the energy required for a given path."""
        energy = 0
        for coords in path:
            energy += self.environment.terrain_object_grid[coords].energy_requirement
        return energy 
    
    def find_shortest_tsp_path(self):
        nodes = list(self.G.nodes())
        shortest_path = None
        min_path_length = float('inf')

        for permutation in permutations(nodes):
            cycle = permutation + (permutation[0],)
            path_length = sum(self.G.edges[cycle[n], cycle[n+1]]['weight'] for n in range(len(cycle) - 1))

            if path_length < min_path_length:
                shortest_path = cycle
                min_path_length = path_length

        return shortest_path, min_path_length
    def get_target_trade_route(self):
        self.create_grid_graph(self.outpost_locations)
        shortest_path_indices, min_path_length = self.find_shortest_tsp_path()
        shortest_path_coords = [self.outpost_locations[index] for index in shortest_path_indices]
        return shortest_path_coords, min_path_length

    @staticmethod
    def calculate_distance(coord1, coord2):
        """Calculate Manhattan distance between two points."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def create_grid_graph(self, outpost_coords):
        """Create a fully connected graph from outpost coordinates."""
        # Convert outpost coordinates to indices in a sorted list to handle arbitrary coordinates
        coord_to_index = {coord: index for index, coord in enumerate(sorted(outpost_coords))}

        # Add nodes
        for coord, index in coord_to_index.items():
            self.G.add_node(index, pos=coord)

        # Add edges - each node is connected to every other node
        for coord1 in outpost_coords:
            for coord2 in outpost_coords:
                if coord1 != coord2:
                    from_id = coord_to_index[coord1]
                    to_id = coord_to_index[coord2]
                    weight = self.calculate_distance(coord1, coord2)
                    self.G.add_edge(from_id, to_id, weight=weight)

    def get_energy_required(self, path):
        """Calculate the total energy required for the paths between outposts in the path."""
        total_energy = 0
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            total_energy += self.calculate_path_energy(start, end)
        return total_energy

    def get_cell_energy(self, x, y):
        """Retrieve the energy requirement of the cell at (x, y)."""
        return self.environment.terrain_object_grid[x, y].energy_requirement

    def get_energy_neighbous(self, x, y):
        """Retrieve the energy requirements of the neighbors of the cell at (x, y)."""
        neighbors = self.environment.get_neighbors(x, y)
        return [self.get_cell_energy(x, y) for x, y in neighbors]

    def calculate_least_energy_path(self, start, end):
        """
        Calculate the path with the least energy required from start to end.
        Uses a variation of Dijkstra's algorithm adapted for energy costs.
        """
        # Initialize distance map with infinity
        energy_cost = {coord: float('inf') for coord in np.ndindex(self.energy_req_grid.shape)}
        energy_cost[start] = self.get_cell_energy(*start)  # Start cell energy
        prev = {coord: None for coord in np.ndindex(self.energy_req_grid.shape)}

        pq = PriorityQueue()
        pq.put((self.get_cell_energy(*start), start))  # Priority queue of (energy, coord)

        while not pq.empty():
            current_energy, current_coord = pq.get()
            if current_coord == end:
                break  # End reached, construct path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Manhattan neighbors
                neighbor = (current_coord[0] + dx, current_coord[1] + dy)
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:  # Check bounds
                    # Calculate new cost to neighbor
                    new_energy = current_energy + self.get_cell_energy(*neighbor)
                    if new_energy < energy_cost[neighbor]:
                        energy_cost[neighbor] = new_energy
                        prev[neighbor] = current_coord
                        pq.put((new_energy, neighbor))

        # Reconstruct the path from end to start
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()  # Reverse it to start -> end
        return path

    def calculate_path_energy(self, start, end):
        """Calculate the least energy required to move from start to end."""
        path = self.calculate_least_energy_path(start, end)
        return sum(self.get_cell_energy(*coord) for coord in path)
    
    def get_route_energy(self, route):
        """Calculate the total energy required for the path."""
        total_energy = 0
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            total_energy += self.calculate_path_energy(start, end)
        return total_energy
