import networkx as nx
from scipy.spatial.distance import pdist, squareform
from itertools import permutations

class Target_Manager:
    def __init__(self, environment):
        self.environment = environment
        self.width, self.height = environment.width, environment.height
        self.terrain_index_grid = environment.terrain_index_grid
        self.entity_index_grid = environment.entity_index_grid
        self.outpost_locations = environment.outpost_locations

        # print("Entity Grid: ")
        # print(self.entity_index_grid)
        print(f"Outpost locations: {self.outpost_locations}")
        # print(f"Outpost id: {self.environment.terrain_object_grid[self.outpost_locations[0]].entity_on_tile.id}")
        self.G = nx.Graph()
        self.shortest_path, self.min_path_length = self.get_target_trade_route()
        
        print(f"Shortest path: {self.shortest_path}, length: {self.min_path_length}")

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
        print("Graph edges:", self.G.edges(data=True))

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
        print(f"Coordinate to index: {coord_to_index}")

        # Add nodes
        for coord, index in coord_to_index.items():
            self.G.add_node(index, pos=coord)
            print(f"Added node: {index} at {coord}")

        # Add edges - each node is connected to every other node
        for coord1 in outpost_coords:
            for coord2 in outpost_coords:
                if coord1 != coord2:
                    from_id = coord_to_index[coord1]
                    to_id = coord_to_index[coord2]
                    weight = self.calculate_distance(coord1, coord2)
                    self.G.add_edge(from_id, to_id, weight=weight)
                    print(f"Added edge from {from_id} to {to_id} with weight {weight}")

    def get_route_between_cells(self, start, end):
        """Get the shortest path between two cells."""
        return nx.shortest_path(self.G, source=start, target=end)
