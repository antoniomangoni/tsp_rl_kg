"""Executable static-world energy baseline; modifications can improve on it."""

from itertools import permutations

import numpy as np

from tsp_rl_kg.rl.pathfinding import reconstruct_path, shortest_paths


class Target_Manager:
    def __init__(self, environment):
        self.environment = environment
        self.width, self.height = environment.width, environment.height
        self.outpost_locations = list(environment.outpost_locations)
        if not 2 <= len(self.outpost_locations) <= 6:
            raise ValueError("Route calculation requires 2–6 outposts")
        self.energy_req_grid = self.get_energy_grid()
        self.passable = np.array(
            [
                [
                    getattr(environment.terrain_object_grid[x, y], "passable", True)
                    for y in range(self.height)
                ]
                for x in range(self.width)
            ]
        )
        self._paths = {
            p: shortest_paths(self.energy_req_grid, self.passable, p)
            for p in self.outpost_locations
        }
        start = self.outpost_locations[0]
        routes = ((start, *middle, start) for middle in permutations(self.outpost_locations[1:]))
        self.shortest_path = list(min(routes, key=self.get_route_energy))
        self.target_route_energy = self.get_route_energy(self.shortest_path)
        if not np.isfinite(self.target_route_energy):
            raise ValueError("outposts are not connected by passable tiles")
        self.min_path_length = self.target_route_energy
        self.route_tiles = [start]
        for left, right in zip(self.shortest_path, self.shortest_path[1:], strict=False):
            self.route_tiles.extend(self.calculate_least_energy_path(left, right)[1:])

    def get_energy_grid(self):
        return np.array(
            [[self.get_cell_energy(x, y) for y in range(self.height)] for x in range(self.width)],
            dtype=float,
        )

    def get_cell_energy(self, x, y):
        return self.environment.terrain_object_grid[x, y].energy_requirement

    @staticmethod
    def calculate_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _query(self, start):
        if start in self._paths:
            return self._paths[start]
        return shortest_paths(self.energy_req_grid, self.passable, start)

    def calculate_least_energy_path(self, start, end):
        _, parents = self._query(start)
        return reconstruct_path(parents, start, end)

    def calculate_path_energy(self, start, end):
        distances, _ = self._query(start)
        return float(distances[end])

    def get_route_energy(self, route):
        return sum(self.calculate_path_energy(a, b) for a, b in zip(route, route[1:], strict=False))

    def get_energy_required(self, path):
        return self.get_route_energy(path)

    def get_target_trade_route(self):
        return self.shortest_path, self.target_route_energy
