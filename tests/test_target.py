"""Tests for Target_Manager in tsp_rl_kg.rl.target.

``Target_Manager`` computes the algorithmic-optimum trade route and, critically,
``target_route_energy`` - the baseline the reward function scores the agent
against. A regression here silently corrupts every training signal and ablation
result, so it is worth pinning down with hand-computed cases.

The tests drive a lightweight fake environment (rather than a full procedural
world) so the terrain-energy grid and outpost placement are fully controlled and
the expected TSP / Dijkstra results can be computed by hand.
"""

from __future__ import annotations

import numpy as np

from tsp_rl_kg.rl.target import Target_Manager


class _Cell:
    def __init__(self, energy: int) -> None:
        self.energy_requirement = energy


class _FakeEnvironment:
    """Minimal stand-in exposing only what Target_Manager reads."""

    def __init__(self, energy_grid, outpost_locations) -> None:
        arr = np.asarray(energy_grid, dtype=int)
        self.width, self.height = arr.shape
        self.terrain_index_grid = arr.copy()
        self.entity_index_grid = np.zeros_like(arr)
        self.outpost_locations = list(outpost_locations)
        self.terrain_object_grid = np.empty(arr.shape, dtype=object)
        for x in range(self.width):
            for y in range(self.height):
                self.terrain_object_grid[x, y] = _Cell(int(arr[x, y]))


def _uniform_env(size, outposts, energy=1):
    return _FakeEnvironment(np.full((size, size), energy, dtype=int), outposts)


# ---------------------------------------------------------------------------
# Energy grid
# ---------------------------------------------------------------------------


class TestEnergyGrid:
    def test_energy_grid_mirrors_terrain(self):
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # Outposts unused by this assertion but required for construction.
        tm = Target_Manager(_FakeEnvironment(grid, [(0, 0), (2, 2), (0, 2)]))
        assert np.array_equal(tm.energy_req_grid, np.asarray(grid))

    def test_get_cell_energy(self):
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tm = Target_Manager(_FakeEnvironment(grid, [(0, 0), (2, 2), (0, 2)]))
        assert tm.get_cell_energy(1, 2) == 6
        assert tm.get_cell_energy(2, 0) == 7


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------


class TestDistance:
    def test_manhattan_distance(self):
        assert Target_Manager.calculate_distance((0, 0), (3, 4)) == 7
        assert Target_Manager.calculate_distance((2, 2), (2, 2)) == 0


# ---------------------------------------------------------------------------
# Brute-force TSP ordering
# ---------------------------------------------------------------------------


class TestTSPRoute:
    def test_three_outposts_tour_is_triangle_perimeter(self):
        # For three nodes every cyclic order has the same length: the perimeter.
        outposts = [(0, 0), (0, 3), (4, 0)]
        tm = Target_Manager(_uniform_env(5, outposts))
        expected = 3 + 4 + 7  # d(A,B) + d(B,C) + d(C,A) in Manhattan
        assert tm.min_path_length == expected

    def test_four_corner_tour_is_rectangle_perimeter(self):
        # The optimal Hamiltonian cycle over the four corners is the rectangle
        # perimeter (16); any diagonal-crossing order is strictly longer.
        outposts = [(0, 0), (0, 4), (4, 0), (4, 4)]  # already sorted
        tm = Target_Manager(_uniform_env(5, outposts))
        assert tm.min_path_length == 16

    def test_route_is_a_closed_cycle_over_all_outposts(self):
        outposts = [(0, 0), (0, 4), (4, 0), (4, 4)]
        tm = Target_Manager(_uniform_env(5, outposts))
        # get_target_trade_route returns the cycle with the start repeated at end.
        assert tm.shortest_path[0] == tm.shortest_path[-1]
        assert set(tm.shortest_path) == set(outposts)
        assert len(tm.shortest_path) == len(outposts) + 1


# ---------------------------------------------------------------------------
# Least-energy (Dijkstra) pathing
# ---------------------------------------------------------------------------


class TestLeastEnergyPath:
    def test_straight_path_on_uniform_grid(self):
        tm = Target_Manager(_uniform_env(3, [(0, 0), (0, 2), (2, 0)]))
        # (0,0)->(0,1)->(0,2): three unit-energy cells.
        assert tm.calculate_path_energy((0, 0), (0, 2)) == 3

    def test_detours_around_high_energy_wall(self):
        # A wall of energy 9 blocks columns 0<->2 except at (2,1). The least-energy
        # path must detour along the bottom row (all energy 1) rather than crossing.
        grid = [
            [1, 9, 1],
            [1, 9, 1],
            [1, 1, 1],
        ]
        tm = Target_Manager(_FakeEnvironment(grid, [(0, 0), (0, 2), (2, 2)]))
        # Direct crossing would cost 1+9+1 = 11; the 7-cell detour costs 7.
        assert tm.calculate_path_energy((0, 0), (0, 2)) == 7


# ---------------------------------------------------------------------------
# Route energy baseline + determinism
# ---------------------------------------------------------------------------


class TestRouteEnergy:
    def test_target_route_energy_hand_value_on_uniform_grid(self):
        outposts = [(0, 0), (0, 4), (4, 0), (4, 4)]
        tm = Target_Manager(_uniform_env(5, outposts))
        # Optimal cycle is the rectangle: four legs of Manhattan length 4. On a
        # uniform unit grid each leg costs (distance + 1) cells, so
        # 4 legs * (4 + 1) = 20.
        assert tm.target_route_energy == 20

    def test_deterministic_for_same_environment(self):
        outposts = [(0, 0), (0, 4), (4, 0), (4, 4)]
        first = Target_Manager(_uniform_env(5, outposts))
        second = Target_Manager(_uniform_env(5, outposts))
        assert first.shortest_path == second.shortest_path
        assert first.min_path_length == second.min_path_length
        assert first.target_route_energy == second.target_route_energy
