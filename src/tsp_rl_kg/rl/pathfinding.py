"""Shared path queries. Costs charge only the destination of each movement."""

from heapq import heappop, heappush

import numpy as np

DELTAS = ((-1, 0), (1, 0), (0, 1), (0, -1))
TERRAIN_ENERGY = np.array([12, 8, 2, 3, 5, 6], dtype=float)


def remembered_grid(state):
    costs = TERRAIN_ENERGY[state.terrain].copy()
    costs[state.entities == 5] = 0
    paths = state.entities == 6
    costs[paths] = np.maximum(0, costs[paths] - 2)
    passable = state.known_tiles & ~np.isin(state.entities, [1, 2, 3, 4])
    return costs, passable


def shortest_paths(costs, passable, start):
    distances = np.full(costs.shape, np.inf)
    distances[start] = 0.0
    parents = {}
    queue = [(0.0, start)]
    while queue:
        distance, position = heappop(queue)
        if distance != distances[position]:
            continue
        for dx, dy in DELTAS:
            nxt = (position[0] + dx, position[1] + dy)
            if not (0 <= nxt[0] < costs.shape[0] and 0 <= nxt[1] < costs.shape[1]):
                continue
            if not passable[nxt]:
                continue
            candidate = distance + float(costs[nxt])
            if candidate < distances[nxt]:
                distances[nxt] = candidate
                parents[nxt] = position
                heappush(queue, (candidate, nxt))
    return distances, parents


def reconstruct_path(parents, start, end):
    if end != start and end not in parents:
        return []
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return path[::-1]
