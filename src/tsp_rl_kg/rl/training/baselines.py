"""Action-producing references evaluated through the same environment lifecycle."""

import numpy as np

from tsp_rl_kg.game_world.actions import MOVEMENT_DELTAS, ActionType
from tsp_rl_kg.rl.pathfinding import reconstruct_path, remembered_grid, shortest_paths


def movement(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    return int(next(action for action, offset in MOVEMENT_DELTAS.items() if offset == delta))


class RandomBaseline:
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):
        return int(self.rng.integers(self.env.action_space.n)), None


class FullKnowledgeBaseline:
    def __init__(self, env):
        self.env = env
        self.path = []

    def predict(self, observation, deterministic=True):
        if self.env.episode.steps == 0:
            cycle = self.env.current_gm.target_manager.route_tiles[:-1]
            offset = cycle.index(self.env.episode.start)
            self.path = cycle[offset:] + cycle[:offset] + [self.env.episode.start]
        index = self.env.episode.steps
        return movement(self.path[index], self.path[index + 1]), None


class RememberedBFSBaseline:
    def __init__(self, env):
        self.env = env

    def predict(self, observation, deterministic=True):
        memory = self.env.kg.state
        position = self.env.kg.player_pos
        _, passable = remembered_grid(memory)
        distances, parents = shortest_paths(np.ones(passable.shape), passable, position)
        targets = [
            p
            for p in self.env.episode.targets
            if memory.known_tiles[p] and np.isfinite(distances[p]) and p != position
        ]
        if not targets:
            # Walk to a frontier whose ordinary sensing can reveal new tiles.
            frontier = []
            radius = self.env.vision_range
            for x, y in np.argwhere(passable & np.isfinite(distances)):
                region = memory.known_tiles[
                    max(0, x - radius) : x + radius + 1, max(0, y - radius) : y + radius + 1
                ]
                if not region.all() and (x, y) != position:
                    frontier.append((int(x), int(y)))
            targets = frontier
        if not targets:
            return int(ActionType.SCOUT), None
        target = min(targets, key=lambda p: (distances[p], p))
        path = reconstruct_path(parents, position, target)
        return movement(path[0], path[1]), None
