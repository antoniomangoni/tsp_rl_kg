"""Single source of truth for closed-tour episode state and sensed-only shaping."""

from dataclasses import dataclass, field

import numpy as np

from tsp_rl_kg.knowledge.state import KnowledgeState
from tsp_rl_kg.rl.pathfinding import remembered_grid, shortest_paths


def efficiency(reference, energy):
    if reference == 0:
        return 1.0 if energy == 0 else 0.0
    return float(reference / max(energy, 1e-8))


@dataclass
class EpisodeState:
    outposts: tuple
    start: tuple
    visited: set = field(default_factory=set)
    energy: float = 0.0
    steps: int = 0
    no_progress: int = 0
    best_distance: float = float("inf")
    success: bool = False
    done: bool = False
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(11, dtype=int))

    def __post_init__(self):
        self.visited.add(self.start)

    @property
    def targets(self):
        return set(self.outposts) - self.visited or {self.start}

    @property
    def returning(self):
        return len(self.visited) == len(self.outposts)

    def distance(self, memory, position):
        _, passable = remembered_grid(memory)
        distances, _ = shortest_paths(np.ones(passable.shape), passable, position)
        known = [p for p in self.targets if memory.known_tiles[p]]
        return min((distances[p] for p in known), default=float("inf"))

    def potential(self, memory, position):
        if self.success:
            return 0.0
        distance = self.distance(memory, position)
        return -min(float(distance) / memory.known_tiles.size, 1.0)

    def update(self, position, energy, action, newly_sensed, memory):
        old_targets = self.targets
        new_visit = position in self.outposts and position not in self.visited
        if new_visit:
            self.visited.add(position)
        self.energy = float(energy)
        self.steps += 1
        self.action_counts[action] += 1
        self.success = self.returning and position == self.start
        distance = self.distance(memory, position)
        if old_targets != self.targets:
            self.best_distance = float("inf")
        if new_visit or newly_sensed or distance < self.best_distance:
            self.no_progress = 0
        else:
            self.no_progress += 1
        self.best_distance = min(self.best_distance, distance)
        return new_visit

    def status(self, agent, limits):
        return np.asarray(
            [
                *[p in self.visited for p in self.outposts],
                *[p == self.start for p in self.outposts],
                agent.wood / agent.resource_max,
                agent.stone / agent.resource_max,
                max(0, 1 - self.steps / limits.max_episode_steps),
                self.returning,
                min(1, self.no_progress / limits.max_steps_without_progress),
            ],
            dtype=np.float32,
        )


def sensing_memory(environment, seed, template):
    return KnowledgeState(environment, completeness=0, seed=seed, template=template)
