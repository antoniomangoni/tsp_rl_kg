"""Remembered observations and reproducible, nested initial map priors."""

from __future__ import annotations

import math

import numpy as np

from tsp_rl_kg.config import validate_completeness


class KnowledgeState:
    def __init__(self, environment, completeness: float, seed: int = 0, template=None):
        from tsp_rl_kg.game_world.world_template import WorldTemplate

        self.completeness = validate_completeness(completeness)
        shape = environment.terrain_index_grid.shape
        self.known_tiles = np.zeros(shape, dtype=bool)
        self.terrain = np.zeros(shape, dtype=np.int64)
        self.entities = np.zeros(shape, dtype=np.int64)
        template = template or WorldTemplate.capture(environment)
        fingerprint = bytes.fromhex(template.fingerprint)
        entropy = [int(seed), *np.frombuffer(fingerprint, dtype="<u4").tolist()]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        self.prior_tiles = np.zeros(shape, dtype=bool)
        indices = rng.permutation(self.known_tiles.size)[
            : math.floor(completeness * self.known_tiles.size)
        ]
        self.prior_tiles.flat[indices] = True
        self.observe(environment, self.prior_tiles)

    def observe(self, environment, tiles: np.ndarray) -> None:
        self.known_tiles |= tiles
        self.terrain[tiles] = environment.terrain_index_grid[tiles]
        self.entities[tiles] = environment.entity_index_grid[tiles]

    def sense(self, environment, center: tuple[int, int], radius: int) -> np.ndarray:
        tiles = np.zeros_like(self.known_tiles)
        x, y = center
        tiles[max(0, x - radius) : x + radius + 1, max(0, y - radius) : y + radius + 1] = True
        self.observe(environment, tiles)
        for x, y in np.argwhere(tiles):
            environment.discover_coordinate(int(x), int(y))
        return tiles

    @property
    def known_fraction(self) -> float:
        return float(self.known_tiles.mean())
