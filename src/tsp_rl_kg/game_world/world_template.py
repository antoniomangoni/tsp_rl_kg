"""Immutable, deterministic episode starting state (no surfaces or RNG state)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldTemplate:
    terrain: tuple[tuple[int, ...], ...]
    entities: tuple[tuple[int, ...], ...]
    spawn: tuple[int, int]
    outposts: tuple[tuple[int, int], ...]

    @classmethod
    def capture(cls, environment):
        return cls(
            tuple(tuple(int(v) for v in row) for row in environment.terrain_index_grid),
            tuple(tuple(int(v) for v in row) for row in environment.entity_index_grid),
            (int(environment.player.grid_x), int(environment.player.grid_y)),
            tuple(tuple(int(v) for v in p) for p in environment.outpost_locations),
        )

    @property
    def fingerprint(self) -> str:
        payload = (self.terrain, self.entities, self.spawn, self.outposts)
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def restore(self, *, tile_size: int, headless: bool):
        from tsp_rl_kg.game_world.environment import Environment

        return Environment(
            np.array(self.terrain),
            tile_size=tile_size,
            number_of_outposts=len(self.outposts),
            headless=headless,
            template=self,
        )
