"""Composable feature encoding for graph nodes and edges.

FeatureEncoder is the Protocol that controls how raw terrain/entity/player
integers become tensors and how edge attributes are encoded.  Two concrete
implementations are provided:

* **RawIntEncoder** — replicates the legacy 1-dim integer behaviour.
* **OneHotEncoder** — one-hot terrain (6) + entity (8) + player flag, padded
  to a uniform ``node_dim``.  Edge features include normalised distance plus
  a 3-class one-hot for edge type.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch

# Edge-type constants used by FeatureEncoder.encode_edge
EDGE_ADJACENCY = "adjacency"
EDGE_ENTITY_TERRAIN = "entity_terrain"
EDGE_PLAYER_TERRAIN = "player_terrain"

_EDGE_TYPE_ORDER = [EDGE_ADJACENCY, EDGE_ENTITY_TERRAIN, EDGE_PLAYER_TERRAIN]


@runtime_checkable
class FeatureEncoder(Protocol):
    """Strategy for encoding raw ints into graph tensors."""

    @property
    def node_dim(self) -> int: ...

    @property
    def edge_dim(self) -> int: ...

    def encode_terrain(self, raw: int) -> torch.Tensor: ...

    def encode_entity(self, raw: int) -> torch.Tensor: ...

    def encode_player(self) -> torch.Tensor: ...

    def encode_edge(self, distance: int, edge_type: str) -> torch.Tensor: ...


class RawIntEncoder:
    """Replicates the legacy converter=None behaviour: 1-dim integer features."""

    @property
    def node_dim(self) -> int:
        return 1

    @property
    def edge_dim(self) -> int:
        return 1

    def encode_terrain(self, raw: int) -> torch.Tensor:
        return torch.tensor([raw], dtype=torch.float)

    def encode_entity(self, raw: int) -> torch.Tensor:
        return torch.tensor([raw], dtype=torch.float)

    def encode_player(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.float)

    def encode_edge(self, distance: int, edge_type: str) -> torch.Tensor:
        return torch.tensor([distance], dtype=torch.float)


# Number of known terrain / entity classes
_NUM_TERRAIN_TYPES = 6  # 0–5
_NUM_ENTITY_TYPES = 8  # 0–7


class OneHotEncoder:
    """One-hot encoding with uniform padding across z-levels.

    Node features: terrain one-hot (6) + entity one-hot (8) + player flag (1) = 15-dim.
    Only the relevant slice is hot; the rest is zero.

    Edge features: normalised distance (1) + edge-type one-hot (3) = 4-dim.
    """

    def __init__(self, grid_size: int = 20):
        self._grid_size = grid_size
        self._diag = math.sqrt(2) * grid_size

    @property
    def node_dim(self) -> int:
        return _NUM_TERRAIN_TYPES + _NUM_ENTITY_TYPES + 1  # 15

    @property
    def edge_dim(self) -> int:
        return 1 + len(_EDGE_TYPE_ORDER)  # 4

    def encode_terrain(self, raw: int) -> torch.Tensor:
        t = torch.zeros(self.node_dim, dtype=torch.float)
        t[raw] = 1.0
        return t

    def encode_entity(self, raw: int) -> torch.Tensor:
        t = torch.zeros(self.node_dim, dtype=torch.float)
        t[_NUM_TERRAIN_TYPES + raw] = 1.0
        return t

    def encode_player(self) -> torch.Tensor:
        t = torch.zeros(self.node_dim, dtype=torch.float)
        t[_NUM_TERRAIN_TYPES + _NUM_ENTITY_TYPES] = 1.0
        return t

    def encode_edge(self, distance: int, edge_type: str) -> torch.Tensor:
        norm_dist = distance / self._diag if self._diag > 0 else 0.0
        t = torch.zeros(self.edge_dim, dtype=torch.float)
        t[0] = norm_dist
        idx = _EDGE_TYPE_ORDER.index(edge_type)
        t[1 + idx] = 1.0
        return t
