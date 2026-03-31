from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


@runtime_checkable
class ProjectionPolicy(Protocol):
    def project(self, graph: Data, edge_index: torch.Tensor, player_node_idx: int) -> Data: ...


class KHopProjection:
    """Extract a k-hop subgraph around a given node."""

    def __init__(self, distance: int):
        self._distance = distance

    @property
    def distance(self) -> int:
        return self._distance

    def project(self, graph: Data, edge_index: torch.Tensor, player_node_idx: int) -> Data:
        subset, sub_edge_index, _mapping, edge_mask = k_hop_subgraph(
            node_idx=player_node_idx,
            num_hops=self._distance,
            edge_index=edge_index,
        )
        return Data(
            x=graph.x[subset],
            edge_index=sub_edge_index,
            edge_attr=graph.edge_attr[edge_mask],
        )


class FullGraphProjection:
    """Return the entire graph unfiltered — useful as a testing baseline."""

    @property
    def distance(self) -> None:
        return None

    def project(self, graph: Data, edge_index: torch.Tensor, player_node_idx: int) -> Data:
        return Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
        )


class CompletenessProjection:
    """Compute a k-hop distance from a completeness fraction and delegate to KHopProjection."""

    def __init__(self, completeness: float, vision_range: int, grid_width: int):
        completeness = min(completeness, 1.0)
        self._distance = max(int(completeness * grid_width), vision_range)
        self._inner = KHopProjection(self._distance)

    @property
    def distance(self) -> int:
        return self._distance

    def project(self, graph: Data, edge_index: torch.Tensor, player_node_idx: int) -> Data:
        return self._inner.project(graph, edge_index, player_node_idx)
