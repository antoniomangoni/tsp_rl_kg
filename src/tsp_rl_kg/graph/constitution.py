from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch_geometric.data import Data

from tsp_rl_kg.graph.graph_idx_manager import Graph_Manager


@runtime_checkable
class GraphConstitution(Protocol):
    def build(
        self,
        environment,
        player_pos: tuple[int, int],
        discovered_grid: np.ndarray,
        converter=None,
    ) -> tuple[Data, Graph_Manager]: ...


class DefaultGridConstitution:
    """Builds a 3-layer spatial graph (terrain z=0, entity z=1, player z=2)."""

    TERRAIN_Z = 0
    ENTITY_Z = 1
    PLAYER_Z = 2

    def build(
        self,
        environment,
        player_pos: tuple[int, int],
        discovered_grid: np.ndarray,
        converter=None,
    ) -> tuple[Data, Graph_Manager]:
        w, h = environment.width, environment.height
        terrain_array = environment.terrain_index_grid
        entity_array = environment.entity_index_grid

        gm = Graph_Manager()

        # --- compute sizes ---
        num_nodes = w * h * 2 + 1  # terrain + entity + player
        terrain_edges = 2 * (w * (h - 1) + h * (w - 1))
        entity_edges = w * h * 4  # 2 to terrain, 2 to player
        num_edges = terrain_edges + entity_edges

        gm.set_max_nodes(num_nodes)
        gm.set_max_edges(num_edges)

        # --- allocate tensors ---
        if converter:
            feat_size = converter.embedding_dim
        else:
            feat_size = 1

        graph = Data(
            x=torch.full((num_nodes, feat_size), -1, dtype=torch.int),
            edge_index=torch.full((2, num_edges), -1, dtype=torch.int),
            edge_attr=torch.full((num_edges, 2), -1, dtype=torch.int),  # [distance, mask]
        )

        # --- helpers ---
        def _embed(z_level, x, y):
            if z_level == self.TERRAIN_Z:
                raw = int(terrain_array[x, y])
                return converter.terrain_embedding_lookup[raw] if converter else raw
            elif z_level == self.ENTITY_Z:
                raw = 0 if (x, y) == player_pos else int(entity_array[x, y])
                return converter.entity_embedding_lookup[raw] if converter else raw
            elif z_level == self.PLAYER_Z:
                return converter.agent_embedding if converter else 0
            raise ValueError(f"Invalid z-level: {z_level}")

        def _create_node(coords, z_level, mask=0):
            features = _embed(z_level, coords[0], coords[1])
            idx = gm.create_idx(coords, z_level)
            if converter:
                graph.x[idx] = torch.tensor(features, dtype=torch.float64)
            else:
                graph.x[idx] = torch.tensor(features, dtype=torch.int)
            return idx

        def _add_edge(idx1, c1, idx2, c2, distance=None, active=None):
            if active is None:
                active = 1  # is_node_active always returns True
            if distance is None:
                distance = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            d_idx, r_idx = gm.create_edge_idx(idx1, idx2)
            graph.edge_index[:, d_idx] = torch.tensor([idx1, idx2], dtype=torch.int)
            graph.edge_index[:, r_idx] = torch.tensor([idx2, idx1], dtype=torch.int)
            graph.edge_attr[d_idx] = torch.tensor([distance, active], dtype=torch.float)
            graph.edge_attr[r_idx] = torch.tensor([distance, active], dtype=torch.float)

        # --- add nodes ---
        gm.player_idx = _create_node(player_pos, self.PLAYER_Z, mask=1)
        for y in range(h):
            for x in range(w):
                _create_node((x, y), self.TERRAIN_Z, mask=discovered_grid[x, y])
                _create_node((x, y), self.ENTITY_Z, mask=discovered_grid[x, y])

        # --- terrain adjacency edges ---
        for x in range(w):
            for y in range(h):
                cur = gm.get_node_idx((x, y), self.TERRAIN_Z)
                if x < w - 1:
                    right = gm.get_node_idx((x + 1, y), self.TERRAIN_Z)
                    _add_edge(cur, (x, y), right, (x + 1, y), distance=1, active=1)
                if y < h - 1:
                    bottom = gm.get_node_idx((x, y + 1), self.TERRAIN_Z)
                    _add_edge(cur, (x, y), bottom, (x, y + 1), distance=1, active=1)

        # --- entity edges (terrain + player) ---
        for x in range(w):
            for y in range(h):
                eidx = gm.get_node_idx((x, y), self.ENTITY_Z)
                tidx = gm.get_node_idx((x, y), self.TERRAIN_Z)
                _add_edge(eidx, (x, y), tidx, (x, y), 0)
                _add_edge(eidx, (x, y), gm.player_idx, player_pos)

        # --- verify ---
        assert torch.all(graph.x[:, 0] >= 0), "Some nodes are uninitialized."
        assert torch.all(graph.edge_index >= 0), "Some edges are uninitialized."
        assert torch.all(graph.edge_attr[:, 1] >= 0), "Some edge attributes are uninitialized."

        return graph, gm
