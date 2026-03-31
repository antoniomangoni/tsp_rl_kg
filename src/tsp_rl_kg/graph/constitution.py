from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch_geometric.data import Data

from tsp_rl_kg.graph.feature_encoder import (
    EDGE_ADJACENCY,
    EDGE_ENTITY_TERRAIN,
    EDGE_PLAYER_TERRAIN,
    FeatureEncoder,
    RawIntEncoder,
)
from tsp_rl_kg.graph.graph_idx_manager import Graph_Manager


@runtime_checkable
class GraphConstitution(Protocol):
    def build(
        self,
        environment,
        player_pos: tuple[int, int],
        discovered_grid: np.ndarray,
        feature_encoder: FeatureEncoder | None = None,
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
        feature_encoder: FeatureEncoder | None = None,
    ) -> tuple[Data, Graph_Manager]:
        if feature_encoder is None:
            feature_encoder = RawIntEncoder()

        w, h = environment.width, environment.height
        terrain_array = environment.terrain_index_grid
        entity_array = environment.entity_index_grid

        gm = Graph_Manager()

        # --- compute sizes ---
        num_nodes = w * h * 2 + 1  # terrain + entity + player
        terrain_edges = 2 * (w * (h - 1) + h * (w - 1))
        entity_edges = w * h * 2  # entity→terrain only (no entity→player)
        player_edges = 2  # 1 bidirectional player→terrain pair
        num_edges = terrain_edges + entity_edges + player_edges

        gm.set_max_nodes(num_nodes)
        gm.set_max_edges(num_edges)

        # --- allocate tensors ---
        feat_size = feature_encoder.node_dim
        edge_feat_size = feature_encoder.edge_dim

        graph = Data(
            x=torch.full((num_nodes, feat_size), -1, dtype=torch.float),
            edge_index=torch.full((2, num_edges), -1, dtype=torch.int),
            edge_attr=torch.full((num_edges, edge_feat_size), -1, dtype=torch.float),
        )

        # --- helpers ---
        def _embed(z_level, x, y):
            if z_level == self.TERRAIN_Z:
                raw = int(terrain_array[x, y])
                return feature_encoder.encode_terrain(raw)
            elif z_level == self.ENTITY_Z:
                raw = 0 if (x, y) == player_pos else int(entity_array[x, y])
                return feature_encoder.encode_entity(raw)
            elif z_level == self.PLAYER_Z:
                return feature_encoder.encode_player()
            raise ValueError(f"Invalid z-level: {z_level}")

        def _create_node(coords, z_level, mask=0):
            features = _embed(z_level, coords[0], coords[1])
            idx = gm.create_idx(coords, z_level)
            graph.x[idx] = features
            return idx

        def _add_edge(idx1, c1, idx2, c2, distance=None, edge_type=EDGE_ADJACENCY):
            if distance is None:
                distance = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            d_idx, r_idx = gm.create_edge_idx(idx1, idx2)
            graph.edge_index[:, d_idx] = torch.tensor([idx1, idx2], dtype=torch.int)
            graph.edge_index[:, r_idx] = torch.tensor([idx2, idx1], dtype=torch.int)
            attr = feature_encoder.encode_edge(distance, edge_type)
            graph.edge_attr[d_idx] = attr
            graph.edge_attr[r_idx] = attr

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
                    _add_edge(cur, (x, y), right, (x + 1, y), distance=1, edge_type=EDGE_ADJACENCY)
                if y < h - 1:
                    bottom = gm.get_node_idx((x, y + 1), self.TERRAIN_Z)
                    _add_edge(cur, (x, y), bottom, (x, y + 1), distance=1, edge_type=EDGE_ADJACENCY)

        # --- entity→terrain edges ---
        for x in range(w):
            for y in range(h):
                eidx = gm.get_node_idx((x, y), self.ENTITY_Z)
                tidx = gm.get_node_idx((x, y), self.TERRAIN_Z)
                _add_edge(eidx, (x, y), tidx, (x, y), 0, edge_type=EDGE_ENTITY_TERRAIN)

        # --- single player→terrain edge ---
        player_terrain_idx = gm.get_node_idx(player_pos, self.TERRAIN_Z)
        d_idx, r_idx = gm.create_edge_idx(gm.player_idx, player_terrain_idx)
        pi, ti = gm.player_idx, player_terrain_idx
        graph.edge_index[:, d_idx] = torch.tensor([pi, ti], dtype=torch.int)
        graph.edge_index[:, r_idx] = torch.tensor([ti, pi], dtype=torch.int)
        attr = feature_encoder.encode_edge(0, EDGE_PLAYER_TERRAIN)
        graph.edge_attr[d_idx] = attr
        graph.edge_attr[r_idx] = attr
        gm.player_edge_direct_idx = d_idx
        gm.player_edge_reverse_idx = r_idx

        # --- verify ---
        assert torch.all(graph.x[:, 0] >= 0), "Some nodes are uninitialized."
        assert torch.all(graph.edge_index >= 0), "Some edges are uninitialized."
        assert torch.all(graph.edge_attr[:, 0] >= 0), "Some edge attributes are uninitialized."

        return graph, gm
