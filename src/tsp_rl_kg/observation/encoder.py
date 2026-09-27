from __future__ import annotations

from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch_geometric.data import Data


@runtime_checkable
class ObservationEncoder(Protocol):
    def encode(
        self, subgraph: Data, vision: np.ndarray, status=None, path_queries=None
    ) -> dict[str, np.ndarray]: ...

    def observation_space(self) -> gym.spaces.Dict: ...


class PaddedPyGObservationEncoder:
    """Pads variable-size PyG subgraphs into fixed-size arrays for Gym."""

    def __init__(
        self,
        max_nodes: int,
        max_edges: int,
        num_node_features: int,
        num_edge_features: int,
        vision_shape: tuple[int, int, int],
        status_dim: int = 11,
        num_outposts: int = 3,
    ):
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._num_node_features = num_node_features
        self._num_edge_features = num_edge_features
        self._vision_shape = vision_shape
        self._status_dim = status_dim
        self._num_outposts = num_outposts

    def observation_space(self) -> gym.spaces.Dict:
        vision_space = spaces.Box(low=0, high=1, shape=self._vision_shape, dtype=np.float16)

        node_feature_space = spaces.Box(
            low=-1.0,
            high=1e4,
            shape=(self._max_nodes, self._num_node_features),
            dtype=np.float16,
        )

        edge_attr_space = spaces.Box(
            low=-1.0,
            high=1e4,
            shape=(self._max_edges, self._num_edge_features),
            dtype=np.float16,
        )
        edge_index_space = spaces.Box(
            low=0,
            high=self._max_nodes - 1,
            shape=(2, self._max_edges),
            dtype=np.int64,
        )

        return spaces.Dict(
            {
                "status": spaces.Box(0, 1, shape=(self._status_dim,), dtype=np.float32),
                "path_queries": spaces.Box(-1, 1, shape=(self._num_outposts, 6), dtype=np.float32),
                "num_nodes": spaces.Box(low=1, high=self._max_nodes, shape=(1,), dtype=np.int64),
                "num_edges": spaces.Box(low=0, high=self._max_edges, shape=(1,), dtype=np.int64),
                "vision": vision_space,
                "node_features": node_feature_space,
                "edge_attr": edge_attr_space,
                "edge_index": edge_index_space,
            }
        )

    def encode(
        self, subgraph: Data, vision: np.ndarray, status=None, path_queries=None
    ) -> dict[str, np.ndarray]:
        n, e = subgraph.num_nodes, subgraph.num_edges
        if n is None or not 1 <= n <= self._max_nodes or not 0 <= e <= self._max_edges:
            raise ValueError("Graph exceeds observation capacity or contains no nodes")
        if subgraph.x.shape != (n, self._num_node_features):
            raise ValueError("Invalid node feature dimensions")
        if subgraph.edge_attr.shape != (e, self._num_edge_features):
            raise ValueError("Invalid edge attribute dimensions")
        edges = subgraph.edge_index.detach().cpu().numpy()
        if edges.shape != (2, e) or not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("edge_index must be an integer array of shape (2, num_edges)")
        if e and (edges.min() < 0 or edges.max() >= n):
            raise ValueError("Graph edge references an invalid local node")
        for features in (subgraph.x, subgraph.edge_attr):
            values = features.detach().cpu().numpy()
            if not np.isfinite(values).all() or (values < -1).any() or (values > 1e4).any():
                raise ValueError("Graph features are nonfinite or outside observation bounds")
        if vision.shape != self._vision_shape or not np.isfinite(vision).all():
            raise ValueError("Invalid vision dimensions or values")
        if (vision < 0).any() or (vision > 255).any():
            raise ValueError("Vision must contain RGB values in [0, 255]")
        node_features = np.zeros((self._max_nodes, subgraph.num_node_features), dtype=np.float16)
        node_features[: subgraph.num_nodes, :] = subgraph.x.detach().cpu().numpy()

        edge_attr = np.zeros((self._max_edges, subgraph.num_edge_features), dtype=np.float16)
        edge_attr[: subgraph.num_edges, :] = subgraph.edge_attr.detach().cpu().numpy()

        edge_index = np.zeros((2, self._max_edges), dtype=np.int64)
        edge_index[:, : subgraph.num_edges] = edges

        status = np.zeros(self._status_dim, dtype=np.float32) if status is None else status
        path_queries = (
            np.zeros((self._num_outposts, 6), dtype=np.float32)
            if path_queries is None
            else path_queries
        )
        if not self.observation_space()["status"].contains(status):
            raise ValueError("Invalid task status")
        if not self.observation_space()["path_queries"].contains(path_queries):
            raise ValueError("Invalid path queries")
        return {
            "status": status,
            "path_queries": path_queries,
            "num_nodes": np.array([n], dtype=np.int64),
            "num_edges": np.array([e], dtype=np.int64),
            "vision": (
                vision.astype(np.float16)
                if self._vision_shape[0] == 15
                else vision.astype(np.float16) / 255.0
            ),
            "node_features": node_features,
            "edge_attr": edge_attr,
            "edge_index": edge_index,
        }
