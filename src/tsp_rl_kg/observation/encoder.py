from __future__ import annotations

from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch_geometric.data import Data


@runtime_checkable
class ObservationEncoder(Protocol):
    def encode(self, subgraph: Data, vision: np.ndarray) -> dict[str, np.ndarray]: ...

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
        converter=None,
    ):
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._num_node_features = num_node_features
        self._num_edge_features = num_edge_features
        self._vision_shape = vision_shape
        self._converter = converter

    def observation_space(self) -> gym.spaces.Dict:
        vision_space = spaces.Box(low=0, high=255, shape=self._vision_shape, dtype=np.float16)

        if self._converter is None:
            node_feature_space = spaces.Box(
                low=0,
                high=7,
                shape=(self._max_nodes, self._num_node_features),
                dtype=np.uint8,
            )
        else:
            node_feature_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self._max_nodes, self._converter.embedding_dim),
                dtype=np.float64,
            )

        edge_attr_space = spaces.Box(
            low=0,
            high=self._max_edges - 1,
            shape=(self._max_edges, self._num_edge_features),
            dtype=np.uint8,
        )
        edge_index_space = spaces.Box(
            low=0,
            high=self._max_nodes - 1,
            shape=(2, self._max_edges),
            dtype=np.int64,
        )

        return spaces.Dict(
            {
                "vision": vision_space,
                "node_features": node_feature_space,
                "edge_attr": edge_attr_space,
                "edge_index": edge_index_space,
            }
        )

    def encode(self, subgraph: Data, vision: np.ndarray) -> dict[str, np.ndarray]:
        node_features = np.zeros((self._max_nodes, subgraph.num_node_features), dtype=np.float16)
        node_features[: subgraph.num_nodes, :] = subgraph.x.numpy()

        edge_attr = np.zeros((self._max_edges, subgraph.num_edge_features), dtype=np.float16)
        edge_attr[: subgraph.num_edges, :] = subgraph.edge_attr.numpy()

        edge_index = np.zeros((2, self._max_edges), dtype=np.int64)
        edge_index[:, : subgraph.num_edges] = subgraph.edge_index.numpy()

        return {
            "vision": vision.astype(np.float16) / 255.0,
            "node_features": node_features,
            "edge_attr": edge_attr,
            "edge_index": edge_index,
        }
