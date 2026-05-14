"""Tests for the ObservationEncoder protocol and PaddedPyGObservationEncoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tsp_rl_kg.observation.encoder import (
    ObservationEncoder,
    PaddedPyGObservationEncoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_NODES = 20
MAX_EDGES = 40
NUM_NODE_FEATURES = 4
NUM_EDGE_FEATURES = 2
VISION_SHAPE = (3, 32, 32)


def _make_subgraph(num_nodes: int = 5, num_edges: int = 8) -> Data:
    """Build a small synthetic PyG Data object."""
    x = torch.randint(0, 7, (num_nodes, NUM_NODE_FEATURES), dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )
    edge_attr = torch.randint(0, 5, (num_edges, NUM_EDGE_FEATURES), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _make_vision() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=VISION_SHAPE, dtype=np.uint8)


@pytest.fixture
def encoder() -> PaddedPyGObservationEncoder:
    return PaddedPyGObservationEncoder(
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        vision_shape=VISION_SHAPE,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance(encoder: PaddedPyGObservationEncoder):
    assert isinstance(encoder, ObservationEncoder)


# ---------------------------------------------------------------------------
# encode() tests
# ---------------------------------------------------------------------------


def test_encode_output_shapes(encoder: PaddedPyGObservationEncoder):
    obs = encoder.encode(_make_subgraph(), _make_vision())

    assert obs["vision"].shape == VISION_SHAPE
    assert obs["node_features"].shape == (MAX_NODES, NUM_NODE_FEATURES)
    assert obs["edge_attr"].shape == (MAX_EDGES, NUM_EDGE_FEATURES)
    assert obs["edge_index"].shape == (2, MAX_EDGES)


def test_encode_padding(encoder: PaddedPyGObservationEncoder):
    num_nodes, num_edges = 3, 4
    subgraph = _make_subgraph(num_nodes=num_nodes, num_edges=num_edges)
    obs = encoder.encode(subgraph, _make_vision())

    # Actual data region should be non-zero (statistically; random ints 0-6)
    assert obs["node_features"][:num_nodes].any()
    # Padding region must be all zeros
    np.testing.assert_array_equal(obs["node_features"][num_nodes:], 0)
    np.testing.assert_array_equal(obs["edge_attr"][num_edges:], 0)
    np.testing.assert_array_equal(obs["edge_index"][:, num_edges:], 0)


def test_encode_deterministic(encoder: PaddedPyGObservationEncoder):
    subgraph = _make_subgraph()
    vision = _make_vision()

    obs1 = encoder.encode(subgraph, vision)
    obs2 = encoder.encode(subgraph, vision)

    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])


def test_vision_normalised(encoder: PaddedPyGObservationEncoder):
    vision = np.full(VISION_SHAPE, 255, dtype=np.uint8)
    obs = encoder.encode(_make_subgraph(), vision)

    np.testing.assert_allclose(obs["vision"], 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# observation_space() tests
# ---------------------------------------------------------------------------


def test_observation_space_matches_encode_output(encoder: PaddedPyGObservationEncoder):
    space = encoder.observation_space()
    obs = encoder.encode(_make_subgraph(), _make_vision())

    for key in space.spaces:
        assert key in obs, f"Missing key {key} in encoded observation"
        assert obs[key].shape == space[key].shape, (
            f"Shape mismatch for {key}: obs={obs[key].shape}, space={space[key].shape}"
        )


def test_observation_space_keys(encoder: PaddedPyGObservationEncoder):
    space = encoder.observation_space()
    assert set(space.spaces.keys()) == {"vision", "node_features", "edge_attr", "edge_index"}
