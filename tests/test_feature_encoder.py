"""Tests for FeatureEncoder protocol and implementations."""

from __future__ import annotations

import torch

from tsp_rl_kg.graph.feature_encoder import (
    EDGE_ADJACENCY,
    EDGE_ENTITY_TERRAIN,
    EDGE_PLAYER_TERRAIN,
    FeatureEncoder,
    OneHotEncoder,
    RawIntEncoder,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_raw_int_encoder_protocol_conformance():
    assert isinstance(RawIntEncoder(), FeatureEncoder)


def test_onehot_encoder_protocol_conformance():
    assert isinstance(OneHotEncoder(), FeatureEncoder)


# ---------------------------------------------------------------------------
# RawIntEncoder
# ---------------------------------------------------------------------------


class TestRawIntEncoder:
    def test_node_dim(self):
        assert RawIntEncoder().node_dim == 1

    def test_edge_dim(self):
        assert RawIntEncoder().edge_dim == 1

    def test_encode_terrain_range(self):
        enc = RawIntEncoder()
        for raw in range(6):
            t = enc.encode_terrain(raw)
            assert t.shape == (1,)
            assert t.item() == raw

    def test_encode_entity_range(self):
        enc = RawIntEncoder()
        for raw in range(8):
            t = enc.encode_entity(raw)
            assert t.shape == (1,)
            assert t.item() == raw

    def test_encode_player(self):
        t = RawIntEncoder().encode_player()
        assert t.shape == (1,)
        assert t.item() == 0

    def test_encode_edge(self):
        enc = RawIntEncoder()
        for edge_type in [EDGE_ADJACENCY, EDGE_ENTITY_TERRAIN, EDGE_PLAYER_TERRAIN]:
            t = enc.encode_edge(5, edge_type)
            assert t.shape == (1,)
            assert t.item() == 5


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------


class TestOneHotEncoder:
    def test_node_dim(self):
        assert OneHotEncoder().node_dim == 15

    def test_edge_dim(self):
        assert OneHotEncoder().edge_dim == 4

    def test_terrain_is_onehot(self):
        enc = OneHotEncoder()
        for raw in range(6):
            t = enc.encode_terrain(raw)
            assert t.shape == (15,)
            assert t.sum().item() == 1.0
            assert t[raw].item() == 1.0

    def test_entity_is_onehot(self):
        enc = OneHotEncoder()
        for raw in range(8):
            t = enc.encode_entity(raw)
            assert t.shape == (15,)
            assert t.sum().item() == 1.0
            assert t[6 + raw].item() == 1.0  # offset by 6 terrain slots

    def test_player_is_onehot(self):
        enc = OneHotEncoder()
        t = enc.encode_player()
        assert t.shape == (15,)
        assert t.sum().item() == 1.0
        assert t[14].item() == 1.0  # last slot

    def test_node_dim_consistent_across_zlevels(self):
        enc = OneHotEncoder()
        assert enc.encode_terrain(0).shape[0] == enc.node_dim
        assert enc.encode_entity(0).shape[0] == enc.node_dim
        assert enc.encode_player().shape[0] == enc.node_dim

    def test_edge_has_distance_and_type(self):
        enc = OneHotEncoder(grid_size=20)
        t = enc.encode_edge(10, EDGE_ADJACENCY)
        assert t.shape == (4,)
        # First element is normalised distance
        assert t[0].item() > 0
        # One-hot for edge type: adjacency = index 0
        assert t[1].item() == 1.0
        assert t[2].item() == 0.0
        assert t[3].item() == 0.0

    def test_edge_entity_terrain_type(self):
        enc = OneHotEncoder()
        t = enc.encode_edge(0, EDGE_ENTITY_TERRAIN)
        assert t[2].item() == 1.0

    def test_edge_player_terrain_type(self):
        enc = OneHotEncoder()
        t = enc.encode_edge(0, EDGE_PLAYER_TERRAIN)
        assert t[3].item() == 1.0

    def test_edge_distance_normalised(self):
        enc = OneHotEncoder(grid_size=10)
        t = enc.encode_edge(0, EDGE_ADJACENCY)
        assert t[0].item() == 0.0
