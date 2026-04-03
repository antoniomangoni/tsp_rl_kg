"""Tests for FeatureEncoder protocol and implementations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tsp_rl_kg.config import FeatureEncodingConfig
from tsp_rl_kg.graph.feature_encoder import (
    EDGE_ADJACENCY,
    EDGE_ENTITY_TERRAIN,
    EDGE_PLAYER_TERRAIN,
    EmbeddingLookupEncoder,
    FeatureEncoder,
    OneHotEncoder,
    RawIntEncoder,
    build_feature_encoder,
    embedding_metadata_path,
)
from tsp_rl_kg.rl.encoders import GraphEncoder

_SCHEMA_TEXT = """
[meta]
version = 1

[terrain]
"0" = "deep water terrain"
"1" = "shallow water terrain"
"2" = "open plains terrain"
"3" = "rolling hills terrain"
"4" = "rugged mountain terrain"
"5" = "snow-covered highland terrain"

[entity]
"0" = "empty tile"
"1" = "fish resource"
"2" = "forest tree resource"
"3" = "mossy rock obstacle"
"4" = "snowy rock obstacle"
"5" = "remote outpost destination"
"6" = "wooden path marker"
"7" = "reserved player entity slot"

[player]
descriptor = "travelling player agent"
""".strip()


def _write_schema_file(path: Path) -> None:
    path.write_text(_SCHEMA_TEXT, encoding="utf-8")


def _write_embedding_fixture(
    embedding_path: Path,
    schema_path: Path,
    *,
    embed_dim: int = 7,
    rows: int = 15,
) -> None:
    embeddings = np.arange(rows * embed_dim, dtype=np.float32).reshape(rows, embed_dim)
    np.save(embedding_path, embeddings)
    embedding_metadata_path(embedding_path).write_text(
        json.dumps(
            {
                "descriptor_count": rows,
                "embed_dim": embed_dim,
                "schema_hash": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "model_name": "fixture-model",
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_raw_int_encoder_protocol_conformance():
    assert isinstance(RawIntEncoder(), FeatureEncoder)


def test_onehot_encoder_protocol_conformance():
    assert isinstance(OneHotEncoder(), FeatureEncoder)


def test_embedding_lookup_encoder_protocol_conformance(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    embedding_path = tmp_path / "semantic_embeddings.npy"
    _write_schema_file(schema_path)
    _write_embedding_fixture(embedding_path, schema_path)

    assert isinstance(EmbeddingLookupEncoder(embedding_path, schema_path), FeatureEncoder)


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


class TestEmbeddingLookupEncoder:
    def test_node_dim_matches_embedding_width(self, tmp_path):
        schema_path = tmp_path / "semantic_schema.toml"
        embedding_path = tmp_path / "semantic_embeddings.npy"
        _write_schema_file(schema_path)
        _write_embedding_fixture(embedding_path, schema_path, embed_dim=11)

        enc = EmbeddingLookupEncoder(embedding_path, schema_path)

        assert enc.node_dim == 11
        assert enc.edge_dim == 4
        assert enc.encode_terrain(0).shape == (11,)
        assert enc.encode_entity(5).shape == (11,)
        assert enc.encode_player().shape == (11,)

    def test_rejects_wrong_descriptor_count(self, tmp_path):
        schema_path = tmp_path / "semantic_schema.toml"
        embedding_path = tmp_path / "semantic_embeddings.npy"
        _write_schema_file(schema_path)
        _write_embedding_fixture(embedding_path, schema_path, rows=14)

        with pytest.raises(ValueError, match="row count"):
            EmbeddingLookupEncoder(embedding_path, schema_path)

    def test_rejects_schema_hash_mismatch(self, tmp_path):
        schema_path = tmp_path / "semantic_schema.toml"
        embedding_path = tmp_path / "semantic_embeddings.npy"
        _write_schema_file(schema_path)
        _write_embedding_fixture(embedding_path, schema_path)
        embedding_metadata_path(embedding_path).write_text(
            json.dumps(
                {
                    "descriptor_count": 15,
                    "embed_dim": 7,
                    "schema_hash": "wrong-hash",
                    "model_name": "fixture-model",
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="schema_hash"):
            EmbeddingLookupEncoder(embedding_path, schema_path)


def test_build_feature_encoder_supports_one_hot_and_embedding_lookup(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    embedding_path = tmp_path / "semantic_embeddings.npy"
    _write_schema_file(schema_path)
    _write_embedding_fixture(embedding_path, schema_path, embed_dim=9)

    one_hot = build_feature_encoder(FeatureEncodingConfig(strategy="one_hot"), grid_size=5)
    embedding_lookup = build_feature_encoder(
        FeatureEncodingConfig(
            strategy="embedding_lookup",
            schema_path=str(schema_path),
            embedding_path=str(embedding_path),
        ),
        grid_size=5,
    )

    assert isinstance(one_hot, OneHotEncoder)
    assert isinstance(embedding_lookup, EmbeddingLookupEncoder)
    assert one_hot.node_dim == 15
    assert embedding_lookup.node_dim == 9


def test_graph_encoder_accepts_embedding_lookup_dimension(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    embedding_path = tmp_path / "semantic_embeddings.npy"
    _write_schema_file(schema_path)
    _write_embedding_fixture(embedding_path, schema_path, embed_dim=13)

    feature_encoder = EmbeddingLookupEncoder(embedding_path, schema_path)
    graph_encoder = GraphEncoder(
        num_graph_node_features=feature_encoder.node_dim,
        graph_params={"num_gat_layers": 2, "gat_heads": [2, 2], "fc_dims": [32]},
        output_dim=16,
        gat_hidden_dim=8,
        num_edge_features=feature_encoder.edge_dim,
    )

    x = torch.rand((6, feature_encoder.node_dim))
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.rand((6, feature_encoder.edge_dim))
    batch = torch.zeros(6, dtype=torch.long)

    output = graph_encoder(x, edge_index, batch, edge_attr)

    assert output.shape == (1, 16)
