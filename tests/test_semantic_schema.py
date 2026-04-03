from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tsp_rl_kg.graph.feature_encoder import (
    EmbeddingLookupEncoder,
    embedding_metadata_path,
    load_semantic_schema,
    semantic_schema_descriptors,
)


def _write_schema(path: Path, *, include_entity_seven: bool = True) -> None:
    entity_seven = '"7" = "reserved player entity slot"\n' if include_entity_seven else ""
    path.write_text(
        (
            "[meta]\n"
            "version = 1\n\n"
            "[terrain]\n"
            '"0" = "deep water terrain"\n'
            '"1" = "shallow water terrain"\n'
            '"2" = "open plains terrain"\n'
            '"3" = "rolling hills terrain"\n'
            '"4" = "rugged mountain terrain"\n'
            '"5" = "snow-covered highland terrain"\n\n'
            "[entity]\n"
            '"0" = "empty tile"\n'
            '"1" = "fish resource"\n'
            '"2" = "forest tree resource"\n'
            '"3" = "mossy rock obstacle"\n'
            '"4" = "snowy rock obstacle"\n'
            '"5" = "remote outpost destination"\n'
            '"6" = "wooden path marker"\n'
            f"{entity_seven}\n"
            "[player]\n"
            'descriptor = "travelling player agent"\n'
        ),
        encoding="utf-8",
    )


def _write_embedding_file(embedding_path: Path, schema_path: Path, embed_dim: int = 5) -> None:
    descriptors = semantic_schema_descriptors(schema_path)
    embeddings = np.arange(len(descriptors) * embed_dim, dtype=np.float32).reshape(
        len(descriptors),
        embed_dim,
    )
    np.save(embedding_path, embeddings)
    embedding_metadata_path(embedding_path).write_text(
        json.dumps(
            {
                "descriptor_count": len(descriptors),
                "embed_dim": embed_dim,
                "schema_hash": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "model_name": "fixture-model",
            }
        ),
        encoding="utf-8",
    )


def test_repo_semantic_schema_loads_and_uses_ml_descriptors():
    schema = load_semantic_schema("configs/semantic_schema.toml")

    assert schema.terrain[3] == "rolling hills terrain"
    assert schema.entity[5] == "remote outpost destination"
    assert schema.player == "travelling player agent"
    assert schema.terrain[3] != "Hills"
    assert schema.player != "Player"


def test_semantic_schema_descriptors_follow_canonical_order(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    _write_schema(schema_path)

    descriptors = semantic_schema_descriptors(schema_path)

    assert len(descriptors) == 15
    assert descriptors[0] == "deep water terrain"
    assert descriptors[6] == "empty tile"
    assert descriptors[-1] == "travelling player agent"


def test_semantic_schema_requires_complete_entity_mapping(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    _write_schema(schema_path, include_entity_seven=False)

    with pytest.raises(ValueError, match="missing required ids"):
        load_semantic_schema(schema_path)


def test_embedding_lookup_uses_schema_validated_fixture(tmp_path):
    schema_path = tmp_path / "semantic_schema.toml"
    embedding_path = tmp_path / "semantic_embeddings.npy"
    _write_schema(schema_path)
    _write_embedding_file(embedding_path, schema_path, embed_dim=5)

    encoder = EmbeddingLookupEncoder(embedding_path, schema_path)

    assert encoder.node_dim == 5
    assert encoder.metadata["descriptor_count"] == 15
