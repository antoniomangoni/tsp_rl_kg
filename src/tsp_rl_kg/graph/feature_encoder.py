"""Composable feature encoding for graph nodes and edges.

FeatureEncoder is the Protocol that controls how raw terrain/entity/player
integers become tensors and how edge attributes are encoded. Supported node
encoders are:

* **RawIntEncoder** — legacy 1-dim integer features.
* **OneHotEncoder** — one-hot terrain/entity/player features.
* **EmbeddingLookupEncoder** — pre-computed semantic embeddings loaded once
    at startup from a numpy file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from tsp_rl_kg.utils.config_files import load_config_file

if TYPE_CHECKING:
    from tsp_rl_kg.config import FeatureEncodingConfig

# Edge-type constants used by FeatureEncoder.encode_edge
EDGE_ADJACENCY = "adjacency"
EDGE_ENTITY_TERRAIN = "entity_terrain"
EDGE_PLAYER_TERRAIN = "player_terrain"

_EDGE_TYPE_ORDER = [EDGE_ADJACENCY, EDGE_ENTITY_TERRAIN, EDGE_PLAYER_TERRAIN]

_NUM_TERRAIN_TYPES = 6  # 0–5
_NUM_ENTITY_TYPES = 8  # 0–7
_PLAYER_INDEX = _NUM_TERRAIN_TYPES + _NUM_ENTITY_TYPES
_EXPECTED_EMBEDDING_ROWS = _PLAYER_INDEX + 1


@dataclass(frozen=True)
class SemanticSchema:
    terrain: dict[int, str]
    entity: dict[int, str]
    player: str


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


def _encode_categorical_edge(distance: int, edge_type: str, diag: float) -> torch.Tensor:
    norm_dist = distance / diag if diag > 0 else 0.0
    t = torch.zeros(1 + len(_EDGE_TYPE_ORDER), dtype=torch.float)
    t[0] = norm_dist
    idx = _EDGE_TYPE_ORDER.index(edge_type)
    t[1 + idx] = 1.0
    return t


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _schema_hash(path: str | Path) -> str:
    resolved = _resolve_path(path)
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _required_id_mapping(
    raw_mapping: object,
    *,
    section_name: str,
    required_ids: range,
) -> dict[int, str]:
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"semantic schema section '{section_name}' must be a mapping")

    parsed: dict[int, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"semantic schema section '{section_name}' contains non-integer key {raw_key!r}"
            ) from exc

        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(
                f"semantic schema section '{section_name}' contains invalid descriptor for id {key}"
            )

        parsed[key] = raw_value.strip()

    missing_ids = [idx for idx in required_ids if idx not in parsed]
    if missing_ids:
        raise ValueError(
            f"semantic schema section '{section_name}' is missing required ids {missing_ids}"
        )

    return parsed


def load_semantic_schema(path: str | Path) -> SemanticSchema:
    data = load_config_file(_resolve_path(path))
    terrain = _required_id_mapping(
        data.get("terrain"),
        section_name="terrain",
        required_ids=range(_NUM_TERRAIN_TYPES),
    )
    entity = _required_id_mapping(
        data.get("entity"),
        section_name="entity",
        required_ids=range(_NUM_ENTITY_TYPES),
    )

    player_section = data.get("player")
    if isinstance(player_section, dict):
        player_descriptor = player_section.get("descriptor")
    else:
        player_descriptor = player_section

    if not isinstance(player_descriptor, str) or not player_descriptor.strip():
        raise ValueError("semantic schema section 'player' must define a non-empty descriptor")

    return SemanticSchema(
        terrain=terrain,
        entity=entity,
        player=player_descriptor.strip(),
    )


def semantic_schema_descriptors(path: str | Path) -> list[str]:
    schema = load_semantic_schema(path)
    return [
        *[schema.terrain[idx] for idx in range(_NUM_TERRAIN_TYPES)],
        *[schema.entity[idx] for idx in range(_NUM_ENTITY_TYPES)],
        schema.player,
    ]


def embedding_metadata_path(embedding_path: str | Path) -> Path:
    return _resolve_path(embedding_path).with_suffix(".json")


def load_embedding_lookup_table(
    embedding_path: str | Path,
    schema_path: str | Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    resolved_embedding_path = _resolve_path(embedding_path)
    metadata_path = embedding_metadata_path(resolved_embedding_path)

    if not resolved_embedding_path.exists():
        raise ValueError(f"embedding file not found: {resolved_embedding_path}")
    if not metadata_path.exists():
        raise ValueError(f"embedding metadata file not found: {metadata_path}")

    embeddings = np.load(resolved_embedding_path)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embedding file must contain a 2D array, got shape {tuple(embeddings.shape)}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    expected_rows = len(semantic_schema_descriptors(schema_path))
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"embedding row count {embeddings.shape[0]} does not match semantic schema count "
            f"{expected_rows}"
        )

    metadata_rows = metadata.get("descriptor_count")
    if metadata_rows != expected_rows:
        raise ValueError(
            f"embedding metadata descriptor_count {metadata_rows!r} does not match schema count "
            f"{expected_rows}"
        )

    metadata_dim = metadata.get("embed_dim")
    if metadata_dim != int(embeddings.shape[1]):
        raise ValueError(
            f"embedding metadata embed_dim {metadata_dim!r} does not match array width "
            f"{embeddings.shape[1]}"
        )

    metadata_schema_hash = metadata.get("schema_hash")
    current_schema_hash = _schema_hash(schema_path)
    if metadata_schema_hash != current_schema_hash:
        raise ValueError(
            "embedding metadata schema_hash does not match the current semantic schema"
        )

    if embeddings.shape[0] != _EXPECTED_EMBEDDING_ROWS:
        raise ValueError(
            f"embedding row count must be {_EXPECTED_EMBEDDING_ROWS}, got {embeddings.shape[0]}"
        )

    return torch.tensor(embeddings, dtype=torch.float32), metadata


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
        return _encode_categorical_edge(distance, edge_type, self._diag)


class EmbeddingLookupEncoder:
    """Use pre-computed semantic embeddings for terrain/entity/player nodes."""

    def __init__(self, embedding_path: str | Path, schema_path: str | Path, grid_size: int = 20):
        self._grid_size = grid_size
        self._diag = math.sqrt(2) * grid_size
        self._embedding_path = _resolve_path(embedding_path)
        self._schema_path = _resolve_path(schema_path)
        self._embeddings, self._metadata = load_embedding_lookup_table(
            self._embedding_path,
            self._schema_path,
        )

    @property
    def node_dim(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def edge_dim(self) -> int:
        return 1 + len(_EDGE_TYPE_ORDER)

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def encode_terrain(self, raw: int) -> torch.Tensor:
        return self._embeddings[raw].clone()

    def encode_entity(self, raw: int) -> torch.Tensor:
        return self._embeddings[_NUM_TERRAIN_TYPES + raw].clone()

    def encode_player(self) -> torch.Tensor:
        return self._embeddings[_PLAYER_INDEX].clone()

    def encode_edge(self, distance: int, edge_type: str) -> torch.Tensor:
        return _encode_categorical_edge(distance, edge_type, self._diag)


def build_feature_encoder(
    feature_config: FeatureEncodingConfig,
    *,
    grid_size: int = 20,
) -> FeatureEncoder:
    if feature_config.strategy == "raw_int":
        return RawIntEncoder()
    if feature_config.strategy == "one_hot":
        return OneHotEncoder(grid_size=grid_size)
    if feature_config.strategy == "embedding_lookup":
        return EmbeddingLookupEncoder(
            embedding_path=feature_config.embedding_path,
            schema_path=feature_config.schema_path,
            grid_size=grid_size,
        )

    raise ValueError(f"Unsupported feature encoding strategy: {feature_config.strategy}")
