from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tsp_rl_kg.graph.feature_encoder import embedding_metadata_path, semantic_schema_descriptors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate semantic node embeddings from a schema using sentence-transformers."
    )
    parser.add_argument("--schema", required=True, help="Path to semantic_schema.toml")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name")
    parser.add_argument("--output", required=True, help="Output .npy file path")
    return parser


def _schema_hash(schema_path: Path) -> str:
    return hashlib.sha256(schema_path.read_bytes()).hexdigest()


def main() -> int:
    args = _build_parser().parse_args()

    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for this script. Run `uv sync --extra embed-gen`."
        ) from exc

    schema_path = Path(args.schema).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptors = semantic_schema_descriptors(schema_path)
    model = sentence_transformers.SentenceTransformer(args.model)
    embeddings = model.encode(descriptors)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(output_path, embeddings)
    metadata = {
        "model_name": args.model,
        "descriptor_count": len(descriptors),
        "embed_dim": int(embeddings.shape[1]),
        "schema_hash": _schema_hash(schema_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    embedding_metadata_path(output_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "model_name": args.model,
                "descriptor_count": len(descriptors),
                "embed_dim": int(embeddings.shape[1]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
