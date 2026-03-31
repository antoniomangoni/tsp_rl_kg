"""Tests for the reusable encoder core and the SB3 AgentModel adapter."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from tsp_rl_kg.config import AgentModelConfig
from tsp_rl_kg.rl.agent_model import AgentModel
from tsp_rl_kg.rl.encoders import HybridEncoder

MAX_NODES = 8
MAX_EDGES = 12
NUM_NODE_FEATURES = 4
NUM_EDGE_FEATURES = 2
VISION_SHAPE = (3, 32, 32)
SMALL_VISION_SHAPE = (3, 12, 12)
BATCH_SIZE = 2


def _make_observation_space(vision_shape: tuple[int, int, int] = VISION_SHAPE) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "vision": gym.spaces.Box(low=0.0, high=1.0, shape=vision_shape, dtype=np.float32),
            "node_features": gym.spaces.Box(
                low=-1.0,
                high=1e4,
                shape=(MAX_NODES, NUM_NODE_FEATURES),
                dtype=np.float32,
            ),
            "edge_attr": gym.spaces.Box(
                low=-1.0,
                high=1e4,
                shape=(MAX_EDGES, NUM_EDGE_FEATURES),
                dtype=np.float32,
            ),
            "edge_index": gym.spaces.Box(
                low=0,
                high=MAX_NODES - 1,
                shape=(2, MAX_EDGES),
                dtype=np.int64,
            ),
        }
    )


def _make_observations(
    vision_shape: tuple[int, int, int] = VISION_SHAPE,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return {
        "vision": torch.rand((BATCH_SIZE, *vision_shape), generator=generator),
        "node_features": torch.rand(
            (BATCH_SIZE, MAX_NODES, NUM_NODE_FEATURES),
            generator=generator,
        ),
        "edge_attr": torch.rand(
            (BATCH_SIZE, MAX_EDGES, NUM_EDGE_FEATURES),
            generator=generator,
        ),
        "edge_index": torch.randint(
            low=0,
            high=MAX_NODES,
            size=(BATCH_SIZE, 2, MAX_EDGES),
            generator=generator,
        ),
    }


def test_hybrid_encoder_core_output_shape():
    config = AgentModelConfig(features_dim=192)
    encoder = HybridEncoder(
        observation_space=_make_observation_space(),
        features_dim=config.features_dim,
        model_config=config,
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(_make_observations())

    assert output.shape == (BATCH_SIZE, config.features_dim)


def test_hybrid_encoder_supports_small_vision_inputs():
    config = AgentModelConfig(features_dim=192)
    encoder = HybridEncoder(
        observation_space=_make_observation_space(SMALL_VISION_SHAPE),
        features_dim=config.features_dim,
        model_config=config,
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(_make_observations(SMALL_VISION_SHAPE))

    assert output.shape == (BATCH_SIZE, config.features_dim)


def test_agent_model_adapter_matches_core_output():
    observation_space = _make_observation_space()
    config = AgentModelConfig(features_dim=192)
    observations = _make_observations()

    torch.manual_seed(7)
    core = HybridEncoder(
        observation_space=observation_space,
        features_dim=config.features_dim,
        model_config=config,
    )
    torch.manual_seed(7)
    adapter = AgentModel(
        observation_space=observation_space,
        features_dim=config.features_dim,
        model_config=config,
    )
    core.eval()
    adapter.eval()

    with torch.no_grad():
        core_output = core(observations)
        adapter_output = adapter(observations)

    assert torch.allclose(adapter_output, core_output)


def test_agent_model_exposes_core_modules_for_callbacks():
    adapter = AgentModel(
        observation_space=_make_observation_space(),
        model_config=AgentModelConfig(),
    )

    assert adapter.vision_processor is adapter.encoder.vision_processor
    assert adapter.graph_processor is adapter.encoder.graph_processor
    assert adapter.fc is adapter.encoder.fc
    assert adapter.dropout is adapter.encoder.dropout
