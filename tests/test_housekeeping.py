"""Configuration, degenerate curricula, and render-resource regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from tsp_rl_kg.config import AgentModelConfig, GameManagerConfig, TrainingConfig
from tsp_rl_kg.game_world.world_template import WorldTemplate
from tsp_rl_kg.rl.simulation_manager import SimulationManager


@pytest.mark.parametrize(
    "key",
    ["replay", "sequence", "world_model", "replay_config", "sequence_config", "world_model_config"],
)
def test_obsolete_training_settings_rejected(key):
    with pytest.raises(ValueError, match="Remove these keys"):
        TrainingConfig.from_dict({key: {}})


@pytest.mark.parametrize(
    "values",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"seeds": [-1]},
        {"total_timesteps": 0},
        {"model_args": {"num_actions": 4}},
    ],
)
def test_invalid_training_relationships(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"vision_num_conv_layers": 5},
        {"graph_num_gat_layers": 5},
        {"graph_fc_dims": []},
        {"dropout": float("nan")},
        {"features_dim": 0},
    ],
)
def test_invalid_model_relationships(values):
    with pytest.raises(ValueError):
        AgentModelConfig(**values)


@pytest.mark.parametrize("energies", [[4], [4, 4, 4], [4, 2, 2, 6]])
def test_curriculum_covers_every_world(energies):
    worlds = [
        SimpleNamespace(target_manager=SimpleNamespace(target_route_energy=e)) for e in energies
    ]
    manager = SimulationManager(GameManagerConfig(), game_managers=worlds, number_of_curricula=10)
    assert manager.number_of_environments == len(worlds)
    visited = set()
    for level, start in enumerate(manager.curriculum_indices):
        manager.current_curriculum_index = level
        current = start
        while current not in visited:
            visited.add(current)
            current = manager.get_next_game_in_curriculum(current)
    assert visited == set(range(len(worlds)))
    assert manager.get_next_game_manager() is False
    assert manager.advance_curriculum() == -1


def test_empty_world_pool_rejected():
    with pytest.raises(ValueError, match="at least one valid world"):
        SimulationManager(GameManagerConfig(), game_managers=[])


def test_render_modes_are_per_world(headless_environment):
    template = WorldTemplate.capture(headless_environment)
    rendered = template.restore(tile_size=8, headless=False)
    headless = template.restore(tile_size=8, headless=True)
    for env in (rendered, headless):
        x, y = env.player.grid_x, env.player.grid_y
        env.place_path(x, y)
        path = env.terrain_object_grid[x, y].entity_on_tile
        assert path._headless == env.headless
        if not env.headless:
            assert path.image.get_size() == (8, 8)
        env.drop_rock_in_water(x, y, 0)
        assert env.terrain_object_grid[x, y]._headless == env.headless
    np.testing.assert_array_equal(rendered.entity_index_grid, headless.entity_index_grid)


def test_world_generation_accepts_last_retry(monkeypatch):
    import tsp_rl_kg.rl.simulation_manager as module

    calls = []
    world = SimpleNamespace(
        environment=SimpleNamespace(outpost_locations=[(0, 0), (1, 1), (2, 2)]),
        target_manager=SimpleNamespace(target_route_energy=4),
    )

    def generate(**kwargs):
        calls.append(True)
        if len(calls) < 20:
            raise ValueError("world requires more outposts")
        return world

    monkeypatch.setattr(module, "GameManager", generate)
    manager = SimulationManager(GameManagerConfig(), number_of_environments=1)
    assert len(calls) == 20
    assert manager.game_managers == [world]
