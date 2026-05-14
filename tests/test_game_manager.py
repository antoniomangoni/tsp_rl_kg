from __future__ import annotations

from types import SimpleNamespace

import pygame

from tsp_rl_kg.config import GameManagerConfig
from tsp_rl_kg.game_world.actions import ActionType
from tsp_rl_kg.game_world.game_manager import GameManager

def test_start_game_logs_human_controls(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = GameManagerConfig(
        num_tiles=5,
        screen_size=250,
        vision_range=1,
        headless=True,
        human_mode=True,
    )
    game_manager = GameManager(config=config)
    messages: list[str] = []

    class DummyRecorder:
        def __init__(self, _config):
            self.paths = SimpleNamespace(run_dir="dummy-run")

        def write_run_start(self, **_kwargs):
            return None

        @staticmethod
        def count_discovered_tiles(_grid):
            return 0

    monkeypatch.setattr(game_manager, "init_knowledge_graph", lambda projection: None)
    monkeypatch.setattr(
        "tsp_rl_kg.game_world.game_manager.PlayRecorder",
        DummyRecorder,
    )
    monkeypatch.setattr(
        "tsp_rl_kg.game_world.game_manager.logger.info",
        lambda message: messages.append(str(message)),
    )

    game_manager.start_game()

    assert messages[: len(GameManager.HUMAN_CONTROL_LINES)] == list(GameManager.HUMAN_CONTROL_LINES)


def _headless_game_manager(monkeypatch, tmp_path) -> GameManager:
    monkeypatch.chdir(tmp_path)
    config = GameManagerConfig(
        num_tiles=5,
        screen_size=250,
        vision_range=1,
        headless=True,
    )
    return GameManager(config=config)


def test_build_status_includes_resources_and_route_scores(monkeypatch, tmp_path):
    game_manager = _headless_game_manager(monkeypatch, tmp_path)
    game_manager.agent_controler.wood = 2
    game_manager.agent_controler.stone = 1

    rows = game_manager._build_status()

    assert isinstance(rows, list) and len(rows) == 2
    fields = {key: value for row in rows for key, value in row.items()}
    resource_max = game_manager.agent_controler.resource_max
    assert fields["Wood"] == f"2/{resource_max}"
    assert fields["Stone"] == f"1/{resource_max}"
    assert fields["Best route"] == game_manager.target_manager.target_route_energy
    assert fields["Current route"] == 0


def test_update_route_tracking_records_completed_route(monkeypatch, tmp_path):
    game_manager = _headless_game_manager(monkeypatch, tmp_path)
    outposts = list(game_manager.environment.outpost_locations)
    assert outposts, "world should generate at least one outpost"

    for index, (outpost_x, outpost_y) in enumerate(outposts, start=1):
        game_manager.agent.grid_x, game_manager.agent.grid_y = outpost_x, outpost_y
        game_manager.agent_controler.energy_spent = index * 10
        game_manager._update_route_tracking()

    assert game_manager.route_energy_list == [len(outposts) * 10]
    assert game_manager.visited_outposts == set()
    assert game_manager.route_start_energy == game_manager.agent_controler.energy_spent
