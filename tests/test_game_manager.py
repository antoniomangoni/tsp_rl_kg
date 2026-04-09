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
