from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsp_rl_kg.config import GameManagerConfig
from tsp_rl_kg.game_world.actions import ActionType


@dataclass
class PlayRunPaths:
    run_dir: Path
    metadata_path: Path
    steps_path: Path
    visual_dir: Path


class PlayRecorder:
    """Persist play-mode metadata and per-step telemetry to disk."""

    def __init__(self, config: GameManagerConfig, base_dir: str | Path = "results") -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.paths = self._create_paths(Path(base_dir), timestamp)
        self._config = config
        self._step_count = 0

    @staticmethod
    def _create_paths(base_dir: Path, timestamp: str) -> PlayRunPaths:
        run_dir = base_dir / f"play_{timestamp}"
        visual_dir = run_dir / "visual_benchmark"
        run_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)
        return PlayRunPaths(
            run_dir=run_dir,
            metadata_path=run_dir / "run_metadata.json",
            steps_path=run_dir / "step_logs.jsonl",
            visual_dir=visual_dir,
        )

    def write_run_start(self, *, player_pos: tuple[int, int], discovered_tiles: int) -> None:
        payload = {
            "run_started_at": datetime.now(UTC).isoformat(),
            "config": asdict(self._config),
            "run_dir": str(self.paths.run_dir),
            "visual_benchmark_dir": str(self.paths.visual_dir),
            "initial_state": {
                "player_pos": [int(player_pos[0]), int(player_pos[1])],
                "discovered_tiles": int(discovered_tiles),
            },
            "status": "running",
        }
        self.paths.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_step(
        self,
        *,
        step_index: int,
        action: ActionType,
        player_pos: tuple[int, int],
        energy_spent: int,
        wood: int,
        stone: int,
        discovered_tiles: int,
        frame_path: str | None,
    ) -> None:
        payload = {
            "step": int(step_index),
            "action": action.name,
            "action_id": int(action.value),
            "player_pos": [int(player_pos[0]), int(player_pos[1])],
            "energy_spent": int(energy_spent),
            "inventory": {"wood": int(wood), "stone": int(stone)},
            "discovered_tiles": int(discovered_tiles),
            "frame_path": frame_path,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self.paths.steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))
            fh.write("\n")
        self._step_count += 1

    def write_run_end(self, *, end_reason: str) -> None:
        payload = json.loads(self.paths.metadata_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "status": "completed",
                "run_completed_at": datetime.now(UTC).isoformat(),
                "total_steps": int(self._step_count),
                "end_reason": end_reason,
            }
        )
        self.paths.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def count_discovered_tiles(discovered_grid: Any) -> int:
        return int(discovered_grid.sum())
