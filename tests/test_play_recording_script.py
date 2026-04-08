from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_play_recordings import summarize_run
from tsp_rl_kg.config import GameManagerConfig


def test_game_manager_config_rejects_invalid_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        GameManagerConfig(max_steps=0)


def test_summarize_run_compares_visual_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "play_20260101_000000"
    visual_dir = run_dir / "visual_benchmark"
    visual_dir.mkdir(parents=True)

    metadata = {
        "visual_benchmark_dir": visual_dir.as_posix(),
        "total_steps": 25,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "step_logs.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    (visual_dir / "game_world_0.jpeg").write_bytes(b"0")
    (visual_dir / "game_world_20.jpeg").write_bytes(b"1")

    summary = summarize_run(run_dir)
    assert summary.expected_visual_frames == 3
    assert summary.visual_frames_found == 2
    assert summary.missing_frame_indices == [10]
    assert summary.logged_steps == 2
