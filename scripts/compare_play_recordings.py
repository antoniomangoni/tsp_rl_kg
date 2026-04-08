#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

FRAME_PATTERN = re.compile(r"game_world_(\d+)\.jpe?g$")


@dataclass
class RunSummary:
    run_dir: Path
    total_steps: int
    logged_steps: int
    expected_visual_frames: int
    visual_frames_found: int
    missing_frame_indices: list[int]


def _load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _count_logged_steps(run_dir: Path) -> int:
    step_logs = run_dir / "step_logs.jsonl"
    if not step_logs.exists():
        return 0
    with step_logs.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _collect_visual_indices(visual_dir: Path) -> list[int]:
    if not visual_dir.exists():
        return []
    indices: list[int] = []
    for file in visual_dir.iterdir():
        match = FRAME_PATTERN.search(file.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def summarize_run(run_dir: Path) -> RunSummary:
    metadata = _load_metadata(run_dir)
    total_steps = int(metadata.get("total_steps", 0))
    logged_steps = _count_logged_steps(run_dir)
    visual_dir = Path(metadata.get("visual_benchmark_dir", run_dir / "visual_benchmark"))

    expected = list(range(0, total_steps, 10))
    found = _collect_visual_indices(visual_dir)
    missing = [idx for idx in expected if idx not in set(found)]

    return RunSummary(
        run_dir=run_dir,
        total_steps=total_steps,
        logged_steps=logged_steps,
        expected_visual_frames=len(expected),
        visual_frames_found=len(found),
        missing_frame_indices=missing,
    )


def _resolve_runs(run_paths: list[str], root: str | None) -> list[Path]:
    if run_paths:
        return [Path(p) for p in run_paths]
    if root is None:
        raise ValueError("Provide at least one run path or --root.")
    root_path = Path(root)
    return sorted(path for path in root_path.glob("play_*") if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate play recorder outputs and compare recorder expectations to visual-only "
            "benchmark frame files."
        )
    )
    parser.add_argument("run", nargs="*", help="Specific play run directories to compare.")
    parser.add_argument("--root", help="Root directory containing play_* run folders.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to save machine-readable comparison output.",
    )
    args = parser.parse_args()

    runs = _resolve_runs(args.run, args.root)
    summaries = [summarize_run(run) for run in runs]

    for summary in summaries:
        print(f"Run: {summary.run_dir}")
        print(
            "  total_steps={0} logged_steps={1} expected_frames={2} found_frames={3}".format(
                summary.total_steps,
                summary.logged_steps,
                summary.expected_visual_frames,
                summary.visual_frames_found,
            )
        )
        if summary.missing_frame_indices:
            print(f"  missing_frames={summary.missing_frame_indices}")
        else:
            print("  missing_frames=[]")

    if args.json_output is not None:
        payload = {
            "runs": [
                {
                    "run_dir": s.run_dir.as_posix(),
                    "total_steps": s.total_steps,
                    "logged_steps": s.logged_steps,
                    "expected_visual_frames": s.expected_visual_frames,
                    "visual_frames_found": s.visual_frames_found,
                    "missing_frame_indices": s.missing_frame_indices,
                }
                for s in summaries
            ]
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
