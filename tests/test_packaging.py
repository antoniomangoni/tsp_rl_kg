"""Build and import the wheel outside the source checkout, without dependency downloads."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_wheel_import_and_cli_outside_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build = tmp_path / "build"
    build.mkdir()
    shutil.copytree(
        root / "src/tsp_rl_kg",
        build / "src/tsp_rl_kg",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(root / name, build / name)
    subprocess.run(
        [
            sys.executable,
            "-c",
            'from setuptools.build_meta import build_wheel; build_wheel("dist")',
        ],
        cwd=build,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (wheel,) = (build / "dist").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        assert "tsp_rl_kg/assets/pixel_art/player.png" in archive.namelist()
        archive.extractall(installed)
    code = """
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import tsp_rl_kg
assert pathlib.Path(tsp_rl_kg.__file__).is_relative_to(sys.argv[1])
from tsp_rl_kg.game_world.entities import Player
assert Player(0, 0, 8).image.get_size() == (8, 8)
assert Player(0, 0, 16).image.get_size() == (16, 16)
from tsp_rl_kg.main import main
assert main(["--help"]) == 0
"""
    subprocess.run(
        [sys.executable, "-c", code, str(installed)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(installed)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
