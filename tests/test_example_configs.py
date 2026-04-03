"""Regression tests for the shipped example config files."""

from __future__ import annotations

from pathlib import Path

from tsp_rl_kg import main as main_module
from tsp_rl_kg.config import AlgorithmName
from tsp_rl_kg.main import _build_training_config
from tsp_rl_kg.rl.training.run import create_default_ablation_study
from tsp_rl_kg.utils.config_files import load_config_file


def test_example_train_json_loads_into_training_config():
    config_path = Path("configs/train.json")

    loaded_config = load_config_file(config_path)
    config = _build_training_config(loaded_config)

    assert config.algorithm.algorithm == AlgorithmName.PPO
    assert config.total_timesteps == 8
    assert config.evaluation.n_eval_episodes == 1
    assert config.feature_encoding.strategy == "one_hot"


def test_example_train_namespaced_json_loads_into_training_config():
    config_path = Path("configs/train_namespaced.json")

    loaded_config = main_module._load_cli_config(
        config_path,
        ("main", "train"),
        ("train",),
        ("training",),
        ("base_config",),
    )
    config = _build_training_config(loaded_config)

    assert config.algorithm.algorithm == AlgorithmName.PPO
    assert config.total_timesteps == 8
    assert config.simulation_manager.number_of_environments == 8
    assert config.feature_encoding.strategy == "one_hot"


def test_example_ablation_toml_loads_into_study():
    config_path = Path("configs/ablation.toml")

    study = create_default_ablation_study(config_path=config_path)

    assert study.base_config.algorithm.algorithm == AlgorithmName.PPO
    assert study.base_config.total_timesteps == 8
    assert study.base_config.feature_encoding.strategy == "one_hot"
    assert len(study.experiments) == 4
    assert study.experiments[-2]["algorithm"]["algorithm"] == "DQN"
    assert study.experiments[-1]["config_overrides"]["feature_encoding"]["strategy"] == "raw_int"
