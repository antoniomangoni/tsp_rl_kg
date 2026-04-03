"""Tests for config-driven training entrypoints and experiment overrides."""

from __future__ import annotations

import json

from tsp_rl_kg import main as main_module
from tsp_rl_kg.config import AlgorithmName
from tsp_rl_kg.rl.training import run as run_module
from tsp_rl_kg.rl.training.ablation_study import AblationStudy
from tsp_rl_kg.rl.training.run import build_base_config


def test_build_base_config_allows_backend_selection():
    config = build_base_config(
        algorithm=AlgorithmName.DQN,
        total_timesteps=64,
        seeds=[7],
        number_of_environments=8,
        number_of_curricula=2,
    )

    assert config.algorithm.algorithm == AlgorithmName.DQN
    assert config.total_timesteps == 64
    assert config.simulation_manager.number_of_environments == 8
    assert "buffer_size" in config.algorithm.hyperparameters
    assert "n_steps" not in config.algorithm.hyperparameters
    assert config.feature_encoding.strategy == "one_hot"


def test_ablation_study_merges_algorithm_and_config_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_config = build_base_config(
        total_timesteps=32,
        seeds=[11],
        number_of_environments=8,
        number_of_curricula=2,
    )
    study = AblationStudy(base_config, kg_completeness_values=[0.5], experiments=[])

    experiment_config, kg_completeness, ablation = study._build_experiment_config(
        {
            "name": "dqn_override",
            "kg_completeness": 0.75,
            "algorithm": {
                "algorithm": AlgorithmName.DQN.value,
                "hyperparameters": {"buffer_size": 256, "learning_starts": 0},
            },
            "config_overrides": {"evaluation": {"n_eval_episodes": 2}},
        }
    )

    assert kg_completeness == 0.75
    assert ablation.disable_vision is False
    assert experiment_config.algorithm.algorithm == AlgorithmName.DQN
    assert experiment_config.algorithm.hyperparameters["buffer_size"] == 256
    assert experiment_config.algorithm.hyperparameters["learning_rate"] == 1e-4
    assert experiment_config.evaluation.n_eval_episodes == 2
    assert base_config.algorithm.algorithm == AlgorithmName.PPO


def test_ablation_study_merges_feature_encoding_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_config = build_base_config(
        total_timesteps=32,
        seeds=[11],
        number_of_environments=8,
        number_of_curricula=2,
    )
    study = AblationStudy(base_config, kg_completeness_values=[0.5], experiments=[])

    experiment_config, _, _ = study._build_experiment_config(
        {
            "name": "raw_int_override",
            "config_overrides": {"feature_encoding": {"strategy": "raw_int"}},
        }
    )

    assert experiment_config.feature_encoding.strategy == "raw_int"
    assert base_config.feature_encoding.strategy == "one_hot"


def test_main_builds_training_config_for_requested_algorithm():
    config = main_module._build_training_config(
        algorithm=AlgorithmName.DQN,
        timesteps=64,
    )

    assert config.algorithm.algorithm == AlgorithmName.DQN
    assert config.total_timesteps == 64
    assert config.game_manager.headless is True
    assert config.algorithm.hyperparameters["buffer_size"] == 1_024
    assert config.episode.max_episode_steps == 128


def test_main_train_loads_external_json_config(monkeypatch, tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "main": {
                    "train": {
                        "algorithm": {
                            "algorithm": "PPO",
                            "hyperparameters": {"n_steps": 16, "batch_size": 8},
                        },
                        "simulation_manager": {
                            "number_of_environments": 9,
                            "number_of_curricula": 3,
                        },
                        "total_timesteps": 99,
                        "seeds": [5],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_training_mode(config):
        captured["config"] = config

    monkeypatch.setattr(main_module, "_run_training_mode", fake_run_training_mode)

    exit_code = main_module.main(
        ["train", "--config", str(config_path), "--algorithm", "DQN", "--timesteps", "64"]
    )

    assert exit_code == 0
    assert captured["config"].algorithm.algorithm == AlgorithmName.DQN
    assert captured["config"].total_timesteps == 64
    assert captured["config"].simulation_manager.number_of_environments == 9
    assert captured["config"].algorithm.hyperparameters["buffer_size"] == 1_024


def test_main_dispatches_train_mode(monkeypatch):
    called = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_training_mode(config):
        called["algorithm"] = config.algorithm.algorithm

    monkeypatch.setattr(main_module, "_run_training_mode", fake_run_training_mode)

    exit_code = main_module.main(["train", "--algorithm", "DQN"])

    assert exit_code == 0
    assert called == {"algorithm": AlgorithmName.DQN}


def test_run_module_loads_external_toml_study_config(monkeypatch, tmp_path):
    config_path = tmp_path / "study.toml"
    config_path.write_text(
        """
[study]
mlflow_experiment_name = "external-study"
kg_completeness_values = [0.4]

[study.base_config.algorithm]
algorithm = "DQN"

[study.base_config.game_manager]
num_tiles = 6
screen_size = 24
vision_range = 1
headless = true

[study.base_config.simulation_manager]
number_of_environments = 12
number_of_curricula = 3
min_episodes_per_curriculum = 2

[study.base_config]
total_timesteps = 77
seeds = [9]

[[study.experiments]]
name = "loaded_experiment"
kg_completeness = 0.4

[study.experiments.algorithm]
algorithm = "DQN"

[study.experiments.algorithm.hyperparameters]
buffer_size = 128
learning_starts = 0
""".strip(),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(run_module, "configure_logging", lambda **kwargs: None)

    def fake_run(self):
        captured["algorithm"] = self.base_config.algorithm.algorithm
        captured["timesteps"] = self.base_config.total_timesteps
        captured["mlflow_experiment_name"] = self.mlflow_experiment_name
        captured["experiments"] = self.experiments

    monkeypatch.setattr(run_module.AblationStudy, "run", fake_run)

    exit_code = run_module.main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured["algorithm"] == AlgorithmName.DQN
    assert captured["timesteps"] == 77
    assert captured["mlflow_experiment_name"] == "external-study"
    assert captured["experiments"][0]["name"] == "loaded_experiment"
