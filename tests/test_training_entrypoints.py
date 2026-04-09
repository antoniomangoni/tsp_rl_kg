"""Tests for config-driven training entrypoints and experiment overrides."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from tsp_rl_kg import main as main_module
from tsp_rl_kg.config import AlgorithmName
from tsp_rl_kg.rl.training import run as run_module
from tsp_rl_kg.rl.training.run import build_base_config


def test_project_scripts_include_short_and_long_cli_names():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["tsp"] == "tsp_rl_kg.main:main"
    assert scripts["tsp-study"] == "tsp_rl_kg.rl.training.run:main"
    assert scripts["tsp-rl-kg"] == "tsp_rl_kg.main:main"
    assert scripts["tsp-rl-kg-study"] == "tsp_rl_kg.rl.training.run:main"


def test_main_module_does_not_eager_import_runtime_dependencies():
    assert not hasattr(main_module, "GameManager")
    assert not hasattr(main_module, "SimulationManager")
    assert not hasattr(main_module, "Trainer")


def test_study_module_does_not_eager_import_runtime_dependencies():
    assert not hasattr(run_module, "AblationStudy")


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
    from tsp_rl_kg.rl.training.ablation_study import AblationStudy

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
    from tsp_rl_kg.rl.training.ablation_study import AblationStudy

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


def test_main_run_app_uses_canonical_prog_name_for_explicit_argv(monkeypatch):
    captured = {}

    def fake_app(*, args, prog_name, standalone_mode):
        captured["args"] = args
        captured["prog_name"] = prog_name
        captured["standalone_mode"] = standalone_mode
        return 0

    monkeypatch.setattr(main_module, "app", fake_app)

    exit_code = main_module._run_app(["--help"])

    assert exit_code == 0
    assert captured["args"] == ["--help"]
    assert captured["prog_name"] == "tsp-rl-kg"
    assert captured["standalone_mode"] is False


def test_main_run_app_derives_prog_name_from_sys_argv(monkeypatch):
    captured = {}

    def fake_app(*, args, prog_name, standalone_mode):
        captured["args"] = args
        captured["prog_name"] = prog_name
        captured["standalone_mode"] = standalone_mode
        return 0

    monkeypatch.setattr(main_module, "app", fake_app)
    monkeypatch.setattr(main_module.sys, "argv", ["/tmp/bin/tsp"])

    exit_code = main_module._run_app()

    assert exit_code == 0
    assert captured["args"] is None
    assert captured["prog_name"] == "tsp"
    assert captured["standalone_mode"] is False


def test_main_help_lists_command_descriptions_without_pygame_banner(capsys):
    exit_code = main_module._run_app(["--help"])
    captured = capsys.readouterr()
    combined_output = captured.out + captured.err

    assert exit_code == 0
    assert "Run a manual world session." in captured.out
    assert "Run the RL training workflow." in captured.out
    assert "Generate and export worlds." in captured.out
    assert "pygame-ce" not in combined_output


def test_main_dispatches_play_mode_with_keyboard_control(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_play_mode(config):
        captured["config"] = config

    monkeypatch.setattr(main_module, "_run_play_mode", fake_run_play_mode)

    exit_code = main_module.main(["play"])

    assert exit_code == 0
    assert captured["config"].human_mode is True
    assert captured["config"].use_random_human_actions is False
    assert captured["config"].headless is False


def test_main_play_random_actions_flag_disables_human_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_play_mode(config):
        captured["config"] = config

    monkeypatch.setattr(main_module, "_run_play_mode", fake_run_play_mode)

    exit_code = main_module.main(["play", "--random-actions"])

    assert exit_code == 0
    assert captured["config"].human_mode is False
    assert captured["config"].use_random_human_actions is False
    assert captured["config"].headless is False


def test_main_play_human_flag_overrides_config(monkeypatch, tmp_path):
    config_path = tmp_path / "play_config.json"
    config_path.write_text(
        json.dumps(
            {
                "main": {
                    "play": {
                        "game_manager": {
                            "human_mode": False,
                            "use_random_human_actions": True,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_play_mode(config):
        captured["config"] = config

    monkeypatch.setattr(main_module, "_run_play_mode", fake_run_play_mode)

    exit_code = main_module.main(["play", "--config", str(config_path), "--human"])

    assert exit_code == 0
    assert captured["config"].human_mode is True
    assert captured["config"].use_random_human_actions is False


def test_main_play_rejects_headless_keyboard_control(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)
    called = {"run_play_mode": False}

    def fake_run_play_mode(config):
        called["run_play_mode"] = True

    monkeypatch.setattr(main_module, "_run_play_mode", fake_run_play_mode)

    exit_code = main_module.main(["play", "--headless"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert called["run_play_mode"] is False
    assert "Keyboard-controlled play requires a visible window." in captured.err


def test_main_play_allows_headless_random_actions(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_play_mode(config):
        captured["config"] = config

    monkeypatch.setattr(main_module, "_run_play_mode", fake_run_play_mode)

    exit_code = main_module.main(["play", "--headless", "--random-actions"])

    assert exit_code == 0
    assert captured["config"].headless is True
    assert captured["config"].human_mode is False


def test_main_simulate_keeps_non_human_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_simulation_mode(game_manager_config, simulation_manager_config):
        captured["game_manager_config"] = game_manager_config
        captured["simulation_manager_config"] = simulation_manager_config

    monkeypatch.setattr(main_module, "_run_simulation_mode", fake_run_simulation_mode)

    exit_code = main_module.main(["simulate"])

    assert exit_code == 0
    assert captured["game_manager_config"].human_mode is False
    assert captured["game_manager_config"].use_random_human_actions is False


def test_study_run_app_uses_canonical_prog_name_for_explicit_argv(monkeypatch):
    captured = {}

    def fake_app(*, args, prog_name, standalone_mode):
        captured["args"] = args
        captured["prog_name"] = prog_name
        captured["standalone_mode"] = standalone_mode
        return 0

    monkeypatch.setattr(run_module, "app", fake_app)

    exit_code = run_module._run_app(["--help"])

    assert exit_code == 0
    assert captured["args"] == ["--help"]
    assert captured["prog_name"] == "tsp-rl-kg-study"
    assert captured["standalone_mode"] is False


def test_study_run_app_derives_prog_name_from_sys_argv(monkeypatch):
    captured = {}

    def fake_app(*, args, prog_name, standalone_mode):
        captured["args"] = args
        captured["prog_name"] = prog_name
        captured["standalone_mode"] = standalone_mode
        return 0

    monkeypatch.setattr(run_module, "app", fake_app)
    monkeypatch.setattr(run_module.sys, "argv", ["/tmp/bin/tsp-study"])

    exit_code = run_module._run_app()

    assert exit_code == 0
    assert captured["args"] is None
    assert captured["prog_name"] == "tsp-study"
    assert captured["standalone_mode"] is False


def test_study_help_includes_description_without_pygame_banner(capsys):
    exit_code = run_module._run_app(["--help"])
    captured = capsys.readouterr()
    combined_output = captured.out + captured.err

    assert exit_code == 0
    assert "Run ablation studies across TSP RL training configurations." in captured.out
    assert "pygame-ce" not in combined_output


def test_main_train_benchmark_writes_standardized_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_training_mode(config):
        assert config.ablation.disable_graph is True
        assert config.ablation.disable_vision is False
        return {
            "mean_reward": 1.25,
            "std_reward": 0.5,
            "metrics_file": "results/manual_x/metrics.csv",
            "model_path": "results/manual_x/model.zip",
            "stats_file": "results/manual_x/profile_stats.txt",
        }

    monkeypatch.setattr(main_module, "_run_training_mode", fake_run_training_mode)

    exit_code = main_module.main(["train", "--benchmark"])

    assert exit_code == 0
    benchmark_files = sorted((tmp_path / "results").glob("benchmark_*.json"))
    assert len(benchmark_files) == 1

    payload = json.loads(benchmark_files[0].read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "vision_only"
    assert "timestamp_utc" in payload
    assert payload["ablation"]["disable_graph"] is True

    required_metric_keys = {"mean_reward", "std_reward"}
    assert required_metric_keys.issubset(payload["metrics"])


def test_run_module_loads_external_toml_study_config(monkeypatch, tmp_path):
    from tsp_rl_kg.rl.training.ablation_study import AblationStudy

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

    monkeypatch.setattr(AblationStudy, "run", fake_run)

    exit_code = run_module.main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured["algorithm"] == AlgorithmName.DQN
    assert captured["timesteps"] == 77
    assert captured["mlflow_experiment_name"] == "external-study"
    assert captured["experiments"][0]["name"] == "loaded_experiment"
