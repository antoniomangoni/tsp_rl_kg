"""Tests for config-driven training entrypoints and experiment overrides."""

from __future__ import annotations

from tsp_rl_kg import main as main_module
from tsp_rl_kg.config import AlgorithmName
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


def test_main_builds_training_config_for_requested_algorithm():
    args = main_module.build_parser().parse_args(
        ["train", "--algorithm", "DQN", "--timesteps", "64"]
    )
    config = main_module._build_training_config(args)

    assert config.algorithm.algorithm == AlgorithmName.DQN
    assert config.total_timesteps == 64
    assert config.game_manager.headless is True
    assert config.algorithm.hyperparameters["buffer_size"] == 1_024
    assert config.episode.max_episode_steps == 128


def test_main_dispatches_train_mode(monkeypatch):
    called = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda **kwargs: None)

    def fake_run_training_mode(args):
        called["algorithm"] = args.algorithm

    monkeypatch.setattr(main_module, "_run_training_mode", fake_run_training_mode)

    exit_code = main_module.main(["train", "--algorithm", "DQN"])

    assert exit_code == 0
    assert called == {"algorithm": "DQN"}
