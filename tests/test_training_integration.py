"""Bounded CPU integration tests; all artifacts live in pytest's temporary directory."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN, PPO

from tsp_rl_kg.config import (
    AgentModelConfig,
    AlgorithmConfig,
    EpisodeConfig,
    EvaluationConfig,
    GameManagerConfig,
    ModelArgs,
    SimulationManagerConfig,
    TrainingConfig,
)
from tsp_rl_kg.rl.training.ablation_study import AblationStudy, StudyFailedError
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.rl.training.trainer import Trainer

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_cpu_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    old_threads = torch.get_num_threads()
    old_uri = mlflow.get_tracking_uri()
    torch.set_num_threads(1)
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path / "mlflow.db"}')
    yield
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)
    torch.set_num_threads(old_threads)


def small_config(algorithm="PPO"):
    hyperparameters = (
        {"n_steps": 8, "batch_size": 4, "n_epochs": 1}
        if algorithm == "PPO"
        else {
            "buffer_size": 32,
            "learning_starts": 0,
            "batch_size": 4,
            "train_freq": 1,
        }
    )
    return TrainingConfig(
        game_manager=GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        simulation_manager=SimulationManagerConfig(number_of_environments=8, number_of_curricula=2),
        episode=EpisodeConfig(max_episode_steps=4, max_steps_without_progress=3),
        evaluation=EvaluationConfig(eval_freq=4, n_eval_episodes=2),
        algorithm=AlgorithmConfig(algorithm=algorithm, verbose=0, hyperparameters=hyperparameters),
        agent_model=AgentModelConfig(
            features_dim=8,
            vision_num_conv_layers=1,
            vision_conv_channels=[4],
            vision_fc_dims=[8],
            graph_num_gat_layers=1,
            graph_gat_heads=[1],
            graph_fc_dims=[8],
            gat_hidden_dim=4,
            dropout=0,
        ),
        total_timesteps=8,
        seeds=[42],
    )


def test_supplied_study_has_every_seed_and_artifact(tmp_path):
    from tsp_rl_kg.rl.training.run import main

    assert main(["--config", str(ROOT / "configs/ablation.toml")]) == 0
    (result_path,) = tmp_path.glob("results/*/ablation_study_results.json")
    results = json.loads(result_path.read_text())
    assert set(results) == {"ppo_baseline", "vision_only", "dqn_baseline", "raw_int_baseline"}
    for experiment in results.values():
        assert experiment["succeeded"] == experiment["attempted"] == 1
        assert experiment["failed"] == 0
        result = experiment["seed_results"][0]["result"]
        assert np.isfinite(result["mean_reward"])
        assert result["config"]["observation_schema_version"] == 2
        assert result["config"]["knowledge_semantics"] == "initial_prior_v1"
        assert Path(result["model_path"]).is_file()
        assert len(Path(result["metrics_file"]).read_text().splitlines()) > 1
        assert (Path(result["model_path"]).parent / "eval/static_data.csv").is_file()
    assert not list(tmp_path.glob("results/play_*"))
    assert not (tmp_path / "Writing").exists()


@pytest.mark.parametrize("algorithm", ["PPO", "DQN"])
def test_real_update_save_reload_and_periodic_evaluation(tmp_path, algorithm):
    trainer = Trainer(0.5, results_dir=str(tmp_path / "results"))
    trainer.setup(small_config(algorithm), seed=42)
    backend = trainer.model_trainer.backend
    initial = {
        key: value.detach().clone() for key, value in backend.model.policy.state_dict().items()
    }
    result = trainer.run("training")
    assert backend.model._n_updates > 0
    assert any(
        not torch.equal(initial[key], value)
        for key, value in backend.model.policy.state_dict().items()
    )
    evaluations = np.load(tmp_path / "results/training/evaluations.npz")
    assert evaluations["results"].shape == (2, 2)
    assert np.isfinite(evaluations["results"]).all()
    model_type = PPO if algorithm == "PPO" else DQN
    loaded = model_type.load(result["model_path"], device="cpu")
    obs, _ = trainer.eval_env.reset(seed=42, options={"world_index": 0})
    old_action, _ = backend.predict(obs)
    action, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_array_equal(old_action, action)


def test_held_out_pool_and_fixed_evaluation_schedule():
    cfg = small_config()
    manager = EnvironmentManager(
        cfg.game_manager, cfg.simulation_manager, ModelArgs(), None, seed=42
    )
    train = manager.make_env()
    evaluation = manager.make_eval_env(train.simulation_manager.game_managers)
    repeat = manager.make_eval_env(train.simulation_manager.game_managers)
    try:
        train_ids = {g.world_template.fingerprint for g in train.simulation_manager.game_managers}
        eval_ids = [
            g.world_template.fingerprint for g in evaluation.simulation_manager.game_managers
        ]
        assert train_ids.isdisjoint(eval_ids)
        assert eval_ids == [
            g.world_template.fingerprint for g in repeat.simulation_manager.game_managers
        ]
        sequences = []
        for _ in range(2):
            evaluation.begin_evaluation()
            sequence = []
            for _ in range(3):
                evaluation.reset()
                sequence.append(evaluation.current_game_index)
            sequences.append(sequence)
        assert sequences == [[0, 1, 2], [0, 1, 2]]
        assert evaluation.simulation_manager.current_curriculum_episodes == 0
    finally:
        for env in (train, evaluation, repeat):
            env.close()


def test_curriculum_advances_only_when_episode_resets(monkeypatch):
    cfg = small_config()
    cfg.episode.max_episode_steps = 1
    manager = EnvironmentManager(
        cfg.game_manager,
        cfg.simulation_manager,
        ModelArgs(),
        None,
        episode_config=cfg.episode,
        seed=42,
    )
    env = manager.make_env()
    try:
        env.reset(seed=42)
        monkeypatch.setattr(env.simulation_manager, "should_advance_curriculum", lambda: True)
        calls = []
        original = env.simulation_manager.advance_curriculum

        def advance():
            calls.append(True)
            return original()

        monkeypatch.setattr(env.simulation_manager, "advance_curriculum", advance)
        env.step(0)
        assert not calls
        env.reset()
        assert len(calls) == 1
    finally:
        env.close()


def test_real_dqn_crosses_weight_logging_boundary(tmp_path, monkeypatch):
    from tsp_rl_kg.rl.training.callbacks import CurriculumCallback

    cfg = small_config("DQN")
    cfg.algorithm.hyperparameters["learning_starts"] = 1001
    cfg.evaluation.eval_freq = 2000
    cfg.ablation.disable_curriculum = True
    trainer = Trainer(0.5, results_dir=str(tmp_path))
    trainer.setup(cfg, seed=42)
    calls = []
    original = CurriculumCallback.print_weight_statistics

    def record(callback):
        calls.append(callback.n_calls)
        original(callback)

    monkeypatch.setattr(CurriculumCallback, "print_weight_statistics", record)
    try:
        trainer.model_trainer.backend.train(1001, output_dir=str(tmp_path / "dqn"))
        assert calls == [1000]
        assert trainer.model_trainer.backend.model.num_timesteps == 1001
    finally:
        trainer.close()


def test_failed_seed_marks_parent_mlflow_run_failed(tmp_path, monkeypatch):
    cfg = small_config()
    cfg.seeds = [41, 42]
    study = AblationStudy(cfg, kg_completeness_values=[0.5], mlflow_experiment_name="failure-test")
    original = Trainer.setup

    def fail_one(self, config, seed=None):
        if seed == 41:
            raise RuntimeError("deliberate failure")
        return original(self, config, seed)

    monkeypatch.setattr(Trainer, "setup", fail_one)
    with pytest.raises(StudyFailedError):
        study.run()
    result = study.results["kg_completeness_0.5"]
    assert (result["attempted"], result["succeeded"], result["failed"]) == (2, 1, 1)
    experiment = mlflow.get_experiment_by_name("failure-test")
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(parents) == 1 and parents[0].info.status == "FAILED"
    assert sorted(run.info.status for run in runs) == ["FAILED", "FAILED", "FINISHED"]


def test_cli_failed_study_returns_nonzero_and_keeps_results(tmp_path):
    from tsp_rl_kg.rl.training.run import main

    config = {
        "study": {
            "base_config": small_config().to_dict(),
            "experiments": [{"name": "broken", "algorithm": {"algorithm": "invalid"}}],
        }
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(config))
    assert main(["--config", str(path)]) == 1
    (summary_path,) = tmp_path.glob("results/*/study_summary.json")
    assert json.loads(summary_path.read_text())["failed"] == 1
