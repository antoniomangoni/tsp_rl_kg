"""Regression tests for reliable study reporting and environment lifecycle."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tsp_rl_kg.config import AlgorithmConfig, TrainingConfig
from tsp_rl_kg.rl.training.ablation_study import AblationStudy, StudyFailedError
from tsp_rl_kg.rl.training.trainer import Trainer


def test_switching_algorithm_replaces_hyperparameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    study = AblationStudy(TrainingConfig(), kg_completeness_values=[0.5])
    for experiment in [
        {"algorithm": {"algorithm": "DQN"}},
        {"config_overrides": {"algorithm": {"algorithm": "DQN"}}},
    ]:
        config, _, _ = study._build_experiment_config(experiment)
        assert "n_steps" not in config.algorithm.hyperparameters
        assert config.algorithm.hyperparameters["buffer_size"] == 100_000
    assert "n_steps" in study.base_config.algorithm.hyperparameters


def test_ppo_keeps_nonlegacy_parameters():
    cfg = TrainingConfig(
        algorithm=AlgorithmConfig(hyperparameters={"ent_coef": 0.01, "n_epochs": 2})
    )
    assert cfg.algorithm.hyperparameters["ent_coef"] == 0.01
    assert TrainingConfig.from_dict(cfg.to_dict()).algorithm.hyperparameters["n_epochs"] == 2


@pytest.mark.parametrize("failed_seeds", [{2}, {1, 2}])
def test_study_persists_failures_and_raises(tmp_path, monkeypatch, failed_seeds):
    monkeypatch.chdir(tmp_path)
    closed = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.seed = None

        def setup(self, config, seed):
            self.seed = seed
            if seed in failed_seeds:
                raise RuntimeError(f"broken seed {seed}")

        def run(self, name):
            return {"mean_reward": 1.0, "std_reward": 0.0}

        def close(self):
            closed.append(self.seed)

    study = AblationStudy(TrainingConfig(seeds=[1, 2]), kg_completeness_values=[0.5])
    fake_mlflow = MagicMock()
    fake_mlflow.active_run.return_value = None
    with (
        patch("tsp_rl_kg.rl.training.ablation_study.Trainer", FakeTrainer),
        patch("tsp_rl_kg.rl.training.ablation_study.mlflow", fake_mlflow),
        pytest.raises(StudyFailedError),
    ):
        study.run()
    assert closed == [1, 2]
    result = study.results["kg_completeness_0.5"]
    assert result["attempted"] == 2
    assert result["failed"] == len(failed_seeds)
    assert result["succeeded"] == 2 - len(failed_seeds)
    assert result["incomplete"]
    saved = json.loads((tmp_path / study.results_dir / "ablation_study_results.json").read_text())
    assert saved == study.results
    for attempt in result["attempts"]:
        if attempt["status"] == "failed":
            assert (tmp_path / attempt["error_path"]).is_file()


def test_empty_study_rejected_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        AblationStudy(TrainingConfig(), experiments=[])
    with pytest.raises(ValueError, match="at least one"):
        AblationStudy(TrainingConfig(seeds=[]), kg_completeness_values=[0.5])
    assert not (tmp_path / "results").exists()


def test_setup_failure_closes_created_environments(tmp_path, monkeypatch):
    trainer = Trainer(0.5, results_dir=str(tmp_path))
    env, eval_env = MagicMock(), MagicMock()

    def fail(config, seed):
        trainer.env = env
        trainer.eval_env = eval_env
        raise RuntimeError("setup failed")

    monkeypatch.setattr(trainer, "_setup", fail)
    with pytest.raises(RuntimeError, match="setup failed"):
        trainer.setup(TrainingConfig())
    trainer.close()
    env.close.assert_called_once()
    eval_env.close.assert_called_once()


def test_run_failure_closes_environments(tmp_path, monkeypatch):
    trainer = Trainer(0.5, results_dir=str(tmp_path))
    trainer.env = MagicMock()
    monkeypatch.setattr(trainer, "_run", MagicMock(side_effect=RuntimeError("training failed")))
    with pytest.raises(RuntimeError):
        trainer.run("run")
    trainer.env.close.assert_called_once()


def test_dqn_weight_logging_uses_online_network():
    from tsp_rl_kg.rl.training.callbacks import CurriculumCallback

    callback = CurriculumCallback(SimpleNamespace(unwrapped=SimpleNamespace()), MagicMock(), 3)
    extractor = MagicMock()
    callback.model = SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=None, q_net=SimpleNamespace(features_extractor=extractor)
        )
    )
    callback.print_weight_statistics()
    extractor.vision_processor.named_modules.assert_called_once()


def test_output_directories_are_unique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = AblationStudy(TrainingConfig(), kg_completeness_values=[0.5])
    b = AblationStudy(TrainingConfig(), kg_completeness_values=[0.5])
    assert a.results_dir != b.results_dir


def test_external_config_switch_does_not_inherit_ppo_defaults():
    from tsp_rl_kg.main import _build_training_config
    from tsp_rl_kg.utils.config_files import merge_training_config

    override = {"algorithm": {"algorithm": "DQN", "hyperparameters": {"batch_size": 4}}}
    merged = merge_training_config(TrainingConfig().to_dict(), override)
    assert "n_steps" not in merged["algorithm"]["hyperparameters"]
    assert "n_steps" not in _build_training_config(override).algorithm.hyperparameters
