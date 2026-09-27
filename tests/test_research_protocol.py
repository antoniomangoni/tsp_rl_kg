import json

import numpy as np
import pytest

from tsp_rl_kg.config import EvaluationConfig, GameManagerConfig, ModelArgs, SimulationManagerConfig
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.rl.training.research import require_gate
from tsp_rl_kg.rl.training.statistics import iqm, paired_bootstrap, pilot_decision


def test_iqm_fractional_boundaries():
    assert iqm([0, 1, 2, 100]) == 1.5
    assert iqm([0, 2, 100]) == pytest.approx(18)
    assert iqm([7]) == 7


def test_bootstrap_preserves_pairs_and_rejects_missing_cells():
    values = np.arange(15).reshape(3, 5)
    results = paired_bootstrap(np.stack([values, values + 2]), repetitions=100)
    assert results["paired_difference_intervals"][1] == pytest.approx([2, 2])
    assert results == paired_bootstrap(np.stack([values, values + 2]), repetitions=100)
    with pytest.raises(ValueError):
        paired_bootstrap([[[np.nan]]])


def test_pilot_gates_and_budget_escalation():
    stalled = dict(success_rate=0.1, efficiency=0.1, median_success_energy=100)
    improving = dict(stalled, success_rate=0.15)
    good = dict(success_rate=0.95, efficiency=0.95, median_success_energy=11)
    assert pilot_decision([good], 1, 10, 1_000_000)["status"] == "passed"
    assert pilot_decision([stalled] * 3, 1, 10, 1_000_000)["status"] == "diagnose"
    assert pilot_decision([stalled, stalled, improving], 1, 10, 1_000_000)["budget"] == 3_000_000
    assert pilot_decision([stalled, stalled, improving], 1, 10, 10_000_000)["status"] == "diagnose"


def test_study_requires_passed_current_gate(tmp_path):
    with pytest.raises(RuntimeError, match="Missing"):
        require_gate(tmp_path, "smoke")
    (tmp_path / "smoke_gate.json").write_text(json.dumps(dict(passed=True, source_hash="old")))
    with pytest.raises(RuntimeError, match="stale"):
        require_gate(tmp_path, "smoke")


def test_terrain_disjoint_reproducible_stratified_pools():
    manager = EnvironmentManager(
        GameManagerConfig(num_tiles=6, number_of_outposts=2, headless=True),
        SimulationManagerConfig(number_of_environments=6),
        ModelArgs(),
        None,
        seed=41,
        evaluation_config=EvaluationConfig(n_eval_episodes=5, final_eval_episodes=5),
    )
    train = manager.make_env()
    validation = manager.make_eval_env()
    test = manager.make_test_env()
    try:
        pools = [set(t.terrain_fingerprint for t in pool) for pool in manager.pools.values()]
        assert all(a.isdisjoint(b) for i, a in enumerate(pools) for b in pools[i + 1 :])
        assert len(validation.simulation_manager.game_managers) == 5
        sequence = []
        for _ in range(2):
            validation.begin_evaluation()
            ids = []
            for _ in range(5):
                validation.reset()
                ids.append(validation.current_gm.world_template.fingerprint)
            sequence.append(ids)
        assert sequence[0] == sequence[1]
        np.random.seed(12)
        expected = np.random.random(3)
        np.random.seed(12)
        manager.make_env().close()
        np.testing.assert_array_equal(expected, np.random.random(3))
    finally:
        for env in (train, validation, test):
            env.close()


def test_conditional_efficiency_reports_undefined_instead_of_inventing_success():
    from tsp_rl_kg.rl.training.statistics import successful_tour_bootstrap

    result = successful_tour_bootstrap(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), repetitions=20)
    assert result["iqm"] == [None]
    assert result["confidence_intervals"] == [None]
    assert result["undefined_resamples"] == [20]


def test_full_config_matches_locked_preset_and_cannot_bypass_gates(tmp_path, monkeypatch):
    from pathlib import Path

    from tsp_rl_kg.config import TrainingConfig
    from tsp_rl_kg.rl.training.research import config_for
    from tsp_rl_kg.rl.training.run import main
    from tsp_rl_kg.utils.config_files import load_config_file

    path = Path("configs/ablation_full.toml").resolve()
    data = load_config_file(path)["study"]
    expected = config_for("study")
    expected.total_timesteps = 1_000_000
    assert TrainingConfig.from_dict(data["base_config"]).to_dict() == expected.to_dict()
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(path)]) == 1
    assert not (tmp_path / "results").exists()


def test_study_summary_checks_pairing_and_reports_conditional_metrics(monkeypatch):
    from tsp_rl_kg.rl.training import research

    monkeypatch.setattr(research, "SEEDS", [1, 2])
    result = dict(
        deterministic=dict(
            episodes=[
                dict(world_id="a", success=True, efficiency=0.8),
                dict(world_id="b", success=False, efficiency=0),
            ]
        ),
        stochastic=dict(
            episodes=[
                dict(world_id="a", success=True, efficiency=0.6),
                dict(world_id="b", success=False, efficiency=0),
            ]
        ),
    )
    data = {"vision": [result, result], "gnn": [result, result]}
    summary = research.summarize_study(data)
    assert "deterministic.successful_tour_efficiency" in summary
    assert summary["deterministic.successful_tour_efficiency"]["iqm"] == [0.8, 0.8]
    import copy

    bad = {key: copy.deepcopy(value) for key, value in data.items()}
    bad["gnn"][0]["deterministic"]["episodes"][0]["world_id"] = "different"
    with pytest.raises(ValueError, match="paired"):
        research.summarize_study(bad)
