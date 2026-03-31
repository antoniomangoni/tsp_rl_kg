"""Tests for the backend-neutral curriculum service."""

from __future__ import annotations

from dataclasses import dataclass

from tsp_rl_kg.rl.training.curriculum import CurriculumService
from tsp_rl_kg.rl.training.metrics import TrainingMetrics


@dataclass
class DummySimulationManager:
    should_advance: bool = False
    new_index: int = 1
    current_curriculum_index: int = 0

    def should_advance_curriculum(self):
        return self.should_advance

    def advance_curriculum(self):
        if self.new_index >= 0:
            self.current_curriculum_index = self.new_index
        return self.new_index


class DummyEnv:
    def __init__(self, *, early_stop=False, should_advance=False, new_index=1):
        self.early_stop = early_stop
        self.simulation_manager = DummySimulationManager(
            should_advance=should_advance,
            new_index=new_index,
        )

    def get_metrics(self):
        return {
            "performance": 10.0,
            "game_manager_index": 2,
            "best_route_energy": 5.0,
            "curriculum_level": self.simulation_manager.current_curriculum_index,
            "target_route_energy": 8.0,
            "best_efficiency": 0.75,
            "improvement": 2.5,
            "gap": 1.25,
        }


def test_curriculum_service_records_metrics_and_requests_reset():
    metrics_sink = TrainingMetrics(num_actions=3)
    service = CurriculumService(metrics_sink)
    env = DummyEnv(should_advance=True, new_index=2)

    decision = service.on_step(10, env, [1, 2, 3])

    assert decision.continue_training is True
    assert decision.should_reset_environments is True
    assert decision.should_stop is False
    assert metrics_sink.steps == [10]
    assert metrics_sink.performances == [10.0]
    assert metrics_sink.action_counts[0] == [1]
    assert metrics_sink.action_counts[1] == [2]
    assert metrics_sink.action_counts[2] == [3]


def test_curriculum_service_stops_on_early_stop():
    service = CurriculumService(TrainingMetrics(num_actions=2))
    env = DummyEnv(early_stop=True)

    decision = service.on_step(4, env, [0, 1])

    assert decision.continue_training is False
    assert decision.should_stop is True
    assert decision.should_reset_environments is False


def test_curriculum_service_stops_when_all_curricula_complete():
    service = CurriculumService(TrainingMetrics(num_actions=2))
    env = DummyEnv(should_advance=True, new_index=-1)

    decision = service.on_step(12, env, [0, 0])

    assert decision.continue_training is False
    assert decision.should_stop is True
