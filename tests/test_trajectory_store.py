"""Tests for the parallel world-model data path abstractions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from tsp_rl_kg.rl.training.sequence_sampler import (
    PeriodicModelUpdateScheduler,
    RandomSequenceSampler,
)
from tsp_rl_kg.rl.training.trajectory_store import (
    InMemoryTrajectoryStore,
    OnlineTrajectoryCollector,
)


def _make_transition(*, episode_id: int, step_id: int, terminated: bool = False):
    return {
        "obs": {"value": np.array([step_id], dtype=np.float32)},
        "action": step_id,
        "reward": float(step_id),
        "terminated": terminated,
        "truncated": False,
        "next_obs": {"value": np.array([step_id + 1], dtype=np.float32)},
        "info": {"index": step_id},
        "episode_id": episode_id,
        "step_id": step_id,
    }


class DummyBackend:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, observation, deterministic: bool = True):
        self.predict_calls += 1
        return int(self.predict_calls % 2), None


class DummyCollectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= 2
        return np.ones((1,), dtype=np.float32) * self._step_count, 1.0, terminated, False, {}


def test_trajectory_store_appends_and_finishes_episode():
    store = InMemoryTrajectoryStore()
    transition = _make_transition(episode_id=0, step_id=0)

    store.append(transition)
    store.finish_episode(0)
    transition["obs"]["value"][0] = 999.0

    stored_episode = store.get_episode(0)
    assert store.num_transitions == 1
    assert store.num_episodes == 1
    assert store.get_completed_episode_ids() == [0]
    assert stored_episode[0]["obs"]["value"][0] == 0.0


def test_online_transition_collector_collects_steps_and_marks_episodes():
    store = InMemoryTrajectoryStore()
    collector = OnlineTrajectoryCollector()

    stats = collector.collect(
        backend=DummyBackend(),
        env=DummyCollectorEnv(),
        store=store,
        max_steps=5,
        deterministic=True,
        start_episode_id=3,
    )

    assert stats.collected_steps == 5
    assert stats.completed_episodes == 2
    assert stats.last_episode_id == 5
    assert store.num_transitions == 5
    assert store.get_completed_episode_ids() == [3, 4]
    assert len(store.get_episode(5)) == 1


def test_random_sequence_sampler_returns_fixed_length_sequences():
    store = InMemoryTrajectoryStore()
    for step_id in range(4):
        store.append(_make_transition(episode_id=0, step_id=step_id, terminated=step_id == 3))
    for step_id in range(5):
        store.append(_make_transition(episode_id=1, step_id=step_id, terminated=step_id == 4))

    sampler = RandomSequenceSampler(store, rng_seed=7)
    batch = sampler.sample(batch_size=3, sequence_length=2)

    assert len(batch.sequences) == 3
    assert batch.sequence_length == 2
    for sequence in batch.sequences:
        assert len(sequence) == 2
        assert sequence[1]["step_id"] == sequence[0]["step_id"] + 1


def test_periodic_model_update_scheduler_respects_warmup_and_interval():
    scheduler = PeriodicModelUpdateScheduler(
        start_after_steps=4,
        update_every_steps=3,
        min_completed_episodes=1,
    )

    assert scheduler.should_update(total_steps=3, completed_episodes=1) is False
    assert scheduler.should_update(total_steps=4, completed_episodes=0) is False
    assert scheduler.should_update(total_steps=4, completed_episodes=1) is True

    scheduler.record_update(total_steps=4, completed_episodes=1)

    assert scheduler.should_update(total_steps=6, completed_episodes=1) is False
    assert scheduler.should_update(total_steps=7, completed_episodes=1) is True
