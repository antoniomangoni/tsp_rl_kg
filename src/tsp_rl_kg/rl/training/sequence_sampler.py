from __future__ import annotations

import numpy as np

from tsp_rl_kg.rl.training.backends.base import ModelUpdateScheduler, SequenceBatch, SequenceSampler
from tsp_rl_kg.rl.training.trajectory_store import InMemoryTrajectoryStore


class RandomSequenceSampler(SequenceSampler):
    """Sample fixed-length transition sequences from completed episodes."""

    def __init__(self, trajectory_store: InMemoryTrajectoryStore, rng_seed: int | None = None):
        self.trajectory_store = trajectory_store
        self._rng = np.random.default_rng(rng_seed)

    def sample(self, batch_size: int, sequence_length: int) -> SequenceBatch:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        eligible_episode_ids = [
            episode_id
            for episode_id in self.trajectory_store.get_completed_episode_ids()
            if len(self.trajectory_store.get_episode(episode_id)) >= sequence_length
        ]
        if not eligible_episode_ids:
            raise ValueError("No completed episodes contain enough transitions to sample")

        sequences: list[tuple] = []
        episode_ids: list[int] = []
        start_step_ids: list[int] = []

        for _ in range(batch_size):
            episode_id = int(self._rng.choice(eligible_episode_ids))
            transitions = self.trajectory_store.get_episode(episode_id)
            max_start = len(transitions) - sequence_length
            start_index = int(self._rng.integers(0, max_start + 1))
            sequence = tuple(transitions[start_index : start_index + sequence_length])
            sequences.append(sequence)
            episode_ids.append(episode_id)
            start_step_ids.append(int(sequence[0]["step_id"]))

        return SequenceBatch(
            sequences=tuple(sequences),
            episode_ids=tuple(episode_ids),
            start_step_ids=tuple(start_step_ids),
            sequence_length=sequence_length,
        )


class PeriodicModelUpdateScheduler(ModelUpdateScheduler):
    """Simple scheduler for replay-backed or world-model update triggers."""

    def __init__(
        self,
        *,
        start_after_steps: int = 0,
        update_every_steps: int = 1,
        min_completed_episodes: int = 0,
    ):
        if start_after_steps < 0:
            raise ValueError(f"start_after_steps must be >= 0, got {start_after_steps}")
        if update_every_steps < 1:
            raise ValueError(f"update_every_steps must be >= 1, got {update_every_steps}")
        if min_completed_episodes < 0:
            raise ValueError(f"min_completed_episodes must be >= 0, got {min_completed_episodes}")

        self.start_after_steps = start_after_steps
        self.update_every_steps = update_every_steps
        self.min_completed_episodes = min_completed_episodes
        self._last_update_step: int | None = None

    def should_update(self, total_steps: int, completed_episodes: int) -> bool:
        if total_steps < self.start_after_steps:
            return False
        if completed_episodes < self.min_completed_episodes:
            return False
        if self._last_update_step is None:
            return True
        return (total_steps - self._last_update_step) >= self.update_every_steps

    def record_update(self, total_steps: int, completed_episodes: int) -> None:
        if total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {total_steps}")
        if completed_episodes < 0:
            raise ValueError(f"completed_episodes must be >= 0, got {completed_episodes}")
        self._last_update_step = total_steps
