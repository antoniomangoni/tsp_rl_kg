from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator

from tsp_rl_kg.rl.training.backends.base import (
    TrainingBackend,
    TrajectoryStore,
    Transition,
    TransitionCollectionStats,
)


@dataclass(frozen=True)
class EpisodeTrajectory:
    episode_id: int
    transitions: tuple[Transition, ...]
    completed: bool


class InMemoryTrajectoryStore(TrajectoryStore):
    """Simple in-memory trajectory store for future world-model training paths."""

    def __init__(self):
        self._episodes: dict[int, list[Transition]] = {}
        self._completed_episode_ids: list[int] = []
        self._completed_episode_id_set: set[int] = set()
        self._transition_count = 0

    @property
    def num_transitions(self) -> int:
        return self._transition_count

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def append(self, transition: Transition) -> None:
        episode_id = int(transition["episode_id"])
        if episode_id in self._completed_episode_id_set:
            raise ValueError(f"Cannot append to completed episode {episode_id}")

        stored_transition = copy.deepcopy(transition)
        self._episodes.setdefault(episode_id, []).append(stored_transition)
        self._transition_count += 1

        if stored_transition["terminated"] or stored_transition["truncated"]:
            self.finish_episode(episode_id)

    def finish_episode(self, episode_id: int) -> None:
        if episode_id not in self._episodes:
            raise KeyError(f"Unknown episode_id {episode_id}")
        if episode_id in self._completed_episode_id_set:
            return

        self._completed_episode_id_set.add(episode_id)
        self._completed_episode_ids.append(episode_id)

    def get_episode(self, episode_id: int) -> list[Transition]:
        return copy.deepcopy(self._episodes.get(episode_id, []))

    def get_completed_episode_ids(self) -> list[int]:
        return list(self._completed_episode_ids)

    def iter_completed_episodes(self) -> Iterator[EpisodeTrajectory]:
        for episode_id in self._completed_episode_ids:
            yield EpisodeTrajectory(
                episode_id=episode_id,
                transitions=tuple(self.get_episode(episode_id)),
                completed=True,
            )

    def get_completed_episodes(self) -> list[EpisodeTrajectory]:
        return list(self.iter_completed_episodes())


class OnlineTrajectoryCollector:
    """Collect real-environment transitions through a backend-neutral predict hook."""

    def collect(
        self,
        backend: TrainingBackend,
        env,
        store: TrajectoryStore,
        max_steps: int,
        *,
        deterministic: bool = False,
        start_episode_id: int = 0,
    ) -> TransitionCollectionStats:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        collected_steps = 0
        completed_episodes = 0
        episode_id = start_episode_id
        step_id = 0
        obs, _ = env.reset()

        while collected_steps < max_steps:
            action, _ = backend.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            store.append(
                {
                    "obs": obs,
                    "action": int(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "next_obs": next_obs,
                    "info": dict(info),
                    "episode_id": episode_id,
                    "step_id": step_id,
                }
            )

            collected_steps += 1
            obs = next_obs
            step_id += 1

            if terminated or truncated:
                store.finish_episode(episode_id)
                completed_episodes += 1
                episode_id += 1
                step_id = 0
                if collected_steps < max_steps:
                    obs, _ = env.reset()

        return TransitionCollectionStats(
            collected_steps=collected_steps,
            completed_episodes=completed_episodes,
            last_episode_id=episode_id,
        )
