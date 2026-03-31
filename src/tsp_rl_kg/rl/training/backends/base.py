from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypedDict, runtime_checkable

ObservationDict = dict[str, Any]
MetricValue = str | int | float | bool | list[int] | list[float]
MetricsDict = dict[str, MetricValue]


@dataclass(frozen=True)
class CurriculumDecision:
    continue_training: bool = True
    should_reset_environments: bool = False
    should_stop: bool = False


class Transition(TypedDict):
    obs: ObservationDict
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_obs: ObservationDict
    info: dict[str, Any]
    episode_id: int
    step_id: int


@runtime_checkable
class TrainingBackend(Protocol):
    name: str

    def build(self) -> None: ...

    def train(self, total_timesteps: int, output_dir: str | None = None) -> None: ...

    def predict(self, observation: ObservationDict, deterministic: bool = True): ...

    def save(self, path: str) -> str: ...

    def collect_metrics(self) -> MetricsDict: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        backend: TrainingBackend,
        env: Any,
        n_episodes: int,
    ) -> MetricsDict: ...


@runtime_checkable
class CurriculumController(Protocol):
    def on_step(
        self,
        step: int,
        env: Any,
        action_counts: Sequence[int],
    ) -> CurriculumDecision: ...


@runtime_checkable
class MetricsSink(Protocol):
    def record(self, step: int, metrics: MetricsDict) -> None: ...


@runtime_checkable
class TrajectoryStore(Protocol):
    def append(self, transition: Transition) -> None: ...

    def finish_episode(self, episode_id: int) -> None: ...


@runtime_checkable
class SequenceSampler(Protocol):
    def sample(self, batch_size: int, sequence_length: int): ...
