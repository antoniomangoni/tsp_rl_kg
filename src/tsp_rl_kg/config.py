from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GameManagerConfig:
    num_tiles: int = 32
    screen_size: int = 800
    vision_range: int = 2
    headless: bool = False

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.screen_size < self.num_tiles:
            raise ValueError(
                f"screen_size ({self.screen_size}) must be >= num_tiles ({self.num_tiles})"
            )
        if self.vision_range < 0:
            raise ValueError(f"vision_range must be >= 0, got {self.vision_range}")


@dataclass
class SimulationManagerConfig:
    number_of_environments: int = 500
    number_of_curricula: int = 10
    min_episodes_per_curriculum: int = 1

    def __post_init__(self) -> None:
        if self.number_of_environments < 1:
            raise ValueError(
                f"number_of_environments must be >= 1, got {self.number_of_environments}"
            )
        if self.number_of_curricula < 1:
            raise ValueError(f"number_of_curricula must be >= 1, got {self.number_of_curricula}")
        if self.min_episodes_per_curriculum < 1:
            raise ValueError(
                f"min_episodes_per_curriculum must be >= 1, got {self.min_episodes_per_curriculum}"
            )


@dataclass
class ModelArgs:
    num_actions: int = 11

    def __post_init__(self) -> None:
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {self.num_actions}")


@dataclass
class ModelConfig:
    """PPO hyperparameters passed directly to stable-baselines3."""

    n_steps: int = 4096
    batch_size: int = 512
    learning_rate: float = 6e-4
    gamma: float = 0.995

    def to_dict(self) -> dict:
        """Convert to dict for passing as **kwargs to PPO."""
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
        }


@dataclass
class CurriculumConfig:
    min_episodes_per_curriculum: int = 4
    performance_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.performance_threshold < 0 or self.performance_threshold > 1:
            raise ValueError(
                f"performance_threshold must be in [0, 1], got {self.performance_threshold}"
            )


@dataclass
class AgentConfig:
    resource_max: int = 5
    action_energy_cost: int = 3
    scout_vision_multiplier: int = 2

    def __post_init__(self) -> None:
        if self.resource_max < 1:
            raise ValueError(f"resource_max must be >= 1, got {self.resource_max}")
        if self.action_energy_cost < 0:
            raise ValueError(f"action_energy_cost must be >= 0, got {self.action_energy_cost}")
        if self.scout_vision_multiplier < 1:
            raise ValueError(
                f"scout_vision_multiplier must be >= 1, got {self.scout_vision_multiplier}"
            )


@dataclass
class RewardConfig:
    """All reward weights, penalties, and thresholds for the reward system."""

    # Base rewards
    new_outpost_reward: float = 30.0
    completion_reward: float = 100.0
    route_improvement_reward: float = 200.0
    better_route_than_algo_reward: float = 200.0

    # Penalties
    penalty_per_step: float = -0.5
    farther_from_outpost_penalty: float = -1.0
    circular_behavior_penalty: float = -2.0

    # Positive reinforcements
    closer_to_outpost_reward: float = 0.55

    # Scaling factors
    time_penalty_factor: float = -0.01
    outpost_reward_increase_factor: float = 0.5
    completion_time_bonus_factor: float = 1.0

    # Route improvement tracking
    max_not_improvement_routes: int = 5

    # Normalisation
    normalisation_scale: float = 100.0


@dataclass
class EpisodeConfig:
    max_episode_steps: int = 2048 * 8
    max_steps_without_progress: int = 2048 * 4
    max_game_worlds_trained_in: int = 100

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1, got {self.max_episode_steps}")
        if self.max_steps_without_progress < 1:
            raise ValueError(
                f"max_steps_without_progress must be >= 1, got {self.max_steps_without_progress}"
            )


@dataclass
class AgentModelConfig:
    # VisionProcessor defaults
    vision_num_conv_layers: int = 4
    vision_conv_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    vision_fc_dims: list[int] = field(default_factory=lambda: [512])

    # GraphProcessor defaults
    graph_num_gat_layers: int = 3
    graph_gat_heads: list[int] = field(default_factory=lambda: [4, 4, 4])
    graph_fc_dims: list[int] = field(default_factory=lambda: [256])
    gat_hidden_dim: int = 48

    # Shared
    features_dim: int = 192
    dropout: float = 0.25

    def to_vision_params(self) -> dict:
        return {
            "num_conv_layers": self.vision_num_conv_layers,
            "conv_channels": self.vision_conv_channels,
            "fc_dims": self.vision_fc_dims,
        }

    def to_graph_params(self) -> dict:
        return {
            "num_gat_layers": self.graph_num_gat_layers,
            "gat_heads": self.graph_gat_heads,
            "fc_dims": self.graph_fc_dims,
        }


@dataclass
class TrainingConfig:
    """Top-level config container aggregating all sub-configs."""

    game_manager: GameManagerConfig = field(default_factory=GameManagerConfig)
    simulation_manager: SimulationManagerConfig = field(default_factory=SimulationManagerConfig)
    model_args: ModelArgs = field(default_factory=ModelArgs)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    agent_model: AgentModelConfig = field(default_factory=AgentModelConfig)
    total_timesteps: int = 100_000
    kg_completeness: float = 0.5

    @staticmethod
    def from_dict(d: dict) -> TrainingConfig:
        """Construct from the legacy nested-dict format for backwards compatibility."""
        return TrainingConfig(
            game_manager=GameManagerConfig(**d.get("game_manager_args", {})),
            simulation_manager=SimulationManagerConfig(**d.get("simulation_manager_args", {})),
            model_args=ModelArgs(**d.get("model_args", {})),
            model_config=ModelConfig(**d.get("model_config", {})),
            curriculum=CurriculumConfig(**d.get("curriculum_config", {})),
            total_timesteps=d.get("total_timesteps", 100_000),
            kg_completeness=d.get("kg_completeness", 0.5),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)
