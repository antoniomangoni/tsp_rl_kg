from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class RLBackend(str, Enum):
    SB3 = "sb3"
    CUSTOM = "custom"

    @classmethod
    def from_value(cls, value: RLBackend | str) -> RLBackend:
        if isinstance(value, cls):
            return value
        return cls(value.lower())


class AlgorithmName(str, Enum):
    PPO = "PPO"
    DQN = "DQN"
    A2C = "A2C"
    SAC = "SAC"

    @classmethod
    def from_value(cls, value: AlgorithmName | str) -> AlgorithmName:
        if isinstance(value, cls):
            return value
        return cls(value.upper())


class RewardComponent(str, Enum):
    PROXIMITY = "proximity"
    CIRCULAR_PENALTY = "circular_penalty"
    ROUTE_IMPROVEMENT = "route_improvement"


@dataclass
class GameManagerConfig:
    num_tiles: int = 32
    screen_size: int = 800
    vision_range: int = 2
    headless: bool = False
    human_mode: bool = False
    use_random_human_actions: bool = False
    target_fps: int = 30
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.screen_size < self.num_tiles:
            raise ValueError(
                f"screen_size ({self.screen_size}) must be >= num_tiles ({self.num_tiles})"
            )
        if self.vision_range < 0:
            raise ValueError(f"vision_range must be >= 0, got {self.vision_range}")
        if self.target_fps < 1:
            raise ValueError(f"target_fps must be >= 1, got {self.target_fps}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 when provided, got {self.max_steps}")


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
    """Legacy PPO hyperparameters kept for backward-compatible config parsing."""

    n_steps: int = 4096
    batch_size: int = 512
    learning_rate: float = 6e-4
    gamma: float = 0.995

    def to_dict(self) -> dict:
        """Convert to dict for use as default PPO backend hyperparameters."""
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
        }


def default_algorithm_hyperparameters(
    algorithm: AlgorithmName | str = AlgorithmName.PPO,
) -> dict[str, int | float | bool | str]:
    """Return baseline hyperparameters for the selected algorithm."""

    algorithm = AlgorithmName.from_value(algorithm)

    if algorithm == AlgorithmName.PPO:
        return ModelConfig().to_dict()
    if algorithm == AlgorithmName.DQN:
        return {
            "learning_rate": 1e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "train_freq": 4,
            "gamma": 0.99,
        }
    if algorithm == AlgorithmName.A2C:
        return {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
        }
    if algorithm == AlgorithmName.SAC:
        return {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 100,
            "batch_size": 256,
            "train_freq": 1,
            "gamma": 0.99,
        }

    raise NotImplementedError(f"No default hyperparameters defined for {algorithm.value}")


@dataclass
class AlgorithmConfig:
    """Backend and algorithm selection for the training stack."""

    backend: RLBackend = RLBackend.SB3
    algorithm: AlgorithmName = AlgorithmName.PPO
    policy_name: str = "MultiInputPolicy"
    verbose: int = 1
    tensorboard_run_name: str | None = None
    hyperparameters: dict[str, int | float | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = RLBackend.from_value(self.backend)
        self.algorithm = AlgorithmName.from_value(self.algorithm)
        defaults = default_algorithm_hyperparameters(self.algorithm)
        raw_hyperparameters = self.hyperparameters or {}
        self.hyperparameters = {**defaults, **dict(raw_hyperparameters)}
        if not self.policy_name:
            raise ValueError("policy_name must be a non-empty string")
        if self.verbose < 0:
            raise ValueError(f"verbose must be >= 0, got {self.verbose}")

    @classmethod
    def from_legacy_model_config(
        cls,
        model_config: ModelConfig | dict | None = None,
        *,
        policy_name: str = "MultiInputPolicy",
    ) -> AlgorithmConfig:
        if model_config is None:
            model_config = ModelConfig()
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        return cls(
            backend=RLBackend.SB3,
            algorithm=AlgorithmName.PPO,
            policy_name=policy_name,
            hyperparameters=model_config.to_dict(),
        )


@dataclass
class EvaluationConfig:
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    deterministic: bool = True
    render: bool = False

    def __post_init__(self) -> None:
        if self.eval_freq < 1:
            raise ValueError(f"eval_freq must be >= 1, got {self.eval_freq}")
        if self.n_eval_episodes < 1:
            raise ValueError(f"n_eval_episodes must be >= 1, got {self.n_eval_episodes}")


@dataclass
class ReplayConfig:
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 1

    def __post_init__(self) -> None:
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")
        if self.learning_starts < 0:
            raise ValueError(f"learning_starts must be >= 0, got {self.learning_starts}")
        if self.train_freq < 1:
            raise ValueError(f"train_freq must be >= 1, got {self.train_freq}")


@dataclass
class SequenceConfig:
    sequence_length: int = 16
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class WorldModelConfig:
    enabled: bool = False
    latent_dim: int = 128
    imagination_horizon: int = 15

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.imagination_horizon < 1:
            raise ValueError(f"imagination_horizon must be >= 1, got {self.imagination_horizon}")


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

    # Ablation flags
    disable_vision: bool = False
    disable_graph: bool = False

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
class FeatureEncodingConfig:
    strategy: str = "one_hot"
    schema_path: str | None = None
    embedding_path: str | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy.lower()
        allowed_strategies = {"raw_int", "one_hot", "embedding_lookup"}
        if self.strategy not in allowed_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(allowed_strategies)}, got {self.strategy!r}"
            )

        if self.strategy == "embedding_lookup":
            if not self.schema_path:
                raise ValueError("schema_path is required when strategy='embedding_lookup'")
            if not self.embedding_path:
                raise ValueError("embedding_path is required when strategy='embedding_lookup'")


@dataclass
class AblationConfig:
    disable_vision: bool = False
    disable_graph: bool = False
    disable_reward_components: list[RewardComponent] = field(default_factory=list)
    disable_curriculum: bool = False

    def __post_init__(self) -> None:
        if self.disable_vision and self.disable_graph:
            raise ValueError("Cannot disable both vision and graph processors")
        # Normalize strings to enum values
        self.disable_reward_components = [
            RewardComponent(c) if isinstance(c, str) else c for c in self.disable_reward_components
        ]


@dataclass
class TrainingConfig:
    """Top-level config container aggregating all sub-configs."""

    game_manager: GameManagerConfig = field(default_factory=GameManagerConfig)
    simulation_manager: SimulationManagerConfig = field(default_factory=SimulationManagerConfig)
    model_args: ModelArgs = field(default_factory=ModelArgs)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    agent_model: AgentModelConfig = field(default_factory=AgentModelConfig)
    feature_encoding: FeatureEncodingConfig = field(default_factory=FeatureEncodingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    total_timesteps: int = 100_000
    kg_completeness: float = 0.5
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    def __post_init__(self) -> None:
        if isinstance(self.game_manager, dict):
            self.game_manager = GameManagerConfig(**self.game_manager)
        if isinstance(self.simulation_manager, dict):
            self.simulation_manager = SimulationManagerConfig(**self.simulation_manager)
        if isinstance(self.model_args, dict):
            self.model_args = ModelArgs(**self.model_args)
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        if isinstance(self.algorithm, dict):
            self.algorithm = AlgorithmConfig(**self.algorithm)
        if isinstance(self.evaluation, dict):
            self.evaluation = EvaluationConfig(**self.evaluation)
        if isinstance(self.replay, dict):
            self.replay = ReplayConfig(**self.replay)
        if isinstance(self.sequence, dict):
            self.sequence = SequenceConfig(**self.sequence)
        if isinstance(self.world_model, dict):
            self.world_model = WorldModelConfig(**self.world_model)
        if isinstance(self.curriculum, dict):
            self.curriculum = CurriculumConfig(**self.curriculum)
        if isinstance(self.episode, dict):
            self.episode = EpisodeConfig(**self.episode)
        if isinstance(self.agent_model, dict):
            self.agent_model = AgentModelConfig(**self.agent_model)
        if isinstance(self.feature_encoding, dict):
            self.feature_encoding = FeatureEncodingConfig(**self.feature_encoding)
        if isinstance(self.ablation, dict):
            self.ablation = AblationConfig(**self.ablation)

        self._synchronise_algorithm_config()

    def _synchronise_algorithm_config(self) -> None:
        default_model_config = ModelConfig().to_dict()
        default_algorithm = AlgorithmConfig()
        current_model_config = self.model_config.to_dict()

        if self.algorithm == default_algorithm and current_model_config != default_model_config:
            self.algorithm = AlgorithmConfig.from_legacy_model_config(self.model_config)
            return

        if (
            self.algorithm.backend == RLBackend.SB3
            and self.algorithm.algorithm == AlgorithmName.PPO
        ):
            merged_hyperparameters = {**default_model_config, **self.algorithm.hyperparameters}
            self.model_config = ModelConfig(**merged_hyperparameters)
            self.algorithm.hyperparameters = self.model_config.to_dict()

    @staticmethod
    def from_dict(d: dict) -> TrainingConfig:
        """Construct from the legacy nested-dict format for backwards compatibility."""
        game_manager_data = d.get("game_manager", d.get("game_manager_args", {}))
        simulation_manager_data = d.get(
            "simulation_manager",
            d.get("simulation_manager_args", {}),
        )
        model_args_data = d.get("model_args", {})
        model_config_data = d.get("model_config", {})
        algorithm_data = d.get("algorithm", d.get("algorithm_config"))
        evaluation_data = d.get("evaluation", d.get("evaluation_config", {}))
        replay_data = d.get("replay", d.get("replay_config", {}))
        sequence_data = d.get("sequence", d.get("sequence_config", {}))
        world_model_data = d.get("world_model", d.get("world_model_config", {}))
        curriculum_data = d.get("curriculum", d.get("curriculum_config", {}))
        episode_data = d.get("episode", d.get("episode_config", {}))
        agent_model_data = d.get("agent_model", d.get("agent_model_config", {}))
        feature_encoding_data = d.get("feature_encoding", d.get("feature_encoding_config", {}))
        ablation_data = d.get("ablation", d.get("ablation_config", {}))

        model_config = ModelConfig(**model_config_data)
        algorithm = (
            AlgorithmConfig(**algorithm_data)
            if algorithm_data is not None
            else AlgorithmConfig.from_legacy_model_config(model_config)
        )

        return TrainingConfig(
            game_manager=GameManagerConfig(**game_manager_data),
            simulation_manager=SimulationManagerConfig(**simulation_manager_data),
            model_args=ModelArgs(**model_args_data),
            model_config=model_config,
            algorithm=algorithm,
            evaluation=EvaluationConfig(**evaluation_data),
            replay=ReplayConfig(**replay_data),
            sequence=SequenceConfig(**sequence_data),
            world_model=WorldModelConfig(**world_model_data),
            curriculum=CurriculumConfig(**curriculum_data),
            episode=EpisodeConfig(**episode_data),
            agent_model=AgentModelConfig(**agent_model_data),
            feature_encoding=FeatureEncodingConfig(**feature_encoding_data),
            ablation=AblationConfig(**ablation_data),
            total_timesteps=d.get("total_timesteps", 100_000),
            kg_completeness=d.get("kg_completeness", 0.5),
            seeds=d.get("seeds", [42, 123, 456]),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)
