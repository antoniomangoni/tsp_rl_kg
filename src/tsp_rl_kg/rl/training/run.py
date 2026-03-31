import traceback

from loguru import logger

from tsp_rl_kg.config import (
    AblationConfig,
    AlgorithmConfig,
    AlgorithmName,
    CurriculumConfig,
    GameManagerConfig,
    ModelArgs,
    RewardComponent,
    SimulationManagerConfig,
    TrainingConfig,
    default_algorithm_hyperparameters,
)
from tsp_rl_kg.rl.training.ablation_study import AblationStudy
from tsp_rl_kg.utils.logger import configure_logging

# Uncomment for windows
# os.environ['PYGAME_DETECT_AVX2'] = '1'
MIN_EPISODES_PER_CURRICULUM = 4
DEFAULT_KG_COMPLETENESS_VALUES = [0.25, 0.5, 0.75, 1.0]


def build_base_config(
    *,
    algorithm: AlgorithmName | str = AlgorithmName.PPO,
    algorithm_hyperparameters: dict[str, int | float | bool | str] | None = None,
    total_timesteps: int = 100_000,
    seeds: list[int] | None = None,
    number_of_environments: int = 3_000,
    number_of_curricula: int = 30,
) -> TrainingConfig:
    algorithm = AlgorithmName.from_value(algorithm)
    if algorithm_hyperparameters is None:
        algorithm_hyperparameters = default_algorithm_hyperparameters(algorithm)

    return TrainingConfig(
        game_manager=GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        simulation_manager=SimulationManagerConfig(
            number_of_environments=number_of_environments,
            number_of_curricula=number_of_curricula,
            min_episodes_per_curriculum=MIN_EPISODES_PER_CURRICULUM,
        ),
        model_args=ModelArgs(num_actions=11),
        algorithm=AlgorithmConfig(
            algorithm=algorithm,
            hyperparameters=algorithm_hyperparameters,
        ),
        curriculum=CurriculumConfig(
            min_episodes_per_curriculum=MIN_EPISODES_PER_CURRICULUM,
            performance_threshold=0.85,
        ),
        total_timesteps=total_timesteps,
        seeds=seeds or [42, 123, 456],
    )


def build_default_experiments(
    kg_completeness_values: list[float] | None = None,
) -> list[dict]:
    kg_completeness_values = kg_completeness_values or DEFAULT_KG_COMPLETENESS_VALUES

    return [
        *[
            {"name": f"kg_{kg}", "kg_completeness": kg, "ablation": AblationConfig()}
            for kg in kg_completeness_values
        ],
        {
            "name": "dqn_baseline",
            "kg_completeness": 0.5,
            "algorithm": {
                "algorithm": AlgorithmName.DQN.value,
                "hyperparameters": {
                    **default_algorithm_hyperparameters(AlgorithmName.DQN),
                    "gamma": 0.995,
                },
            },
        },
        {
            "name": "vision_only",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_graph=True),
        },
        {
            "name": "graph_only",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_vision=True),
        },
        {
            "name": "no_curriculum",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_curriculum=True),
        },
        {
            "name": "no_proximity",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_reward_components=[RewardComponent.PROXIMITY]),
        },
    ]


def create_default_ablation_study(
    *,
    base_config: TrainingConfig | None = None,
    kg_completeness_values: list[float] | None = None,
    experiments: list[dict] | None = None,
) -> AblationStudy:
    if kg_completeness_values is None:
        kg_completeness_values = list(DEFAULT_KG_COMPLETENESS_VALUES)
    if base_config is None:
        base_config = build_base_config()
    if experiments is None:
        experiments = build_default_experiments(kg_completeness_values)
    return AblationStudy(base_config, kg_completeness_values, experiments=experiments)


def run_ablation_study(
    *,
    base_config: TrainingConfig | None = None,
    kg_completeness_values: list[float] | None = None,
    experiments: list[dict] | None = None,
) -> AblationStudy:
    configure_logging(log_dir="logs", level="INFO")
    ablation_study = create_default_ablation_study(
        base_config=base_config,
        kg_completeness_values=kg_completeness_values,
        experiments=experiments,
    )
    ablation_study.run()
    return ablation_study


def main() -> int:
    try:
        run_ablation_study()
    except Exception as exc:
        logger.error(f"An error occurred during the ablation study: {str(exc)}")
        logger.error(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
