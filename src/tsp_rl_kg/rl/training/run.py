import traceback

from tsp_rl_kg.config import (
    AblationConfig,
    CurriculumConfig,
    GameManagerConfig,
    ModelArgs,
    ModelConfig,
    RewardComponent,
    SimulationManagerConfig,
    TrainingConfig,
)
from tsp_rl_kg.rl.training.ablation_study import AblationStudy
from tsp_rl_kg.utils.logger import Logger

# Uncomment for windows
# os.environ['PYGAME_DETECT_AVX2'] = '1'
min_episodes_per_curriculum = 4
base_config = TrainingConfig(
    game_manager=GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
    simulation_manager=SimulationManagerConfig(
        number_of_environments=3000,
        number_of_curricula=30,
        min_episodes_per_curriculum=min_episodes_per_curriculum,
    ),
    model_args=ModelArgs(num_actions=11),
    model_config=ModelConfig(
        n_steps=2048 * 2,
        batch_size=512,
        learning_rate=6e-4,
        gamma=0.995,
    ),
    curriculum=CurriculumConfig(
        min_episodes_per_curriculum=min_episodes_per_curriculum,
        performance_threshold=0.85,
    ),
    total_timesteps=100000,
    seeds=[42, 123, 456],
)

kg_completeness_values = [0.25, 0.5, 0.75, 1.0]

# Optional: define a custom experiment matrix for component ablation
experiments = [
    # KG completeness sweep
    *[
        {"name": f"kg_{kg}", "kg_completeness": kg, "ablation": AblationConfig()}
        for kg in kg_completeness_values
    ],
    # Vision-only (graph disabled)
    {"name": "vision_only", "kg_completeness": 0.5, "ablation": AblationConfig(disable_graph=True)},
    # Graph-only (vision disabled)
    {"name": "graph_only", "kg_completeness": 0.5, "ablation": AblationConfig(disable_vision=True)},
    # No curriculum
    {
        "name": "no_curriculum",
        "kg_completeness": 0.5,
        "ablation": AblationConfig(disable_curriculum=True),
    },
    # No proximity reward
    {
        "name": "no_proximity",
        "kg_completeness": 0.5,
        "ablation": AblationConfig(disable_reward_components=[RewardComponent.PROXIMITY]),
    },
]

logger = Logger("ablation_study.log")
ablation_study = AblationStudy(base_config, kg_completeness_values, logger, experiments=experiments)

try:
    ablation_study.run()
except Exception as e:
    logger.error(f"An error occurred during the ablation study: {str(e)}")
    logger.error(traceback.format_exc())
