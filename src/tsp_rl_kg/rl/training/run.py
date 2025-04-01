import os
import traceback
from tsp_rl_kg.rl.training.ablation_study import AblationStudy
from tsp_rl_kg.utils.logger import Logger

# Uncomment for windows
# os.environ['PYGAME_DETECT_AVX2'] = '1'
min_episodes_per_curriculum = 4
base_config = {
    'model_args': {'num_actions': 11},
    'simulation_manager_args': {
        'number_of_environments': 3000,
        'number_of_curricula': 30,
        'min_episodes_per_curriculum': min_episodes_per_curriculum},
    'game_manager_args': {'num_tiles': 5, 'screen_size': 20, 'vision_range': 1},
    'model_config': {
        'n_steps': 2048 * 2,
        'batch_size': 512,
        'learning_rate': 6e-4,
        'gamma': 0.995
    },
    'curriculum_config': {
    'min_episodes_per_curriculum': min_episodes_per_curriculum,
    'performance_threshold': 0.85,
    },
    'total_timesteps': 100000
}

kg_completeness_values = [0.25, 0.5, 0.75, 1.0]

logger = Logger('ablation_study.log')
ablation_study = AblationStudy(base_config, kg_completeness_values, logger)

try:
    ablation_study.run()
except Exception as e:
    logger.error(f"An error occurred during the ablation study: {str(e)}")
    logger.error(traceback.format_exc())
