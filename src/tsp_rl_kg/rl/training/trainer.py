import torch
import os
import cProfile
import pstats
from datetime import datetime
from stable_baselines3.common.callbacks import EvalCallback
from tsp_rl_kg.rl.custom_env import CustomEnv
from tsp_rl_kg.rl.training.model_trainer import ModelTrainer
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.utils.logger import Logger

class Trainer:
    def __init__(self, current_kg_completeness, ablation_study):
        self.logger = Logger('ablation_study.log')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.current_kg_completeness = current_kg_completeness
        self.ablation_study = ablation_study

    def setup(self, config):
        self.config = config
        self.env_manager: EnvironmentManager = EnvironmentManager(config['game_manager_args'], 
                                              config['simulation_manager_args'], 
                                              config['model_args'],
                                              self.ablation_study.converter)
        
        self.logger.info("Creating environment", logger_name='training')
        self.env: CustomEnv = self.env_manager.make_env()
        self.env.unwrapped.simulation_manager.min_episodes_per_curriculum = config['curriculum_config']['min_episodes_per_curriculum']
        self.env.unwrapped.simulation_manager.performance_threshold = config['curriculum_config']['performance_threshold']
        self.logger.info("Environment created successfully", logger_name='training')

        self.logger.info("Creating evaluation environment", logger_name='eval')
        self.eval_env: CustomEnv = self.env_manager.make_env()
        self.eval_env.unwrapped.simulation_manager.min_episodes_per_curriculum = config['curriculum_config']['min_episodes_per_curriculum']
        self.eval_env.unwrapped.simulation_manager.performance_threshold = config['curriculum_config']['performance_threshold']
        self.logger.info("Evaluation environment created successfully", logger_name='eval')

        self.model_trainer = ModelTrainer(self.env, self.eval_env, self.logger, self.device)
        self.model_trainer.create_model(config['model_config'])

    def run(self, experiment_name):
        # Create a subdirectory for this experiment within the results directory
        experiment_dir = os.path.join(self.ablation_study.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        eval_callback = EvalCallback(self.eval_env, best_model_save_path=experiment_dir,
                                    log_path=experiment_dir, eval_freq=10000,
                                    deterministic=True, render=False)

        profiler = cProfile.Profile()
        profiler.enable()

        self.model_trainer.train(
            total_timesteps=self.config['total_timesteps'], 
            eval_callback=eval_callback,
            timeout=3600)
        
        # Save metrics
        metrics_file = os.path.join(experiment_dir, f"{experiment_name}_metrics.csv")
        self.model_trainer.metrics.save_to_csv(metrics_file)

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats_file = os.path.join(experiment_dir, 'profile_stats.txt')
        stats.dump_stats(stats_file)
        self.logger.info(f"Profiling stats saved to {stats_file}")

        model_path = os.path.join(experiment_dir, f"ppo_custom_env_{experiment_name}")
        self.model_trainer.save_model(model_path)
        mean_reward, std_reward = self.model_trainer.evaluate_model(self.eval_env)

        self.logger.info("Closing environments", logger_name='training')
        self.env_manager.set_kg_completeness(self.env, self.current_kg_completeness)
        self.env.close()
        self.env_manager.set_kg_completeness(self.eval_env, self.current_kg_completeness)
        self.eval_env.close()
        self.logger.info("Environments closed successfully", logger_name='training')

        self.logger.info("Training and evaluation completed.")

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'config': self.config,
            'model_path': model_path,
            'stats_file': stats_file
        }
