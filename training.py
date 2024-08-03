import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

import cProfile
import pstats
import logging
import traceback
import time
import warnings
import json

from custom_env import CustomEnv
from agent_model import AgentModel

class Logger:
    def __init__(self, log_file='training.log'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        warnings.filterwarnings("always")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

class EnvironmentManager:
    def __init__(self, game_manager_args, simulation_manager_args, model_args):
        self.game_manager_args = game_manager_args
        self.simulation_manager_args = simulation_manager_args
        self.model_args = model_args

    def make_env(self):
        env = CustomEnv(self.game_manager_args, self.simulation_manager_args, self.model_args)
        return Monitor(env)

    def set_kg_completeness(self, env, completeness):
        env.kg_completeness = completeness

class ModelTrainer:
    def __init__(self, env, logger, device):
        self.env = env
        self.logger = logger
        self.device = device
        self.rl_model = None

    def create_model(self, model_config):
        self.logger.info("Creating PPO model")
        self.rl_model = PPO("MultiInputPolicy", 
                    self.env, 
                    policy_kwargs={
                        'features_extractor_class': AgentModel,
                        'features_extractor_kwargs': {'features_dim': 256},
                    },
                    **model_config,
                    device=self.device,
                    verbose=1
        )
        self.logger.info("PPO model created successfully")

    def train(self, total_timesteps, eval_callback, timeout=3600):
        self.logger.info("Starting model training")
        start_time = time.time()
        try:
            for i in range(0, total_timesteps, 2048):
                self.logger.info(f"Training iteration {i//2048 + 1}, timesteps {i}-{min(i+2048, total_timesteps)}")
                
                if time.time() - start_time > timeout:
                    self.logger.warning("Training timed out after 1 hour")
                    break

                with warnings.catch_warnings(record=True) as w:
                    self.rl_model.learn(total_timesteps=2048, callback=eval_callback, reset_num_timesteps=False)
                    if len(w) > 0:
                        self.logger.warning(f"Warnings during training: {[str(warn.message) for warn in w]}")

                self.log_training_stats()

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        self.logger.info("Model training completed or stopped")

    def log_training_stats(self):
        mean_reward = self.calculate_mean_reward()
        mean_episode_length = self.calculate_mean_episode_length()
        self.logger.info(f"Recent mean reward: {mean_reward:.2f}")
        self.logger.info(f"Recent mean episode length: {mean_episode_length:.2f}")
        self.logger.info(f"Episode info buffer size: {len(self.rl_model.ep_info_buffer)}")
        if len(self.rl_model.ep_info_buffer) > 0:
            self.logger.info(f"Sample episode info: {self.rl_model.ep_info_buffer[-1]}")
        self.logger.info(f"Recent policy loss: {self.rl_model.logger.name_to_value['train/policy_loss']:.5f}")
        self.logger.info(f"Recent value loss: {self.rl_model.logger.name_to_value['train/value_loss']:.5f}")

    def calculate_mean_reward(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["r"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def calculate_mean_episode_length(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["l"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def save_model(self, path):
        self.logger.info(f"Saving model to {path}")
        self.rl_model.save(path)
        self.logger.info("Model saved successfully")

    def evaluate_model(self, eval_env, n_eval_episodes=10):
        self.logger.info("Starting final model evaluation")
        mean_reward, std_reward = evaluate_policy(self.rl_model, eval_env, n_eval_episodes=n_eval_episodes)
        self.logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

class AblationStudy:
    def __init__(self, base_config, kg_completeness_values, logger):
        self.base_config = base_config
        self.kg_completeness_values = kg_completeness_values
        self.logger = logger
        self.results = {}

    def run(self):
        self.logger.info("Starting Ablation Study")
        for kg_completeness in self.kg_completeness_values:
            experiment_name = f"kg_completeness_{kg_completeness}"
            self.logger.info(f"Running experiment: {experiment_name}")
            
            trainer = Trainer()
            trainer.setup(self.base_config)
            trainer.env_manager.set_kg_completeness(trainer.env, kg_completeness)
            trainer.env_manager.set_kg_completeness(trainer.eval_env, kg_completeness)
            
            result = trainer.run(experiment_name)
            
            self.results[experiment_name] = result
            
            self.logger.info(f"Experiment {experiment_name} completed")
        
        self._save_results()
        self.logger.info("Ablation Study completed")

    def _save_results(self):
        with open('ablation_study_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        self.logger.info("Ablation study results saved to ablation_study_results.json")

class Trainer:
    def __init__(self):
        self.logger = Logger('ablation_study.log')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def setup(self, config):
        self.config = config
        self.env_manager = EnvironmentManager(config['game_manager_args'], 
                                              config['simulation_manager_args'], 
                                              config['model_args'])
        
        self.logger.info("Creating environment")
        self.env = self.env_manager.make_env()
        self.logger.info("Environment created successfully")

        self.logger.info("Creating evaluation environment")
        self.eval_env = self.env_manager.make_env()
        self.logger.info("Evaluation environment created successfully")

        # # Explicitly set KG completeness for both training and evaluation environments
        # try:
        #     kg_completeness = config['kg_completeness']
        # except KeyError:
        #     logger.error("KG completeness not specified in config")
        # self.env.set_kg_completeness(kg_completeness)
        # self.eval_env.set_kg_completeness(kg_completeness)
        # self.logger.info(f"KG completeness set to {kg_completeness} for both environments")

        self.model_trainer = ModelTrainer(self.env, self.logger, self.device)
        self.model_trainer.create_model(config['model_config'])

    def run(self, experiment_name):
        eval_callback = EvalCallback(self.eval_env, best_model_save_path='./logs/',
                                    log_path='./logs/', eval_freq=10000,
                                    deterministic=True, render=False)

        profiler = cProfile.Profile()
        profiler.enable()

        self.model_trainer.train(total_timesteps=self.config['total_timesteps'], 
                                 eval_callback=eval_callback)

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

        self.model_trainer.save_model(f"ppo_custom_env_{experiment_name}")
        mean_reward, std_reward = self.model_trainer.evaluate_model(self.eval_env)

        self.logger.info("Closing environments")
        self.env.close()
        self.eval_env.close()
        self.logger.info("Environments closed successfully")

        self.logger.info("Training and evaluation completed.")

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'config': self.config
        }

if __name__ == '__main__':
    os.environ['PYGAME_DETECT_AVX2'] = '1'

    base_config = {
        'model_args': {'num_actions': 11},
        'simulation_manager_args': {'number_of_environments': 10, 'number_of_curricula': 1},
        'game_manager_args': {'num_tiles': 8, 'screen_size': 200, 'vision_range': 1},
        'model_config': {
            'n_steps': 2048,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'gamma': 0.99
        },
        'total_timesteps': 1#0
    }

    kg_completeness_values = [0.25, 0.5, 0.75, 1.0]

    logger = Logger('ablation_study.log')
    ablation_study = AblationStudy(base_config, kg_completeness_values, logger)

    try:
        ablation_study.run()
    except Exception as e:
        logger.error(f"An error occurred during the ablation study: {str(e)}")
        logger.error(traceback.format_exc())
        