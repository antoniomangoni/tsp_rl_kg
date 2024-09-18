import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
torch.set_default_dtype(torch.float32)
import cProfile
import pstats
import traceback
import time
import warnings
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from custom_env import CustomEnv
from agent_model import AgentModel
from logger import Logger

torch.cuda.empty_cache()

class EnvironmentManager:
    def __init__(self, game_manager_args, simulation_manager_args, model_args):
        self.game_manager_args = game_manager_args
        self.simulation_manager_args = simulation_manager_args
        self.model_args = model_args

    def make_env(self):
        env = CustomEnv(self.game_manager_args, self.simulation_manager_args, self.model_args, plot = False)
        return Monitor(env)

    def set_kg_completeness(self, env, completeness):
        # Access the unwrapped environment to set KG completeness
        env.unwrapped.set_kg_completeness(completeness)

class TrainingMetrics:
    def __init__(self, num_actions):
        self.steps = []
        self.performances = []
        self.game_manager_indices = []
        self.best_route_energies = []
        self.curriculum_steps = []
        self.target_route_energies = []
        self.efficiency = []
        self.improvement = []
        self.gap = []
        self.num_actions = num_actions
        self.action_counts = [[] for _ in range(num_actions)]

    def add_metric(self, step, performance, game_manager_index,
                   best_route_energy, curriculum_step, target_route_energy,
                   efficiency, improvement, gap, action_counts):
        self.steps.append(step)
        self.performances.append(performance)
        self.game_manager_indices.append(game_manager_index)
        self.best_route_energies.append(best_route_energy)
        self.curriculum_steps.append(curriculum_step)
        self.target_route_energies.append(target_route_energy)
        self.efficiency.append(efficiency)
        self.improvement.append(improvement)
        self.gap.append(gap)
        for i, count in enumerate(action_counts):
            self.action_counts[i].append(count)

    def save_to_csv(self, filename):
        # replace any special characters and spaces in the filename with underscores
        filename = filename.replace(' ', '_')
        filename = filename.replace('-', '_')
        filename = filename.replace('.', '_')
        filename = ''.join(e for e in filename if e.isalnum() or e == '_')
        # if file ends with _csv replace it with .csv
        if filename.endswith('_csv'):
            filename = filename[:-4] + '.csv'

        df = pd.DataFrame({
            'Step': self.steps,
            'Performance': self.performances,
            'Game Manager Index': self.game_manager_indices,
            'Best Route Energy': self.best_route_energies,
            'Curriculum Step': self.curriculum_steps,
            'Target Route Energy': self.target_route_energies,
            'Efficiency': self.efficiency,
            'Improvement': self.improvement,
            'Gap': self.gap
        })

        total_actions = np.sum(self.action_counts, axis=0)
        # Add columns for each action
        for i in range(self.num_actions):
            df[f'Action_{i}'] = self.action_counts[i] / total_actions

        df.to_csv(filename, index=False)

class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, custom_logger, metrics, print_weight_stats_freq=1000, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.custom_logger = custom_logger
        self.metrics = metrics
        self.should_stop = False
        self.print_weight_stats_freq = print_weight_stats_freq
        self.action_counts = np.zeros(metrics.num_actions, dtype=int)
        self.num_envs = getattr(eval_env.unwrapped, 'num_envs', 1)  # Use unwrapped to access num_envs

    def _on_step(self) -> bool:
        # Update action counts based on the actions taken
        actions = self.locals['actions']
        if isinstance(actions, np.ndarray):
            self.action_counts += np.bincount(actions, minlength=self.metrics.num_actions)
        else:
            self.action_counts[actions] += 1

        if self.n_calls % self.model.n_steps == 0:
            self.custom_logger.info(f"Step {self.n_calls}", logger_name='training')
            
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            unwrapped_env = env.unwrapped  # Get the unwrapped environment

            if unwrapped_env.early_stop:
                self.custom_logger.info("Early stop condition met. Stopping training.", logger_name='training')
                self.should_stop = True
                return False 
            
            metrics = unwrapped_env.get_metrics()

            performance = metrics.get('performance', 0)
            game_manager_index = metrics.get('game_manager_index', 0)
            best_route_energy = metrics.get('best_route_energy', 0)
            curriculum_step = metrics.get('curriculum_step', 0)
            target_route_energy = metrics.get('target_route_energy', 0)
            efficiency = metrics.get('best_efficiency', 0)
            improvement = metrics.get('improvement', 0)
            gap = metrics.get('gap', 0)

            self.metrics.add_metric(
                self.n_calls, performance, game_manager_index,
                best_route_energy, curriculum_step, target_route_energy,
                efficiency, improvement, gap,
                self.action_counts
            )
            
            # Reset action counts
            self.action_counts = np.zeros(self.metrics.num_actions, dtype=int)
            
            # Check if curriculum should advance
            if unwrapped_env.simulation_manager.should_advance_curriculum():
                unwrapped_env.simulation_manager.advance_curriculum()
                self.custom_logger.info(f"Advancing to curriculum level {unwrapped_env.simulation_manager.current_curriculum_index}", logger_name='training')
                
                # Reset both training and eval environments
                self.training_env.reset()
                self.eval_env.reset()

            # Print weight statistics
            if self.n_calls % self.print_weight_stats_freq == 0:
                self.print_weight_statistics()

        return True

    def print_weight_statistics(self):
        self.custom_logger.info("Weight Statistics:", logger_name='training')
        agent_model = self.model.policy.features_extractor

        # Vision Processor
        self.print_module_statistics(agent_model.vision_processor, "Vision Processor")

        # Graph Processor
        self.print_module_statistics(agent_model.graph_processor, "Graph Processor")

        # Combined Fully Connected Layers
        self.print_module_statistics(agent_model.fc, "Combined FC")

        # Dropout layer doesn't have learnable parameters, so we skip it

    def print_module_statistics(self, module, module_name):
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, (nn.Conv2d, nn.Linear, GATConv)):
                self.print_layer_statistics(sub_module, f"{module_name} - {name}")

    def print_layer_statistics(self, layer, layer_name):
        if hasattr(layer, 'weight'):
            weights = layer.weight.data
            weight_stats = self.compute_stats(weights)
            self.custom_logger.info(f"{layer_name} weights - {weight_stats}", logger_name='training')
            # print(f"{layer_name} weights - {weight_stats}")

        if hasattr(layer, 'bias') and layer.bias is not None:
            bias = layer.bias.data
            bias_stats = self.compute_stats(bias)
            self.custom_logger.info(f"{layer_name} bias - {bias_stats}", logger_name='training')
            # print(f"{layer_name} bias - {bias_stats}")

    def compute_stats(self, tensor):
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item()
        }

class ModelTrainer:
    def __init__(self, env, eval_env, logger, device):
        self.env = env
        self.eval_env = eval_env
        self.logger = logger
        self.device = device
        self.rl_model = None
        self.metrics = TrainingMetrics(env.action_space.n)

    def create_model(self, model_config):
        self.logger.info("Creating PPO model", logger_name='training')
        self.rl_model = PPO("MultiInputPolicy", 
                    self.env, 
                    policy_kwargs={
                        'features_extractor_class': AgentModel,
                        'features_extractor_kwargs': {'features_dim': 64}
                    },
                    **model_config,
                    device=self.device,
                    verbose=1
        )
        self.logger.info("PPO model created successfully", logger_name='training')

    def train(self, total_timesteps, eval_callback, timeout=3600):
        self.logger.info("Starting model training", logger_name='training')
        
        curriculum_callback = CurriculumCallback(self.eval_env, self.logger, self.metrics)
        
        try:
            self.rl_model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, curriculum_callback],
                reset_num_timesteps=False,
                tb_log_name="PPO",
                progress_bar=True
            )
        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        if curriculum_callback.should_stop:
            self.logger.info("Training ended early due to early stop condition", logger_name='training')
        else:
            self.logger.info("Model training completed", logger_name='training')
        
        # Save metrics after training
        self.metrics.save_to_csv("training_metrics.csv")

    def log_training_stats(self):
        mean_reward = self.calculate_mean_reward()
        mean_episode_length = self.calculate_mean_episode_length()
        self.logger.info(f"Recent mean reward: {mean_reward:.2f}", logger_name='training')
        self.logger.info(f"Recent mean episode length: {mean_episode_length:.2f}", logger_name='training')
        self.logger.info(f"Episode info buffer size: {len(self.rl_model.ep_info_buffer)}", logger_name='training')
        if len(self.rl_model.ep_info_buffer) > 0:
            self.logger.info(f"Sample episode info: {self.rl_model.ep_info_buffer[-1]}", logger_name='training')
        self.logger.info(f"Recent policy loss: {self.rl_model.logger.name_to_value['train/policy_loss']:.5f}", logger_name='training')
        self.logger.info(f"Recent value loss: {self.rl_model.logger.name_to_value['train/value_loss']:.5f}", logger_name='training')

    def calculate_mean_reward(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["r"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def calculate_mean_episode_length(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["l"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def save_model(self, path):
        self.logger.info(f"Saving model to {path}", logger_name='training')
        self.rl_model.save(path)
        self.logger.info("Model saved successfully", logger_name='training')

    def evaluate_model(self, eval_env, n_eval_episodes=10):
        self.logger.info("Starting final model evaluation", logger_name='eval')
        
        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        self.logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", logger_name='eval')
        return mean_reward, std_reward

class AblationStudy:
    def __init__(self, base_config, kg_completeness_values, logger):
        self.base_config = base_config
        self.kg_completeness_values = kg_completeness_values
        self.logger = logger
        self.results = {}
        self.results_dir = self._create_results_directory()

    def _create_results_directory(self):
        # Create a 'results' folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Create a subfolder with the current datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join('results', current_time)
        os.makedirs(result_dir)
        
        self.logger.info(f"Created results directory: {result_dir}")
        return result_dir

    def run(self):
        self.logger.info("Starting Ablation Study")
        for kg_completeness in self.kg_completeness_values:
            experiment_name = f"kg_completeness_{kg_completeness}"
            self.logger.info(f"Running experiment: {experiment_name}")
            
            try:
                trainer = Trainer(kg_completeness, ablation_study=self)
                trainer.setup(self.base_config)
                trainer.env_manager.set_kg_completeness(trainer.env, kg_completeness)
                trainer.env_manager.set_kg_completeness(trainer.eval_env, kg_completeness)
                
                result = trainer.run(experiment_name)
                
                self.results[experiment_name] = result
                
                self.logger.info(f"Experiment {experiment_name} completed")
            except Exception as e:
                self.logger.error(f"An error occurred during experiment {experiment_name}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        self._save_results()
        self.logger.info("Ablation Study completed")

    def _save_results(self):
        results_file = os.path.join(self.results_dir, 'ablation_study_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        self.logger.info(f"Ablation study results saved to {results_file}")

        # Save individual experiment results
        for experiment_name, result in self.results.items():
            experiment_file = os.path.join(self.results_dir, f"{experiment_name}_results.json")
            with open(experiment_file, 'w') as f:
                json.dump(result, f, indent=4)
            self.logger.info(f"Individual experiment results saved to {experiment_file}")

        # Save the base configuration
        config_file = os.path.join(self.results_dir, 'base_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.base_config, f, indent=4)
        self.logger.info(f"Base configuration saved to {config_file}")

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
                                              config['model_args'])
        
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

if __name__ == '__main__':
    os.environ['PYGAME_DETECT_AVX2'] = '1'
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
