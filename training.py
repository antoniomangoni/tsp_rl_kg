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

from custom_env import CustomEnv
from agent_model import AgentModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Capture warnings
warnings.filterwarnings("always")  # Change to "error" to raise warnings as exceptions

os.environ['PYGAME_DETECT_AVX2'] = '1'

# Define the arguments for the environment and model
model_args = {
    'num_actions': 11
}
simulation_manager_args = {
    'number_of_environments': 10,
    'number_of_curricula': 3
}
game_manager_args = {
    'num_tiles': 8,
    'screen_size': 200,
    'kg_completeness': 1,
    'vision_range': 1
}

def calculate_mean_reward(ep_info_buffer):
    if len(ep_info_buffer) > 0:
        return np.mean([ep_info["r"] for ep_info in ep_info_buffer])
    return 0.0

def calculate_mean_episode_length(ep_info_buffer):
    if len(ep_info_buffer) > 0:
        return np.mean([ep_info["l"] for ep_info in ep_info_buffer])
    return 0.0

def make_env():
    env = CustomEnv(game_manager_args, simulation_manager_args, model_args)
    return Monitor(env)

def main():
    try:
        logger.info("Starting main function")
        
        # Create a single environment
        logger.info("Creating environment")
        env = make_env()
        logger.info("Environment created successfully")

        # Set up the device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Instantiate the PPO agent with MultiInputPolicy
        logger.info("Creating PPO model")
        rl_model = PPO("MultiInputPolicy", 
                    env, 
                    policy_kwargs={
                        'features_extractor_class': AgentModel,
                        'features_extractor_kwargs': {'features_dim': 256},
                    },
                    n_steps=2048,
                    batch_size=64,  # Reduced batch size for single environment
                    learning_rate=3e-4,
                    gamma=0.99,
                    device=device,
                    verbose=1
        )
        logger.info("PPO model created successfully")

        # Create an evaluation environment
        logger.info("Creating evaluation environment")
        eval_env = make_env()
        logger.info("Evaluation environment created successfully")

        # Set up evaluation callback
        logger.info("Setting up evaluation callback")
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                    log_path='./logs/', eval_freq=10000,
                                    deterministic=True, render=False)
        logger.info("Evaluation callback set up successfully")

        # Profile the training
        profiler = cProfile.Profile()
        profiler.enable()

        # Run the training
        logger.info("Starting model training")
        total_timesteps = 1000000
        timeout = 3600  # 1 hour timeout
        start_time = time.time()
        try:
            for i in range(0, total_timesteps, 2048):  # 2048 is the n_steps parameter
                logger.info(f"Training iteration {i//2048 + 1}, timesteps {i}-{min(i+2048, total_timesteps)}")
                
                # Check for timeout
                if time.time() - start_time > timeout:
                    logger.warning("Training timed out after 1 hour")
                    break

                with warnings.catch_warnings(record=True) as w:
                    rl_model.learn(total_timesteps=2048, callback=eval_callback, reset_num_timesteps=False)
                    if len(w) > 0:
                        logger.warning(f"Warnings during training: {[str(warn.message) for warn in w]}")

                logger.info(f"Completed training iteration {i//2048 + 1}")

                # Log some training statistics
                mean_reward = calculate_mean_reward(rl_model.ep_info_buffer)
                mean_episode_length = calculate_mean_episode_length(rl_model.ep_info_buffer)
                logger.info(f"Recent mean reward: {mean_reward:.2f}")
                logger.info(f"Recent mean episode length: {mean_episode_length:.2f}")

                # Log the size of the episode info buffer
                logger.info(f"Episode info buffer size: {len(rl_model.ep_info_buffer)}")

                # Log a sample of the episode info buffer
                if len(rl_model.ep_info_buffer) > 0:
                    logger.info(f"Sample episode info: {rl_model.ep_info_buffer[-1]}")

                # Log policy and value losses
                logger.info(f"Recent policy loss: {rl_model.logger.name_to_value['train/policy_loss']:.5f}")
                logger.info(f"Recent value loss: {rl_model.logger.name_to_value['train/value_loss']:.5f}")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("Model training completed or stopped")

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

        # Save the final trained model
        logger.info("Saving final trained model")
        rl_model.save("ppo_custom_env_final")
        logger.info("Final trained model saved successfully")

        # Evaluate the model
        logger.info("Starting final model evaluation")
        mean_reward, std_reward = evaluate_policy(rl_model, eval_env, n_eval_episodes=10)
        logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Close the environments
        logger.info("Closing environments")
        env.close()
        eval_env.close()
        logger.info("Environments closed successfully")

        # Save any additional data
        logger.info("Saving additional data")
        env.simulation_manager.save_data()
        logger.info("Additional data saved successfully")

        logger.info("Training and evaluation completed.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
