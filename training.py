import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from custom_env import CustomEnv
from agent_model import AgentModel
import cProfile
import pstats
import multiprocessing

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

# Function to create a vectorized environment
def make_env():
    def _init():
        return CustomEnv(game_manager_args, simulation_manager_args, model_args)
    return _init

# Function to run the training
def train(rl_model, total_timesteps, callback):
    rl_model.learn(total_timesteps=total_timesteps, callback=callback)

def main():
    # Create a vectorized environment
    num_envs = 8  # Adjust based on your CPU cores
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the PPO agent with MultiInputPolicy
    rl_model = PPO("MultiInputPolicy", 
                env, 
                policy_kwargs={
                    'features_extractor_class': AgentModel,
                    'features_extractor_kwargs': {'features_dim': 256},  # You can adjust this value
                },
                n_steps=2048,
                batch_size=256,
                learning_rate=3e-4,
                gamma=0.99,
                device=device,
                verbose=1
    )
    
    # Create an evaluation environment
    eval_env = CustomEnv(game_manager_args, simulation_manager_args, model_args, plot=False)
    
    # Set up evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    
    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the training
    train(rl_model, 1000000, eval_callback)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)
    
    # Save the final trained model
    rl_model.save("ppo_custom_env_final")
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(rl_model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Close the environments
    env.close()
    eval_env.close()
    
    # Save any additional data
    single_env.simulation_manager.save_data()
    
    print("Training and evaluation completed.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()