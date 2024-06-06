from stable_baselines3 import PPO
from custom_env import CustomEnv
import numpy as np

# Define the arguments for the environment and model
model_args = {
    'num_actions': 11
}

simulation_manager_args = {
    'number_of_environments': 10,
    'number_of_curricula': 3
}

game_manager_args = {
    'map_pixel_size': 16,
    'screen_size': 800,
    'kg_completeness': 1,
    'vision_range': 2
}

# Initialize the custom environment
env = CustomEnv(game_manager_args, simulation_manager_args, model_args)

# Add evaluate_policy method to CustomEnv class
def evaluate_policy(self, model, env, n_eval_episodes=10):
    all_episode_rewards = []

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward

        all_episode_rewards.append(episode_rewards)

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    return mean_reward, std_reward

# Bind the new method to the CustomEnv instance
CustomEnv.evaluate_policy = evaluate_policy

# Instantiate the RL model (PPO in this case)
model = PPO('MultiInputPolicy', env, verbose=2)

# Train the model
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_custom_env")
env.simulation_manager.save_data()

# Evaluate the model
mean_reward, std_reward = env.evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Close the environment
env.close()
