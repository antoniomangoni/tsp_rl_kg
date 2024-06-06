from stable_baselines3 import PPO
from custom_env import CustomEnv

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
    'kg_completness': 1,
    'vision_range': 2
}

# Initialize the custom environment
env = CustomEnv(game_manager_args, simulation_manager_args, model_args)

# Instantiate the RL model (PPO in this case)
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_custom_env")

# Optionally, evaluate the model
def evaluate_model(model, env, num_environments=10):
    """
    Evaluate the trained model across different environments of increasing difficulty.
    """
    for i in range(num_environments):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Environment {i + 1}: Total Reward = {total_reward}")

# Evaluate the trained model across the different environments
evaluate_model(model, env, num_environments=simulation_manager_args['number_of_environments'])
