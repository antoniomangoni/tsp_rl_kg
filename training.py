from stable_baselines3 import PPO
from custom_env import CustomEnv
import numpy as np

from agent_model import AgentModel

# Define the arguments for the environment and model
model_args = {
    'num_actions': 11
}

simulation_manager_args = {
    'number_of_environments': 10,
    'number_of_curricula': 3
}

game_manager_args = {
    'map_pixel_size': 32,
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

# Define model initialization parameters directly from environment setup
vision_shape = (3, 2 * game_manager_args['vision_range'] + 1, 2 * game_manager_args['vision_range'] + 1)
num_graph_features = env.kg.graph.x.shape[1]
num_edge_features = env.kg.graph.edge_attr.shape[1]

model = PPO("MultiInputPolicy", env, policy_kwargs={
    'features_extractor_class': AgentModel,
    'features_extractor_kwargs': {
        'num_graph_features': num_graph_features,
        'num_edge_names': num_edge_features,
        'num_actions': model_args['num_actions']
    }
}, verbose=1)
"""
the observation is the vision and the whole graph.
Then the graph is separated into its nodes and egdes in the model.
also why are the num_actions in the features if they are the outputs?
"""


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
