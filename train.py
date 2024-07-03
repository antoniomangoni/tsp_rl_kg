from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import set_random_seed
import torch
from custom_env import CustomEnv
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
    'num_tiles': 8,
    'screen_size': 400,
    'kg_completeness': 1,
    'vision_range': 1
}

# Initialize the custom environment
env = CustomEnv(game_manager_args, simulation_manager_args, model_args, plot=False)

# Set random seed for reproducibility
set_random_seed(42)

# Subclass PPO to use the custom AgentModel
class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(CustomPPO, self).__init__(*args, **kwargs)

    def _build_model(self) -> ActorCriticPolicy:
        policy_kwargs = dict(
            net_arch=[dict(pi=[], vf=[]),  # No shared MLP
                       dict(pi=[], vf=[]),  # Separate MLPs for actor/critic
                       ],
            activation_fn=torch.nn.ReLU,
            squash_output=False,
            features_extractor_class=lambda observation_space: AgentModel(observation_space, env.kg.graph.num_node_features, model_args['num_actions']),
            features_extractor_kwargs=dict(features_dim=256),
        )

        return super()._build_model(policy_kwargs=policy_kwargs)

# Initialize the PPO model with the custom environment
model = CustomPPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_custom_model")

# Load the saved model (optional)
# model = CustomPPO.load("ppo_custom_model")

# Evaluate the trained model
n_eval_episodes = 10
mean_reward, std_reward = model.evaluate_policy(env, n_eval_episodes=n_eval_episodes)
print(f"Evaluation over {n_eval_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")

# Close the environment
env.close()
