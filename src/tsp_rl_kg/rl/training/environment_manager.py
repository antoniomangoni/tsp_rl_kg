from tsp_rl_kg.rl.custom_env import CustomEnv
from stable_baselines3.common.monitor import Monitor

class EnvironmentManager:
    def __init__(self, game_manager_args, simulation_manager_args, model_args, converter):
        self.game_manager_args = game_manager_args
        self.simulation_manager_args = simulation_manager_args
        self.model_args = model_args
        self.converter = converter

    def make_env(self):
        env = CustomEnv(self.game_manager_args, self.simulation_manager_args, self.model_args, self.converter, plot = False)
        return Monitor(env)

    def set_kg_completeness(self, env, completeness):
        # Access the unwrapped environment to set KG completeness
        env.unwrapped.set_kg_completeness(completeness)