from stable_baselines3.common.monitor import Monitor

from tsp_rl_kg.config import GameManagerConfig, ModelArgs, SimulationManagerConfig
from tsp_rl_kg.rl.custom_env import CustomEnv


class EnvironmentManager:
    def __init__(
        self,
        game_manager_config: GameManagerConfig | dict,
        simulation_manager_config: SimulationManagerConfig | dict,
        model_args: ModelArgs | dict,
        converter,
    ):
        self.game_manager_config = game_manager_config
        self.simulation_manager_config = simulation_manager_config
        self.model_args = model_args
        self.converter = converter

    def make_env(self):
        env = CustomEnv(
            self.game_manager_config,
            self.simulation_manager_config,
            self.model_args,
            self.converter,
            plot=False,
        )
        return Monitor(env)

    def set_kg_completeness(self, env, completeness):
        # Access the unwrapped environment to set KG completeness
        env.unwrapped.set_kg_completeness(completeness)
