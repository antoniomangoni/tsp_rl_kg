import copy

from tsp_rl_kg.config import (
    AblationConfig,
    EpisodeConfig,
    GameManagerConfig,
    ModelArgs,
    SimulationManagerConfig,
)
from tsp_rl_kg.rl.custom_env import CustomEnv


class EnvironmentManager:
    def __init__(
        self,
        game_manager_config: GameManagerConfig | dict,
        simulation_manager_config: SimulationManagerConfig | dict,
        model_args: ModelArgs | dict,
        feature_encoder,
        episode_config: EpisodeConfig | None = None,
        ablation_config: AblationConfig | None = None,
        kg_completeness: float = 0.5,
    ):
        self.game_manager_config = game_manager_config
        self.simulation_manager_config = simulation_manager_config
        self.model_args = model_args
        self.feature_encoder = feature_encoder
        self.episode_config = episode_config
        self.ablation_config = ablation_config if ablation_config is not None else AblationConfig()
        self.kg_completeness = kg_completeness

    def make_env(self):
        return CustomEnv(
            self.game_manager_config,
            self.simulation_manager_config,
            self.model_args,
            self.feature_encoder,
            plot=False,
            episode_config=self.episode_config,
            ablation_config=self.ablation_config,
            kg_completeness=self.kg_completeness,
        )

    def make_eval_env(self, train_game_managers):
        eval_game_managers = copy.deepcopy(train_game_managers)
        return CustomEnv(
            self.game_manager_config,
            self.simulation_manager_config,
            self.model_args,
            self.feature_encoder,
            plot=False,
            episode_config=self.episode_config,
            game_managers=eval_game_managers,
            ablation_config=self.ablation_config,
            kg_completeness=self.kg_completeness,
        )

    def set_kg_completeness(self, env, completeness):
        # Access the unwrapped environment to set KG completeness
        env.unwrapped.set_kg_completeness(completeness)
