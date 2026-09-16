import random

import numpy as np

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
        seed: int = 0,
    ):
        self.game_manager_config = game_manager_config
        self.simulation_manager_config = simulation_manager_config
        self.model_args = model_args
        self.feature_encoder = feature_encoder
        self.episode_config = episode_config
        self.ablation_config = ablation_config if ablation_config is not None else AblationConfig()
        self.kg_completeness = kg_completeness
        self.seed = seed

    def _make_seeded_env(self, stream, *, evaluation=False):
        # World generation still uses Python/NumPy globals; isolate it from model RNG.
        py_state, np_state = random.getstate(), np.random.get_state()
        world_seed = int(np.random.SeedSequence([self.seed, stream]).generate_state(1)[0])
        try:
            random.seed(world_seed)
            np.random.seed(world_seed)
            return CustomEnv(
                self.game_manager_config,
                self.simulation_manager_config,
                self.model_args,
                self.feature_encoder,
                plot=False,
                episode_config=self.episode_config,
                ablation_config=self.ablation_config,
                kg_completeness=self.kg_completeness,
                seed=self.seed,
                evaluation=evaluation,
            )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def make_env(self):
        return self._make_seeded_env(0)

    def make_eval_env(self, train_game_managers):
        train_ids = {gm.world_template.fingerprint for gm in train_game_managers}
        for stream in range(1, 11):
            env = self._make_seeded_env(stream, evaluation=True)
            eval_ids = {
                gm.world_template.fingerprint for gm in env.simulation_manager.game_managers
            }
            if train_ids.isdisjoint(eval_ids):
                return env
            env.close()
        raise ValueError("Could not construct a disjoint held-out world pool after 10 attempts")

    def set_kg_completeness(self, env, completeness):
        # Access the unwrapped environment to set KG completeness
        env.unwrapped.set_kg_completeness(completeness)
