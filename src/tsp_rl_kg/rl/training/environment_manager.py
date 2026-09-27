"""Reproducible terrain-disjoint pools, shared by all workers and policy arms."""

from dataclasses import asdict, replace

import numpy as np

from tsp_rl_kg.config import (
    AblationConfig,
    CurriculumConfig,
    EvaluationConfig,
    GameManagerConfig,
    RewardConfig,
    SimulationManagerConfig,
)
from tsp_rl_kg.game_world.game_manager import GameManager
from tsp_rl_kg.rl.custom_env import CustomEnv, typed


class EnvironmentManager:
    def __init__(
        self,
        game_manager_config,
        simulation_manager_config,
        model_args,
        feature_encoder,
        episode_config=None,
        ablation_config=None,
        kg_completeness=0.5,
        seed=0,
        reward_config=None,
        curriculum_config=None,
        evaluation_config=None,
        gamma=0.99,
    ):
        self.game_manager_config = typed(game_manager_config, GameManagerConfig)
        self.simulation_manager_config = typed(simulation_manager_config, SimulationManagerConfig)
        self.model_args = model_args
        self.feature_encoder = feature_encoder
        self.episode_config = episode_config
        self.ablation_config = ablation_config or AblationConfig()
        self.kg_completeness = kg_completeness
        self.seed = seed
        self.reward_config = reward_config or RewardConfig()
        self.curriculum_config = curriculum_config or CurriculumConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.gamma = gamma
        self.pools = {}

    def _pool(self, label, count, excluded=()):
        if label in self.pools:
            return self.pools[label]
        stream = {"train": 0, "validation": 1, "test": 2}[label]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, stream]))
        used = set(excluded)
        candidates = []
        # Generate cheap immutable templates first; build policy graphs only for selected worlds.
        wanted = count if label == "train" else max(count, 5 * count)
        for _ in range(max(100, wanted * 100)):
            if len(candidates) == wanted:
                break
            try:
                gm = GameManager(
                    config=self.game_manager_config, feature_encoder=self.feature_encoder, rng=rng
                )
            except ValueError as error:
                if "outposts" not in str(error):
                    raise
                continue
            template = gm.world_template
            start_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(template.terrain_fingerprint[:8], 16)])
            )
            template = replace(
                template, spawn=template.outposts[int(start_rng.integers(len(template.outposts)))]
            )
            if len(template.outposts) != self.game_manager_config.number_of_outposts:
                continue
            if template.terrain_fingerprint in used:
                continue
            used.add(template.terrain_fingerprint)
            candidates.append((gm.target_manager.target_route_energy, template))
        if len(candidates) < wanted:
            raise ValueError(
                f"Cannot generate {wanted} unique reachable {label} terrains; enlarge the map"
            )
        candidates.sort(key=lambda item: (item[0], item[1].fingerprint))
        if label != "train":
            bins = np.array_split(np.arange(len(candidates)), min(5, count))
            selected = []
            for i in range(count):
                bucket = bins[i % len(bins)]
                available = [int(j) for j in bucket if int(j) not in selected]
                selected.append(int(rng.choice(available)))
            candidates = [candidates[i] for i in selected]
        self.pools[label] = [template for _, template in candidates]
        return self.pools[label]

    def _make(self, templates, *, evaluation=False, worker=0):
        games = [
            GameManager(
                config=self.game_manager_config, feature_encoder=self.feature_encoder, template=t
            )
            for t in templates
        ]
        env = CustomEnv(
            self.game_manager_config,
            self.simulation_manager_config,
            self.model_args,
            self.feature_encoder,
            game_managers=games,
            episode_config=self.episode_config,
            ablation_config=self.ablation_config,
            kg_completeness=self.kg_completeness,
            seed=self.seed,
            evaluation=evaluation,
            reward_config=self.reward_config,
            curriculum_config=self.curriculum_config,
            gamma=self.gamma,
        )
        # Worker scheduling differs, but template/prior identity remains paired across arms.
        env.np_random = np.random.default_rng(np.random.SeedSequence([self.seed, 10, worker]))
        if evaluation:
            by_terrain = {
                g.world_template.terrain_fingerprint: g
                for g in env.simulation_manager.game_managers
            }
            env.simulation_manager.game_managers = [
                by_terrain[t.terrain_fingerprint] for t in templates
            ]
            env.begin_evaluation()
        return env

    def make_env(self, worker=0):
        return self._make(
            self._pool("train", self.simulation_manager_config.number_of_environments),
            worker=worker,
        )

    def make_eval_env(self, train_game_managers=None):
        train = self._pool("train", self.simulation_manager_config.number_of_environments)
        excluded = {t.terrain_fingerprint for t in train}
        if train_game_managers:
            excluded.update(g.world_template.terrain_fingerprint for g in train_game_managers)
        return self._make(
            self._pool("validation", self.evaluation_config.n_eval_episodes, excluded),
            evaluation=True,
        )

    def make_test_env(self):
        if "validation" not in self.pools:
            raise RuntimeError("Create validation pool before the final-test pool")
        excluded = {t.terrain_fingerprint for ts in self.pools.values() for t in ts}
        return self._make(
            self._pool("test", self.evaluation_config.final_eval_episodes, excluded),
            evaluation=True,
        )

    def manifest(self):
        return {
            label: [
                dict(terrain_id=t.terrain_fingerprint, world_id=t.fingerprint, template=asdict(t))
                for t in templates
            ]
            for label, templates in self.pools.items()
        }

    def set_kg_completeness(self, env, completeness):
        env.unwrapped.set_kg_completeness(completeness)
