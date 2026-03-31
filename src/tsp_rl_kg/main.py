from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
from loguru import logger

from tsp_rl_kg.config import (
    AlgorithmConfig,
    AlgorithmName,
    CurriculumConfig,
    GameManagerConfig,
    ModelArgs,
    SimulationManagerConfig,
    TrainingConfig,
    default_algorithm_hyperparameters,
)
from tsp_rl_kg.game_world.game_manager import GameManager
from tsp_rl_kg.rl.simulation_manager import SimulationManager
from tsp_rl_kg.rl.training.trainer import Trainer
from tsp_rl_kg.utils.logger import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entry points for tsp_rl_kg")
    parser.add_argument(
        "mode",
        nargs="?",
        default="play",
        choices=["play", "train", "simulate"],
        help="Run a single game world, a config-driven training demo, or export worlds.",
    )
    parser.add_argument(
        "--algorithm",
        default=AlgorithmName.PPO.value,
        choices=[algorithm.value for algorithm in AlgorithmName],
        help="Training backend algorithm for train mode.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=512, help="Training timesteps in train mode."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used in train mode.")
    parser.add_argument(
        "--kg-completeness", type=float, default=0.5, help="KG completeness used in train mode."
    )
    parser.add_argument(
        "--num-tiles", type=int, default=None, help="Override the world width and height in tiles."
    )
    parser.add_argument(
        "--screen-size", type=int, default=None, help="Override the renderer screen size."
    )
    parser.add_argument(
        "--vision-range", type=int, default=None, help="Override the agent vision range."
    )
    parser.add_argument(
        "--num-environments",
        type=int,
        default=None,
        help="Override the number of generated environments for train or simulate mode.",
    )
    parser.add_argument(
        "--num-curricula",
        type=int,
        default=None,
        help="Override the number of curriculum buckets for train or simulate mode.",
    )
    parser.add_argument(
        "--min-episodes-per-curriculum",
        type=int,
        default=2,
        help="Curriculum pacing used in train mode.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless world generation for play or simulate mode.",
    )
    return parser


def _default_game_manager_config(mode: str, args) -> GameManagerConfig:
    if mode == "train":
        return GameManagerConfig(
            num_tiles=args.num_tiles or 5,
            screen_size=args.screen_size or 20,
            vision_range=args.vision_range or 1,
            headless=True,
        )

    return GameManagerConfig(
        num_tiles=args.num_tiles or 50,
        screen_size=args.screen_size or 800,
        vision_range=args.vision_range or 2,
        headless=args.headless,
    )


def _default_simulation_manager_config(mode: str, args) -> SimulationManagerConfig:
    if mode == "train":
        return SimulationManagerConfig(
            number_of_environments=args.num_environments or 32,
            number_of_curricula=args.num_curricula or 4,
            min_episodes_per_curriculum=args.min_episodes_per_curriculum,
        )

    return SimulationManagerConfig(
        number_of_environments=args.num_environments or 2,
        number_of_curricula=args.num_curricula or 3,
        min_episodes_per_curriculum=1,
    )


def _demo_algorithm_hyperparameters(
    algorithm: AlgorithmName,
) -> dict[str, int | float | bool | str]:
    hyperparameters = default_algorithm_hyperparameters(algorithm)

    if algorithm == AlgorithmName.PPO:
        hyperparameters.update({"n_steps": 64, "batch_size": 32, "learning_rate": 3e-4})
    elif algorithm == AlgorithmName.DQN:
        hyperparameters.update(
            {
                "buffer_size": 1_024,
                "learning_starts": 32,
                "batch_size": 32,
                "train_freq": 1,
            }
        )

    return hyperparameters


def _build_training_config(args) -> TrainingConfig:
    algorithm = AlgorithmName.from_value(args.algorithm)
    return TrainingConfig(
        game_manager=_default_game_manager_config("train", args),
        simulation_manager=_default_simulation_manager_config("train", args),
        model_args=ModelArgs(num_actions=11),
        algorithm=AlgorithmConfig(
            algorithm=algorithm,
            hyperparameters=_demo_algorithm_hyperparameters(algorithm),
        ),
        curriculum=CurriculumConfig(
            min_episodes_per_curriculum=args.min_episodes_per_curriculum,
            performance_threshold=0.85,
        ),
        total_timesteps=args.timesteps,
        kg_completeness=args.kg_completeness,
        seeds=[args.seed],
    )


def _create_results_directory(prefix: str) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{prefix}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
    return results_dir


def _run_training_mode(args) -> None:
    config = _build_training_config(args)
    results_dir = _create_results_directory("manual")
    experiment_name = f"manual_{config.algorithm.algorithm.value.lower()}"

    trainer = Trainer(
        config.kg_completeness,
        results_dir=results_dir,
    )
    trainer.setup(config, seed=args.seed)
    trainer.env_manager.set_kg_completeness(trainer.env, config.kg_completeness)
    trainer.env_manager.set_kg_completeness(trainer.eval_env, config.kg_completeness)
    trainer.run(experiment_name)


def _run_play_mode(args) -> None:
    game_manager = GameManager(config=_default_game_manager_config("play", args))
    game_manager.run()


def _run_simulation_mode(args) -> None:
    game_manager_config = _default_game_manager_config("simulate", args)
    simulation_manager_config = _default_simulation_manager_config("simulate", args)
    simulation_manager = SimulationManager(
        game_manager_config,
        sim_config=simulation_manager_config,
    )

    game_world_array = np.zeros(
        (
            simulation_manager_config.number_of_environments,
            2,
            game_manager_config.num_tiles,
            game_manager_config.num_tiles,
        ),
        dtype=np.uint8,
    )

    for i, game_manager in enumerate(simulation_manager.game_managers):
        logger.info(
            f"Terrain index grid for environment {i}:\n"
            f"{game_manager.environment.terrain_index_grid}"
        )
        logger.info(
            f"Entity index grid for environment {i}:\n"
            f"{game_manager.environment.entity_index_grid}"
        )
        game_world_array[i, 0] = game_manager.environment.terrain_index_grid
        game_world_array[i, 1] = game_manager.environment.entity_index_grid

    np.save("game_world.npy", game_world_array)
    logger.info(f"Combined game world array shape: {game_world_array.shape}")
    logger.info(f"Unique terrain ids: {np.unique(game_world_array[:, 0]).tolist()}")

    terrain_dict = {
        0: "Deep Water",
        1: "Water",
        2: "Grass",
        3: "Hill",
        4: "Mountain",
        5: "Snow",
    }
    entity_dict = {
        0: "None",
        1: "Fish",
        2: "Tree",
        3: "Mossy Rock",
        4: "Snowy Rock",
        5: "Outpost",
        6: "Wood Path",
        7: "Player",
    }

    with open("terrain_dict.json", "w", encoding="utf-8") as terrain_file:
        json.dump(terrain_dict, terrain_file)
    with open("entity_dict.json", "w", encoding="utf-8") as entity_file:
        json.dump(entity_dict, entity_file)


def main(argv: list[str] | None = None) -> int:
    configure_logging(log_dir="logs", level="INFO")
    args = build_parser().parse_args(argv)

    if args.mode == "train":
        _run_training_mode(args)
    elif args.mode == "simulate":
        _run_simulation_mode(args)
    else:
        _run_play_mode(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
