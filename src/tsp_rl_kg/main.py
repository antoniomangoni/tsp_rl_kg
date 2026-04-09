from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from click.core import ParameterSource
from click.exceptions import ClickException
from click.exceptions import Exit as ClickExit
from loguru import logger

from tsp_rl_kg.config import (
    AlgorithmConfig,
    AlgorithmName,
    CurriculumConfig,
    EpisodeConfig,
    GameManagerConfig,
    ModelArgs,
    SimulationManagerConfig,
    TrainingConfig,
    default_algorithm_hyperparameters,
)
from tsp_rl_kg.game_world.game_manager import GameManager
from tsp_rl_kg.rl.simulation_manager import SimulationManager
from tsp_rl_kg.rl.training.trainer import Trainer
from tsp_rl_kg.utils.config_files import find_mapping_section, load_config_file, merge_nested_dicts
from tsp_rl_kg.utils.logger import configure_logging

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _load_cli_config(
    config_path: Path | None,
    *candidate_sections: tuple[str, ...],
) -> dict[str, Any] | None:
    if config_path is None:
        return None

    try:
        loaded_config = load_config_file(config_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    return find_mapping_section(loaded_config, *candidate_sections) or loaded_config


def _filter_config_fields(data: dict[str, Any], field_names: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in field_names}


def _cli_override_if_explicit(
    ctx: typer.Context,
    parameter_name: str,
    value: Any,
) -> Any | None:
    if ctx.get_parameter_source(parameter_name) is ParameterSource.COMMANDLINE:
        return value
    return None


def _default_game_manager_config(
    mode: str,
    *,
    num_tiles: int | None = None,
    screen_size: int | None = None,
    vision_range: int | None = None,
    headless: bool | None = None,
    max_steps: int | None = None,
) -> GameManagerConfig:
    if mode == "play":
        return GameManagerConfig(
            num_tiles=num_tiles or 50,
            screen_size=screen_size or 800,
            vision_range=vision_range or 2,
            headless=False if headless is None else headless,
            human_mode=True,
            use_random_human_actions=False,
            max_steps=max_steps,
        )

    if mode == "train":
        return GameManagerConfig(
            num_tiles=num_tiles or 5,
            screen_size=screen_size or 20,
            vision_range=vision_range or 1,
            headless=True if headless is None else headless,
            max_steps=max_steps,
        )

    if mode == "simulate":
        return GameManagerConfig(
            num_tiles=num_tiles or 50,
            screen_size=screen_size or 800,
            vision_range=vision_range or 2,
            headless=False if headless is None else headless,
            max_steps=max_steps,
        )

    raise ValueError(f"Unsupported game manager mode: {mode}")


def _default_simulation_manager_config(
    mode: str,
    *,
    num_environments: int | None = None,
    num_curricula: int | None = None,
    min_episodes_per_curriculum: int | None = None,
) -> SimulationManagerConfig:
    if mode == "train":
        return SimulationManagerConfig(
            number_of_environments=num_environments or 32,
            number_of_curricula=num_curricula or 4,
            min_episodes_per_curriculum=min_episodes_per_curriculum or 2,
        )

    return SimulationManagerConfig(
        number_of_environments=num_environments or 2,
        number_of_curricula=num_curricula or 3,
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


def _default_training_config() -> TrainingConfig:
    return TrainingConfig(
        game_manager=GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        simulation_manager=SimulationManagerConfig(
            number_of_environments=32,
            number_of_curricula=4,
            min_episodes_per_curriculum=2,
        ),
        model_args=ModelArgs(num_actions=11),
        algorithm=AlgorithmConfig(
            algorithm=AlgorithmName.PPO,
            hyperparameters=_demo_algorithm_hyperparameters(AlgorithmName.PPO),
        ),
        curriculum=CurriculumConfig(
            min_episodes_per_curriculum=2,
            performance_threshold=0.85,
        ),
        episode=EpisodeConfig(max_episode_steps=128, max_steps_without_progress=64),
        total_timesteps=512,
        kg_completeness=0.5,
        seeds=[42],
    )


def _build_game_manager_config(
    mode: str,
    loaded_config: dict[str, Any] | None = None,
    *,
    num_tiles: int | None = None,
    screen_size: int | None = None,
    vision_range: int | None = None,
    headless: bool | None = None,
    human_mode: bool | None = None,
    max_steps: int | None = None,
) -> GameManagerConfig:
    default_config = asdict(
        _default_game_manager_config(
            mode,
            num_tiles=num_tiles,
            screen_size=screen_size,
            vision_range=vision_range,
            headless=headless,
            max_steps=max_steps,
        )
    )
    config_data = {}
    if loaded_config is not None:
        raw_config = (
            find_mapping_section(
                loaded_config,
                ("game_manager",),
                ("game_manager_args",),
            )
            or loaded_config
        )
        config_data = _filter_config_fields(raw_config, set(GameManagerConfig.__dataclass_fields__))

    merged_config = merge_nested_dicts(default_config, config_data)
    if num_tiles is not None:
        merged_config["num_tiles"] = num_tiles
    if screen_size is not None:
        merged_config["screen_size"] = screen_size
    if vision_range is not None:
        merged_config["vision_range"] = vision_range
    if headless is not None:
        merged_config["headless"] = headless
    if human_mode is not None:
        merged_config["human_mode"] = human_mode
        merged_config["use_random_human_actions"] = False
    if max_steps is not None:
        merged_config["max_steps"] = max_steps

    return GameManagerConfig(**merged_config)


def _build_simulation_manager_config(
    mode: str,
    loaded_config: dict[str, Any] | None = None,
    *,
    num_environments: int | None = None,
    num_curricula: int | None = None,
    min_episodes_per_curriculum: int | None = None,
) -> SimulationManagerConfig:
    default_config = asdict(
        _default_simulation_manager_config(
            mode,
            min_episodes_per_curriculum=min_episodes_per_curriculum,
        )
    )
    config_data = {}
    if loaded_config is not None:
        raw_config = (
            find_mapping_section(
                loaded_config,
                ("simulation_manager",),
                ("simulation_manager_args",),
            )
            or loaded_config
        )
        config_data = _filter_config_fields(
            raw_config,
            set(SimulationManagerConfig.__dataclass_fields__),
        )

    merged_config = merge_nested_dicts(default_config, config_data)
    if num_environments is not None:
        merged_config["number_of_environments"] = num_environments
    if num_curricula is not None:
        merged_config["number_of_curricula"] = num_curricula
    if min_episodes_per_curriculum is not None:
        merged_config["min_episodes_per_curriculum"] = min_episodes_per_curriculum

    return SimulationManagerConfig(**merged_config)


def _build_training_config(
    loaded_config: dict[str, Any] | None = None,
    *,
    algorithm: AlgorithmName | None = None,
    timesteps: int | None = None,
    seed: int | None = None,
    kg_completeness: float | None = None,
    num_tiles: int | None = None,
    screen_size: int | None = None,
    vision_range: int | None = None,
    num_environments: int | None = None,
    num_curricula: int | None = None,
    min_episodes_per_curriculum: int | None = None,
) -> TrainingConfig:
    default_config = _default_training_config()
    if loaded_config is not None:
        config = TrainingConfig.from_dict(
            merge_nested_dicts(default_config.to_dict(), loaded_config)
        )
    else:
        config = default_config

    if algorithm is not None:
        if algorithm == config.algorithm.algorithm:
            hyperparameters = dict(config.algorithm.hyperparameters)
        else:
            hyperparameters = _demo_algorithm_hyperparameters(algorithm)
        config.algorithm = AlgorithmConfig(
            backend=config.algorithm.backend,
            algorithm=algorithm,
            policy_name=config.algorithm.policy_name,
            verbose=config.algorithm.verbose,
            tensorboard_run_name=config.algorithm.tensorboard_run_name,
            hyperparameters=hyperparameters,
        )

    if timesteps is not None:
        config.total_timesteps = timesteps
    if seed is not None:
        config.seeds = [seed]
    if kg_completeness is not None:
        config.kg_completeness = kg_completeness

    if num_tiles is not None:
        config.game_manager.num_tiles = num_tiles
    if screen_size is not None:
        config.game_manager.screen_size = screen_size
    if vision_range is not None:
        config.game_manager.vision_range = vision_range
    if num_environments is not None:
        config.simulation_manager.number_of_environments = num_environments
    if num_curricula is not None:
        config.simulation_manager.number_of_curricula = num_curricula
    if min_episodes_per_curriculum is not None:
        config.simulation_manager.min_episodes_per_curriculum = min_episodes_per_curriculum
        config.curriculum.min_episodes_per_curriculum = min_episodes_per_curriculum

    return config


def _create_results_directory(prefix: str) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{prefix}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
    return results_dir


def _run_training_mode(config: TrainingConfig) -> dict[str, Any]:
    results_dir = _create_results_directory("manual")
    experiment_name = f"manual_{config.algorithm.algorithm.value.lower()}"
    seed = config.seeds[0] if config.seeds else None

    trainer = Trainer(
        config.kg_completeness,
        results_dir=results_dir,
    )
    trainer.setup(config, seed=seed)
    trainer.env_manager.set_kg_completeness(trainer.env, config.kg_completeness)
    trainer.env_manager.set_kg_completeness(trainer.eval_env, config.kg_completeness)
    return trainer.run(experiment_name)


def _write_benchmark_summary(
    *,
    benchmark_result: dict[str, Any],
    config: TrainingConfig,
    benchmark_name: str = "vision_only",
) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_file = os.path.join("results", f"benchmark_{timestamp}.json")
    summary = {
        "benchmark_name": benchmark_name,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "algorithm": config.algorithm.algorithm.value,
        "total_timesteps": config.total_timesteps,
        "seed": config.seeds[0] if config.seeds else None,
        "kg_completeness": config.kg_completeness,
        "ablation": {
            "disable_vision": config.ablation.disable_vision,
            "disable_graph": config.ablation.disable_graph,
            "disable_curriculum": config.ablation.disable_curriculum,
            "disable_reward_components": [c.value for c in config.ablation.disable_reward_components],
        },
        "metrics": {
            "mean_reward": float(benchmark_result["mean_reward"]),
            "std_reward": float(benchmark_result["std_reward"]),
        },
        "artifacts": {
            "metrics_file": benchmark_result["metrics_file"],
            "model_path": benchmark_result["model_path"],
            "stats_file": benchmark_result["stats_file"],
        },
    }
    with open(benchmark_file, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    logger.info(f"Benchmark summary written to {benchmark_file}")
    return benchmark_file


def _run_benchmark_mode(config: TrainingConfig) -> str:
    benchmark_config = TrainingConfig.from_dict(config.to_dict())
    benchmark_config.ablation.disable_graph = True
    benchmark_config.ablation.disable_vision = False
    benchmark_result = _run_training_mode(benchmark_config)
    return _write_benchmark_summary(
        benchmark_result=benchmark_result,
        config=benchmark_config,
        benchmark_name="vision_only",
    )


def _validate_play_config(config: GameManagerConfig) -> None:
    if config.headless and config.human_mode:
        raise typer.BadParameter(
            "Keyboard-controlled play requires a visible window. "
            "Use --no-headless for manual play or --random-actions for autoplay.",
            param_hint="--headless",
        )


def _run_play_mode(config: GameManagerConfig) -> None:
    game_manager = GameManager(config=config)
    game_manager.run()


def _run_simulation_mode(
    game_manager_config: GameManagerConfig,
    simulation_manager_config: SimulationManagerConfig,
) -> None:
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


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    configure_logging(log_dir="logs", level=log_level.upper())
    if ctx.invoked_subcommand is None:
        play_config = _build_game_manager_config("play")
        _validate_play_config(play_config)
        _run_play_mode(play_config)


@app.command()
def play(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Load play settings from a JSON or TOML file."),
    ] = None,
    num_tiles: Annotated[
        int | None,
        typer.Option(help="Override the world width and height in tiles."),
    ] = None,
    screen_size: Annotated[
        int | None,
        typer.Option(help="Override the renderer screen size."),
    ] = None,
    vision_range: Annotated[
        int | None,
        typer.Option(help="Override the agent vision range."),
    ] = None,
    headless: Annotated[
        bool | None,
        typer.Option(
            "--headless/--no-headless",
            help="Override headless rendering from config or defaults.",
        ),
    ] = None,
    human_control: Annotated[
        bool,
        typer.Option(
            "--human/--random-actions",
            help="Choose keyboard-controlled play or random autoplay.",
        ),
    ] = True,
    max_steps: Annotated[
        int | None,
        typer.Option(help="Optional max number of play-loop steps before auto-stop."),
    ] = None,
) -> None:
    loaded_config = _load_cli_config(config, ("main", "play"), ("play",), ("base_config",))
    play_config = _build_game_manager_config(
        "play",
        loaded_config,
        num_tiles=num_tiles,
        screen_size=screen_size,
        vision_range=vision_range,
        headless=headless,
        human_mode=_cli_override_if_explicit(ctx, "human_control", human_control),
        max_steps=max_steps,
    )
    _validate_play_config(play_config)
    _run_play_mode(play_config)


@app.command()
def train(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Load training settings from a JSON or TOML file."),
    ] = None,
    algorithm: Annotated[
        AlgorithmName | None,
        typer.Option(help="Training backend algorithm override."),
    ] = None,
    timesteps: Annotated[
        int | None,
        typer.Option(help="Training timesteps override."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(help="Seed override."),
    ] = None,
    kg_completeness: Annotated[
        float | None,
        typer.Option(help="KG completeness override."),
    ] = None,
    num_tiles: Annotated[
        int | None,
        typer.Option(help="Override the world width and height in tiles."),
    ] = None,
    screen_size: Annotated[
        int | None,
        typer.Option(help="Override the renderer screen size."),
    ] = None,
    vision_range: Annotated[
        int | None,
        typer.Option(help="Override the agent vision range."),
    ] = None,
    num_environments: Annotated[
        int | None,
        typer.Option(help="Override the number of generated environments."),
    ] = None,
    num_curricula: Annotated[
        int | None,
        typer.Option(help="Override the number of curriculum buckets."),
    ] = None,
    min_episodes_per_curriculum: Annotated[
        int | None,
        typer.Option(help="Curriculum pacing override."),
    ] = None,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark/--no-benchmark",
            help="Run standardized benchmark workflow (vision-only ablation) and write results/benchmark_*.json.",
        ),
    ] = False,
) -> None:
    loaded_config = _load_cli_config(
        config,
        ("main", "train"),
        ("train",),
        ("training",),
        ("base_config",),
    )
    training_config = _build_training_config(
        loaded_config,
        algorithm=algorithm,
        timesteps=timesteps,
        seed=seed,
        kg_completeness=kg_completeness,
        num_tiles=num_tiles,
        screen_size=screen_size,
        vision_range=vision_range,
        num_environments=num_environments,
        num_curricula=num_curricula,
        min_episodes_per_curriculum=min_episodes_per_curriculum,
    )
    if benchmark:
        _run_benchmark_mode(training_config)
    else:
        _run_training_mode(training_config)


@app.command()
def simulate(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Load simulation settings from a JSON or TOML file."),
    ] = None,
    num_tiles: Annotated[
        int | None,
        typer.Option(help="Override the world width and height in tiles."),
    ] = None,
    screen_size: Annotated[
        int | None,
        typer.Option(help="Override the renderer screen size."),
    ] = None,
    vision_range: Annotated[
        int | None,
        typer.Option(help="Override the agent vision range."),
    ] = None,
    num_environments: Annotated[
        int | None,
        typer.Option(help="Override the number of generated environments."),
    ] = None,
    num_curricula: Annotated[
        int | None,
        typer.Option(help="Override the number of curriculum buckets."),
    ] = None,
    headless: Annotated[
        bool | None,
        typer.Option(
            "--headless/--no-headless",
            help="Override headless rendering from config or defaults.",
        ),
    ] = None,
) -> None:
    loaded_config = _load_cli_config(
        config,
        ("main", "simulate"),
        ("simulate",),
        ("base_config",),
    )
    _run_simulation_mode(
        _build_game_manager_config(
            "simulate",
            loaded_config,
            num_tiles=num_tiles,
            screen_size=screen_size,
            vision_range=vision_range,
            headless=headless,
        ),
        _build_simulation_manager_config(
            "simulate",
            loaded_config,
            num_environments=num_environments,
            num_curricula=num_curricula,
        ),
    )


def _run_app(argv: list[str] | None = None) -> int:
    prog_name = "tsp-rl-kg" if argv is not None else (Path(sys.argv[0]).name or "tsp-rl-kg")
    try:
        result = app(args=argv, prog_name=prog_name, standalone_mode=False)
    except ClickExit as exc:
        return exc.exit_code
    except ClickException as exc:
        exc.show()
        return exc.exit_code

    return result if isinstance(result, int) else 0


def main(argv: list[str] | None = None) -> int:
    return _run_app(argv)


if __name__ == "__main__":
    raise SystemExit(main())
