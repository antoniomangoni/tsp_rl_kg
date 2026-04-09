from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from click.exceptions import ClickException
from click.exceptions import Exit as ClickExit
from loguru import logger

from tsp_rl_kg.config import (
    AblationConfig,
    AlgorithmConfig,
    AlgorithmName,
    CurriculumConfig,
    GameManagerConfig,
    ModelArgs,
    RewardComponent,
    SimulationManagerConfig,
    TrainingConfig,
    default_algorithm_hyperparameters,
)
from tsp_rl_kg.rl.training.ablation_study import AblationStudy
from tsp_rl_kg.utils.config_files import (
    find_list_section,
    find_mapping_section,
    load_config_file,
    merge_nested_dicts,
)
from tsp_rl_kg.utils.logger import configure_logging

app = typer.Typer(add_completion=False, invoke_without_command=True)

# Uncomment for windows
# os.environ['PYGAME_DETECT_AVX2'] = '1'
MIN_EPISODES_PER_CURRICULUM = 4
DEFAULT_KG_COMPLETENESS_VALUES = [0.25, 0.5, 0.75, 1.0]


def build_base_config(
    *,
    algorithm: AlgorithmName | str = AlgorithmName.PPO,
    algorithm_hyperparameters: dict[str, int | float | bool | str] | None = None,
    total_timesteps: int = 100_000,
    seeds: list[int] | None = None,
    number_of_environments: int = 3_000,
    number_of_curricula: int = 30,
) -> TrainingConfig:
    algorithm = AlgorithmName.from_value(algorithm)
    if algorithm_hyperparameters is None:
        algorithm_hyperparameters = default_algorithm_hyperparameters(algorithm)

    return TrainingConfig(
        game_manager=GameManagerConfig(num_tiles=5, screen_size=20, vision_range=1, headless=True),
        simulation_manager=SimulationManagerConfig(
            number_of_environments=number_of_environments,
            number_of_curricula=number_of_curricula,
            min_episodes_per_curriculum=MIN_EPISODES_PER_CURRICULUM,
        ),
        model_args=ModelArgs(num_actions=11),
        algorithm=AlgorithmConfig(
            algorithm=algorithm,
            hyperparameters=algorithm_hyperparameters,
        ),
        curriculum=CurriculumConfig(
            min_episodes_per_curriculum=MIN_EPISODES_PER_CURRICULUM,
            performance_threshold=0.85,
        ),
        total_timesteps=total_timesteps,
        seeds=seeds or [42, 123, 456],
    )


def _load_study_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None

    try:
        loaded_config = load_config_file(config_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    return (
        find_mapping_section(
            loaded_config,
            ("ablation",),
            ("study",),
            ("run",),
        )
        or loaded_config
    )


def _create_ablation_study_from_external_config(config_path: Path) -> dict[str, Any]:
    study_config = _load_study_config(config_path)
    if study_config is None:
        return {}

    default_base_config = build_base_config()
    base_config_data = (
        find_mapping_section(
            study_config,
            ("base_config",),
            ("training",),
            ("training_config",),
        )
        or study_config
    )
    base_config = TrainingConfig.from_dict(
        merge_nested_dicts(default_base_config.to_dict(), base_config_data)
    )

    study_kwargs: dict[str, Any] = {"base_config": base_config}

    kg_completeness_values = find_list_section(
        study_config,
        ("kg_completeness_values",),
        ("kg_values",),
    )
    if kg_completeness_values is not None:
        study_kwargs["kg_completeness_values"] = kg_completeness_values

    experiments = find_list_section(study_config, ("experiments",))
    if experiments is not None:
        study_kwargs["experiments"] = experiments

    mlflow_experiment_name = study_config.get("mlflow_experiment_name")
    if mlflow_experiment_name is not None:
        study_kwargs["mlflow_experiment_name"] = mlflow_experiment_name

    mlflow_tracking_uri = study_config.get("mlflow_tracking_uri")
    if mlflow_tracking_uri is not None:
        study_kwargs["mlflow_tracking_uri"] = mlflow_tracking_uri

    return study_kwargs


def build_default_experiments(
    kg_completeness_values: list[float] | None = None,
) -> list[dict]:
    kg_completeness_values = kg_completeness_values or DEFAULT_KG_COMPLETENESS_VALUES

    return [
        *[
            {"name": f"kg_{kg}", "kg_completeness": kg, "ablation": AblationConfig()}
            for kg in kg_completeness_values
        ],
        {
            "name": "dqn_baseline",
            "kg_completeness": 0.5,
            "algorithm": {
                "algorithm": AlgorithmName.DQN.value,
                "hyperparameters": {
                    **default_algorithm_hyperparameters(AlgorithmName.DQN),
                    "gamma": 0.995,
                },
            },
        },
        {
            "name": "vision_only",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_graph=True),
        },
        {
            "name": "graph_only",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_vision=True),
        },
        {
            "name": "no_curriculum",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_curriculum=True),
        },
        {
            "name": "no_proximity",
            "kg_completeness": 0.5,
            "ablation": AblationConfig(disable_reward_components=[RewardComponent.PROXIMITY]),
        },
    ]


def create_default_ablation_study(
    *,
    base_config: TrainingConfig | None = None,
    kg_completeness_values: list[float] | None = None,
    experiments: list[dict] | None = None,
    config_path: str | Path | None = None,
    mlflow_experiment_name: str = "tsp_rl_kg_ablation",
    mlflow_tracking_uri: str | None = None,
) -> AblationStudy:
    loaded_kwargs = {}
    if config_path is not None:
        loaded_kwargs = _create_ablation_study_from_external_config(Path(config_path))

    if base_config is None:
        base_config = loaded_kwargs.get("base_config")
    if kg_completeness_values is None:
        kg_completeness_values = loaded_kwargs.get("kg_completeness_values")
    if base_config is None:
        base_config = build_base_config()
    if kg_completeness_values is None:
        kg_completeness_values = list(DEFAULT_KG_COMPLETENESS_VALUES)
    if experiments is None:
        experiments = loaded_kwargs.get("experiments")
    if experiments is None:
        experiments = build_default_experiments(kg_completeness_values)

    mlflow_experiment_name = loaded_kwargs.get(
        "mlflow_experiment_name",
        mlflow_experiment_name,
    )
    mlflow_tracking_uri = loaded_kwargs.get("mlflow_tracking_uri", mlflow_tracking_uri)

    return AblationStudy(
        base_config,
        kg_completeness_values,
        experiments=experiments,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def run_ablation_study(
    *,
    base_config: TrainingConfig | None = None,
    kg_completeness_values: list[float] | None = None,
    experiments: list[dict] | None = None,
    config_path: str | Path | None = None,
    mlflow_experiment_name: str = "tsp_rl_kg_ablation",
    mlflow_tracking_uri: str | None = None,
) -> AblationStudy:
    configure_logging(log_dir="logs", level="INFO")
    ablation_study = create_default_ablation_study(
        base_config=base_config,
        kg_completeness_values=kg_completeness_values,
        experiments=experiments,
        config_path=config_path,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    ablation_study.run()
    return ablation_study


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Load a full ablation-study config from a JSON or TOML file.",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    configure_logging(log_dir="logs", level=log_level.upper())
    if ctx.invoked_subcommand is None:
        try:
            run_ablation_study(config_path=config)
        except Exception as exc:
            logger.error(f"An error occurred during the ablation study: {str(exc)}")
            logger.error(traceback.format_exc())
            raise typer.Exit(code=1) from exc


def _run_app(argv: list[str] | None = None) -> int:
    prog_name = (
        "tsp-rl-kg-study" if argv is not None else (Path(sys.argv[0]).name or "tsp-rl-kg-study")
    )
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
