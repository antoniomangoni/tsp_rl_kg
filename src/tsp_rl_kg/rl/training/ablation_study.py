import copy
import json
import os
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum

import mlflow
import numpy as np
from loguru import logger

from tsp_rl_kg.config import AblationConfig, AlgorithmConfig, TrainingConfig
from tsp_rl_kg.rl.training.trainer import Trainer


class AblationStudy:
    def __init__(
        self,
        base_config: TrainingConfig | dict,
        kg_completeness_values=None,
        feature_encoder=None,
        experiments: list[dict] | None = None,
        mlflow_experiment_name: str = "tsp_rl_kg_ablation",
        mlflow_tracking_uri: str | None = None,
    ):
        if isinstance(base_config, dict):
            base_config = TrainingConfig.from_dict(base_config)
        self.base_config = base_config
        self.feature_encoder = feature_encoder
        self.kg_completeness_values = kg_completeness_values or []
        self.seeds = base_config.seeds
        self.results = {}
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.results_dir = self._create_results_directory()

        if experiments is not None:
            self.experiments = experiments
        else:
            self.experiments = self._build_default_experiments()

    def _build_default_experiments(self) -> list[dict]:
        """Build default experiment list from kg_completeness_values."""
        experiments = []
        for kg in self.kg_completeness_values:
            experiments.append(
                {
                    "name": f"kg_completeness_{kg}",
                    "kg_completeness": kg,
                    "ablation": AblationConfig(),
                }
            )
        return experiments

    def _create_results_directory(self):
        # Create a 'results' folder if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Create a subfolder with the current datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("results", current_time)
        os.makedirs(result_dir, exist_ok=True)

        logger.info(f"Created results directory: {result_dir}")
        return result_dir

    def _configure_mlflow(self) -> None:
        if self.mlflow_tracking_uri is not None:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

    def _normalise_mlflow_value(self, value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return json.dumps(
                [item.value if isinstance(item, Enum) else item for item in value],
                sort_keys=True,
            )
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        return value

    def _log_mlflow_params(self, params: dict) -> None:
        if not mlflow.active_run():
            return
        mlflow.log_params({k: self._normalise_mlflow_value(v) for k, v in params.items()})

    def _log_result_metrics(self, result: dict) -> None:
        if not mlflow.active_run():
            return

        metrics = {
            key: float(value) for key, value in result.items() if isinstance(value, (int, float))
        }
        if metrics:
            mlflow.log_metrics(metrics)

    def _to_plain_value(self, value):
        if is_dataclass(value):
            return self._to_plain_value(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [self._to_plain_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_plain_value(item) for key, item in value.items()}
        return value

    def _merge_nested_dicts(self, base: dict, overrides: dict) -> dict:
        merged = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_nested_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _build_experiment_config(
        self, experiment: dict
    ) -> tuple[TrainingConfig, float, AblationConfig]:
        kg_completeness = experiment.get("kg_completeness", self.base_config.kg_completeness)

        raw_ablation = experiment.get("ablation", AblationConfig())
        if isinstance(raw_ablation, AblationConfig):
            ablation_config = raw_ablation
        else:
            ablation_config = AblationConfig(**raw_ablation)

        config_overrides = self._to_plain_value(experiment.get("config_overrides", {}))

        algorithm_override = experiment.get("algorithm")
        if algorithm_override is not None:
            plain_override = self._to_plain_value(algorithm_override)
            target_backend = plain_override.get("backend", self.base_config.algorithm.backend)
            target_algorithm = plain_override.get("algorithm", self.base_config.algorithm.algorithm)

            if (
                target_backend == self.base_config.algorithm.backend
                and target_algorithm == self.base_config.algorithm.algorithm
            ):
                base_algorithm = self._to_plain_value(self.base_config.algorithm)
            else:
                base_algorithm = self._to_plain_value(
                    AlgorithmConfig(
                        backend=target_backend,
                        algorithm=target_algorithm,
                        policy_name=self.base_config.algorithm.policy_name,
                        verbose=self.base_config.algorithm.verbose,
                        tensorboard_run_name=self.base_config.algorithm.tensorboard_run_name,
                    )
                )

            merged_algorithm = self._merge_nested_dicts(
                base_algorithm,
                plain_override,
            )
            config_overrides = self._merge_nested_dicts(
                config_overrides,
                {"algorithm": merged_algorithm},
            )

        config_overrides = self._merge_nested_dicts(
            config_overrides,
            {"ablation": self._to_plain_value(ablation_config)},
        )

        experiment_config = TrainingConfig.from_dict(
            self._merge_nested_dicts(self.base_config.to_dict(), config_overrides)
        )
        return experiment_config, kg_completeness, ablation_config

    def run(self):
        logger.info("Starting Ablation Study")
        self._configure_mlflow()
        study_run_name = f"ablation_study_{os.path.basename(self.results_dir)}"

        with mlflow.start_run(run_name=study_run_name):
            self._log_mlflow_params(
                {
                    "study.results_dir": self.results_dir,
                    "study.num_experiments": len(self.experiments),
                    "study.num_seeds": len(self.seeds),
                }
            )

            for experiment in self.experiments:
                experiment_name = experiment["name"]
                experiment_config, kg_completeness, ablation_config = self._build_experiment_config(
                    experiment
                )
                logger.info(f"Running experiment: {experiment_name}")

                seed_results = []
                for seed in self.seeds:
                    seed_name = f"{experiment_name}_seed_{seed}"
                    logger.info(f"Running {seed_name}")

                    try:
                        seed_config = copy.deepcopy(experiment_config)

                        with mlflow.start_run(run_name=seed_name, nested=True):
                            self._log_mlflow_params(
                                {
                                    "experiment.name": experiment_name,
                                    "experiment.seed": seed,
                                    "experiment.kg_completeness": kg_completeness,
                                    "experiment.backend": seed_config.algorithm.backend,
                                    "experiment.algorithm": seed_config.algorithm.algorithm,
                                    "experiment.policy_name": seed_config.algorithm.policy_name,
                                    "experiment.algorithm_hyperparameters": (
                                        seed_config.algorithm.hyperparameters
                                    ),
                                    "ablation.disable_vision": ablation_config.disable_vision,
                                    "ablation.disable_graph": ablation_config.disable_graph,
                                    "ablation.disable_curriculum": (
                                        ablation_config.disable_curriculum
                                    ),
                                    "ablation.disable_reward_components": (
                                        ablation_config.disable_reward_components
                                    ),
                                }
                            )
                            mlflow.log_dict(
                                seed_config.to_dict(),
                                f"configs/{seed_name}_config.json",
                            )

                            trainer = Trainer(
                                kg_completeness,
                                results_dir=self.results_dir,
                                feature_encoder=self.feature_encoder,
                            )
                            trainer.setup(seed_config, seed=seed)
                            trainer.env_manager.set_kg_completeness(trainer.env, kg_completeness)
                            trainer.env_manager.set_kg_completeness(
                                trainer.eval_env, kg_completeness
                            )

                            result = trainer.run(seed_name)
                            seed_results.append({"seed": seed, "result": result})
                            self._log_result_metrics(result)

                        logger.info(f"{seed_name} completed")
                    except Exception as e:
                        logger.error(f"An error occurred during {seed_name}: {str(e)}")
                        logger.error(traceback.format_exc())

                self.results[experiment_name] = {
                    "seed_results": seed_results,
                    "aggregated": self._aggregate_seed_results(seed_results),
                }
                logger.info(f"Experiment {experiment_name} completed")

            self._save_results()
        logger.info("Ablation Study completed")

    def _aggregate_seed_results(self, seed_results):
        if not seed_results:
            return {}
        valid_results = [sr["result"] for sr in seed_results if sr["result"] is not None]
        if not valid_results:
            return {}

        aggregated = {}
        all_keys = set()
        for r in valid_results:
            if isinstance(r, dict):
                all_keys.update(r.keys())

        for key in all_keys:
            values = []
            for r in valid_results:
                if isinstance(r, dict) and key in r:
                    val = r[key]
                    if isinstance(val, (int, float)):
                        values.append(val)
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_seeds": len(values),
                }
        return aggregated

    def _save_results(self):
        results_file = os.path.join(self.results_dir, "ablation_study_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Ablation study results saved to {results_file}")

        # Save individual experiment results
        for experiment_name, result in self.results.items():
            experiment_file = os.path.join(self.results_dir, f"{experiment_name}_results.json")
            with open(experiment_file, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Individual experiment results saved to {experiment_file}")

        # Save the base configuration
        config_file = os.path.join(self.results_dir, "base_config.json")
        with open(config_file, "w") as f:
            json.dump(self.base_config.to_dict(), f, indent=4)
        logger.info(f"Base configuration saved to {config_file}")

        if mlflow.active_run():
            mlflow.log_artifacts(self.results_dir, artifact_path="study_outputs")
