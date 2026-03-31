import copy
import json
import os
import traceback
from datetime import datetime

import numpy as np

from tsp_rl_kg.config import AblationConfig, TrainingConfig
from tsp_rl_kg.rl.training.trainer import Trainer


class AblationStudy:
    def __init__(
        self,
        base_config: TrainingConfig | dict,
        kg_completeness_values=None,
        logger=None,
        feature_encoder=None,
        experiments: list[dict] | None = None,
    ):
        if isinstance(base_config, dict):
            base_config = TrainingConfig.from_dict(base_config)
        self.base_config = base_config
        self.feature_encoder = feature_encoder
        self.kg_completeness_values = kg_completeness_values or []
        self.seeds = base_config.seeds
        self.logger = logger
        self.results = {}
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
        if not os.path.exists("results"):
            os.makedirs("results")

        # Create a subfolder with the current datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("results", current_time)
        os.makedirs(result_dir)

        self.logger.info(f"Created results directory: {result_dir}")
        return result_dir

    def run(self):
        self.logger.info("Starting Ablation Study")
        for experiment in self.experiments:
            experiment_name = experiment["name"]
            kg_completeness = experiment.get("kg_completeness", self.base_config.kg_completeness)
            ablation_config = experiment.get("ablation", AblationConfig())
            self.logger.info(f"Running experiment: {experiment_name}")

            seed_results = []
            for seed in self.seeds:
                seed_name = f"{experiment_name}_seed_{seed}"
                self.logger.info(f"Running {seed_name}")

                try:
                    # Build per-experiment config with ablation overrides
                    experiment_config = copy.deepcopy(self.base_config)
                    experiment_config.ablation = ablation_config

                    trainer = Trainer(kg_completeness, ablation_study=self, logger=self.logger)
                    trainer.setup(experiment_config, seed=seed)
                    trainer.env_manager.set_kg_completeness(trainer.env, kg_completeness)
                    trainer.env_manager.set_kg_completeness(trainer.eval_env, kg_completeness)

                    result = trainer.run(seed_name)
                    seed_results.append({"seed": seed, "result": result})

                    self.logger.info(f"{seed_name} completed")
                except Exception as e:
                    self.logger.error(f"An error occurred during {seed_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())

            self.results[experiment_name] = {
                "seed_results": seed_results,
                "aggregated": self._aggregate_seed_results(seed_results),
            }
            self.logger.info(f"Experiment {experiment_name} completed")

        self._save_results()
        self.logger.info("Ablation Study completed")

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
        self.logger.info(f"Ablation study results saved to {results_file}")

        # Save individual experiment results
        for experiment_name, result in self.results.items():
            experiment_file = os.path.join(self.results_dir, f"{experiment_name}_results.json")
            with open(experiment_file, "w") as f:
                json.dump(result, f, indent=4)
            self.logger.info(f"Individual experiment results saved to {experiment_file}")

        # Save the base configuration
        config_file = os.path.join(self.results_dir, "base_config.json")
        with open(config_file, "w") as f:
            json.dump(self.base_config.to_dict(), f, indent=4)
        self.logger.info(f"Base configuration saved to {config_file}")
