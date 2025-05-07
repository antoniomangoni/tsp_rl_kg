import os
import json
import traceback
from datetime import datetime
from typing import Literal
from tsp_rl_kg.rl.training.trainer import Trainer
from tsp_rl_kg.utils.logger import Logger
from tsp_rl_kg.rl.training.environment_manager import EnvironmentManager
from tsp_rl_kg.rl.simulation_manager import SimulationManager
from tsp_rl_kg.rl.training.trainer import Trainer

class AblationStudy:
    def __init__(self, base_config, kg_completeness_values, converter, logger):
        self.base_config = base_config
        self.converter = converter
        self.kg_completeness_values = kg_completeness_values
        self.logger = logger
        self.results = {}
        self.results_dir = self._create_results_directory()

    def _create_results_directory(self):
        # Create a 'results' folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Create a subfolder with the current datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join('results', current_time)
        os.makedirs(result_dir)
        
        self.logger.info(f"Created results directory: {result_dir}")
        return result_dir

    def run(self):
        self.logger.info("Starting Ablation Study")
        for kg_completeness in self.kg_completeness_values:
            experiment_name = f"kg_completeness_{kg_completeness}"
            self.logger.info(f"Running experiment: {experiment_name}")
            
            try:
                trainer = Trainer(kg_completeness, ablation_study=self)
                trainer.setup(self.base_config)
                trainer.env_manager.set_kg_completeness(trainer.env, kg_completeness)
                trainer.env_manager.set_kg_completeness(trainer.eval_env, kg_completeness)
                
                result = trainer.run(experiment_name)
                
                self.results[experiment_name] = result
                
                self.logger.info(f"Experiment {experiment_name} completed")
            except Exception as e:
                self.logger.error(f"An error occurred during experiment {experiment_name}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        self._save_results()
        self.logger.info("Ablation Study completed")

    def _save_results(self):
        results_file = os.path.join(self.results_dir, 'ablation_study_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        self.logger.info(f"Ablation study results saved to {results_file}")

        # Save individual experiment results
        for experiment_name, result in self.results.items():
            experiment_file = os.path.join(self.results_dir, f"{experiment_name}_results.json")
            with open(experiment_file, 'w') as f:
                json.dump(result, f, indent=4)
            self.logger.info(f"Individual experiment results saved to {experiment_file}")

        # Save the base configuration
        config_file = os.path.join(self.results_dir, 'base_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.base_config, f, indent=4)
        self.logger.info(f"Base configuration saved to {config_file}")
