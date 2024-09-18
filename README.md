# Knowledge Graph-Enhanced Reinforcement Learning for NPC Decision Making

## Master Thesis Overview

This master thesis explores the integration of Knowledge Graphs (KGs) with Reinforcement Learning (RL) to enhance the decision-making capabilities of Non-Player Characters (NPCs) in video games. It features a custom-built game environment simulating a Travelling Salesman Problem (TSP) with procedurally generated terrain and resources.

## Key Features

- Custom game environment with procedurally generated terrain
- Dynamic Knowledge Graph integration
- Hybrid CNN-GAT model for processing visual and graph-based inputs
- Proximal Policy Optimization (PPO) algorithm for agent training
- Ablation study across different KG completeness levels

## Project Structure

Flow diagrams can be found in the `mermaid_diagrams` folder:

- `flow.md`: Overall project structure
- `environment.md`: Custom game environment creation via Perlin noise
- `game_manager.md`: Game world creation and target energy route calculation
- `pipeline.md`: Data flow into the agent model
- `agent_model.md`: Structure of the agent model
- `reward.md`: Decision flow of the reward structure

## Running the Project

### Non-RL Version

To run the project without Reinforcement Learning:

1. Navigate to `main.py`
2. Choose between:
   - Running one game world via a game manager
   - Creating a simulation manager to generate multiple game managers sorted by route energy requirements

### RL Version

To run the Reinforcement Learning training:

1. Navigate to `training.py`
2. Adjust the configuration parameters at the end of the script:

```python
min_episodes_per_curriculum = 4
base_config = {
    'model_args': {'num_actions': 11},
    'simulation_manager_args': {
        'number_of_environments': 3000,
        'number_of_curricula': 30,
        'min_episodes_per_curriculum': min_episodes_per_curriculum},
    'game_manager_args': {'num_tiles': 5, 'screen_size': 20, 'vision_range': 1},
    'model_config': {
        'n_steps': 2048 * 2,
        'batch_size': 512,
        'learning_rate': 6e-4,
        'gamma': 0.995
    },
    'curriculum_config': {
    'min_episodes_per_curriculum': min_episodes_per_curriculum,
    'performance_threshold': 0.85,
    },
    'total_timesteps': 100000
}
kg_completeness_values = [0.25, 0.5, 0.75, 1.0]
```

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

There is a `requirements.txt` (which does not include plotly used in `test.py`) and an `environment_droplet.yml` for pip and conda environments respectively. 

## Contact

Antonio Mangoni: antoniomangoni@gmail.com