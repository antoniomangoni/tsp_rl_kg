# Knowledge Graph-Enhanced Reinforcement Learning for NPC Decision Making

## Master Thesis Overview

*Neuro-Symbolic Creation of Non-Playable Characters*

This master thesis explores the integration of Knowledge Graphs (KGs) with Reinforcement Learning (RL) to enhance the decision-making capabilities of Non-Player Characters (NPCs) in video games. It features a custom-built game environment simulating a Travelling Salesman Problem (TSP) with procedurally generated terrain and resources.

[Read the full thesis (PDF)](https://gupea.ub.gu.se/bitstream/handle/2077/86450/CSE%2024-36%20AM.pdf?sequence=1&isAllowed=y)

## Key Features

- Custom game environment with procedurally generated terrain
- Dynamic Knowledge Graph integration
- Hybrid CNN-GAT model for processing visual and graph-based inputs
- Composable RL backend layer with SB3 PPO and DQN support today
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

1. Run a single interactive world:

```bash
uv run tsp-rl-kg play
```

2. Export a small batch of generated worlds:

```bash
uv run tsp-rl-kg simulate
```

### RL Version

To run the Reinforcement Learning training:

1. For a quick config-driven training run through the shared trainer path:

```bash
uv run tsp-rl-kg train --algorithm PPO
uv run tsp-rl-kg train --algorithm DQN
uv run tsp-rl-kg train --config configs/train.json
uv run tsp-rl-kg train --config configs/train_namespaced.json
```

Example config files are included in `configs/train.json`, `configs/train_namespaced.json`, and `configs/ablation.toml`. `configs/train_namespaced.json` shows the `main.train` wrapper shape accepted by the CLI.

2. For the ablation study runner, either use the Typer CLI or run the module file directly:

```bash
uv run tsp-rl-kg-study --config configs/ablation.toml
uv run python src/tsp_rl_kg/rl/training/run.py --config configs/ablation.toml
```

3. The CLIs accept JSON or TOML config files. `tsp-rl-kg train` loads a `TrainingConfig`-shaped mapping from the root, `train`, `training`, `main.train`, or `base_config`. `tsp-rl-kg-study` loads a study spec from the root, `study`, `ablation`, or `run`.

4. The training stack is configured via `TrainingConfig.algorithm`, not just legacy PPO-only `model_config` values. A typical config now looks like:

```python
min_episodes_per_curriculum = 4
base_config = {
    "model_args": {"num_actions": 11},
    "game_manager": {"num_tiles": 5, "screen_size": 20, "vision_range": 1, "headless": True},
    "simulation_manager": {
        "number_of_environments": 3000,
        "number_of_curricula": 30,
        "min_episodes_per_curriculum": min_episodes_per_curriculum,
    },
    "algorithm": {
        "backend": "sb3",
        "algorithm": "PPO",
        "policy_name": "MultiInputPolicy",
        "hyperparameters": {
            "n_steps": 4096,
            "batch_size": 512,
            "learning_rate": 6e-4,
            "gamma": 0.995,
        },
    },
    "curriculum": {
        "min_episodes_per_curriculum": min_episodes_per_curriculum,
        "performance_threshold": 0.85,
    },
    "total_timesteps": 100000,
}
```

The environment, knowledge-graph, and observation contracts stay stable; backend choice and algorithm cadence live in the training layer.

## Contributing

All changes to `main` must go through a pull request. Direct pushes to `main` are not allowed, including for the maintainer.

Before opening a pull request, bootstrap the repository and run the same checks that CI enforces:

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files --show-diff-on-failure
uv run pytest tests/ -v
```

Create a topic branch from `main` for each change using one of these prefixes: `feature/`, `fix/`, `chore/`, `docs/`, `refactor/`, or `test/`.

Maintainer-authored pull requests may merge once all required checks pass. Pull requests authored by anyone else require an approving review from `@antoniomangoni` before merge.

## Contact

Antonio Mangoni: antoniomangoni@gmail.com
