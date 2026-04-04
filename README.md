# Knowledge Graph-Enhanced Reinforcement Learning for NPC Decision Making

## Master Thesis Overview

*Neuro-Symbolic Creation of Non-Playable Characters*

This master thesis explores the integration of Knowledge Graphs (KGs) with Reinforcement Learning (RL) to enhance the decision-making capabilities of Non-Player Characters (NPCs) in video games. It features a custom-built game environment simulating a Travelling Salesman Problem (TSP) with procedurally generated terrain and resources.

[Read the full thesis (PDF)](https://gupea.ub.gu.se/bitstream/handle/2077/86450/CSE%2024-36%20AM.pdf?sequence=1&isAllowed=y)

## Key Features

- Custom game environment with procedurally generated terrain
- Dynamic Knowledge Graph integration
- Hybrid CNN-GAT model for processing visual and graph-based inputs
- Composable RL backend layer with SB3 PPO and DQN support
- Ablation study across different KG completeness levels

## Project Structure

<<<<<<< ours
### `src/tsp_rl_kg/game_world`
Core simulation domain and world state.

- `actions.py` — available in-world action definitions.
- `agent.py` — agent state and behavior in the world.
- `entities.py` — game entity/data structures.
- `environment.py` — environment step/reset coordination.
- `game_manager.py` — world orchestration and episode lifecycle support.
- `heightmap_generator.py` — procedural terrain generation helpers.
- `terrains.py` — terrain types and terrain-specific constants/logic.

### `src/tsp_rl_kg/knowledge`
Knowledge graph construction and index helpers.

- `knowledge_graph.py` — KG representation and update/query routines.
- `graph_idx_manager.py` — index mapping utilities used by KG nodes/edges.

### `src/tsp_rl_kg/graph`
Graph feature processing and schema/constitution helpers.

- `constitution.py` — graph constitution/schema-related definitions.
- `feature_encoder.py` — feature encoding for graph inputs.
- `graph_idx_manager.py` — graph index management utilities.
- `projection.py` — projection/transformation utilities for graph outputs.

### `src/tsp_rl_kg/observation`
Observation encoding boundary between environment and model.

- `encoder.py` — observation encoding and model-ready tensor conversion.

### `src/tsp_rl_kg/rl`
RL-facing environment wrappers, models, and reward logic.

- `agent_model.py` — RL policy/model architecture entrypoint.
- `custom_env.py` — Gymnasium-compatible environment wrapper.
- `encoders.py` — RL encoder components.
- `reward.py` — reward shaping and reward aggregation logic.
- `simulation_manager.py` — simulation coordination for RL runs.
- `target.py` — target/task abstractions used by RL components.

### `src/tsp_rl_kg/rl/training`
Training orchestration, studies, evaluation, and backend adapters.

- `run.py` — ablation-study CLI entrypoint (`tsp-rl-kg-study`).
- `trainer.py` — core training loop orchestration.
- `model_trainer.py` — model training execution utilities.
- `environment_manager.py` — vectorized/single env setup management.
- `callbacks.py` — training callback hooks.
- `curriculum.py` — curriculum scheduling logic.
- `evaluation.py` — evaluation and benchmark routines.
- `metrics.py` — metric calculation/reporting helpers.
- `sequence_sampler.py` — sequence sampling helpers.
- `trajectory_store.py` — trajectory persistence/buffering utilities.
- `ablation_study.py` — ablation experiment coordinator.
- `backends/base.py` — backend interface/contract.
- `backends/sb3.py` — Stable-Baselines3 backend implementation.

### `src/tsp_rl_kg/utils`
Shared utility helpers.

- `config_files.py` — JSON/TOML loading, section extraction, and dict merge helpers.
- `helper_functions.py` — generic utility helpers.
- `logger.py` — logging setup/configuration.

## Diagrams

Mermaid design/flow docs live under `docs/mermaid_diagrams/`:

- [`flow.md`](docs/mermaid_diagrams/flow.md) — overall project flow.
- [`environment.md`](docs/mermaid_diagrams/environment.md) — environment creation flow.
- [`game_manager.md`](docs/mermaid_diagrams/game_manager.md) — game manager/world lifecycle.
- [`pipeline.md`](docs/mermaid_diagrams/pipeline.md) — input pipeline and data flow.
- [`agent_model.md`](docs/mermaid_diagrams/agent_model.md) — model architecture.
- [`reward.md`](docs/mermaid_diagrams/reward.md) — reward decision flow.
- [`kg.md`](docs/mermaid_diagrams/kg.md) — knowledge graph modeling flow.
- [`sim_env.md`](docs/mermaid_diagrams/sim_env.md) — simulation environment behavior.
=======
Architecture diagrams are maintained in `docs/mermaid_diagrams/` and aligned with the current code:

- `system_overview.md`: `main.py` entrypoints and package boundaries.
- `training_backend_protocols.md`: backend contracts in `rl/training/backends/base.py`.
- `training_sb3_backend.md`: SB3 backend internals (`rl/training/backends/sb3.py`).
- `training_orchestration.md`: interaction across `trainer.py`, `environment_manager.py`, `curriculum.py`, `evaluation.py`, and `trajectory_store.py`.
- `kg_observation_flow.md`: knowledge-graph constitution/projection/encoding and observation assembly.
>>>>>>> theirs

## Running the Project

Installed console scripts (from `pyproject.toml`):

- `tsp-rl-kg = tsp_rl_kg.main:main`
- `tsp-rl-kg-study = tsp_rl_kg.rl.training.run:main`

### Non-RL / world commands (`tsp-rl-kg`)

```bash
uv run tsp-rl-kg play
uv run tsp-rl-kg simulate
```

### RL training via main CLI (`tsp-rl-kg`)

```bash
uv run tsp-rl-kg train --algorithm PPO
uv run tsp-rl-kg train --algorithm DQN
uv run tsp-rl-kg train --config configs/train.json
uv run tsp-rl-kg train --config configs/train_namespaced.json
```

### Ablation study CLI (`tsp-rl-kg-study`)

```bash
uv run tsp-rl-kg-study --config configs/ablation.toml
```

## Configuration Shapes (Concise)

Both CLIs accept JSON or TOML.

- `tsp-rl-kg` (`src/tsp_rl_kg/main.py`) loads training config from the first matching mapping among:
  - root object
  - `train`
  - `training`
  - `main.train`
  - `base_config`
- `tsp-rl-kg-study` (`src/tsp_rl_kg/rl/training/run.py`) loads study config from the first matching mapping among:
  - root object
  - `ablation`
  - `study`
  - `run`

Within study config, the base training config can be provided under `base_config` (or `training` / `training_config`).

## Docs Index

- Root documentation:
  - [`README.md`](README.md)
- Module READMEs:
  - _No module-level `README.md` files are currently present under `src/tsp_rl_kg/`._
- Mermaid diagrams:
  - [`docs/mermaid_diagrams/flow.md`](docs/mermaid_diagrams/flow.md)
  - [`docs/mermaid_diagrams/environment.md`](docs/mermaid_diagrams/environment.md)
  - [`docs/mermaid_diagrams/game_manager.md`](docs/mermaid_diagrams/game_manager.md)
  - [`docs/mermaid_diagrams/pipeline.md`](docs/mermaid_diagrams/pipeline.md)
  - [`docs/mermaid_diagrams/agent_model.md`](docs/mermaid_diagrams/agent_model.md)
  - [`docs/mermaid_diagrams/reward.md`](docs/mermaid_diagrams/reward.md)
  - [`docs/mermaid_diagrams/kg.md`](docs/mermaid_diagrams/kg.md)
  - [`docs/mermaid_diagrams/sim_env.md`](docs/mermaid_diagrams/sim_env.md)

Training-layer diagram references:

- [Backend protocols](docs/mermaid_diagrams/training_backend_protocols.md)
- [SB3 backend internals](docs/mermaid_diagrams/training_sb3_backend.md)
- [Training orchestration](docs/mermaid_diagrams/training_orchestration.md)

KG/observation diagram reference:

- [KG and observation flow](docs/mermaid_diagrams/kg_observation_flow.md)

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
