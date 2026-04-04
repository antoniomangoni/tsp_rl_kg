# `tsp_rl_kg.rl`

## Purpose / ownership
Owns RL-facing runtime components: Gymnasium environment wrapper, policy feature extractor model, simulation orchestration across worlds, reward shaping, and target routing helpers.

## Main identifiers
- `CustomEnv` (`custom_env.py`): primary Gymnasium environment.
- `SimulationManager` (`simulation_manager.py`): multi-world pool + curriculum sequencing.
- `AgentModel` (`agent_model.py`): SB3 feature extractor binding graph/vision encoders.
- `VisionEncoder`, `GraphEncoder`, `HybridEncoder` (`encoders.py`).
- `RewardCalculator` + `manhattan_distance` (`reward.py`).
- `Target_Manager` (`target.py`).

## Inputs / outputs and neighboring package interactions
- **Inputs:** training/runtime configs, game manager instances, encoded graph/vision observations, and actions from learning algorithms.
- **Outputs:** Gym step/reset tuples (`obs, reward, terminated, truncated, info` pattern), metrics-relevant signals, and model feature vectors.
- **Neighbor interactions:**
  - Builds on `game_world` for environment dynamics.
  - Consumes `knowledge`/`graph` outputs through observation encoders.
  - Delegates training orchestration to `rl.training` package.

## Extension points
- Add alternative reward components/ablation flags.
- Add new policy encoders/architectures in `encoders.py`.
- Add new simulation scheduling strategies in `SimulationManager`.

## Cross-links
- [Training module](./training/README.md)
- [Observation module](../observation/README.md)
- [Game world module](../game_world/README.md)

## Tests
- `tests/test_custom_env.py`
- `tests/test_reward.py`
- `tests/test_agent_model.py`
- `tests/test_state_sync.py`
