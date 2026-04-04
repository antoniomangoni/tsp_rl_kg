# `tsp_rl_kg.observation`

## Purpose / ownership
Owns observation packaging from simulation state into fixed-shape tensors/spaces expected by RL agents.

## Main identifiers
- `ObservationEncoder` protocol (`encoder.py`): contract for observation-space + encoding behavior.
- `PaddedPyGObservationEncoder` (`encoder.py`): pads PyG graph tensors and vision crops into bounded, Gymnasium-compatible observation dictionaries.

## Inputs / outputs and neighboring package interactions
- **Inputs:** PyG `Data` graph, rendered/local vision tensor, and max node/edge dimensions supplied by environment setup.
- **Outputs:** observation dict (graph + vision components) and corresponding `gymnasium.spaces` definitions.
- **Neighbor interactions:**
  - Used by `rl.custom_env.CustomEnv` to expose model-ready observations.
  - Consumes graph structures from `knowledge`/`graph` and visual state from `game_world`.

## Extension points
- Add new observation encoder variants (e.g., graph-only, vision-only, sequence-based).
- Extend the observation contract while preserving compatibility with `rl.agent_model.AgentModel`.

## Cross-links
- [RL module](../rl/README.md)
- [Knowledge module](../knowledge/README.md)
- [Game world module](../game_world/README.md)

## Tests
- `tests/test_observation_encoder.py`
- `tests/test_custom_env.py`
