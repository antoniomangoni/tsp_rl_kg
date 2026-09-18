# `tsp_rl_kg.observation`

## Purpose / ownership
Owns observation packaging from simulation state into fixed-shape tensors/spaces expected by RL agents.

## Main identifiers
- `ObservationEncoder` protocol (`encoder.py`): contract for observation-space + encoding behavior.
- `PaddedPyGObservationEncoder` (`encoder.py`): pads PyG graph tensors and vision crops into bounded, Gymnasium-compatible observation dictionaries.

## Inputs / outputs and neighboring package interactions
- **Inputs:** PyG `Data` graph, unnormalized channel-first semantic RGB array, and max node/edge dimensions supplied by environment setup.
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

## Observation schema v2

`num_nodes` and `num_edges` are integer arrays of shape `(1,)` identifying valid prefixes.
Only these prefixes enter GAT and pooling; padding is never inferred from feature values.
The pure NumPy semantic renderer supplies normalized channel-first RGB in both display
modes, including terrain, entity and player markers. Tiny tiles use at least two pixels
per axis to keep overlay markers distinct. Old checkpoints require fresh training.

### Observation dictionary

Let `N`/`E` be the configured graph capacities and `F_n`/`F_e` the feature widths.

| Key | Shape | NumPy dtype | Meaning |
| --- | --- | --- | --- |
| `vision` | `(3, H, W)` | `float16` | Semantic RGB normalized to `[0, 1]` |
| `node_features` | `(N, F_n)` | `float16` | Node features, zero-padded after `num_nodes` |
| `edge_attr` | `(E, F_e)` | `float16` | Edge features, zero-padded after `num_edges` |
| `edge_index` | `(2, E)` | `int64` | Local node indices, zero-padded after `num_edges` |
| `num_nodes` | `(1,)` | `int64` | Number of valid nodes, at least one |
| `num_edges` | `(1,)` | `int64` | Number of valid edges, possibly zero |

`encode()` accepts RGB values in `[0, 255]` and normalizes them. It validates graph
capacity, local edge indices, feature dimensions/ranges, and finite input values.
The semantic renderer is `semantic_vision.render_semantic_vision`; Pygame frames
and HUD pixels do not enter the policy observation.
