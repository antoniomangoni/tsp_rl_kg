# G21 — Graph and observation semantics

## Goals and non-goals

Correct and reproduce world → remembered knowledge → projected graph → padded observation → model encoding. Fresh training only: no legacy checkpoint compatibility. Completeness means initial map knowledge, knowledge remembers last observations, each episode restores the original world, and headless/displayed training share semantic vision.

Retain terrain/entity/player topology, encoding strategies, action IDs, reward formulas and model dimensions. Do not tune learning or redesign the human renderer. Use Python 3.14, uv, NumPy, Gymnasium, PyTorch Geometric and SB3 without adding dependencies.

## Knowledge contract

- Reject nonfinite completeness or values outside [0,1].
- Reveal floor(completeness × tile_count) tiles from an isolated seeded permutation, derived from run seed and stable pristine-world fingerprint including spawn. Reveal terrain and non-player entity together.
- Reuse the permutation across completeness levels to give nested prior subsets. Union prior with sensed tiles; report requested completeness and actual known fraction.
- KnowledgeState owns known_tiles and remembered terrain/entity IDs. Ordinary sensing uses a clipped Chebyshev square of vision_range at reset and after every action. Scout additionally senses the configured multiplied radius.
- Only sensing and successful direct actions refresh remembered values. Unseen world changes remain unseen. Visual discovery never depends on KG prior.
- Unknown tiles contribute no nodes/incident edges to policy graphs. Always include player and its sensed terrain tile. Completeness 1 is full initial knowledge, not permanent omniscience.

## Episode contract

Capture an immutable world template after generation and before play: terrain, non-player entities, spawn and outposts. Reconstruct mutable state at reset, including inventory, energy, discovery and knowledge; never copy live renderers/recorders. Same world/seed gives the same start and prior. Keep curriculum history outside episode state.

Player occupancy is separate from underlying entities. Moving off a path must preserve it in world state, graph memory and vision. Do not encode player as an ordinary entity.

## Graph contract

Build policy-facing features from memory, not unknown live data. Keep stable internal IDs; emit compact local IDs. Use the induced graph over known terrain/entity nodes and player, preserving edge/attribute alignment. Standalone KHopProjection must relabel nodes. Remove completeness-to-hop behavior from training; retain FullGraphProjection as a diagnostic utility.

Re-encode player features through encode_player(); represent location by its terrain edge. Never write coordinates into feature columns. Visualization must obtain coordinates/layers from metadata/index mappings, not encoded values.

## Observation and batch interfaces

Keep vision, node_features, edge_index and edge_attr. Add num_nodes and num_edges: integer arrays of shape (1,), counting valid leading rows/columns. Padding is zero; zero-valued real features remain valid. Reject invalid dimensions, capacities, nonfinite features and out-of-range local indices; never truncate. Require at least one node; zero edges are valid.

Set observation_schema_version=2 and knowledge_semantics="initial_prior_v1" in run metadata and saved config. Model adapters require counts and report a retraining-required error for legacy observations.

Slice real rows per sample, concatenate nodes, offset edges by cumulative real-node counts, concatenate matching attributes, and create memberships only for real nodes. Padding cannot enter GAT, self-loops or pooling. Skip disabled encoder branches and substitute correctly shaped zero vectors.

## Canonical vision

Tiny tiles use at least two pixels per axis so player and entity markers remain distinct. Pure NumPy renderer shared by headless/displayed training: centered RGB terrain window, distinguishable entity markers and player overlay preserving underlying markers. Outside-world pixels are black. Expose locally sensed content only; no clamped viewport exposing unsensed tiles. Scout expansion is clipped to fixed observation window. Normalize channel-first output with matching [0,1] bounds. Keep human sprites/HUD separate.

## Sequence and target files

1. Add regressions for projection indices, cross-sample edges, padding influence and player-feature corruption.
2. Add world-template reconstruction and player occupancy separation.
3. Add knowledge memory, nested priors and sensing/action updates; remove completeness-dependent discovery.
4. Extract knowledge graphs and canonical vision.
5. Add counts, compact batching, validation and version checks.
6. Thread seeds/metadata through environment creation; update fixtures, config comments and graph-flow docs.
7. Run focused and full validation.

Targets: graph/projection.py, graph/constitution.py, knowledge/knowledge_graph.py plus new knowledge/state.py; game_world/environment.py, agent.py, game_manager.py plus new world_template.py; observation/encoder.py plus new semantic_vision.py; rl/custom_env.py, encoders.py, agent_model.py; training/environment_manager.py and trainer.py; config.py; corresponding tests, module READMEs and docs/mermaid_diagrams/kg_observation_flow.md. Add focused modules rather than expanding CustomEnv with all responsibilities.

## Testing and conformance

- Original graph nodes 3/4 compact to 0/1 with aligned attributes.
- Batch sizes 1/2/3 have no cross-sample edges; ordering and standalone embedding invariance hold within tolerance (hybrid eval mode).
- Padding size/content cannot affect embeddings; zero real features, disconnected components and zero-edge graphs work.
- Player movement preserves all three feature encodings.
- Completeness endpoints/nested subsets, deterministic seeds and isolated RNG are tested.
- Hidden mutation is invisible until sensed; all actions refresh only permitted knowledge.
- Reset restores edits/inventory/spawn/discovery/prior with no mutable sharing.
- Paths survive movement in world, memory and vision.
- Entity/player markers are distinct; headless/displayed policy observations agree; completeness cannot alter fixed-trace vision.
- Boundary/small maps retain correct dimensions/visibility.
- Real observations satisfy spaces and pass SB3 preprocessing/model forward/backward with finite outputs/gradients.
- Legacy observations, invalid counts, overflows and invalid edges fail clearly.

Commands: uv run pytest tests/ -v; uv run ruff check .; uv run black --check .; uv run isort --check-only . Run focused tests first. Follow Black 100-column/isort/Ruff rules and use a topic branch and PR.

## Boundaries and rejection criteria

Always preserve artifacts, add behavioral regressions with fixes, update this spec for changed requirements and record validation in the roadmap. No new dependencies or full research sweeps. Do not edit virtualenv code or hide failing tests. New external services or destructive migrations require separate authorization.

Reject completion if samples interact, padding affects embeddings, hidden state leaks, reset retains episode history, completeness alters visual discovery or movement corrupts player features. Shape tests alone are insufficient.

## Validation

Implemented and validated: 271 tests pass, including graph isolation, padding invariance, prior nesting, memory isolation, reset restoration, paths and schema validation. Ruff, Black and isort pass.
