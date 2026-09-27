# G21 — Graph and observation semantics

## Active contract: schema 4, closed-tour task

This supersedes schema 2's RGB policy vision, mean pooling, and repeated open laps.
The intermediate schema-3 compact-vision changes are included in schema 4; neither
schema 2 nor schema 3 checkpoints can be resumed. Fresh training is required.
Knowledge semantics remain `initial_prior_v1`.

## Knowledge and world lifecycle

- Completeness is a seeded initial prior, not continuing omniscience. Reveal nested
  tile subsets, union them with sensed tiles, and remember last-observed terrain and
  entities. Unknown live state never enters graph or query features.
- Prior knowledge never changes visual discovery. Sense once after each action;
  scout uses its expanded sensing radius while policy vision retains a fixed window.
- Keep immutable templates, isolated NumPy generators, complete-world fingerprints,
  and terrain-only fingerprints. Train/validation/test terrains are disjoint.
- Training starts on a seeded outpost, visits every other outpost, and returns to
  that start. Outpost IDs follow the template's randomized placement order, never
  coordinate sorting. The initial outpost counts as visited but earns no arrival bonus.
- EpisodeState owns task progress, inventory-independent visit history, energy,
  completion and no-progress counters. Reward calculation cannot mutate task state.
- Success terminates; step/no-progress limits truncate. Success wins at a limit.
  Preserve terminal observations and records until reset. Restore world changes,
  inventory, knowledge, discovery, counters and energy at every reset.
- The full-world reference is a passable, energy-optimal static closed tour. It
  charges destination cells once per movement and can be beaten by world modification.

## Policy observation

`vision`: `(15, 2r+1, 2r+1)`, channel/y/x. Six terrain classes, seven entity classes
(including empty), player and visible mask. Unknown/outside-world values are zero.
Display size cannot change this tensor. Keep the RGB renderer for humans.

`status`: visited flags, one-hot starting-outpost identity, normalized wood and stone,
remaining-time fraction, return-home flag, normalized no-progress count. This exposes
task bookkeeping without revealing unknown goal locations. Navigation remains partially observed.

`node_features`: semantic encoder output followed by named fields
`relative_x, relative_y, is_player, visited, is_start`. Relative offsets are divided
by `max(1, map_side-1)`. Task flags occur on known outpost entity nodes. Never overwrite
semantic columns; preserve raw-int, one-hot and embedding-lookup encodings.

`edge_attr`: semantic edge encoding followed by signed `dx, dy`; reverse edges negate
these two values. Entity/player attachment edges have zero displacement.

`path_queries`: `(outpost_count, 6)` with columns `known, reachable, visited,
relative_x, relative_y, normalized_energy`. Use remembered passable paths, divide energy
by `12 × map_area`, zero unknown rows, and use a false reachable mask plus zero cost
for disconnected goals. Queries are an explicit ablation; disabling graph also disables queries.

`edge_index`, `num_nodes`, `num_edges`: compact local indices and valid leading counts.
Padding is zero and cannot participate in message passing, self-loops, attention or
validation of real data. Zero-valued semantic features remain valid. Never truncate
capacity overflow or accept invalid counts, indices or nonfinite real features.

## Graph and policy

Keep stable internal IDs and node-to-tile metadata. Policy graphs contain known
terrain nodes, real known entity nodes, and one player; omit empty entity slots.
Use vectorized masks for induced subgraphs and batch construction, relabel local
indices, and offset edges by cumulative real-node counts. No cross-sample edges.

Read all real node embeddings with player-query attention; preserve relative geometry
alongside message-passed features. Disconnected prior regions and distant outposts must
reach the readout. No BatchNorm or dropout in the policy. PPO must not overwrite the
shared feature-extractor initialization.

## Rewards, metrics and experiment isolation

Charge actual action energy, first visits and bounded completion efficiency. Potential
shaping runs on every transition using a separate sensing-only memory, independent of
initial completeness. True terminal potential is zero; truncations retain the successor
potential. Algorithm and shaping discounts agree. No historical improvement bonus,
growing time penalty, circular penalty or environment reward clipping.

Emit exactly one terminal record per episode per worker. Record raw returns, success,
closed-tour efficiency, energy, outposts reached, known fraction, action counts and world
identity. Reward normalization is training-only; pair every checkpoint with saved
normalization and schema metadata. Evaluation statistics remain frozen.

## Conformance gates

Regression coverage must establish executable reference energy, correct closure and
truncation, reset restoration, fixed-trace completeness invariance of task signals,
unknown-state isolation, nested priors, display-independent vision, valid graph
geometry, padding and batch isolation, masked path queries, deterministic train/eval
log probabilities, and PPO/DQN save/reload behavior.

Run unit/integration checks on PRs. The nightly smoke, pilot and graph-only learning
gates provide separate evidence of learning; shape/gradient tests do not certify them.
See [the research review and delivery log](../plans/rl-dl-kg-improvements.md).
