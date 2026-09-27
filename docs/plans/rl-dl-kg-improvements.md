# RL, deep learning and knowledge-graph review

Reviewed against `main` at `5c51100` on 2026-09-27. Scope: the RL environment and reward,
the policy network, the knowledge-graph pipeline, and the training/evaluation protocol.
Rendering, CLI and packaging were only checked where they touch these.

## Verdict

The engineering scaffolding is in good shape: typed and validated configs, reproducible
seeded worlds and nested priors, schema-versioned observations, isolated graph batching
with padding-invariance tests, and a study runner that records failures honestly.

The learning problem built on top of that scaffolding is not yet sound enough to answer
the research question. That question is whether knowledge-graph completeness improves NPC
route finding. Four independent problems each make a null or noisy result likely,
whatever the truth is:

1. **Reward and termination defects** (P0 below). Proximity shaping almost never runs.
   Productive episodes are cut off by a no-progress counter that cannot reset. The
   curriculum's success signal is almost never recorded. The route-quality bonus is
   clipped away.
2. **The reference route is infeasible.** The target-route oracle routes through trees,
   rocks and fish that block movement, so efficiency and gap are not comparable.
3. **The observation is not Markov.** The policy cannot see which outposts it has
   visited or what it is carrying, but the reward depends on both.
4. **The graph branch cannot carry the information the KG exists to provide.** Mean
   pooling over hundreds of direction-free lattice nodes discards where outposts are
   relative to the player.

No stored run has trained long enough to show learning. The `results/20260722_130521`
"sweep" ran 64 timesteps per experiment, and every evaluation episode ended at the
no-progress limit.

Recommended order: fix P0 → make the observation Markov and right-size the network →
redesign the KG readout → run the study with enough seeds, steps and baselines to
interpret it.

## Evidence

Measured with throwaway probe scripts (not committed) on the `configs/ablation_full.toml`
world settings: 12×12 tiles, 480 px screen, vision range 2, 300-step episodes and one-hot
features. Machine: 16 CPU threads and an RTX 3060 Laptop GPU (6 GB).

| Probe | Result |
| --- | --- |
| Oracle target routes that pass through impassable tiles | **144 / 150** worlds |
| Worlds where every outpost is reachable from spawn without clearing obstacles | **9 / 20** |
| Distinct terrain grids in a 150-world training pool | **78 / 150** |
| Held-out eval worlds whose terrain also appears in the training pool | **113 / 150** |
| Steps on which `proximity_reward` is evaluated (heuristic agent) | **199 / 4,946** (4%) |
| Laps completed vs successes recorded for the curriculum (30 episodes) | **77 vs 1** |
| Episodes ended by the no-progress limit (same heuristic agent) | **27 / 30** |
| Share of observation bytes that is the vision image (3×200×200) | **240 KB of 262 KB** (92%) |
| Parameters: vision branch / graph branch / whole policy | **5.85 M / 0.18 M / 7.35 M** |
| PPO ratio outside the [0.8, 1.2] clip range at *identical* weights (train vs eval mode) | **64%** of samples |
| Same, between two train-mode passes (dropout noise alone) | **53%** of samples |
| GPU training memory per sample; configured batch of 64 | ~59 MB; **out of memory** on 6 GB |
| GPU forward, batch 32: vision / GAT / Python graph-batch loop | 32 ms / 5 ms / 9 ms |
| CPU forward+backward, batch 64 | ~8 s |
| Environment throughput, single process | ~358 steps/s, reset ~14 ms |

The ratio probe re-initialised the action head with gain 1 to emulate a barely trained
policy (entropy 2.33 nats against 2.40 for uniform). At SB3's default 0.01 gain the logits
are near zero and the effect is hidden until training moves them.

## P0 — Defects that invalidate current experiments

Each fix should land with a behavioural regression test that goes through
`RewardCalculator.calculate()` or `CustomEnv.step()`. The current reward tests call the
component methods directly, which is how several of these survived.

### 1. Proximity shaping only runs when the agent reaches an outpost

[reward.py:204-246](../../src/tsp_rl_kg/rl/reward.py#L204-L246): the `else:` holding
`proximity_reward` belongs to `if len(outposts_visited) == len(outpost_coords)`, which sits
inside the new-outpost branch. Shaping therefore fires only on arrival at a non-final
outpost, and compares distances between arrivals rather than between steps. In the probe
it was evaluated on 4% of steps.

**Fix.** Evaluate shaping every step. Better, replace it with potential-based shaping,
`F = γ·Φ(s′) − Φ(s)` with `Φ(s) = −d(s, nearest unvisited outpost) / D`. That form
provably leaves the optimal policy unchanged (Ng, Harada & Russell 1999). Use a path
distance over *known passable* tiles rather than Manhattan distance, which ignores
obstacles.

### 2. The no-progress counter can never reset after the first near-approach

[custom_env.py:310-338](../../src/tsp_rl_kg/rl/custom_env.py#L310-L338):
`best_distance_to_unvisited` only ever decreases within an episode. On arrival the outpost
leaves the unvisited set *before* the check, so reaching distance 0 never counts as
progress. Once the agent has stood next to any outpost, `best ≤ 1` and the counter only
grows. After a completed lap `best = 0` and cannot be beaten again.

In the probe, a BFS agent finishing a lap every 10–20 steps was terminated for "no
progress" after 5–6 laps in 3 of 6 episodes, with the counter rising monotonically across
arrivals.

**Fix.** Reset `best_distance_to_unvisited` and `steps_without_progress` whenever
`outposts_visited` changes. Alternatively define progress as "visited a new outpost or
discovered a new tile in the last N steps".

**Related: time-outs should be truncations.** The no-progress cut-off and the
`max_not_improvement_routes` early stop are returned as `terminated=True`. SB3 bootstraps
the value only for truncations (`TimeLimit.truncated`), so these time-outs teach the
critic `V(s) = 0` in states that were not terminal. Return them as `truncated`.

### 3. Success is recorded only if a lap ends on the episode's final step

[custom_env.py:273-278](../../src/tsp_rl_kg/rl/custom_env.py#L273-L278): `success =
all_visited` is true only on the step that completes a lap, and episodes do not end on lap
completion. The heuristic agent completed 77 laps; the curriculum saw 1 success. The
success criterion is therefore almost never met, and advancement falls back to the
50-episode plateau rule.

**Fix.** Count laps per episode, set `success = laps ≥ 1`, and put `success`, `laps` and
lap efficiency into `info`. Log them through `Monitor(info_keywords=...)` so SB3 reports
`rollout/success_rate`.

### 4. The curriculum's performance test compares incommensurable units

[simulation_manager.py:158-166](../../src/tsp_rl_kg/rl/simulation_manager.py#L158-L166):
`avg_performance > performance_threshold * target_route_energy`. The left side is a sum
of rewards clipped to [−1, 1] per step; the right side is in terrain-energy units. In
practice this is unreachable. The hard-coded thresholds at lines 69 and 73 are then
partly overwritten by `Trainer`, so the configured value only takes effect because of a
later mutation.

**Fix.** Advance on success rate plus a normalised efficiency (`target_energy /
agent_energy` over closed laps). Keep the thresholds in `CurriculumConfig` only.

### 5. The target route is not a valid reference

[target.py](../../src/tsp_rl_kg/rl/target.py) has three problems:

- **Obstacles are ignored.** Dijkstra ([target.py:90-115](../../src/tsp_rl_kg/rl/target.py#L90-L115))
  uses terrain energy only. Tiles holding trees, rocks or fish are impassable
  ([terrains.py:31-34](../../src/tsp_rl_kg/game_world/terrains.py#L31-L34)), so 144 of 150
  target routes cannot be walked. Meanwhile 11 of 20 worlds cannot be completed at all
  without clearing obstacles. Efficiency and gap are measured against a relaxation whose
  error varies from world to world.
- **The legs differ.** The oracle is a closed tour o₁→o₂→o₃→o₁
  ([target.py:37](../../src/tsp_rl_kg/rl/target.py#L37)). The agent's first "route" is the
  open path spawn→o₁→o₂→o₃, because energy resets only at episode start and lap completion.
  First-lap efficiency is therefore not comparable. Laps two onward are closed tours and
  are comparable.
- **Tour order uses Manhattan distance, not energy**
  ([target.py:59-75](../../src/tsp_rl_kg/rl/target.py#L59-L75)). This is harmless with 3
  outposts, where every tour is the same cycle, but wrong for more.

The step penalty also uses the energy grid computed once from the pristine world
([custom_env.py:234](../../src/tsp_rl_kg/rl/custom_env.py#L234)). Paths and filled water
never lower it, and non-move actions are charged the tile's terrain cost instead of
`action_energy_cost`. The reward and the energy metric therefore disagree.

**Fix.** Compute the oracle over a graph that models passability. Charge each obstacle
tile its clearing cost plus terrain cost, and compute an energy-optimal closed tour
(3–6 outposts can be enumerated exactly). Compare closed laps only. Derive the step
penalty from the energy actually spent in the step. Describe the oracle as a strong
*baseline*, since path building can legitimately beat it.

### 6. Route quality has almost no explicit reward signal

[reward.py:92-136](../../src/tsp_rl_kg/rl/reward.py#L92-L136) and
[reward.py:261-278](../../src/tsp_rl_kg/rl/reward.py#L261-L278):

- **First lap:** `previous_best_route_energy = inf`, so improvement is `inf`. The reward
  becomes `inf` and is clipped to 1.0, and `improvement = inf` is recorded in metrics.
- **Later laps:** the completion reward (≥100) plus the final-outpost reward (60) already
  exceed `reward_scale = 100`. The step is clipped to +1 *before* the improvement bonus
  or the +200 "better than algorithm" bonus can matter. A route 10% better than the last
  and one 50% worse earn the same lap reward. Quality enters only indirectly through
  per-step penalties.
- `RewardConfig.route_improvement_reward` is never read.
- The time penalty `−0.01 · episode_step` grows every step and the policy cannot observe
  the step count, so the reward is non-stationary. With the default
  `max_episode_steps = 16384`, every step after step 10,000 saturates at −1.

**Fix.** Make the per-lap reward a bounded function of efficiency, for example
`c · min(target / lap_energy, 2)`. Drop per-step clipping and normalise returns with
`VecNormalize(norm_obs=False, norm_reward=True)`. Remove the growing time penalty, or
add normalised time to the observation.

### 7. The circular-behaviour penalty punishes every non-move action

[reward.py:173-177](../../src/tsp_rl_kg/rl/reward.py#L173-L177) with `recent_path`
`maxlen = len(outposts)` (line 37). Scout, collect, build path and place rock leave the
position unchanged, so each one costs −2. Blocked moves do too. These are exactly the
actions that grow the KG and let the agent beat the oracle.

**Fix.** Apply the penalty to movement actions only, or replace it with a count-based
novelty bonus.

## P1 — Observation and task design

### 8. The policy cannot see the task state

The observation holds only `vision`, the graph and its counts. Nothing tells the agent
which outposts it has visited, its wood and stone, its lap progress or the time remaining.
All of these drive the reward and the valid actions. For a feed-forward policy, "go to the
next unvisited outpost" cannot be expressed.

**Fix.** Add a small `status` Box: visited flags in a canonical outpost order, normalised
inventory, fraction of the episode remaining and lap count. Mirror the visited flag on
outpost entity nodes in the graph. A recurrent policy (sb3-contrib `RecurrentPPO`) is the
heavier alternative and needs a new dependency.

### 9. Pose a well-defined episodic task first

The current task is continuing: laps repeat until an early stop after 5 non-improving
laps, a time-out or a no-progress cut. That makes success, return and efficiency hard to
interpret. Start by terminating on the first completed lap. Success and efficiency then
become per-episode quantities, and route improvement shows up *across* training rather
than within an episode. Reintroduce multi-lap episodes as a later variant.

### 10. The world distribution is narrower and harder than it looks

- **Terrain diversity.** [heightmap_generator.py:28](../../src/tsp_rl_kg/game_world/heightmap_generator.py#L28)
  draws the noise base from `randint(0, 100)`, which allows at most 101 terrains per map
  size. Only 78 of 150 training terrains were distinct, and 113 of 150 "held-out"
  evaluation worlds share terrain with a training world. The disjointness check in
  `make_eval_env` compares whole-world fingerprints, so a different entity layout on the
  same terrain passes as held out. **Fix:** add random coordinate offsets from the full
  seed space and split train/eval on terrain.
- **Obstacle density.** 20–40% of tiles hold an impassable entity
  ([environment.py:67-72](../../src/tsp_rl_kg/game_world/environment.py#L67-L72)), and only
  9 of 20 worlds are solvable without clearing. That is a hard exploration problem to
  combine with the KG question. **Fix:** guarantee reachability at generation (reject or
  repair worlds), and make clearing a later curriculum stage.
- **Default study world.** `tsp-study` without a config uses 5×5 maps
  ([run.py:62](../../src/tsp_rl_kg/rl/training/run.py#L62)). With vision range 1 the agent
  already sees 36% of the map, which leaves little room for prior knowledge to matter.
- **Evaluation bias.** The eval pool is sorted by target energy, and `FixedWorldEvalCallback`
  restarts at index 0, so periodic evaluation always uses the *n easiest* held-out worlds
  (5 in the full config). The environment still builds the whole pool. **Fix:** sample a
  fixed, stratified subset and build only those worlds.

### 11. Per-episode metrics

`CurriculumService` samples `env.get_metrics()` from `envs[0]` every `n_steps` steps. That
gives a snapshot of whatever episode is running, not an episode aggregate. Log success,
laps, closed-lap efficiency, outposts reached, final known fraction and action histograms
per episode (see #3).

## P1 — Network architecture

### 12. Vision spends almost all the compute on a 5×5 symbol grid

[semantic_vision.py:17-38](../../src/tsp_rl_kg/observation/semantic_vision.py#L17-L38)
paints each tile of a (2r+1)² window at the *display* tile size. That is 40 px in the
full config (3×200×200) and 25 px at defaults (3×125×125). The image carries no more
information than a 5×5×15 one-hot tensor (375 values against 120,000), yet it accounts
for:

- 92% of observation bytes;
- 5.85 M of the policy's 7.35 M parameters;
- nearly all training memory (the configured batch of 64 does not fit on a 6 GB GPU).

It also couples the policy input shape to `screen_size`, a display setting.

**Fix.** Emit a `(C, 2r+1, 2r+1)` one-hot tensor (terrain, entity, player, visible mask)
and use a two-layer 3×3 CNN or a small MLP. Keep the RGB renderer for humans. Beyond
speed, this matters for the research design: "vision-only vs vision+KG" currently pits a
6 M-parameter CNN against a 0.18 M GAT, so any difference confounds information with
capacity.

### 13. Remove BatchNorm and dropout from the policy

BatchNorm2d at [encoders.py:34](../../src/tsp_rl_kg/rl/encoders.py#L34) and dropout 0.25
at [encoders.py:177/230](../../src/tsp_rl_kg/rl/encoders.py#L177) are in the shared
feature extractor. SB3 collects rollouts in eval mode and optimises in train mode, so the
"old" log-probabilities come from a different function. At identical weights, 64% of
samples already fall outside PPO's clip range (53% from dropout alone), so clipping mostly
reacts to noise. For DQN, dropout also makes the target network stochastic. Use no
normalisation, or LayerNorm, and set `dropout = 0`.

### 14. Smaller fixes

- **Initialisation differs between algorithms.** `HybridEncoder._initialize_weights`
  (Kaiming) is overwritten for PPO: SB3's `ortho_init` re-initialises every Conv/Linear in
  the feature extractor with orthogonal gain √2 (`stable_baselines3/common/policies.py`
  617-630). DQN keeps Kaiming. Choose one scheme explicitly so algorithm comparisons are
  not also init comparisons.
- **The fusion MLP is deep and oddly sized.** Five linear layers follow the concatenation
  (384→768→768→384→192→192), before SB3's own `[64, 64]` heads. The sizes are derived from
  `vision_fc_dims`/`graph_fc_dims`
  ([encoders.py:145-146](../../src/tsp_rl_kg/rl/encoders.py#L145-L146)), which read as
  per-branch settings. One or two fusion layers plus `net_arch=dict(pi=[128], vf=[128])`
  is enough.
- **Graph batching loops in Python.** `_prepare_graph_batch` loops per sample and calls
  `.item()`, forcing a GPU sync each time. At batch 32 it costs more than the GAT. Build
  node masks from `arange(max_nodes) < num_nodes` and use cumulative offsets instead.
- **DQN replay does not fit in memory.** SB3's dict replay buffer stores observations
  *and* next observations. At ~262 KB per observation, the default `buffer_size = 100_000`
  needs about 52 GB. The compact vision in #12 and #18 largely removes this.

## P1 — Knowledge-graph representation and readout

### 15. Mean pooling discards the player-relative information the KG should contribute

[encoders.py:114-118](../../src/tsp_rl_kg/rl/encoders.py#L114-L118) runs three GAT layers,
then `global_mean_pool` over every known node: up to 289 nodes at 12×12 and 2,049 at
32×32. Three things combine:

- Node features carry no position, by the g21 design.
- The player is one node, attached to one terrain node.
- Adjacency edges all have `distance = 1`, so `edge_attr` encodes only the edge type,
  never the direction.

As a result, message passing is invariant to the lattice's rotations and reflections, and
the pooled vector is mostly a histogram of known terrain and entity types. It cannot
represent "an unvisited outpost lies 7 tiles east through hills", while every movement
action is directional. **The architecture therefore predicts no effect of
`kg_completeness`, independent of the hypothesis.**

Options, roughly in order of effort:

1. **Player-anchored readout.** Use the player's node embedding, or attention pooling with
   the player embedding as the query, instead of a global mean.
2. **Directional or relative encodings.** Give edges a direction (`dx, dy ∈ {−1, 0, 1}`),
   and/or add each node's offset from the player as a feature. This is relative
   geometry, not absolute coordinates, so it does not conflict with g21's concern about
   corrupting feature columns, but record the decision in g21.
3. **Longer range.** Three hops cannot reach an outpost 10+ tiles away. Add a virtual
   global node, hierarchical coarsening, or shortcut edges (see #16).
4. **Symbolic queries as features.** Run Dijkstra over the *remembered* passable graph and
   feed the energy from the player to each known outpost, with visited flags, into the
   policy. This is the most direct neuro-symbolic use of the KG: a planning query the CNN
   cannot make, which degrades gracefully as completeness falls.

### 16. The graph is a spatial lattice, not yet a knowledge graph

[constitution.py](../../src/tsp_rl_kg/graph/constitution.py) builds a 4-connected terrain
grid, one entity node per tile (including "empty", id 0) and one player node. The only
relations are adjacency, entity–terrain and player–terrain. A GNN over a 4-connected
lattice approximates a CNN at a much higher cost, so the "KG" arm adds little that a map
image would not. Suggestions:

- Instantiate entity nodes only for real entities, or fold entities into terrain-node
  features. This roughly halves node and edge counts.
- Add semantic structure:
  - outposts as first-class nodes with `visited` state;
  - `route` edges between outposts annotated with known-map path cost, or "unknown";
  - resource and affordance relations (tree → wood → path → lower cost).

  Encode these with typed message passing (PyG `HeteroData` with HGT or RGCN) rather than
  a homogeneous GAT with a one-hot edge type.
- For a clean ablation, compare three arms that hold the **same information** with
  different inductive biases: (a) remembered map as a multi-channel image plus CNN,
  (b) the same memory as a graph plus GNN, (c) the semantic graph from the previous bullet.

### 17. Other KG notes

- **The prior is salt-and-pepper.** `KnowledgeState` reveals a uniform random subset of
  tiles. More realistic, and more likely to show a KG effect: contiguous revealed regions,
  landmark-only priors (outposts known, terrain unknown), or stale priors that sensing
  must correct.
- **Sentence-transformer embeddings** of 15 fixed descriptors add only a frozen similarity
  prior over one-hot. Keep the ablation, but add a learned `nn.Embedding(15, d)` baseline.
- **Per-step cost.** `get_subgraph` loops over all nodes in Python on every step
  ([knowledge_graph.py:104-110](../../src/tsp_rl_kg/knowledge/knowledge_graph.py#L104-L110)).
  A move senses twice: once in `move_player_node` (line 102) and again in `agent_action`
  ([agent.py:69](../../src/tsp_rl_kg/game_world/agent.py#L69)); scout does the same.
  Precompute a node→tile index to vectorise selection, and sense once per action.

## P2 — Training protocol and evaluation

18. **Budget and statistics.** The research config requests 150k steps × 2 seeds. For
    sparse, partially observed, multi-goal navigation, plan for 1–10 M steps and at least
    5 seeds. Report interquartile mean with bootstrap confidence intervals (Agarwal et al.
    2021, *rliable*), and evaluate on at least 50 held-out episodes, reporting both
    deterministic and stochastic evaluation. With #12 fixed, `SubprocVecEnv` with 8
    environments should give several thousand steps/s on this machine.
19. **Baselines.** Add three non-learning references so RL numbers can be interpreted:
    random; greedy BFS to the nearest known unvisited outpost over the remembered map; and
    a full-knowledge planner. The greedy BFS agent also makes a useful regression test
    for #2 and #3.
20. **Action masking.** Most of the 11 actions are no-ops in most states (collect with
    nothing adjacent, build with no wood, place rock off water). `MaskablePPO` would help
    sample efficiency substantially, but it needs sb3-contrib, which is a new dependency.
21. **Exploration.** A bonus for newly known tiles is cheap because `KnowledgeState`
    already tracks them, and it lines up with the KG theme.
22. **PPO settings.** Add `ent_coef ≈ 0.01` and reward normalisation (#6). Revisit
    `gamma = 0.995` once episodes are short (#9).

## P3 — Code health (brief)

- Unused config fields: `RewardConfig.route_improvement_reward` and
  `EpisodeConfig.max_game_worlds_trained_in`. `min_episodes_per_curriculum` is duplicated
  across `SimulationManagerConfig` and `CurriculumConfig`.
- `logger.info` runs inside per-step reward and termination paths
  ([reward.py:161-175](../../src/tsp_rl_kg/rl/reward.py#L161-L175)), which floods logs over
  long runs. Demote these to `debug`.
- `Target_Manager` uses `queue.PriorityQueue` (thread-safe, slow) with per-call dicts over
  every cell. `heapq` plus NumPy, or `scipy.sparse.csgraph`, would be faster.
- World generation relies on the global `random` and `np.random` state, swapped in and out
  by `EnvironmentManager`. Pass a `np.random.Generator` explicitly instead.
- `MOVE_UP = (0, +1)` moves the player *down* the screen (the `S` key maps to it). Rename
  to compass directions so action histograms read correctly.
- `SimulationManager.plot_curriculum` writes to a hard-coded `Writing/` directory and calls
  `plt.show()`.

## Suggested sequence

Each phase follows the reliability roadmap's pattern: focused PRs, regression tests with
every fix, and a written exit gate.

**Phase A — correct the task (P0 #1–7).** Fix shaping, the no-progress counter,
truncation, success accounting, the curriculum criterion, the oracle and reward scaling.
Add a *learnability smoke test*: a 6×6 world with 2 reachable outposts, compact
observation and PPO, which must exceed a stated success rate within a fixed CPU budget.
Run it nightly rather than on every PR. *Exit gate:* the BFS baseline is never terminated
mid-lap, its success and efficiency are recorded correctly, and the smoke test learns.

**Phase B — Markov, right-sized agent (#8–14).** Add the status observation and visited
flags, compact vision, no BatchNorm or dropout, per-episode metrics, terrain-disjoint
held-out worlds with guaranteed reachability, and the three baselines. *Exit gate:* on
fully known 12×12 worlds PPO approaches the greedy BFS baseline, and vision-only vs
vision+graph arms have comparable parameter counts.

**Phase C — make the KG matter (#15–17).** Add a player-anchored readout, directional
edges, symbolic path-cost features, entity-node pruning, and optionally the semantic
graph. *Exit gate:* with completeness 1.0 and vision disabled, the agent can navigate to
outposts outside its vision window. That is a direct test that the graph branch conveys
location.

**Phase D — the study.** Pre-register the hypothesis: success and efficiency rise with
completeness for KG agents and stay flat for vision-only agents. Then run ≥5 seeds over
{0, 0.25, 0.5, 0.75, 1.0} plus baselines at a budget set by Phase B learning curves, and
report IQM with confidence intervals on stratified held-out worlds.

## Implementation delivery log — 2026-09-27

The review above is preserved as the pre-change evidence. The implementation uses a
single closed tour from a seeded starting outpost, with reachable worlds and a static
passable-world reference. Clearing-required worlds and the other optional research
variants remain deferred.

| Delivery | Implemented behavior | Evidence / remaining gate |
| --- | --- | --- |
| A1 | EpisodeState owns visits, closure, energy and progress; executable energy-optimal reference; successful termination and timeout truncation | Reference execution matches predicted energy; closure/reset/limit regressions pass |
| A2 | Actual-energy, arrival and bounded efficiency rewards; sensing-only potential; reward-only normalization with checkpoint sidecars | Behavioral reward, PPO/DQN update/reload, and normalization tests pass; long learnability gate not run locally |
| B1 | Compact tile vision, task status, deterministic compact policy, explicit capacity matching, vectorized batching | Spaces, gradients, mode parity and parameter matching pass; 6 GiB GPU test added but hardware unavailable |
| B2 | Isolated RNG, terrain-disjoint stratified pools/manifests, per-worker terminal records, configured curriculum, three executable baselines | Pool isolation, fixed evaluation schedule, BFS/planner and two-worker tests pass |
| C1 | Relative geometry, signed directions, player-query attention, empty-entity pruning and once-per-action sensing | Padding/batch isolation, direction reversal and disconnected distant-node influence pass |
| C2 | Masked remembered-map path queries and separate query ablation | Unknown/disconnected masking and disabled-graph isolation pass; learned graph-only navigation gate remains unrun |
| D1 | Three-seed smoke, gated 1M→3M→10M pilot, validation-only graph gate, immutable source/dependency fingerprints | Gate logic and stale/missing-gate rejection pass; no learning result is claimed |
| D2 | Five-seed, five-completeness, three-arm study; paired IQM bootstrap, explicit undefined conditional metrics, baseline statistics and plots | Statistical fixtures, world pairing and locked-config validation pass; full study remains gated |

The final observation version is **4**. Compact vision/status (the planned schema-3
milestone) and directional graphs/queries are delivered together; no intermediate
schema-3 checkpoints are supported. The authoritative contract is
[the updated G21 specification](../specs/g21-graph-semantics.md).

### Local validation

- Full software suite: **328 passed, 2 skipped**. One skip is the GPU memory gate;
  the other is an existing generated-world fixture whose required condition was absent.
- Additional final gate/CLI checks after reserving final-test worlds: **31 passed**.
- Ruff, Black/isort verification across 87 Python files, `git diff --check`, and
  `uv lock --check --offline`: passed. No dependencies were added.
- Real PPO and DQN updates, periodic evaluation, model/normalization reload, and a
  two-process training run were exercised. The complete eight-process study, nightly
  200k-step learning runs, pilot, and graph-only learning gate were not executed locally.
- Existing historical experiments/checkpoints were preserved. Changed schemas require
  fresh training; a checkpoint now includes model, normalization and schema sidecars.

### Running the learning gates

Use one source revision and output directory for the sequence documented in the
[README](../../README.md#gated-closed-tour-research-protocol): `research smoke`,
`research pilot`, `research graph`, then `research study`. The nightly workflow runs
only the CPU smoke and retains evidence on failure. Each smoke seed must independently
reach 80% success. The graph gate uses validation worlds, leaving the 12×12 study's
final-test set reserved for final evaluation. Gate failures or source changes prevent
the full study from starting. Pilot seeds run sequentially so only one worker pool
is resident at a time; extended budgets retain the preceding checkpoint and evidence.

`configs/ablation_full.toml` is the locked research preset and dispatches through these
gates. The small `ablation.toml` and training examples remain bounded integration
examples and are not learning evidence.
