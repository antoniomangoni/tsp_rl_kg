# Policy observations

`PaddedPyGObservationEncoder` implements the schema-4 dictionary consumed by the
SB3 adapter. `compact_semantic_vision` creates display-independent channel/y/x tile
symbols; `render_semantic_vision` remains available for human-facing RGB diagnostics.

| Key | Shape | Meaning |
| --- | --- | --- |
| `vision` | `(15, 2r+1, 2r+1)` | Terrain/entity one-hot, player, visible mask |
| `status` | `(2 × outposts + 5,)` | Visits, start identity, inventory, remaining time, return phase, no-progress |
| `node_features` | `(max_nodes, semantic_dim + 5)` | Remembered semantics, relative geometry and task flags |
| `edge_attr` | `(max_edges, semantic_dim + 2)` | Relation features and signed direction |
| `edge_index` | `(2, max_edges)` | Compact local indices |
| `num_nodes`, `num_edges` | `(1,)` | Real leading entries, independent of feature values |
| `path_queries` | `(outposts, 6)` | Known/reachable masks, visits, relative offsets and remembered path energy |

Compact vision contains values in `[0,1]`; it is not divided by 255. Status and queries
use float32; padded features and vision use float16. Empty entity nodes are pruned.
Unknown prior tiles do not enter graphs. Disabled graph branches cannot consume path
queries. Graph batching masks padding before message passing and attention.

See the [complete semantics contract](../../../docs/specs/g21-graph-semantics.md) for
field ordering, episode behavior, migration requirements and conformance tests.
