# Maintenance notes

Reviewed against the implementation on 2026-09-18. The two original reports below
have corresponding implementations and regression coverage.

- **HUD inventory and route scores: implemented.** `GameManager._build_status()`
  supplies wood/stone capacities, target route energy, and current route energy.
  `Renderer.render_ui()` draws the status rows. See the
  [completed HUD spec](../specs/g20-hud-player-status.md),
  `tests/test_game_manager.py`, and `tests/test_renderer.py`.
- **Paths disappearing when the player moves: covered by the occupancy fix.**
  The player is an overlay on the underlying entity grid. The regression
  `test_template_reset_and_path_persistence` in `tests/test_graph_semantics.py`
  places a path, moves the player, and checks that the path remains in both the
  environment and remembered knowledge. See the
  [graph semantics contract](../specs/g21-graph-semantics.md).

The [reliability roadmap](repository-reliability-roadmap.md) retains historical
implementation and validation records; its test counts and publication notes
refer to that work, not the current checkout's validation status.
