# Repository guidance

Use Python 3.14 and uv. Install the existing locked environment with `uv sync --locked`;
do not add dependencies without a concrete requirement. Source lives under
`src/tsp_rl_kg`; use typed configuration dataclasses in `config.py`.

Read `docs/specs/g21-graph-semantics.md` before changing observations. Schema 2 uses
explicit valid-node/edge counts, compact graph batches, remembered map knowledge,
and pristine episode resets. Completeness is an initial prior, independent of vision.
Policy vision is pure NumPy; pygame resources are for display. Assets are packaged
under `src/tsp_rl_kg/assets/pixel_art` and loaded with `importlib.resources`.

Training supports SB3 PPO/DQN. Trajectory/sequence/world-model utilities are experimental;
they are not active training configuration. Studies must retain failed seed diagnostics,
write partial results, close environments, and exit nonzero after any seed failure.

Run `uv run pytest tests/ -v`, `uv run ruff check .`, `uv run black --check .`, and
`uv run isort --check-only .`. Tests marked `integration` use temporary local MLflow
storage and bounded CPU training. Use `MPLBACKEND=Agg SDL_VIDEODRIVER=dummy` for CI.
Preserve historical results/checkpoints and do not run the full research sweep for fixes.

Use focused topic branches and pull requests; do not push directly to main. See README
for contribution policy and docs/plans/repository-reliability-roadmap.md for validation.
