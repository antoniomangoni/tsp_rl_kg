# Repository reliability roadmap

## Objective and boundaries

Make training inputs correct, experiment failures explicit, and research runs reproducible before further architectural cleanup. Preserve Python 3.14, uv, Gymnasium, PyTorch Geometric, and SB3 PPO/DQN. Add no dependencies or navigation tools. Preserve historical artifacts; corrected observations require fresh training.

Implement phases sequentially through focused pull requests. Regression tests accompany fixes; phase 3 adds cross-subsystem coverage. Follow Black/isort/Ruff conventions. Do not run the full research sweep.

## Phase 1 — Fix graph and observation semantics

Status: implemented. Validation: 271 tests pass; Ruff, Black and isort pass. Detailed contract: [g21-graph-semantics](../specs/g21-graph-semantics.md).

- Correct subgraph indices, minibatch connectivity, padding and player features.
- Completeness becomes seeded initial map knowledge, separate from visual discovery.
- Remember observations, restore pristine worlds at episode boundaries, and render canonical semantic vision independent of display mode.
- Version observations and knowledge semantics; require fresh training.

Exit gate: connectivity, isolation, knowledge, reset and visual tests pass; old observations fail clearly. No learning-improvement claim is required.

## Phase 2 — Make experiments fail visibly

Status: implemented. Validation: 281 tests pass; real example study: attempted 4, succeeded 4, failed 0; Ruff/Black/isort pass.

- Replace algorithm-specific hyperparameters on algorithm switches. Preserve arbitrary PPO options while synchronizing only legacy fields.
- Record every attempted seed, status, errors and artifact paths. Preserve successful-result fields; add attempted/succeeded/failed counts and incomplete aggregate labels.
- Continue remaining seeds after failures; persist partial results, then raise a study error and return nonzero. Reject empty experiments/seeds before starting.
- Clean up initialized environments in finally blocks, including setup failures. Preserve failed MLflow runs and fail their parent study.
- Fix DQN weight logging through its online Q-network extractor. Honor periodic evaluation episode counts and final evaluation determinism.
- Use held-out reproducible evaluation worlds and a fixed schedule independent of training curricula.
- Advance curricula at episode boundaries, never through mid-transition callback resets.
- Use unique run directories, scoped simulation exports, and play-only recording.

Exit gate: all four example experiments produce successful seeds; injected failure preserves diagnostics and partial results with nonzero exit.

## Phase 3 — Add integration tests

Status: implemented. Validation: 9 CPU integration tests passed in 33.42 seconds (281 unit tests deselected), including all real study experiments, model updates/reload, logging boundary and MLflow failure status. Wheel import works; sprite checks are completed with phase 4.

- Execute the small study in temporary storage with local MLflow; check experiments, seeds, metrics and models.
- Exercise PPO/DQN updates, save/reload, prediction, periodic evaluation, curriculum transitions and DQN logging boundary.
- Cover partial/all-seed failure, invalid configuration, cleanup and isolated output directories.
- Check held-out worlds, deterministic resets and nested paired priors.
- Build/inspect a wheel and exercise installed behavior outside the checkout.
- Run bounded headless CPU tests in CI, checking finite outputs and behavior rather than reward gains.

Exit gate: regressions are detected without external services, long training or writes to existing outputs.

## Phase 4 — Finish housekeeping

Status: pending.

- Package sprites and load through package resources.
- Reconcile README, module docs, diagrams, examples and stale agent guidance. Fix broken links and ignored workflow/plan files.
- Remove unused timeout (completed with phase-2 lifecycle changes) and active replay/sequence/world-model settings; reject obsolete config keys with actionable errors. Keep standalone trajectory utilities experimental.
- Require uv sync --locked in CI.
- Handle empty/single/equal-energy world pools and validate configuration relationships.
- Avoid unrelated renaming/abstraction changes.

Exit gate: documented commands and installed assets work; obsolete settings cannot silently succeed; all checks pass.

## Delivery and validation log

Explicitly stage these planning files despite the existing ignore rules; revise the broader rules in phase 4. Keep phase-specific implementation and validation recorded here. Never delete historical experiments, rewrite checkpoints, install tools, or bypass tests to obtain green checks.

Publication: automatic approval review blocked external GitHub pushes/PR creation. Local phase branches and commits are retained; explicit publication approval is required.
