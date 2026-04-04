# `tsp_rl_kg.utils`

## Purpose / ownership
Owns reusable support utilities for logging, config file loading/merging, and helper decorators used across simulation and training code.

## Main identifiers
- `configure_logging`, `InterceptHandler` (`logger.py`).
- `load_config_file`, `merge_nested_dicts`, `find_mapping_section`, `find_list_section` (`config_files.py`).
- `time_function` decorator (`helper_functions.py`).

## Inputs / outputs and neighboring package interactions
- **Inputs:** file paths for JSON/TOML configs, nested mapping overrides, logger sink settings.
- **Outputs:** validated/merged config mappings, structured logging behavior, and timing instrumentation wrappers.
- **Neighbor interactions:**
  - Used by training CLI and orchestration (`rl.training.run`, `heightmap_generator`, and other modules importing logger/config helpers).

## Extension points
- Add schema-aware config validation helpers.
- Add structured logging sinks/formatters for experiment tracking.
- Add profiling/diagnostics decorators used by environment/training modules.

## Cross-links
- [Top-level package](../README.md)
- [Training module](../rl/training/README.md)

## Tests
- `tests/test_training_entrypoints.py`
- `tests/test_example_configs.py`
- `tests/test_config.py`
