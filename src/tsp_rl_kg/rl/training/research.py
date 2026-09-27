"""Gated research entrypoint: smoke → pilot → paired five-seed study.

Run with python -m tsp_rl_kg.rl.training.research STAGE --output DIRECTORY.
Gate artifacts are written only after all required seeds pass. Failed attempts retain data.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from tsp_rl_kg.config import (
    AblationConfig,
    AlgorithmConfig,
    EpisodeConfig,
    EvaluationConfig,
    GameManagerConfig,
    SimulationManagerConfig,
    TrainingConfig,
)
from tsp_rl_kg.rl.training.baselines import (
    FullKnowledgeBaseline,
    RandomBaseline,
    RememberedBFSBaseline,
)
from tsp_rl_kg.rl.training.statistics import (
    paired_bootstrap,
    pilot_decision,
    successful_tour_bootstrap,
)
from tsp_rl_kg.rl.training.trainer import Trainer

SEEDS = [42, 7, 123, 456, 789]
COMPLETENESS = [0, 0.25, 0.5, 0.75, 1]
ARMS = ["vision", "gnn", "queries"]


def revision():
    root = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    files = list((root / "src").rglob("*.py")) + [root / "uv.lock", root / "pyproject.toml"]
    for file in sorted(files):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def config_for(stage, arm="gnn", completeness=1.0, n_envs=8):
    smoke = stage == "smoke"
    return TrainingConfig(
        game_manager=GameManagerConfig(
            num_tiles=6 if smoke else 12,
            number_of_outposts=2 if smoke else 3,
            screen_size=480,
            vision_range=2,
            headless=True,
        ),
        simulation_manager=SimulationManagerConfig(
            number_of_environments=20 if smoke else 150, number_of_curricula=1
        ),
        episode=EpisodeConfig(
            max_episode_steps=100 if smoke else 300, max_steps_without_progress=50 if smoke else 100
        ),
        algorithm=AlgorithmConfig(
            verbose=0,
            hyperparameters=dict(
                n_steps=250,
                batch_size=50,
                n_epochs=5,
                learning_rate=0.0003,
                gamma=0.99,
                ent_coef=0.01,
            ),
        ),
        evaluation=EvaluationConfig(
            eval_freq=25_000 if smoke else 100_000, n_eval_episodes=10, final_eval_episodes=50
        ),
        ablation=AblationConfig(
            disable_vision=arm == "graph_only",
            disable_graph=arm == "vision",
            use_path_queries=arm == "queries",
            disable_curriculum=True,
        ),
        total_timesteps=200_000 if smoke else 1_000_000,
        n_envs=1 if smoke else n_envs,
        kg_completeness=completeness,
        seeds=SEEDS[:3] if smoke else SEEDS,
    )


class Deadline(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.deadline = time.monotonic() + seconds

    def _on_step(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Learnability seed exceeded its 30-minute CPU budget")
        return True


def metrics_summary(result):
    records = result["episodes"]
    successes = [r for r in records if r["success"]]
    return dict(
        success_rate=result["success_rate"],
        efficiency=float(np.mean([r["efficiency"] for r in records])),
        median_success_energy=(
            float(np.median([r["energy_spent"] for r in successes])) if successes else None
        ),
    )


def prepare(config, seed, directory, *, cpu=False):
    trainer = Trainer(config.kg_completeness, results_dir=str(directory))
    if cpu:
        trainer.device = torch.device("cpu")
    trainer.source_hash = revision()
    try:
        trainer.setup(config, seed=seed)
        # Construct final-test worlds now and freeze the manifest before training.
        trainer.test_env = trainer.env_manager.make_test_env()
        write(directory / "config.json", config.to_dict())
        write(directory / "manifests.json", trainer.env_manager.manifest())
        write(
            directory / "versions.json",
            dict(
                source_hash=revision(),
                python=platform.python_version(),
                packages={
                    name: importlib.metadata.version(name)
                    for name in ["torch", "stable_baselines3", "numpy", "torch_geometric"]
                },
            ),
        )
    except BaseException:
        trainer.close()
        raise
    return trainer


def train_to(trainer, budget, directory, deadline=None):
    if trainer.source_hash != revision():
        raise RuntimeError("Source changed during research run; restart with a fixed revision")
    backend = trainer.model_trainer.backend
    remaining = max(0, budget - backend.model.num_timesteps)
    if remaining:
        backend.train(
            remaining, str(directory), extra_callbacks=[] if deadline is None else [deadline]
        )
    backend.save(str(directory / "model.zip"))


def evaluate(trainer, directory, final=False):
    if trainer.source_hash != revision():
        raise RuntimeError("Source changed during research run; gate cannot be certified")
    env = trainer.test_env if final else trainer.eval_env
    count = (
        trainer.config.evaluation.final_eval_episodes
        if final
        else trainer.config.evaluation.n_eval_episodes
    )
    results = {}
    for deterministic in (True, False):
        label = "deterministic" if deterministic else "stochastic"
        # Reproducible sampling independent of preceding evaluations.
        with torch.random.fork_rng():
            torch.manual_seed(12345)
            results[label] = trainer.model_trainer.evaluator.evaluate(
                trainer.model_trainer.backend, env, count, deterministic=deterministic
            )
    write(directory / ("final.json" if final else "validation.json"), results)
    return results


def smoke(output):
    if (output / "smoke").exists():
        raise FileExistsError("Smoke evidence already exists; choose a fresh output directory")
    source_hash = revision()
    results = []
    for seed in SEEDS[:3]:
        directory = output / "smoke" / str(seed)
        trainer = None
        started = time.monotonic()
        try:
            trainer = prepare(config_for("smoke"), seed, directory, cpu=True)
            train_to(
                trainer, 200_000, directory, Deadline(max(0, 1800 - (time.monotonic() - started)))
            )
            result = evaluate(trainer, directory, final=True)["deterministic"]
            results.append(
                dict(
                    seed=seed,
                    success_rate=result["success_rate"],
                    elapsed_seconds=time.monotonic() - started,
                    passed=result["success_rate"] >= 0.8 and time.monotonic() - started <= 1800,
                )
            )
        except Exception as error:
            results.append(dict(seed=seed, passed=False, error=repr(error)))
        finally:
            if trainer is not None:
                trainer.close()
            write(output / "smoke" / "attempts.json", results)
    passed = all(r["passed"] for r in results) and source_hash == revision()
    write(output / "smoke_gate.json", dict(passed=passed, source_hash=source_hash, results=results))
    if not passed:
        raise RuntimeError("Learnability gate failed; inspect seed evidence before the pilot")


def require_gate(output, stage):
    path = output / f"{stage}_gate.json"
    if not path.exists():
        raise RuntimeError(f"Missing {stage} gate; run and pass it before continuing")
    gate = json.loads(path.read_text())
    if not gate.get("passed") or gate.get("source_hash") != revision():
        raise RuntimeError(f"{stage} gate is failed or stale for this source revision")
    return gate


def pilot(output, n_envs):
    require_gate(output, "smoke")
    if (output / "pilot").exists():
        raise FileExistsError(
            "Pilot evidence already exists; preserve it and use a fresh output directory"
        )
    histories = {seed: [] for seed in SEEDS[:3]}
    references = {}
    previous_budget = None
    for budget in (1_000_000, 3_000_000, 10_000_000):
        # Only one eight-worker pool/model is resident at a time.
        for seed in SEEDS[:3]:
            directory = output / "pilot" / str(seed)
            phase_directory = directory / str(budget)
            config = config_for("pilot", n_envs=n_envs)
            config.total_timesteps = budget
            trainer = prepare(config, seed, phase_directory)
            try:
                if budget > 1_000_000:
                    trainer.model_trainer.backend.load(
                        str(directory / str(previous_budget) / "model.zip")
                    )
                else:
                    reference = trainer.model_trainer.evaluator.evaluate(
                        RememberedBFSBaseline(trainer.eval_env),
                        trainer.eval_env,
                        trainer.config.evaluation.n_eval_episodes,
                    )
                    references[seed] = metrics_summary(reference)
                    write(directory / "reference.json", reference)
                while trainer.model_trainer.backend.model.num_timesteps < budget:
                    next_step = min(
                        budget, trainer.model_trainer.backend.model.num_timesteps + 100_000
                    )
                    train_to(trainer, next_step, phase_directory)
                    histories[seed].append(
                        metrics_summary(evaluate(trainer, phase_directory)["deterministic"])
                    )
                    write(directory / "history.json", histories[seed])
            finally:
                trainer.close()
        previous_budget = budget
        decisions = []
        for seed in SEEDS[:3]:
            history, reference = histories[seed], references[seed]
            sanitized = [
                {
                    **record,
                    "median_success_energy": (
                        record["median_success_energy"]
                        if record["median_success_energy"] is not None
                        else float("inf")
                    ),
                }
                for record in history
            ]
            decisions.append(
                pilot_decision(
                    sanitized, reference["success_rate"], reference["median_success_energy"], budget
                )
            )
        write(output / "pilot" / "decisions.json", decisions)
        if all(d["status"] == "passed" for d in decisions):
            write(
                output / "pilot_gate.json",
                dict(passed=True, source_hash=revision(), budget=budget, seeds=SEEDS[:3]),
            )
            return
        if any(d["status"] == "diagnose" for d in decisions):
            raise RuntimeError("Pilot stalled; diagnose learning before the study")


def graph_gate(output, n_envs):
    if (output / "graph").exists():
        raise FileExistsError("Graph evidence already exists; use a fresh output directory")
    budget = require_gate(output, "pilot")["budget"]
    results = []
    for seed in SEEDS[:3]:
        directory = output / "graph" / str(seed)
        config = config_for("graph", arm="graph_only", n_envs=n_envs)
        config.total_timesteps = budget
        trainer = prepare(config, seed, directory)
        try:
            train_to(trainer, budget, directory)
            records = evaluate(trainer, directory, final=False)["deterministic"]["episodes"]
            distant_ids = {
                gm.world_template.fingerprint
                for gm in trainer.eval_env.simulation_manager.game_managers
                if any(
                    max(
                        abs(p[0] - gm.world_template.spawn[0]),
                        abs(p[1] - gm.world_template.spawn[1]),
                    )
                    > trainer.config.game_manager.vision_range
                    for p in gm.world_template.outposts
                )
            }
            distant = [r for r in records if r["world_id"] in distant_ids]
            rate = float(np.mean([r["success"] for r in distant])) if distant else 0.0
            results.append(dict(seed=seed, success_rate=rate, episodes=len(distant)))
            if not distant or rate < 0.8:
                raise RuntimeError("Graph-only distant-navigation gate failed")
        finally:
            trainer.close()
    write(output / "graph_gate.json", dict(passed=True, source_hash=revision(), results=results))


def study(output, n_envs):
    require_gate(output, "smoke")
    budget = require_gate(output, "pilot")["budget"]
    require_gate(output, "graph")
    protocol = dict(
        hypothesis=(
            "KG success and efficiency increase with initial completeness; "
            "vision-only remains flat."
        ),
        arms=ARMS,
        completeness=COMPLETENESS,
        seeds=SEEDS,
        budget=budget,
        source_hash=revision(),
        n_envs=n_envs,
    )
    protocol_path = output / "study" / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(
            "Study directory already exists; choose a fresh output directory or inspect its results"
        )
    write(protocol_path, protocol)
    attempts = []
    all_results = {}
    baseline_results = {}
    for arm in ARMS:
        for completeness in COMPLETENESS:
            key = f"{arm}_{completeness}"
            all_results[key] = []
            for seed in SEEDS:
                directory = output / "study" / key / str(seed)
                trainer = None
                try:
                    config = config_for("study", arm, completeness, n_envs)
                    config.total_timesteps = budget
                    trainer = prepare(config, seed, directory)
                    train_to(trainer, budget, directory)
                    result = evaluate(trainer, directory, final=True)
                    if arm == ARMS[0]:
                        baselines = {}
                        for name, cls in [
                            ("random", RandomBaseline),
                            ("remembered_bfs", RememberedBFSBaseline),
                            ("full_knowledge", FullKnowledgeBaseline),
                        ]:
                            baselines[name] = trainer.model_trainer.evaluator.evaluate(
                                cls(trainer.test_env), trainer.test_env, 50
                            )
                        for name, reference in baselines.items():
                            baseline_results.setdefault(f"{name}_{completeness}", []).append(
                                dict(deterministic=reference, stochastic=reference)
                            )
                        write(directory / "baselines.json", baselines)
                    all_results[key].append(result)
                    attempts.append(dict(arm=key, seed=seed, status="passed"))
                except Exception as error:
                    attempts.append(dict(arm=key, seed=seed, status="failed", error=repr(error)))
                    raise
                finally:
                    if trainer is not None:
                        trainer.close()
                    write(output / "study" / "attempts.json", attempts)
                    write(
                        output / "study" / "summary.json",
                        dict(
                            expected=len(ARMS) * len(COMPLETENESS) * len(SEEDS),
                            attempted=len(attempts),
                            failed=sum(a["status"] == "failed" for a in attempts),
                            incomplete=len(attempts) != len(ARMS) * len(COMPLETENESS) * len(SEEDS),
                        ),
                    )
    reports = summarize_study(all_results)
    write(output / "study" / "statistics.json", dict(arms=list(all_results), reports=reports))
    baseline_reports = summarize_study(baseline_results)
    write(
        output / "study" / "baseline_statistics.json",
        dict(arms=list(baseline_results), reports=baseline_reports),
    )
    plot_results(output / "study", reports)


def summarize_study(all_results):
    reports = {}
    for seed_index in range(len(SEEDS)):
        ids = [
            [r["world_id"] for r in results[seed_index]["deterministic"]["episodes"]]
            for results in all_results.values()
        ]
        if any(worlds != ids[0] for worlds in ids[1:]):
            raise ValueError("Evaluation worlds are not paired across arms")
    for mode in ("deterministic", "stochastic"):
        for metric in ("success", "efficiency"):
            values = [
                [[float(r[metric]) for r in result[mode]["episodes"]] for result in results]
                for results in all_results.values()
            ]
            reports[f"{mode}.{metric}"] = paired_bootstrap(values)
        efficiencies = [
            [[r["efficiency"] for r in result[mode]["episodes"]] for result in results]
            for results in all_results.values()
        ]
        successes = [
            [[r["success"] for r in result[mode]["episodes"]] for result in results]
            for results in all_results.values()
        ]
        reports[f"{mode}.successful_tour_efficiency"] = successful_tour_bootstrap(
            efficiencies, successes
        )
    return reports


def plot_results(directory, reports):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("success", "efficiency"), strict=True):
        report = reports[f"deterministic.{metric}"]
        for i, arm in enumerate(ARMS):
            indices = slice(i * len(COMPLETENESS), (i + 1) * len(COMPLETENESS))
            means = np.asarray(report["iqm"])[indices]
            bounds = np.asarray(report["confidence_intervals"])[indices]
            axis.plot(COMPLETENESS, means, marker="o", label=arm)
            axis.fill_between(COMPLETENESS, bounds[:, 0], bounds[:, 1], alpha=0.15)
        axis.set(xlabel="Initial completeness", ylabel=f"IQM {metric}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "pilot", "graph", "study"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, choices=[1, 2, 4, 8], default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "pilot":
        pilot(args.output, args.n_envs)
    elif args.stage == "graph":
        graph_gate(args.output, args.n_envs)
    else:
        study(args.output, args.n_envs)


if __name__ == "__main__":
    main()
