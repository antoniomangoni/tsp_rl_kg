"""Paired hierarchical bootstrap; no silent omission of missing experiment cells."""

import numpy as np


def iqm(values):
    values = np.sort(np.asarray(values, dtype=float).ravel())
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("IQM requires nonempty finite values")
    # Fractional boundary weights make small-sample IQM well defined.
    left, right = len(values) * 0.25, len(values) * 0.75
    weights = np.maximum(
        0, np.minimum(np.arange(len(values)) + 1, right) - np.maximum(np.arange(len(values)), left)
    )
    return float(np.dot(values, weights) / weights.sum())


def paired_bootstrap(values, *, seed=0, repetitions=2000):
    """Input (arms, training seeds, common worlds); shared resamples preserve pairing."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Complete finite arm/seed/world tensor required")
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, values.shape[0]))
    for i in range(repetitions):
        seeds = rng.integers(values.shape[1], size=values.shape[1])
        worlds = rng.integers(values.shape[2], size=(values.shape[1], values.shape[2]))
        for arm in range(values.shape[0]):
            samples[i, arm] = iqm(np.take_along_axis(values[arm][seeds], worlds, axis=1))
    return {
        "iqm": [iqm(arm) for arm in values],
        "confidence_intervals": np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
        "paired_difference_intervals": np.quantile(
            samples - samples[:, :1], [0.025, 0.975], axis=0
        ).T.tolist(),
        "repetitions": repetitions,
        "seed": seed,
    }


def pilot_decision(evaluations, baseline_success, baseline_median_energy, budget):
    """Lock the first passing common budget; otherwise extend only with measured learning."""
    if baseline_median_energy is None:
        return {"status": "diagnose", "budget": budget}
    latest = evaluations[-1]
    passed = (
        latest["success_rate"] >= baseline_success - 0.1
        and latest["median_success_energy"] <= 1.2 * baseline_median_energy
    )
    if passed:
        return {"status": "passed", "budget": budget}
    improving = len(evaluations) >= 3 and any(
        latest[key] - evaluations[-3][key] >= 0.02 for key in ("success_rate", "efficiency")
    )
    budgets = [1_000_000, 3_000_000, 10_000_000]
    if budget not in budgets:
        raise ValueError("Unsupported pilot budget")
    if improving and budget < budgets[-1]:
        return {"status": "extend", "budget": budgets[budgets.index(budget) + 1]}
    return {"status": "diagnose", "budget": budget}


def successful_tour_bootstrap(efficiencies, successes, *, seed=0, repetitions=2000):
    """Conditional IQM with explicit undefined intervals when resamples have no successes."""
    values = np.asarray(efficiencies, dtype=float)
    mask = np.asarray(successes, dtype=bool)
    if values.ndim != 3 or values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("Complete matching arm/seed/world tensors required")
    rng = np.random.default_rng(seed)
    samples = np.full((repetitions, values.shape[0]), np.nan)
    for i in range(repetitions):
        seeds = rng.integers(values.shape[1], size=values.shape[1])
        worlds = rng.integers(values.shape[2], size=(values.shape[1], values.shape[2]))
        for arm in range(values.shape[0]):
            selected = np.take_along_axis(values[arm][seeds], worlds, axis=1)[
                np.take_along_axis(mask[arm][seeds], worlds, axis=1)
            ]
            if len(selected):
                samples[i, arm] = iqm(selected)
    return {
        "iqm": [iqm(v[m]) if m.any() else None for v, m in zip(values, mask, strict=True)],
        "successful_episodes": mask.sum(axis=(1, 2)).tolist(),
        "undefined_resamples": np.isnan(samples).sum(axis=0).tolist(),
        "confidence_intervals": [
            np.quantile(column, [0.025, 0.975]).tolist() if np.isfinite(column).all() else None
            for column in samples.T
        ],
    }
