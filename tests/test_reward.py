"""Behavioral regression tests for stationary closed-tour rewards."""

from dataclasses import replace

import numpy as np
import pytest

from tsp_rl_kg.config import RewardComponent, RewardConfig
from tsp_rl_kg.rl.episode import efficiency
from tsp_rl_kg.rl.reward import RewardCalculator, RewardTransition


def transition(**changes):
    return replace(RewardTransition(2, 20, 20, False, False, -0.5, -0.4), **changes)


def test_calculate_charges_actual_energy_and_every_step_shaping():
    calc = RewardCalculator(RewardConfig(), gamma=0.9)
    assert calc.calculate(transition()) == pytest.approx(-0.02 + 0.9 * -0.4 + 0.5)
    assert calc.calculate(transition(energy_delta=3, potential_after=-0.5)) == pytest.approx(
        -0.03 + 0.05
    )


def test_first_arrival_reward_once_and_quality_not_clipped():
    calc = RewardCalculator(RewardConfig(shaping_weight=0))
    assert calc.calculate(transition(new_outpost=True)) == pytest.approx(0.98)
    good = calc.calculate(transition(success=True, episode_energy=20))
    poor = calc.calculate(transition(success=True, episode_energy=40))
    assert good == pytest.approx(9.98)
    assert poor == pytest.approx(4.98)
    assert good > poor > 1


def test_shaping_ablation_is_explicit():
    calc = RewardCalculator(RewardConfig(), disabled_components=[RewardComponent.PROXIMITY])
    assert calc.calculate(transition()) == pytest.approx(-0.02)


def test_terminal_potential_and_truncation_are_distinct():
    calc = RewardCalculator(RewardConfig(), gamma=0.99)
    terminal = calc.calculate(transition(success=True, potential_after=0))
    timeout = calc.calculate(transition(potential_after=-0.4))
    assert terminal == pytest.approx(10.48)
    assert timeout == pytest.approx(0.084)


@pytest.mark.parametrize(
    "reference,energy,expected", [(0, 0, 1), (0, 10, 0), (10, 20, 0.5), (20, 10, 2)]
)
def test_efficiency(reference, energy, expected):
    assert efficiency(reference, energy) == expected


def test_zero_energy_reward_is_finite_and_bounded():
    calc = RewardCalculator(RewardConfig())
    for reference in (0, 20):
        reward = calc.calculate(
            transition(success=True, episode_energy=0, reference_energy=reference)
        )
        assert np.isfinite(reward)
        assert reward < 21


@pytest.mark.parametrize(
    "field,value",
    [("penalty_per_step", 1), ("shaping_weight", -1), ("completion_reward", float("inf"))],
)
def test_invalid_weights_rejected(field, value):
    with pytest.raises(ValueError):
        RewardConfig(**{field: value})
