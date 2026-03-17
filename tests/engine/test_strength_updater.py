"""Tests for InPlayStrengthUpdater Bayesian update logic."""

from src.engine.strength_updater import InPlayStrengthUpdater

# Shared defaults
_A_H_INIT = 0.3
_A_A_INIT = -0.1
_SIGMA_A = 0.5
_SIGMA_A_SQ = _SIGMA_A ** 2


def _make_updater(pre_match_home_prob: float = 0.50) -> InPlayStrengthUpdater:
    return InPlayStrengthUpdater(
        a_H_init=_A_H_INIT,
        a_A_init=_A_A_INIT,
        sigma_a_sq=_SIGMA_A_SQ,
        pre_match_home_prob=pre_match_home_prob,
    )


def test_no_update_early_game() -> None:
    """Very small mu_elapsed → shrinkage ≈ 0 → a barely moves."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="home", mu_H_elapsed=0.01, mu_A_elapsed=0.01,
    )
    assert abs(new_a_H - _A_H_INIT) < 0.05
    assert abs(new_a_A - _A_A_INIT) < 0.05


def test_strong_update_late_game() -> None:
    """Large mu_elapsed → meaningful shrinkage → a_H shifts noticeably."""
    updater = _make_updater()
    new_a_H, _ = updater.update_on_goal(
        team="home", mu_H_elapsed=1.4, mu_A_elapsed=1.0,
    )
    assert updater.n_H == 1
    assert abs(new_a_H - _A_H_INIT) > 0.05


def test_zero_goals_penalized() -> None:
    """Away goal: home gets 0 goals vs expectation → a_H drops.
    Away exceeds expectation → a_A rises."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="away", mu_H_elapsed=1.2, mu_A_elapsed=0.3,
    )
    # Home scored 0 vs 1.2 expected → penalized downward
    assert new_a_H < _A_H_INIT
    # Away scored 1 vs 0.3 expected → rewarded upward
    assert new_a_A > _A_A_INIT


def test_classify_goal() -> None:
    """Classification based on scoring team's pre-match win probability."""
    # Away scores when home is heavy favourite (0.70) → away prob = 0.30 < 0.35
    updater = _make_updater(pre_match_home_prob=0.70)
    result = updater.classify_goal(team="away")
    assert result.label == "SURPRISE"

    # Home scores when home is heavy favourite (0.70) → home prob = 0.70 > 0.60
    result = updater.classify_goal(team="home")
    assert result.label == "EXPECTED"

    # Home scores when evenly matched (0.50) → 0.35 ≤ 0.50 ≤ 0.60
    updater2 = _make_updater(pre_match_home_prob=0.50)
    result = updater2.classify_goal(team="home")
    assert result.label == "NEUTRAL"
