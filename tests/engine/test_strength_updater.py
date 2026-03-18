"""Tests for InPlayStrengthUpdater (v5 EKF-based)."""

from src.engine.strength_updater import InPlayStrengthUpdater

_A_H_INIT = 0.3
_A_A_INIT = -0.1
_SIGMA_A_SQ = 0.25


def _make_updater(pre_match_home_prob: float = 0.50) -> InPlayStrengthUpdater:
    return InPlayStrengthUpdater(
        a_H_init=_A_H_INIT, a_A_init=_A_A_INIT,
        sigma_a_sq=_SIGMA_A_SQ, pre_match_home_prob=pre_match_home_prob,
    )


def test_no_update_early_game() -> None:
    """v4 compat: small mu_elapsed → a barely moves."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="home", mu_H_elapsed=0.01, mu_A_elapsed=0.01,
    )
    assert abs(new_a_H - _A_H_INIT) < 0.05
    assert abs(new_a_A - _A_A_INIT) < 0.05


def test_strong_update_late_game() -> None:
    """v4 compat: large mu_elapsed → a_H shifts noticeably."""
    updater = _make_updater()
    new_a_H, _ = updater.update_on_goal(
        team="home", mu_H_elapsed=1.4, mu_A_elapsed=1.0,
    )
    assert updater.n_H == 1
    assert abs(new_a_H - _A_H_INIT) > 0.05


def test_zero_goals_penalized() -> None:
    """v4 compat: away goal penalizes home, rewards away."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="away", mu_H_elapsed=1.2, mu_A_elapsed=0.3,
    )
    assert new_a_H < _A_H_INIT
    assert new_a_A > _A_A_INIT


def test_classify_goal() -> None:
    """Classification thresholds unchanged from v4."""
    updater = _make_updater(pre_match_home_prob=0.70)
    assert updater.classify_goal("away").label == "SURPRISE"
    assert updater.classify_goal("home").label == "EXPECTED"

    updater2 = _make_updater(pre_match_home_prob=0.50)
    assert updater2.classify_goal("home").label == "NEUTRAL"


def test_ekf_path_goal_update() -> None:
    """v5: EKF path via lambda kwargs."""
    updater = _make_updater()
    new_a_H, _ = updater.update_on_goal(
        team="home", lambda_H=0.03, lambda_A=0.02, dt=1.0,
    )
    assert updater.n_H == 1
    assert new_a_H != _A_H_INIT  # EKF updated


def test_ekf_predict() -> None:
    """v5: predict step increases uncertainty."""
    updater = _make_updater()
    P_before = updater.ekf.P_H
    updater.predict(dt=1.0)
    assert updater.ekf.P_H > P_before
