import numpy as np

from src.calibration.step_1_3_ml_prior import train_xgboost_prior, compute_C_time


def test_mle_fallback():
    """With no odds data, should fall back to league MLE."""
    # Fake matches with known goal counts
    matches = [{"match_id": f"m{i}", "home_goals": 1, "away_goals": 1,
                "home_team": f"Team{i}", "away_team": f"Team{i+1}",
                "goal_events": [], "red_card_events": []} for i in range(50)]
    model, features, a_H, a_A = train_xgboost_prior(matches, {}, "1204")
    assert model is None  # no odds → MLE fallback
    assert len(a_H) == 50
    # MLE for 1 goal/match: a = ln(1/90) ≈ -4.50
    assert -5.0 < a_H[0] < -4.0


def test_C_time_default():
    C = compute_C_time(np.zeros(6))
    assert C == 90.0  # 6 periods * 15 min each
